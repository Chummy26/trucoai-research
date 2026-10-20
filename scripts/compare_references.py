"""E4 — comparativo justo entre nossos agentes e os de repositórios de referência.

Protocolo (tudo gravado no JSON de saída, em results/<timestamp>_<tag>.json):
  * regulamento ``PLAYTRUCO_O1`` (o observado na PlayTruco) para TODOS os confrontos;
  * partidas espelhadas: cada distribuição é jogada nos dois assentos (padrão da arena);
  * sementes de avaliação disjuntas das de treino (``EVAL_BASE`` da arena) e dos
    conjuntos já usados em E1-E3 (checado abaixo);
  * etapa 0 (validação, sementes próprias): escolhe o melhor checkpoint
    MaskablePPO do truco-de-malandro; a etapa 1 (teste) não escolhe nada;
  * etapa 1: round-robin entre os agentes principais (família de comparações para
    Bonferroni) + confrontos descritivos dos agentes secundários contra os principais;
  * por confronto: vitórias de partida com IC de Wilson 95%, IC de Wilson com
    Bonferroni, IC pareado por distribuição (cluster-robusto), p exato bilateral
    (binomial), pontos/mão, ms/decisão e taxa de ação ilegal;
  * benchmark de decisão em processo único, sobre as MESMAS observações.

Tudo é SIMULAÇÃO contra bots; não é evidência de força contra humanos.

Uso:
    python scripts/compare_references.py --deals 1000 --secondary-deals 500 --workers 6
    python scripts/compare_references.py --resume results/<arquivo>.json   # retoma (novo arquivo)
"""

from __future__ import annotations

import argparse
import itertools
import json
import math
import platform
import statistics
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

import numpy as np  # noqa: E402

from trucoai import __version__  # noqa: E402
from trucoai.agents import make_agent  # noqa: E402
from trucoai.arena import EVAL_BASE, play_match, run_matchup, wilson  # noqa: E402
from trucoai.engine import new_match  # noqa: E402
from trucoai.observation import observe  # noqa: E402
from trucoai.rules import DEFAULT_RULES, PLAYTRUCO_O1  # noqa: E402

BLUEPRINT = "runs/mccfr_rs_v1_s1/blueprint.pkl"
HYBRID = f"hybrid:{BLUEPRINT}@2000"
CHECKPOINTS = ["truco_liga_v2", "truco_ppo_1M", "truco_ppo", "truco_ppo_200k"]
SECONDARY_FIXED = ["random", "pimc:24", "mccfr_simplex", "mccfr_simple"]
# Sementes de avaliação já usadas antes (E1-E3): não reutilizar no teste.
USED_SEEDS = [(0, 99), (1000, 1399), (5000, 5299), (9000, 9599)]
RULES = PLAYTRUCO_O1


def label(spec: str) -> str:
    return "hybrid(rs_v1,@2000)" if spec == HYBRID else spec


def heavy(spec: str) -> bool:
    return spec.startswith(("hybrid", "blueprint"))


def cap_workers(workers: int, a: str, b: str, heavy_cap: int) -> int:
    """O blueprint ocupa ~650 MB por processo: limita workers quando entra no confronto."""
    return min(workers, heavy_cap) if (heavy(a) or heavy(b)) else workers


# ----------------------------------------------------------------------------
# Estatística
# ----------------------------------------------------------------------------
def binom_p(k: int, n: int) -> float:
    """p-valor bilateral exato de H0: p=0.5 (soma das caudas com P <= P(k))."""
    if n == 0:
        return 1.0
    lp = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2)
          for i in range(n + 1)]
    ref = lp[k] + 1e-9
    return min(1.0, sum(math.exp(x) for x in lp if x <= ref))


def zq(alpha: float) -> float:
    return statistics.NormalDist().inv_cdf(1 - alpha / 2)


def pair_stats(row: dict, alpha_family: float) -> dict:
    n, k = row["games"], row["a_wins"]
    p = k / n
    d = row["deal_wins"]
    x = [w / 2 for w in d]
    mean = sum(x) / len(x)
    sd = statistics.pstdev(x) * math.sqrt(len(x) / (len(x) - 1)) if len(x) > 1 else 0.0
    se = sd / math.sqrt(len(x)) if len(x) else 0.0
    z_c = (mean - 0.5) / se if se > 0 else 0.0
    p_cluster = 2 * (1 - statistics.NormalDist().cdf(abs(z_c)))
    lo_b, hi_b = wilson(k, n, z=zq(alpha_family))
    zc = zq(alpha_family)
    return {
        "wilson95": row["ci95"], "wilson_bonf": [lo_b, hi_b], "alpha_family": alpha_family,
        "cluster95": [mean - 1.96 * se, mean + 1.96 * se],
        "cluster_bonf": [mean - zc * se, mean + zc * se],
        "p_binom": binom_p(k, n), "p_cluster": p_cluster,
        "sig_bonf": bool(binom_p(k, n) < alpha_family and p_cluster < alpha_family),
        "win_rate_a": p,
    }


# ----------------------------------------------------------------------------
# Benchmark de decisão (processo único, mesmas observações)
# ----------------------------------------------------------------------------
def collect_observations(n_games: int, seed0: int) -> list:
    a, b = make_agent("heuristic"), make_agent("malandro_rule")
    obs_list = []
    import random
    for g in range(n_games):
        rng = random.Random(seed0 + g)
        st = new_match(RULES, first_dealer=1)
        agents = (a, b) if g % 2 == 0 else (b, a)
        for ag in agents:
            ag.reset(g)
        hands = 0
        while st.match_winner is None and hands < 40:
            st.deal(rng)
            hands += 1
            while not st.is_terminal():
                o = observe(st, st.current)
                if len(o.legal_actions) > 1:
                    obs_list.append(o)
                st.apply(agents[st.current].act(o))
    return obs_list


def bench_decisions(specs: list, obs_list: list) -> dict:
    out = {}
    for spec in specs:
        ag = make_agent(spec)
        ag.reset(123)
        for o in obs_list[:20]:  # aquecimento
            ag.act(o)
        ts = []
        for o in obs_list:
            t0 = time.perf_counter()
            ag.act(o)
            ts.append((time.perf_counter() - t0) * 1000)
        out[spec] = {"n": len(ts), "mean_ms": float(np.mean(ts)), "median_ms": float(np.median(ts)),
                     "p95_ms": float(np.percentile(ts, 95))}
        print(f"  bench {label(spec):<32} média {out[spec]['mean_ms']:.3f} ms "
              f"(mediana {out[spec]['median_ms']:.3f}, p95 {out[spec]['p95_ms']:.3f})", flush=True)
    return out


def hybrid_coverage(n_deals: int, start: int) -> dict:
    """Quanto das decisões do híbrido vem do blueprint (treinado sob h0) em O1 vs h0."""
    res = {}
    for rules in (DEFAULT_RULES, RULES):
        hy = make_agent(HYBRID)
        opp = make_agent("malandro_rule")
        for s in range(start, start + n_deals):
            for seat in (0, 1):
                play_match(hy, opp, s, seat, rules=rules)
        tot = hy.hits + hy.misses
        res[rules.version] = {"blueprint_hits": hy.hits, "fallback": hy.misses,
                              "hit_rate": hy.hits / max(tot, 1), "decisions_with_choice": tot}
    return res


# ----------------------------------------------------------------------------
def report(path: str) -> None:
    """Imprime tabelas em Markdown a partir de um JSON gravado (não executa partidas)."""
    d = json.loads(Path(path).read_text(encoding="utf-8"))
    rows, sm, bench = d["matchups"], d["summary"], d.get("bench", {})

    def pct(x):
        return f"{100 * x:.1f}"

    print("### Validação (escolha do checkpoint)\n")
    print("| Checkpoint | média de vitória vs heuristic/malandro_rule/pimc:64 |")
    print("|---|---|")
    for k, v in d["validation"]["scores"].items():
        print(f"| {k} | {pct(v)}%{' (escolhido)' if k == d['validation']['chosen'] else ''} |")
    print("\n### Confrontos principais (partidas espelhadas)\n")
    print(f"Bonferroni: m={sm['bonferroni_m']}, alfa por comparação={sm['alpha_family']:.4f}\n")
    print("| A | B | Partidas | Vitória A | IC95 Wilson | IC Bonf. Wilson | IC95 pareado | "
          "p exato | Pts/mão A | Signif. |")
    print("|---|---|---|---|---|---|---|---|---|---|")
    for r in rows:
        if r["stage"] != "principal":
            continue
        s = r["stats"]
        print(f"| {label(r['a'])} | {label(r['b'])} | {r['games']} | {pct(r['win_rate_a'])}% | "
              f"[{pct(s['wilson95'][0])}, {pct(s['wilson95'][1])}] | "
              f"[{pct(s['wilson_bonf'][0])}, {pct(s['wilson_bonf'][1])}] | "
              f"[{pct(s['cluster95'][0])}, {pct(s['cluster95'][1])}] | {s['p_binom']:.4f} | "
              f"{r['point_diff_per_hand_a']:+.2f} ± {1.96 * r['point_diff_se']:.2f} | "
              f"{'sim' if s['sig_bonf'] else 'não'} |")
    print("\n### Agentes principais\n")
    print("| Agente | Vitória média vs. os outros 4 | Partidas (total) | "
          "ms/dec (bench, 1 processo) | ms/dec (arena) |")
    print("|---|---|---|---|---|")
    nan = float("nan")
    for k in sm["ranking_by_mean_win_rate"]:
        t = sm["agents"][k]
        b = bench.get(k, {})
        print(f"| {t['label']} | {pct(t['mean_win_rate_vs_field'])}% | {t['pooled_games']} | "
              f"{b.get('mean_ms', nan):.3f} (p95 {b.get('p95_ms', nan):.3f}) | "
              f"{t['ms_per_decision_in_arena']:.3f} |")
    print("\n### Secundários (descritivo, IC95 de Wilson sem correção)\n")
    mains = sm["ranking_by_mean_win_rate"]
    print("| Agente (A) | " + " | ".join(label(m) for m in mains) + " | ms/dec bench |")
    print("|---|" + "---|" * (len(mains) + 1))
    seen = []
    for r in rows:
        if r["stage"] == "secundario" and r["a"] not in seen:
            seen.append(r["a"])
    for a_ in seen:
        cells = []
        for m_ in mains:
            r = next(x for x in rows if x["stage"] == "secundario" and x["a"] == a_ and x["b"] == m_)
            lo, hi = r["ci95"]
            cells.append(f"{pct(r['win_rate_a'])}% [{pct(lo)}, {pct(hi)}]")
        print(f"| {label(a_)} | " + " | ".join(cells)
              + f" | {bench.get(a_, {}).get('mean_ms', nan):.3f} |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--report", default=None, help="imprime tabelas Markdown de um JSON e sai")
    ap.add_argument("--deals", type=int, default=1000, help="distribuições por confronto principal")
    ap.add_argument("--secondary-deals", type=int, default=500)
    ap.add_argument("--val-deals", type=int, default=150)
    ap.add_argument("--start-seed", type=int, default=20000)
    ap.add_argument("--val-seed", type=int, default=21000)
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--heavy-workers", type=int, default=4, help="máx. workers com blueprint")
    ap.add_argument("--tag", default="e4_references")
    ap.add_argument("--resume", default=None, help="JSON parcial: pula etapas/confrontos prontos")
    ap.add_argument("--skip-bench", action="store_true")
    ap.add_argument("--sensitivity", action="store_true",
                    help="só a análise de sensibilidade: checkpoint escolhido x principais sob "
                         "o regulamento equivalente ao do truco-de-malandro")
    a = ap.parse_args()
    if a.report:
        report(a.report)
        return

    for lo, hi in USED_SEEDS:
        for s0, n in ((a.start_seed, a.deals), (a.val_seed, a.val_deals)):
            assert not (s0 <= hi and s0 + n - 1 >= lo), f"sementes {s0}..{s0 + n - 1} reusam {lo}-{hi}"
    assert not (a.start_seed <= a.val_seed + a.val_deals - 1 and a.val_seed <= a.start_seed + a.deals - 1), \
        "validação e teste com sementes sobrepostas"

    global RULES
    if a.sensitivity:
        from trucoai.agents.references.malandro import MALANDRO_MATCHED
        RULES = MALANDRO_MATCHED
        a.tag = a.tag if a.tag != "e4_references" else "e4_sensibilidade_regras_malandro"
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{a.tag}.json"
    assert not out.exists()

    prev = json.loads(Path(a.resume).read_text(encoding="utf-8")) if a.resume else {}
    done = {(r["a"], r["b"], r["stage"]): r for r in prev.get("matchups", [])}
    rows = list(prev.get("matchups", []))

    from trucoai.agents.references import malandro as M
    bp = ROOT / BLUEPRINT
    meta = {
        "experiment": "E4 comparativo com referências",
        "nota": "simulação contra bots; NÃO é evidência de desempenho contra humanos",
        "code_version": __version__, "rules_version": RULES.version,
        "rules_fingerprint": RULES.fingerprint, "rules_status": RULES.status,
        "python": platform.python_version(), "numpy": np.__version__,
        "machine": {"platform": platform.platform(), "processor": platform.processor()},
        "args": vars(a), "eval_base": EVAL_BASE,
        "test_seeds": [a.start_seed, a.start_seed + a.deals - 1],
        "validation_seeds": [a.val_seed, a.val_seed + a.val_deals - 1],
        "seeds_used_before_e4": USED_SEEDS,
        "blueprint": {"path": BLUEPRINT, "sha256": M.file_sha256(bp),
                      "trained_under": "paulista-1v1-h0 (fingerprint 0a1e69535d56a4b1); avaliado sob O1",
                      "min_visits": 2000},
        "malandro_checkpoints": {c: M.file_sha256(M.resolve_checkpoint(c)) for c in CHECKPOINTS},
        "resumed_from": a.resume,
    }

    def save(extra=None):
        doc = {"meta": meta, "matchups": rows}
        doc.update(extra or {})
        out.write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")

    def play(stage, x, y, deals, start, alpha=None):
        key = (x, y, stage)
        if key in done:
            print(f"[{stage}] {label(x)} x {label(y)}: já pronto", flush=True)
            return done[key]
        t0 = time.time()
        r = run_matchup(make_agent, x, y, deals=deals, start_seed=start,
                        workers=cap_workers(a.workers, x, y, a.heavy_workers), rules=RULES)
        r["wall_s"] = round(time.time() - t0, 1)
        r["stage"] = stage
        rows.append(r)
        lo, hi = r["ci95"]
        print(f"[{stage}] {label(x):>26} x {label(y):<26} A={r['win_rate_a']:.3f} [{lo:.3f},{hi:.3f}] "
              f"pts/mão={r['point_diff_per_hand_a']:+.2f}±{1.96 * r['point_diff_se']:.2f} "
              f"ms A={r['a_metrics']['ms_per_decision']:.2f} B={r['b_metrics']['ms_per_decision']:.2f} "
              f"ilegais={r['a_metrics']['illegal_rate']:.4f}/{r['b_metrics']['illegal_rate']:.4f} "
              f"({r['wall_s']}s)", flush=True)
        save()
        return r

    if a.sensitivity:
        ck = "malandro_ppo:truco_liga_v2"
        opps = ["malandro_rule", "heuristic", "pimc:64", HYBRID]
        for o in opps:
            r = play("sensibilidade", ck, o, a.deals, a.start_seed)
            r["stats"] = pair_stats(r, 0.05)
        save({"nota_sensibilidade": "regulamento: " + RULES.version + " (sem encoberta, sem correr "
                                    "na vez, ferro vale 3); IC95 de Wilson sem correção"})
        print(f"gravado em {out}")
        return

    # ---------------- Etapa 0: escolha do melhor checkpoint (validação) ----------------
    val_opps = ["heuristic", "malandro_rule", "pimc:64"]
    val_score = {}
    for c in CHECKPOINTS:
        spec = f"malandro_ppo:{c}"
        wr = []
        for o in val_opps:
            r = play("validacao", spec, o, a.val_deals, a.val_seed)
            wr.append(r["win_rate_a"])
        val_score[spec] = sum(wr) / len(wr)
    best_ckpt = max(val_score, key=val_score.get)
    print("validação (média de vitória vs heuristic, malandro_rule, pimc:64):",
          {k: round(v, 3) for k, v in val_score.items()}, "-> melhor:", best_ckpt, flush=True)
    save({"validation": {"scores": val_score, "chosen": best_ckpt, "opponents": val_opps}})

    # ---------------- Etapa 1: round-robin principal ----------------
    main_agents = ["pimc:64", HYBRID, "heuristic", "malandro_rule", best_ckpt]
    pairs = list(itertools.combinations(main_agents, 2))
    m = len(pairs)
    alpha_family = 0.05 / m
    for x, y in pairs:
        play("principal", x, y, a.deals, a.start_seed)

    # ---------------- Etapa 2: secundários (descritivo) ----------------
    others = [f"malandro_ppo:{c}" for c in CHECKPOINTS if f"malandro_ppo:{c}" != best_ckpt]
    secondary = SECONDARY_FIXED + others
    for s in secondary:
        for y in main_agents:
            play("secundario", s, y, a.secondary_deals, a.start_seed)

    # ---------------- Análise ----------------
    for r in rows:
        if r["stage"] == "principal":
            r["stats"] = pair_stats(r, alpha_family)
        else:
            r["stats"] = pair_stats(r, 0.05)  # sem correção: descritivo

    table = {}
    for ag in main_agents:
        wrs, wins, games, dec, tms = [], 0, 0, 0, 0.0
        beats = []
        for r in rows:
            if r["stage"] != "principal":
                continue
            if r["a"] == ag:
                wr, w, side, other = r["win_rate_a"], r["a_wins"], r["a_metrics"], r["b"]
            elif r["b"] == ag:
                wr, w, side, other = 1 - r["win_rate_a"], r["games"] - r["a_wins"], r["b_metrics"], r["a"]
            else:
                continue
            wrs.append(wr)
            wins += w
            games += r["games"]
            dec += side["decisions"]
            tms += side["time_s"]
            # supera `other` com significância (família Bonferroni)?
            sig = r["stats"]["sig_bonf"]
            beats.append({"vs": other, "win_rate": wr, "significant": bool(sig),
                          "direction": "vence" if wr > 0.5 else "perde"})
        table[ag] = {"label": label(ag), "mean_win_rate_vs_field": sum(wrs) / len(wrs),
                     "pooled_wins": wins, "pooled_games": games,
                     "ms_per_decision_in_arena": 1000 * tms / max(dec, 1), "vs": beats,
                     "significantly_beats_all": all(b["significant"] and b["direction"] == "vence"
                                                    for b in beats)}
    ranking = sorted(table, key=lambda k: -table[k]["mean_win_rate_vs_field"])
    best_sig = [k for k in ranking if table[k]["significantly_beats_all"]]

    bench = {}
    cov = {}
    if not a.skip_bench:
        print("benchmark de decisão (processo único):", flush=True)
        obs_list = collect_observations(60, 777)[:2500]
        bench = bench_decisions(main_agents + secondary, obs_list)
        bench["_n_observations"] = len(obs_list)
        print("cobertura do blueprint pelo híbrido:", flush=True)
        cov = hybrid_coverage(40, a.val_seed + 5000)
        print(" ", cov, flush=True)

    summary = {"best_checkpoint": best_ckpt, "bonferroni_m": m, "alpha_family": alpha_family,
               "ranking_by_mean_win_rate": ranking, "significantly_best": best_sig,
               "agents": table}
    save({"summary": summary, "bench": bench, "hybrid_blueprint_coverage": cov,
          "validation": {"scores": val_score, "chosen": best_ckpt, "opponents": val_opps}})

    print("\n=== Principal (Bonferroni m=%d, alpha=%.4f) ===" % (m, alpha_family))
    for r in rows:
        if r["stage"] != "principal":
            continue
        s = r["stats"]
        print(f"{label(r['a']):>26} x {label(r['b']):<26} {r['win_rate_a']:.3f} "
              f"W95[{s['wilson95'][0]:.3f},{s['wilson95'][1]:.3f}] "
              f"Wbonf[{s['wilson_bonf'][0]:.3f},{s['wilson_bonf'][1]:.3f}] "
              f"p={s['p_binom']:.4f} sig={s['sig_bonf']}")
    print("ranking:", [label(k) for k in ranking])
    print("significativamente melhor que todos:", [label(k) for k in best_sig] or "NENHUM")
    print(f"gravado em {out}")


if __name__ == "__main__":
    main()
