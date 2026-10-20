"""Teste A/B: cada candidato contra um agente de referência (campeão atual).

Partidas espelhadas sob PLAYTRUCO_O1, sementes de avaliação disjuntas do treino,
estatística pareada por distribuição (reaproveita ``compare_references``) e
Bonferroni sobre o número de candidatos. Grava JSON em results/ (nome novo).

Uso:
    python scripts/ab_test.py --baseline "hybrid:runs/X/compact@2000" \
        --candidates "search:runs/X/compact" "search:runs/X/compact?belief=0" \
        --deals 300 --start-seed 40000 --workers 4 --tag e5_triagem
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(ROOT / "scripts"))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from compare_references import pair_stats  # noqa: E402

from trucoai import __version__  # noqa: E402
from trucoai.agents import make_agent  # noqa: E402
from trucoai.arena import run_matchup  # noqa: E402
from trucoai.rules import PLAYTRUCO_O1  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--baseline", nargs="+", required=True,
                    help="um ou mais adversários; com vários, vira um torneio candidato x adversário")
    ap.add_argument("--candidates", nargs="+", required=True)
    ap.add_argument("--deals", type=int, default=300)
    ap.add_argument("--start-seed", type=int, default=40000)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--tag", default="ab")
    a = ap.parse_args()
    out = ROOT / "results" / f"{time.strftime('%Y%m%d_%H%M%S')}_{a.tag}.json"
    alpha = 0.05 / (len(a.candidates) * len(a.baseline))
    rows = []

    def save():
        out.write_text(json.dumps({
            "meta": {"code_version": __version__, "rules": PLAYTRUCO_O1.version,
                     "rules_fingerprint": PLAYTRUCO_O1.fingerprint, "args": vars(a),
                     "alpha_bonferroni": alpha,
                     "nota": "simulação contra bots; não é evidência contra humanos"},
            "matchups": rows}, indent=2, ensure_ascii=False), encoding="utf-8")

    for c, base in [(c, b) for c in a.candidates for b in a.baseline]:
        t0 = time.time()
        r = run_matchup(make_agent, c, base, deals=a.deals, start_seed=a.start_seed,
                        workers=a.workers, rules=PLAYTRUCO_O1)
        r["wall_s"] = round(time.time() - t0, 1)
        r["stats"] = pair_stats(r, alpha)
        rows.append(r)
        save()
        s = r["stats"]
        lo, hi = r["ci95"]
        print(f"{c}  x  {base}\n   vitória={r['win_rate_a']:.3f} [{lo:.3f},{hi:.3f}] "
              f"pts/mão={r['point_diff_per_hand_a']:+.2f} p={s['p_binom']:.4f} "
              f"signif(Bonferroni)={s['sig_bonf']} ms/dec={r['a_metrics']['ms_per_decision']:.0f} "
              f"({r['wall_s']}s)", flush=True)
    if len(a.baseline) > 1:
        print("\nmédia de vitória contra os adversários:")
        for c in a.candidates:
            rs = [r for r in rows if r["a"] == c]
            print(f"  {sum(r['win_rate_a'] for r in rs) / len(rs):.3f}  {c}")
    print(f"gravado em {out}")


if __name__ == "__main__":
    main()
