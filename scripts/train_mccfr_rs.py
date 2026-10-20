"""Treina o blueprint MCCFR com o núcleo Rust (rust/mccfr) e grava em runs/<id>/.

Mesmo algoritmo, abstração e chave de `trucoai/cfr` (equivalência verificada em
tests/test_rust_core.py), ~100x mais rápido por núcleo. As threads compartilham
uma tabela (CFR assíncrono) em vez de somar tabelas independentes no fim.

Grava em runs/<id>/: config.json (argumentos, sementes, regras, versão do
código, hash do binário e da distribuição de equidade), equity_dist.json
(cópia da usada), equity_table.txt, summary.json (saída do Rust: iterações,
histograma de visitas, podas), policy.tsv (opcional, --keep-tsv) e
blueprint.pkl (formato de `MCCFRTrainer.save`, lido por BlueprintAgent).

Uso:
    python scripts/train_mccfr_rs.py --minutes 30 --threads 6 --seed 1 \
        --max-nodes 6000000 --run-id mccfr_rs_v1_s1
"""

from __future__ import annotations

import argparse
import hashlib
import json
import pickle
import platform
import shutil
import subprocess
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from trucoai import __version__  # noqa: E402
from trucoai.equity import default_table  # noqa: E402
from trucoai.rules import DEFAULT_RULES, PLAYTRUCO_O1  # noqa: E402

RULESETS = {"h0": DEFAULT_RULES, "o1": PLAYTRUCO_O1}

CRATE = ROOT / "rust" / "mccfr"
BIN = CRATE / "target" / "release" / ("mccfr.exe" if sys.platform == "win32" else "mccfr")
EQ_FILE = ROOT / "data" / "equity_dist.json"


def sha256(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def load_policy_tsv(path: Path, min_visits: int):
    policy, visits = {}, {}
    with open(path, encoding="ascii") as f:
        for line in f:
            key, v, dist = line.rstrip("\n").split("\t")
            v = int(v)
            if v < min_visits:
                continue
            policy[key] = {int(a): float(p) for a, p in (x.split("=") for x in dist.split(","))}
            visits[key] = v
    return policy, visits


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--threads", type=int, default=6)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--iterations", type=int, default=None)
    ap.add_argument("--max-nodes", type=int, default=6_000_000)
    ap.add_argument("--dump-min-visits", type=int, default=100,
                    help="limiar para policy.tsv (Rust)")
    ap.add_argument("--save-min-visits", type=int, default=300,
                    help="limiar para blueprint.pkl")
    ap.add_argument("--keep-tsv", action="store_true")
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--rules", choices=sorted(RULESETS), default="h0",
                    help="h0 (padrão histórico) ou o1 (observado na PlayTruco)")
    a = ap.parse_args()

    out = ROOT / "runs" / a.run_id
    if out.exists():
        raise SystemExit(f"{out} já existe: não sobrescrevo artefatos de experimentos")
    rules = RULESETS[a.rules]
    subprocess.run(["cargo", "build", "--release"], cwd=CRATE, check=True)
    out.mkdir(parents=True)

    eq = default_table()
    if EQ_FILE.exists():
        shutil.copy(EQ_FILE, out / "equity_dist.json")
    (out / "equity_table.txt").write_text(
        "\n".join(" ".join(repr(eq(x, y)) for y in range(12)) for x in range(12)) + "\n")
    cfg = {
        "run_id": a.run_id, "engine": "rust/mccfr", "args": vars(a),
        "threads": a.threads, "seed": a.seed,
        "thread_seeds": f"seed*1000003 + i, i=0..{a.threads - 1} (xoshiro256**)",
        "equity_dist_file": str(EQ_FILE.relative_to(ROOT)) if EQ_FILE.exists() else None,
        "equity_dist_sha256": sha256(EQ_FILE) if EQ_FILE.exists() else None,
        "equity_dist": eq.dist,
        "binary_sha256": sha256(BIN), "rust_src_sha256": sha256(CRATE / "src" / "main.rs"),
        "code_version": __version__, "python": platform.python_version(),
        "machine": platform.processor(),
        "rules_version": rules.version, "rules_fingerprint": rules.fingerprint,
        "rules": rules.to_dict(),
        "started": time.strftime("%Y-%m-%d %H:%M:%S"),
    }
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    cmd = [str(BIN), "train", "--eq", str(out / "equity_table.txt"),
           "--out", str(out / "policy.tsv"), "--summary", str(out / "summary.json"),
           "--seed", str(a.seed), "--threads", str(a.threads),
           "--seconds", str(a.minutes * 60), "--max-nodes", str(a.max_nodes),
           "--save-min-visits", str(a.dump_min_visits), "--log-every", "60",
           "--rules", a.rules]
    if a.iterations:
        cmd += ["--iterations", str(a.iterations)]
    print(" ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)

    summ = json.loads((out / "summary.json").read_text())
    if summ["rules_version"] != rules.version:
        raise SystemExit(f"binário treinou {summ['rules_version']}, esperado {rules.version}")
    policy, visits = load_policy_tsv(out / "policy.tsv", a.save_min_visits)
    meta = {
        "rules_version": rules.version, "rules_fingerprint": rules.fingerprint,
        "iterations": summ["iterations"], "seed": a.seed, "elapsed_s": summ["elapsed_s"],
        "nodes": summ["nodes_final"],
        "algorithm": "ES-MCCFR + RM+ + linear averaging (Rust, tabela compartilhada entre "
                     "threads); utility = match equity delta",
        "equity_dist": eq.dist, "run_id": a.run_id, "save_min_visits": a.save_min_visits,
        "saved_nodes": len(policy), "code_version": __version__,
    }
    with open(out / "blueprint.pkl", "wb") as f:
        pickle.dump({"policy": policy, "visits": visits, "meta": meta}, f)
    if not a.keep_tsv:
        (out / "policy.tsv").unlink()
    cfg["finished"] = time.strftime("%Y-%m-%d %H:%M:%S")
    cfg["summary"] = summ
    cfg["saved_nodes_pkl"] = len(policy)
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    print(f"ok: {out} iterações={summ['iterations']} nós no pkl={len(policy)} "
          f"(>= {a.save_min_visits} visitas)")


if __name__ == "__main__":
    main()
