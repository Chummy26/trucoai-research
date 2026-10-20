"""Treino "turbo": núcleo CPU v2 (Rust) e kernel GPU em ondas, em paralelo, com mescla.

Mesmo método do treino atual (ES-MCCFR, RM+, média linear, utilidade = Δ equidade,
mesma abstração e regras), com:
  * CPU: tabela sem travas + filtro de admissão + poda estilo Pluribus (rust/mccfr/src/v2.rs);
  * GPU: travessia em ondas (uma thread por nó de cada profundidade) com a mesma poda
    (trucoai/gpu/mccfr_kernel.cu);
  * mescla: as duas tabelas são treinadas de forma independente (sementes próprias) e
    combinadas no fim — estratégia média ponderada pelas visitas de cada lado
    (aproximação: média de execuções independentes, como no E2).

Uso:
    python scripts/train_turbo.py --minutes 20 --run-id turbo_o1_s11 --rules o1
    python scripts/train_turbo.py --minutes 5 --run-id x --no-gpu        # só CPU
"""

from __future__ import annotations

import argparse
import json
import os
import platform
import subprocess
import sys
import threading
import time
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from trucoai import __version__  # noqa: E402
from trucoai.cfr.compact import finalize_unsorted  # noqa: E402
from trucoai.equity import default_table  # noqa: E402
from trucoai.rules import DEFAULT_RULES, ONLINE_O1  # noqa: E402

CRATE = ROOT / "rust" / "mccfr"
BIN = CRATE / "target" / "release" / ("mccfr.exe" if sys.platform == "win32" else "mccfr")
RULESETS = {"h0": DEFAULT_RULES, "o1": ONLINE_O1}


def merge_unsorted(parts, out_dir: Path) -> dict:
    """Combina exportações ``*_unsorted.npy``: probabilidades ponderadas por visitas."""
    keys = np.concatenate([np.load(p / "keys_unsorted.npy") for p in parts])
    visits = np.concatenate([np.load(p / "visits_unsorted.npy").astype(np.float64) for p in parts])
    probs = np.concatenate([np.load(p / "probs_unsorted.npy").astype(np.float64) for p in parts])
    order = np.argsort(keys, kind="stable")
    keys, visits, probs = keys[order], visits[order], probs[order]
    uniq, start = np.unique(keys, return_index=True)
    w = probs * visits[:, None]
    wsum = np.add.reduceat(w, start, axis=0)
    vsum = np.add.reduceat(visits, start)
    p = wsum / np.maximum(wsum.sum(1, keepdims=True), 1e-12)
    q = np.round(p * 255).astype(np.uint8)
    zero = q.sum(1) == 0
    q[zero, p[zero].argmax(1)] = 1
    out_dir.mkdir(parents=True, exist_ok=True)
    np.save(out_dir / "keys_unsorted.npy", uniq)
    np.save(out_dir / "visits_unsorted.npy", np.minimum(vsum, 2**32 - 1).astype(np.uint32))
    np.save(out_dir / "probs_unsorted.npy", q)
    return {"nodes_total": int(len(keys)), "nodes_merged": int(len(uniq)),
            "shared_nodes": int(len(keys) - len(uniq))}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-id", required=True)
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--rules", choices=sorted(RULESETS), default="o1")
    ap.add_argument("--seed", type=int, default=11)
    ap.add_argument("--cpu-threads", type=int, default=(os.cpu_count() or 16) - 1)
    ap.add_argument("--log2-cap", type=int, default=25)
    ap.add_argument("--gate", type=int, default=4)
    ap.add_argument("--prune", type=float, default=0.95)
    ap.add_argument("--prune-after", type=int, default=2_000_000)
    ap.add_argument("--save-min-visits", type=int, default=30)
    ap.add_argument("--gpu-batch", type=int, default=16384)
    ap.add_argument("--abstraction", choices=["v1", "v2a", "v2b"], default="v1")
    ap.add_argument("--no-gpu", action="store_true")
    ap.add_argument("--no-cpu", action="store_true")
    a = ap.parse_args()

    out = ROOT / "runs" / a.run_id
    if out.exists():
        raise SystemExit(f"{out} já existe: não sobrescrevo artefatos de experimentos")
    subprocess.run(["cargo", "build", "--release"], cwd=CRATE, check=True)
    out.mkdir(parents=True)
    rules = RULESETS[a.rules]
    eq = default_table()
    eq_txt = out / "equity_table.txt"
    eq_txt.write_text("\n".join(" ".join(repr(eq(x, y)) for y in range(12)) for x in range(12)) + "\n")
    secs = a.minutes * 60
    cfg = {"run_id": a.run_id, "engine": "turbo (cpu-v2 + gpu-wavefront)", "args": vars(a),
           "abstraction": a.abstraction,
           "rules_version": rules.version, "rules_fingerprint": rules.fingerprint,
           "equity_dist": eq.dist, "code_version": __version__,
           "python": platform.python_version(), "started": time.strftime("%Y-%m-%d %H:%M:%S")}
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))

    parts, results = [], {}
    cpu_proc = None
    if not a.no_cpu:
        cpu_cmd = [str(BIN), "train2", "--rules", a.rules, "--eq", str(eq_txt),
                   "--out-dir", str(out / "cpu"), "--summary", str(out / "cpu_summary.json"),
                   "--seed", str(a.seed), "--threads", str(a.cpu_threads), "--seconds", str(secs),
                   "--log2-cap", str(a.log2_cap), "--gate", str(a.gate), "--prune", str(a.prune),
                   "--prune-after", str(a.prune_after), "--save-min-visits", str(a.save_min_visits),
                   "--log-every", "60", "--abstraction", a.abstraction]
        print(" ".join(cpu_cmd), flush=True)
        cpu_proc = subprocess.Popen(cpu_cmd)
        parts.append(out / "cpu")
    if not a.no_gpu:
        from trucoai.gpu.train import WavefrontTrainer
        g = WavefrontTrainer(eq_txt, o1=a.rules == "o1", log2_cap=a.log2_cap, gate=a.gate,
                             seed=a.seed * 7919 + 1, batch=a.gpu_batch, prune=a.prune,
                             prune_after=a.prune_after, abstraction=a.abstraction)
        results["gpu"] = {**g.train(secs), **g.summary()}
        n = g.export_unsorted(out / "gpu", a.save_min_visits)
        results["gpu"]["saved_nodes"] = n
        print(f"[gpu] {results['gpu']}", flush=True)
        del g
        parts.append(out / "gpu")
    if cpu_proc is not None:
        if cpu_proc.wait() != 0:
            raise SystemExit("treino CPU falhou")
        results["cpu"] = json.loads((out / "cpu_summary.json").read_text())

    meta = {"run_id": a.run_id, "rules_version": rules.version, "abstraction": a.abstraction,
            "rules_fingerprint": rules.fingerprint, "equity_dist": eq.dist,
            "iterations": sum(r["iterations"] for r in results.values()),
            "sources": results, "merge": "probabilidades ponderadas por visitas"}
    if len(parts) > 1:
        meta["merge_stats"] = merge_unsorted(parts, out / "compact")
    else:
        import shutil
        shutil.copytree(parts[0], out / "compact")
    n = finalize_unsorted(out / "compact", meta)
    cfg.update({"finished": time.strftime("%Y-%m-%d %H:%M:%S"), "results": results, "saved_nodes": n,
                "iterations_total": meta["iterations"]})
    (out / "config.json").write_text(json.dumps(cfg, indent=2, ensure_ascii=False))
    print(f"ok: {out / 'compact'} iterações={meta['iterations']} nós={n}")


if __name__ == "__main__":
    main()
