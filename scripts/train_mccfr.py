"""Treina o blueprint MCCFR em paralelo e grava em runs/<id>/.

Cada processo treina uma tabela independente (semente própria) pelo tempo
dado; ao fim as tabelas são somadas (média de execuções independentes — uma
aproximação documentada em docs/ARQUITETURA.md). Nós com poucas visitas são
descartados antes da soma para caber em memória.

Uso:
    python scripts/train_mccfr.py --minutes 20 --workers 12 --seed 0
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from trucoai import __version__  # noqa: E402
from trucoai.cfr.mccfr import VISITS, MCCFRTrainer  # noqa: E402
from trucoai.rules import DEFAULT_RULES  # noqa: E402


def _worker(args):
    seed, seconds, keep, max_nodes = args
    tr = MCCFRTrainer(seed=seed)
    tr.train(10**12, time_limit=seconds, max_nodes=max_nodes)
    nodes = {k: n for k, n in tr.nodes.items() if n[VISITS] >= keep}
    return tr.iterations, len(tr.nodes), nodes


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--minutes", type=float, default=10)
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--keep-visits", type=int, default=3)
    ap.add_argument("--max-nodes", type=int, default=1_200_000)
    ap.add_argument("--save-min-visits", type=int, default=10)
    ap.add_argument("--run-id", default=None)
    a = ap.parse_args()
    run_id = a.run_id or time.strftime("mccfr_%Y%m%d_%H%M%S")
    out = ROOT / "runs" / run_id
    if out.exists():
        raise SystemExit(f"{out} já existe: não sobrescrevo artefatos de experimentos")
    out.mkdir(parents=True)
    t0 = time.time()
    master = MCCFRTrainer(seed=a.seed)
    total_it, raw_nodes = 0, 0
    seeds = [a.seed * 1000 + i for i in range(a.workers)]
    with ProcessPoolExecutor(a.workers) as ex:
        for it, n_all, nodes in ex.map(_worker, [(s, a.minutes * 60, a.keep_visits, a.max_nodes) for s in seeds]):
            total_it += it
            raw_nodes += n_all
            master.merge(nodes)
            print(f"worker: {it} iterações, {n_all} nós ({len(nodes)} mantidos)", flush=True)
    master.iterations = total_it
    master.elapsed = time.time() - t0
    meta = {"run_id": run_id, "workers": a.workers, "worker_seeds": seeds,
            "minutes_per_worker": a.minutes, "keep_visits": a.keep_visits,
            "max_nodes_per_worker": a.max_nodes,
            "save_min_visits": a.save_min_visits, "raw_nodes_total": raw_nodes,
            "code_version": __version__, "python": platform.python_version(),
            "machine": platform.processor(), "rules": DEFAULT_RULES.to_dict()}
    master.save(out / "blueprint.pkl", meta, min_visits=a.save_min_visits)
    (out / "config.json").write_text(json.dumps(meta, indent=2, ensure_ascii=False))
    print(f"ok: {out} iterações={total_it} nós salvos (>= {a.save_min_visits} visitas)")


if __name__ == "__main__":
    main()
