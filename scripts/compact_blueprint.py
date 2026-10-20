"""Converte um blueprint (policy.tsv do Rust ou blueprint.pkl) para o formato compacto.

Uso:
    python scripts/compact_blueprint.py runs/mccfr_rs_o1_s7 --min-visits 30
    python scripts/compact_blueprint.py runs/mccfr_rs_v1_s1            # a partir do .pkl

Grava em ``<run>/compact`` (ou ``--out``); nunca sobrescreve uma pasta existente.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from trucoai.cfr.compact import build_from_pkl, build_from_tsv  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("run")
    ap.add_argument("--min-visits", type=int, default=30)
    ap.add_argument("--out", default=None)
    a = ap.parse_args()
    run = Path(a.run)
    out = Path(a.out) if a.out else run / "compact"
    t0 = time.time()
    tsv = run / "policy.tsv"
    if tsv.exists():
        cfg = json.loads((run / "config.json").read_text(encoding="utf-8"))
        summ = cfg.get("summary", {})
        meta = {"run_id": cfg.get("run_id"), "rules_version": cfg.get("rules_version"),
                "rules_fingerprint": cfg.get("rules_fingerprint"),
                "iterations": summ.get("iterations"), "seed": cfg.get("seed"),
                "equity_dist": cfg.get("equity_dist"), "source": str(tsv)}
        n = build_from_tsv(tsv, out, a.min_visits, meta)
    else:
        n = build_from_pkl(run / "blueprint.pkl", out, a.min_visits)
    size = sum(f.stat().st_size for f in out.iterdir()) / 1e6
    print(f"ok: {out} nós={n} ({size:.0f} MB) em {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main()
