"""Torneio entre agentes com partidas espelhadas; grava JSON em results/.

Uso:
    python scripts/evaluate.py --agents random heuristic pimc hybrid:runs/X/blueprint.pkl \
        --deals 100 --workers 8 --tag prelim

Cada confronto usa ``deals`` sequências de distribuição, cada uma jogada nos
dois assentos (2*deals partidas). Resultados são de SIMULAÇÃO contra bots e não
são evidência de desempenho contra humanos.
"""

from __future__ import annotations

import argparse
import itertools
import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from trucoai import __version__  # noqa: E402
from trucoai.agents import make_agent  # noqa: E402
from trucoai.arena import run_matchup  # noqa: E402
from trucoai.rules import DEFAULT_RULES  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agents", nargs="+", required=True)
    ap.add_argument("--deals", type=int, default=100)
    ap.add_argument("--workers", type=int, default=4)
    ap.add_argument("--start-seed", type=int, default=0)
    ap.add_argument("--tag", default="eval")
    a = ap.parse_args()
    out_dir = ROOT / "results"
    out_dir.mkdir(exist_ok=True)
    out = out_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{a.tag}.json"
    rows = []
    for x, y in itertools.combinations(a.agents, 2):
        t0 = time.time()
        r = run_matchup(make_agent, x, y, deals=a.deals, start_seed=a.start_seed,
                        workers=a.workers)
        r["wall_s"] = round(time.time() - t0, 1)
        rows.append(r)
        lo, hi = r["ci95"]
        print(f"{x:>40} x {y:<40} vitória A={r['win_rate_a']:.3f} [{lo:.3f},{hi:.3f}] "
              f"pts/mão={r['point_diff_per_hand_a']:+.2f}±{1.96 * r['point_diff_se']:.2f} "
              f"ms/dec A={r['a_metrics']['ms_per_decision']:.1f} ({r['wall_s']}s)", flush=True)
    meta = {"code_version": __version__, "rules_fingerprint": DEFAULT_RULES.fingerprint,
            "rules_version": DEFAULT_RULES.version, "args": vars(a),
            "nota": "simulação contra bots; não é evidência contra humanos"}
    out.write_text(json.dumps({"meta": meta, "matchups": rows}, indent=2, ensure_ascii=False))
    print(f"gravado em {out}")


if __name__ == "__main__":
    main()
