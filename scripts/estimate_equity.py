"""Estima a distribuição de pontos por mão em autojogo e grava data/equity_dist.json.

A tabela de equidade (trucoai/equity.py) usa essa distribuição. Rode de novo
quando o agente principal mudar muito; registre a versão usada nos experimentos.
"""

from __future__ import annotations

import argparse
import collections
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from trucoai.agents import make_agent  # noqa: E402
from trucoai.engine import NORMAL, new_match  # noqa: E402
from trucoai.observation import observe  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="heuristic")
    ap.add_argument("--matches", type=int, default=300)
    ap.add_argument("--seed", type=int, default=99)
    a = ap.parse_args()
    agents = [make_agent(a.agent), make_agent(a.agent)]
    for i, ag in enumerate(agents):
        ag.reset(a.seed + i)
    rng = random.Random(a.seed)
    counts = collections.Counter()
    for _ in range(a.matches):
        st = new_match()
        while st.match_winner is None:
            st.deal(rng)
            normal = st.special == NORMAL
            while not st.is_terminal():
                st.apply(agents[st.current].act(observe(st, st.current)))
            if normal:  # mãos especiais seguem regras próprias
                counts[st.hand_points] += 1
    tot = sum(counts.values())
    dist = {int(k): v / tot for k, v in sorted(counts.items())}
    out = ROOT / "data" / "equity_dist.json"
    out.parent.mkdir(exist_ok=True)
    out.write_text(json.dumps({"dist": dist, "target": 12, "source": vars(a), "hands": tot}, indent=2))
    print(dist, tot)


if __name__ == "__main__":
    main()
