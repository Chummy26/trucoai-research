"""Partida local humano x IA no terminal (para testes e estudos com consentimento).

O humano é informado de que joga contra uma IA. Nenhum dado é gravado, a menos
que ``--log`` seja passado (grava só ações e cartas, sem identificadores).

Uso:
    python scripts/play_cli.py --agent heuristic
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))
if hasattr(sys.stdout, "reconfigure"):
    sys.stdout.reconfigure(encoding="utf-8")

from trucoai import cards as C  # noqa: E402
from trucoai.agents import make_agent  # noqa: E402
from trucoai.engine import ACTION_NAMES, new_match  # noqa: E402
from trucoai.observation import observe  # noqa: E402


def show(obs):
    hand = " ".join(f"[{i}]{'??' if obs.blind else C.name(c)}" for i, c in enumerate(obs.my_hand)
                    if obs.my_slots_filled[i])
    table = " ".join("encoberta" if c < 0 else C.name(c) for _, c in (obs.rounds[-1] if obs.rounds else ()))
    print(f"\nPlacar você {obs.scores[obs.me]} x {obs.scores[obs.opp]} IA | vira {C.name(obs.vira)} "
          f"| mão vale {obs.stake}{' (pedido: ' + str(obs.pending) + ')' if obs.pending else ''}")
    print(f"Rodadas: {['você' if r == obs.me else 'IA' if r == obs.opp else 'empate' for r in obs.round_results]}")
    print(f"Mesa: {table or '-'} | Sua mão: {hand}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--agent", default="heuristic")
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--log", default=None)
    a = ap.parse_args()
    print("Aviso: você vai jogar contra uma INTELIGÊNCIA ARTIFICIAL (projeto de pesquisa).")
    ai = make_agent(a.agent)
    ai.reset(a.seed)
    rng = random.Random(a.seed)
    human = rng.randrange(2)
    st = new_match()
    log = []
    while st.match_winner is None:
        st.deal(rng)
        while not st.is_terminal():
            p = st.current
            obs = observe(st, p)
            if p == human:
                show(obs)
                opts = {str(x): ACTION_NAMES[x] for x in obs.legal_actions}
                choice = None
                while choice not in opts:
                    choice = input(f"Ação {opts}: ").strip()
                act = int(choice)
            else:
                act = ai.act(obs)
                print(f"IA: {ACTION_NAMES[act]}")
            log.append({"hand": st.hand_no, "human": p == human, "action": act})
            st.apply(act)
        print(f"Fim da mão: {'você' if st.hand_winner == human else 'IA' if st.hand_winner is not None else 'ninguém'}"
              f" (+{st.hand_points})")
    print("Você venceu!" if st.match_winner == human else "A IA venceu.")
    if a.log:
        Path(a.log).write_text(json.dumps(log))


if __name__ == "__main__":
    main()
