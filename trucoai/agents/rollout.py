"""Políticas rápidas para simulação e estimativa de força de mão.

Estas funções operam sobre estados *amostrados* (``sample_state``), nunca
sobre o estado verdadeiro de uma partida em andamento. Em um estado amostrado,
cada jogador simulado usa apenas a própria mão e a mesa.
"""

from __future__ import annotations

import random
from typing import Optional

from .. import cards as C
from ..engine import ACCEPT, RAISE, RUN, GameState
from ..equity import default_table
from ..observation import Observation, sample_state


def _slots(state: GameState, p: int):
    return [(i, c) for i, c in enumerate(state.hands[p]) if c is not None]


def greedy_card(state: GameState, rng: Optional[random.Random] = None) -> int:
    """Escolhe uma carta com regras simples, usando só a própria mão e a mesa."""
    p, vira = state.current, state.vira
    slots = _slots(state, p)
    slots.sort(key=lambda ic: C.strength(ic[1], vira))
    table = state.rounds[-1]
    ri = state.round_index
    can_cover = (state.rules.allow_covered and ri >= state.rules.covered_from_round
                 and state.special != "iron")
    if table:  # respondendo à carta adversária
        _, oc, ocov = table[0]
        os_ = -1 if ocov else C.strength(oc, vira)
        beating = [ic for ic in slots if C.strength(ic[1], vira) > os_]
        tying = [ic for ic in slots if C.strength(ic[1], vira) == os_]
        if beating:
            # Rodada 1 com vitória prévia minha: empatar basta? ganhar é melhor.
            return beating[0][0]
        if tying and (ri > 0 and state.round_results[0] == p or ri == 0):
            return tying[0][0]
        weakest = slots[0][0]
        return 3 + weakest if can_cover else weakest
    # abrindo a rodada
    if ri == 0:
        if len(slots) >= 2 and C.strength(slots[-1][1], vira) >= 12 \
                and C.strength(slots[-2][1], vira) >= 8:
            return slots[-2][0]
        return slots[-1][0]
    if state.round_results and state.round_results[0] == p and ri == 1:
        return slots[0][0]  # ganhei a primeira: guardo a forte
    return slots[-1][0]


def playout(state: GameState, rng: random.Random, respond_fn=None) -> GameState:
    """Joga até o fim da mão sem novos aumentos; responde pedidos pendentes."""
    s = state
    while not s.is_terminal():
        ph = s.phase
        if ph == "play":
            s.apply(greedy_card(s, rng))
        elif ph == "respond":
            s.apply(respond_fn(s, rng) if respond_fn else ACCEPT)
        else:  # mao11
            s.apply(ACCEPT)
    return s


def win_prob(obs: Observation, rng: random.Random, n: int = 40) -> float:
    """P(vencer a mão) estimada por determinização + playouts gulosos.

    Empates contam como meia vitória.
    """
    me = obs.me
    tot = 0.0
    for _ in range(n):
        s = sample_state(obs, rng)
        s.pending = None
        if s.phase in ("respond", "mao11"):
            s.phase = "play"
            if s.resume_player is not None:
                s.current = s.resume_player
            elif obs.phase == "mao11":
                s.current = s.leader
        s.resume_player = s.raiser = None
        playout(s, rng)
        if s.hand_winner is None:
            tot += 0.5
        elif s.hand_winner == me:
            tot += 1.0
    return tot / n


def equity_after(state: GameState, player: int) -> float:
    eq = default_table()
    return eq(state.scores[player], state.scores[1 - player])


def ev_accept(p: float, me: int, opp: int, value: int) -> float:
    eq = default_table()
    return p * eq(me + value, opp) + (1 - p) * eq(me, opp + value)
