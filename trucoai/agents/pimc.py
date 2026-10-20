"""Busca por determinização (Perfect Information Monte Carlo) sobre a observação.

Para cada ação legal, sorteia ``n`` mundos compatíveis com a observação
(``sample_state``), aplica a ação e simula até o fim da mão. O valor é a
equidade da partida após a mão (não pontos). O oponente simulado responde a
pedidos comparando equidades com a sua probabilidade estimada de vitória.

Limitações conhecidas (documentadas): PIMC sofre de *strategy fusion* e não
blefa por conta própria; por isso o agente híbrido (``hybrid.py``) usa o
blueprint MCCFR para apostas quando ele cobre o conjunto de informação.
"""

from __future__ import annotations

import random

from ..engine import ACCEPT, RUN, GameState
from ..equity import default_table
from ..observation import Observation, observe, sample_state
from .base import Agent
from .rollout import ev_accept, playout, win_prob


def _respond_model(nested: int):
    def respond(state: GameState, rng: random.Random) -> int:
        p_obs = observe(state, state.current)
        p = win_prob(p_obs, rng, nested)
        eq = default_table()
        me, opp = state.scores[state.current], state.scores[1 - state.current]
        acc = ev_accept(p, me, opp, state.pending)
        return ACCEPT if acc >= eq(me, opp + state.stake) else RUN
    return respond


class PIMCAgent(Agent):
    name = "pimc"

    def __init__(self, n: int = 24, nested: int = 6):
        super().__init__()
        self.n = n
        self.respond = _respond_model(nested)

    def action_values(self, obs: Observation):
        eq = default_table()
        me = obs.me
        vals = {a: 0.0 for a in obs.legal_actions}
        for _ in range(self.n):
            world = sample_state(obs, self.rng)
            for a in obs.legal_actions:
                s = world.clone()
                s.apply(a)
                playout(s, self.rng, self.respond)
                vals[a] += eq(s.scores[me], s.scores[1 - me])
        return {a: v / self.n for a, v in vals.items()}

    def act(self, obs: Observation) -> int:
        if len(obs.legal_actions) == 1:
            return obs.legal_actions[0]
        vals = self.action_values(obs)
        return max(vals, key=vals.get)
