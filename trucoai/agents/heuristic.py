"""Agente heurístico baseado em equidade da partida.

Melhor que os ``RulePlayer``/``RationalAgent`` das referências porque decide
aceitar/correr/aumentar comparando a equidade esperada de cada opção, usando a
probabilidade de vencer a mão estimada por determinização da observação.
"""

from __future__ import annotations

from ..engine import ACCEPT, RAISE, RUN
from ..equity import default_table
from ..observation import Observation, sample_state
from .base import Agent
from .rollout import ev_accept, greedy_card, win_prob


class HeuristicAgent(Agent):
    name = "heuristic"

    def __init__(self, sims: int = 40, raise_margin: float = 0.08, bluff: float = 0.06):
        super().__init__()
        self.sims = sims
        self.raise_margin = raise_margin
        self.bluff = bluff

    def act(self, obs: Observation) -> int:
        legal = obs.legal_actions
        if len(legal) == 1:
            return legal[0]
        eq = default_table()
        me, opp = obs.scores[obs.me], obs.scores[obs.opp]
        if obs.phase == "mao11":
            p = win_prob(obs, self.rng, self.sims)
            play = ev_accept(p, me, opp, obs.rules.mao11_value)
            return ACCEPT if play >= eq(me, opp + obs.rules.mao11_run_points) else RUN
        if obs.phase == "respond":
            p = win_prob(obs, self.rng, self.sims)
            run = eq(me, opp + obs.stake)
            acc = ev_accept(p, me, opp, obs.pending)
            if RAISE in legal and p > 0.75 + self.raise_margin:
                return RAISE
            return ACCEPT if acc >= run else RUN
        # fase de jogo
        if RAISE in legal:
            p = win_prob(obs, self.rng, self.sims)
            if p > 0.70 + self.raise_margin or (p < 0.25 and self.rng.random() < self.bluff):
                return RAISE
        state = sample_state(obs, self.rng)  # só a própria mão e a mesa importam
        return greedy_card(state, self.rng)
