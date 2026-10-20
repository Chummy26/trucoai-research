"""Interface comum de agentes.

Um agente só recebe :class:`~trucoai.observation.Observation`. Nunca recebe o
``GameState`` verdadeiro — isso garante as mesmas restrições de informação
contra bots e contra humanos (e na integração com uma interface externa).
"""

from __future__ import annotations

import random
from typing import Optional

from ..observation import Observation


class Agent:
    name = "agent"

    def reset(self, seed: Optional[int] = None) -> None:
        self.rng = random.Random(seed)

    def act(self, obs: Observation) -> int:
        raise NotImplementedError

    def __init__(self) -> None:
        self.rng = random.Random(0)


class RandomAgent(Agent):
    name = "random"

    def act(self, obs: Observation) -> int:
        return self.rng.choice(obs.legal_actions)
