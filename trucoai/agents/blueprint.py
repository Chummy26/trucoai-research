"""Agentes que usam a política média treinada pelo MCCFR.

``BlueprintAgent``: amostra da política média; cai para o heurístico quando o
conjunto de informação não foi visitado o suficiente.

``HybridAgent`` (candidato principal): apostas (truco/aceitar/correr/mão de
onze) vêm do blueprint — que aprende blefes e frequências equilibradas — e a
escolha de carta vem da busca PIMC, que é mais precisa no jogo de cartas.
"""

from __future__ import annotations

import pickle
from pathlib import Path
from typing import Optional

from ..cfr.abstraction import info_key, key_fn
from ..cfr.mccfr import to_concrete
from ..engine import ACCEPT, RAISE, RUN
from ..observation import Observation
from .base import Agent
from .heuristic import HeuristicAgent
from .pimc import PIMCAgent

BET_ACTIONS = {RAISE, ACCEPT, RUN}

# Cache por processo: o pkl do blueprint é grande (~650 MB em RAM) e a arena
# recria os agentes a cada bloco de partidas. O conteúdo é só leitura.
_BLOB_CACHE: dict = {}


def _load_blob(path) -> dict:
    key = str(Path(path).resolve())
    if key not in _BLOB_CACHE:
        with open(path, "rb") as f:
            _BLOB_CACHE[key] = pickle.load(f)
    return _BLOB_CACHE[key]


class _DictPolicy:
    """Blueprint em pickle (dict de strings)."""

    def __init__(self, blob: dict):
        self.policy, self.visits, self.meta = blob["policy"], blob["visits"], blob["meta"]

    def lookup(self, key: str):
        return self.policy.get(key), self.visits.get(key, 0)

    def __len__(self) -> int:
        return len(self.policy)


def load_policy(path):
    """Pasta compacta (``keys.npy``...) ou ``blueprint.pkl``; cache por processo."""
    p = Path(path)
    key = str(p.resolve())
    if key not in _BLOB_CACHE:
        if p.is_dir():
            from ..cfr.compact import CompactPolicy
            _BLOB_CACHE[key] = CompactPolicy(p)
        else:
            _BLOB_CACHE[key] = _DictPolicy(_load_blob(p))
    return _BLOB_CACHE[key]


class BlueprintAgent(Agent):
    name = "blueprint"

    def __init__(self, path: Path, min_visits: float = 30.0, fallback: Optional[Agent] = None,
                 expected_rules: Optional[str] = None):
        super().__init__()
        self.store = load_policy(path)
        self.meta = self.store.meta
        self.key = key_fn(self.meta.get("abstraction", "v1"))  # versão da chave do blueprint
        if expected_rules and self.meta["rules_fingerprint"] != expected_rules:
            raise ValueError("checkpoint treinado com outro regulamento")
        self.min_visits = min_visits
        self.fallback = fallback or HeuristicAgent()
        self.hits = 0
        self.misses = 0

    def reset(self, seed=None):
        super().reset(seed)
        self.fallback.reset(seed)

    def lookup(self, obs: Observation):
        dist, visits = self.store.lookup(self.key(obs))
        if dist is None or visits < self.min_visits:
            return None
        return dist

    def _sample(self, obs: Observation, dist) -> Optional[int]:
        acts = list(dist)
        a = self.rng.choices(acts, weights=[dist[x] for x in acts])[0]
        concrete = to_concrete(obs, a)
        return concrete if concrete in obs.legal_actions else None

    def act(self, obs: Observation) -> int:
        if len(obs.legal_actions) == 1:
            return obs.legal_actions[0]
        dist = self.lookup(obs)
        if dist is not None:
            a = self._sample(obs, dist)
            if a is not None:
                self.hits += 1
                return a
        self.misses += 1
        return self.fallback.act(obs)


class HybridAgent(BlueprintAgent):
    name = "hybrid"

    def __init__(self, path: Path, min_visits: float = 30.0, pimc_n: int = 24, **kw):
        super().__init__(path, min_visits, fallback=PIMCAgent(n=pimc_n), **kw)

    def act(self, obs: Observation) -> int:
        legal = obs.legal_actions
        if len(legal) == 1:
            return legal[0]
        dist = self.lookup(obs)
        if dist is not None:
            a = self._sample(obs, dist)
            # Apostas: segue o blueprint. Na fase de jogo, se o blueprint
            # escolheu não pedir truco, a carta vem da busca.
            if a is not None and (a in BET_ACTIONS or obs.phase != "play"):
                self.hits += 1
                return a
            if a is not None:
                vals = self.fallback.action_values(obs)
                vals.pop(RAISE, None)
                self.hits += 1
                return max(vals, key=vals.get)
        self.misses += 1
        return self.fallback.act(obs)
