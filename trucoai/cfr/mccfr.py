"""MCCFR com amostragem externa, Regret Matching+ e média linear.

Combina o que funcionou nas referências:
  * amostragem externa (OpenSpiel ``external_sampling_mccfr``, filevich ESVMCCFR);
  * RM+ e média ponderada por iteração (CFR+/Linear CFR, como Deep-CFR/DCFR),
    que convergem bem mais rápido que o CFR vanilla usado no truco-mccfr-ai;
  * utilidade = variação de equidade da partida (ver ``equity.py``), em vez de
    pontos brutos;
  * contexto de placar amostrado na raiz, para cobrir mão de onze/ferro e
    situações de fim de partida.

O jogo treinado é a MÃO (do deal ao fim), com ações *abstratas*: jogar/cobrir a
k-ésima carta mais fraca restante, truco, aceitar, correr.
"""

from __future__ import annotations

import pickle
import random
import time
from array import array
from pathlib import Path
from typing import Dict, List, Optional, Tuple

from .. import cards as C
from ..engine import NUM_ACTIONS, GameState, new_match
from ..equity import default_table
from ..observation import Observation, observe
from ..rules import DEFAULT_RULES, RuleSet
from .abstraction import info_key


def sorted_slots(obs: Observation) -> List[int]:
    """Posições da mão ordenadas da carta mais fraca para a mais forte."""
    slots = [i for i in range(3) if obs.my_slots_filled[i]]
    if obs.blind:
        return slots
    return sorted(slots, key=lambda i: (C.strength(obs.my_hand[i], obs.vira), i))


def to_abstract(obs: Observation, action: int) -> int:
    if action >= 6:
        return action
    order = sorted_slots(obs)
    return (action // 3) * 3 + order.index(action % 3)


def to_concrete(obs: Observation, abstract: int) -> int:
    if abstract >= 6:
        return abstract
    order = sorted_slots(obs)
    return (abstract // 3) * 3 + order[abstract % 3]


def abstract_legal(obs: Observation) -> List[int]:
    return sorted(to_abstract(obs, a) for a in obs.legal_actions)


# Nó compacto: [regret x9 | média x9 | visitas] em float32 (~150 bytes/nó).
R0, A0, VISITS, NODE_LEN = 0, NUM_ACTIONS, 2 * NUM_ACTIONS, 2 * NUM_ACTIONS + 1


def new_node() -> array:
    return array("f", bytes(4 * NODE_LEN))


def regret_matching(node, legal: List[int]) -> List[float]:
    pos = [max(node[R0 + a], 0.0) for a in legal]
    tot = sum(pos)
    if tot <= 0:
        return [1.0 / len(legal)] * len(legal)
    return [x / tot for x in pos]


class MCCFRTrainer:
    def __init__(self, rules: RuleSet = DEFAULT_RULES, seed: int = 0):
        self.rules = rules
        self.rng = random.Random(seed)
        self.seed = seed
        self.nodes: Dict[str, array] = {}
        self.iterations = 0
        self.eq = default_table()
        self.elapsed = 0.0

    # Raiz: placar e quem dá as cartas sorteados (inclui 11 para mão de onze/ferro).
    def _sample_root(self) -> GameState:
        st = new_match(self.rules, first_dealer=self.rng.randrange(2))
        r = self.rng.random()
        t = self.rules.target_score
        if r < 0.5:
            a, b = self.rng.randrange(t), self.rng.randrange(t)
        elif r < 0.85:
            a, b = self.rng.randrange(min(4, t)), self.rng.randrange(min(4, t))
        else:
            a, b = t - 1, self.rng.randrange(t)
            if self.rng.random() < 0.5:
                a, b = b, a
        st.scores = [a, b]
        st.deal(self.rng)
        return st

    def _utility(self, s: GameState, p: int, before: Tuple[int, int]) -> float:
        return (self.eq(s.scores[p], s.scores[1 - p])
                - self.eq(before[p], before[1 - p]))

    def _traverse(self, s: GameState, traverser: int, before, t: int) -> float:
        if s.is_terminal():
            return self._utility(s, traverser, before)
        p = s.current
        obs = observe(s, p)
        legal = abstract_legal(obs)
        key = info_key(obs)
        node = self.nodes.get(key)
        if node is None:
            node = self.nodes[key] = new_node()
        node[VISITS] += 1
        sigma = regret_matching(node, legal)
        if p == traverser:
            vals = []
            for a in legal:
                child = s.clone()
                child.apply(to_concrete(obs, a))
                vals.append(self._traverse(child, traverser, before, t))
            v = sum(pr * x for pr, x in zip(sigma, vals))
            for a, x in zip(legal, vals):
                node[R0 + a] = max(0.0, node[R0 + a] + (x - v))  # RM+
            return v
        # oponente: acumula estratégia média (ponderação linear) e amostra
        for a, pr in zip(legal, sigma):
            node[A0 + a] += t * pr
        a = self.rng.choices(legal, weights=sigma)[0]
        s.apply(to_concrete(obs, a))
        return self._traverse(s, traverser, before, t)

    def prune(self, max_nodes: int) -> None:
        """Remove nós pouco visitados até caber em ``max_nodes`` (limite de RAM).

        Nós raros perdem o arrependimento acumulado; o agente usa a busca como
        reserva nesses conjuntos de informação de qualquer forma.
        """
        k = 2
        while len(self.nodes) > max_nodes:
            self.nodes = {key: n for key, n in self.nodes.items() if n[VISITS] >= k}
            k *= 2

    def train(self, iterations: int, log_every: int = 0, time_limit: Optional[float] = None,
              max_nodes: Optional[int] = None):
        t0 = time.time()
        for _ in range(iterations):
            self.iterations += 1
            root = self._sample_root()
            before = (root.scores[0], root.scores[1])
            for traverser in (0, 1):
                self._traverse(root.clone(), traverser, before, self.iterations)
            if log_every and self.iterations % log_every == 0:
                dt = time.time() - t0
                print(f"it={self.iterations} nodes={len(self.nodes)} {self.iterations / max(dt, 1e-9):.0f} it/s")
            if max_nodes and len(self.nodes) > max_nodes:
                self.prune(int(max_nodes * 0.6))
            if time_limit and time.time() - t0 > time_limit:
                break
        self.elapsed += time.time() - t0

    # ------------------------------------------------------------------
    def merge(self, other_nodes: Dict[str, array]) -> None:
        """Soma tabelas de treinos independentes (média de execuções)."""
        for k, n in other_nodes.items():
            mine = self.nodes.get(k)
            if mine is None:
                self.nodes[k] = n
            else:
                for i in range(NODE_LEN):
                    mine[i] += n[i]

    def average_policy(self, min_visits: int = 1) -> Dict[str, Dict[int, float]]:
        pol = {}
        for k, n in self.nodes.items():
            if n[VISITS] < min_visits:
                continue
            avg = n[A0:A0 + NUM_ACTIONS]
            tot = sum(avg)
            if tot > 0:
                pol[k] = {a: x / tot for a, x in enumerate(avg) if x > 0}
        return pol

    def save(self, path: Path, meta: Optional[dict] = None, min_visits: int = 1) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        policy = self.average_policy(min_visits)
        blob = {
            "policy": policy,
            "visits": {k: int(self.nodes[k][VISITS]) for k in policy},
            "meta": {
                "rules_version": self.rules.version,
                "rules_fingerprint": self.rules.fingerprint,
                "iterations": self.iterations,
                "seed": self.seed,
                "elapsed_s": round(self.elapsed, 1),
                "nodes": len(self.nodes),
                "algorithm": "ES-MCCFR + RM+ + linear averaging; utility = match equity delta",
                "equity_dist": self.eq.dist,
                **(meta or {}),
            },
        }
        with open(path, "wb") as f:
            pickle.dump(blob, f)
