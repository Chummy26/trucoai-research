"""Arena de avaliação: partidas espelhadas, IC de Wilson e métricas por agente.

Protocolo (herdado e ampliado de truco-de-malandro/arena):
  * cada sequência de distribuições é jogada duas vezes, trocando os assentos
    (controle de variância da sorte das cartas e da posição);
  * a distribuição da mão k da partida s é ``Random((EVAL_BASE + s) * 1000 + k)``
    — sementes disjuntas das usadas no treino;
  * ação ilegal é contada e substituída por uma ação legal aleatória;
  * tempo de decisão é medido por agente.
"""

from __future__ import annotations

import math
import random
import time
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass, field
from typing import Callable, Dict, List, Optional, Tuple

from . import cards as C
from .engine import RAISE, RUN, new_match
from .observation import observe
from .rules import DEFAULT_RULES, RuleSet

EVAL_BASE = 10**9


@dataclass
class Side:
    decisions: int = 0
    illegal: int = 0
    time_s: float = 0.0
    raises: int = 0
    runs: int = 0
    responses: int = 0
    points: int = 0

    def add(self, o: "Side") -> None:
        for k in self.__dataclass_fields__:
            setattr(self, k, getattr(self, k) + getattr(o, k))


@dataclass
class MatchResult:
    seed: int
    a_seat: int
    a_won: bool
    hands: int
    a_points: int
    b_points: int
    sides: Dict[str, Side] = field(default_factory=dict)


def play_match(agent_a, agent_b, seed: int, a_seat: int,
               rules: RuleSet = DEFAULT_RULES, max_hands: int = 200,
               log: Optional[list] = None) -> MatchResult:
    agents = [None, None]
    agents[a_seat], agents[1 - a_seat] = agent_a, agent_b
    agent_a.reset(seed * 2 + a_seat)
    agent_b.reset(seed * 2 + 1 - a_seat + 7919)
    sides = [Side(), Side()]
    st = new_match(rules, first_dealer=1)
    hands = 0
    while st.match_winner is None and hands < max_hands:
        st.deal(random.Random((EVAL_BASE + seed) * 1000 + hands))
        hands += 1
        while not st.is_terminal():
            p = st.current
            obs = observe(st, p)
            t0 = time.perf_counter()
            a = agents[p].act(obs)
            sides[p].time_s += time.perf_counter() - t0
            sides[p].decisions += 1
            if a not in obs.legal_actions:
                sides[p].illegal += 1
                a = random.Random(seed + hands).choice(obs.legal_actions)
            if obs.phase == "respond":
                sides[p].responses += 1
            if a == RAISE:
                sides[p].raises += 1
            elif a == RUN:
                sides[p].runs += 1
            if log is not None:
                log.append({"seed": seed, "hand": hands, "p": p, "phase": obs.phase,
                            "action": a, "legal": list(obs.legal_actions)})
            st.apply(a)
    for p in (0, 1):
        sides[p].points = st.scores[p]
    a_won = st.match_winner == a_seat
    return MatchResult(seed, a_seat, a_won, hands, st.scores[a_seat], st.scores[1 - a_seat],
                       {"a": sides[a_seat], "b": sides[1 - a_seat]})


def wilson(k: int, n: int, z: float = 1.96) -> Tuple[float, float]:
    if n == 0:
        return 0.0, 1.0
    p = k / n
    d = 1 + z * z / n
    c = (p + z * z / (2 * n)) / d
    h = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / d
    return max(0.0, c - h), min(1.0, c + h)


def summarize(results: List[MatchResult]) -> dict:
    n = len(results)
    w = sum(r.a_won for r in results)
    a, b = Side(), Side()
    for r in results:
        a.add(r.sides["a"])
        b.add(r.sides["b"])
    hands = sum(r.hands for r in results)
    # Diferença de pontos por mão com erro padrão (pareada por partida).
    diffs = [(r.a_points - r.b_points) / max(r.hands, 1) for r in results]
    mean = sum(diffs) / n if n else 0.0
    sd = math.sqrt(sum((x - mean) ** 2 for x in diffs) / (n - 1)) if n > 1 else 0.0

    def side(s: Side):
        return {"illegal_rate": s.illegal / max(s.decisions, 1),
                "decisions": s.decisions, "time_s": s.time_s,
                "ms_per_decision": 1000 * s.time_s / max(s.decisions, 1),
                "raise_per_hand": s.raises / max(hands, 1),
                "run_rate_on_response": s.runs / max(s.responses, 1)}

    return {"games": n, "a_wins": w, "win_rate_a": w / n if n else 0.0,
            "ci95": list(wilson(w, n)), "hands": hands,
            "point_diff_per_hand_a": mean, "point_diff_se": sd / math.sqrt(n) if n else 0.0,
            "a": side(a), "b": side(b),
            "by_seat": {s: sum(r.a_won for r in results if r.a_seat == s) /
                        max(1, sum(1 for r in results if r.a_seat == s)) for s in (0, 1)}}


# ---------------------------------------------------------------------------
_CACHE: Dict[str, object] = {}


def _chunk(args):
    factory, spec_a, spec_b, seeds = args[:4]
    rules = args[4] if len(args) > 4 else DEFAULT_RULES
    a = factory(spec_a)
    b = factory(spec_b)
    out = []
    for s in seeds:
        for seat in (0, 1):
            out.append(play_match(a, b, s, seat, rules=rules))
    return out


def run_matchup(factory: Callable[[str], object], spec_a: str, spec_b: str,
                deals: int = 100, start_seed: int = 0, workers: int = 1,
                chunk: int = 10, rules: RuleSet = DEFAULT_RULES) -> dict:
    """``deals`` sequências de distribuição, cada uma jogada nos dois assentos.

    ``rules`` escolhe o regulamento (padrão: h0, como antes). A saída inclui
    ``deal_wins``: vitórias de A (0, 1 ou 2) em cada sequência de distribuição,
    na ordem das sementes, para análises pareadas por distribuição.
    """
    seeds = list(range(start_seed, start_seed + deals))
    chunks = [seeds[i:i + chunk] for i in range(0, len(seeds), chunk)]
    results: List[MatchResult] = []
    if workers <= 1:
        for c in chunks:
            results += _chunk((factory, spec_a, spec_b, c, rules))
    else:
        with ProcessPoolExecutor(workers) as ex:
            for r in ex.map(_chunk, [(factory, spec_a, spec_b, c, rules) for c in chunks]):
                results += r
    out = summarize(results)
    wins_by_seed: Dict[int, int] = {}
    for r in results:
        wins_by_seed[r.seed] = wins_by_seed.get(r.seed, 0) + int(r.a_won)
    out.update({"a": spec_a, "b": spec_b, "deals": deals, "start_seed": start_seed,
                "rules_version": rules.version, "rules_fingerprint": rules.fingerprint,
                "deal_wins": [wins_by_seed[s] for s in seeds],
                "a_metrics": out.pop("a"), "b_metrics": out.pop("b")})
    return out
