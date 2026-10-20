"""Observação legítima de um jogador e reconstrução de estados compatíveis.

``observe(state, player)`` é a ÚNICA ponte entre o estado completo e uma
política. Ela contém apenas:
  * as próprias cartas (exceto na mão de ferro às cegas);
  * as próprias cartas encobertas já jogadas (o dono sabe quais eram);
  * eventos públicos: vira, placar, cartas abertas, pedidos e respostas.

Nunca inclui cartas do adversário na mão, cartas encobertas do adversário,
ordem do baralho ou sementes. ``tests/test_observation.py`` verifica isso
trocando a informação oculta e exigindo observações idênticas.

``sample_state(obs, rng)`` cria um ``GameState`` completo que é consistente com
a observação, sorteando a informação oculta. É o que os agentes de busca usam:
assim eles funcionam também a partir de observações vindas de uma interface
externa, sem jamais tocar no estado verdadeiro.
"""

from __future__ import annotations

import random
from dataclasses import dataclass
from typing import List, Optional, Tuple

from . import cards as C
from .engine import IRON, GameState
from .rules import RuleSet


@dataclass(frozen=True)
class Observation:
    rules: RuleSet
    me: int
    scores: Tuple[int, int]
    dealer: int
    hand_no: int
    phase: str
    special: str
    mao11_player: Optional[int]
    vira: int
    my_hand: Tuple[Optional[int], ...]  # None = já jogada ou desconhecida (às cegas)
    my_slots_filled: Tuple[bool, bool, bool]
    blind: bool
    opp_cards_left: int
    my_covered: Tuple[int, ...]  # minhas cartas encobertas já jogadas
    # Rodadas públicas: (jogador, carta ou -1 se encoberta)
    rounds: Tuple[Tuple[Tuple[int, int], ...], ...]
    round_results: Tuple[int, ...]
    leader: int
    current: int
    stake: int
    pending: Optional[int]
    raiser: Optional[int]
    last_raiser: Optional[int]
    resume_player: Optional[int]
    public_log: Tuple[Tuple[str, int, int], ...]
    legal_actions: Tuple[int, ...]

    @property
    def opp(self) -> int:
        return 1 - self.me

    def known_cards(self) -> List[int]:
        """Cartas cuja identidade o jogador conhece (não podem estar ocultas)."""
        known = [self.vira]
        known += [c for c in self.my_hand if c is not None]
        known += list(self.my_covered)
        for rnd in self.rounds:
            known += [c for _, c in rnd if c >= 0]
        return known

    def unknown_pool(self) -> List[int]:
        k = set(self.known_cards())
        return [c for c in range(C.NUM_CARDS) if c not in k]

    def key(self) -> str:
        """Chave textual completa (sem abstração) do conjunto de informação."""
        hand = ",".join("_" if c is None else str(c) for c in self.my_hand)
        log = ";".join(f"{t}{p}:{v}" for t, p, v in self.public_log)
        return f"{self.me}|{self.scores}|{hand}|{','.join(map(str, self.my_covered))}|{log}"


def observe(state: GameState, player: int) -> Observation:
    blind = state.special == IRON and state.rules.iron_blind
    my_hand = tuple(None if blind else c for c in state.hands[player])
    my_covered = tuple(c for rnd in state.rounds for (p, c, cov) in rnd if p == player and cov)
    rounds = tuple(tuple((p, -1 if cov else c) for (p, c, cov) in rnd) for rnd in state.rounds)
    opp_left = sum(c is not None for c in state.hands[1 - player])
    legal = tuple(state.legal_actions()) if state.current == player else ()
    return Observation(
        rules=state.rules, me=player, scores=(state.scores[0], state.scores[1]),
        dealer=state.dealer, hand_no=state.hand_no, phase=state.phase,
        special=state.special, mao11_player=state.mao11_player, vira=state.vira,
        my_hand=my_hand, my_slots_filled=tuple(c is not None for c in state.hands[player]),
        blind=blind, opp_cards_left=opp_left, my_covered=my_covered, rounds=rounds,
        round_results=tuple(state.round_results), leader=state.leader,
        current=state.current, stake=state.stake, pending=state.pending,
        raiser=state.raiser, last_raiser=state.last_raiser,
        resume_player=state.resume_player, public_log=tuple(state.public_log),
        legal_actions=legal,
    )


def sample_state(obs: Observation, rng: random.Random,
                 opp_weights=None) -> GameState:
    """Sorteia um estado completo consistente com ``obs``.

    ``opp_weights`` (opcional) é uma função ``carta -> peso`` para amostragem
    não uniforme da mão adversária (modelo de oponente / crença).
    """
    pool = obs.unknown_pool()
    rng.shuffle(pool)
    me, opp = obs.me, obs.opp
    s = GameState(rules=obs.rules)
    s.scores = list(obs.scores)
    s.dealer, s.hand_no, s.phase, s.special = obs.dealer, obs.hand_no, obs.phase, obs.special
    s.mao11_player, s.vira = obs.mao11_player, obs.vira
    s.leader, s.current, s.stake, s.pending = obs.leader, obs.current, obs.stake, obs.pending
    s.raiser, s.last_raiser, s.resume_player = obs.raiser, obs.last_raiser, obs.resume_player
    s.round_results = list(obs.round_results)
    s.public_log = list(obs.public_log)
    hands: List[List[Optional[int]]] = [[None] * 3, [None] * 3]
    # Minhas cartas (às cegas: sorteadas também).
    for i in range(3):
        if obs.my_slots_filled[i]:
            hands[me][i] = obs.my_hand[i] if obs.my_hand[i] is not None else pool.pop()
    # Cartas do adversário: amostragem ponderada opcional.
    if opp_weights is not None and obs.opp_cards_left:
        chosen = _weighted_sample(pool, obs.opp_cards_left, opp_weights, rng)
        pool = [c for c in pool if c not in chosen]
    else:
        chosen = [pool.pop() for _ in range(obs.opp_cards_left)]
    for i, c in enumerate(chosen):
        hands[opp][i] = c
    s.hands = hands
    my_cov = list(obs.my_covered)
    rounds = []
    for rnd in obs.rounds:
        r = []
        for p, c in rnd:
            if c >= 0:
                r.append((p, c, False))
            elif p == me:
                r.append((p, my_cov.pop(0), True))
            else:
                r.append((p, pool.pop(), True))
        rounds.append(r)
    s.rounds = rounds
    return s


def _weighted_sample(pool, k, weight_fn, rng):
    items = list(pool)
    out = []
    for _ in range(k):
        ws = [max(1e-9, weight_fn(c)) for c in items]
        c = rng.choices(items, weights=ws)[0]
        out.append(c)
        items.remove(c)
    return out
