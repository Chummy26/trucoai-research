"""Jogo-sombra: reconstrói a mão a partir dos eventos visíveis da mesa real.

A mesa externa (PlayTruco) só mostra ao jogador o que ele pode ver. O leitor
da mesa converte isso em eventos normalizados; esta classe os aplica no motor
local, com cartas *fictícias* no lugar das cartas ocultas do adversário. Assim:

  * a ``Observation`` entregue ao agente é exatamente a do motor local
    (as cartas fictícias nunca aparecem nela — ver ``observe``);
  * o motor valida cada evento: se a mesa real fizer algo que nossas regras
    não permitem, ou pontuar diferente, isso vira uma ``Divergence``
    registrada — evidência para verificar o regulamento (docs/REGRAS.md);
  * após uma divergência a mão fica dessincronizada e a IA não age (falha fechada).

Convenção: no jogo-sombra a IA é sempre o jogador 0 e o adversário o 1.
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Sequence, Union

from ..engine import ACCEPT, COVER_0, PLAY_0, RAISE, RUN, GameState
from ..observation import Observation, observe
from ..rules import DEFAULT_RULES, RuleSet

ME, OPP = 0, 1


# --- eventos normalizados -----------------------------------------------------
@dataclass(frozen=True)
class HandStart:
    my_cards: Optional[Sequence[int]]  # None = mão de ferro às cegas
    vira: int
    my_score: int
    opp_score: int
    i_am_mao: bool  # True se a IA abre a primeira rodada (não deu as cartas)


@dataclass(frozen=True)
class Play:
    who: int  # ME ou OPP
    card: Optional[int]  # None = encoberta (só para OPP; para ME informe a carta)
    covered: bool = False
    slot: Optional[int] = None  # posição clicada (útil às cegas, quando só a IA sabe)


@dataclass(frozen=True)
class Raise:
    who: int


@dataclass(frozen=True)
class Accept:
    who: int


@dataclass(frozen=True)
class Run:
    who: int


@dataclass(frozen=True)
class HandEnd:
    my_score: int
    opp_score: int
    gave_up: bool = False  # a mesa indicou que alguém correu


Event = Union[HandStart, Play, Raise, Accept, Run, HandEnd]


@dataclass
class Divergence:
    hand_no: int
    event: str
    detail: str


@dataclass
class ShadowGame:
    rules: RuleSet = DEFAULT_RULES
    state: Optional[GameState] = None
    synced: bool = False
    hands: int = 0
    divergences: List[Divergence] = field(default_factory=list)
    blind: bool = False
    blind_slot_hint: Optional[int] = None  # posição que o executor clicou às cegas
    _rng: random.Random = field(default_factory=lambda: random.Random(0))

    # ------------------------------------------------------------------
    def feed(self, ev: Event) -> None:
        try:
            if isinstance(ev, HandStart):
                self._start(ev)
            elif not self.synced:
                return
            elif isinstance(ev, Play):
                self._play(ev)
            elif isinstance(ev, Raise):
                self._act(ev.who, RAISE, ev)
            elif isinstance(ev, Accept):
                self._act(ev.who, ACCEPT, ev)
            elif isinstance(ev, Run):
                self._act(ev.who, RUN, ev)
            elif isinstance(ev, HandEnd):
                self._end(ev)
        except ValueError as e:
            self._diverge(ev, str(e))

    def observation(self) -> Optional[Observation]:
        """Observação da IA, ou ``None`` se não for a vez dela / dessincronizado."""
        s = self.state
        if not self.synced or s is None or s.is_terminal() or s.current != ME:
            return None
        return observe(s, ME)

    # ------------------------------------------------------------------
    def _diverge(self, ev, detail: str) -> None:
        self.divergences.append(Divergence(self.hands, type(ev).__name__, f"{ev}: {detail}"))
        self.synced = False

    def _start(self, ev: HandStart) -> None:
        self.blind = ev.my_cards is None
        if self.blind:
            # Mão de ferro às cegas: nem a IA conhece as próprias cartas.
            pool = [c for c in range(40) if c != ev.vira]
            self._rng.shuffle(pool)
            mine, pool = pool[:3], pool[3:]
        else:
            mine = list(ev.my_cards)
            if len(mine) != 3 or len(set(mine + [ev.vira])) != 4:
                raise ValueError("mão inicial inválida")
            pool = [c for c in range(40) if c not in mine and c != ev.vira]
            self._rng.shuffle(pool)
        fake_opp = pool[:3]  # fictícias: nunca chegam à observação
        mao = ME if ev.i_am_mao else OPP
        hands = {mao: mine if mao == ME else fake_opp, 1 - mao: fake_opp if mao == ME else mine}
        deck = []
        for i in range(3):
            deck += [hands[mao][i], hands[1 - mao][i]]
        deck += [ev.vira] + pool[3:]
        s = GameState(rules=self.rules, dealer=1 - mao)
        s.scores = [ev.my_score, ev.opp_score]
        s.deal(self._rng, deck)
        if self.blind and not (s.special == "iron" and self.rules.iron_blind):
            raise ValueError("mão às cegas fora da mão de ferro")
        self.state, self.synced = s, True
        self.hands += 1

    def _expect_turn(self, who: int) -> None:
        if self.state.current != who:
            raise ValueError(f"evento do jogador {who}, mas a vez é do {self.state.current}")

    def _act(self, who: int, action: int, ev) -> None:
        self._expect_turn(who)
        self.state.apply(action)

    def _reveal(self, who: int, card: int, slot_hint: Optional[int] = None) -> int:
        """Posição da mão de ``who`` que passa a conter ``card`` (revelada pela mesa).

        Cartas fictícias (do adversário, ou as minhas na mão às cegas) são
        trocadas para manter o baralho consistente.
        """
        s = self.state
        hand = s.hands[who]
        if card in hand:
            j = hand.index(card)
            if slot_hint is not None and slot_hint != j and hand[slot_hint] is not None \
                    and (who == OPP or self.blind):
                hand[j], hand[slot_hint] = hand[slot_hint], hand[j]  # ambas fictícias
                return slot_hint
            return j
        if who == ME and not self.blind:
            raise ValueError("carta jogada não está na mão da IA")
        real_mine = [] if self.blind else [c for c in s.hands[ME] if c is not None]
        if card == s.vira or card in real_mine or any(
                card == c and not cov for rnd in s.rounds for _, c, cov in rnd):
            raise ValueError("carta revelada já é conhecida em outro lugar")
        if slot_hint is not None and hand[slot_hint] is not None:
            slot = slot_hint
        else:
            slot = next(i for i, c in enumerate(hand) if c is not None)
        old = hand[slot]
        other_hand = s.hands[1 - who]
        if card in other_hand:  # só pode ser fictícia
            other_hand[other_hand.index(card)] = old
        for r in s.rounds:
            for j, (p, c, cov) in enumerate(r):
                if c == card:
                    r[j] = (p, old, cov)
        hand[slot] = card
        return slot

    def _play(self, ev: Play) -> None:
        s = self.state
        self._expect_turn(ev.who)
        hand = s.hands[ev.who]
        covered = ev.covered or ev.card is None
        if ev.card is None:
            if ev.who == ME and not self.blind:
                raise ValueError("carta da IA sem identidade")
            slot = next(i for i, c in enumerate(hand) if c is not None)
        else:
            hint = ev.slot if ev.slot is not None else (
                self.blind_slot_hint if ev.who == ME else None)
            slot = self._reveal(ev.who, ev.card, hint)
            if ev.who == ME:
                self.blind_slot_hint = None
        s.apply((COVER_0 if covered else PLAY_0) + slot)

    def _end(self, ev: HandEnd) -> None:
        s = self.state
        if not s.is_terminal() and ev.gave_up and s.phase == "play":
            s.apply(RUN)  # "correr" na própria vez (sem pedido pendente)
        if not s.is_terminal():
            raise ValueError("mesa encerrou a mão antes do motor local")
        # A mesa limita o placar exibido ao alvo (ex.: 11 + 3 aparece como 12).
        target = self.rules.target_score
        if [min(ev.my_score, target), min(ev.opp_score, target)] != [min(x, target) for x in s.scores]:
            raise ValueError(f"placar da mesa {ev.my_score}x{ev.opp_score} != motor {s.scores}")
