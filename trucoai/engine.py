"""Motor de regras de Truco Paulista 1v1 (sem I/O, sem política).

O estado completo (``GameState``) contém informação privada dos dois jogadores.
Políticas nunca devem recebê-lo: use :func:`trucoai.observation.observe`.

Ações (inteiros, espaço fixo ``NUM_ACTIONS``):
    0..2  PLAY_0..2     joga a carta da posição i aberta
    3..5  COVER_0..2    joga a carta da posição i encoberta
    6     RAISE         pede truco / aumenta (seis, nove, doze)
    7     ACCEPT        aceita o pedido (ou joga a mão de onze)
    8     RUN           corre (recusa o pedido, ou recusa a mão de onze)

As posições 0..2 são fixas durante a mão: a carta jogada deixa ``None`` no lugar.
Isso permite jogar às cegas na mão de ferro escolhendo só a posição.

Fases:
    "mao11"   jogador com 11 decide jogar (ACCEPT) ou correr (RUN)
    "play"    vez de jogar carta (ou pedir aumento antes)
    "respond" responder a pedido pendente
    "hand_over" / "match_over"
"""

from __future__ import annotations

import random
from dataclasses import dataclass, field
from typing import List, Optional, Tuple

from . import cards as C
from .rules import DEFAULT_RULES, RuleSet

PLAY_0, PLAY_1, PLAY_2, COVER_0, COVER_1, COVER_2, RAISE, ACCEPT, RUN = range(9)
NUM_ACTIONS = 9
ACTION_NAMES = ["jogar0", "jogar1", "jogar2", "cobrir0", "cobrir1", "cobrir2",
                "truco", "aceitar", "correr"]

NORMAL, MAO11, IRON = "normal", "mao11", "iron"
TIE = -1


def other(p: int) -> int:
    return 1 - p


@dataclass
class GameState:
    rules: RuleSet = DEFAULT_RULES
    scores: List[int] = field(default_factory=lambda: [0, 0])
    dealer: int = 1
    hand_no: int = 0
    phase: str = "hand_over"
    special: str = NORMAL
    mao11_player: Optional[int] = None
    vira: int = -1
    hands: List[List[Optional[int]]] = field(default_factory=lambda: [[None] * 3, [None] * 3])
    # Rodadas: lista de jogadas (jogador, carta, encoberta) por rodada.
    rounds: List[List[Tuple[int, int, bool]]] = field(default_factory=list)
    round_results: List[int] = field(default_factory=list)  # 0, 1 ou TIE
    leader: int = 0
    current: int = 0
    stake: int = 1
    pending: Optional[int] = None
    raiser: Optional[int] = None
    last_raiser: Optional[int] = None  # quem fez o último aumento aceito
    resume_player: Optional[int] = None
    hand_winner: Optional[int] = None
    hand_points: int = 0
    match_winner: Optional[int] = None
    # Log público: (tipo, jogador, valor). Cartas encobertas aparecem como -1.
    public_log: List[Tuple[str, int, int]] = field(default_factory=list)

    # ------------------------------------------------------------------
    def clone(self) -> "GameState":
        s = GameState.__new__(GameState)
        s.__dict__.update(self.__dict__)
        s.scores = list(self.scores)
        s.hands = [list(self.hands[0]), list(self.hands[1])]
        s.rounds = [list(r) for r in self.rounds]
        s.round_results = list(self.round_results)
        s.public_log = list(self.public_log)
        return s

    @property
    def mao(self) -> int:
        """Jogador que é "mão" (abre a primeira rodada): quem não deu as cartas."""
        return other(self.dealer)

    @property
    def round_index(self) -> int:
        return len(self.round_results)

    def is_terminal(self) -> bool:
        return self.phase in ("hand_over", "match_over")

    # ------------------------------------------------------------------
    # Distribuição
    # ------------------------------------------------------------------
    def deal(self, rng: random.Random, deck: Optional[List[int]] = None) -> None:
        """Inicia uma nova mão. ``deck`` permite fixar a ordem (testes)."""
        if self.match_winner is not None:
            raise RuntimeError("partida encerrada")
        if deck is None:
            deck = C.full_deck()
            rng.shuffle(deck)
        if self.hand_no > 0:
            self.dealer = other(self.dealer)
        self.hand_no += 1
        mao = self.mao
        # Distribuição alternada a partir do mão; depois a vira.
        self.hands = [[None] * 3, [None] * 3]
        for i in range(3):
            self.hands[mao][i] = deck[2 * i]
            self.hands[other(mao)][i] = deck[2 * i + 1]
        self.vira = deck[6]
        self.rounds = [[]]
        self.round_results = []
        self.leader = mao
        self.current = mao
        self.pending = self.raiser = self.last_raiser = self.resume_player = None
        self.hand_winner, self.hand_points = None, 0
        self.public_log = [("deal", self.dealer, self.vira)]
        target = self.rules.target_score
        at11 = [s == target - 1 for s in self.scores]
        self.mao11_player = None
        if at11[0] and at11[1]:
            self.special, self.stake, self.phase = IRON, self.rules.iron_value, "play"
        elif at11[0] or at11[1]:
            self.special, self.stake = MAO11, 1
            self.mao11_player = 0 if at11[0] else 1
            self.phase, self.current = "mao11", self.mao11_player
        else:
            self.special, self.stake, self.phase = NORMAL, 1, "play"

    # ------------------------------------------------------------------
    # Ações legais
    # ------------------------------------------------------------------
    def _next_stake(self, value: int) -> Optional[int]:
        ladder = self.rules.stake_ladder
        if value not in ladder:
            return None
        i = ladder.index(value)
        if i + 1 >= len(ladder):
            return None
        nxt = ladder[i + 1]
        if self.rules.cap_raise_at_needed:
            needed = self.rules.target_score - min(self.scores)
            if value >= needed:
                return None
        return nxt

    def _raises_allowed(self) -> bool:
        if self.special == IRON:
            return False
        if self.special == MAO11 and not self.rules.mao11_allows_truco:
            return False
        return True

    def legal_actions(self) -> List[int]:
        ph = self.phase
        if ph == "mao11":
            return [ACCEPT, RUN]
        if ph == "respond":
            acts = [ACCEPT, RUN]
            if self._raises_allowed() and self._next_stake(self.pending) is not None:
                acts.insert(0, RAISE)
            return acts
        if ph != "play":
            return []
        p = self.current
        hand = self.hands[p]
        acts = [i for i in range(3) if hand[i] is not None]
        if (self.rules.allow_covered and self.round_index >= self.rules.covered_from_round
                and self.special != IRON):
            acts += [3 + i for i in range(3) if hand[i] is not None]
        if (self._raises_allowed() and self.last_raiser != p
                and self._next_stake(self.stake) is not None):
            acts.append(RAISE)
        if (self.rules.allow_fold_on_turn and self.special == NORMAL
                and self.round_index <= self.rules.fold_on_turn_max_round):
            acts.append(RUN)
        return acts

    def legal_mask(self) -> List[bool]:
        m = [False] * NUM_ACTIONS
        for a in self.legal_actions():
            m[a] = True
        return m

    # ------------------------------------------------------------------
    # Transição
    # ------------------------------------------------------------------
    def apply(self, action: int) -> None:
        if action not in self.legal_actions():
            raise ValueError(f"ação ilegal {action} na fase {self.phase}")
        p = self.current
        ph = self.phase
        if ph == "mao11":
            self.public_log.append(("mao11", p, action))
            if action == ACCEPT:
                self.stake = self.rules.mao11_value
                self.phase = "play"
                self.current = self.leader
            else:
                self._end_hand(other(p), self.rules.mao11_run_points)
            return
        if ph == "respond":
            if action == ACCEPT:
                self.public_log.append(("accept", p, self.pending))
                self.stake = self.pending
                self.last_raiser = self.raiser
                self.pending = self.raiser = None
                self.current = self.resume_player
                self.resume_player = None
                self.phase = "play"
            elif action == RUN:
                self.public_log.append(("run", p, self.pending))
                self._end_hand(self.raiser, self.stake)
            else:  # contra-aumento: aceita implicitamente o pedido anterior
                self.stake = self.pending
                self.pending = self._next_stake(self.pending)
                self.raiser = p
                self.public_log.append(("raise", p, self.pending))
                self.current = other(p)
            return
        # fase "play"
        if action == RAISE:
            self.pending = self._next_stake(self.stake)
            self.raiser = p
            self.resume_player = p
            self.public_log.append(("raise", p, self.pending))
            self.phase = "respond"
            self.current = other(p)
            return
        if action == RUN:
            self.public_log.append(("fold", p, self.stake))
            self._end_hand(other(p), self.stake)
            return
        slot = action % 3
        covered = action >= 3
        card = self.hands[p][slot]
        self.hands[p][slot] = None
        self.rounds[-1].append((p, card, covered))
        self.public_log.append(("play", p, -1 if covered else card))
        if len(self.rounds[-1]) == 2:
            self._resolve_round()
        else:
            self.current = other(p)

    def _resolve_round(self) -> None:
        (pa, ca, cova), (pb, cb, covb) = self.rounds[-1]
        sa = C.COVERED_STRENGTH if cova else C.strength(ca, self.vira)
        sb = C.COVERED_STRENGTH if covb else C.strength(cb, self.vira)
        if sa > sb:
            res = pa
        elif sb > sa:
            res = pb
        else:
            res = TIE
        self.round_results.append(res)
        self.public_log.append(("round", res, len(self.round_results) - 1))
        done, winner = hand_outcome(self.round_results, self.mao, self.rules)
        if done:
            if winner is None:
                self._end_hand(None, 0)
            else:
                self._end_hand(winner, self.stake)
            return
        if res == TIE:
            nxt = self.leader if self.rules.tie_next_leader == "same_leader" else self.mao
        else:
            nxt = res
        self.rounds.append([])
        self.leader = self.current = nxt

    def _end_hand(self, winner: Optional[int], points: int) -> None:
        self.hand_winner = winner
        self.hand_points = points if winner is not None else 0
        self.pending = self.raiser = self.resume_player = None
        self.public_log.append(("hand_end", -1 if winner is None else winner, self.hand_points))
        if winner is not None:
            self.scores[winner] += points
            if self.scores[winner] >= self.rules.target_score:
                self.match_winner = winner
                self.phase = "match_over"
                return
        self.phase = "hand_over"


def hand_outcome(results: List[int], mao: int, rules: RuleSet) -> Tuple[bool, Optional[int]]:
    """Decide a mão pelas rodadas concluídas. Retorna ``(decidida, vencedor)``."""
    wins = [results.count(0), results.count(1)]
    if wins[0] >= 2:
        return True, 0
    if wins[1] >= 2:
        return True, 1
    n = len(results)
    if n < 2:
        return False, None
    first = results[0]
    if first == TIE:
        decided = [r for r in results[1:] if r != TIE]
        if decided:
            return True, decided[0]
        if n < 3:
            return False, None
        return True, (mao if rules.all_tied == "mao" else None)
    if TIE in results[1:]:
        return True, first
    if n < 3:
        return False, None  # 1 a 1: terceira rodada decide
    return True, first  # inalcançável: 3 rodadas sem empate têm 2 vitórias


def new_match(rules: RuleSet = DEFAULT_RULES, first_dealer: int = 1) -> GameState:
    return GameState(rules=rules, dealer=first_dealer)
