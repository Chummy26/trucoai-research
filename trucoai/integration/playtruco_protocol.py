"""Leitor do protocolo da mesa PlayTruco (gateway-v3, socket.io + CBOR).

Formato observado em 19/10/2026 (captura ``captures/20261019_014228``, partida
1v1 Paulista). O servidor manda ``451-["gameEvent",{_placeholder}]`` seguido de
um anexo binário CBOR ``{sq, tp, dt, ...}``. Tipos usados:

  game_sync        dt.ef (entrada em centavos), dt.mt ("p" = Paulista), dt.np,
                   dt.ps[{pos, tm, mp (= sou eu), cs}], dt.sc.rp (placar por time)
  start_round      nova MÃO: dt.vr (vira), dt.ps[pos].cs (as minhas; "x" = oculta),
                   dt.pp (posição do pé = quem deu as cartas)
  run_turn         dt.cpp (de quem é a vez), catip (pode pedir truco),
                   cgr (pode correr), chc (pode virar carta)
  play_card        dt.cd ("s4", "h12"...; "x" se oculta), dt.pos, dt.hd (encoberta)
  increase_points  dt.akt (time que pediu), dt.ant (time que responde), dt.npi
  player_decision  dt.pos, dt.dt ("a" aceitou, "r" recusou)
  finish_turn      fim de rodada (vaza)
  finish_round     fim da MÃO: dt.sc.rp (placar por time), dt.gu (alguém correu)
  finish_game      fim da última mão e da partida (mesmos campos)
  last_point       mão de onze: dt.tid (time que tem 11 e decide), calp/crlp
                   (posso aceitar/recusar). Não há evento da decisão: run_turn com
                   crp=3 = jogou; finish_round sem jogadas = correu (+1 ao outro)
  startGame (JSON) partida nova (placar 0x0; aguarda game_sync para o valor)

O servidor envia os eventos a TODAS as conexões da conta: a posição do jogador
é inferida pelas cartas visíveis em start_round.

Enviados pelo cliente: playCard {value, suit, hidden}, increasePoints (pedir),
reraiseIncreasePoints (aumentar em resposta), acceptIncreasePoints,
refuseIncreasePoints (recusar), giveUpRound (correr na vez), revealCards.
Ainda não observado: decidir a mão de onze.

Naipes: s=espadas, h=copas, c=paus, d=ouros. Valores 1=A, 13=K; 11 < 12 em força
(ver ``_VALUE``).

Tudo o que não for reconhecido marca o leitor como dessincronizado até a
próxima mão, e a IA não age (falha fechada).
"""

from __future__ import annotations

import base64
from typing import Any, Dict, List, Optional

from .. import cards as C
from .cbor import loads as cbor_loads
from .playtruco import TableState
from .shadow import ME, OPP, Accept, HandEnd, HandStart, Play, Raise, Run

_SUIT = {"s": "e", "h": "c", "c": "p", "d": "o"}
# Na PlayTruco a força por código é 4 5 6 7 11 12 13 1 2 3: a figura 11 é a mais
# fraca, depois a 12 (observado: vira 11 tornou manilha a 12, não a 13). Mapeamos
# 11 e 12 para as posições de Q e J da nossa ordem (4 5 6 7 Q J K A 2 3), o que
# preserva a força; o nome exibido pode diferir do desenho da carta no site.
_VALUE = {1: "A", 2: "2", 3: "3", 4: "4", 5: "5", 6: "6", 7: "7", 11: "Q", 12: "J", 13: "K"}


def card_from_code(code: str) -> Optional[int]:
    """``"s4"`` -> 4 de espadas; ``"x"`` -> None (carta oculta)."""
    if not code or code == "x":
        return None
    suit, value = code[0], int(code[1:])
    return C.make(_VALUE[value], _SUIT[suit])


def code_from_card(card: int) -> Dict[str, Any]:
    """Carta -> ``{"value", "suit"}`` no formato do ``playCard`` enviado pelo cliente."""
    rank, suit = C.RANKS[C.rank_of(card)], C.SUITS[C.suit_of(card)]
    value = {v: k for k, v in _VALUE.items()}[rank]
    suit_name = {"e": "spades", "c": "hearts", "p": "clubs", "o": "diamonds"}[suit]
    return {"value": value, "suit": suit_name}


def card_from_playcard(msg: Any) -> Optional[int]:
    """Mensagem ``playCard`` do cliente -> carta do motor (None se não decifrar)."""
    try:
        value, suit = msg[1]["value"], msg[1]["suit"]
        rank = {v: k for k, v in _VALUE.items()}[value]
        return C.make(rank, {"spades": "e", "hearts": "c", "clubs": "p",
                             "diamonds": "o"}[suit])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


class PlayTrucoProtocolReader:
    implemented = True

    def __init__(self) -> None:
        self.me_pos: Optional[int] = None
        self.my_team: Optional[int] = None
        self.entry_cents: Optional[int] = None
        self.mode: Optional[str] = None
        self.n_players: Optional[int] = None
        self.scores: Optional[List[int]] = None  # por time
        self.current: Optional[int] = None
        self.flags: Dict[str, Any] = {}
        self.in_hand = False
        self.unknown: List[str] = []
        self.sent: List[Any] = []
        self._events: List[Any] = []
        self._pending_my_hidden: Optional[int] = None
        self.mao11_pos: Optional[int] = None  # quem decide a mão de onze nesta mão
        self.plays_in_hand = 0

    # ------------------------------------------------------------------
    def on_frame(self, direction: str, data: Any) -> None:
        j = data.get("json") if isinstance(data, dict) else None
        if direction == "sent":
            if isinstance(j, list) and j:
                self.sent.append(j)
            return
        if isinstance(j, list) and j and j[0] == "startGame":
            self.new_game()
            return
        if isinstance(data, dict) and "b64" in data:
            try:
                ev = cbor_loads(base64.b64decode(data["b64"]))
            except Exception as e:  # anexo que não é evento de jogo
                self.unknown.append(f"cbor: {e}")
                return
            if isinstance(ev, dict) and "tp" in ev:
                self.on_event(ev)

    def new_game(self) -> None:
        """Partida nova: placar 0x0; valor/modo ficam desconhecidos até o game_sync."""
        self.entry_cents = self.mode = self.n_players = None
        self.scores = [0, 0]
        self.in_hand = False
        self.mao11_pos = None

    def who(self, pos: int) -> int:
        return ME if pos == self.me_pos else OPP

    def _my_opp_scores(self, rp) -> Optional[tuple]:
        if rp is None or self.my_team is None:
            return None
        return rp[self.my_team], rp[1 - self.my_team]

    def on_event(self, ev: Dict[str, Any]) -> None:
        tp, dt = ev.get("tp"), ev.get("dt") or {}
        if tp == "game_sync":
            self.entry_cents = dt.get("ef")
            self.mode, self.n_players = dt.get("mt"), dt.get("np")
            for p in dt.get("ps", []):
                if p.get("mp"):
                    self.me_pos, self.my_team = p.get("pos"), p.get("tm")
            sc = (dt.get("sc") or {}).get("rp")
            if sc is not None:
                self.scores = list(sc)
            self.current = (dt.get("tn") or {}).get("cpp")
            self.in_hand = False  # entrou no meio da mão: espera a próxima
        elif tp == "start_round":
            visible = [p for p in dt.get("ps", []) if p.get("cs") and "x" not in p["cs"]]
            if len(visible) == 1 and visible[0]["pos"] != self.me_pos:
                # as cartas visíveis são as minhas: corrige a posição (1v1: time == posição)
                self.me_pos = self.my_team = visible[0]["pos"]
            if self.me_pos is None or self.scores is None:
                return
            self.mao11_pos, self.plays_in_hand = None, 0
            mine = next((p["cs"] for p in dt.get("ps", []) if p.get("pos") == self.me_pos), None)
            cards = [card_from_code(c) for c in (mine or [])]
            my, opp = self._my_opp_scores(self.scores)
            iron = my == opp == 11
            if len(cards) == 3 and all(c is None for c in cards) and iron:
                cards = None  # mão de ferro às cegas
            elif len(cards) != 3 or any(c is None for c in cards):
                self.in_hand = False
                self.unknown.append("start_round sem as 3 cartas visíveis")
                return
            self._events.append(HandStart(cards, card_from_code(dt["vr"]), my, opp,
                                          i_am_mao=dt.get("pp") != self.me_pos))
            self.in_hand = True
        elif not self.in_hand:
            if tp in ("finish_round", "finish_game"):
                self.scores = list((dt.get("sc") or {}).get("rp", self.scores or []))
            return
        elif tp == "last_point":
            # Mão de onze: decide o jogador do time ``tid`` (1v1: posição == time).
            self.mao11_pos = dt.get("tid")
            self.current = self.mao11_pos
            self.flags = {"calp": dt.get("calp"), "crlp": dt.get("crlp")}
        elif tp == "run_turn":
            if self.mao11_pos is not None:
                # Não há evento da decisão: começar a jogar valendo 3 = aceitou.
                if dt.get("crp", 1) > 1:
                    self._events.append(Accept(self.who(self.mao11_pos)))
                    self.mao11_pos = None
                else:
                    self.unknown.append("mão de onze sem valor 3 após decisão")
                    self.in_hand = False
                    return
            self.current = dt.get("cpp")
            self.flags = {k: dt.get(k) for k in ("catip", "cgr", "chc", "crp", "urp")}
        elif tp == "play_card":
            self.plays_in_hand += 1
            hidden = bool(dt.get("hd"))
            card = card_from_code(dt.get("cd", "x"))
            if dt.get("pos") == self.me_pos and card is None:
                card = self._pending_my_hidden
            self._events.append(Play(self.who(dt["pos"]), card, covered=hidden))
        elif tp == "increase_points":
            # ``akt`` = time que pediu (1v1: posição). ``akp`` não é confiável:
            # já veio 0 num pedido do time 1 (captura 20261019_020041, t=96 s).
            asker = dt["akt"] if "akt" in dt else dt["akp"]
            self._events.append(Raise(self.who(asker)))
            self.current = dt.get("ant")
        elif tp == "player_decision":
            kind = dt.get("dt")
            if kind == "a":
                self._events.append(Accept(self.who(dt["pos"])))
            elif kind == "r":
                self._events.append(Run(self.who(dt["pos"])))
            else:
                self.unknown.append(f"player_decision {kind!r}")
                self.in_hand = False
        elif tp in ("finish_round", "finish_game"):  # finish_game = última mão
            if self.mao11_pos is not None and self.plays_in_hand == 0:
                self._events.append(Run(self.who(self.mao11_pos)))  # correu da mão de onze
                self.mao11_pos = None
            rp = (dt.get("sc") or {}).get("rp")
            if rp is not None:
                self.scores = list(rp)
                my, opp = self._my_opp_scores(rp)
                self._events.append(HandEnd(my, opp, gave_up=bool(dt.get("gu"))))
            self.in_hand = False
        elif tp in ("finish_turn", "cardsRevealed"):
            pass
        else:
            self.unknown.append(f"evento {tp!r}")

    def note_sent_card(self, card: int) -> None:
        """Registra a carta que a IA vai jogar encoberta (o servidor não a devolve)."""
        self._pending_my_hidden = card

    # ------------------------------------------------------------------
    def table_value_text(self) -> Optional[str]:
        """Valor da mesa para a trava; ``None`` se não for 1v1 Paulista identificado."""
        if self.entry_cents is None or self.mode != "p" or self.n_players != 2:
            return None
        return f"R$ {self.entry_cents / 100:.2f}".replace(".", ",")

    def poll(self, dom: Optional[dict] = None) -> TableState:
        evs, self._events = self._events, []
        return TableState(events=evs, table_value_text=self.table_value_text(),
                          my_turn=self.current == self.me_pos and self.in_hand,
                          buttons=dict(self.flags))
