"""Leitor do protocolo da mesa Online (gateway-v3, socket.io + CBOR).

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
import hashlib
import json
import math
import time
from typing import Any, Dict, List, Optional

from .. import cards as C
from .cbor import loads as cbor_loads
from .online import TableState
from .shadow import ME, OPP, Accept, HandEnd, HandStart, MatchStart, Play, Raise, Run

_SUIT = {"s": "e", "h": "c", "c": "p", "d": "o"}
# Na plataforma de referência a força por código é 4 5 6 7 11 12 13 1 2 3: a figura 11 é a mais
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
        rank = _VALUE[value]
        return C.make(rank, {"spades": "e", "hearts": "c", "clubs": "p",
                             "diamonds": "o"}[suit])
    except (KeyError, IndexError, TypeError, ValueError):
        return None


class OnlineProtocolReader:
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
        self.match_no = 0
        self.match_finished = False
        self.notices: List[dict] = []
        self.afk = {}
        self.sync_required = False
        self.pause_reason = ''
        self.last_sq = None
        self._last_event = None
        self.turn_deadline = None
        self.timer_source = {}
        self.protocol_events = []
        self.public_reveals = []
        self._room_key = None
        self._boundary_unnotified = False
        self._start_notified = False

    @staticmethod
    def _room_hash(value):
        return hashlib.sha256(value.encode()).hexdigest() if isinstance(value, str) and value else None

    def _start_notice(self, msg):
        key = self._room_hash(msg[1] if len(msg) > 1 else None)
        same = key is not None and key == self._room_key and not self.match_finished
        pending = (self.match_no > 0 and not self.match_finished and self._boundary_unnotified
                   and (key is None or self._room_key is None))
        anonymous_repeat = (key is None and self._start_notified and not self.match_finished
                            and self.scores == [0, 0])
        if same or pending or anonymous_repeat:
            if key is not None:
                self._room_key = key
            self._boundary_unnotified, self._start_notified = False, True
            return False
        self.new_game(key)
        self._start_notified = True
        return True

    def _desync(self, reason, require_sync=True):
        self.in_hand = False
        self.sync_required |= require_sync
        self.pause_reason = reason
        self.unknown.append(reason)
        self.turn_deadline = None

    def on_disconnect(self):
        self._desync('websocket disconnected')

    def remaining_s(self):
        if self.turn_deadline is None:
            return None
        return max(0., self.turn_deadline - time.monotonic())

    def is_actionable(self):
        remaining = self.remaining_s()
        return (self.in_hand and not self.sync_required and not self.afk.get(ME, False)
                and self.current == self.me_pos
                and (remaining is None or remaining > .4))

    def public_context(self):
        return {'afk': dict(self.afk), 'remaining_s': self.remaining_s(),
                'timer_source': dict(self.timer_source), 'last_sq': self.last_sq,
                'sync_required': self.sync_required, 'pause_reason': self.pause_reason,
                'public_reveals': list(self.public_reveals)}

    # ------------------------------------------------------------------
    def on_frame(self, direction: str, data: Any) -> None:
        j = data.get("json") if isinstance(data, dict) else None
        if direction == "sent":
            if isinstance(j, list) and j:
                self.sent.append(j)
                if j[0] == 'playCard' and len(j)>1 and isinstance(j[1], dict) and j[1].get('hidden'):
                    self._pending_my_hidden = card_from_playcard(j)
            return
        if isinstance(j, list) and j and j[0] == "startGame":
            fresh = self._start_notice(j)
            self.protocol_events.append({'tp': 'startGame', 'duplicate': not fresh, 'sq': None})
            return
        if isinstance(data, dict) and "b64" in data:
            try:
                ev = cbor_loads(base64.b64decode(data["b64"]))
            except Exception as e:  # anexo que não é evento de jogo
                self._desync(f'cbor: {type(e).__name__}')
                return
            if isinstance(ev, dict) and "tp" in ev:
                self.on_event(ev)
            else:
                self._desync('invalid game event envelope')

    def new_game(self, room_key=None) -> None:
        """Partida nova: placar 0x0; valor/modo ficam desconhecidos até o game_sync."""
        self.entry_cents = self.mode = self.n_players = None
        self.scores = [0, 0]
        self.in_hand = False
        self.mao11_pos = None
        self.current, self.flags = None, {}
        self._pending_my_hidden = None
        self.match_finished = False
        self.last_sq, self._last_event = None, None
        self.sync_required, self.pause_reason = False, ''
        self.turn_deadline, self.timer_source = None, {}
        self.afk, self.public_reveals = {}, []
        self._room_key = room_key
        self._boundary_unnotified, self._start_notified = False, False
        self.match_no += 1
        self._events.append(MatchStart(self.match_no))

    def who(self, pos: int) -> int:
        return ME if pos == self.me_pos else OPP

    def _my_opp_scores(self, rp) -> Optional[tuple]:
        if rp is None or self.my_team is None:
            return None
        return rp[self.my_team], rp[1 - self.my_team]

    def on_event(self, ev: Dict[str, Any]) -> None:
        if not isinstance(ev, dict) or not isinstance(ev.get('tp'), str) or not isinstance(ev.get('dt', {}), dict):
            self._desync('malformed game event')
            return
        tp, sq = ev['tp'], ev.get('sq')
        signature = json.dumps(ev, sort_keys=True, default=str)
        # AFK notifications and game_sync are snapshots and can share sq with
        # the preceding transition (observed in capture 20261019_020417).
        snapshot = tp in ('game_sync', 'player_afk')
        if sq is not None:
            if type(sq) is not int or sq < 0:
                self._desync('invalid event sequence')
                return
            if tp != 'game_sync' and self.last_sq is not None:
                if sq == self.last_sq and signature == self._last_event:
                    return  # exact retransmission, never apply a play twice
                if sq != self.last_sq + 1 and not (snapshot and sq == self.last_sq):
                    self._desync(f'event sequence gap: {self.last_sq}->{sq}')
                    self.last_sq, self._last_event = sq, signature
                    return
        self.protocol_events.append({'tp': tp, 'sq': sq,
                                     'timing': {k: ev['dt'][k] for k in ('rt', 'td') if k in ev.get('dt', {})}})
        try:
            self._apply_event(ev)
        except (ValueError, TypeError, KeyError, IndexError, StopIteration):
            self._desync(f'malformed {tp}')
        if sq is not None:
            self.last_sq, self._last_event = sq, signature

    def _apply_event(self, ev: Dict[str, Any]) -> None:
        tp, dt = ev.get("tp"), ev.get("dt") or {}
        if tp in ('run_turn', 'increase_points', 'last_point'):
            rt = dt.get('rt')
            self.timer_source = {k: dt[k] for k in ('rt', 'td') if k in dt}
            if rt is not None:
                if not isinstance(rt, (int, float)) or not math.isfinite(rt) or not 0 <= rt <= 3600:
                    self._desync('invalid turn timer')
                    return
                self.turn_deadline = time.monotonic() + rt
            else:
                self.turn_deadline = None
        if tp in ('player_afk', 'cardsRevealed'):
            notice = {'type': tp, 'who': self.who(dt['pos']) if 'pos' in dt and self.me_pos is not None else None}
            if tp == 'player_afk':
                notice['afk'] = dt.get('afk')
                if notice['who'] is not None and type(dt.get('afk')) is bool:
                    self.afk[notice['who']] = dt['afk']
            if tp == 'cardsRevealed':
                notice['cards'] = [card_from_code(c) for c in dt.get('cs', [])]
                notice['in_hand'] = self.in_hand
                self.public_reveals.append(dict(notice))
                if self.in_hand and notice['who'] == OPP and any(c is not None for c in notice['cards']):
                    # E9's frozen observation cannot represent remaining revealed
                    # opponent cards. Do not decide using an incomplete input.
                    self._desync('unsupported public opponent reveal', require_sync=False)
            self.notices.append(notice)
            return
        if tp == 'finish_game':
            self.match_finished = True
        if tp == "game_sync":
            key = self._room_hash(dt.get('rid'))
            fresh = (self.match_no == 0 or self.match_finished
                     or (key is not None and self._room_key is not None and key != self._room_key))
            if fresh:
                self.new_game(key)
                self._boundary_unnotified = True
            elif key is not None:
                self._room_key = key
            self.sync_required, self.pause_reason = False, ''
            self.turn_deadline = None
            self.entry_cents = dt.get("ef")
            self.mode, self.n_players = dt.get("mt"), dt.get("np")
            for p in dt.get("ps", []):
                if p.get("mp"):
                    self.me_pos, self.my_team = p.get("pos"), p.get("tm")
            sc = (dt.get("sc") or {}).get("rp")
            if sc is not None:
                self.scores = list(sc)
                if not fresh and self.scores != [0, 0]:
                    self._boundary_unnotified = False
            self.current = (dt.get("tn") or {}).get("cpp")
            self.in_hand = False  # entrou no meio da mão: espera a próxima
        elif tp == "start_round":
            self.current, self.flags = None, {}
            self.turn_deadline, self.timer_source = None, {}
            self.public_reveals = []
            if self.sync_required:
                return  # missing score/history requires an authoritative sync first
            self.pause_reason = ''
            visible = [p for p in dt.get("ps", []) if p.get("cs") and "x" not in p["cs"]]
            if len(visible) == 1 and visible[0]["pos"] != self.me_pos:
                # as cartas visíveis são as minhas: corrige a posição (1v1: time == posição)
                self.me_pos = self.my_team = visible[0]["pos"]
            if self.me_pos is None or self.scores is None:
                return
            self.mao11_pos, self.plays_in_hand = None, 0
            self._pending_my_hidden = None
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
            self.current, self.flags = None, {}
            self.turn_deadline = None
            self.plays_in_hand += 1
            hidden = bool(dt.get("hd"))
            card = card_from_code(dt.get("cd", "x"))
            if dt.get("pos") == self.me_pos and card is None:
                card = self._pending_my_hidden
            if dt.get("pos") == self.me_pos:
                self._pending_my_hidden = None
            self._events.append(Play(self.who(dt["pos"]), card, covered=hidden))
        elif tp == "increase_points":
            # ``akt`` = time que pediu (1v1: posição). ``akp`` não é confiável:
            # já veio 0 num pedido do time 1 (captura 20261019_020041, t=96 s).
            asker = dt["akt"] if "akt" in dt else dt["akp"]
            self._events.append(Raise(self.who(asker)))
            self.current = dt.get("ant")
        elif tp == "player_decision":
            self.current, self.flags = None, {}
            self.turn_deadline = None
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
            self.in_hand = False

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
        notices, self.notices = self.notices, []
        return TableState(events=evs, table_value_text=self.table_value_text(),
                          my_turn=self.is_actionable(),
                          buttons=dict(self.flags), notices=notices)
