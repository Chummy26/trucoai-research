"""Executa jogadas na mesa da PlayTruco clicando na interface, com verificação.

Cada clique é conferido pela mensagem que o próprio cliente da PlayTruco envia
ao servidor (lida pelo leitor do protocolo):

    jogar carta  -> ["playCard", {value, suit, hidden}] com a carta pretendida
    truco        -> ["increasePoints"]; aumento em resposta -> ["reraiseIncreasePoints"]
    aceitar      -> ["acceptIncreasePoints"]
    recusar      -> ["refuseIncreasePoints"]; correr na vez -> ["giveUpRound"]

Se a mensagem não chegar em ``timeout_s`` ou for diferente, o executor se
desativa e a IA para de agir (falha fechada). Botão de ação que some com a
animação do turno antes do toque é re-esperado; aumento sem botão re-decide
sem a aposta (``fallback_needed``), mantendo a mão viva. Hipóteses ainda não verificadas
ao vivo: as cartas da mão são ``<button>`` sem texto na parte de baixo da mesa.
A carta de cada botão é identificada pelo recorte do spritesheet (``sprite_card``),
não pela ordem: a ordem na tela já divergiu da do servidor.
Carta encoberta: clica "Virar" (não é <button>: localizado pelo texto) e depois a
carta; confere ``playCard`` com ``hidden: true`` (fluxo ainda não confirmado ao vivo).
Mão de ferro (às cegas): clica a posição escolhida; confere só que houve ``playCard``.
Mão de onze (fase "mao11"): botões "Aceitar"/"Correr"; correr envia
``refuseLastPoint`` (observado); aceitar deve enviar ``acceptLastPoint``
(deduzido). ``_mao11_manual`` fica como reserva para o modo remoto.
"""

from __future__ import annotations

import time
from typing import List, Optional

from ..engine import ACCEPT, RAISE, RUN
from .human import HumanCursor
from .playtruco_protocol import PlayTrucoProtocolReader, code_from_card

_RAISE_LABELS = ("Truco!", "Seis!", "Nove!", "Doze!")

HAND_BUTTONS_JS = r"""
() => {
  const vh = window.innerHeight;
  return [...document.querySelectorAll('button')]
    .filter(b => !(b.innerText || '').trim() && !b.getAttribute('aria-label'))
    .map(b => { const r = b.getBoundingClientRect();
      const spr = [b, ...b.querySelectorAll('*')].map(e => getComputedStyle(e))
        .find(cs => (cs.backgroundImage || '').includes('cards'));
      return {x: r.x + r.width / 2, y: r.y + r.height / 2, w: r.width, h: r.height,
              pos: spr ? spr.backgroundPosition : null, size: spr ? spr.backgroundSize : null}; })
    .filter(r => !!r.pos && r.w >= 14 && r.h >= 18 && r.h >= r.w * 0.8 &&
                 r.w > 0 && r.h > 0 && r.y > vh * 0.45)
    .sort((a, b) => a.x - b.x);
}
"""

# Spritesheet das cartas (cards-spritesheet.*.webp): 41 quadros na horizontal
# (quadro 0 = verso). Naipes em blocos de 10: paus 1-10, ouros 11-20,
# espadas 21-30, copas 31-40; dentro do naipe, valores 1..7, 11, 12, 13.
# Derivado de 39 cliques verificados (logs/playtruco/*_remote.jsonl).
_SPRITE_SUITS = ["c", "d", "s", "h"]
_SPRITE_VALUES = [1, 2, 3, 4, 5, 6, 7, 11, 12, 13]


def sprite_card(pos: Optional[str], size: Optional[str]) -> Optional[int]:
    """Carta desenhada no botão, pelo recorte do spritesheet; None se desconhecida."""
    from .playtruco_protocol import card_from_code
    try:
        x = -float(pos.split()[0].replace("px", ""))
        frame_w = float(size.split()[0].replace("px", "")) / 41
    except (AttributeError, ValueError, IndexError):
        return None
    frame = round(x / frame_w)
    if abs(frame * frame_w - x) > 1 or not 1 <= frame <= 40:
        return None
    suit, idx = divmod(frame - 1, 10)
    return card_from_code(f"{_SPRITE_SUITS[suit]}{_SPRITE_VALUES[idx]}")


# Sonda passiva (só leitura): botões da mão com o recorte do spritesheet de cada
# carta, e rótulos dos botões visíveis. Usada no shadow para validar o executor.
PROBE_JS = r"""
() => {
  const vh = window.innerHeight;
  const hand = [...document.querySelectorAll('button')]
    .filter(b => !(b.innerText || '').trim() && !b.getAttribute('aria-label'))
    .map(b => {
      const r = b.getBoundingClientRect();
      const spr = [b, ...b.querySelectorAll('*')].map(e => getComputedStyle(e))
        .find(cs => (cs.backgroundImage || '').includes('cards'));
      return {x: Math.round(r.x + r.width / 2), y: Math.round(r.y + r.height / 2),
              w: Math.round(r.width), h: Math.round(r.height),
              pos: spr ? spr.backgroundPosition : null, size: spr ? spr.backgroundSize : null};
    })
    .filter(r => !!r.pos && r.w >= 14 && r.h >= 18 && r.h >= r.w * 0.8 &&
                 r.w > 0 && r.h > 0 && r.y > vh * 0.45)
    .sort((a, b) => a.x - b.x);
  const labels = [...document.querySelectorAll('button')]
    .filter(b => { const r = b.getBoundingClientRect(); return r.width > 0 && r.height > 0; })
    .map(b => (b.innerText || '').trim()).filter(Boolean);
  const virar = [...document.querySelectorAll('body *')]
    .filter(e => (e.innerText || '').trim() === 'Virar' && e.children.length <= 2)
    .map(e => { const r = e.getBoundingClientRect();
      return {tag: e.tagName.toLowerCase(), role: e.getAttribute('role'), w: Math.round(r.width),
              h: Math.round(r.height), cls: (typeof e.className === 'string' ? e.className : '').slice(0, 80)}; })
    .filter(v => v.w > 0 && v.h > 0);
  return {hand, labels, virar, vw: window.innerWidth, vh};
}
"""


class PlayTrucoActor:
    supports_cover = True

    def __init__(self, reader: PlayTrucoProtocolReader, timeout_s: float = 2.5):
        self.reader = reader
        self.timeout_s = timeout_s
        self.enabled = True
        self.reason = ""
        self.fallback_needed = False  # botão ausente mas re-decidível (só RAISE)
        self.last_mismatch: Optional[dict] = None  # clique que jogou carta ≠ da escolhida
        self.hand: List[int] = []  # cartas restantes na ordem do servidor
        self.learned_mao11: Optional[dict] = None
        self.cursor = HumanCursor()

    # chamados pelo laço principal
    def start_hand(self, cards: List[int]) -> None:
        self.hand = list(cards)
        self.cursor.new_hand()

    def played(self, card: Optional[int]) -> None:
        if card in self.hand:
            self.hand.remove(card)

    def _disable(self, why: str) -> None:
        self.enabled = False
        self.reason = why

    def _stable_hand_buttons(self, page, wait_s: float = 6.0):
        """Botões da mão em estado estável (duas leituras idênticas espaçadas).

        A IA pode juntar-se à mesa NO MEIO de uma mão: o DOM mostra só as cartas
        que restam — a contagem não precisa bater com a mão do protocolo. O que
        interessa é estabilidade (a distribuição terminar de animar)."""
        deadline = time.monotonic() + wait_s
        while True:
            buttons = page.evaluate(HAND_BUTTONS_JS)

            def _sig(bs) -> tuple:
                return tuple((round(b["x"]), round(b["y"]), b["pos"]) for b in bs)

            page.wait_for_timeout(250)
            again = page.evaluate(HAND_BUTTONS_JS)
            if len(again) > 0 and len(again) == len(buttons) and _sig(again) == _sig(buttons):
                return again  # estável (contagem qualquer; quem cham decide)
            if time.monotonic() >= deadline:
                return again if again else buttons  # o mais recente vence: a mão anima sem parar

    def _match_card_button(self, page, card: int, patience_s: float = 8.0):
        """Botão que mostra ``card`` entre os estáveis. Exige que TODAS as cartas
        da tela sejam reconhecíveis e subconjunto da mão registrada (a IA pode ter
        entrado no meio da mão: menos cartas na tela que na mão é estado válido).
        Paciente com a distribuição: na primeira partida a vez pode chegar antes
        das cartas terminarem de animar (tela vazia não é desistência imediata)."""
        deadline = time.monotonic() + patience_s
        while True:
            buttons = self._stable_hand_buttons(page)
            shown = [sprite_card(b.get("pos"), b.get("size")) for b in (buttons or [])]
            if not shown or any(s is None for s in shown):
                if time.monotonic() >= deadline:
                    return None, f"cartas ilegíveis na tela: {shown}"
                page.wait_for_timeout(500)
                continue
            hand_set = set(self.hand)
            shown_set = set(shown)
            if not shown_set <= hand_set:
                if time.monotonic() >= deadline:
                    return None, f"cartas na tela {sorted(shown_set)} fora da mão {sorted(hand_set)}"
                page.wait_for_timeout(400)
                continue  # distribuição ainda anima: pode mostrar recortes provisórios
            if card not in shown_set:
                if time.monotonic() >= deadline:
                    return None, f"carta {card} não está visível na mesa (mostradas {sorted(shown_set)})"
                page.wait_for_timeout(400)
                continue  # as cartas chegam uma a uma na distribuição
            if shown.count(card) != 1 and len(shown) != len(set(shown)):
                if time.monotonic() >= deadline:
                    return None, f"carta duplicada na tela: {shown}"
                page.wait_for_timeout(400)
                continue
            return buttons[shown.index(card)], ""

    def _wait_sent(self, page, start: int, check, disable: bool = True) -> bool:
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            page.wait_for_timeout(100)  # processa eventos do WebSocket
            for msg in self.reader.sent[start:]:
                if check(msg):
                    return True
                if msg and msg[0] not in ("syncGameState", "isInGame"):
                    self._disable(f"mensagem inesperada após clique: {msg}")
                    return False
        if disable:
            self._disable("nenhuma mensagem do cliente após o clique")
        return False

    def _click_card_retry(self, page, b: dict, check, phase: str = "card",
                          intended: Optional[int] = None, attempts: int = 3) -> bool:
        """Clica a carta e espera a mensagem, re-tentando: logo após um truco
        aceito o cliente engole cliques durante a acomodação da mesa."""
        for attempt in range(attempts):
            start = len(self.reader.sent)
            self._human_click(page, b, phase)
            ok, msg = self._wait_card_msg(page, start, check)
            if ok:
                return True
            if not self.enabled:  # mensagem inesperada: falha fechada, já desligado
                return False
            if msg is not None:
                # Jogada que partiu com OUTRA carta: clique que deslizou (ou a mesa
                # resolveu outro alvo). A partida segue coerente com o leitor —
                # registra a divergência e aceita a jogada, sem desativar.
                from .playtruco_protocol import card_from_playcard
                got = card_from_playcard(msg)
                got_in_hand = got is not None and got in self.hand
                if got_in_hand:
                    self.last_mismatch = {
                        "intended": intended, "sent": msg,
                        "clicked": {"x": b["x"], "y": b["y"], "w": b["w"], "h": b["h"]},
                    }
                    return True
                return False  # carta estranha ao motor: dessincronização visível
            self.cursor.pause_ms(320)
        self._disable("nenhuma mensagem do cliente após o clique (3 tentativas)")
        return False

    def _wait_card_msg(self, page, start: int, check):
        """Espera a mensagem da jogada. ``ok`` diz se a carta esperada veio;
        ``msg`` traz o primeiro envio relevante encontrado."""
        deadline = time.monotonic() + self.timeout_s
        while time.monotonic() < deadline:
            page.wait_for_timeout(100)  # processa eventos do WebSocket
            for msg in self.reader.sent[start:]:
                if check(msg):
                    return True, msg
                if msg and msg[0] == "playCard":
                    return False, msg
                if msg and msg[0] not in ("syncGameState", "isInGame"):
                    self._disable(f"mensagem inesperada após clique: {msg}")
                    return False, msg
        return None, None

    def _human_click(self, page, b: dict, phase: str) -> None:
        self.cursor.click(page, b["x"], b["y"], b.get("w", 36.0), b.get("h", 54.0), phase)

    def _locate(self, page, label):
        """Botão com o rótulo; se não houver, o elemento visível com o texto exato
        (ex.: "Virar" não é <button> na PlayTruco)."""
        loc = page.get_by_role("button", name=label, exact=True)
        if loc.count() == 1 and loc.is_visible():
            return loc
        loc = page.get_by_text(label, exact=True)
        vis = [loc.nth(i) for i in range(loc.count()) if loc.nth(i).is_visible()]
        return vis[-1] if len(vis) >= 1 else None  # o mais interno/último visível

    def _wait_locate(self, page, labels, wait_s: float):
        """Espera qualquer um dos rótulos existir no DOM (a barra de ação da mesa
        aparece e desaparece com a animação do turno)."""
        deadline = time.monotonic() + wait_s
        while True:
            for label in labels:
                loc = self._locate(page, label)
                if loc is not None:
                    return loc
            if time.monotonic() >= deadline:
                return None
            page.wait_for_timeout(200)

    def _hit_wait(self, page, loc, bb: dict, wait_s: float = 4.0) -> bool:
        """Espera o elemento ficar de fato acessível no ponto: capas/overlays
        translúcidos interceptam o toque e nenhuma mensagem sai do cliente."""
        el = loc.element_handle()
        if el is None:
            return False
        deadline = time.monotonic() + wait_s
        while True:
            cx = bb["x"] + bb["width"] / 2.0
            cy = bb["y"] + bb["height"] / 2.0
            hit = page.evaluate(
                """([x, y]) => { const e = document.elementFromPoint(x, y);
                    return e ? {tag: e.tagName, cls: (typeof e.className === 'string'? e.className:'')}
                              : null; }""", [cx, cy])
            cur = page.evaluate(
                """(el) => { if (!el || !el.isConnected) return null;
                    const r = el.getBoundingClientRect();
                    return {x: r.x, y: r.y, w: r.width, h: r.height}; }""", el)
            if cur is not None and hit is not None:
                owner = page.evaluate(
                    """([el, x, y]) => { const e = document.elementFromPoint(x, y);
                        return !!(e && (e === el || el.contains(e))); }""",
                    [el, cur["x"] + cur["w"] / 2, cur["y"] + cur["h"] / 2])
                if owner and not (time.monotonic() >= deadline):
                    bb.update(x=cur["x"], y=cur["y"], width=cur["w"], height=cur["h"])
                    return True
            if time.monotonic() >= deadline:
                return cur is not None
            page.wait_for_timeout(250)

    def _click_text(self, page, labels, phase: str = "card") -> bool:
        """Uma tentativa de clique pelo rótulo; o re-tentativo mora em ``_click_expect``."""
        for label in labels:
            loc = self._locate(page, label)
            if loc is not None:
                bb = loc.bounding_box()
                if bb is not None:
                    if self._hit_wait(page, loc, bb):
                        cx = bb["x"] + bb["width"] / 2.0
                        cy = bb["y"] + bb["height"] / 2.0
                        self._human_click(page, {"x": cx, "y": cy,
                                                 "w": bb["width"], "h": bb["height"]}, phase)
                        return True
                    loc.click(timeout=900)  # caminho ancorado do Playwright
                    return True
                loc.click()
                return True
        return False

    def _click_expect(self, page, labels, want: str, phase: str,
                      attempts: int = 3, absent_fallback: bool = False) -> bool:
        """Clica pelo rótulo e espera a mensagem esperada, re-tentando o clique:
        o banner da mesa engole o primeiro toque com frequência. Botão que some
        com a animação do turno antes do toque é re-esperado (janela crescente),
        não um desligamento imediato."""
        if self._wait_locate(page, labels, 6.0) is None:
            # A mesa não desenhou o botão: em acordos onde o motor jura
            # legalidade (aumento no Paulista), o laço re-decide sem essa ação
            # em vez de matar a mão.
            if absent_fallback:
                self.fallback_needed = True
                return False
            self._disable(f"botão {labels} não apareceu em 6 s")
            return False
        for attempt in range(attempts):
            if self._click_text(page, labels, phase):
                start = len(self.reader.sent)
                if self._wait_sent(page, start, lambda m: m[0] == want, disable=False):
                    return True
                if not self.enabled:
                    return False
            else:
                # A barra de ação some com a animação do turno entre a espera
                # e o clique: re-espera o botão em vez de matar o executor na
                # primeira falha (era a queda recorrente em pedidas de truco).
                # Só desativa (ou aplica o fallback do aumento) quando o botão
                # não volta dentro da janela de re-espera.
                if self._wait_locate(page, labels, 3.0 + attempt) is None:
                    if absent_fallback:
                        self.fallback_needed = True
                        return False
                    self._disable(f"botão {labels} não encontrado")
                    return False
            self.cursor.pause_ms(350)
        self._disable(f"nenhuma mensagem {want} após o clique ({attempts} tentativas)")
        return False

    def _mao11_manual(self, page, action: int, start: int) -> bool:
        """Pede o clique humano na decisão da IA e aprende a mensagem enviada."""
        what = "JOGAR (aceitar a mão de onze)" if action == ACCEPT else "CORRER (recusar a mão de onze)"
        print(f"\n>>> MÃO DE ONZE: a IA decidiu {what}. Clique esse botão na mesa agora. <<<\n",
              flush=True)
        deadline = time.monotonic() + 12.0
        while time.monotonic() < deadline:
            page.wait_for_timeout(100)
            new = [m for m in self.reader.sent[start:] if m and m[0] != "playCard"]
            if new:
                self.learned_mao11 = {"action": "accept" if action == ACCEPT else "run",
                                      "message": new[0]}
                print(f"[executor] mensagem da mão de onze gravada: {new[0]}", flush=True)
                return True
        self._disable("mão de onze: nenhum clique registrado a tempo")
        return False

    def _button_for(self, buttons, card: int):
        """Botão que mostra ``card`` (pelo sprite). Exige que TODAS as cartas da
        tela sejam reconhecidas e batam com a mão registrada."""
        shown = [sprite_card(b.get("pos"), b.get("size")) for b in buttons]
        if None in shown or sorted(shown) != sorted(self.hand):
            return None, f"cartas na tela {shown} != mão {self.hand}"
        return buttons[shown.index(card)], ""

    def dry_run(self, page, obs, action: int) -> dict:
        """Ensaia o clique que ``do`` faria, SEM clicar (só leitura e ``trial``).

        Confere: o botão do rótulo existe, está visível e receberia o clique
        (``click(trial=True)``); para cartas, que o número de cartas na tela bate
        e que o ponto do clique cai no botão escolhido (``elementFromPoint``).
        """
        try:
            if obs.phase == "mao11":
                label = "Aceitar" if action == ACCEPT else "Correr"
                return {"ok": self._trial(page, (label,)), "what": ("mao11", label)}
            if action < 6:
                buttons = page.evaluate(HAND_BUTTONS_JS)
                if obs.blind:
                    slots = [i for i in range(3) if obs.my_slots_filled[i]]
                    idx = slots.index(action % 3)
                    n_expected = len(slots)
                else:
                    card = obs.my_hand[action % 3]
                    if card not in self.hand:
                        return {"ok": False, "why": "carta fora da mão registrada"}
                    n_expected = len(self.hand)
                if len(buttons) != n_expected:
                    return {"ok": False, "why": f"{len(buttons)} cartas na tela, {n_expected} esperadas"}
                if obs.blind:
                    b = buttons[idx]
                else:
                    b, why = self._button_for(buttons, card)
                    if b is None:
                        return {"ok": False, "why": why}
                    idx = buttons.index(b)
                hit = page.evaluate("""([x, y]) => { const e = document.elementFromPoint(x, y);
                    const btn = e && e.closest('button'); if (!btn) return null;
                    const r = btn.getBoundingClientRect(); return [r.x + r.width / 2, r.y + r.height / 2]; }""",
                                    [b["x"], b["y"]])
                ok = hit is not None and abs(hit[0] - b["x"]) < 2 and abs(hit[1] - b["y"]) < 2
                res = {"ok": ok, "what": "carta", "idx": idx, "n": len(buttons)}
                if action >= 3:
                    res["virar"] = self._trial(page, ("Virar",))
                    res["ok"] = ok and res["virar"]
                return res
            labels = {RAISE: _RAISE_LABELS, ACCEPT: ("Aceitar",),
                      RUN: ("Recusar",) if obs.phase == "respond" else ("Correr",)}[action]
            return {"ok": self._trial(page, labels), "what": labels}
        except Exception as e:
            return {"ok": False, "why": f"erro: {e}"}

    def _trial(self, page, labels) -> bool:
        for label in labels:
            loc = self._locate(page, label)
            if loc is not None:
                loc.click(trial=True, timeout=800)  # checagens do clique, sem clicar
                return True
        return False

    def do(self, page, obs, action: int) -> bool:
        if not self.enabled:
            return False
        self.last_mismatch = None
        start = len(self.reader.sent)
        if action < 6:
            covered = action >= 3
            if obs.blind:
                # Às cegas: a posição escolhida entre as cartas restantes, da esquerda.
                slots = [i for i in range(3) if obs.my_slots_filled[i]]
                buttons = self._stable_hand_buttons(page)
                if buttons is None or len(buttons) != len(slots):
                    self._disable(f"{len(buttons or [])} cartas na tela, {len(slots)} esperadas (mão de ferro)")
                    return False
                b = buttons[slots.index(action % 3)]
                want = lambda m: m[0] == "playCard"
                return self._click_card_retry(page, b, want, phase="card")
            card = obs.my_hand[action % 3]
            if card not in self.hand:
                self._disable("carta escolhida não está na mão registrada")
                return False
            if covered:
                self.reader.note_sent_card(card)
                if not self._click_text(page, ("Virar",), phase="virar"):
                    return False
            b, why = self._match_card_button(page, card)
            if b is None:
                self._disable(why)
                return False
            want = code_from_card(card)
            check = lambda m: m[0] == "playCard" and isinstance(
                m[1], dict) and m[1].get("value") == want["value"] \
                and m[1].get("suit") == want["suit"] and bool(m[1].get("hidden")) == covered
            return self._click_card_retry(page, b, check, phase="card", intended=card)
        if action == RAISE:
            want = "reraiseIncreasePoints" if obs.phase == "respond" else "increasePoints"
            return self._click_expect(page, _RAISE_LABELS, want, phase="raise",
                                      absent_fallback=True)
        if obs.phase == "mao11":
            # Observado: botões "Aceitar"/"Correr"; correr envia refuseLastPoint
            # (logs/playtruco/20261019_022512_remote.jsonl). acceptLastPoint é deduzido
            # por simetria: se vier outro nome, o executor se desativa.
            label, want = ("Aceitar", "acceptLastPoint") if action == ACCEPT else ("Correr", "refuseLastPoint")
            ok = self._click_expect(page, (label,), want, phase="mao11")
            if ok:
                self.learned_mao11 = {"action": label, "message": want}
            return ok
        if action == ACCEPT:
            return self._click_expect(page, ("Aceitar",), "acceptIncreasePoints",
                                      phase="accept")
        if action == RUN:
            respond = obs.phase == "respond"
            label = ("Recusar",) if respond else ("Correr",)
            want = "refuseIncreasePoints" if respond else "giveUpRound"
            return self._click_expect(page, label, want, phase="run")
        self._disable(f"ação {action} não suportada")
        return False
