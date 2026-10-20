"""Integração com a PlayTruco via navegador controlado (Playwright).

Modos:
  capture  só grava o que o navegador recebe (frames WebSocket e texto/estrutura
           da página) enquanto o HUMANO joga. A IA não age nem sugere jogadas.
  shadow   lê a mesa, mantém o jogo-sombra e registra o que a IA faria.
  play     igual a shadow, mas executa a jogada. Exige trava liberada
           (mesa grátis ou paga até o teto configurado) antes de CADA ação
           e seletores marcados como verificados.

O login é feito pelo próprio usuário na janela aberta; o script nunca lê,
digita nem armazena senha. O perfil do navegador fica em ``.pw-profile/``
(fora do controle de versão).

O leitor do protocolo fica em ``playtruco_protocol.py`` e o executor de jogadas
(cliques conferidos pela mensagem enviada) em ``playtruco_actor.py``.
"""

from __future__ import annotations

import base64
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

from .guard import StakeGuard, max_stake_from_env
from .shadow import Event

SITE = "https://www.playtruco.com/"
_SECRET_KEYS = re.compile(r"(token|auth|password|senha|cookie|session|secret|jwt|bearer)", re.I)
_JWT = re.compile(r"eyJ[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}\.[A-Za-z0-9_-]{8,}")


def redact(obj: Any) -> Any:
    """Remove credenciais óbvias antes de gravar em disco."""
    if isinstance(obj, dict):
        return {k: ("<redigido>" if _SECRET_KEYS.search(str(k)) else redact(v))
                for k, v in obj.items()}
    if isinstance(obj, list):
        return [redact(v) for v in obj]
    if isinstance(obj, str):
        return _JWT.sub("<jwt-redigido>", obj)
    return obj


def parse_payload(payload: Any) -> Any:
    """Tenta decodificar JSON (inclusive no formato socket.io ``42[...]``)."""
    if isinstance(payload, (bytes, bytearray)):
        # Anexos binários do socket.io (ex.: ``gameEvent``): guarda o conteúdo.
        return {"binary_len": len(payload), "b64": base64.b64encode(bytes(payload)).decode()}
    text = str(payload)
    m = re.match(r"^(\d+)(.*)$", text, re.S)
    body = m.group(2) if m else text
    try:
        return {"prefix": m.group(1) if m else "", "json": json.loads(body)} if body else {"raw": text}
    except (ValueError, TypeError):
        return {"raw": text[:4000]}


# Script que resume a página: texto visível e elementos interativos.
DOM_SNAPSHOT_JS = r"""
() => {
  const vis = el => { const r = el.getBoundingClientRect(); return r.width > 0 && r.height > 0; };
  const pick = el => ({
    tag: el.tagName.toLowerCase(),
    text: (el.innerText || el.getAttribute('aria-label') || el.alt || '').trim().slice(0, 80),
    cls: (el.className && el.className.baseVal === undefined ? el.className : '').toString().slice(0, 120),
    src: el.src ? el.src.split('/').slice(-2).join('/') : undefined,
    id: el.id || undefined,
    box: (() => { const r = el.getBoundingClientRect(); return [Math.round(r.x), Math.round(r.y), Math.round(r.width), Math.round(r.height)]; })(),
  });
  const els = [...document.querySelectorAll('button, [role=button], img, a, [class*=card], [class*=Card], [class*=carta]')].filter(vis).slice(0, 300).map(pick);
  return { url: location.href, title: document.title, text: document.body.innerText.slice(0, 6000), elements: els };
}
"""


@dataclass
class TableState:
    """Saída de um leitor de mesa a cada sondagem."""
    events: List[Event] = field(default_factory=list)
    table_value_text: Optional[str] = None  # texto bruto do valor da mesa
    my_turn: bool = False
    buttons: Dict[str, Any] = field(default_factory=dict)  # ação -> alvo clicável


def _launch(pw, profile: Path, headless: bool = False, cdp: Optional[str] = None):
    """Abre um Chrome próprio ou conecta ao Chrome do usuário (``cdp``).

    Com ``cdp`` (ex.: ``http://127.0.0.1:9222``) o script se liga ao Chrome que o
    usuário abriu e onde já fez login (inclusive com conta Google). Nada de
    credencial passa pelo script. Ao sair, apenas desconecta.
    """
    if cdp:
        browser = pw.chromium.connect_over_cdp(cdp)
        ctx = browser.contexts[0]
        page = next((p for p in ctx.pages if "playtruco" in p.url), None) or ctx.new_page()
        ctx._trucoai_external = True
        return ctx, page
    profile.mkdir(parents=True, exist_ok=True)
    ctx = pw.chromium.launch_persistent_context(
        str(profile), channel="chrome", headless=headless, viewport={"width": 1280, "height": 860})
    page = ctx.pages[0] if ctx.pages else ctx.new_page()
    return ctx, page


def capture(out_dir: Path, profile: Path, minutes: float = 30.0, every_s: float = 1.5,
            screenshots: bool = True, cdp: Optional[str] = None) -> Path:
    """Grava a sessão enquanto o humano joga. Encerre fechando a janela ou Ctrl+C."""
    from playwright.sync_api import sync_playwright

    out_dir.mkdir(parents=True, exist_ok=True)
    frames_f = open(out_dir / "frames.jsonl", "a", encoding="utf-8")
    dom_f = open(out_dir / "dom.jsonl", "a", encoding="utf-8")
    t0 = time.time()

    def log_frame(url: str, direction: str, payload: Any) -> None:
        frames_f.write(json.dumps({"t": round(time.time() - t0, 3), "url": url, "dir": direction,
                                   "data": redact(parse_payload(payload))}, ensure_ascii=False) + "\n")
        frames_f.flush()

    def on_ws(ws) -> None:
        url = ws.url.split("?")[0]
        log_frame(url, "open", "")
        ws.on("framereceived", lambda p: log_frame(url, "recv", p))
        ws.on("framesent", lambda p: log_frame(url, "sent", p))

    with sync_playwright() as pw:
        ctx, page = _launch(pw, profile, cdp=cdp)
        ctx.on("page", lambda p: p.on("websocket", on_ws))
        for p in ctx.pages:
            p.on("websocket", on_ws)
        # Recarrega para gravar a conexão da mesa desde o início.
        page.goto(SITE)
        print("Gravando. Faça login (se preciso) e jogue normalmente. "
              "A IA NÃO joga nem sugere jogadas neste modo. Ctrl+C para encerrar.")
        last_text, n = None, 0
        try:
            while time.time() - t0 < minutes * 60 and ctx.pages:
                p = ctx.pages[-1]
                try:
                    snap = p.evaluate(DOM_SNAPSHOT_JS)
                except Exception:
                    snap = None
                if snap and snap["text"] != last_text:
                    last_text = snap["text"]
                    dom_f.write(json.dumps({"t": round(time.time() - t0, 3), **redact(snap)},
                                           ensure_ascii=False) + "\n")
                    dom_f.flush()
                    if screenshots and "/truco" in snap["url"]:
                        n += 1
                        p.screenshot(path=str(out_dir / f"shot_{n:04d}.png"))
                p.wait_for_timeout(int(every_s * 1000))
        except KeyboardInterrupt:
            pass
        finally:
            if not getattr(ctx, "_trucoai_external", False):
                try:
                    ctx.close()
                except Exception:
                    pass
    frames_f.close()
    dom_f.close()
    return out_dir


def filter_legal(obs, flags: Dict[str, Any]):
    """Restringe as ações legais do motor às que a mesa oferece agora.

    ``catip`` (pode pedir truco), ``cgr`` (pode correr na vez) e ``chc`` (pode
    virar carta) vêm do ``run_turn``. Sem flags, não restringe.
    """
    import dataclasses

    from ..engine import RAISE, RUN
    if obs.phase != "play" or not flags:
        return obs
    legal = []
    for a in obs.legal_actions:
        if a == RAISE and flags.get("catip") is False:
            continue
        if a == RUN and flags.get("cgr") is False:
            continue
        if 3 <= a < 6 and flags.get("chc") is False:
            continue
        legal.append(a)
    return dataclasses.replace(obs, legal_actions=tuple(legal))


def run_agent(mode: str, agent_spec: str, profile: Path, log_dir: Path,
              cdp: Optional[str] = None, max_hands: Optional[int] = None,
              max_stake: Optional[float] = None) -> None:
    """Laço de shadow/play sobre o protocolo da mesa.

    shadow: decide e registra, sem clicar. play: decide e clica (executor
    verificado). Mesas 1v1 Paulista grátis sempre liberam; mesas pagas liberam
    até o teto ``max_stake`` (reais, por ``--max-stake`` ou ``TRUCOAI_MAX_STAKE``).
    """
    from playwright.sync_api import sync_playwright

    from ..agents import make_agent
    from ..engine import COVER_0, PLAY_0
    from ..rules import PLAYTRUCO_O1
    from .guard import parse_brl
    from .playtruco_actor import PROBE_JS, PlayTrucoActor
    from .playtruco_protocol import PlayTrucoProtocolReader
    from .shadow import ME, HandEnd, HandStart, Play, ShadowGame
    from .silent import SilentDecisions

    reader = PlayTrucoProtocolReader()
    actor = PlayTrucoActor(reader) if mode == "play" else None
    # No shadow, um executor "de ensaio" só acompanha a mão para ensaiar cliques.
    rehearsal = PlayTrucoActor(reader) if mode == "shadow" else None
    agent = make_agent(agent_spec)
    agent.reset(int(time.time()))
    if max_stake is None:
        max_stake = max_stake_from_env()
    guard = StakeGuard(max_stake=max_stake)
    shadow = ShadowGame(rules=PLAYTRUCO_O1)
    log_dir.mkdir(parents=True, exist_ok=True)
    log = open(log_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_{mode}.jsonl", "a", encoding="utf-8")

    def write(rec: dict) -> None:
        rec["t"] = round(time.time(), 3)
        log.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        log.flush()

    write({"start": mode, "agent": agent_spec, "rules": PLAYTRUCO_O1.version,
           "rules_fingerprint": PLAYTRUCO_O1.fingerprint, "meta": getattr(agent, "meta", None)})
    last_block, decided_at, hands, divs, last_table = "", None, 0, 0, None
    turn_seen = None
    silent = SilentDecisions()
    probe_due = 0.0  # instante (monotônico) da próxima sonda passiva da interface

    def probe(tag: str) -> None:
        try:
            write({"ui_probe": tag, **page.evaluate(PROBE_JS)})
        except Exception as e:  # sonda nunca derruba o laço
            write({"ui_probe_error": str(e)})

    def flush_silent() -> None:
        out = silent.flush()
        if out["n"]:
            write(out)
            dr = out.get("dry_run_ok")
            print(f"[shadow] partida encerrada: {out['n']} decisões silenciosas gravadas no log"
                  + (f"; ensaios de clique ok: {dr:.0%}" if dr is not None else ""))

    with sync_playwright() as pw:
        ctx, page = _launch(pw, profile, cdp=cdp)

        def on_ws(ws) -> None:
            ws.on("framereceived", lambda p: reader.on_frame("recv", parse_payload(p)))
            ws.on("framesent", lambda p: reader.on_frame("sent", parse_payload(p)))

        page.on("websocket", on_ws)
        page.goto(SITE)  # recarrega para acompanhar a conexão da mesa desde o início
        print(f"[{mode}] Sente numa mesa 1v1 Paulista"
              + (f" grátis ou paga até R$ {max_stake:.2f}" if max_stake is not None else " GRÁTIS")
              + ". Acima do teto a IA não age.")
        try:
            while ctx.pages and (actor is None or actor.enabled):
                page.wait_for_timeout(150)
                ts = reader.poll()
                guard.update(parse_brl(ts.table_value_text), "protocolo (game_sync.ef)")
                table = (reader.mode, reader.n_players, reader.entry_cents)
                if table != last_table and reader.entry_cents is not None:
                    write({"mesa": {"mode": reader.mode, "n_players": reader.n_players,
                                    "entry_text": ts.table_value_text,
                                    "entry_cents": reader.entry_cents,
                                    "allowed": guard.check().allowed}})
                    print(f"[mesa] modo={reader.mode} jogadores={reader.n_players} "
                          f"entrada={ts.table_value_text or '?'} -> "
                          f"{'IA pode jogar' if guard.check().allowed else 'IA bloqueada'}")
                    last_table = table
                for ev in ts.events:
                    silent.observe_event(ev)
                    shadow.feed(ev)
                    if isinstance(ev, HandEnd) and shadow.state is not None                         and shadow.state.match_winner is not None:
                        if hasattr(agent, "hits"):
                            h, m = agent.hits, agent.misses
                            write({"agent_coverage": {
                                "hits": h, "misses": m,
                                "hit_rate": round(h / max(1, h + m), 3)}})
                            agent.hits = 0
                            agent.misses = 0
                        flush_silent()  # só depois do fim da partida
                    if isinstance(ev, HandEnd):
                        ok = not shadow.divergences
                        print(f"[sombra] mão {hands} encerrada {ev.my_score}x{ev.opp_score} "
                              f"{'- motor bate com a mesa' if ok else '- DIVERGÊNCIA'} "
                              f"(total: {hands} mãos, {divs + len(shadow.divergences)} divergências)")
                    if isinstance(ev, HandStart):
                        hands += 1
                        decided_at = None
                        probe_due = time.monotonic() + 1.5  # após a animação de distribuição
                        for ac in (actor, rehearsal):
                            if ac and ev.my_cards is not None:
                                ac.start_hand(list(ev.my_cards))
                    elif isinstance(ev, Play) and ev.who == ME:
                        for ac in (actor, rehearsal):
                            if ac:
                                ac.played(ev.card)
                    write({"event": repr(ev)})
                for d in shadow.divergences:
                    divs += 1
                    write({"divergence": d.__dict__})
                    print(f"[sombra] divergência: {d.detail}")
                shadow.divergences.clear()
                if reader.unknown:
                    write({"unknown": reader.unknown})
                    reader.unknown.clear()
                if max_hands and hands > max_hands:
                    break
                if probe_due and time.monotonic() >= probe_due:
                    probe_due = 0.0
                    probe("inicio_da_mao")
                obs = shadow.observation()
                if not ts.my_turn or obs is None:
                    continue
                key = (len(obs.public_log), obs.phase)
                if turn_seen != key:
                    turn_seen = key
                    write({"my_turn": obs.phase, "legal": obs.legal_actions, "flags": ts.buttons,
                           "hand_cards": [c for c in obs.my_hand if c is not None]})
                    probe("minha_vez")
                    print(f"[sombra] vez do jogador detectada (fase {obs.phase}, "
                          f"{len(obs.legal_actions)} ações legais)")
                if decided_at == key:
                    continue  # já decidiu neste ponto; aguardando a mesa
                decision = guard.check()
                if not decision.allowed:
                    if decision.reason != last_block:
                        print(f"[trava] {decision.reason}")
                        write({"blocked": decision.reason})
                        last_block = decision.reason
                    if mode == "shadow" and decision.value and decision.value > 0:
                        # Mesa paga identificada: decide em silêncio (nada impresso,
                        # log só no fim da partida). Nunca executa.
                        sobs = filter_legal(obs, ts.buttons)
                        if sobs.legal_actions:
                            t0 = time.perf_counter()
                            a = agent.act(sobs)
                            ms = round(1000 * (time.perf_counter() - t0), 1)
                            silent.decide(sobs, a, ms, rehearsal.dry_run(page, sobs, a))
                        decided_at = key
                    continue
                obs = filter_legal(obs, ts.buttons)
                if not obs.legal_actions:
                    write({"no_legal_actions": True, "flags": ts.buttons})
                    continue
                t0 = time.perf_counter()
                action = agent.act(obs)
                ms = round(1000 * (time.perf_counter() - t0), 1)
                if actor and 3 <= action < 6 and not actor.supports_cover:
                    action = PLAY_0 + (action - COVER_0)  # "Virar" ainda não suportado
                decided_at = key
                write({"decision": action, "ms": ms, "legal": obs.legal_actions, "phase": obs.phase})
                if actor:
                    if not guard.check().allowed:  # confere de novo imediatamente antes do clique
                        continue
                    if obs.blind and action < 6:
                        shadow.blind_slot_hint = action % 3
                    ok = actor.do(page, obs, action)
                    if not ok and actor.enabled and actor.fallback_needed:
                        # A mesa não desenhou o botão do aumento: decide de novo
                        # sem essa opção e executa a alternativa (a mão segue viva).
                        actor.fallback_needed = False
                        import dataclasses
                        alt = tuple(a for a in obs.legal_actions if a != action)
                        write({"fallback": {"absent_action": action, "legal": obs.legal_actions}})
                        if alt:
                            if not guard.check().allowed:
                                continue
                            altobs = dataclasses.replace(obs, legal_actions=alt)
                            t1 = time.perf_counter()
                            new_action = agent.act(altobs)
                            ms = round(1000 * (time.perf_counter() - t1), 1)
                            if obs.blind and new_action < 6:
                                shadow.blind_slot_hint = new_action % 3
                            write({"decision": new_action, "ms": ms,
                                   "legal": alt, "phase": obs.phase, "fallback": True})
                            action = new_action
                            ok = actor.do(page, obs, action)
                    write({"acted": ok, "reason": actor.reason,
                           "learned_mao11": actor.learned_mao11,
                           "misclick": actor.last_mismatch})
                    if not ok and not actor.enabled:
                        print(f"[executor] desativado: {actor.reason}")
        except KeyboardInterrupt:
            print("[shadow/play] encerrado pelo usuário")
        finally:
            flush_silent()
            log.close()



def action_label(obs, a: int) -> str:
    """Descrição de uma ação para o humano (modo remoto)."""
    from .. import cards as C
    from ..engine import ACCEPT, RAISE, RUN
    if a < 6:
        c = obs.my_hand[a % 3]
        name = "?" if c is None else C.name(c)
        return f"{'virar (encoberta)' if a >= 3 else 'jogar'} {name}"
    if a == RAISE:
        nxt = {1: "truco", 3: "seis", 6: "nove", 9: "doze"}
        base = obs.pending if obs.phase == "respond" else obs.stake
        return f"pedir {nxt.get(base, 'aumento')}"
    if a == ACCEPT:
        return "jogar a mão de onze" if obs.phase == "mao11" else "aceitar"
    if a == RUN:
        return {"mao11": "correr da mão de onze", "respond": "recusar"}.get(obs.phase, "correr")
    return str(a)


def run_remote(profile: Path, log_dir: Path, cdp: Optional[str] = None) -> None:
    """Controle remoto: o HUMANO escolhe cada jogada no terminal e o executor
    clica na mesa, conferindo a mensagem enviada. Nenhuma IA é carregada:
    serve para testar os cliques em qualquer mesa sem a IA decidir nada.
    Digite o número da opção, ou "m" para clicar você mesmo na mesa.
    """
    from playwright.sync_api import sync_playwright

    from ..rules import PLAYTRUCO_O1
    from .playtruco_actor import PROBE_JS, PlayTrucoActor
    from .playtruco_protocol import PlayTrucoProtocolReader
    from .shadow import ME, HandEnd, HandStart, Play, ShadowGame

    reader = PlayTrucoProtocolReader()
    actor = PlayTrucoActor(reader)
    shadow = ShadowGame(rules=PLAYTRUCO_O1)
    log_dir.mkdir(parents=True, exist_ok=True)
    log = open(log_dir / f"{time.strftime('%Y%m%d_%H%M%S')}_remote.jsonl", "a", encoding="utf-8")

    def write(rec: dict) -> None:
        rec["t"] = round(time.time(), 3)
        log.write(json.dumps(rec, ensure_ascii=False, default=str) + "\n")
        log.flush()

    write({"start": "remote", "rules": PLAYTRUCO_O1.version})
    results = []
    with sync_playwright() as pw:
        ctx, page = _launch(pw, profile, cdp=cdp)

        def on_ws(ws) -> None:
            ws.on("framereceived", lambda p: reader.on_frame("recv", parse_payload(p)))
            ws.on("framesent", lambda p: reader.on_frame("sent", parse_payload(p)))

        page.on("websocket", on_ws)
        page.goto(SITE)
        print("[remoto] Sente numa mesa. Na sua vez, escolha a jogada aqui; o executor clica.")
        asked = None
        try:
            while ctx.pages:
                page.wait_for_timeout(150)
                ts = reader.poll()
                for ev in ts.events:
                    shadow.feed(ev)
                    write({"event": repr(ev)})
                    if isinstance(ev, HandStart) and ev.my_cards is not None:
                        actor.start_hand(list(ev.my_cards))
                    elif isinstance(ev, Play) and ev.who == ME:
                        actor.played(ev.card)
                    elif isinstance(ev, HandEnd):
                        print(f"[remoto] mão encerrada {ev.my_score}x{ev.opp_score}")
                for d in shadow.divergences:
                    write({"divergence": d.__dict__})
                    print(f"[sombra] divergência: {d.detail}")
                shadow.divergences.clear()
                obs = shadow.observation()
                if not ts.my_turn or obs is None:
                    continue
                key = (len(obs.public_log), obs.phase)
                if asked == key:
                    continue
                asked = key
                obs = filter_legal(obs, ts.buttons)
                opts = list(obs.legal_actions)
                print("\n[remoto] sua vez: " + "  ".join(
                    f"[{i}] {action_label(obs, a)}" for i, a in enumerate(opts)) + "  [m] eu clico")
                choice = input("> ").strip().lower()
                if choice == "m" or not choice.isdigit() or int(choice) >= len(opts):
                    write({"manual": choice})
                    continue
                a = opts[int(choice)]
                write({"probe": page.evaluate(PROBE_JS), "dry_run": actor.dry_run(page, obs, a)})
                actor.enabled, actor.reason = True, ""
                ok = actor.do(page, obs, a)
                results.append({"action": action_label(obs, a), "ok": ok, "reason": actor.reason,
                                "sent": reader.sent[-1:] if reader.sent else None})
                write({"click": results[-1], "learned_mao11": actor.learned_mao11})
                print(f"[remoto] clique {'OK' if ok else 'FALHOU: ' + actor.reason}")
                if not ok:
                    print("[remoto] faça essa jogada você mesmo na mesa.")
        except KeyboardInterrupt:
            pass
        finally:
            ok_n = sum(r["ok"] for r in results)
            write({"summary": {"clicks": len(results), "ok": ok_n}})
            print(f"[remoto] cliques: {ok_n}/{len(results)} ok. Log em {log.name}")
            log.close()
