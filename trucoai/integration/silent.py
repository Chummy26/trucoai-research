"""Decisões silenciosas da IA durante partidas jogadas pelo humano em mesa paga.

No modo ``shadow`` em mesa com valor, a IA não pode jogar nem ajudar. Para
validar o caminho de decisão ao vivo sem virar "copiloto", ela decide em
segundo plano e as decisões ficam SÓ na memória: nada é impresso, e elas vão
para o log apenas quando a partida termina (``flush``).

Para cada decisão registra-se também a jogada que o humano fez de fato, o que
permite medir concordância IA x humano (dado de pesquisa, não de desempenho).
"""

from __future__ import annotations

from typing import List, Optional

from ..engine import ACCEPT, COVER_0, PLAY_0, RAISE, RUN
from ..observation import Observation
from .shadow import ME, Accept, HandEnd, Play, Raise, Run


class SilentDecisions:
    def __init__(self) -> None:
        self._buffer: List[dict] = []
        self._pending: Optional[dict] = None

    def decide(self, obs: Observation, action: int, ms: float,
               dry_run: Optional[dict] = None) -> None:
        self._pending = {"phase": obs.phase, "legal": list(obs.legal_actions),
                         "ai": action, "ms": ms, "my_hand": list(obs.my_hand)}
        if dry_run is not None:
            self._pending["dry_run"] = dry_run

    def human_action(self, ev) -> Optional[int]:
        p = self._pending
        if isinstance(ev, Play) and ev.who == ME:
            if ev.card is not None and ev.card in p["my_hand"]:
                slot = p["my_hand"].index(ev.card)
            elif ev.slot is not None:
                slot = ev.slot
            else:
                return None
            return (COVER_0 if ev.covered else PLAY_0) + slot
        if isinstance(ev, (Raise, Accept, Run)) and ev.who == ME:
            return {Raise: RAISE, Accept: ACCEPT, Run: RUN}[type(ev)]
        if isinstance(ev, HandEnd) and ev.gave_up:
            return RUN  # correu na própria vez
        return None

    def observe_event(self, ev) -> None:
        if self._pending is None:
            return
        human = self.human_action(ev)
        if human is None and not isinstance(ev, HandEnd):
            return
        rec = dict(self._pending)
        rec.pop("my_hand")
        rec["human"] = human
        rec["agree"] = human == rec["ai"] if human is not None else None
        self._buffer.append(rec)
        self._pending = None

    def flush(self) -> dict:
        recs, self._buffer, self._pending = self._buffer, [], None
        judged = [r for r in recs if r["agree"] is not None]
        dry = [r["dry_run"]["ok"] for r in recs if r.get("dry_run", {}).get("ok") is not None]
        return {"silent_decisions": recs, "n": len(recs),
                "dry_run_ok": (sum(dry) / len(dry)) if dry else None,
                "agreement": (sum(r["agree"] for r in judged) / len(judged)) if judged else None}
