"""Trava de mesa: libera mesas grátis e, opcionalmente, mesas com entrada paga.

Falha fechada: qualquer dúvida (valor ausente, ilegível ou desatualizado)
bloqueia. A trava é consultada antes de CADA ação, não só ao sentar.

Mesas valendo dinheiro: defina ``TRUCOAI_MAX_STAKE`` (reais, ex.: ``2.00``)
ou use ``--max-stake`` no CLI. Sem o limite definido a IA não joga mesa paga
(é assim que ``online.py`` monta a guarda por padrão).
"""

from __future__ import annotations

import os
import re
import time
from dataclasses import dataclass
from typing import Optional

_BRL = re.compile(r"R\$\s*([0-9]{1,3}(?:\.[0-9]{3})*(?:,[0-9]{1,2})?|[0-9]+(?:[.,][0-9]{1,2})?)")
_FREE_WORDS = ("grátis", "gratis", "gratuita", "gratuito", "free")


def max_stake_from_env() -> Optional[float]:
    """Limite de entrada pago vindo de ``TRUCOAI_MAX_STAKE`` (reais). ``None`` = não configurado."""
    raw = os.environ.get("TRUCOAI_MAX_STAKE")
    if raw is None or not raw.strip():
        return None
    value = parse_brl(raw.replace("R$", ""))
    return value


def parse_brl(text: Optional[str]) -> Optional[float]:
    """Extrai um valor em reais de um texto. ``None`` se não houver valor claro."""
    if not text:
        return None
    low = text.lower()
    values = []
    for m in _BRL.finditer(text):
        raw = m.group(1)
        if "," in raw:
            raw = raw.replace(".", "").replace(",", ".")
        values.append(float(raw))
    if values:
        # Mais de um valor diferente (ex.: entrada e taxa) -> usa o maior: conservador.
        return max(values)
    if any(w in low for w in _FREE_WORDS):
        return 0.0
    return None


@dataclass
class GuardDecision:
    allowed: bool
    reason: str
    value: Optional[float] = None


class StakeGuard:
    def __init__(self, max_age_s: float = 30.0, max_stake: Optional[float] = None):
        self.max_age_s = max_age_s
        # ``None`` = só mesas grátis. Valor >= 0 = libera também mesa paga até o teto.
        self.max_stake = max_stake
        self._value: Optional[float] = None
        self._source: str = ""
        self._at: float = 0.0
        self.blocked_count = 0

    def update(self, value: Optional[float], source: str) -> None:
        """Registra o valor da mesa lido da interface (``None`` = ilegível)."""
        self._value, self._source, self._at = value, source, time.monotonic()

    def check(self) -> GuardDecision:
        d = self._decide()
        if not d.allowed:
            self.blocked_count += 1
        return d

    def _decide(self) -> GuardDecision:
        if self._at == 0.0:
            return GuardDecision(False, "valor da mesa ainda não lido")
        if time.monotonic() - self._at > self.max_age_s:
            return GuardDecision(False, "leitura do valor da mesa desatualizada", self._value)
        if self._value is None:
            return GuardDecision(False, f"valor da mesa ilegível ({self._source})")
        if self._value == 0.0:
            return GuardDecision(True, f"mesa grátis ({self._source})", 0.0)
        if self.max_stake is None:
            return GuardDecision(False, f"mesa com valor R$ {self._value:.2f}: limite de mesa paga não configurado",
                                 self._value)
        if self._value > self.max_stake:
            return GuardDecision(False,
                                 f"mesa com valor R$ {self._value:.2f} acima do teto R$ {self.max_stake:.2f}",
                                 self._value)
        return GuardDecision(True, f"mesa paga R$ {self._value:.2f} (teto R$ {self.max_stake:.2f}; {self._source})",
                             self._value)
