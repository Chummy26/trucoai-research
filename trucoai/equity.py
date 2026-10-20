"""Tabela de equidade da partida: P(vencer a partida | placar).

No 1v1 até 12, o objetivo é vencer a PARTIDA, não maximizar pontos por mão.
Um mesmo truco vale muito mais com 10x10 do que com 0x0. Os repositórios de
referência otimizam pontos (CFR) ou só ±1 no fim (PPO, sinal esparso); aqui a
utilidade de cada mão é a variação de equidade, o que dá sinal denso E
alinhado ao objetivo real.

Modelo (HIPÓTESE, documentado em docs/ARQUITETURA.md): cada mão é vencida por
qualquer lado com probabilidade 1/2, com distribuição de pontos ``dist``
estimada empiricamente (``scripts/estimate_equity.py``). Ignora o efeito de
quem é mão e as decisões especiais da mão de onze.
"""

from __future__ import annotations

import json
from functools import lru_cache
from pathlib import Path
from typing import Dict, List, Optional

TARGET = 12
# Distribuição padrão (pontos por mão, incluindo empate=0). Substituível pela
# estimativa empírica gravada em data/equity_dist.json.
DEFAULT_DIST: Dict[int, float] = {0: 0.02, 1: 0.55, 3: 0.33, 6: 0.07, 9: 0.02, 12: 0.01}


class EquityTable:
    def __init__(self, dist: Optional[Dict[int, float]] = None, target: int = TARGET):
        self.dist = {int(k): float(v) for k, v in (dist or DEFAULT_DIST).items()}
        tot = sum(self.dist.values())
        self.dist = {k: v / tot for k, v in self.dist.items()}
        self.target = target
        self.table = self._build()

    def _build(self) -> List[List[float]]:
        t = self.target
        W = [[0.0] * (t + 1) for _ in range(t + 1)]
        p_draw = self.dist.get(0, 0.0)
        # W[a][b]: equidade de quem tem a pontos contra b. Processa do fim.
        for total in range(2 * t - 2, -1, -1):
            for a in range(t):
                b = total - a
                if not (0 <= b < t):
                    continue
                v = 0.0
                for k, pk in self.dist.items():
                    if k == 0:
                        continue
                    win = 1.0 if a + k >= t else W[a + k][b]
                    lose = 0.0 if b + k >= t else W[a][b + k]
                    v += pk * 0.5 * (win + lose)
                W[a][b] = v / (1.0 - p_draw) if p_draw < 1 else 0.5
        return W

    def __call__(self, me: int, opp: int) -> float:
        if me >= self.target:
            return 1.0
        if opp >= self.target:
            return 0.0
        return self.table[me][opp]

    def save(self, path: Path) -> None:
        Path(path).write_text(json.dumps({"dist": self.dist, "target": self.target}, indent=2))

    @classmethod
    def load(cls, path: Path) -> "EquityTable":
        d = json.loads(Path(path).read_text())
        return cls(d["dist"], d["target"])


@lru_cache(maxsize=1)
def default_table() -> EquityTable:
    p = Path(__file__).resolve().parent.parent / "data" / "equity_dist.json"
    if p.exists():
        return EquityTable.load(p)
    return EquityTable()
