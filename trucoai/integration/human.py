"""Pulso humano no cursor: resposta, caminho e clique com o mesmo desenho
de ruído de um jogador na mesa.

Cada ação sai com: latência de reação amostrada log-normal com moduladores
por contexto (pedido de truco demora mais, mão de onze bem mais),8% de
hesitação, 5% de reflexo rápido; deslocamento por bezier com passos de
8–16 px espaçados 6–14 ms, havendo 30% de roça (drift) durante o pensar,
25% de ultrapassar o alvo e corrigir, e flop gaussiano dentro do retângulo
do botão (nunca o centro exato); aperto com mousedown→mouseup separados por
30–110 ms. A cadência acumula calma pela sessão (1.00→1.30, acorde por mão
0.85–1.25) para não cravar intervalos geminados.
"""

from __future__ import annotations

import math
import random
import time

_PRESS_MS = (30, 110)


def _ease(t: float) -> float:
    """suavização cônica — arranque lento, volta lenta, meio rápido."""
    return t * t * (3.0 - 2.0 * t)


class HumanCursor:
    def __init__(self):
        self.rng = random.SystemRandom()
        self.lx: float | None = None
        self.ly: float | None = None
        self.calm = 1.0  # multiplicador de latência, cresce devagar na sessão

    def new_hand(self) -> None:
        k = self.rng.uniform(0.85, 1.25)
        self.calm = max(0.85, min(1.30, self.calm * (0.85 + k) / 2.0))

    # ------------------------------------------------------------------
    def _react_ms(self, phase: str) -> int:
        base = {"card": 1.0, "virar": 0.9, "raise": 1.55, "accept": 1.25,
                "run": 1.15, "mao11": 2.1}.get(phase, 1.0)
        u1 = self.rng.random() or 1e-9
        u2 = self.rng.random()
        z = math.sqrt(-2.0 * math.log(u1)) * math.cos(2.0 * math.pi * u2)
        ms = 430.0 * math.exp(0.36 * z) * base
        if self.rng.random() < 0.05:            # reflexo: jogou sem pensar
            ms *= self.rng.uniform(0.35, 0.55)
        if self.rng.random() < 0.08:            # hesitação: voltou a olhar
            ms += self.rng.uniform(700.0, 2400.0)
        ms *= self.calm
        return int(max(180.0, min(4200.0, ms)))

    def _wander(self, page) -> None:
        if self.lx is None:
            return
        x, y = self.lx, self.ly
        for _ in range(self.rng.randint(1, 3)):
            x += self.rng.gauss(0.0, 22.0)
            y += self.rng.gauss(0.0, 16.0)
            page.mouse.move(max(2.0, x), max(2.0, y), steps=1)
            page.wait_for_timeout(self.rng.randint(14, 42))
        self.lx, self.ly = x, y

    def _path(self, page, tx: float, ty: float) -> None:
        if self.lx is None:
            # pequena flutuação perto do alvo antes de encostar
            self.lx = tx + self.rng.gauss(0.0, 40.0)
            self.ly = ty + self.rng.gauss(0.0, 40.0)
            page.mouse.move(self.lx, self.ly, steps=1)
        sx, sy = self.lx, self.ly
        dist = math.hypot(tx - sx, ty - sy)
        if dist < 3.0:
            return
        # bezier: pontos de controle deslocados do reto
        ang = math.atan2(ty - sy, tx - sx) + math.pi / 2.0
        bow = math.copysign(dist * self.rng.uniform(0.05, 0.20), self.rng.random() - 0.5)
        cx = sx + (tx - sx) * 0.42 + math.cos(ang) * bow
        cy = sy + (ty - sy) * 0.42 + math.sin(ang) * bow
        n = max(4, min(26, int(dist / 9.0)))
        for i in range(1, n + 1):
            t = i / n
            q = _ease(t)
            x = (1 - q) ** 2 * sx + 2 * (1 - q) * q * cx + q * q * tx
            y = (1 - q) ** 2 * sy + 2 * (1 - q) * q * cy + q * q * ty
            x += self.rng.gauss(0.0, 1.4)
            y += self.rng.gauss(0.0, 1.4)
            page.mouse.move(x, y, steps=1)
            page.wait_for_timeout(self.rng.randint(6, 14))
        page.mouse.move(tx, ty, steps=1)
        self.lx, self.ly = tx, ty

    def _press(self, page) -> None:
        page.mouse.down()
        page.wait_for_timeout(self.rng.randint(*_PRESS_MS))
        page.mouse.up()
        self.calm = max(1.0, self.calm + 0.006)  # a sessão cansa: intervalo cresce

    # ------------------------------------------------------------------
    def click(self, page, cx: float, cy: float, w: float, h: float, phase: str) -> None:
        page.wait_for_timeout(self._react_ms(phase))
        if self.rng.random() < 0.30:
            self._wander(page)
        inset = min(6.0, w * 0.18)
        tx = cx + self.rng.gauss(0.0, max(1.8, (w / 8.0)))
        ty = cy + self.rng.gauss(0.0, max(2.2, (h / 9.0)))
        tx = max(cx - w / 2.0 + inset, min(cx + w / 2.0 - inset, tx))
        ty = max(cy - h / 2.0 + inset, min(cy + h / 2.0 - inset, ty))
        if self.rng.random() < 0.25:  # passa do alvo e volta
            over = self.rng.uniform(12.0, 48.0)
            d = math.hypot(tx - (self.lx or tx), ty - (self.ly or ty))
            if d > over:
                ux = (tx - (self.lx or tx)) / d
                uy = (ty - (self.ly or ty)) / d
                self._path(page, tx + ux * over, ty + uy * over)
                page.wait_for_timeout(self.rng.randint(40, 140))
        self._path(page, tx, ty)
        self._press(page)

    def pause_ms(self, ms: int, spread: float = 0.5) -> None:
        """espera com folga proporcional — nada de intervalo cravado."""
        lo = ms * (1.0 - spread)
        hi = ms * (1.0 + spread)
        val = int(self.rng.uniform(lo, hi))
        if val > 0:
            time.sleep(val / 1000.0)
