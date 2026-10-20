"""Abstração de conjuntos de informação para o MCCFR.

Inspirada nas abstrações A1/A2/A3 de filevich/truco-mccfr-ai, mas adaptada ao
Paulista: a vira define a manilha e os naipes ordenam as manilhas, então a
abstração natural é a *força relativa*, aqui agrupada em 8 faixas (4-7, Q-K, A, 2, 3,
manilhas fracas, copas, zap). Perde-se a distinção entre cartas da mesma faixa e a
identidade exata usada em contagem de cartas (manilhas por naipe são mantidas
em parte). A busca PIMC usa as cartas exatas e compensa essa perda.

Diferenças em relação à referência: o placar entra na chave (faixa "perto do fim"), porque
a decisão de aceitar/aumentar depende fortemente dele no 1v1 até 12.

A chave só usa campos de :class:`Observation` — não há caminho para informação
privada do adversário.
"""

from __future__ import annotations

from .. import cards as C
from ..observation import Observation


def score_bucket(s: int) -> int:
    # 0-8 / 9-10 (perto do fim). 11 é tratado por ``special`` (onze/ferro).
    return 0 if s < 9 else 1


# Força 0..13 -> faixa 0..7: {4,5,6,7} {Q,J,K} A 2 3 {manilha ♦/♠} ♥ ♣
_BUCKET = [0, 0, 0, 0, 1, 1, 1, 2, 3, 4, 5, 5, 6, 7]


def card_bucket(card: int, vira: int) -> int:
    return _BUCKET[C.strength(card, vira)]


def info_key(obs: Observation) -> str:
    me, vira = obs.me, obs.vira
    if obs.blind:
        hand = "B" + "".join("1" if f else "0" for f in obs.my_slots_filled)
    else:
        hand = "".join(str(x) for x in sorted(
            card_bucket(c, vira) for c in obs.my_hand if c is not None))
    hist = []
    for t, p, v in obs.public_log:
        rel = "m" if p == me else "o"
        if t == "play":
            hist.append(f"{rel}{'x' if v < 0 else card_bucket(v, vira)}")
        elif t == "raise":
            hist.append(f"{rel}R{v}")
        elif t == "accept":
            hist.append(f"{rel}A")
        elif t == "mao11":
            hist.append(f"{rel}M{v}")
        elif t == "round":
            hist.append("|" if p < 0 else ("|w" if p == me else "|l"))
    return (f"{obs.special[0]}{score_bucket(obs.scores[me])}{score_bucket(obs.scores[1 - me])}"
            f":{hand}:{''.join(hist)}:{obs.phase[0]}")


# ---------------------------------------------------------------------------
# Abstrações versionadas (E7). "v1" = ``info_key`` acima (Zap v1, não muda).
#   v2a: força EXATA das cartas (14 classes, letras a..n) na mão e no histórico;
#        placar em 6 faixas: 0-3, 4-6, 7-8, 9, 10, 11.
#   v2b: força exata só na própria mão; histórico em faixas (como v1); placar em 6 faixas.
# Implementadas igual em rust/mccfr (``--abstraction``) e no kernel GPU (``absv``).
# ---------------------------------------------------------------------------
ABSTRACTIONS = ("v1", "v2a", "v2b")
_LETTERS = "abcdefghijklmn"


def score_bucket_v2(s: int) -> int:
    return 0 if s < 4 else 1 if s < 7 else 2 if s < 9 else s - 6


def info_key_v(obs: Observation, version: str = "v1") -> str:
    if version == "v1":
        return info_key(obs)
    if version not in ABSTRACTIONS:
        raise ValueError(f"abstração desconhecida: {version}")
    exact_hist = version == "v2a"
    me, vira = obs.me, obs.vira
    if obs.blind:
        hand = "B" + "".join("1" if f else "0" for f in obs.my_slots_filled)
    else:
        hand = "".join(_LETTERS[x] for x in sorted(
            C.strength(c, vira) for c in obs.my_hand if c is not None))
    hist = []
    for t, p, v in obs.public_log:
        rel = "m" if p == me else "o"
        if t == "play":
            if v < 0:
                hist.append(f"{rel}x")
            else:
                hist.append(rel + (_LETTERS[C.strength(v, vira)] if exact_hist else str(card_bucket(v, vira))))
        elif t == "raise":
            hist.append(f"{rel}R{v}")
        elif t == "accept":
            hist.append(f"{rel}A")
        elif t == "mao11":
            hist.append(f"{rel}M{v}")
        elif t == "round":
            hist.append("|" if p < 0 else ("|w" if p == me else "|l"))
    return (f"{obs.special[0]}{score_bucket_v2(obs.scores[me])}{score_bucket_v2(obs.scores[1 - me])}"
            f":{hand}:{''.join(hist)}:{obs.phase[0]}")


def key_fn(version: str = "v1"):
    """Função ``obs -> chave`` da versão de abstração usada por um blueprint."""
    if version == "v1":
        return info_key
    return lambda obs: info_key_v(obs, version)
