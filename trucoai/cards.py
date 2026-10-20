"""Cartas como inteiros 0..39 (``rank_idx * 4 + naipe``).

Ordem de força sem manilha (``rank_idx``): 4 5 6 7 Q J K A 2 3.
Naipes (força entre manilhas): 0=Ouros < 1=Espadas < 2=Copas < 3=Paus.
"""

from __future__ import annotations

from typing import List

RANKS = ["4", "5", "6", "7", "Q", "J", "K", "A", "2", "3"]
SUITS = ["o", "e", "c", "p"]  # ouros, espadas, copas, paus
SUIT_SYMBOLS = ["♦", "♠", "♥", "♣"]
NUM_CARDS = 40
COVERED_STRENGTH = -1
# Forças possíveis: 0..9 (cartas comuns) e 10..13 (manilhas por naipe).
NUM_STRENGTHS = 14


def rank_of(card: int) -> int:
    return card >> 2


def suit_of(card: int) -> int:
    return card & 3


def make(rank: str, suit: str) -> int:
    return RANKS.index(rank) * 4 + SUITS.index(suit)


def parse(text: str) -> int:
    """``"4p"`` -> 4 de paus; aceita também símbolos de naipe."""
    text = text.strip()
    r, s = text[:-1].upper(), text[-1]
    if s in SUIT_SYMBOLS:
        s = SUITS[SUIT_SYMBOLS.index(s)]
    return make(r, s.lower())


def name(card: int) -> str:
    return RANKS[rank_of(card)] + SUIT_SYMBOLS[suit_of(card)]


def manilha_rank(vira: int) -> int:
    return (rank_of(vira) + 1) % 10


def strength(card: int, vira: int) -> int:
    r = rank_of(card)
    if r == manilha_rank(vira):
        return 10 + suit_of(card)
    return r


def full_deck() -> List[int]:
    return list(range(NUM_CARDS))
