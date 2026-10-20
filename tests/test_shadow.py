"""O jogo-sombra, alimentado só com eventos visíveis à IA, reproduz a observação real."""

import dataclasses
import random

from trucoai.agents import make_agent
from trucoai.engine import ACCEPT, RAISE, RUN, new_match
from trucoai.integration.shadow import (ME, OPP, Accept, HandEnd, HandStart, Play, Raise, Run,
                                        ShadowGame)
from trucoai.observation import observe


def strip(o):
    return dataclasses.replace(o, hand_no=0)


def run_games(n_matches=30, seed=0):
    rng = random.Random(seed)
    agents = [make_agent("heuristic"), make_agent("random")]
    checked = 0
    for m in range(n_matches):
        for a in agents:
            a.reset(m)
        st = new_match()
        sh = ShadowGame()
        while st.match_winner is None:
            st.deal(rng)
            sh.feed(HandStart(list(st.hands[ME]), st.vira, st.scores[ME], st.scores[OPP],
                              i_am_mao=st.mao == ME))
            while not st.is_terminal():
                p = st.current
                if p == ME:
                    assert strip(sh.observation()) == strip(observe(st, ME))
                    checked += 1
                else:
                    assert sh.observation() is None
                a = agents[p].act(observe(st, p))
                if a < 6:
                    card = st.hands[p][a % 3]
                    covered = a >= 3
                    ev = Play(p, None if (covered and p == OPP) else card, covered)
                else:
                    ev = {RAISE: Raise, ACCEPT: Accept, RUN: Run}[a](p)
                st.apply(a)
                sh.feed(ev)
            sh.feed(HandEnd(st.scores[ME], st.scores[OPP]))
            assert not sh.divergences, sh.divergences
    return checked


def test_sombra_reproduz_observacao():
    assert run_games() > 300


def test_divergencia_quando_mesa_viola_regras():
    sh = ShadowGame()
    sh.feed(HandStart([0, 1, 2], 39, 0, 0, i_am_mao=True))
    sh.feed(Play(OPP, 10))  # adversário jogando fora da vez
    assert sh.divergences and not sh.synced and sh.observation() is None


def test_divergencia_de_placar():
    sh = ShadowGame()
    sh.feed(HandStart([0, 1, 2], 39, 0, 0, i_am_mao=True))
    sh.feed(Raise(ME))
    sh.feed(Run(OPP))
    sh.feed(HandEnd(3, 0))  # nossas regras dão 1 ponto
    assert sh.divergences


def test_mao_de_ferro_as_cegas_na_sombra():
    """Mão de ferro: a IA não vê as próprias cartas; a mesa revela cada carta jogada."""
    rng = random.Random(11)
    checked = 0
    for m in range(200):
        st = new_match()
        st.scores = [11, 11]
        st.deal(rng)
        sh = ShadowGame()
        sh.feed(HandStart(None, st.vira, 11, 11, i_am_mao=st.mao == ME))
        while not st.is_terminal():
            p = st.current
            if p == ME:
                o = sh.observation()
                assert o is not None and o.blind and all(c is None for c in o.my_hand)
                assert strip(o) == strip(observe(st, ME))
                checked += 1
            a = rng.choice(st.legal_actions())
            card = st.hands[p][a % 3]
            st.apply(a)
            sh.feed(Play(p, card, slot=a % 3 if p == ME else None))
        sh.feed(HandEnd(st.scores[ME], st.scores[OPP]))
        assert not sh.divergences, sh.divergences
    assert checked > 200
