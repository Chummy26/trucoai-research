"""Fronteira de informação: a observação não pode depender do que é oculto."""

import random

from trucoai.cfr.abstraction import info_key
from trucoai.engine import new_match
from trucoai.observation import observe, sample_state


def random_states(n=400, seed=3):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        st = new_match()
        st.scores = [rng.randrange(12), rng.randrange(12)]
        st.deal(rng)
        for _ in range(rng.randrange(7)):
            if st.is_terminal():
                break
            st.apply(rng.choice(st.legal_actions()))
        if not st.is_terminal():
            out.append(st)
    return out


def scramble_hidden(st, viewer, rng):
    """Troca toda informação oculta ao ``viewer`` mantendo o que é público."""
    s = st.clone()
    opp = 1 - viewer
    known = {s.vira} | {c for c in s.hands[viewer] if c is not None}
    for rnd in s.rounds:
        for p, c, cov in rnd:
            if not cov or p == viewer:
                known.add(c)
    pool = [c for c in range(40) if c not in known]
    rng.shuffle(pool)
    s.hands[opp] = [pool.pop() if c is not None else None for c in s.hands[opp]]
    s.rounds = [[(p, pool.pop() if (cov and p == opp) else c, cov) for p, c, cov in rnd]
                for rnd in s.rounds]
    return s


def test_observacao_invariante_a_informacao_oculta():
    rng = random.Random(0)
    for st in random_states():
        for viewer in (0, 1):
            o1 = observe(st, viewer)
            o2 = observe(scramble_hidden(st, viewer, rng), viewer)
            assert o1 == o2
            assert info_key(o1) == info_key(o2)


def test_observacao_nao_contem_cartas_do_adversario():
    for st in random_states(100):
        for viewer in (0, 1):
            o = observe(st, viewer)
            opp_hidden = {c for c in st.hands[1 - viewer] if c is not None}
            opp_hidden |= {c for rnd in st.rounds for p, c, cov in rnd if cov and p != viewer}
            seen = set(o.known_cards()) | {v for t, _, v in o.public_log
                                           if t in ("play", "deal")}
            assert not (opp_hidden & seen)


def test_mao_de_ferro_as_cegas_esconde_as_proprias_cartas():
    rng = random.Random(1)
    st = new_match()
    st.scores = [11, 11]
    st.deal(rng)
    o = observe(st, 0)
    assert o.blind and all(c is None for c in o.my_hand)


def test_sample_state_consistente_com_a_observacao():
    rng = random.Random(5)
    for st in random_states(200):
        viewer = st.current
        o = observe(st, viewer)
        s = sample_state(o, rng)
        assert observe(s, viewer) == o
        cards = [c for h in s.hands for c in h if c is not None]
        cards += [c for rnd in s.rounds for _, c, _ in rnd] + [s.vira]
        assert len(cards) == len(set(cards))
        assert sorted(s.legal_actions()) == sorted(o.legal_actions)
