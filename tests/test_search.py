"""Agente de busca com crença bayesiana (agents/search.py)."""

import random
from pathlib import Path

import pytest

from trucoai.agents import make_agent
from trucoai.agents.search import _hand_start
from trucoai.arena import play_match
from trucoai.engine import new_match
from trucoai.observation import observe, sample_state
from trucoai.rules import PLAYTRUCO_O1

COMPACT = Path(__file__).resolve().parent.parent / "runs" / "mccfr_rs_v1_s1" / "compact"
pytestmark = pytest.mark.skipif(not COMPACT.exists(), reason="blueprint compacto ausente")


def random_mid_hand_states(n=150, seed=4):
    rng = random.Random(seed)
    out = []
    while len(out) < n:
        st = new_match(PLAYTRUCO_O1)
        st.scores = [rng.randrange(12), rng.randrange(12)]
        st.deal(rng)
        for _ in range(rng.randrange(1, 7)):
            if st.is_terminal():
                break
            st.apply(rng.choice(st.legal_actions()))
        if not st.is_terminal():
            out.append(st)
    return out


def test_reconstrucao_do_inicio_da_mao_reproduz_o_log():
    """Do início da mão, reaplicar o log público num mundo sorteado leva ao mesmo log."""
    from trucoai.engine import ACCEPT, COVER_0, PLAY_0, RAISE, RUN
    agent = make_agent(f"search:{COMPACT}")
    rng = random.Random(0)
    for st in random_mid_hand_states():
        obs = observe(st, st.current)
        w = sample_state(obs, rng)
        s, plays = _hand_start(w, obs)
        pi = 0
        for t, p, v in obs.public_log[1:]:
            if t in ("round", "hand_end"):
                continue
            if t == "play":
                pl, card, cov = plays[pi]
                pi += 1
                a = (COVER_0 if cov else PLAY_0) + s.hands[pl].index(card)
            else:
                a = {"raise": RAISE, "accept": ACCEPT, "run": RUN, "fold": RUN}.get(t, v)
            s.apply(a)
        assert s.public_log[1:] == list(obs.public_log[1:])
        assert 0 < agent._likelihood(w, obs) <= 1


def test_crenca_favorece_manilha_de_quem_pediu_truco():
    """Depois de um truco do adversário, a crença dá mais peso a mãos com manilha."""
    from trucoai import cards as C
    from trucoai.engine import RAISE
    agent = make_agent(f"search:{COMPACT}?n=400&k=4")
    agent.reset(1)
    rng = random.Random(3)
    more = 0
    trials = 0
    while trials < 25:
        st = new_match(PLAYTRUCO_O1)
        st.deal(rng)
        if st.current != 0 or RAISE not in st.legal_actions():
            continue
        st.apply(RAISE)  # o mão (jogador 0) pede truco logo de cara; IA é o 1
        obs = observe(st, 1)
        man = lambda w: sum(C.strength(c, w.vira) >= 10 for c in w.hands[0] if c is not None)  # noqa: E731
        agent.belief = False
        uni = sum(man(w) for w in agent.worlds(obs)) / agent.n
        agent.belief = True
        bel = sum(man(w) for w in agent.worlds(obs)) / agent.n
        more += bel > uni
        trials += 1
    assert more >= 18  # em pelo menos ~70% dos casos


def test_agente_joga_partidas_sem_acoes_ilegais():
    a = make_agent(f"search:{COMPACT}?n=8&k=2&adaptive=1")
    b = make_agent("heuristic")
    for seed in range(3):
        r = play_match(a, b, seed=seed, a_seat=seed % 2, rules=PLAYTRUCO_O1)
        assert r.sides["a"].illegal == 0


def test_zap_aponta_para_o_vencedor_do_e5():
    a = make_agent("zap")
    assert a.name == "Zap v1" and a.meta["name"] == "Zap v1"
    assert "mccfr_rs_o1_s7" in a.meta["spec"] and a.meta["rules_version"] == "paulista-1v1-o1"
