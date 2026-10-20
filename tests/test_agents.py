import random

from trucoai.agents import make_agent
from trucoai.arena import play_match, run_matchup
from trucoai.cfr.mccfr import MCCFRTrainer, abstract_legal, to_abstract, to_concrete
from trucoai.engine import new_match
from trucoai.equity import EquityTable
from trucoai.observation import observe


def test_equidade_simetrica_e_monotona():
    eq = EquityTable()
    for a in range(12):
        for b in range(12):
            assert abs(eq(a, b) + eq(b, a) - 1) < 1e-9
            if a < 11:
                assert eq(a + 1, b) >= eq(a, b) - 1e-12
    assert abs(eq(0, 0) - 0.5) < 1e-9


def test_mapeamento_de_acoes_abstratas_e_bijetivo():
    rng = random.Random(0)
    for _ in range(200):
        st = new_match()
        st.deal(rng)
        for _ in range(rng.randrange(5)):
            if st.is_terminal():
                break
            st.apply(rng.choice(st.legal_actions()))
        if st.is_terminal():
            continue
        o = observe(st, st.current)
        for a in o.legal_actions:
            assert to_concrete(o, to_abstract(o, a)) == a
        assert len(abstract_legal(o)) == len(o.legal_actions)


def test_arena_deterministica_e_sem_acoes_ilegais():
    r1 = run_matchup(make_agent, "heuristic", "random", deals=5)
    r2 = run_matchup(make_agent, "heuristic", "random", deals=5)
    assert r1 == r2 or r1["a_wins"] == r2["a_wins"]
    assert r1["a_metrics"]["illegal_rate"] == 0.0


def test_mccfr_treina_salva_e_agente_joga(tmp_path):
    tr = MCCFRTrainer(seed=1)
    tr.train(30)
    assert tr.nodes
    path = tmp_path / "bp.pkl"
    tr.save(path)
    for spec in (f"blueprint:{path}", f"hybrid:{path}"):
        a, b = make_agent(spec), make_agent("random")
        res = play_match(a, b, seed=0, a_seat=0)
        assert res.sides["a"].illegal == 0
