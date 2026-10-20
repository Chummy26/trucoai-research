import random
import pytest
from trucoai.engine import new_match
from trucoai.observation import observe
from trucoai.rules import PLAYTRUCO_O1


def test_tail_value_uses_expected_opponent_and_best_own_terminal_action():
    from trucoai.agents.refined import RefinedSearchAgent
    agent = object.__new__(RefinedSearchAgent)
    agent.own_greedy = False
    agent._policy_probs = lambda o: {a: 1 / len(o.legal_actions) for a in o.legal_actions}
    from trucoai.equity import default_table
    found = 0
    for seed in range(100):
        s = new_match(PLAYTRUCO_O1)
        s.deal(random.Random(seed))
        s.stake = 12
        while not s.is_terminal():
            children = []
            for a in s.legal_actions():
                child = s.clone()
                child.apply(a)
                children.append(child)
            if children and all(c.is_terminal() for c in children):
                me = s.current
                values = [default_table()(c.scores[me], c.scores[1-me]) for c in children]
                assert agent._tail(s, me) == pytest.approx(sum(values)/len(values))
                agent.own_greedy = True
                assert agent._tail(s, me) == pytest.approx(max(values))
                agent.own_greedy = False
                found += 1
            s.apply(s.legal_actions()[0])
    assert found > 10


def test_refined_hidden_state_invariance_and_legal_actions():
    from scripts.e10_factory import factory
    import json
    from pathlib import Path
    root = Path(__file__).resolve().parents[1]
    cfg = json.loads((root/'runs/e9_frozen_20261020/portfolio.json').read_text())
    spec = cfg['policies'][-1].replace('learned:', 'refined:', 1) + '&tail=1&own_greedy=1'
    a, b = factory(spec), factory(spec)
    s = new_match(PLAYTRUCO_O1)
    s.deal(random.Random(121))
    hidden = s.clone()
    hidden.hands[1-s.current] = [35, 36, 37]
    oa, ob = observe(s, s.current), observe(hidden, s.current)
    assert oa == ob
    a.reset(91)
    b.reset(91)
    assert a.act(oa) == b.act(ob)
    assert a.act(oa) in oa.legal_actions
