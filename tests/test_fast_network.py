import random
import numpy as np
from pathlib import Path
from trucoai.agents.learned import PolicyNetwork
from trucoai.engine import new_match
from trucoai.observation import observe
from trucoai.rules import ONLINE_O1


def test_cached_network_is_bit_exact_on_legal_observations():
    from trucoai.agents.fast_network import FastNetwork
    root = Path(__file__).resolve().parents[1]
    original = PolicyNetwork(root/'runs/e9_iteration_s43_warm/policy.npz')
    fast = FastNetwork(original)
    checked = 0
    for seed in range(30):
        s = new_match(ONLINE_O1)
        s.deal(random.Random(seed))
        rng = random.Random(seed)
        while not s.is_terminal():
            o = observe(s,s.current)
            assert np.array_equal(original.logits(o),fast.logits(o))
            assert original.probs(o) == fast.probs(o) == fast.probs(o)
            s.apply(rng.choice(o.legal_actions))
            checked += 1
    assert checked > 100


def test_acceleration_preserves_seeded_portfolio_decisions():
    from trucoai.agents.fast_network import accelerate
    from scripts.portfolio_factory import factory
    root = Path(__file__).resolve().parents[1]
    spec='portfolio:'+(root/'runs/e9_frozen_20261020/portfolio.json').as_posix()
    a, b = factory(spec), accelerate(factory(spec))
    a.reset(11); b.reset(11)
    s = new_match(ONLINE_O1)
    s.deal(random.Random(11))
    while not s.is_terminal():
        o = observe(s,s.current)
        x,y = a.act(o), b.act(o)
        assert x == y
        s.apply(x)
