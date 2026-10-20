import importlib.util
import random
import numpy as np
import pytest
from trucoai.engine import new_match
from trucoai.observation import observe
from trucoai.rules import ONLINE_O1


def agent_type():
    assert importlib.util.find_spec("trucoai.agents.payoff_portfolio"), "payoff-aware selector missing"
    from trucoai.agents.payoff_portfolio import PayoffPortfolioAgent
    return PayoffPortfolioAgent


class Model:
    def __init__(self, likelihood):
        self.likelihood = likelihood

    def _likelihood(self, world, obs):
        return self.likelihood


class Policy:
    def __init__(self, slot):
        self.slot = slot
        self.resets = 0

    def reset(self, seed=None):
        self.resets += 1

    def act(self, obs):
        return obs.legal_actions[self.slot]


def public_obs():
    s = new_match(ONLINE_O1)
    s.deal(random.Random(42))
    s.apply(s.legal_actions()[0])
    return s, observe(s, s.current)


def test_selector_uses_expected_payoff_not_largest_style_probability():
    cls = agent_type()
    # The most likely style is 0, but response 1 is better under uncertainty.
    a = cls([Model(.6), Model(.4)], [Policy(0), Policy(1)],
            [[.8, .1], [.7, .7]], style_n=2, forgetting=0)
    _, o = public_obs()
    assert a.act(o) == o.legal_actions[1]
    assert np.allclose(a.belief.posterior, [.6, .4])
    assert np.allclose(a.last_scores, [.52, .7])
    # Repeating a partial history must replace, not square, its evidence.
    a.act(o)
    assert np.allclose(a.belief.posterior, [.6, .4])


def test_reset_and_private_cards_cannot_change_policy_selection():
    cls = agent_type()
    policies = [Policy(0), Policy(1)]
    a = cls([Model(.8), Model(.2)], policies, [[.9, .1], [.1, .9]], style_n=2)
    s, o = public_obs()
    a.reset(123)
    action = a.act(o)
    replacements = iter(o.unknown_pool())
    s.hands[1-s.current] = [
        next(replacements) if c is not None else None for c in s.hands[1-s.current]
    ]
    assert observe(s, s.current) == o
    a.reset(123)
    assert a.belief.hand_no is None
    assert np.allclose(a.belief.posterior, [.5, .5])
    assert a.act(observe(s, s.current)) == action
    assert all(p.resets == 3 for p in policies)


def test_invalid_payoff_cannot_silently_select_a_policy():
    cls = agent_type()
    for payoff in ([[.5]], [[float("nan"), .5]], [[1.1, .5]]):
        with pytest.raises(ValueError):
            cls([Model(1), Model(1)], [Policy(0)], payoff)
