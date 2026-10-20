from importlib.util import find_spec
import random
import pytest


def test_portfolio_uses_public_likelihoods_and_resets_all_policy_memory():
    assert find_spec('trucoai.agents.portfolio') is not None, 'public-style portfolio missing'
    from trucoai.agents.portfolio import PortfolioAgent
    from trucoai.agents.base import Agent
    from trucoai.engine import new_match
    from trucoai.observation import observe, Observation
    from trucoai.rules import PLAYTRUCO_O1

    class Policy(Agent):
        def __init__(self, last=False):
            super().__init__()
            self.last = last
            self.calls = 0

        def reset(self, seed=None):
            super().reset(seed)
            self.calls = 0

        def act(self, obs):
            assert isinstance(obs, Observation)
            self.calls += 1
            return obs.legal_actions[-1 if self.last else 0]

    class Model:
        def __init__(self, likelihood):
            self.likelihood = likelihood

        def _likelihood(self, hypothetical_world, obs):
            assert isinstance(obs, Observation)
            return self.likelihood

    policies = [Policy(), Policy(True)]
    fallback = Policy()
    portfolio = PortfolioAgent([Model(0.01), Model(0.99)], policies, fallback,
                               style_n=4, threshold=0.7)
    state = new_match(PLAYTRUCO_O1)
    state.deal(random.Random(42))
    start = observe(state, state.current)
    assert portfolio.act(start) == start.legal_actions[0]
    assert fallback.calls == 1
    state.apply(0)
    obs = observe(state, state.current)
    assert portfolio.act(obs) == obs.legal_actions[-1]
    assert policies[1].calls == 1 and policies[0].calls == 0
    portfolio.reset(55)
    assert portfolio.belief.posterior == pytest.approx([.5, .5])
    assert all(p.calls == 0 for p in policies+[fallback])
    assert portfolio.selection_counts == [0, 0, 0]


def test_portfolio_rejects_misaligned_models_or_invalid_budgets():
    assert find_spec('trucoai.agents.portfolio') is not None, 'public-style portfolio missing'
    from trucoai.agents.portfolio import PortfolioAgent
    with pytest.raises(ValueError):
        PortfolioAgent([object()], [], object())
    with pytest.raises(ValueError):
        PortfolioAgent([object()], [object()], object(), style_n=0)


def test_portfolio_file_loads_exact_policies_and_completes_legal_match(tmp_path):
    assert find_spec('scripts.portfolio_factory') is not None, 'portfolio loader missing'
    import json
    from pathlib import Path
    from scripts.portfolio_factory import factory
    from trucoai.arena import play_match
    from trucoai.rules import PLAYTRUCO_O1
    root = Path(__file__).resolve().parents[1]
    net = root/'runs/e9_iteration_s43_warm/policy.npz'
    spec = f'learned:{root}/runs/mccfr_rs_o1_s7/compact?net={net}&nbets=1&n=2&k=2&bets=0'
    path = tmp_path/'portfolio.json'
    path.write_text(json.dumps({'models':[spec], 'policies':[spec], 'fallback':'heuristic',
                                'style_n':2, 'threshold':0.5}))
    ag = factory(f'portfolio:{path}')
    assert ag.models[0].network.path == str(net.resolve())
    result = play_match(ag, factory('heuristic'), 132001, 0, rules=PLAYTRUCO_O1)
    assert max(result.a_points, result.b_points) >= 12
    assert result.sides['a'].illegal == 0
    assert ag.meta['portfolio_file'] == str(path.resolve())
    # Actual hidden cards may change while the legal observation stays fixed.
    from trucoai.engine import new_match
    from trucoai.observation import observe
    state = new_match(PLAYTRUCO_O1)
    state.deal(random.Random(8))
    obs = observe(state, state.current)
    alternate = state.clone()
    alternate.hands[1-state.current] = list(reversed(obs.unknown_pool()[-3:]))
    other_obs = observe(alternate, state.current)
    assert other_obs == obs
    twin = factory(f'portfolio:{path}')
    ag.reset(88)
    twin.reset(88)
    assert ag.act(obs) == twin.act(other_obs)
    assert ag.belief.posterior == pytest.approx(twin.belief.posterior)
