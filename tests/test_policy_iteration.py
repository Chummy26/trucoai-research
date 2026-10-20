import importlib.util
import numpy as np
from pathlib import Path


def test_match_collection_is_reproducible_and_keeps_player_memories_separate():
    path = Path(__file__).resolve().parents[1] / 'scripts/collect_iteration.py'
    assert path.exists(), 'full-match policy-iteration collector missing'
    from scripts.collect_iteration import collect_match
    from trucoai.agents.base import RandomAgent
    from trucoai.observation import Observation
    agents = []

    class Tracked(RandomAgent):
        def __init__(self):
            super().__init__()
            self.seats = set()

        def act(self, obs):
            assert isinstance(obs, Observation)
            self.seats.add(obs.me)
            return super().act(obs)

    def factory(spec):
        ag = Tracked()
        agents.append(ag)
        return ag

    args = ('random', 'random', 600001)
    a = collect_match(args, factory=factory)
    b = collect_match(args, factory=factory)
    assert len(agents) == 4 and all(len(ag.seats) == 1 for ag in agents)
    for key in ('x', 'y', 'mask', 'group'):
        assert np.array_equal(a[key], b[key])
    assert np.unique(a['group']).tolist() == [600001]
    assert len(a['y']) > 0 and a['mask'][np.arange(len(a['y'])), a['y']].all()
    assert a['receipt']['completed'] and a['receipt']['teacher_seat'] == 1


def test_asymmetric_search_uses_the_opponent_model_only_for_the_other_seat():
    from importlib.util import find_spec
    assert find_spec('trucoai.agents.iteration') is not None, 'asymmetric search missing'
    from trucoai.agents.iteration import AsymmetricSearchAgent
    from trucoai.agents.learned import LearnedSearchAgent
    from trucoai.engine import new_match
    from trucoai.observation import observe, sample_state
    from trucoai.rules import PLAYTRUCO_O1
    from dataclasses import replace
    import random
    root = Path(__file__).resolve().parents[1]
    bp = root/'runs/mccfr_rs_o1_s7/compact'
    own = root/'runs/e8_mixture_s20261021/policy.npz'
    opp = root/'runs/e8_imitation_s20261020/policy.npz'
    agent = AsymmetricSearchAgent(bp, own, opponent_network_path=opp, n=4,
                                 neural_bets=True, bets_from_blueprint=False)
    st = new_match(PLAYTRUCO_O1)
    st.deal(random.Random(1))
    obs = observe(st, st.current)
    agent.act(obs)
    assert agent._policy_probs(obs) == agent.network.probs(obs)
    st.apply(next(a for a in obs.legal_actions if a < 3))
    other = observe(st, st.current)
    assert agent._policy_probs(other) == agent.opponent_network.probs(other)
    # Likelihood reconstruction must establish perspective even without act().
    agent.reset(1)
    agent._likelihood(sample_state(obs, random.Random(2)), obs)
    assert agent._root_me == obs.me
    agent.reset(2)
    assert agent._root_me is None


def test_iteration_factory_loads_explicit_models_and_finishes_a_legal_match():
    path = Path(__file__).resolve().parents[1] / 'scripts/evaluate_iteration.py'
    assert path.exists(), 'iteration evaluation entrypoint missing'
    from scripts.evaluate_iteration import factory
    from trucoai.arena import play_match
    from trucoai.rules import PLAYTRUCO_O1
    root = Path(__file__).resolve().parents[1]
    own = root/'runs/e8_mixture_s20261021/policy.npz'
    opp = root/'runs/e8_imitation_s20261020/policy.npz'
    spec = f'asymmetric:{root}/runs/mccfr_rs_o1_s7/compact?net={own}&oppnet={opp}&n=2&k=2&temper=0.5'
    agent = factory(spec)
    assert agent.network.path == str(own.resolve())
    assert agent.opponent_network.path == str(opp.resolve())
    result = play_match(agent, factory('heuristic'), 130001, 0, rules=PLAYTRUCO_O1)
    assert max(result.a_points, result.b_points) >= 12
    assert result.sides['a'].illegal == 0


def test_validation_hash_split_keeps_matches_whole_without_fixing_a_seat():
    from scripts import train_imitation
    assert hasattr(train_imitation, 'validation_mask'), 'hash split missing'
    groups = np.repeat(np.arange(600000, 600512), 3)
    valid = train_imitation.validation_mask(groups, 'hash')
    assert np.array_equal(valid, train_imitation.validation_mask(groups, 'hash'))
    assert (valid.reshape(-1, 3).min(1) == valid.reshape(-1, 3).max(1)).all()
    assert set(groups[valid] % 2) == {0, 1}
    assert set(((groups[valid]-600000)//2) % 4) == {0, 1, 2, 3}
    assert not set(groups[valid]) & set(groups[~valid])
    assert np.array_equal(train_imitation.validation_mask(groups, 'modulo'), groups % 10 == 0)


def test_warm_start_preserves_exported_network_logits():
    from scripts import train_imitation
    assert hasattr(train_imitation, 'load_initial'), 'warm start missing'
    import torch
    from trucoai.agents.learned import PolicyNetwork
    path = Path(__file__).resolve().parents[1]/'runs/e8_mixture_s20261021/policy.npz'
    net = torch.nn.Sequential(torch.nn.Linear(224, 96), torch.nn.ReLU(),
                              torch.nn.Linear(96, 96), torch.nn.ReLU(), torch.nn.Linear(96, 9))
    train_imitation.load_initial(net, path)
    portable = PolicyNetwork(path)
    x = np.random.default_rng(4).normal(size=(10, 224)).astype(np.float32)
    expected = x.copy()
    for i, (w, b) in enumerate(portable.layers):
        expected = expected @ w + b
        if i+1 < len(portable.layers):
            expected = np.maximum(expected, 0)
    with torch.no_grad():
        assert np.max(np.abs(net(torch.tensor(x)).numpy()-expected)) < 1e-4
