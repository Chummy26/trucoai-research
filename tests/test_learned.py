"""Information boundary and portable inference for the E8 learned policy."""
import importlib.util
import random

import numpy as np
import pytest

from trucoai.engine import new_match
from trucoai.observation import observe, sample_state
from trucoai.rules import ONLINE_O1, DEFAULT_RULES


def api():
    assert importlib.util.find_spec('trucoai.agents.learned') is not None, 'learned policy missing'
    from trucoai.agents import learned
    return learned


def test_features_only_depend_on_legal_observation():
    m = api()
    rng = random.Random(132)
    for _ in range(100):
        s = new_match(ONLINE_O1)
        s.deal(rng)
        for _ in range(rng.randrange(5)):
            if s.is_terminal():
                break
            s.apply(rng.choice(s.legal_actions()))
        if s.is_terminal():
            continue
        o = observe(s, s.current)
        alternative = sample_state(o, rng)
        assert np.array_equal(m.features(o), m.features(observe(alternative, o.me)))
        assert np.isfinite(m.features(o)).all()


def test_export_masks_illegal_actions_and_checks_rules(tmp_path):
    m = api()
    import torch
    torch.manual_seed(42)
    s = new_match(ONLINE_O1)
    s.deal(random.Random(2))
    o = observe(s, s.current)
    net = torch.nn.Sequential(torch.nn.Linear(len(m.features(o)), 32), torch.nn.ReLU(),
                              torch.nn.Linear(32, 9))
    p = tmp_path / 'policy.npz'
    m.export_policy(net, p, {'rules_fingerprint': ONLINE_O1.fingerprint})
    portable = m.PolicyNetwork(p)
    expected = net(torch.tensor(m.features(o))).detach().numpy()
    assert np.max(np.abs(portable.logits(o) - expected)) < 1e-5
    probs = portable.probs(o)
    assert set(probs) == set(o.legal_actions)
    assert sum(probs.values()) == pytest.approx(1)
    s2 = new_match(DEFAULT_RULES)
    s2.deal(random.Random(2))
    with pytest.raises(ValueError, match='rules'):
        portable.probs(observe(s2, s2.current))


def test_collection_is_reproducible_and_groups_complete_hands():
    m = api()
    a = m.collect_chunk(('heuristic', 300000, 12))
    b = m.collect_chunk(('heuristic', 300000, 12))
    for k in ('x', 'y', 'mask', 'group'):
        assert np.array_equal(a[k], b[k])
    assert len(set(a['group'])) == 12
    assert a['mask'][np.arange(len(a['y'])), a['y']].all()
    assert np.isfinite(a['x']).all()


def test_learned_search_registration_and_legal_play(tmp_path):
    m = api()
    import torch
    from pathlib import Path
    from trucoai.agents import make_agent
    from trucoai.arena import play_match
    s = new_match(ONLINE_O1)
    s.deal(random.Random(4))
    net = torch.nn.Sequential(torch.nn.Linear(len(m.features(observe(s, s.current))), 9))
    p = tmp_path / 'network.npz'
    m.export_policy(net, p, {'rules_fingerprint': ONLINE_O1.fingerprint})
    bp = Path(__file__).resolve().parents[1] / 'runs' / 'mccfr_rs_o1_s7' / 'compact'
    agent = make_agent(f'learned:{bp}?net={p}&n=4&k=2&belief=1&bcards=1')
    for seed in range(2):
        result = play_match(agent, make_agent('heuristic'), seed=seed, a_seat=seed, rules=ONLINE_O1)
        assert result.sides['a'].illegal == 0
    assert make_agent('zap:v1').name == 'Zap v1'


def test_mixture_rejects_overlapping_hand_groups():
    m = api()
    assert hasattr(m, 'combine_datasets')
    a = m.collect_chunk(('heuristic', 555000, 2))
    b = m.collect_chunk(('random', 556000, 2))
    mixed = m.combine_datasets([a, b])
    assert len(mixed['y']) == len(a['y']) + len(b['y'])
    with pytest.raises(ValueError, match='overlap'):
        m.combine_datasets([a, a])


def test_network_can_model_all_actions_when_requested(tmp_path):
    m = api()
    import torch
    from pathlib import Path
    s = new_match(ONLINE_O1)
    s.deal(random.Random(4))
    s.apply(6)
    o = observe(s, s.current)
    net = torch.nn.Sequential(torch.nn.Linear(len(m.features(o)), 9))
    p = tmp_path / 'network.npz'
    m.export_policy(net, p, {'rules_fingerprint': ONLINE_O1.fingerprint})
    bp = Path(__file__).resolve().parents[1] / 'runs' / 'mccfr_rs_o1_s7' / 'compact'
    agent = m.LearnedSearchAgent(bp, p, neural_bets=True)
    assert agent._policy_probs(o) == pytest.approx(agent.network.probs(o))


def test_fold_signal_uses_only_unambiguous_public_transition():
    m = api()
    assert hasattr(m, 'fold_signal')
    from trucoai.engine import RUN, RAISE
    rng = random.Random(88)
    s = new_match(ONLINE_O1)
    s.deal(rng)
    before = observe(s, s.current)
    s.apply(0)
    s.apply(RUN)
    s.deal(rng)
    after = observe(s, before.me)
    assert m.fold_signal(before, 0, after)
    assert not m.fold_signal(before, RAISE, after)
    from dataclasses import replace
    assert not m.fold_signal(before, 0, replace(after, hand_no=after.hand_no+1))
    assert not m.fold_signal(before, 0, replace(after, scores=before.scores))


def test_adaptive_expert_preserves_champion_against_nonfolding_style():
    m = api()
    assert hasattr(m, 'AdaptiveLearnedAgent')
    from pathlib import Path
    from trucoai.agents import make_agent
    from trucoai.arena import play_match
    root = Path(__file__).resolve().parents[1]
    net = root/'runs/e8_imitation_s20261020/policy.npz'
    if not net.exists():
        pytest.skip('E8 checkpoint unavailable')
    bp = root/'runs/mccfr_rs_o1_s7/compact'
    wrapper = m.AdaptiveLearnedAgent(m.LearnedSearchAgent(bp, net, n=4, belief=False))
    for seed in range(3):
        logs = [[], []]
        r1 = play_match(wrapper, make_agent('heuristic'), seed, 0, rules=ONLINE_O1, log=logs[0])
        r2 = play_match(make_agent('zap:v1'), make_agent('heuristic'), seed, 0, rules=ONLINE_O1, log=logs[1])
        assert not wrapper.expert_enabled
        assert logs[0] == logs[1]
        assert r1.a_won == r2.a_won


def test_adaptive_can_load_general_network_and_resets_between_matches():
    m = api()
    from pathlib import Path
    from trucoai.agents import make_agent
    root = Path(__file__).resolve().parents[1]
    specialist = root/'runs/e8_imitation_s20261020/policy.npz'
    general = root/'runs/e8_mixture_s20261021/policy.npz'
    if not specialist.exists() or not general.exists():
        pytest.skip('E8 checkpoints unavailable')
    bp = root/'runs/mccfr_rs_o1_s7/compact'
    agent = make_agent(f'learned:{bp}?net={specialist}&general={general}&gate=1&n=4')
    assert isinstance(agent.baseline, m.LearnedSearchAgent)
    assert agent.baseline.network.path == str(general.resolve())
    agent.expert_enabled = True
    agent.reset(123)
    assert not agent.expert_enabled and agent.previous is None


def test_style_belief_does_not_count_same_hand_evidence_twice():
    m = api()
    assert hasattr(m, 'StyleBelief')
    belief = m.StyleBelief(2)
    first = belief.update(1, [0.8, 0.2])
    assert first == pytest.approx([0.8, 0.2])
    assert belief.update(1, [0.8, 0.2]) == pytest.approx(first)
    # Replacement evidence for the current hand, not another independent hand.
    assert belief.update(1, [0.6, 0.4]) == pytest.approx([0.6, 0.4])
    assert belief.update(2, [0.8, 0.2]) == pytest.approx([0.48/0.56, 0.08/0.56])
    belief.reset()
    assert belief.posterior == pytest.approx([0.5, 0.5])


def test_style_agent_registration_legality_and_reset():
    m = api()
    from pathlib import Path
    from trucoai.agents import make_agent
    from trucoai.arena import play_match
    root = Path(__file__).resolve().parents[1]
    paths = [root/f'runs/{r}/policy.npz' for r in ['e8_imitation_s20261020',
             'e8_mixture_s20261021', 'e8_style_heuristic_s31', 'e8_style_malandro_s32']]
    if not all(p.exists() for p in paths):
        pytest.skip('E8 checkpoints unavailable')
    bp = root/'runs/mccfr_rs_o1_s7/compact'
    spec = (f'learned:{bp}?net={paths[0]}&general={paths[1]}&style_models={paths[2]},{paths[3]}'
            '&style=1&style_n=4&n=4&belief=1&bcards=1&bets=0&mix=0&temper=1')
    agent = make_agent(spec)
    assert isinstance(agent, m.StyleAdaptiveAgent)
    result = play_match(agent, make_agent('heuristic'), 55, 0, rules=ONLINE_O1)
    assert result.sides['a'].illegal == 0
    agent.reset(55)
    assert agent.belief.posterior == pytest.approx([1/3]*3)
    assert agent.expert_decisions == agent.general_decisions == 0
