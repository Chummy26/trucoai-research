"""E8: imitation of legal Zap observations; portable NumPy inference.

Training labels are actions, never hidden cards. The network can also replace
the coarse card policy inside search and public-history likelihoods.
"""
from __future__ import annotations

import json
import random
from pathlib import Path

import numpy as np

from .. import cards as C
from ..cfr.mccfr import abstract_legal, sorted_slots, to_abstract
from ..engine import new_match
from ..observation import Observation, observe
from ..rules import ONLINE_O1
from .search import SearchAgent
from .base import Agent

FEATURE_VERSION = 1


def features(o: Observation) -> np.ndarray:
    """Relative seats and sorted card slots; input is exclusively Observation."""
    out = []

    def onehot(value, size):
        out.extend(float(i == value) for i in range(size))

    slots = sorted_slots(o)
    strengths = [C.strength(o.my_hand[i], o.vira) if o.my_hand[i] is not None else -1
                 for i in slots]
    for v in strengths + [-1] * (3 - len(strengths)):
        onehot(v + 1, 15)
    for p in (o.me, o.opp):
        played = [c for rnd in o.rounds for who, c in rnd if who == p]
        for c in played + [-2] * (3 - len(played)):
            onehot(c + 2 if c < 0 else C.strength(c, o.vira) + 2, 16)
    for p in (o.me, o.opp):
        onehot(min(o.scores[p], 11), 12)
    onehot(('play', 'respond', 'mao11').index(o.phase), 3)
    onehot(('normal', 'mao11', 'iron').index(o.special), 3)
    out.extend((o.stake / 12, (o.pending or 0) / 12, o.opp_cards_left / 3,
                float(o.leader == o.me), float(o.dealer == o.me), float(o.blind),
                float(o.last_raiser == o.me), float(o.last_raiser == o.opp),
                float(o.resume_player == o.me), float(o.mao11_player == o.me)))
    for r in list(o.round_results) + [-2] * (3 - len(o.round_results)):
        onehot(0 if r == -2 else 1 if r == -1 else 2 if r == o.me else 3, 4)
    for p in (o.me, o.opp):
        for t in ('raise', 'accept'):
            events = [v for kind, who, v in o.public_log if kind == t and who == p]
            out.extend((len(events) / 4, max(events, default=0) / 12))
    counts = [0] * 14
    for c in o.unknown_pool():
        counts[C.strength(c, o.vira)] += 1
    out.extend(v / 4 for v in counts)
    legal = abstract_legal(o)
    out.extend(float(a in legal) for a in range(9))
    return np.asarray(out, dtype=np.float32)


def export_policy(net, path, meta):
    """Export linear/ReLU network without requiring torch at inference."""
    arrays = {}
    i = 0
    for layer in net:
        if hasattr(layer, 'weight'):
            arrays[f'w{i}'] = layer.weight.detach().cpu().numpy().T.copy()
            arrays[f'b{i}'] = layer.bias.detach().cpu().numpy().copy()
            i += 1
    arrays['meta'] = np.asarray(json.dumps({**meta, 'feature_version': FEATURE_VERSION, 'layers': i}))
    with Path(path).open('xb') as f:
        np.savez(f, **arrays)


class PolicyNetwork:
    def __init__(self, path):
        self.path = str(Path(path).resolve())
        with np.load(path, allow_pickle=False) as z:
            self.meta = json.loads(str(z['meta']))
            if self.meta['feature_version'] != FEATURE_VERSION:
                raise ValueError('incompatible features')
            self.layers = [(z[f'w{i}'].copy(), z[f'b{i}'].copy()) for i in range(self.meta['layers'])]

    def logits(self, obs):
        if obs.rules.fingerprint != self.meta['rules_fingerprint']:
            raise ValueError('incompatible rules for learned policy')
        x = features(obs)
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i + 1 < len(self.layers):
                x = np.maximum(x, 0)
        return x

    def probs(self, obs):
        logits = self.logits(obs)
        acts = list(obs.legal_actions)
        v = np.array([logits[to_abstract(obs, a)] for a in acts])
        p = np.exp(v - v.max())
        p /= p.sum()
        return {a: float(q) for a, q in zip(acts, p)}


def collect_chunk(args):
    """Synthetic independent hands, deterministic per hand, no human data."""
    from . import make_agent
    spec, start, count = args
    agent = make_agent(spec)
    xs, ys, masks, groups = [], [], [], []
    for seed in range(start, start + count):
        rng = random.Random(seed)
        agent.reset(seed + 314159)
        s = new_match(ONLINE_O1, first_dealer=rng.randrange(2))
        s.scores = [rng.randrange(12), rng.randrange(12)]
        s.deal(rng)
        while not s.is_terminal():
            o = observe(s, s.current)
            a = agent.act(o)
            if a not in o.legal_actions:
                raise RuntimeError('illegal teacher action')
            xs.append(features(o))
            ys.append(to_abstract(o, a))
            masks.append([j in abstract_legal(o) for j in range(9)])
            groups.append(seed)
            s.apply(a)
    return {'x': np.asarray(xs, np.float32), 'y': np.asarray(ys, np.int64),
            'mask': np.asarray(masks, bool), 'group': np.asarray(groups, np.int64)}


def combine_datasets(parts):
    seen = set()
    for part in parts:
        groups = set(np.unique(part['group']).tolist())
        if seen & groups:
            raise ValueError('overlap between hand groups')
        seen.update(groups)
    return {k: np.concatenate([p[k] for p in parts]) for k in ('x', 'y', 'mask', 'group')}


class LearnedSearchAgent(SearchAgent):
    """Search with a trained card rollout policy and optional posterior inference."""
    name = 'learned-search'

    def __init__(self, policy_path, network_path, neural_bets=False, **kwargs):
        super().__init__(policy_path, **kwargs)
        self.network = PolicyNetwork(network_path)
        self.neural_bets = neural_bets
        self.meta = {**self.meta, 'network': self.network.path, 'network_meta': self.network.meta}

    def _policy_probs(self, obs):
        if self.neural_bets:
            return self.network.probs(obs)
        base = super()._policy_probs(obs)
        if obs.phase != 'play':
            return base or self.network.probs(obs)
        learned = self.network.probs(obs)
        if not base:
            return learned
        mass = sum(p for a, p in base.items() if a < 6)
        denom = sum(p for a, p in learned.items() if a < 6)
        if denom <= 1e-12:
            return base
        return {a: learned[a] * mass / denom if a < 6 else p for a, p in base.items()}


def fold_signal(previous, action, current):
    """Infer an on-turn fold only when the public transition proves it.

    After our opening card of trick one, the hand cannot end without us
    acting again unless the opponent folds on turn. One adjacent hand and
    exactly the awarded stake exclude skipped hands and unrelated scores.
    """
    if previous is None:
        return False
    me, opp = previous.me, previous.opp
    return (current.me == me and current.hand_no == previous.hand_no + 1
            and previous.phase == 'play' and previous.special == 'normal'
            and not previous.round_results and not previous.rounds[-1]
            and action in (0, 1, 2)
            and current.scores[me] == previous.scores[me] + previous.stake
            and current.scores[opp] == previous.scores[opp])


class AdaptiveLearnedAgent(Agent):
    """Conservative expert selection from public behavior, reset each match.

    The specialized network is enabled only after an observed on-turn fold.
    This is a documented behavior gate, not access to opponent identity or
    hidden cards. Unknown styles initially receive the frozen champion policy.
    """
    name = 'adaptive-learned'

    def __init__(self, expert, fallback=None):
        from . import make_agent
        super().__init__()
        self.expert = expert
        self.baseline = fallback if fallback is not None else make_agent('zap:v1')
        self.meta = {**expert.meta, 'gate': 'public-on-turn-fold-v1',
                     'fallback': self.baseline.meta if fallback is not None else 'zap:v1'}
        self.reset()

    def reset(self, seed=None):
        super().reset(seed)
        self.expert.reset(seed)
        self.baseline.reset(seed)
        self.previous = None
        self.previous_action = None
        self.expert_enabled = False
        self.expert_decisions = 0

    def act(self, obs):
        if fold_signal(self.previous, self.previous_action, obs):
            self.expert_enabled = True
        selected = self.expert if self.expert_enabled else self.baseline
        action = selected.act(obs)
        self.expert_decisions += int(self.expert_enabled)
        self.previous, self.previous_action = obs, action
        return action


class StyleBelief:
    """Accumulate whole-hand likelihoods, replacing partial same-hand evidence."""
    def __init__(self, count):
        self.count = count
        self.reset()

    def reset(self):
        self.prior = np.ones(self.count, dtype=np.float64) / self.count
        self.posterior = self.prior.copy()
        self.hand_no = None

    def update(self, hand_no, likelihood):
        if hand_no != self.hand_no:
            self.prior = self.posterior.copy()
            self.hand_no = hand_no
        weights = self.prior * np.maximum(likelihood, 1e-12)
        self.posterior = weights / weights.sum()
        return self.posterior.copy()


class StyleAdaptiveAgent(Agent):
    """Infer a mixture of trained behavior models from public action histories.

    Every model integrates over the SAME sampled possible opponent hands.
    Within a hand, its entire action sequence shares those hidden cards;
    repeated observations replace evidence rather than multiplying marginals.
    No opponent identifier or true hidden state is accepted by this interface.
    """
    name = 'style-adaptive'

    def __init__(self, expert, general, style_models, style_n=48, threshold=0.75):
        super().__init__()
        self.expert, self.general = expert, general
        self.models = [expert] + list(style_models)
        self.style_n, self.threshold = style_n, threshold
        self.belief = StyleBelief(len(self.models))
        self.meta = {**expert.meta, 'general': general.meta,
                     'style_models': [m.meta for m in self.models],
                     'style_n': style_n, 'threshold': threshold,
                     'selection': 'public-history joint likelihood per hand'}
        self.reset()

    def reset(self, seed=None):
        super().reset(seed)
        self.expert.reset(seed)
        self.general.reset(seed)
        self.belief.reset()
        self.expert_decisions = self.general_decisions = 0

    def act(self, obs):
        from ..observation import sample_state
        likelihoods = np.ones(len(self.models))
        if any(p == obs.opp and t in ('play', 'raise', 'accept', 'mao11')
               for t, p, _ in obs.public_log[1:]):
            worlds = [sample_state(obs, self.rng) for _ in range(self.style_n)]
            likelihoods = np.array([sum(model._likelihood(w, obs) for w in worlds) / len(worlds)
                                    for model in self.models])
        posterior = self.belief.update(obs.hand_no, likelihoods)
        use_expert = posterior[0] >= self.threshold
        self.expert_decisions += int(use_expert)
        self.general_decisions += int(not use_expert)
        return (self.expert if use_expert else self.general).act(obs)
