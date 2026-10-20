"""Bounded deterministic memoization for offline experiments.

Keys are immutable legal observations, never simulator private state.
No change to features, matrix operations, masks or random-number consumption.
"""
import numpy as np
from .learned import PolicyNetwork, features


_FEATURES = {}
_FINGERPRINTS = {}


def key(obs):
    # RuleSet.status is a dict despite the frozen dataclass.
    rules = tuple((k, tuple(sorted(v.items())) if isinstance(v,dict) else v)
                  for k,v in vars(obs.rules).items())
    return (rules, *[v for k,v in vars(obs).items() if k != 'rules'])


class FastNetwork(PolicyNetwork):
    def __init__(self, original):
        self.path, self.meta, self.layers = original.path, original.meta, original.layers
        self._probs = {}

    def probs(self, obs):
        k = key(obs)
        if k not in self._probs:
            if len(self._probs) >= 4096:
                self._probs.clear()
            self._probs[k] = super().probs(obs)
        return self._probs[k]

    def logits(self, obs):
        k = key(obs)
        if k[0] not in _FINGERPRINTS:
            if len(_FINGERPRINTS) >= 32:
                _FINGERPRINTS.clear()
            _FINGERPRINTS[k[0]] = obs.rules.fingerprint
        if _FINGERPRINTS[k[0]] != self.meta['rules_fingerprint']:
            raise ValueError('incompatible rules for learned policy')
        if k not in _FEATURES:
            if len(_FEATURES) >= 8192:
                _FEATURES.clear()
            _FEATURES[k] = features(obs)
        x = _FEATURES[k]
        for i, (w, b) in enumerate(self.layers):
            x = x @ w + b
            if i+1 < len(self.layers):
                x = np.maximum(x,0)
        return x


def accelerate(agent):
    seen, networks = set(), {}
    def visit(node):
        if id(node) in seen:
            return
        seen.add(id(node))
        for key in ('network', 'own_network', 'opponent_network'):
            value = getattr(node,key,None)
            if isinstance(value,PolicyNetwork) and not isinstance(value,FastNetwork):
                if value.path not in networks:
                    networks[value.path] = FastNetwork(value)
                setattr(node,key,networks[value.path])
        for key in ('models','policies','style_models'):
            for child in getattr(node,key,[]):
                visit(child)
        for key in ('fallback','expert','general'):
            child=getattr(node,key,None)
            if child is not None:
                visit(child)
    visit(agent)
    return agent
