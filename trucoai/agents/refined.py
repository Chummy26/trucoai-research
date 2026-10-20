"""Experimental rollout variance reduction. Input remains public Observation.

Late hypothetical worlds are integrated exactly under the rollout model.
Greedy continuations are a PIMC approximation, not an information-set solver.
"""
import random
from .learned import LearnedSearchAgent
from ..observation import observe
from ..equity import default_table


class RefinedSearchAgent(LearnedSearchAgent):
    def __init__(self, *args, tail=True, own_greedy=False, **kwargs):
        super().__init__(*args, **kwargs)
        self.tail, self.own_greedy = tail, own_greedy
        self.meta.update(tail=tail, own_greedy=own_greedy)

    def _tail(self, s, me):
        if s.is_terminal():
            return default_table()(s.scores[me], s.scores[1-me])
        obs = observe(s, s.current)
        probs = self._policy_probs(obs)
        if probs is None:
            probs = {a: 1/len(obs.legal_actions) for a in obs.legal_actions}
        values = []
        for a, probability in probs.items():
            if probability <= 0:
                continue
            child = s.clone()
            child.apply(a)
            values.append((self._tail(child, me), probability))
        if self.own_greedy and s.current == me:
            return max(v for v, p in values)
        return sum(v*p for v, p in values) / sum(p for v, p in values)

    def _rollout(self, s, rng, me):
        while not s.is_terminal():
            left = sum(c is not None for hand in s.hands for c in hand)
            if self.tail and left <= 2:
                return self._tail(s, me)
            if self.own_greedy and s.current == me:
                probs = self._policy_probs(observe(s, me))
                action = max(probs, key=probs.get) if probs else self._rollout_action(s, rng, me)
            else:
                action = self._rollout_action(s, rng, me)
            s.apply(action)
        return default_table()(s.scores[me], s.scores[1-me])
