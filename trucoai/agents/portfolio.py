"""Select trained responses using only public action likelihoods."""
import numpy as np
from .base import Agent
from .learned import StyleBelief
from ..observation import sample_state


class PortfolioAgent(Agent):
    name = 'public-style-portfolio'

    def __init__(self, models, policies, fallback, style_n=48, threshold=0.5):
        if not models or len(models) != len(policies):
            raise ValueError('one response policy is required per style model')
        if style_n < 1 or not 0 <= threshold <= 1:
            raise ValueError('invalid style budget or threshold')
        super().__init__()
        self.models, self.policies, self.fallback = models, policies, fallback
        self.style_n, self.threshold = style_n, threshold
        self.belief = StyleBelief(len(models))
        self.meta = {'style_models': [getattr(m, 'meta', {}) for m in models],
                     'response_policies': [getattr(p, 'meta', {}) for p in policies],
                     'fallback': getattr(fallback, 'meta', {}),
                     'style_n': style_n, 'threshold': threshold,
                     'selection': 'whole-hand public-history likelihood; no opponent identity'}
        self.reset()

    def reset(self, seed=None):
        super().reset(seed)
        seen = set()
        for agent in self.models + self.policies + [self.fallback]:
            if id(agent) not in seen:
                seen.add(id(agent))
                if hasattr(agent, 'reset'):
                    agent.reset(seed)
        self.belief.reset()
        self.selection_counts = [0] * (len(self.models) + 1)

    def act(self, obs):
        likelihoods = np.ones(len(self.models))
        if any(p == obs.opp and t in ('play', 'raise', 'accept', 'mao11')
               for t, p, _ in obs.public_log[1:]):
            worlds = [sample_state(obs, self.rng) for _ in range(self.style_n)]
            likelihoods = np.array([sum(m._likelihood(w, obs) for w in worlds) / len(worlds)
                                    for m in self.models])
        posterior = self.belief.update(obs.hand_no, likelihoods)
        selected = int(np.argmax(posterior))
        if posterior[selected] < self.threshold:
            selected = len(self.models)
            policy = self.fallback
        else:
            policy = self.policies[selected]
        self.selection_counts[selected] += 1
        return policy.act(obs)
