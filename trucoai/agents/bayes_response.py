"""Joint posterior over hypothetical hands and behavioral models.

The latent opponent model is held fixed through each simulated hand. Neither
opponent identity nor actual private cards are available to this agent.
"""
import random
import numpy as np
from .base import Agent
from .learned import StyleBelief
from ..observation import observe, sample_state
from ..equity import default_table


def joint_weights(prior, likelihood):
    likelihood = np.maximum(np.asarray(likelihood, dtype=float), 1e-12)
    weights = likelihood * np.asarray(prior)[None, :]
    return weights / weights.sum(), likelihood.mean(axis=0)


class BayesResponseAgent(Agent):
    def __init__(self, models, own_network, n=48, k=2, own_greedy=False,
                 forgetting=0.02):
        if not models or n < 1 or k < 1 or not 0 <= forgetting <= 1:
            raise ValueError('invalid posterior search configuration')
        super().__init__()
        self.models, self.own_network = models, own_network
        self.n, self.k = n, k
        self.own_greedy, self.forgetting = own_greedy, forgetting
        self.belief = StyleBelief(len(models))
        self.meta = dict(algorithm='joint hand/style posterior, fixed latent per rollout',
                         n=n, k=k, own_greedy=own_greedy, forgetting=forgetting,
                         models=[getattr(m, 'meta', {}) for m in models],
                         own_network=getattr(own_network, 'path', 'test'))
        self.reset()

    def reset(self, seed=None):
        super().reset(seed)
        self.belief.reset()
        self.diagnostics = {}

    def _rollout(self, s, model, me, rng):
        while not s.is_terminal():
            obs = observe(s, s.current)
            probs = (self.own_network.probs(obs) if s.current == me
                     else model._policy_probs(obs))
            if not probs:
                probs = {a: 1/len(obs.legal_actions) for a in obs.legal_actions}
            if self.own_greedy and s.current == me:
                a = max(probs, key=probs.get)
            else:
                a = rng.choices(list(probs), weights=list(probs.values()))[0]
            s.apply(a)
        return default_table()(s.scores[me], s.scores[1-me])

    def act(self, obs):
        if self.belief.hand_no != obs.hand_no:
            self.belief.posterior = ((1-self.forgetting)*self.belief.posterior
                                     + self.forgetting/len(self.models))
            prior = self.belief.posterior.copy()
        else:
            prior = self.belief.prior.copy()
        worlds = [sample_state(obs, self.rng) for _ in range(self.n*self.k)]
        likelihood = np.ones((len(worlds), len(self.models)))
        if any(p == obs.opp for _, p, _ in obs.public_log[1:]):
            likelihood = np.array([[m._likelihood(w, obs) for m in self.models]
                                   for w in worlds])
        joint, evidence = joint_weights(prior, likelihood)
        posterior = self.belief.update(obs.hand_no, evidence)
        self.diagnostics = {'posterior': posterior.tolist(), 'worlds': len(worlds)}
        indices = self.rng.choices(range(joint.size), weights=joint.ravel(), k=self.n)
        values = {a: 0. for a in obs.legal_actions}
        for index in indices:
            wi, mi = divmod(index, len(self.models))
            seed = self.rng.getrandbits(32)
            for a in values:
                s = worlds[wi].clone()
                s.apply(a)
                values[a] += self._rollout(s, self.models[mi], obs.me, random.Random(seed))
        return max(values, key=values.get)
