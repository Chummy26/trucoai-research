"""Choose a response by measured payoffs under public-history uncertainty."""
import numpy as np
from .base import Agent
from .learned import StyleBelief
from ..observation import sample_state


class PayoffPortfolioAgent(Agent):
    name = "payoff-public-portfolio"

    def __init__(self, models, policies, payoff, style_n=48, forgetting=.02):
        super().__init__()
        matrix = np.array(payoff, dtype=float, copy=True)
        if (not models or not policies or matrix.shape != (len(policies), len(models))
                or not np.isfinite(matrix).all() or (matrix < 0).any()
                or (matrix > 1).any() or style_n < 1 or not 0 <= forgetting <= 1):
            raise ValueError("invalid payoff portfolio configuration")
        matrix.flags.writeable = False
        self.models, self.policies, self.payoff = models, policies, matrix
        self.style_n, self.forgetting = style_n, forgetting
        self.belief = StyleBelief(len(models))
        self.meta = {
            "algorithm": "maximum empirical response payoff under whole-hand public-history posterior",
            "models": [getattr(m, "meta", {}) for m in models],
            "policies": [getattr(p, "meta", {}) for p in policies],
            "payoff": matrix.tolist(), "style_n": style_n, "forgetting": forgetting,
        }
        self.reset()

    def reset(self, seed=None):
        super().reset(seed)
        seen = set()
        for node in self.models + self.policies:
            if id(node) not in seen:
                seen.add(id(node))
                if hasattr(node, "reset"):
                    node.reset(seed)
        self.belief.reset()
        self.selection_counts = [0] * len(self.policies)
        self.last_scores = np.zeros(len(self.policies))

    def act(self, obs):
        if self.belief.hand_no != obs.hand_no:
            self.belief.posterior = (
                (1-self.forgetting)*self.belief.posterior + self.forgetting/len(self.models)
            )
        evidence = np.ones(len(self.models))
        if any(p == obs.opp and t in ("play", "raise", "accept", "mao11")
               for t, p, _ in obs.public_log[1:]):
            worlds = [sample_state(obs, self.rng) for _ in range(self.style_n)]
            evidence = np.array([
                sum(model._likelihood(w, obs) for w in worlds)/len(worlds)
                for model in self.models
            ])
        posterior = self.belief.update(obs.hand_no, evidence)
        self.last_scores = self.payoff @ posterior
        index = int(np.argmax(self.last_scores))
        self.selection_counts[index] += 1
        return self.policies[index].act(obs)
