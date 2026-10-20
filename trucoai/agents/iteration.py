"""Separate continuation and opponent models for E9 policy iteration."""
from .learned import LearnedSearchAgent, PolicyNetwork


class AsymmetricSearchAgent(LearnedSearchAgent):
    name = 'asymmetric-learned-search'

    def __init__(self, policy_path, network_path, opponent_network_path, **kwargs):
        super().__init__(policy_path, network_path, **kwargs)
        self.opponent_network = PolicyNetwork(opponent_network_path)
        self._root_me = None
        self.meta = {**self.meta, 'opponent_network': self.opponent_network.path,
                     'opponent_network_meta': self.opponent_network.meta,
                     'role_assignment': 'public observation seat at root decision'}

    def reset(self, seed=None):
        super().reset(seed)
        self._root_me = None

    def act(self, obs):
        self._root_me = obs.me
        return super().act(obs)

    def action_values(self, obs, actions=None):
        self._root_me = obs.me
        return super().action_values(obs, actions)

    def _likelihood(self, world, obs):
        self._root_me = obs.me
        return super()._likelihood(world, obs)

    def _policy_probs(self, obs):
        if self._root_me is not None and obs.me != self._root_me:
            return self.opponent_network.probs(obs)
        return super()._policy_probs(obs)
