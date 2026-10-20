"""Match-scoped agent memory shared by the live adapter and offline replay."""


class LiveAgentSession:
    def __init__(self, agent, shadow, seed_base):
        self.agent, self.shadow, self.seed_base = agent, shadow, seed_base
        self.match_no = None

    def feed(self, event):
        self.shadow.feed(event)
        match_no = self.shadow.match_no
        if match_no and match_no != self.match_no:
            seed = self.seed_base + match_no
            self.agent.reset(seed)
            self.match_no = match_no
            return {'match_no': match_no, 'seed': seed}
        return None
