"""Accumulate style posterior mass for hypotheses sharing a response."""
import numpy as np
from .portfolio import PortfolioAgent
from ..observation import sample_state


class GroupedPortfolioAgent(PortfolioAgent):
    def act(self, obs):
        likelihoods=np.ones(len(self.models))
        if any(p==obs.opp and t in ('play','raise','accept','mao11')
               for t,p,_ in obs.public_log[1:]):
            worlds=[sample_state(obs,self.rng) for _ in range(self.style_n)]
            likelihoods=np.array([sum(m._likelihood(w,obs) for w in worlds)/len(worlds)
                                  for m in self.models])
        posterior=self.belief.update(obs.hand_no,likelihoods)
        groups={}
        for i,policy in enumerate(self.policies):
            ident=id(policy)
            if ident not in groups:
                groups[ident]=[policy,0.,i]
            groups[ident][1] += float(posterior[i])
        policy,mass,index=max(groups.values(),key=lambda g:g[1])
        self.last_group_mass=mass
        if mass<self.threshold:
            policy=self.fallback
            index=len(self.models)
        self.selection_counts[index]+=1
        return policy.act(obs)
