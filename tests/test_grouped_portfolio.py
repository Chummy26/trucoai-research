import random
from trucoai.engine import new_match
from trucoai.observation import observe
from trucoai.rules import PLAYTRUCO_O1


def test_related_styles_combine_evidence_for_same_response():
    from trucoai.agents.grouped_portfolio import GroupedPortfolioAgent
    class Model:
        def __init__(self, likelihood):
            self.likelihood = likelihood
        def _likelihood(self, w, o):
            return self.likelihood
    class Policy:
        def __init__(self, index):
            self.index = index
        def act(self, o):
            return o.legal_actions[self.index]
        def reset(self, seed=None):
            pass
    first, second = Policy(0), Policy(1)
    a = GroupedPortfolioAgent([Model(.31),Model(.31),Model(.38)],
                              [first,first,second],second,style_n=2,threshold=.5)
    s=new_match(PLAYTRUCO_O1);s.deal(random.Random(42));s.apply(s.legal_actions()[0])
    obs=observe(s,s.current)
    assert a.act(obs)==obs.legal_actions[0]
    assert a.last_group_mass > .6
    a.reset(3)
    assert a.belief.hand_no is None
