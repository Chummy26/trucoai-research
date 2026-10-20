import random
import numpy as np
from trucoai.engine import new_match
from trucoai.observation import observe
from trucoai.rules import PLAYTRUCO_O1


def test_joint_posterior_normalization_and_no_same_hand_double_count():
    from trucoai.agents.bayes_response import joint_weights
    from trucoai.agents.learned import StyleBelief
    prior = np.array([0.25, 0.75])
    likelihood = np.array([[0.2, 0.8], [0.6, 0.2]])
    weights, evidence = joint_weights(prior, likelihood)
    assert weights.shape == (2, 2)
    assert np.isclose(weights.sum(), 1)
    assert np.allclose(evidence, [.4, .5])
    assert np.allclose(weights.sum(axis=0), prior*evidence/(prior@evidence))
    belief = StyleBelief(2)
    p1 = belief.update(1, evidence)
    p2 = belief.update(1, evidence)
    assert np.allclose(p1, p2)


def test_bayes_reset_and_hidden_state_invariance():
    from trucoai.agents.bayes_response import BayesResponseAgent
    class Uniform:
        meta = {}
        def _likelihood(self, w, o):
            return 1.
        def _policy_probs(self, o):
            return {a: 1/len(o.legal_actions) for a in o.legal_actions}
    class Own:
        def probs(self, o):
            return Uniform()._policy_probs(o)
    a = BayesResponseAgent([Uniform(), Uniform()], Own(), n=4, k=2)
    s = new_match(PLAYTRUCO_O1)
    s.deal(random.Random(42))
    o = observe(s, s.current)
    a.reset(21)
    action = a.act(o)
    s.hands[1-s.current] = [34,35,36]
    assert o == observe(s, s.current)
    a.reset(21)
    assert action == a.act(observe(s, s.current))
    assert action in o.legal_actions
    assert np.allclose(a.belief.posterior, [.5, .5])
    assert a.diagnostics['worlds'] == 8
