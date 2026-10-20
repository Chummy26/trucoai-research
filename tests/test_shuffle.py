"""Public Fisher-Yates contract, not a claim of GLI certification."""
import itertools
import random

from trucoai.engine import new_match
from trucoai.observation import observe
from trucoai.rules import PLAYTRUCO_O1


class TracedRandom(random.Random):
    def __init__(self, seed):
        super().__init__(seed)
        self.bounds = []

    def _randbelow(self, n):
        self.bounds.append(n)
        return super()._randbelow(n)


def test_full_deck_fisher_yates_before_dealing():
    rng = TracedRandom(442)
    state = new_match(PLAYTRUCO_O1)
    state.deal(rng)
    # All 39 exchanges are performed, including cards not distributed.
    assert rng.bounds == list(range(40, 1, -1))
    reference = list(range(40))
    independent = random.Random(442)
    for i in range(39, 0, -1):
        j = independent.randrange(i + 1)
        reference[i], reference[j] = reference[j], reference[i]
    assert state.hands[state.mao] == reference[0:6:2]
    assert state.hands[state.dealer] == reference[1:6:2]
    assert state.vira == reference[6]
    obs = observe(state, state.current)
    assert not {'seed', 'rng', 'deck'} & set(vars(obs))


def test_fisher_yates_exhaustive_small_deck_has_equal_preimages():
    outcomes = []
    for choices in itertools.product(range(4), range(3), range(2)):
        deck = list(range(4))
        for i, j in zip(range(3, 0, -1), choices):
            deck[i], deck[j] = deck[j], deck[i]
        outcomes.append(tuple(deck))
    assert len(outcomes) == len(set(outcomes)) == 24


def test_seeded_shuffling_has_no_repetition_and_replays():
    for seed in range(200):
        a, b = new_match(PLAYTRUCO_O1), new_match(PLAYTRUCO_O1)
        a.deal(random.Random(seed))
        b.deal(random.Random(seed))
        assert a.hands == b.hands and a.vira == b.vira
        cards = a.hands[0] + a.hands[1] + [a.vira]
        assert len(set(cards)) == 7
