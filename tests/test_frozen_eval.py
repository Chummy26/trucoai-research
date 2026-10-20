import importlib.util

from trucoai.agents import make_agent
from trucoai.arena import play_match, summarize
from trucoai.rules import PLAYTRUCO_O1


def test_frozen_summary_requires_complete_mirrored_pairs():
    assert importlib.util.find_spec('scripts.evaluate_frozen') is not None
    from scripts.evaluate_frozen import summarize_pairs
    import pytest
    rows = [play_match(make_agent('random'), make_agent('random'), seed, seat, rules=PLAYTRUCO_O1)
            for seed in (100, 101, 102) for seat in (0, 1)]
    expected = summarize(rows)
    actual = summarize_pairs(rows, 'random', 'random', 100)
    assert actual['games'] == 6 and actual['a_wins'] == expected['a_wins']
    assert len(actual['deal_wins']) == 3
    with pytest.raises(ValueError, match='pair'):
        summarize_pairs(rows[:-1], 'random', 'random', 100)
    with pytest.raises(ValueError, match='pair'):
        summarize_pairs(rows + [rows[0]], 'random', 'random', 100)
    from dataclasses import replace
    unfinished = [replace(rows[0], a_points=3, b_points=4)] + rows[1:]
    with pytest.raises(ValueError, match='unfinished'):
        summarize_pairs(unfinished, 'random', 'random', 100)
