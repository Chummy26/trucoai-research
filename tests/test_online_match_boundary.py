import pytest
from trucoai.integration.online_protocol import OnlineProtocolReader
from trucoai.integration.guard import StakeGuard, parse_brl


def sync(reader, fee=200, scores=(0, 0), sq=0, room=None):
    dt = {
        'ef': fee, 'mt': 'p', 'np': 2, 'sc': {'rp': list(scores)},
        'ps': [{'mp': True, 'pos': 0, 'tm': 0}]}
    if room is not None: dt['rid'] = room
    reader.on_event({'tp': 'game_sync', 'sq': sq, 'dt': dt})


def hand(reader):
    reader.on_event({'tp': 'start_round', 'sq': 1, 'dt': {
        'pp': 1, 'vr': 'd3', 'ps': [{'pos': 0, 'cs': ['s4', 'h5', 'c6']}]}})
    reader.on_event({'tp': 'run_turn', 'sq': 2, 'dt': {
        'cpp': 0, 'catip': True, 'cgr': True, 'chc': False, 'rt': 10., 'td': 10.}})


@pytest.mark.parametrize('order', ['sync_first', 'notification_first'])
def test_initial_boundary_is_once_and_retains_authoritative_table(order):
    r = OnlineProtocolReader()
    if order == 'sync_first':
        sync(r)
        r.on_frame('recv', {'json': ['startGame']})
    else:
        r.on_frame('recv', {'json': ['startGame']})
        sync(r)
    hand(r)
    state = r.poll()
    assert sum(type(e).__name__ == 'MatchStart' for e in state.events) == 1
    assert state.my_turn and r.entry_cents == 200 and r.match_no == 1
    guard = StakeGuard(max_stake=100)
    guard.update(parse_brl(state.table_value_text), 'authoritative game_sync')
    assert guard.check().allowed


@pytest.mark.parametrize('order', ['sync_first', 'notification_first'])
def test_rematch_updates_value_and_never_reuses_previous_table(order):
    r = OnlineProtocolReader()
    sync(r, fee=200)
    hand(r)
    r.on_event({'tp': 'finish_game', 'sq': 3, 'dt': {'sc': {'rp': [8, 12]}}})
    r.poll()
    if order == 'sync_first':
        sync(r, fee=15000)
        r.on_frame('recv', {'json': ['startGame']})
    else:
        r.on_frame('recv', {'json': ['startGame']})
        assert r.entry_cents is None
        sync(r, fee=15000)
    hand(r)
    state = r.poll()
    assert sum(type(e).__name__ == 'MatchStart' for e in state.events) == 1
    assert r.match_no == 2 and r.entry_cents == 15000
    guard = StakeGuard(max_stake=100)
    guard.update(parse_brl(state.table_value_text), 'new authoritative game_sync')
    assert not guard.check().allowed


def test_new_match_without_sync_does_not_inherit_paid_table_authorization():
    r = OnlineProtocolReader()
    sync(r)
    hand(r)
    r.on_event({'tp': 'finish_game', 'sq': 3, 'dt': {'sc': {'rp': [12, 1]}}})
    r.on_frame('recv', {'json': ['startGame']})
    assert r.entry_cents is None and r.table_value_text() is None


def test_missing_new_match_fee_stays_blocked_even_after_duplicate_notice():
    r = OnlineProtocolReader()
    r.on_frame('recv', {'json': ['startGame']})
    r.on_event({'tp': 'game_sync', 'sq': 0, 'dt': {'mt': 'p', 'np': 2, 'sc': {'rp': [0, 0]}}})
    r.on_frame('recv', {'json': ['startGame']})
    assert r.entry_cents is None and r.table_value_text() is None


def test_duplicate_start_does_not_erase_first_hand_or_pending_events():
    r = OnlineProtocolReader()
    sync(r)
    hand(r)
    r.on_frame('recv', {'json': ['startGame']})
    state = r.poll()
    assert r.match_no == 1 and state.my_turn
    assert sum(type(e).__name__ == 'HandStart' for e in state.events) == 1


@pytest.mark.parametrize('sync_first', [True, False])
def test_real_protocol_room_key_coalesces_both_orders(sync_first):
    r = OnlineProtocolReader()
    if sync_first:
        sync(r, room='synthetic-room-a')
        r.on_frame('recv', {'json': ['startGame', 'synthetic-room-a']})
    else:
        r.on_frame('recv', {'json': ['startGame', 'synthetic-room-a']})
        sync(r, room='synthetic-room-a')
    hand(r)
    r.on_frame('recv', {'json': ['startGame', 'synthetic-room-a']})
    assert r.match_no == 1 and r.poll().my_turn and r.entry_cents == 200
    assert 'synthetic-room-a' not in str(r.public_context())


def test_new_room_sync_without_finish_event_requires_fresh_table_value():
    r = OnlineProtocolReader()
    sync(r, room='synthetic-room-a')
    hand(r)
    r.poll()
    sync(r, fee=15000, room='synthetic-room-b')
    r.on_frame('recv', {'json': ['startGame', 'synthetic-room-b']})
    assert r.match_no == 2 and r.entry_cents == 15000
    assert sum(type(e).__name__ == 'MatchStart' for e in r.poll().events) == 1


def test_same_room_reconnect_and_notice_preserve_match_memory():
    r = OnlineProtocolReader()
    sync(r, room='synthetic-room-a')
    hand(r)
    r.poll()
    r.on_disconnect()
    sync(r, room='synthetic-room-a', scores=(4, 2), sq=20)
    r.on_frame('recv', {'json': ['startGame', 'synthetic-room-a']})
    assert r.match_no == 1 and r.scores == [4, 2] and r.entry_cents == 200
    assert not r.poll().events
