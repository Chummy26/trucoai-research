import pytest
from trucoai.integration.online_protocol import OnlineProtocolReader
from trucoai.integration.online_actor import OnlineActor


@pytest.mark.parametrize('want', [
    'increasePoints', 'reraiseIncreasePoints', 'acceptIncreasePoints',
    'refuseIncreasePoints', 'giveUpRound', 'acceptLastPoint', 'refuseLastPoint'])
def test_synchronous_client_message_is_confirmed_without_second_click(monkeypatch, want):
    import trucoai.integration.online_actor as module
    now = [100.]
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    r = OnlineProtocolReader()
    a = OnlineActor(r)
    clicks = []
    class Page:
        def wait_for_timeout(self, ms): now[0] += ms / 1000
    def click(*args, **kwargs):
        clicks.append(want)
        r.on_frame('sent', {'json': [want]})
        return True
    monkeypatch.setattr(a, '_wait_locate', lambda *args: object())
    monkeypatch.setattr(a, '_click_text', click)
    monkeypatch.setattr(a.cursor, 'pause_ms', lambda *args: None)
    assert a._click_expect(Page(), ('label',), want, 'raise')
    assert clicks == [want] and a.enabled


def test_new_turn_for_same_player_cannot_receive_previous_turn_click(monkeypatch):
    import trucoai.integration.online_actor as module
    from trucoai.integration.shadow import ShadowGame, HandStart
    from trucoai.rules import ONLINE_O1
    now = [100.]
    monkeypatch.setattr(module.time, 'monotonic', lambda: now[0])
    r = OnlineProtocolReader()
    r.me_pos = r.my_team = r.current = 0
    r.in_hand = True
    r.last_sq = 10
    r.turn_deadline = 110.
    a = OnlineActor(r)
    g = ShadowGame(rules=ONLINE_O1)
    g.feed(HandStart([0, 1, 2], 39, 0, 0, True))
    a.start_hand([0, 1, 2])
    inputs = []
    class Mouse:
        def down(self): inputs.append('down')
        def up(self): pass
    class Page:
        mouse = Mouse()
        def wait_for_timeout(self, ms):
            now[0] += .1
            r.on_event({'tp': 'run_turn', 'sq': 11,
                        'dt': {'cpp': 0, 'rt': 10., 'td': 10.}})
    class Cursor:
        def pause_ms(self, *args): pass
        def click(self, page, *args):
            page.wait_for_timeout(100)
            page.mouse.down()
            page.mouse.up()
    a.cursor = Cursor()
    monkeypatch.setattr(a, '_match_card_button', lambda *args, **kw: ({'x': 1, 'y': 1}, ''))
    assert not a.do(Page(), g.observation(), 0)
    assert inputs == [] and a.enabled and a.retry_later
