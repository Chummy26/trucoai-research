"""Mão de onze: selected E11, live reader and synchronous client confirmations."""
import json
from pathlib import Path

import pytest

from trucoai.agents import make_agent
from trucoai import cards as C
from trucoai.engine import ACCEPT, RUN, RAISE
from trucoai.integration.online_actor import OnlineActor
from trucoai.integration.online_protocol import OnlineProtocolReader
from trucoai.integration.shadow import ShadowGame, HandStart
from trucoai.rules import ONLINE_O1


def e11():
    root = Path(__file__).resolve().parents[1]
    config = root/'runs/e11_frozen_20261020/candidate.json'
    if not config.exists():
        pytest.skip('Historical E11 configuration/blueprint omitted')
    cfg = json.loads(config.read_text(encoding='utf8'))
    a = make_agent(cfg['candidate'])
    a.reset(20261020)
    assert a.n == 384 and a.neural_bets
    return a


@pytest.mark.parametrize('cards,expected', [
    (['5p', '5c', '5e'], ACCEPT),  # three strongest manilhas: guaranteed win
    (['4p', '4c', '4e'], RUN),     # all remaining cards outrank our whole hand
])
def test_selected_e11_decides_accept_and_run_in_mao11(cards, expected):
    g = ShadowGame(rules=ONLINE_O1)
    g.feed(HandStart([C.parse(c) for c in cards], C.parse('4o'), 11, 10, True))
    obs = g.observation()
    assert obs.phase == 'mao11' and set(obs.legal_actions) == {ACCEPT, RUN}
    assert e11().act(obs) == expected


@pytest.mark.parametrize('action,message', [(ACCEPT, 'acceptLastPoint'), (RUN, 'refuseLastPoint')])
def test_mao11_client_confirmation_during_click_is_not_lost(monkeypatch, action, message):
    r = OnlineProtocolReader()
    r.me_pos = r.my_team = 0
    r.scores = [11, 10]
    r.on_event({'tp': 'start_round', 'sq': 1, 'dt': {
        'pp': 1, 'vr': 'd6', 'ps': [{'pos': 0, 'cs': ['s5', 'd5', 's11']}]}})
    r.on_event({'tp': 'last_point', 'sq': 2, 'dt': {
        'tid': 0, 'cpp': 0, 'calp': True, 'crlp': True, 'rt': 10., 'td': 10.}})
    g = ShadowGame(rules=ONLINE_O1)
    for ev in r.poll().events: g.feed(ev)
    obs = g.observation()
    assert obs.phase == 'mao11' and r.is_actionable()
    a = OnlineActor(r)
    clicks = []
    class Page:
        def wait_for_timeout(self, ms): pass
    def click(page, labels, phase):
        clicks.append(labels)
        r.on_frame('sent', {'json': [message]})
        return True
    monkeypatch.setattr(a, '_wait_locate', lambda *args: object())
    monkeypatch.setattr(a, '_click_text', click)
    assert a.do(Page(), obs, action)
    assert a.enabled and len(clicks) == 1
    assert clicks[0] == (('Aceitar',) if action == ACCEPT else ('Correr',))
    assert a.learned_mao11['message'] == message


def test_confirmed_raise_returns_before_next_mao11_and_observation_reaches_e11(monkeypatch):
    r = OnlineProtocolReader()
    r.on_event({'tp': 'game_sync', 'sq': 0, 'dt': {
        'ef': 0, 'mt': 'p', 'np': 2, 'sc': {'rp': [10, 10]},
        'ps': [{'mp': True, 'pos': 0, 'tm': 0}]}})
    r.on_event({'tp': 'start_round', 'sq': 1, 'dt': {
        'pp': 1, 'vr': 'd2', 'ps': [{'pos': 0, 'cs': ['s3', 'c3', 'h3']}]}})
    r.on_event({'tp': 'run_turn', 'sq': 2, 'dt': {
        'cpp': 0, 'rt': 10., 'td': 10., 'crp': 1, 'catip': True}})
    g = ShadowGame(rules=ONLINE_O1)
    for ev in r.poll().events: g.feed(ev)
    a = OnlineActor(r)
    class Page:
        def wait_for_timeout(self, ms): pass
    def click(page, labels, phase):
        r.on_frame('sent', {'json': ['increasePoints']})
        r.on_event({'tp': 'increase_points', 'sq': 3, 'dt': {
            'akt': 0, 'ant': 1, 'npi': 3, 'rt': 10., 'td': 10.}})
        return True
    monkeypatch.setattr(a, '_wait_locate', lambda *args: object())
    monkeypatch.setattr(a, '_click_text', click)
    assert a.do(Page(), g.observation(), RAISE)
    r.on_event({'tp': 'player_decision', 'sq': 4, 'dt': {'pos': 1, 'dt': 'r'}})
    r.on_event({'tp': 'finish_round', 'sq': 5, 'dt': {'sc': {'rp': [11, 10]}, 'gu': True}})
    r.on_event({'tp': 'start_round', 'sq': 6, 'dt': {
        'pp': 1, 'vr': 'd6', 'ps': [{'pos': 0, 'cs': ['s5', 'd5', 's11']}]}})
    r.on_event({'tp': 'last_point', 'sq': 7, 'dt': {
        'tid': 0, 'cpp': 0, 'calp': True, 'crlp': True, 'rt': 10., 'td': 10.}})
    state = r.poll()
    for ev in state.events: g.feed(ev)
    obs = g.observation()
    assert not g.divergences and state.my_turn and obs.phase == 'mao11'
    assert obs.my_hand == (5, 4, 17) and obs.vira == 8  # screenshot's exact public information
    assert e11().act(obs) in (ACCEPT, RUN)
