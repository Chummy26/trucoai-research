"""Exercise the real live loop and loaded agent, with an offline browser boundary."""
import dataclasses
import hashlib
import json
from pathlib import Path


def test_live_loop_logs_the_exact_observation_received_by_loaded_agent(monkeypatch, tmp_path):
    import pytest
    pytest.importorskip("playwright.sync_api")
    import playwright.sync_api
    from trucoai.integration import online, online_protocol, online_actor
    from trucoai.agents.learned import features

    class Reader(online_protocol.OnlineProtocolReader):
        def __init__(self):
            super().__init__()
            self.on_event({'tp': 'game_sync', 'sq': 0, 'dt': {
                'ef': 0, 'mt': 'p', 'np': 2, 'sc': {'rp': [0, 0]},
                'ps': [{'mp': True, 'pos': 0, 'tm': 0}]}})
            self.on_event({'tp': 'start_round', 'sq': 1, 'dt': {
                'vr': 'd3', 'pp': 1, 'ps': [{'pos': 0, 'cs': ['s4', 'h5', 'c6']}]}})
            self.on_event({'tp': 'run_turn', 'sq': 2, 'dt': {
                'cpp': 0, 'catip': True, 'cgr': True, 'chc': False,
                'crp': 1, 'urp': 3, 'rt': 10., 'td': 10.}})

    class Context:
        pages = [True]

    context = Context()
    delivered = []

    class Page:
        polls = 0
        def on(self, *args): pass
        def goto(self, *args): pass
        def set_default_timeout(self, *args): pass
        def evaluate(self, *args): return {'hand': [], 'labels': [], 'virar': []}
        def wait_for_timeout(self, *args):
            self.polls += 1
            if self.polls == 2: context.pages = []

    class Playwright:
        def __enter__(self): return None
        def __exit__(self, *args): pass

    def execute(self, page, obs, action):
        delivered.append((obs, action))
        return True

    monkeypatch.setattr(playwright.sync_api, 'sync_playwright', Playwright)
    monkeypatch.setattr(online, '_launch', lambda *args, **kwargs: (context, Page()))
    monkeypatch.setattr(online_protocol, 'OnlineProtocolReader', Reader)
    monkeypatch.setattr(online_actor.OnlineActor, 'do', execute)
    root = Path(__file__).resolve().parents[1]
    spec = 'heuristic'
    monkeypatch.setenv('TRUCOAI_SITE_URL', 'https://table.example/game')
    online.run_agent('play', spec, tmp_path, tmp_path, cdp='offline', max_stake=0)
    rows = [json.loads(s) for s in next(tmp_path.glob('*_play.jsonl')).read_text().splitlines()]
    receipt = next(r['policy_input'] for r in rows if 'policy_input' in r)
    decision = next(r for r in rows if 'decision' in r)
    obs, action = delivered[0]
    assert len(delivered) == 1 and decision['decision'] == action
    assert receipt['observation'] == json.loads(json.dumps(dataclasses.asdict(obs)))
    assert receipt['features_sha256'] == hashlib.sha256(features(obs).tobytes()).hexdigest()
    assert decision['input_sha256'] == receipt['observation_sha256']
    assert rows[0]['integration_revision'] == 'online_alignment_20261020_v4'
    assert rows[0]['agent'] == spec and any(r.get('acted') for r in rows)
    assert 0 < receipt['public_context']['remaining_s'] <= 10
