import importlib.util
import pytest
from trucoai.integration import shadow as sh
from trucoai.integration.online_protocol import OnlineProtocolReader
from trucoai.rules import ONLINE_O1


def test_hand_counter_advances_and_restarts_per_match():
    game=sh.ShadowGame(rules=ONLINE_O1)
    for i in range(3):
        game.feed(sh.HandStart([0,1,2],39,i,0,True))
        assert game.observation().hand_no == i+1
    assert hasattr(sh,'MatchStart'), 'explicit match boundary missing'
    game.feed(sh.MatchStart(2))
    assert game.observation() is None
    game.feed(sh.HandStart([0,1,2],39,0,0,True))
    assert game.observation().hand_no == 1 and game.match_no == 2


def test_session_resets_once_per_match_and_preserves_between_hands():
    spec=importlib.util.find_spec('trucoai.integration.lifecycle')
    assert spec is not None, 'live agent lifecycle missing'
    from trucoai.integration.lifecycle import LiveAgentSession
    from trucoai.agents.base import RandomAgent
    class Remember(RandomAgent):
        def reset(self,seed=None):
            super().reset(seed)
            self.style_memory=[]
    agent=Remember(); game=sh.ShadowGame(rules=ONLINE_O1)
    session=LiveAgentSession(agent,game,seed_base=100)
    first=session.feed(sh.MatchStart(1))
    assert first == {'match_no':1,'seed':101}
    assert session.feed(sh.HandStart([0,1,2],39,0,0,True)) is None
    agent.style_memory.append('hand1')
    session.feed(sh.HandStart([0,1,2],39,1,0,True))
    assert agent.style_memory == ['hand1']
    assert game.observation().hand_no == 2
    assert session.feed(sh.MatchStart(2)) == {'match_no':2,'seed':102}
    assert agent.style_memory == []
    session.feed(sh.HandStart([0,1,2],39,0,0,True))
    assert game.observation().hand_no == 1


def test_protocol_emits_initial_and_explicit_match_boundaries_without_duplicates():
    rd=OnlineProtocolReader()
    rd.on_event({'tp':'game_sync','dt':{'ps':[{'mp':True,'pos':0,'tm':0}],
                                     'sc':{'rp':[0,0]},'mt':'p','np':2,'ef':0}})
    events=rd.poll().events
    assert len(events)==1 and type(events[0]).__name__=='MatchStart'
    rd.on_event({'tp':'game_sync','dt':{'sc':{'rp':[1,0]}}})
    assert rd.poll().events == []
    rd.on_frame('recv',{'json':['startGame']})
    ev=rd.poll().events
    assert len(ev)==1 and ev[0].match_no==2
    rd.on_event({'tp':'game_sync','dt':{'sc':{'rp':[0,0]}}})
    assert rd.poll().events == []


def test_public_notices_and_own_hidden_card_are_preserved():
    rd=OnlineProtocolReader();rd.me_pos=0;rd.scores=[0,0];rd.my_team=0
    rd.on_event({'tp':'start_round','dt':{'vr':'d3','pp':1,'ps':[{'pos':0,'cs':['s4','h5','c6']},{'pos':1,'cs':['x','x','x']}]}})
    rd.poll()
    rd.on_frame('sent',{'json':['playCard',{'value':4,'suit':'spades','hidden':True}]})
    rd.on_event({'tp':'play_card','dt':{'pos':0,'cd':'x','hd':True}})
    ev=rd.poll().events[0]
    assert ev.card is not None
    rd.on_event({'tp':'player_afk','dt':{'pos':1}})
    rd.on_event({'tp':'cardsRevealed','dt':{'pos':0,'cs':['s4','h5','c6']}})
    state=rd.poll()
    assert [n['type'] for n in state.notices]==['player_afk','cardsRevealed']
    assert rd.unknown==[] and rd.poll().notices==[]


def test_unknown_game_event_blocks_actions_until_a_new_hand():
    rd=OnlineProtocolReader();rd.me_pos=0;rd.current=0;rd.in_hand=True
    rd.on_event({'tp':'unrecognized_game_transition','dt':{}})
    assert rd.unknown and not rd.poll().my_turn
    rd.scores=[0,0];rd.my_team=0
    rd.on_event({'tp':'start_round','dt':{'vr':'d3','pp':1,'ps':[{'pos':0,'cs':['s4','h5','c6']},{'pos':1,'cs':['x','x','x']}]}})
    assert not rd.poll().my_turn  # a new hand must not inherit the previous turn
    rd.on_event({'tp':'run_turn','dt':{'cpp':0}})
    assert rd.poll().my_turn


def test_actual_e9_policy_replays_captured_matches_with_resets():
    import json
    import numpy as np
    from pathlib import Path
    from scripts.portfolio_factory import factory
    from trucoai.integration.lifecycle import LiveAgentSession
    from trucoai.integration.online import filter_legal
    root=Path(__file__).resolve().parents[1]
    manifest=root/'runs/e9_frozen_20261020/manifest.json'
    if not manifest.exists():
        pytest.skip('frozen E9 policy unavailable')
    agent=factory(json.loads(manifest.read_text())['candidate'])
    assert len(agent.models)==5 and agent.style_n==48
    decisions=resets=hands=0
    for filename in ['online_partida2.jsonl','online_partida3.jsonl']:
        rd=OnlineProtocolReader();game=sh.ShadowGame(rules=ONLINE_O1)
        session=LiveAgentSession(agent,game,seed_base=17)
        decided=None
        for line in (root/'tests/data'/filename).read_text(encoding='utf8').splitlines():
            row=json.loads(line)
            if row['dir']=='recv':
                if 'msg' in row:rd.on_frame('recv',{'json':row['msg']})
                else:rd.on_event(row['ev'])
            else:
                rd.on_frame('sent',{'json':row['msg']})
            state=rd.poll()
            for ev in state.events:
                prior=agent.belief.posterior.copy()
                reset=session.feed(ev)
                if reset:
                    resets+=1
                    assert np.allclose(agent.belief.posterior,np.ones(5)/5)
                if isinstance(ev,sh.HandStart):
                    hands+=1
                    if not reset:assert np.array_equal(agent.belief.posterior,prior)
            obs=game.observation()
            if state.my_turn and obs is not None:
                key=(game.match_no,obs.hand_no,len(obs.public_log),obs.phase)
                if key!=decided:
                    obs=filter_legal(obs,state.buttons)
                    assert agent.act(obs) in obs.legal_actions
                    decisions+=1;decided=key
        assert not game.divergences and rd.unknown==[]
    assert decisions>=30 and hands==26 and resets>=3
