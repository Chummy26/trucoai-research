import random

import pytest

from trucoai import cards as C
from trucoai.engine import (ACCEPT, COVER_0, IRON, MAO11, PLAY_0, PLAY_1, PLAY_2, RAISE,
                            RUN, TIE, hand_outcome, new_match)
from trucoai.rules import RuleSet


def deck_with(mao_cards, pe_cards, vira):
    """Monta um baralho: cartas do mão, do pé e a vira; resto qualquer."""
    m = [C.parse(x) for x in mao_cards]
    p = [C.parse(x) for x in pe_cards]
    v = C.parse(vira)
    used = set(m + p + [v])
    rest = [c for c in range(40) if c not in used]
    deck = []
    for i in range(3):
        deck += [m[i], p[i]]
    return deck + [v] + rest


def start(mao_cards, pe_cards, vira, scores=(0, 0), rules=None):
    st = new_match(rules or RuleSet(), first_dealer=1)  # jogador 0 é mão
    st.scores = list(scores)
    st.deal(random.Random(0), deck_with(mao_cards, pe_cards, vira))
    return st


# --- cartas -----------------------------------------------------------------
def test_ordem_sem_manilha():
    vira = C.parse("4o")  # manilha = 5
    order = ["4p", "6p", "7p", "Qp", "Jp", "Kp", "Ap", "2p", "3p"]
    s = [C.strength(C.parse(x), vira) for x in order]
    assert s == sorted(s) and len(set(s)) == len(s)


def test_manilha_ciclica_e_naipes():
    assert C.manilha_rank(C.parse("3o")) == C.RANKS.index("4")
    assert C.manilha_rank(C.parse("7o")) == C.RANKS.index("Q")
    vira = C.parse("3o")
    zap, copas, espadilha, picafumo = (C.strength(C.parse(x), vira)
                                       for x in ("4p", "4c", "4e", "4o"))
    assert zap > copas > espadilha > picafumo > C.strength(C.parse("3p"), vira)


def test_comuns_mesmo_valor_empatam():
    vira = C.parse("4o")
    assert C.strength(C.parse("3p"), vira) == C.strength(C.parse("3o"), vira)


# --- rodadas e empates --------------------------------------------------------
@pytest.mark.parametrize("results,expected", [
    ([0, 0], (True, 0)),
    ([0, 1], (False, None)),
    ([0, 1, 1], (True, 1)),
    ([TIE, 1], (True, 1)),
    ([TIE, TIE], (False, None)),
    ([TIE, TIE, 0], (True, 0)),
    ([TIE, TIE, TIE], (True, None)),
    ([1, TIE], (True, 1)),
    ([0, 1, TIE], (True, 0)),
    ([0], (False, None)),
])
def test_hand_outcome(results, expected):
    assert hand_outcome(results, 0, RuleSet()) == expected


def test_tres_empates_regra_mao():
    assert hand_outcome([TIE, TIE, TIE], 1, RuleSet(all_tied="mao")) == (True, 1)


def test_mao_completa_vencedor_pontua_1():
    st = start(["3p", "2p", "Ap"], ["4o", "5o", "6o"], "7c")
    for _ in range(2):
        st.apply(PLAY_0 if st.hands[st.current][0] is not None else
                 [a for a in st.legal_actions() if a < 3][0])
        st.apply([a for a in st.legal_actions() if a < 3][0])
    assert st.phase == "hand_over" and st.hand_winner == 0 and st.scores == [1, 0]


def test_vencedor_da_rodada_abre_a_proxima():
    st = start(["4o", "3p", "2p"], ["Ap", "5o", "6o"], "7c")
    st.apply(PLAY_0)  # 4o perde para Ap
    st.apply(PLAY_0)
    assert st.round_results == [1] and st.current == 1 and st.leader == 1


def test_empate_mantem_quem_abriu():
    st = start(["3p", "4o", "5o"], ["3o", "6o", "7o"], "Kc")
    st.apply(PLAY_0)
    st.apply(PLAY_0)
    assert st.round_results == [TIE] and st.current == 0


# --- truco ------------------------------------------------------------------
def test_truco_aceito_vale_3():
    st = start(["3p", "2p", "Ap"], ["4o", "5o", "6o"], "7c")
    st.apply(RAISE)
    assert st.phase == "respond" and st.current == 1 and st.pending == 3
    st.apply(ACCEPT)
    assert st.stake == 3 and st.current == 0 and st.phase == "play"
    assert RAISE not in st.legal_actions()  # quem pediu não pede de novo


def test_truco_recusado_da_1():
    st = start(["3p", "2p", "Ap"], ["4o", "5o", "6o"], "7c")
    st.apply(RAISE)
    st.apply(RUN)
    assert st.scores == [1, 0] and st.phase == "hand_over"


def test_contra_aumento_e_recusa_da_valor_anterior():
    st = start(["3p", "2p", "Ap"], ["4o", "5o", "6o"], "7c")
    st.apply(RAISE)  # truco (3)
    st.apply(RAISE)  # seis
    assert st.pending == 6 and st.current == 0 and st.stake == 3
    st.apply(RUN)
    assert st.scores == [0, 3]


def test_aceite_depois_de_cadeia_devolve_a_vez_a_quem_pediu():
    st = start(["3p", "2p", "Ap"], ["4o", "5o", "6o"], "7c")
    st.apply(PLAY_0)
    st.apply(RAISE)  # pé pede truco antes de responder a carta
    st.apply(RAISE)  # mão pede seis
    st.apply(ACCEPT)
    assert st.stake == 6 and st.current == 1 and st.phase == "play"
    # quem fez o último aumento (jogador 0) não pode pedir nove
    st.apply(PLAY_0)
    assert RAISE in st.legal_actions() if st.current == 1 else RAISE not in st.legal_actions()


def test_escada_termina_em_12():
    st = start(["3p", "2p", "Ap"], ["4o", "5o", "6o"], "7c")
    for _ in range(4):
        st.apply(RAISE)
    assert st.pending == 12 and RAISE not in st.legal_actions()
    st.apply(ACCEPT)
    assert st.stake == 12


# --- encoberta ---------------------------------------------------------------
def test_encoberta_proibida_na_primeira_e_perde_depois():
    st = start(["3p", "2p", "Ap"], ["4o", "5o", "6o"], "7c")
    assert COVER_0 not in st.legal_actions()
    st.apply(PLAY_0)
    st.apply(PLAY_0)  # 0 ganha a primeira
    assert COVER_0 + 1 in st.legal_actions()
    st.apply(COVER_0 + 1)  # 2p encoberta
    st.apply(PLAY_1)  # 5o aberta vence a encoberta
    assert st.round_results == [0, 1]


def test_duas_encobertas_empatam():
    st = start(["3p", "2p", "Ap"], ["4o", "5o", "6o"], "7c")
    st.apply(PLAY_0)
    st.apply(PLAY_0)
    st.apply(COVER_0 + 1)
    st.apply(COVER_0 + 1)
    assert st.round_results == [0, TIE] and st.hand_winner == 0


# --- mão de onze / ferro ------------------------------------------------------
def test_mao_de_onze_jogar_vale_3_sem_truco():
    st = start(["3p", "2p", "Ap"], ["4o", "5o", "6o"], "7c", scores=(5, 11))
    assert st.special == MAO11 and st.phase == "mao11" and st.current == 1
    st.apply(ACCEPT)
    assert st.stake == 3 and st.current == 0
    assert RAISE not in st.legal_actions()


def test_mao_de_onze_correr_da_1_ao_adversario():
    st = start(["3p", "2p", "Ap"], ["4o", "5o", "6o"], "7c", scores=(11, 4))
    assert st.current == 0
    st.apply(RUN)
    assert st.scores == [11, 5]


def test_mao_de_ferro_sem_truco_e_fecha_partida():
    st = start(["3p", "2p", "Ap"], ["4o", "5o", "6o"], "7c", scores=(11, 11))
    assert st.special == IRON and RAISE not in st.legal_actions()
    while not st.is_terminal():
        st.apply(st.legal_actions()[0])
    assert st.phase == "match_over" and st.match_winner == 0


def test_partida_termina_em_12_e_dealer_alterna():
    st = new_match()
    rng = random.Random(1)
    dealers = []
    while st.match_winner is None:
        st.deal(rng)
        dealers.append(st.dealer)
        while not st.is_terminal():
            st.apply(rng.choice(st.legal_actions()))
    assert max(st.scores) >= 12
    assert all(a != b for a, b in zip(dealers, dealers[1:]))


def test_partidas_aleatorias_nunca_quebram():
    rng = random.Random(7)
    for _ in range(300):
        st = new_match()
        while st.match_winner is None:
            st.deal(rng)
            while not st.is_terminal():
                acts = st.legal_actions()
                assert acts
                st.apply(rng.choice(acts))
            assert st.phase in ("hand_over", "match_over")


def test_acao_ilegal_rejeitada():
    st = start(["3p", "2p", "Ap"], ["4o", "5o", "6o"], "7c")
    with pytest.raises(ValueError):
        st.apply(ACCEPT)


def test_online_o1_permite_correr_na_vez():
    from trucoai.rules import ONLINE_O1
    st = start(["3p", "2p", "Ap"], ["4o", "5o", "6o"], "7c", scores=(6, 1), rules=ONLINE_O1)
    assert RUN in st.legal_actions()
    st.apply(RUN)
    assert st.scores == [6, 2]  # como observado na Online
    assert RUN not in start(["3p", "2p", "Ap"], ["4o", "5o", "6o"], "7c").legal_actions()


def test_online_o1_sem_limite_perto_de_12():
    from trucoai.rules import ONLINE_O1
    st = start(["3p", "2p", "Ap"], ["4o", "5o", "6o"], "7c", scores=(8, 10), rules=ONLINE_O1)
    for _ in range(4):
        st.apply(RAISE)
    assert st.pending == 12


def test_online_o1_mao_de_ferro_visivel_sem_correr_nem_virar():
    from trucoai.observation import observe
    from trucoai.rules import ONLINE_O1
    st = start(["3p", "2p", "Ap"], ["4o", "5o", "6o"], "7c", scores=(11, 11), rules=ONLINE_O1)
    assert st.special == IRON and st.stake == 1
    o = observe(st, 0)
    assert not o.blind and all(c is not None for c in o.my_hand)
    assert set(st.legal_actions()) == {PLAY_0, PLAY_1, PLAY_2}  # sem truco nem correr
    st.apply(PLAY_0)
    st.apply(PLAY_0)
    assert all(a < 3 for a in st.legal_actions())  # sem virar na 2ª rodada


def test_online_o1_sem_correr_na_mao_de_onze():
    from trucoai.rules import ONLINE_O1
    st = start(["3p", "2p", "Ap"], ["4o", "5o", "6o"], "7c", scores=(11, 5), rules=ONLINE_O1)
    st.apply(ACCEPT)
    assert RUN not in st.legal_actions()
