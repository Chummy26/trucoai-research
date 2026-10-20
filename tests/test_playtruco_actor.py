"""Identificação das cartas na tela pelo recorte do spritesheet da PlayTruco."""

from trucoai.integration.playtruco_actor import PlayTrucoActor, sprite_card
from trucoai.integration.playtruco_protocol import card_from_code

SIZE = "2050px 75px"

# (código da carta enviada pelo cliente, recorte do botão clicado), de cliques
# verificados ao vivo em logs/playtruco/20261019_0225*_remote e _0230*_remote.
VERIFIED = [("c1", -50), ("c2", -100), ("c5", -250), ("c6", -300), ("c7", -350),
            ("c11", -400), ("c12", -450), ("c13", -500), ("d4", -700), ("d5", -750),
            ("d6", -800), ("d7", -850), ("d12", -950), ("d13", -1000), ("s1", -1050),
            ("s3", -1150), ("s6", -1300), ("s7", -1350), ("s11", -1400), ("s12", -1450),
            ("h2", -1600), ("h5", -1750), ("h11", -1900), ("h12", -1950), ("h13", -2000)]


def test_sprite_bate_com_cliques_verificados():
    for code, x in VERIFIED:
        assert sprite_card(f"{x}px 0px", SIZE) == card_from_code(code), code


def test_sprite_cobre_as_40_cartas_sem_repetir():
    cards = [sprite_card(f"{-50 * f}px 0px", SIZE) for f in range(1, 41)]
    assert None not in cards and len(set(cards)) == 40


def test_sprite_desconhecido_ou_escala_diferente():
    assert sprite_card("0px 0px", SIZE) is None  # verso
    assert sprite_card("-73px 0px", SIZE) is None  # fora do quadro
    assert sprite_card(None, None) is None
    # interface em outra escala: quadros de 25 px
    assert sprite_card("-700px 0px", "1025px 37.5px") == card_from_code("s11")  # quadro 28


def test_botao_escolhido_pela_carta_e_nao_pela_ordem():
    a = PlayTrucoActor(reader=None)
    a.hand = [card_from_code("c7"), card_from_code("c3")]
    # na tela, ordem invertida em relação à mão registrada (caso real da falha)
    buttons = [{"x": 10, "pos": "-150px 0px", "size": SIZE},
               {"x": 20, "pos": "-350px 0px", "size": SIZE}]
    b, _ = a._button_for(buttons, card_from_code("c7"))
    assert b["x"] == 20
    b, why = a._button_for(buttons[:1], card_from_code("c7"))
    assert b is None and why
