import pytest

from trucoai.integration.guard import StakeGuard, max_stake_from_env, parse_brl


@pytest.mark.parametrize("text,value", [
    ("R$ 0,25", 0.25), ("R$1", 1.0), ("R$ 1.000,00", 1000.0), ("R$ 0", 0.0),
    ("R$ 0,00", 0.0), ("Mesa grátis", 0.0), ("Entrada R$ 10 Taxa: 0.70", 10.0),
    ("R$ 5 / R$ 0", 5.0), ("", None), (None, None), ("Mesa 12", None),
    ("Grátis R$ 2", 2.0),
])
def test_parse_brl(text, value):
    assert parse_brl(text) == value


def test_trava_bloqueia_por_padrao():
    g = StakeGuard()
    assert not g.check().allowed


def test_trava_libera_so_zero_sem_teto():
    g = StakeGuard()
    g.update(0.0, "teste")
    assert g.check().allowed
    for v in (0.01, 0.25, 200.0, None):
        g.update(v, "teste")
        assert not g.check().allowed
    g.update(0.25, "teste")
    assert not g.check().allowed


def test_trava_libera_mesa_paga_ate_o_teto():
    g = StakeGuard(max_stake=2.0)
    for v in (0.0, 0.25, 2.0):
        g.update(v, "teste")
        assert g.check().allowed, v
    for v in (2.01, 200.0, None):
        g.update(v, "teste")
        assert not g.check().allowed, v


def test_trava_expira():
    g = StakeGuard(max_age_s=0.0)
    g.update(0.0, "teste")
    assert not g.check().allowed
