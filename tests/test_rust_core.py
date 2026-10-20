"""Equivalência entre o núcleo Rust do MCCFR (rust/mccfr) e o motor Python.

O binário joga partidas aleatórias e imprime, a cada decisão, a chave de
infoset e as ações abstratas legais. Aqui cada jogo é refeito no motor Python e
exige-se igualdade com ``info_key(observe(...))`` e ``abstract_legal`` — como a
chave Python é invariante à informação oculta (test_observation.py), a do Rust
também é. Também confere placar e fase finais.
"""

import shutil
import subprocess
from pathlib import Path

import pytest

from trucoai.cfr.abstraction import info_key, info_key_v
from trucoai.cfr.compact import key_hash_fnv
from trucoai.cfr.mccfr import abstract_legal
from trucoai.engine import new_match
from trucoai.observation import observe
from trucoai.rules import DEFAULT_RULES, PLAYTRUCO_O1

CRATE = Path(__file__).resolve().parent.parent / "rust" / "mccfr"
BIN = CRATE / "target" / "release" / ("mccfr.exe" if shutil.os.name == "nt" else "mccfr")

# Valores de regra fixados no núcleo Rust (ver rust/mccfr/src/main.rs).
RUST_RULES = {
    "version": "paulista-1v1-h0", "target_score": 12, "stake_ladder": [1, 3, 6, 9, 12],
    "cap_raise_at_needed": False, "mao11_value": 3, "mao11_run_points": 1,
    "mao11_allows_truco": False, "iron_value": 1, "iron_blind": True,
    "allow_covered": True, "covered_from_round": 1, "all_tied": "nobody",
    "tie_next_leader": "same_leader", "allow_fold_on_turn": False,
}


@pytest.fixture(scope="module")
def binary():
    if not BIN.exists():
        if shutil.which("cargo") is None:
            pytest.skip("cargo indisponível e binário Rust não compilado")
        subprocess.run(["cargo", "build", "--release"], cwd=CRATE, check=True)
    return BIN


def test_regras_do_rust_iguais_as_default_rules(binary):
    d = DEFAULT_RULES.to_dict()
    for k, v in RUST_RULES.items():
        assert (list(d[k]) if isinstance(v, list) else d[k]) == v, k
    out = subprocess.run([str(binary), "version"], capture_output=True, text=True, check=True)
    assert out.stdout.strip() == DEFAULT_RULES.version


def test_rust_versao_o1(binary):
    out = subprocess.run([str(binary), "version", "--rules", "o1"], capture_output=True,
                         text=True, check=True)
    assert out.stdout.strip() == PLAYTRUCO_O1.version


@pytest.mark.parametrize("absv", ["v1", "v2a", "v2b"])
@pytest.mark.parametrize("rules_flag,rules", [("h0", DEFAULT_RULES), ("o1", PLAYTRUCO_O1)])
@pytest.mark.parametrize("seed", [1, 2])
def test_rust_reproduz_motor_e_chave_python(binary, seed, rules_flag, rules, absv):
    out = subprocess.run([str(binary), "selftest", "--seed", str(seed), "--games", "3000",
                          "--rules", rules_flag, "--abstraction", absv],
                         capture_output=True, text=True, check=True).stdout
    st = None
    games = decisions = folds = 0
    specials = set()
    for line in out.splitlines():
        tag, *f = line.split(" ")
        if tag == "G":
            dealer, s0, s1, *deck = map(int, f)
            st = new_match(rules, first_dealer=dealer)
            st.scores = [s0, s1]
            st.deal(None, deck=deck + [c for c in range(40) if c not in deck])
            specials.add(st.special)
            games += 1
        elif tag == "D":
            p, key, legal, action = int(f[0]), f[1], f[2], int(f[3])
            assert key_hash_fnv(key) == int(f[4])  # hash da tabela v2/GPU
            assert st.current == p
            obs = observe(st, p)
            assert info_key_v(obs, absv) == key
            assert abstract_legal(obs) == [int(x) for x in legal.split(",")]
            folds += st.phase == "play" and action == 8
            st.apply(action)
            decisions += 1
        elif tag == "E":
            assert st.is_terminal()
            assert st.scores == [int(f[0]), int(f[1])]
            assert st.phase == f[2]
    assert games == 3000 and decisions > 3 * games
    assert specials == {"normal", "mao11", "iron"}
    assert (folds > 50) if rules_flag == "o1" else (folds == 0)
