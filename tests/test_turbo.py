"""Pipeline turbo de ponta a ponta: núcleo CPU v2 (e GPU, se houver) -> compacto -> consulta Python."""

import random
import shutil
import subprocess
from pathlib import Path

import pytest

from trucoai.cfr.abstraction import info_key
from trucoai.cfr.compact import CompactPolicy, finalize_unsorted
from trucoai.engine import new_match
from trucoai.observation import observe
from trucoai.rules import PLAYTRUCO_O1

ROOT = Path(__file__).resolve().parent.parent
BIN = ROOT / "rust" / "mccfr" / "target" / "release" / ("mccfr.exe" if shutil.os.name == "nt" else "mccfr")
EQ = ROOT / "runs" / "mccfr_rs_o1_s7" / "equity_table.txt"
pytestmark = pytest.mark.skipif(not (BIN.exists() and EQ.exists()), reason="binário ou tabela ausente")


def root_keys(n=400, seed=0):
    rng = random.Random(seed)
    out = []
    for _ in range(n):
        st = new_match(PLAYTRUCO_O1)
        st.deal(rng)
        out.append(info_key(observe(st, st.current)))
    return out


def _hit_rate(policy, keys):
    return sum(policy.lookup(k)[0] is not None for k in keys) / len(keys)


@pytest.fixture
def proj_tmp():
    """Pasta temporária DENTRO do projeto: o ambiente bloqueia o executável de gravar no %TEMP%."""
    d = ROOT / "logs" / f"pytest_turbo_{random.randrange(1 << 30)}"
    d.mkdir(parents=True)
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.mark.parametrize("prune", ["0", "0.95"])
def test_cpu_v2_exporta_e_python_encontra_os_estados(proj_tmp, prune):
    tmp_path = proj_tmp
    out = tmp_path / "cpu"
    subprocess.run([str(BIN), "train2", "--rules", "o1", "--eq", str(EQ), "--out-dir", str(out),
                    "--summary", str(tmp_path / "s.json"), "--seed", "3", "--threads", "4",
                    "--seconds", "3", "--log2-cap", "20", "--log2-sketch", "20", "--gate", "2",
                    "--prune", prune, "--prune-after", "10000", "--save-min-visits", "5"],
                   check=True, capture_output=True)
    n = finalize_unsorted(out, {"rules_version": "paulista-1v1-o1"})
    p = CompactPolicy(out)
    assert n == len(p) > 1000 and p.meta["key_hash"] == "fnv1a64-splitmix"
    assert _hit_rate(p, root_keys()) > 0.9  # estados de início de mão são os mais visitados


def test_gpu_wavefront_exporta_e_python_encontra_os_estados(tmp_path):
    torch = pytest.importorskip("torch")
    if not torch.cuda.is_available():
        pytest.skip("sem GPU")
    from trucoai.gpu.train import WavefrontTrainer
    g = WavefrontTrainer(EQ, log2_cap=20, log2_sketch=20, gate=2, batch=4096, prune=0.95,
                         prune_after=10000, seed=5)
    g.train(4, log_every=1000)
    n = g.export_unsorted(tmp_path / "gpu", save_min=5)
    del g
    finalize_unsorted(tmp_path / "gpu", {"rules_version": "paulista-1v1-o1"})
    p = CompactPolicy(tmp_path / "gpu")
    assert n == len(p) > 1000
    assert _hit_rate(p, root_keys()) > 0.9
