"""Kernel GPU (trucoai/gpu/mccfr_kernel.cu) reproduz o motor Python decisão a decisão."""

import ctypes

import pytest

torch = pytest.importorskip("torch")
pytestmark = pytest.mark.skipif(not torch.cuda.is_available(), reason="sem GPU CUDA")

from trucoai.cfr.abstraction import info_key_v  # noqa: E402
from trucoai.cfr.compact import key_hash_fnv  # noqa: E402
from trucoai.cfr.mccfr import abstract_legal  # noqa: E402
from trucoai.engine import new_match  # noqa: E402
from trucoai.observation import observe  # noqa: E402
from trucoai.rules import DEFAULT_RULES, ONLINE_O1  # noqa: E402

ST_MAX = 48


@pytest.fixture(scope="module")
def kernel():
    from trucoai.gpu.cuda import Kernel
    k = Kernel()
    k.set_stack(16384)
    return k


@pytest.mark.parametrize("absv,absname", [(0, "v1"), (1, "v2a"), (2, "v2b")])
@pytest.mark.parametrize("o1,rules", [(1, ONLINE_O1), (0, DEFAULT_RULES)])
def test_kernel_reproduz_motor_python(kernel, o1, rules, absv, absname):
    from trucoai.gpu.cuda import ptr
    games = 3000
    head = torch.zeros(games * 12, dtype=torch.int32, device="cuda")
    recs = torch.zeros(games * ST_MAX * 2, dtype=torch.int64, device="cuda")
    nrec = torch.zeros(games, dtype=torch.int32, device="cuda")
    kernel.launch("selftest", (games + 127) // 128, 128,
                  [ctypes.c_uint64(7 + o1), ctypes.c_int(o1), ctypes.c_int(games),
                   ptr(head), ptr(recs), ptr(nrec), ctypes.c_int(absv)])
    kernel.sync()
    head, recs, nrec = head.cpu().tolist(), recs.cpu().tolist(), nrec.cpu().tolist()
    decisions = specials = 0
    seen = set()
    for g in range(games):
        h = head[g * 12:(g + 1) * 12]
        dealer, s0, s1, deck = h[0], h[1], h[2], h[3:10]
        st = new_match(rules, first_dealer=dealer)
        st.scores = [s0, s1]
        st.deal(None, deck=deck + [c for c in range(40) if c not in deck])
        seen.add(st.special)
        for i in range(nrec[g]):
            key_h = recs[(g * ST_MAX + i) * 2] & ((1 << 64) - 1)
            meta = recs[(g * ST_MAX + i) * 2 + 1]
            cur, mask, a = meta >> 32, (meta >> 8) & 0xFFFFFF, meta & 0xFF
            assert st.current == cur
            obs = observe(st, cur)
            assert key_hash_fnv(info_key_v(obs, absname)) == key_h
            assert sum(1 << x for x in abstract_legal(obs)) == mask
            st.apply(a)
            decisions += 1
        assert st.is_terminal()
        assert st.scores == [h[10], h[11]]
    assert decisions > 3 * games
    assert seen == {"normal", "mao11", "iron"}
