"""Blueprint compacto (cfr/compact.py): conversão fiel e consultas equivalentes."""

import pickle
import random

from trucoai.agents.blueprint import BlueprintAgent
from trucoai.cfr.compact import CompactPolicy, build_from_pkl, build_from_tsv, key_hash
from trucoai.cfr.mccfr import MCCFRTrainer
from trucoai.engine import new_match
from trucoai.observation import observe


def _trained(tmp_path):
    tr = MCCFRTrainer(seed=3)
    tr.train(60)
    pkl = tmp_path / "bp.pkl"
    tr.save(pkl)
    return pkl


def test_pkl_para_compacto_preserva_politica(tmp_path):
    pkl = _trained(tmp_path)
    blob = pickle.load(open(pkl, "rb"))
    n = build_from_pkl(pkl, tmp_path / "compact")
    cp = CompactPolicy(tmp_path / "compact")
    assert n == len(blob["policy"]) == len(cp)
    for k, dist in blob["policy"].items():
        d, v = cp.lookup(k)
        assert v == blob["visits"][k]
        for a, p in dist.items():
            assert abs(d.get(a, 0.0) - p) <= 1 / 255 + 1e-9
    assert cp.lookup("nao-existe") == (None, 0)


def test_tsv_para_compacto_com_limiar(tmp_path):
    tsv = tmp_path / "policy.tsv"
    tsv.write_text("a:1\t50\t0=0.25,6=0.75\nb:2\t10\t7=1\nc:3\t30\t8=0.5,7=0.5\n", encoding="ascii")
    n = build_from_tsv(tsv, tmp_path / "c", min_visits=30, meta={"x": 1})
    cp = CompactPolicy(tmp_path / "c")
    assert n == 2 and cp.meta["min_visits"] == 30
    assert cp.lookup("b:2") == (None, 0)
    d, v = cp.lookup("a:1")
    assert v == 50 and abs(d[6] - 0.75) < 0.01


def test_agente_decide_igual_com_pkl_e_compacto(tmp_path):
    pkl = _trained(tmp_path)
    build_from_pkl(pkl, tmp_path / "compact")
    a = BlueprintAgent(pkl, min_visits=1)
    b = BlueprintAgent(tmp_path / "compact", min_visits=1)
    rng = random.Random(5)
    agree = total = 0
    for _ in range(300):
        st = new_match()
        st.deal(rng)
        o = observe(st, st.current)
        da, db = a.lookup(o), b.lookup(o)
        assert (da is None) == (db is None)
        if da:
            total += 1
            agree += max(da, key=da.get) == max(db, key=db.get) or abs(
                da[max(da, key=da.get)] - da.get(max(db, key=db.get), 0)) < 0.01
    assert total and agree == total


def test_hash_estavel():
    assert key_hash("n00:012::p") == key_hash("n00:012::p")
    assert key_hash("a") != key_hash("b")
