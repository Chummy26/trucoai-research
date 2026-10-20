import importlib.util
import json
from pathlib import Path
import pytest
from scripts.evaluate_frozen import summarize_pairs


def test_parallel_panel_matches_seeded_serial_games_and_preserves_outputs(tmp_path):
    assert importlib.util.find_spec("scripts.e13_panel"), "cross-matchup panel runner missing"
    from scripts.e13_panel import run_panel
    from scripts import e10_rapid
    root = Path(__file__).resolve().parents[1]
    model = (f"learned:{root}/runs/mccfr_rs_o1_s7/compact?"
             f"net={root}/runs/e12_frozen_20261020/policy.npz&nbets=1&bcards=1&n=1")
    cfg = tmp_path/"portfolio.json"
    cfg.write_text(json.dumps(dict(models=[model], policies=["heuristic", "random"],
                                   payoff=[[.7], [.3]], style_n=1)))
    plan = {"jobs": [
        dict(name="one", a=f"payoff:{cfg}", b="random", deals=2, start_seed=225000),
        dict(name="two", a="random", b="heuristic", deals=2, start_seed=225010),
    ]}
    p = tmp_path/"plan.json"
    p.write_text(json.dumps(plan))
    out = tmp_path/"results"
    result = run_panel(p, out, workers=2)
    assert result["completed"]
    tracked = result["source_and_artifact_sha256"]
    assert not any(Path(p).name == "enable_playtruco_alignment.py" for p in tracked)
    assert any(Path(p).name == "e13_panel.py" for p in tracked)
    for job, got in zip(plan["jobs"], result["matchups"]):
        rows, _, _ = e10_rapid.batch((job["a"], job["b"], list(range(job["start_seed"], job["start_seed"]+2))))
        expected = summarize_pairs(rows, job["a"], job["b"], job["start_seed"])
        assert got["deal_wins"] == expected["deal_wins"]
        assert got["a_metrics"]["illegal_rate"] == 0
        assert got["games"] == 4
    before = (out/"results.json").read_bytes()
    with pytest.raises(FileExistsError):
        run_panel(p, out, workers=2)
    assert (out/"results.json").read_bytes() == before
