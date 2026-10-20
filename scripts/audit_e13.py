"""Audit exported E13 statistics, without loading agents or original dependencies."""
import hashlib
import json
import math
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]

def audit():
    folder = ROOT / "research/e13"
    for entry in json.loads((folder / "export_manifest.json").read_text()):
        path = ROOT / entry["path"]
        if hashlib.sha256(path.read_bytes()).hexdigest() != entry["public_sha256"]:
            raise ValueError(f"Export changed: {entry['path']}")
    result = json.loads((folder / "results.json").read_text())
    analysis = json.loads((folder / "analysis.json").read_text())
    if not result["completed"] or not analysis["completed"]:
        raise ValueError("Incomplete confirmation")
    families = list(analysis["cells"])
    rows = {m["name"]: m for m in result["matchups"]}
    candidate = control = 0
    differences = [[] for _ in range(40)]
    for family in families:
        a, b = rows[f"candidate_vs_{family}"], rows[f"e9_vs_{family}"]
        for row in (a, b):
            assert row["start_seed"] == 240000 and row["games"] == 80
            assert len(row["deal_wins"]) == 40
            assert all(x in (0, 1, 2) for x in row["deal_wins"])
            assert sum(row["deal_wins"]) == row["a_wins"]
            assert row["a_metrics"]["illegal_rate"] == row["b_metrics"]["illegal_rate"] == 0
        assert a["a_wins"] == analysis["cells"][family]["candidate_wins"]
        assert b["a_wins"] == analysis["cells"][family]["e9_wins"]
        candidate += a["a_wins"]
        control += b["a_wins"]
        for index, (x, y) in enumerate(zip(a["deal_wins"], b["deal_wins"])):
            differences[index].append((x - y) / 2)
    paired = [statistics.mean(d) for d in differences]
    gain = statistics.mean(paired)
    se = statistics.stdev(paired) / math.sqrt(len(paired))
    z = statistics.NormalDist().inv_cdf(0.975)
    ci = [gain - z * se, gain + z * se]
    assert candidate == analysis["primary_wins"]["candidate"] == 363
    assert control == analysis["primary_wins"]["e9"] == 319
    assert math.isclose(gain, analysis["gain"]["mean"], abs_tol=1e-12)
    assert all(math.isclose(x, y, abs_tol=1e-12) for x, y in zip(ci, analysis["gain"]["ci"]))
    print(json.dumps(dict(export_integrity=True, candidate_wins=candidate,
        control_wins=control, games_per_policy=560, gain=gain, paired_normal95=ci,
        scope="Stored simulation statistics only; no arena rerun or live validation."), indent=2))

if __name__ == "__main__":
    audit()
