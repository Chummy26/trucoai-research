"""Interleave complete mirrored pairs across matchups without changing games."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from contextlib import ExitStack
from dataclasses import asdict
import hashlib
import json
import time
import numpy as np
from scripts.e13_rapid import factory
from scripts.e10_rapid import batch
from scripts.evaluate_frozen import summarize_pairs
from trucoai.rules import PLAYTRUCO_O1


def panel_batch(task):
    # Import this module in spawned workers so its research loader is installed.
    return batch(task)


def run_panel(plan_path, out, workers=12):
    plan_path, out = Path(plan_path).resolve(), Path(out)
    plan = json.loads(plan_path.read_bytes())
    out.mkdir(exist_ok=False)
    # Freeze the arena's script dependency closure, not unrelated live tools.
    script_names = ("e13_panel.py", "e13_rapid.py", "e11_rapid.py", "e10_rapid.py",
                    "e10_factory.py", "portfolio_factory.py", "evaluate_iteration.py",
                    "compare_references.py", "evaluate_frozen.py")
    paths = list((ROOT/"trucoai").rglob("*.py")) + [ROOT/"scripts"/p for p in script_names]
    paths += [ROOT/p for p in plan.get("artifacts", [])] + [plan_path]
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in paths}
    receipt = dict(plan=plan, source_and_artifact_sha256=hashes, completed=False,
                   rules_fingerprint=PLAYTRUCO_O1.fingerprint, workers=workers,
                   loaded_agents={s: getattr(factory(s), "meta", {})
                                  for j in plan["jobs"] for s in (j["a"], j["b"])},
                   matchups=[])
    (out/"manifest.json").write_text(json.dumps(receipt, indent=2))
    started = time.time()
    slots = [dict(rows=[], at=[], bt=[]) for _ in plan["jobs"]]
    finished = {}
    total_games = sum(j["deals"]*2 for j in plan["jobs"])
    count = 0
    with ExitStack() as stack:
        files = [stack.enter_context((out/f"job_{i}.jsonl").open("x")) for i in range(len(slots))]
        pool = stack.enter_context(ProcessPoolExecutor(workers))
        futures = {}
        # Round-robin submission avoids stranding cores behind a small, slow cell.
        for offset in range(max(j["deals"] for j in plan["jobs"])):
            for i, job in enumerate(plan["jobs"]):
                if offset < job["deals"]:
                    task = job["a"], job["b"], [job["start_seed"]+offset]
                    futures[pool.submit(panel_batch, task)] = i
        for future in as_completed(futures):
            i = futures[future]
            job, slot = plan["jobs"][i], slots[i]
            rows, at, bt = future.result()
            slot["rows"].extend(rows); slot["at"].extend(at); slot["bt"].extend(bt)
            files[i].write(json.dumps(dict(rows=[asdict(r) for r in rows], a_times=at, b_times=bt))+"\n")
            files[i].flush()
            count += len(rows)
            if count % 40 == 0 or count == total_games:
                print(f"PROGRESS {count}/{total_games} games, {time.time()-started:.1f}s", flush=True)
            if len(slot["rows"]) == job["deals"]*2:
                row = summarize_pairs(slot["rows"], job["a"], job["b"], job["start_seed"], job.get("alpha", .05))
                row["name"] = job["name"]
                row["latency_s"] = {
                    side: dict(zip(("p50", "p95", "p99", "max"), np.quantile(t, [.5, .95, .99, 1]).tolist()))
                    for side, t in (("a", slot["at"]), ("b", slot["bt"]))
                }
                finished[i] = row
                receipt["matchups"] = [finished[k] for k in sorted(finished)]
                (out/"results.json").write_text(json.dumps(receipt, indent=2))
                print(f'DONE {job["name"]}: {row["win_rate_a"]:.3f}', flush=True)
    for path, expected in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest() != expected:
            raise ValueError(f"changed during panel: {path}")
    assert len(finished) == len(slots)
    receipt.update(completed=True, elapsed_s=time.time()-started)
    (out/"results.json").write_text(json.dumps(receipt, indent=2))
    print(f'COMPLETE {receipt["elapsed_s"]:.1f}s', flush=True)
    return receipt


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", required=True)
    ap.add_argument("--tag", required=True)
    ap.add_argument("--workers", type=int, default=12)
    args = ap.parse_args()
    run_panel(args.plan, ROOT/"results"/args.tag, args.workers)
