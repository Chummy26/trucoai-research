"""Small balanced batches, raw decisions' timings, frozen sources and weights."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
from pathlib import Path
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from dataclasses import asdict
import hashlib
import json
import time
import numpy as np
from scripts.e10_factory import factory as old_factory
from scripts.evaluate_frozen import summarize_pairs
from trucoai.arena import play_match
from trucoai.agents.bayes_response import BayesResponseAgent
from trucoai.agents.learned import PolicyNetwork
from trucoai.agents.fast_network import accelerate
from trucoai.rules import PLAYTRUCO_O1


def factory(spec):
    if not spec.startswith('bayes:'):
        return accelerate(old_factory(spec))
    path = Path(spec.partition(':')[2])
    cfg = json.loads(path.read_bytes())
    agent = BayesResponseAgent([old_factory(s) for s in cfg['models']],
        PolicyNetwork(cfg['own_network']), n=cfg.get('n', 48), k=cfg.get('k', 2),
        own_greedy=cfg.get('own_greedy', False), forgetting=cfg.get('forgetting', .02))
    agent.meta.update(config_sha256=hashlib.sha256(path.read_bytes()).hexdigest(), config=cfg)
    return accelerate(agent)


class Timed:
    def __init__(self, agent):
        self.agent = agent
        self.times = []
    def reset(self, seed):
        self.agent.reset(seed)
    def act(self, obs):
        start = time.perf_counter()
        action = self.agent.act(obs)
        self.times.append(time.perf_counter()-start)
        return action


_CACHE = {}
def batch(task):
    a, b, seeds = task
    for spec in (a, b):
        if spec not in _CACHE:
            _CACHE[spec] = factory(spec)
    # Independent memory even for a self-match.
    aa = Timed(_CACHE[a])
    bb = Timed(factory(b) if a == b else _CACHE[b])
    rows = [play_match(aa, bb, seed, seat, rules=PLAYTRUCO_O1)
            for seed in seeds for seat in (0, 1)]
    return rows, aa.times, bb.times


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    plan = json.loads(Path(args.plan).read_bytes())
    out = ROOT/'results'/args.tag
    out.mkdir(exist_ok=False)
    files = (list((ROOT/'trucoai').rglob('*.py')) + list((ROOT/'scripts').glob('*.py'))
             + [ROOT/p for p in plan.get('artifacts', [])] + [Path(args.plan).resolve()])
    hashes = {str(p.resolve()): hashlib.sha256(p.read_bytes()).hexdigest() for p in files}
    receipt = dict(plan=plan, source_and_artifact_sha256=hashes, completed=False,
                   rules_fingerprint=PLAYTRUCO_O1.fingerprint, workers=args.workers,
                   loaded_agents={s: getattr(factory(s), 'meta', {})
                                  for j in plan['jobs'] for s in (j['a'],j['b'])},
                   matchups=[])
    (out/'manifest.json').write_text(json.dumps(receipt, indent=2))
    started = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        for ji, job in enumerate(plan['jobs']):
            rows, at, bt = [], [], []
            start, count = job['start_seed'], job['deals']
            print(f'START {job["name"]}: {count*2} games', flush=True)
            tasks = [(job['a'], job['b'], list(range(s, min(s+2,start+count))))
                     for s in range(start,start+count,2)]
            with (out/f'job_{ji}.jsonl').open('x') as f:
                for future in as_completed([pool.submit(batch,t) for t in tasks]):
                    chunk, ats, bts = future.result()
                    rows.extend(chunk); at.extend(ats); bt.extend(bts)
                    f.write(json.dumps(dict(rows=[asdict(r) for r in chunk], a_times=ats,b_times=bts))+'\n')
                    f.flush()
                    if len(rows)%40==0 or len(rows)==count*2:
                        print(f'{job["name"]} {len(rows)}/{count*2}, {time.time()-started:.1f}s',flush=True)
            result = summarize_pairs(rows, job['a'],job['b'],start,job.get('alpha',.05))
            result['name'] = job['name']
            result['latency_s'] = {
                label: dict(zip(('p50','p95','p99','max'), np.quantile(ts,[.5,.95,.99,1]).tolist()))
                for label,ts in [('a',at),('b',bt)]}
            receipt['matchups'].append(result)
            (out/'results.json').write_text(json.dumps(receipt,indent=2))
            print(f'DONE {job["name"]}: {result["win_rate_a"]:.3f} CI={result["stats"]["cluster95"]}',flush=True)
    for path, expected in hashes.items():
        if hashlib.sha256(Path(path).read_bytes()).hexdigest()!=expected:
            raise ValueError(f'changed during evaluation: {path}')
    receipt.update(completed=True,elapsed_s=time.time()-started)
    (out/'results.json').write_text(json.dumps(receipt,indent=2))
    print(f'COMPLETE {time.time()-started:.1f}s',flush=True)


if __name__=='__main__':
    main()
