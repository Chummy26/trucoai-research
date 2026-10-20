"""Evaluate a pre-frozen manifest; report progress without adaptive stopping."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
from dataclasses import asdict
import hashlib
import json
import os
from pathlib import Path
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.compare_references import pair_stats
from trucoai.agents import make_agent
from trucoai.arena import _chunk, summarize
from trucoai.rules import PLAYTRUCO_O1


def summarize_pairs(rows, a, b, start_seed, alpha=0.05):
    groups = {}
    for r in rows:
        if max(r.a_points, r.b_points) < PLAYTRUCO_O1.target_score:
            raise ValueError('unfinished match cannot be scored as a loss')
        groups.setdefault(r.seed, []).append(r)
    if not groups or any(sorted(r.a_seat for r in g) != [0, 1] for g in groups.values()):
        raise ValueError('incomplete or duplicate mirrored pair')
    if sorted(groups) != list(range(start_seed, start_seed + len(groups))):
        raise ValueError('missing mirrored pair seed')
    result = summarize(rows)
    result.update({'a_metrics': result.pop('a'), 'b_metrics': result.pop('b')})
    result.update({'a': a, 'b': b, 'deals': len(groups), 'start_seed': start_seed,
                   'deal_wins': [sum(r.a_won for r in groups[s]) for s in sorted(groups)],
                   'rules_version': PLAYTRUCO_O1.version,
                   'rules_fingerprint': PLAYTRUCO_O1.fingerprint})
    result['stats'] = pair_stats(result, alpha)
    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--workers', type=int, default=10)
    args = ap.parse_args()
    manifest_path = Path(args.manifest)
    manifest_text = manifest_path.read_text(encoding='utf8')
    manifest = json.loads(manifest_text)
    if manifest['rules_fingerprint'] != PLAYTRUCO_O1.fingerprint:
        raise ValueError('rules fingerprint mismatch')

    def verify_hashes():
        for name, expected in manifest['sha256'].items():
            actual = hashlib.sha256((ROOT/name).read_bytes()).hexdigest()
            if actual != expected:
                raise ValueError(f'frozen source/model changed: {name}')

    verify_hashes()
    out = ROOT/'results'/args.tag
    out.mkdir(exist_ok=False)
    (out/'manifest.json').write_text(manifest_text,encoding='utf8')
    results = []
    start = time.time()
    with ProcessPoolExecutor(args.workers) as ex:
        for j, job in enumerate(manifest['jobs']):
            count, seed0 = job['deals'], job['start_seed']
            tasks = [(make_agent, job['a'], job['b'], list(range(s,min(s+10,seed0+count))), PLAYTRUCO_O1)
                     for s in range(seed0,seed0+count,10)]
            rows = []
            print(f'job {j+1}/{len(manifest["jobs"])}: {count*2} games versus {job["b"]}',flush=True)
            with (out/f'job_{j}_pairs.jsonl').open('x',encoding='utf8') as raw:
                for batch in ex.map(_chunk,tasks):
                    rows.extend(batch)
                    raw.write(json.dumps([asdict(r) for r in batch])+'\n')
                    raw.flush()
                    if len(rows)%100==0 or len(rows)==2*count:
                        print(f'job {j+1}: {len(rows)}/{count*2} games, elapsed {time.time()-start:.1f}s',flush=True)
            result = summarize_pairs(rows,job['a'],job['b'],seed0,job.get('alpha',0.05))
            results.append(result)
            (out/'results.json').write_text(json.dumps({'manifest':manifest,'matchups':results,
                                                       'completed':False},indent=2),encoding='utf8')
            print(f'job {j+1} finished, win_rate={result["win_rate_a"]:.4f}, cluster95={result["stats"]["cluster95"]}',flush=True)
    verify_hashes()
    (out/'results.json').write_text(json.dumps({'manifest':manifest,'matchups':results,'completed':True,
                                               'elapsed_s':time.time()-start,'workers':args.workers},indent=2),encoding='utf8')
    print(f'COMPLETE {out}',flush=True)


if __name__=='__main__':
    main()
