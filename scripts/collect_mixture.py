"""Augment frozen Zap imitation data with independent reference-bot hands."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
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
import numpy as np
from trucoai.agents.learned import collect_chunk, combine_datasets
from trucoai.rules import ONLINE_O1


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--source', required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--hands', type=int, default=15000, help='Independent hands per reference bot')
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    out = ROOT / 'runs' / args.run_id
    out.mkdir(exist_ok=False)
    t0 = time.time()
    source = Path(args.source)
    with np.load(source) as z:
        parts = [{k: z[k].copy() for k in ('x', 'y', 'mask', 'group')}]
    manifest = {'args': vars(args), 'source_sha256': hashlib.sha256(source.read_bytes()).hexdigest(),
                'rules_fingerprint': ONLINE_O1.fingerprint, 'teachers': ['zap:v1', 'heuristic', 'malandro_rule'],
                'new_seed_ranges': [[400000, 400000+args.hands-1], [500000,500000+args.hands-1]],
                'hypothesis': 'A diverse opponent policy reduces exploitation-specific regression against non-Zap bots.'}
    tasks = [(spec, start+i, min(100,args.hands-i))
             for spec,start in [('heuristic',400000),('malandro_rule',500000)]
             for i in range(0,args.hands,100)]
    with ProcessPoolExecutor(args.workers) as ex:
        for j, part in enumerate(ex.map(collect_chunk, tasks)):
            parts.append(part)
            if (j+1)%20==0:
                print(f'new hands {(j+1)*100}/{2*args.hands} elapsed {time.time()-t0:.1f}s',flush=True)
    data = combine_datasets(parts)
    with (out/'data.npz').open('xb') as f:
        np.savez_compressed(f, **data)
    manifest.update({'rows':len(data['y']), 'elapsed_s':time.time()-t0,
                     'sha256':hashlib.sha256((out/'data.npz').read_bytes()).hexdigest()})
    (out/'manifest.json').write_text(json.dumps(manifest,indent=2),encoding='utf8')
    print(f'SAVED {out} rows={len(data["y"])}',flush=True)


if __name__ == '__main__':
    main()
