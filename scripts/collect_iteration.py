"""E9: distill a frozen teacher from full matches, with separate player memory."""
from __future__ import annotations
import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import os
from pathlib import Path
import random
import sys
import time

os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
from trucoai.agents import make_agent
from trucoai.agents.learned import features
from trucoai.arena import EVAL_BASE
from trucoai.cfr.mccfr import abstract_legal, to_abstract
from trucoai.engine import new_match
from trucoai.observation import observe
from trucoai.rules import ONLINE_O1


def collect_match(args, factory=make_agent):
    teacher, opponent, seed = args
    seat = seed % 2
    agents = [factory(teacher), factory(opponent)]
    for i, agent in enumerate(agents):
        agent.reset(seed * 2 + i + 314159)
    seated = agents if seat == 0 else agents[::-1]
    state = new_match(ONLINE_O1, first_dealer=1)
    xs, ys, masks = [], [], []
    hands = 0
    while state.match_winner is None and hands < 200:
        state.deal(random.Random((EVAL_BASE + seed) * 1000 + hands))
        hands += 1
        while not state.is_terminal():
            p = state.current
            obs = observe(state, p)
            action = seated[p].act(obs)
            if action not in obs.legal_actions:
                raise ValueError('illegal collection action')
            if p == seat:
                xs.append(features(obs))
                ys.append(to_abstract(obs, action))
                masks.append([a in abstract_legal(obs) for a in range(9)])
            state.apply(action)
    if state.match_winner is None:
        raise ValueError('unfinished collection match')
    return {'x': np.asarray(xs, np.float32), 'y': np.asarray(ys, np.int64),
            'mask': np.asarray(masks, bool), 'group': np.full(len(ys), seed, np.int64),
            'receipt': {'seed': seed, 'teacher_seat': seat, 'opponent': opponent,
                        'hands': hands, 'teacher_won': state.match_winner == seat,
                        'scores': state.scores, 'completed': True}}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--manifest', required=True)
    ap.add_argument('--run-id', required=True)
    ap.add_argument('--games', type=int, default=512)
    ap.add_argument('--start-seed', type=int, default=600000)
    ap.add_argument('--workers', type=int, default=4)
    args = ap.parse_args()
    manifest = json.loads(Path(args.manifest).read_text(encoding='utf8'))
    for name, expected in manifest['sha256'].items():
        if hashlib.sha256((ROOT/name).read_bytes()).hexdigest() != expected:
            raise ValueError(f'teacher source/model changed: {name}')
    teacher = manifest['candidate']
    out = ROOT/'runs'/args.run_id
    out.mkdir(exist_ok=False)
    sources = list((ROOT/'trucoai').rglob('*.py')) + [Path(__file__)]
    config = {'args': vars(args), 'teacher': teacher, 'rules_fingerprint': ONLINE_O1.fingerprint,
              'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
              'teacher_manifest_sha256': hashlib.sha256(Path(args.manifest).read_bytes()).hexdigest(),
              'provenance': 'Synthetic full matches. Teacher labels only its own legal observations. Separate agent per seat; reset each match.',
              'split': 'Entire matches; seed modulo 10 == 0 validation. No evaluation seed reused.',
              'started': time.strftime('%Y-%m-%d %H:%M:%S')}
    (out/'manifest.json').write_text(json.dumps(config, indent=2), encoding='utf8')
    opponents = ['zap:v1', 'heuristic', 'malandro_rule', teacher]
    # Two seats for every opponent, rather than coupling style with seat parity.
    tasks = [(teacher, opponents[(i//2)%len(opponents)], args.start_seed+i) for i in range(args.games)]
    parts, receipts = [], []
    start = time.time()
    with ProcessPoolExecutor(args.workers) as pool:
        for i, part in enumerate(pool.map(collect_match, tasks)):
            receipts.append(part.pop('receipt'))
            parts.append(part)
            with (out/f'match_{i:04d}.npz').open('xb') as f:
                np.savez_compressed(f, **part)
            if (i+1)%16 == 0 or i+1 == len(tasks):
                print(f'{i+1}/{len(tasks)} matches, {sum(len(p["y"]) for p in parts)} decisions, {time.time()-start:.1f}s', flush=True)
    data = {key: np.concatenate([p[key] for p in parts]) for key in parts[0]}
    with (out/'data.npz').open('xb') as f:
        np.savez_compressed(f, **data)
    config.update({'completed': True, 'elapsed_s': time.time()-start, 'rows': len(data['y']),
                   'receipts': receipts, 'dataset_sha256': hashlib.sha256((out/'data.npz').read_bytes()).hexdigest()})
    (out/'manifest.json').write_text(json.dumps(config, indent=2), encoding='utf8')
    print(f'COMPLETE {out}', flush=True)


if __name__ == '__main__':
    main()
