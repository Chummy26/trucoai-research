"""Development arena for E9; explicit candidate specs and immutable output."""
from __future__ import annotations
import argparse
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
from trucoai.agents import make_agent
from trucoai.agents.iteration import AsymmetricSearchAgent
from trucoai.arena import run_matchup
from trucoai.rules import ONLINE_O1
from scripts.compare_references import pair_stats


def factory(spec):
    kind, _, arg = spec.partition(':')
    if kind != 'asymmetric':
        return make_agent(spec)
    path, _, query = arg.partition('?')
    opts = dict(x.split('=', 1) for x in query.split('&') if x)
    return AsymmetricSearchAgent(path, opts['net'], opts['oppnet'], neural_bets=True,
                                 n=int(opts.get('n', 48)), k=int(opts.get('k', 4)),
                                 belief=opts.get('belief', '1') == '1', belief_cards=True,
                                 temper=float(opts.get('temper', 1)), opp_mix=0,
                                 bets_from_blueprint=False)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument('--plan', required=True)
    ap.add_argument('--tag', required=True)
    ap.add_argument('--workers', type=int, default=8)
    args = ap.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding='utf8'))
    out = ROOT/'results'/args.tag
    out.mkdir(exist_ok=False)
    sources = list((ROOT/'trucoai').rglob('*.py')) + [Path(__file__)]
    receipt = {'plan': plan, 'workers': args.workers, 'completed': False,
               'rules_fingerprint': ONLINE_O1.fingerprint,
               'source_sha256': {str(p.relative_to(ROOT)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
               'matchups': [], 'loaded_agents': {}}
    start = time.time()
    for job in plan['jobs']:
        for spec in (job['a'], job['b']):
            if spec not in receipt['loaded_agents']:
                receipt['loaded_agents'][spec] = getattr(factory(spec), 'meta', {})
        print(f'START {job.get("name", "matchup")} {job["deals"]*2} games', flush=True)
        result = run_matchup(factory, job['a'], job['b'], deals=job['deals'],
                             start_seed=job['start_seed'], workers=args.workers, rules=ONLINE_O1)
        result['stats'] = pair_stats(result, 0.05)
        result['name'] = job.get('name', '')
        receipt['matchups'].append(result)
        (out/'results.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
        print(f'FINISHED {result["name"]}: {result["win_rate_a"]:.4f} {result["stats"]["cluster95"]} elapsed={time.time()-start:.1f}s', flush=True)
    receipt.update({'completed': True, 'elapsed_s': time.time()-start})
    (out/'results.json').write_text(json.dumps(receipt, indent=2), encoding='utf8')
    print(f'COMPLETE {out}', flush=True)


if __name__ == '__main__':
    main()
