"""E10 experiment-only loader. Existing registry and frozen agents unchanged."""
import hashlib
import json
from pathlib import Path
from scripts.portfolio_factory import factory as old_factory
from trucoai.agents.refined import RefinedSearchAgent
from trucoai.agents.portfolio import PortfolioAgent


def factory(spec):
    kind, _, arg = spec.partition(':')
    if kind == 'e10portfolio':
        raw = Path(arg).read_bytes()
        cfg = json.loads(raw)
        cache = {}
        def load(s):
            if s not in cache:
                cache[s] = factory(s)
            return cache[s]
        agent = PortfolioAgent([load(s) for s in cfg['models']],
                               [load(s) for s in cfg['policies']], load(cfg['fallback']),
                               style_n=cfg['style_n'], threshold=cfg['threshold'])
        agent.meta.update(config_sha256=hashlib.sha256(raw).hexdigest(), config=cfg)
        return agent
    if kind == 'refined':
        path, _, query = arg.partition('?')
        opts = dict(x.split('=', 1) for x in query.split('&') if x)
        return RefinedSearchAgent(path, opts['net'],
            neural_bets=opts.get('nbets', '0') == '1',
            n=int(opts.get('n', 48)), k=int(opts.get('k', 4)),
            belief=opts.get('belief', '1') == '1', belief_cards=True,
            temper=float(opts.get('temper', 1)), opp_mix=0, bets_from_blueprint=False,
            tail=opts.get('tail', '1') == '1', own_greedy=opts.get('own_greedy', '0') == '1')
    return old_factory(spec)
