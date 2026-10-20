"""Load an explicit public-style portfolio; ordinary specs keep their meaning."""
import hashlib
import json
from pathlib import Path
from scripts.evaluate_iteration import factory as leaf_factory
from trucoai.agents.portfolio import PortfolioAgent


def factory(spec):
    kind, _, arg = spec.partition(':')
    if kind != 'portfolio':
        return leaf_factory(spec)
    path = Path(arg).resolve()
    raw = path.read_bytes()
    config = json.loads(raw)
    cache = {}

    def load(leaf):
        if leaf.startswith('portfolio:'):
            raise ValueError('nested portfolio is not supported')
        if leaf not in cache:
            cache[leaf] = leaf_factory(leaf)
        return cache[leaf]

    agent = PortfolioAgent([load(s) for s in config['models']],
                           [load(s) for s in config['policies']], load(config['fallback']),
                           style_n=config['style_n'], threshold=config['threshold'])
    agent.meta.update({'portfolio_file': str(path), 'portfolio_sha256': hashlib.sha256(raw).hexdigest(),
                       'model_specs': config['models'], 'policy_specs': config['policies']})
    return agent
