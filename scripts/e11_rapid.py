"""Grouped-family policy selection, using the same frozen rapid arena."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS','1')
os.environ.setdefault('OMP_NUM_THREADS','1')
import sys
from pathlib import Path
sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
import json
import hashlib
from scripts import e10_rapid
from trucoai.agents.grouped_portfolio import GroupedPortfolioAgent
from trucoai.agents.fast_network import accelerate

previous_factory=e10_rapid.factory
def factory(spec):
    if not spec.startswith('grouped:'):
        return previous_factory(spec)
    path=Path(spec.partition(':')[2])
    raw=path.read_bytes();cfg=json.loads(raw);cache={}
    def load(s):
        if s not in cache:
            cache[s]=previous_factory(s)
        return cache[s]
    agent=GroupedPortfolioAgent([load(s) for s in cfg['models']],
         [load(s) for s in cfg['policies']],load(cfg['fallback']),
         style_n=cfg['style_n'],threshold=cfg['threshold'])
    agent.meta.update(config=cfg,config_sha256=hashlib.sha256(raw).hexdigest(),
                      selection='sum posterior of all models with identical response policy')
    return accelerate(agent)

e10_rapid.factory=factory
if __name__=='__main__':
    e10_rapid.main()
