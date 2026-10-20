"""Explicit research loader for a payoff portfolio; reuse the paired arena."""
import os
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("OMP_NUM_THREADS", "1")
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import hashlib
import json
from scripts import e11_rapid, e10_rapid
from trucoai.agents.payoff_portfolio import PayoffPortfolioAgent
from trucoai.agents.fast_network import accelerate

previous_factory = e11_rapid.factory


def factory(spec):
    if not spec.startswith("payoff:"):
        return previous_factory(spec)
    raw = Path(spec.partition(":")[2]).read_bytes()
    config = json.loads(raw)
    cache = {}
    def load(s):
        if s not in cache:
            cache[s] = previous_factory(s)
        return cache[s]
    agent = PayoffPortfolioAgent(
        [load(s) for s in config["models"]],
        [load(s) for s in config["policies"]],
        config["payoff"], style_n=config.get("style_n", 48),
        forgetting=config.get("forgetting", .02),
    )
    agent.meta.update(config_sha256=hashlib.sha256(raw).hexdigest(), config=config)
    return accelerate(agent)


e10_rapid.factory = factory
if __name__ == "__main__":
    e10_rapid.main()
