"""Use the frozen paired evaluator with an explicit portfolio agent factory.

The injected factory is passed to workers as a picklable module function.
The original evaluator and its ordinary entrypoint remain byte-identical.
Manifests include hashes of this entrypoint and portfolio_factory.py as well.
"""
from pathlib import Path
import sys
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import evaluate_frozen
from scripts.portfolio_factory import factory


if __name__ == '__main__':
    evaluate_frozen.make_agent = factory
    evaluate_frozen.main()
