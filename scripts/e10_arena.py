"""Reuse frozen evaluation with E10 factory. Small mirrored chunks for scouts."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import evaluate_frozen
from scripts.e10_factory import factory

evaluate_frozen.make_agent = factory
if __name__ == '__main__':
    evaluate_frozen.main()
