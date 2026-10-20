"""Full-match public E9 teacher collection via the explicit portfolio loader."""
import os
os.environ.setdefault('OPENBLAS_NUM_THREADS', '1')
os.environ.setdefault('OMP_NUM_THREADS', '1')
import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from scripts import collect_iteration
from scripts.e10_factory import factory

original_collect = collect_iteration.collect_match
def collect(args):
    return original_collect(args, factory=factory)

collect_iteration.collect_match = collect
if __name__ == '__main__':
    collect_iteration.main()
