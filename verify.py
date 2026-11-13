"""Reproducibility check: rerun community 0 of every part of the farm study and compare, entry by entry, with the
saved results in paper_runs/farm/results/ that Tables 2-4 and Figures 1 and 3 are computed from. About 1-2 minutes.
Usage: python verify.py"""
import os, sys, pickle, time, warnings
import numpy as np
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, 'paper_runs', 'farm'))
import farm_run as F

RES = os.path.join(HERE, 'paper_runs', 'farm', 'results')
saved = lambda name: [r for r in pickle.load(open(os.path.join(RES, f'farm_{name}.pkl'), 'rb')) if r['seed'] == 0]


def same(a, b):
    if isinstance(a, dict):
        return a.keys() == b.keys() and all(same(a[k], b[k]) for k in a)
    if isinstance(a, np.ndarray):
        return a.shape == b.shape and np.array_equal(np.nan_to_num(a, nan=-9e9), np.nan_to_num(b, nan=-9e9))
    if isinstance(a, float) and isinstance(b, float) and np.isnan(a) and np.isnan(b):
        return True
    return a == b


ok = True; t0 = time.time()
for part in ('plans', 'checks', 'channels', 'wrong_lag', 'physical', 'hidden', 'ours_base_data', 'ours_daily_outputs', 'cgranger'):
    new = F.PARTS[part](0); old = saved(part)
    good = len(new) == len(old) and all(same(a, b) for a, b in zip(new, old))
    ok &= good
    print(f'{part:15s} {len(new):3d} runs  {"identical" if good else "DIFFERENT"}   ({time.time() - t0:.0f} s)', flush=True)
print('all identical to the saved results' if ok else 'MISMATCH: see above')
sys.exit(0 if ok else 1)
