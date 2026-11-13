"""File contract of the notebooks. Notebook 01 writes three kinds of file and each later notebook reads only what it
is allowed to:
    data/measurements/  what a field team measures (no network, no parameters)  -> read by 02, 03, 04
    data/sketch/        the channel sketch: zero patterns only                   -> read by 02, 03, 04
    data/truth/         the networks W, P and the full matrices                  -> read by 04 only (scoring)
The plans, the data a plan provides and the lags from the sketch are in plans.py; the test is in certify.py."""
import numpy as np
import community as C
from plans import CANDIDATES, NAMES, SHARED, PLANS, plan, view, lags, tau0          # noqa: F401 (re-exported)

EXCITATIONS = {0: 'canal-turn deviation', 1: 'pest-pressure innovation'}


def save_measurements(path, D):
    """Write ONLY what a field team measures. D is the output of community.simulate; D['truth'] is dropped."""
    arr = {}
    for o in D['Y']:
        arr[f'Y_{o}'] = D['Y'][o]; arr[f'avail_{o}'] = D['avail'][o]
    for d in D['M']:
        arr[f'M_{d}'] = D['M'][d]; arr[f'Pi_{d}'] = D['Pi'][d]
    arr['sched'] = D['sched']
    for s, v in D['Omega'].items():
        arr[f'Omega_{s}'] = v
    np.savez_compressed(path, **arr)


def load_measurements(path):
    """The data dict that certify.py expects. Contains no network and no parameters."""
    z = np.load(path); keys = list(z.keys())
    assert not any(k in keys for k in ('W', 'P', 'Phi', 'Gamma')), 'truth found in a measurement file'
    R = dict(Y={}, avail={}, M={}, Pi={}, Omega={})
    for k in keys:
        if k.startswith('Y_'): R['Y'][k[2:]] = z[k]
        elif k.startswith('avail_'): R['avail'][k[6:]] = z[k]
        elif k.startswith('M_'): R['M'][int(k[2:])] = z[k]
        elif k.startswith('Pi_'): R['Pi'][int(k[3:])] = z[k]
        elif k.startswith('Omega_'): R['Omega'][k[6:]] = z[k]
    R['Omega'] = {s: R['Omega'][s] for s in SHARED if s in R['Omega']}        # keep the order of the regressors
    R['sched'] = z['sched']
    return R


def save_sketch(path, drop=()):
    """The channel sketch: WHICH entries of Phi_i, H_i, Gamma_i, E_i are non-zero. No values."""
    sk = C.sketch(drop); np.savez(path, channels=np.array(C.CH), **sk)


def load_sketch(path):
    z = np.load(path)
    return {k: z[k] for k in ('Phi', 'H', 'Gamma', 'E')}


def save_truth(path, Cm):
    np.savez(path, W=Cm['W'], P=Cm['P'], Phi=Cm['Phi'], Gamma=Cm['Gamma'], F=Cm['F'])


def load_truth(path):
    z = np.load(path)
    return {k: z[k] for k in z.keys()}
