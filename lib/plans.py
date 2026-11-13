"""Measurement plans for the farming community (Section 4, Table 3), the data a plan provides, and the lags that
the test reads from the channel sketch (Definition 1, Lemma 1, Step 1). Shared by the paper runs and the notebooks."""
import numpy as np
import community as C

CANDIDATES = ['sm', 'gw', 'bel', 'pump', 'spr', 'temp', 'pr']      # channels that can be measured ('talk' never is)
NAMES = dict(sm='soil moisture', gw='GW level', bel='belief survey', pump='pumping', spr='spraying',
             temp='canopy temp. (decoy)', pr='price survey (decoy)')
SHARED = list(C.SHARED)                                             # shared series used as regressors

# A plan M lists the excitations d, the outputs o and the shared series that are measured. 'farms_out' limits an
# output to some farms.
PLANS = {
    'SM':                    dict(excitations=[0], outputs=['sm']),
    'GW level':              dict(excitations=[0], outputs=['gw']),
    'Pump':                  dict(excitations=[0], outputs=['pump']),
    'SM+Pump':               dict(excitations=[0], outputs=['sm', 'pump']),
    'SM+Pump(4 farms)':      dict(excitations=[0], outputs=['sm', 'pump'], farms_out={'pump': [0, 2, 4, 6]}),
    'Survey':                dict(excitations=[0], outputs=['bel']),
    'Spray+Pest':            dict(excitations=[1], outputs=['spr']),
    'Pump+Spray':            dict(excitations=[0, 1], outputs=['pump', 'spr']),
    'All practical':         dict(excitations=[0, 1], outputs=['sm', 'gw', 'pump', 'spr', 'bel']),
    'Pump+Spray, no shared': dict(excitations=[0, 1], outputs=['pump', 'spr'], shared=[]),
    'Decoys':                dict(excitations=[0, 1], outputs=['temp', 'pr']),
}


def plan(name):
    """The plan with its shared series filled in."""
    p = dict(PLANS[name]); p.setdefault('shared', SHARED); return p


def view(D, T, outputs, excitations, shared=SHARED, farms_out=None, keep=None):
    """The data that a plan provides: the first T days of the chosen outputs, excitations and shared series.
    farms_out[o] = farms at which output o is measured; keep = farms that are measured at all."""
    N = D['Y'][outputs[0]].shape[1]
    keep = list(range(N)) if keep is None else keep
    Y = {}; avail = {}
    for o in outputs:
        av = D['avail'][o][:T][:, keep].copy()
        if farms_out and o in farms_out:
            av[:, [k for k, f in enumerate(keep) if f not in farms_out[o]]] = False
        Y[o] = np.where(av, D['Y'][o][:T][:, keep], 0.0); avail[o] = av
    return dict(Y=Y, avail=avail, M={d: D['M'][d][:T][:, keep] for d in excitations},
                Pi={d: D['Pi'][d][:T][:, keep] for d in excitations},
                sched=D['sched'][:T][:, keep], Omega={s: D['Omega'][s][:T] for s in shared})


def plan_view(D, T, name, keep=None):
    p = plan(name)
    return view(D, T, p['outputs'], p['excitations'], p['shared'], p.get('farms_out'), keep)


def lags(sk, physical=False, network='W', detail=False):
    """[Lemma 1] The lag tau_{d,o} = tau^S tested for each (excitation d, output o), from the sketch alone. A pair is
    kept only if network S arrives first: before the other network and before any chain with two or more
    crossings. With detail=True, also returns the first lags of Definition 1 for every pair."""
    other = 'P' if network == 'W' else 'W'
    tau, det = {}, {}
    for d in C.EXC:
        for o in CANDIDATES:
            a = C.first_lags(sk, d, C.IX[o], physical); det[(d, o)] = a
            others = [x for x in (a[other], a['tau2']) if x]
            if a[network] and (not others or a[network] < min(others)):
                tau[(d, o)] = a[network]
    return (tau, det) if detail else tau


def tau0(sk, physical=False, network='W'):
    """[Step 1] tau^0_c for each (excitation d, channel c): the first lag at which channel c can respond to j's
    excitation without a link of the tested network, i.e. through the other network or a chain with two or more
    crossings. None = never. The receiver's own data on c are used as regressors only before this lag."""
    other = 'P' if network == 'W' else 'W'; r = {}
    for d in C.EXC:
        for c in CANDIDATES:
            a = C.first_lags(sk, d, C.IX[c], physical)
            others = [x for x in (a[other], a['tau2']) if x]
            r[(d, c)] = min(others) if others else None
    return r
