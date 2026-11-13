"""Score the saved runs against the true networks and compute every number of Section 4, Appendix C and
Tables 2-4. Reads results/*.pkl, writes results/farm_summary.json. Usage: python farm_analyze.py"""
import os, sys, json, pickle
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'lib'))
import community as C
from certify import threshold
from scoring import score, predicted_found

RES = os.path.join(HERE, 'results'); N = 8; ALPHA = 0.1
OFF = ~np.eye(N, dtype=bool)
WARM = 90 + 14                                    # days before the first term of the statistic (certify.py)

load = lambda name: pickle.load(open(os.path.join(RES, name), 'rb'))
mean = lambda x: float(np.mean(x)) if len(x) else float('nan')


def summarise(rows):
    """Mean over communities of each score; FWER is the share of communities with any false link."""
    s = [score(W_hat, W) for W_hat, W in rows]
    return dict(runs=len(s), FWER=mean([x['any_false'] for x in s]),
                **{k: mean([x[k] for x in s]) for k in ('FP', 'precision', 'recall', 'F1', 'FDR')})


def two_or_three_step(W):
    """Pairs (i, j) without a link that are joined by a directed path j -> l -> i or j -> l -> l' -> i."""
    A = ((W > 0) & OFF).astype(int); A2 = (A @ A) > 0; A3 = (A @ A @ A) > 0
    return (A2 | A3) & (W == 0) & OFF


out = {}
# ---- measurement plans (Table 3, Figure 1a)
R = load('farm_plans.pkl'); plans = list(dict.fromkeys(r['plan'] for r in R)); years = sorted({r['yrs'] for r in R})
tab = []
for p in plans:
    for y in years:
        rs = [r for r in R if r['plan'] == p and r['yrs'] == y]
        s = summarise([(r['W_hat'], r['W']) for r in rs])
        s['predicted'] = mean([predicted_found(r['fam'])[r['W'] > 0].mean() for r in rs]) if rs[0]['fam'] else 0.0
        tab.append(dict(plan=p, yrs=y, **s))
    # predicted years to find 90% of the links: scale the 10-year runs' event counts
    r10 = [r for r in R if r['plan'] == p and r['yrs'] == max(years)]; need = None
    if r10[0]['fam']:
        n10 = max(years) * 365 - WARM
        for yy in np.concatenate([np.arange(0.5, 10, 0.25), np.arange(10, 200.1, 2.0)]):
            sc = max(yy * 365 - WARM, 1) / n10
            if mean([predicted_found(r['fam'], sc)[r['W'] > 0].mean() for r in r10]) >= 0.9:
                need = float(yy); break
    tab.append(dict(plan=p, yrs='T90', predicted_years=need))
out['plans'] = tab
out['plans_total'] = dict(runs=len(R), runs_with_false=int(sum(score(r['W_hat'], r['W'])['any_false'] for r in R)))
gaps = [abs(t['recall'] - t['predicted']) for t in tab if t['yrs'] != 'T90']
out['prediction_gap_max'] = float(max(gaps))
out['links_per_community'] = mean([int(((r['W'] > 0) & OFF).sum()) for r in R if r['plan'] == plans[0] and r['yrs'] == years[0]])

# ---- checks: placebo, lag scan, baselines (Tables 2 and 4)
R = load('farm_checks.pkl')
Zp = np.concatenate([r['Z_placebo'] for r in R])
out['placebo'] = dict(n=int(Zp.size), mean=float(Zp.mean()), sd=float(Zp.std()), max_abs=float(np.abs(Zp).max()))
out['c_alpha_56_pairs'] = threshold(N * (N - 1), ALPHA)[0]
out['scan'] = [dict(yrs=y, **summarise([(r['scan'], r['W']) for r in R if r['yrs'] == y]),
                    lags={str(k): dict(zip(*map(lambda a: a.tolist(), np.unique([str(r['scan_lag'][k]) for r in R if r['yrs'] == y], return_counts=True))))
                          for k in R[0]['scan_lag']}) for y in (1, 2, 3, 5)]
out['scan_total'] = dict(runs=len(R), runs_with_false=int(sum(score(r['scan'], r['W'])['any_false'] for r in R)))
out['baselines'] = [dict(yrs=y, method=m, **summarise([(r['base'][m], r['W']) for r in R if r['yrs'] == y and 'base' in r]))
                    for y in (3, 5) for m in ('dmdc', 'output_granger', 'lasso_granger')]
R = load('farm_ours_base_data.pkl')                # our test on the same data as the baselines
out['baselines'] += [dict(yrs=y, method='ours', **summarise([(r['W_hat'], r['W']) for r in R if r['yrs'] == y])) for y in (3, 5)]
O = {(r['seed'], r['yrs']): r['W_hat'] for r in R}       # our test with the survey (ours_base_data)
R4 = load('farm_ours_daily_outputs.pkl')
out['ours_without_survey'] = dict(runs=len(R4), identical_reports=int(sum(np.array_equal(r['W_hat'], O[(r['seed'], r['yrs'])]) for r in R4)))
R = load('farm_cgranger.pkl')
out['baselines'] += [dict(yrs=y, method='conditional_granger', **summarise([(r['W_hat'], r['W']) for r in R if r['yrs'] == y]))
                     for y in (3, 5)]
fl = sum(int((r['W_hat'] & (r['W'] == 0) & OFF).sum()) for r in R)
fl23 = sum(int((r['W_hat'] & two_or_three_step(r['W'])).sum()) for r in R)
out['cgranger_false_on_2_3_step_paths'] = dict(false_links=fl, on_paths=fl23, share=fl23 / fl)

# ---- channel certificates (Figure 1b)
R = load('farm_channels.pkl')
reached = {(0, 'sm'), (0, 'gw'), (0, 'bel'), (0, 'pump'), (1, 'spr')}      # from the sketch (Lemma 1)
out['channels'] = [dict(yrs=y, excitation=C.EXC[d], output=o, reached=(d, o) in reached,
                        certified=mean([r['cert'][(d, o)] is not None for r in R if r['yrs'] == y]))
                   for y in (1, 2, 3, 5) for (d, o) in R[0]['cert']]

# ---- wrong lag vs true lag vs lag scan (Table 4)
R = load('farm_wrong_lag.pkl')
out['wrong_lag'] = [dict(yrs=y, wrong=summarise([(r['wrong'], r['W']) for r in R if r['yrs'] == y]),
                         true=summarise([(r['true'], r['W']) for r in R if r['yrs'] == y]),
                         scan=summarise([(r['scan'], r['W']) for r in R if r['yrs'] == y])) for y in (3, 5, 10)]
wf = sum(int((r['wrong'] & (r['W'] == 0) & OFF).sum()) for r in R)
wf2 = sum(int((r['wrong'] & ((((r['W'] > 0) & OFF).astype(int) @ ((r['W'] > 0) & OFF).astype(int)) > 0) & (r['W'] == 0) & OFF).sum()) for r in R)
out['wrong_lag_false_links'] = dict(false_links=wf, on_two_step_paths=wf2)
out['wrong_lag_scan'] = dict(runs=len(R), runs_with_false=int(sum(score(r['scan'], r['W'])['any_false'] for r in R)),
                             lags=sorted({str(r['lag']) for r in R}))

# ---- local aquifer (Table 4)
R = load('farm_physical.pkl')
# With seepage the social and physical networks reach pumping at the same lag, so there is no social test.
out['physical'] = [dict(seep=sp, yrs=y, lag_physical=rs[0]['lag_physical'], lag_social=rs[0].get('lag_social'),
                        social=summarise([(r['social'], r['W']) for r in rs if 'social' in r]) if 'social' in rs[0] else None,
                        social_false_on_physical_neighbours=mean([int((r['social'] & (r['W'] == 0) & (r['P'] > 0)).sum())
                                                                  for r in rs if 'social' in r]) if 'social' in rs[0] else None,
                        physical=summarise([(r['physical'], r['P']) for r in rs]))
                   for sp in (0.0, 0.5) for y in (3, 5) for rs in [[r for r in R if r['seep'] == sp and r['yrs'] == y]]]

# ---- two farms not measured (Table 4)
R = load('farm_hidden.pkl')
sub = lambda r, M: M[np.ix_(r['keep'], r['keep'])]


def via_hidden(r):
    """Measured pairs without a link that are joined by a 2- or 3-step path through an unmeasured farm."""
    W = r['W']; A = ((W > 0) & OFF).astype(int); h = [k for k in range(N) if k not in r['keep']]
    Ah = A[:, h] @ A[h, :]                                       # j -> hidden -> i
    through = (Ah > 0) | ((A @ Ah) > 0) | ((Ah @ A) > 0)
    return int(sub(r, through & (W == 0) & OFF).sum())


out['hidden'] = [dict(yrs=y, **summarise([(r['W_hat'], sub(r, r['W'])) for r in rs]),
                      indirect_pairs_via_hidden=mean([via_hidden(r) for r in rs]))
                 for y in (3, 5) for rs in [[r for r in R if r['yrs'] == y]]]

# ---- pooled residual correlation (Section 4)
out['pool_corr'] = json.load(open(os.path.join(RES, 'pool_corr.json')))

# ---- largest spectral radius of Phi over 300 communities, regional and local aquifer (Appendix C: 'at most 0.97')
out['max_spectral_radius_300'] = float(max(np.abs(np.linalg.eigvals(C.build(N, s)['Phi'])).max() for s in range(300)))
out['max_spectral_radius_300_local'] = float(max(np.abs(np.linalg.eigvals(C.build(N, s, scenario='local')['Phi'])).max()
                                            for s in range(300)))

json.dump(out, open(os.path.join(RES, 'farm_summary.json'), 'w'), indent=1, default=str)
print('wrote results/farm_summary.json')
