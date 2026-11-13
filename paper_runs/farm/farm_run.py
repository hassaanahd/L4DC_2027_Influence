"""Simulation study of Section 4 and Appendix C: 40 communities of 8 farms on one watercourse.

Usage (from this folder):  python farm_run.py <part> [--communities 40]
    plans           11 measurement plans x durations 1, 2, 3, 5, 10 years (Table 3, Figure 1a)        ~6 min
    checks          placebo statistics, lag scan, DMDc / output Granger / Lasso-Granger (Tables 2, 4)  ~5 min
    channels        channel certificates from pilots of 1-5 years (Figure 1b)                          ~25 min
    wrong_lag       pest -> spraying tested at a wrong lag (3) vs the true lag (2) and the lag scan     ~1 min
    physical        local aquifer: social links via pumping, physical links via GW level (Table 4)     ~1 min
                    (lags from the sketch of each variant; with seepage pumping is not tested)
    hidden          two farms are not measured (Table 4)                                                ~1 min
    ours_base_data  our test on the same data that the baselines receive (Table 2, Figure 3)          ~1 min
    ours_daily_outputs  the same without the weekly survey (Section 4: identical reports)          ~1 min
    cgranger        conditional Granger baseline (Table 2)                                              ~2 min
    pool_corr       correlation of the residuals pooled by the SM+Pump plan (Section 4)                 ~1 min
Each part writes results/farm_<part>.pkl (pool_corr writes results/pool_corr.json).

The test sees only the data that a plan provides (plans.view) and the lags read from the channel sketch. W, P and
all parameter values are used only to simulate and, afterwards, to score (farm_analyze.py). Community s is built
with seed s and simulated with seed 1000 + s."""
import sys, os, time, pickle, json, argparse
import numpy as np
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', '..', 'lib'))
import community as C, certify as K, plans as PL

N = 8; NCOM = 40; YEARS = (1, 2, 3, 5, 10); ALPHA = 0.1
SK = C.sketch()                                   # the channel sketch: zero pattern only
TAU = PL.lags(SK)                                 # tau_{d,o} for the social network (regional scenario)
TAU0 = PL.tau0(SK)                                # tau^0_c for the regressors of Step 1
BASE_OUTS = ['sm', 'gw', 'pump', 'spr']           # outputs given to the baselines (Appendix C)
RES = os.path.join(HERE, 'results')


def run_plan(Cm, D, T, name, keep=None):
    """Certify one plan on the first T days. Besides the reported links, keeps what eq. (7) needs to predict power:
    the tested lags, event counts n_j, residual spreads, c_alpha and the TRUE impulse responses (for scoring only)."""
    R = PL.plan_view(D, T, name, keep)
    W_hat, Zs, infos = K.certify_plan(R, PL.plan(name), TAU, ALPHA, tau0=TAU0)
    fam = {d: dict(taus=inf['taus'], n_events=inf['n_events'], sd=inf['sd'], c_alpha=inf['c_alpha'],
                   h={o: C.impulse_response(Cm, d, o, t) for o, t in inf['taus'].items()})
           for d, inf in infos.items()}
    return W_hat, Zs, fam, R


def community(s, years, **kw):
    Cm = C.build(N, s, **kw); return Cm, C.simulate(Cm, years * 365, 1000 + s)


def part_plans(s):
    Cm, D = community(s, max(YEARS)); out = []
    for yrs in YEARS:
        for name in PL.PLANS:
            W_hat, Zs, fam, _ = run_plan(Cm, D, yrs * 365, name)
            out.append(dict(seed=s, yrs=yrs, plan=name, W=Cm['W'], W_hat=W_hat, Z=Zs, fam=fam))
    return out


def part_checks(s):
    Cm, D = community(s, 5); out = []
    for yrs in (1, 2, 3, 5):
        T = yrs * 365; p = PL.plan('Pump+Spray'); R = PL.plan_view(D, T, 'Pump+Spray')
        r = dict(seed=s, yrs=yrs, W=Cm['W'])
        # placebo: statistics at lag 1, before any excitation can arrive, so every pair is null
        Z0, _ = K.test_excitation(R, 0, {'pump': 1}, 0.05, [0, 1], PL.SHARED)
        Z1, _ = K.test_excitation(R, 1, {'spr': 1}, 0.05, [0, 1], PL.SHARED)
        r['Z_placebo'] = np.concatenate([Z0[~np.isnan(Z0)], Z1[~np.isnan(Z1)]])
        r['scan'], r['scan_lag'] = K.lag_scan(R, p, ALPHA)
        if yrs in (3, 5):
            r['base'] = K.baselines(PL.view(D, T, BASE_OUTS, [0, 1]), BASE_OUTS, ALPHA)
        out.append(r)
    return out


def part_channels(s):
    Cm, D = community(s, 5); out = []
    for yrs in (1, 2, 3, 5):
        R = PL.view(D, yrs * 365, PL.CANDIDATES, [0, 1])
        out.append(dict(seed=s, yrs=yrs, cert=K.channel_certificates(R, PL.CANDIDATES, [0, 1], PL.SHARED, ALPHA)))
    return out


def part_wrong_lag(s):
    """A sketch with a wrong arrow puts pest -> spraying at lag 3 (via perceived scarcity); the truth is lag 2."""
    Cm, D = community(s, 10); out = []
    for yrs in (3, 5, 10):
        R = PL.view(D, yrs * 365, ['spr'], [1])
        _, wrong = K.test_excitation(R, 1, {'spr': 3}, ALPHA, [1], PL.SHARED, tau0=TAU0)
        _, true = K.test_excitation(R, 1, {'spr': 2}, ALPHA, [1], PL.SHARED, tau0=TAU0)
        scan, lag = K.lag_scan(R, PL.plan('Spray+Pest'), ALPHA)
        out.append(dict(seed=s, yrs=yrs, W=Cm['W'], wrong=wrong, true=true, scan=scan, lag=lag))
    return out


def local_lags(seepage):
    """Lags and tau^0 for the local-aquifer scenario, read from the sketch of that variant (with seepage, canal
    water also enters the groundwater level directly)."""
    sk = C.sketch(seepage=seepage)
    return dict(tau_W=PL.lags(sk, physical=True, network='W'), tau0_W=PL.tau0(sk, physical=True, network='W'),
                tau_P=PL.lags(sk, physical=True, network='P'), tau0_P=PL.tau0(sk, physical=True, network='P'))


LOCAL = {False: local_lags(False), True: local_lags(True)}


def part_physical(s):
    """Local aquifer: the social network is tested through pumping and the physical network through the GW level,
    each at the lag where it arrives first (Lemma 1) and at level alpha / 2. With seepage the social and physical
    networks reach pumping at the same lag, so pumping is not tested for social links."""
    out = []
    for seep in (0.0, 0.5):
        L = LOCAL[seep > 0]
        Cm, D = community(s, 5, scenario='local', seepage=seep)
        for yrs in (3, 5):
            R = PL.view(D, yrs * 365, ['pump', 'gw'], [0])
            r = dict(seed=s, seep=seep, yrs=yrs, W=Cm['W'], P=Cm['P'], lag_physical=L['tau_P'][(0, 'gw')])
            if (0, 'pump') in L['tau_W']:                      # the social network arrives first at pumping
                r['lag_social'] = L['tau_W'][(0, 'pump')]
                _, r['social'] = K.test_excitation(R, 0, {'pump': r['lag_social']}, ALPHA / 2, [0], PL.SHARED, tau0=L['tau0_W'])
            _, r['physical'] = K.test_excitation(R, 0, {'gw': r['lag_physical']}, ALPHA / 2, [0], PL.SHARED, tau0=L['tau0_P'])
            out.append(r)
    return out


def part_hidden(s):
    """Farms 7 and 8 are not measured at all; links through them are indirect for the six measured farms."""
    Cm, D = community(s, 5); keep = [0, 1, 2, 3, 4, 5]; out = []
    for yrs in (3, 5):
        W_hat, _, _, _ = run_plan(Cm, D, yrs * 365, 'Pump+Spray', keep=keep)
        out.append(dict(seed=s, yrs=yrs, W=Cm['W'], keep=keep, W_hat=W_hat))
    return out


def part_ours_base_data(s):
    """Our test ('All practical') on the same simulated data that the baselines receive in part 'checks'."""
    Cm, D = community(s, 5); out = []
    for yrs in (3, 5):
        W_hat, _, _, _ = run_plan(Cm, D, yrs * 365, 'All practical')
        out.append(dict(seed=s, yrs=yrs, W=Cm['W'], W_hat=W_hat))
    return out


def part_ours_daily_outputs(s):
    """Our test on the baselines' data without the weekly survey: only the four daily outputs (Section 4 states that
    the reports are the same as with the survey)."""
    Cm, D = community(s, 5); out = []
    plan = dict(excitations=[0, 1], outputs=BASE_OUTS, shared=PL.SHARED)
    for yrs in (3, 5):
        R = PL.view(D, yrs * 365, plan['outputs'], plan['excitations'])
        W_hat, _, _ = K.certify_plan(R, plan, TAU, ALPHA, tau0=TAU0)
        out.append(dict(seed=s, yrs=yrs, W=Cm['W'], W_hat=W_hat))
    return out


def part_cgranger(s):
    Cm, D = community(s, 5); out = []
    for yrs in (3, 5):
        W_hat = K.conditional_granger(PL.view(D, yrs * 365, BASE_OUTS, [0, 1]), BASE_OUTS, L=4, alpha=ALPHA)
        out.append(dict(seed=s, yrs=yrs, W=Cm['W'], W_hat=W_hat))
    return out


def part_pool_corr(s):
    """Correlation, over canal event days k, between soil-moisture residuals at k + tau_sm and pumping residuals at
    k + tau_pump, as the SM+Pump plan pools them."""
    Cm, D = community(s, 10); R = PL.view(D, 10 * 365, ['sm', 'pump'], [0])
    t0 = {c: v for (d, c), v in TAU0.items() if d == 0}; ts, tg = TAU[(0, 'sm')], TAU[(0, 'pump')]
    Pi = R['Pi'][0]; cs = []
    for i in range(N):
        a = K.residuals(R, 'sm', i, ts, [0], PL.SHARED, tau0=t0)
        b = K.residuals(R, 'pump', i, tg, [0], PL.SHARED, tau0=t0)
        for j in range(N):
            if j == i:
                continue
            k = np.where(Pi[:, j])[0]; k = k[(k > 200) & (k + max(ts, tg) < len(a))]
            x, y = a[k + ts], b[k + tg]; ok = ~np.isnan(x) & ~np.isnan(y)
            cs.append(float(np.corrcoef(x[ok], y[ok])[0, 1]))
    return cs


PARTS = dict(plans=part_plans, checks=part_checks, channels=part_channels, wrong_lag=part_wrong_lag,
             physical=part_physical, hidden=part_hidden, ours_base_data=part_ours_base_data,
             ours_daily_outputs=part_ours_daily_outputs, cgranger=part_cgranger,
             pool_corr=part_pool_corr)

if __name__ == '__main__':
    ap = argparse.ArgumentParser(); ap.add_argument('part', choices=list(PARTS))
    ap.add_argument('--communities', type=int, default=NCOM); a = ap.parse_args()
    os.makedirs(RES, exist_ok=True); t0 = time.time(); out = []
    for s in range(a.communities):
        out += PARTS[a.part](s); print(a.part, 'community', s, f'{time.time() - t0:.0f} s', flush=True)
    if a.part == 'pool_corr':
        cs = np.array(out)
        res = dict(n=len(cs), median=float(np.median(cs)), q05=float(np.quantile(cs, .05)),
                   q95=float(np.quantile(cs, .95)), max_abs=float(np.abs(cs).max()))
        json.dump(res, open(os.path.join(RES, 'pool_corr.json'), 'w'), indent=1); print(res)
    else:
        pickle.dump(out, open(os.path.join(RES, f'farm_{a.part}.pkl'), 'wb'))
    print('done')
