"""The certified link test of Section 3 (Steps 1-3, Algorithm 1, Theorem 2), the lag scan (Proposition 3), the
channel certificates (Corollary 4) and the baselines of Section 4.

The test sees only data: a dict R with
    Y[o]      T x N   output o            (paper: Y_o)
    avail[o]  T x N   mask of the measured entries of Y[o]
    M[d]      T x N   excitation d        (paper: M_d), zero off event days
    Pi[d]     T x N   event days of d     (paper: Pi_d)
    sched     T x N   canal schedule (a farm's own turn days), a regressor
    Omega[s]  T       shared series s     (paper: the columns of Omega)
and, optionally, lags read from a channel sketch. It never sees W, P, parameter values, gains or noise variances.

For excitation d, output o and lag tau = tau_{d,o}, the term for pair j -> i on j's event day k is
    D_k = w_ij[k]' delta_ij[k],   delta_ij[k] = ( Mbar_d(k, j) Rbar_o(k, i) )_{o in O},
    Mbar_d(k, j) = m_{j,d}[k] / sigma_hat_{m,j}[k],   Rbar_o(k, i) = ytilde_{i,o}[k + tau] / sigma_hat_{ytilde,i,o}[k],
and A_ij = sum_k D_k, V2_ij = sum_k D_k^2, Z_ij = A_ij / sqrt(V2_ij)  (eq. 5). Every scale, weight and regression
coefficient uses only data before day k (condition C2)."""
import numpy as np
from math import ceil, log, sqrt

DAILY = ('sm', 'gw', 'pump', 'spr', 'temp')            # outputs measured every day (bel and pr are weekly surveys)


def threshold(n_p, alpha, ratio=1e4, kappas=np.linspace(1.02, 1.6, 59)):
    """[Theorem 2, eq. (6)] c_alpha = sqrt(2 kappa log(2 n_p n_kappa / alpha)), n_kappa = ceil(log_kappa(b_hi/b_lo)),
    with b_hi/b_lo = ratio and kappa chosen on a grid to make c_alpha smallest. Returns (c_alpha, kappa, n_kappa)."""
    best = None
    for kappa in kappas:
        n_kappa = ceil(log(ratio) / log(kappa)); c = sqrt(2 * kappa * log(2 * n_p * n_kappa / alpha))
        if best is None or c < best[0]:
            best = (c, kappa, n_kappa)
    return best


def p_values(Zs, infos, alpha, ratio=1e4):
    """Family-wise adjusted p-values (Appendix B) from the output of certify_plan:
    alpha_hat_ij = min(1, |D| min_d 2 n_p n_kappa exp(-Z_{d,ij}^2 / (2 kappa))), with kappa and n_kappa those of
    the threshold at level alpha / |D|, and a term set to 1 where V2_{d,ij} lies outside [b_lo, b_hi]."""
    nD = len(Zs); best = None
    for d, Z in Zs.items():
        f = infos[d]; _, kappa, n_kappa = threshold(f['n_p'], alpha / nD, ratio)
        n_j = f['n_events'][None, :]
        ok = (f['V2'] >= n_j / np.sqrt(ratio)) & (f['V2'] <= n_j * np.sqrt(ratio))
        p = np.where(ok, 2 * f['n_p'] * n_kappa * np.exp(-np.nan_to_num(Z) ** 2 / (2 * kappa)), 1.0)
        best = p if best is None else np.minimum(best, p)
    p = np.minimum(1, nD * best); np.fill_diagonal(p, 1.0)
    return p


def _own_lags(c, o, tau, tau0):
    """[Step 1] Lags at which receiver i's own data on channel c are regressors for output o tested at lag tau.
    Data at lag l before day k + tau are at lag tau - l from the sender's day-k innovation, and must not have been
    reached by it under the null. Output o itself is null below tau, so lags 1, 2 are safe. Another channel c is
    safe below tau0[c] = tau^0_c (from the sketch). Without a sketch tau^0_c = 2: no channel of another agent can
    respond before lag 2 (one day to enter the state, one to cross)."""
    if c == o:
        return (1, 2)
    t0 = 2 if tau0 is None else tau0.get(c)
    if t0 is None:
        return (1, 2)                                   # the sketch says channel c is never reached
    l0 = max(1, tau - t0 + 1)
    return (l0, l0 + 1)


def residuals(R, o, i, tau, excitations, shared, warm=90, every=7, ridge=1e-2, tau0=None):
    """[Step 1, eq. (4)] Online residual ytilde_{i,o}[k] of receiver i's output o, tested at lag tau. Regressors
    (row k of X_{i,o}): a constant; i's own daily outputs at the two most recent lags allowed by _own_lags; i's own
    excitations and canal schedule at lags 1..tau+1; the shared series at lags 0..3; the other farms' daily outputs
    at lag tau and excitations at lag tau+1. The ridge coefficients are refitted every `every` days on earlier rows
    only, with rho_k = ridge * tr(X'X) / (number of regressors), and are zero during the warm-up."""
    Y = R['Y'][o]; av = R['avail'][o][:, i]; T = Y.shape[0]; L = tau + 4
    t = np.arange(L, T); outs = [c for c in R['Y'] if c in DAILY]
    cols = [np.ones(len(t))]
    for c in outs:
        for l in _own_lags(c, o, tau, tau0):
            cols.append(R['Y'][c][t - l, i])
    for d in excitations:
        for l in range(1, tau + 2):
            cols.append(R['M'][d][t - l, i])
    for l in range(1, tau + 2):
        cols.append(R['sched'][t - l, i])
    for s in shared:
        for l in range(4):
            cols.append(R['Omega'][s][t - l])
    N = Y.shape[1]
    for l_ in range(N):
        if l_ == i:
            continue
        for c in outs:
            cols.append(R['Y'][c][t - tau, l_])
        for d in excitations:
            cols.append(R['M'][d][t - tau - 1, l_])
    X = np.stack(cols, 1); y = Y[t, i]; ok = av[t]; p = X.shape[1]
    XX = np.zeros((p, p)); Xy = np.zeros(p); last = 0; theta = np.zeros(p); out = np.full(T, np.nan)
    for s0 in range(0, len(t), every):
        if s0 > last:
            sl = slice(last, s0); w = ok[sl]; XX += X[sl][w].T @ X[sl][w]; Xy += X[sl][w].T @ y[sl][w]; last = s0
        if t[s0] >= warm and np.trace(XX) > 0:
            theta = np.linalg.solve(XX + ridge * np.trace(XX) / p * np.eye(p), Xy)
        sl = slice(s0, s0 + every); out[t[sl]] = np.where(ok[sl], y[sl] - X[sl] @ theta, np.nan)
    return out


def test_excitation(R, d, taus, alpha, excitations, shared, warm=90, every=7, ridge=1e-2, ratio=1e4,
                    weights='online', senders=None, receivers=None, info=None, tau0=None):
    """[Steps 1-3, Theorem 2] The test for one excitation d, pooling the outputs taus = {o: tau_{d,o}} at level
    alpha. Returns Z_d (N x N, receivers as rows, senders as columns; nan where untested) and the reported links
    W_hat_d = 1{|Z| >= c_alpha and V2 in [b_lo, b_hi]}, with [b_lo, b_hi] = [n_j / sqrt(ratio), n_j * sqrt(ratio)].
    If `info` is a dict, it receives c_alpha, n_p, the event counts n_j and the residual spreads."""
    o0 = next(iter(taus)); N = R['Y'][o0].shape[1]; T = R['Y'][o0].shape[0]
    senders = range(N) if senders is None else senders
    rec = [i for i in (range(N) if receivers is None else receivers) if any(R['avail'][o][:, i].any() for o in taus)]
    if not taus:
        return np.full((N, N), np.nan), np.zeros((N, N), bool)
    t0 = None if tau0 is None else {c: v for (dd, c), v in tau0.items() if dd == d}
    res = {(o, i): residuals(R, o, i, taus[o], excitations, shared, warm, every, ridge, t0) for o in taus for i in rec}
    kmax = T - max(taus.values()); k0 = warm + 14; ks = np.arange(k0, kmax)
    M = R['M'][d]; Pi = R['Pi'][d]
    Z = np.full((N, N), np.nan); V2 = np.zeros((N, N)); n_events = np.zeros(N)
    pairs = [(j, i) for j in senders for i in rec if i != j]
    c_alpha = threshold(max(1, len(pairs)), alpha, ratio)[0]
    for j in senders:
        ev = Pi[ks, j]; n_events[j] = ev.sum()
        m_j = M[:, j]; cm = np.cumsum(m_j ** 2)
        for i in rec:
            if i == j:
                continue
            Rbar = []
            for o in taus:
                yt = res[(o, i)][ks + taus[o]]; okm = ~np.isnan(yt); yt = np.where(okm, yt, 0.0)
                full = res[(o, i)]; f2 = np.where(np.isnan(full), 0.0, full) ** 2
                cnt = np.cumsum(~np.isnan(full)); cy = np.cumsum(f2)
                s_y = np.sqrt(np.maximum(cy[ks - 1], 1e-12) / np.maximum(cnt[ks - 1], 1))   # RMS over earlier days
                Rbar.append(yt / s_y)
            Rbar = np.stack(Rbar, 1)                                                     # (days, |O|)
            s_m = np.sqrt(np.maximum(cm[ks - 1], 1e-12) / np.maximum(np.cumsum(Pi[:, j])[ks - 1], 1))
            delta = (m_j[ks] / s_m)[:, None] * Rbar * ev[:, None]                        # zero off event days
            if weights == 'online' and len(taus) > 1:
                acc = np.cumsum(delta, 0) - delta                                         # earlier events only
                n_prev = np.cumsum(ev) - ev
                w = np.where((n_prev >= 10)[:, None], acc, 1.0)                          # equal before the 10th event
                w = w / np.maximum(np.linalg.norm(w, axis=1, keepdims=True), 1e-12)
            else:
                w = np.ones_like(delta) / np.sqrt(delta.shape[1])
            D = (w * delta).sum(1)
            A = D.sum(); v2 = (D ** 2).sum(); V2[i, j] = v2
            Z[i, j] = A / np.sqrt(v2) if v2 > 0 else 0.0
    n_j = n_events[None, :] * np.ones((N, 1))
    in_range = (V2 >= n_j / np.sqrt(ratio)) & (V2 <= n_j * np.sqrt(ratio))
    W_hat = (np.abs(np.nan_to_num(Z)) >= c_alpha) & in_range; np.fill_diagonal(W_hat, False)
    if info is not None:
        info.update(c_alpha=c_alpha, n_p=len(pairs), n_events=n_events, V2=V2, receivers=rec, ks=ks,
                    sd={(o, i): float(np.nanstd(res[(o, i)][ks + taus[o]])) for o in taus for i in rec})
    return Z, W_hat


def certify_plan(R, plan, taus, alpha=0.1, **kw):
    """[Algorithm 1] One test per excitation of the plan, each at level alpha / |D|, outputs pooled; the reports
    are combined by an elementwise OR. taus = {(d, o): tau_{d,o}} from the sketch (plans.lags).
    Returns W_hat, {d: Z_d} and {d: info}."""
    D_set = [d for d in plan['excitations'] if any(taus.get((d, o)) for o in plan['outputs'])]
    N = R['Y'][plan['outputs'][0]].shape[1]; W_hat = np.zeros((N, N), bool); Zs = {}; infos = {}
    for d in D_set:
        tt = {o: taus[(d, o)] for o in plan['outputs'] if taus.get((d, o))}
        infos[d] = {}
        Z, Wd = test_excitation(R, d, tt, alpha / max(1, len(D_set)), plan['excitations'], plan['shared'], info=infos[d], **kw)
        W_hat |= Wd; Zs[d] = Z; infos[d]['taus'] = tt
    return W_hat, Zs, infos


def lag_scan(R, plan, alpha=0.1, Lbar=6, **kw):
    """[Proposition 3] Without a sketch: for each (excitation d, output o), test lags tau = 1..Lbar at level
    alpha / (Lbar |D| |O|) and stop at the first lag with a report. Returns the union of the reports and the lag
    found for each (d, o) (None if nothing was reported)."""
    N = R['Y'][plan['outputs'][0]].shape[1]; pairs = [(d, o) for d in plan['excitations'] for o in plan['outputs']]
    a = alpha / (Lbar * len(pairs)); W_hat = np.zeros((N, N), bool); found = {}
    for d, o in pairs:
        found[(d, o)] = None
        for tau in range(1, Lbar + 1):
            Z, Wd = test_excitation(R, d, {o: tau}, a, plan['excitations'], plan['shared'], **kw)
            if Wd.any():
                W_hat |= Wd; found[(d, o)] = tau; break
    return W_hat, found


def channel_certificates(R, candidates, excitations, shared, alpha=0.1, Lbar=6, **kw):
    """[Corollary 4] Run the lag scan for each candidate (d, o) at level alpha / (Lbar |D| |O|); (d, o) is certified
    to carry influence if any pair is reported at any lag. Returns {(d, o): first lag with a report, or None}."""
    pairs = [(d, o) for d in excitations for o in candidates]
    a = alpha / (Lbar * len(pairs)); cert = {}
    for d, o in pairs:
        cert[(d, o)] = None
        for tau in range(1, Lbar + 1):
            Z, Wd = test_excitation(R, d, {o: tau}, a, excitations, shared, **kw)
            if Wd.any():
                cert[(d, o)] = tau; break
    return cert


# ---------------- baselines on the same data (Section 4, Appendix C) ----------------
from scipy.stats import norm
from sklearn.linear_model import LassoLarsIC


def _ols_t(X, y):
    b, *_ = np.linalg.lstsq(X, y, rcond=None); r = y - X @ b; s2 = r @ r / max(1, len(y) - X.shape[1])
    cov = s2 * np.linalg.pinv(X.T @ X); return b / np.sqrt(np.maximum(np.diag(cov), 1e-300))


def _design(R, outs, lags, with_inputs):
    """Regressors for predicting every output at day t+1: all farms' outputs at the given lags and, with_inputs,
    the excitations, the canal schedule and the shared series at day t. `owner` gives the farm of each column."""
    T, N = R['Y'][outs[0]].shape; L = max(lags); t = np.arange(L, T - 1); cols = [np.ones(len(t))]; owner = [-1]
    for l in lags:
        for o in outs:
            for j in range(N):
                cols.append(R['Y'][o][t + 1 - l, j]); owner.append(j)
    if with_inputs:
        for d in R['M']:
            for j in range(N):
                cols.append(R['M'][d][t, j]); owner.append(-1)
        for j in range(N):
            cols.append(R['sched'][t, j]); owner.append(-1)
        for s, v in R['Omega'].items():
            cols.append(v[t]); owner.append(-1)
    return np.stack(cols, 1), np.array(owner), t


BASE_FALLBACKS = [0]   # number of Lasso fits that needed the coordinate-descent fallback (Appendix C)


def _lasso_bic(X, y):
    """Lasso with the BIC choice of penalty (LassoLarsIC). If the LARS solver fails on a degenerate design (a known
    scikit-learn failure), the same BIC criterion is applied to the coordinate-descent lasso path instead."""
    try:
        return LassoLarsIC(criterion='bic').fit(X, y).coef_
    except ValueError:
        from sklearn.linear_model import lasso_path
        BASE_FALLBACKS[0] += 1
        yc = y - y.mean(); n_ = len(yc)
        alphas, coefs, _ = lasso_path(X, yc, n_alphas=100)
        rss = ((yc[:, None] - X @ coefs) ** 2).sum(0)
        bic = n_ * np.log(rss / n_ + 1e-300) + (np.abs(coefs) > 0).sum(0) * np.log(n_)
        return coefs[:, int(np.argmin(bic))]


def baselines(R, outs, alpha=0.1):
    """DMDc: least-squares regression of each output at k+1 on all outputs at k, the excitations, the canal
    schedule and the shared series (no rank truncation); report j -> i if any coefficient from j has an OLS |t|
    above the Bonferroni threshold over ordered pairs, coefficients and outputs.
    Output Granger: the same test on the outputs at lags 1-3, without excitations or shared series.
    Lasso-Granger: the standardised DMDc regressors at lags 1 and 2, penalty by BIC; any non-zero coefficient."""
    T, N = R['Y'][outs[0]].shape; out = {}
    for name, lags, inp in (('dmdc', (1,), True), ('output_granger', (1, 2, 3), False)):
        X, own, t = _design(R, outs, lags, inp); ncoef = len(lags) * len(outs)
        z = norm.ppf(1 - alpha / (2 * N * (N - 1) * ncoef * len(outs)))
        W_hat = np.zeros((N, N), bool)
        for o in outs:
            for i in range(N):
                tv = _ols_t(X, R['Y'][o][t + 1, i])
                for j in range(N):
                    if j != i and np.any(np.abs(tv[own == j]) > z):
                        W_hat[i, j] = True
        out[name] = W_hat
    X, own, t = _design(R, outs, (1, 2), True); Xs = (X[:, 1:] - X[:, 1:].mean(0)) / (X[:, 1:].std(0) + 1e-12); own = own[1:]
    W_hat = np.zeros((N, N), bool)
    for o in outs:
        for i in range(N):
            b = _lasso_bic(Xs, R['Y'][o][t + 1, i])
            for j in range(N):
                if j != i and np.any(np.abs(b[own == j]) > 0):
                    W_hat[i, j] = True
    out['lasso_granger'] = W_hat
    return out


def conditional_granger(R, outs, L=4, alpha=0.1):
    """Conditional Granger: regress each output on a constant, every output of every farm at lags 1..L, every
    excitation and the canal schedule at lags 1..L, and the shared series at lags 0..L. For each sender j, a Wald
    test of all of j's output lags with Newey-West (HAC) covariance; report j -> i if any output rejects at level
    alpha / (N (N-1) |O|)."""
    from scipy.stats import chi2
    T, N = R['Y'][outs[0]].shape
    t = np.arange(L, T)
    cols = [np.ones(len(t))]; owner = [-1]
    for o in outs:
        for l in range(1, L + 1):
            for j in range(N):
                cols.append(R['Y'][o][t - l, j]); owner.append(j)
    for d in R['M']:
        for l in range(1, L + 1):
            for j in range(N):
                cols.append(R['M'][d][t - l, j]); owner.append(-1)
    for l in range(1, L + 1):
        for j in range(N):
            cols.append(R['sched'][t - l, j]); owner.append(-1)
    for s, v in R['Omega'].items():
        for l in range(0, L + 1):
            cols.append(v[t - l]); owner.append(-1)
    X = np.stack(cols, 1); owner = np.array(owner)
    sc = X.std(0); sc[0] = 1.0; sc[sc == 0] = 1.0; X = X / sc
    n_, p = X.shape
    XtXi = np.linalg.pinv(X.T @ X)
    bw = int(np.floor(4 * (n_ / 100) ** (2 / 9)))                     # Newey-West bandwidth
    thr = alpha / (N * (N - 1) * len(outs))
    W_hat = np.zeros((N, N), bool)
    for o in outs:
        for i in range(N):
            y = R['Y'][o][t, i]
            b = XtXi @ (X.T @ y); e = y - X @ b
            Xe = X * e[:, None]
            S = Xe.T @ Xe
            for l in range(1, bw + 1):
                G = Xe[l:].T @ Xe[:-l]; S += (1 - l / (bw + 1)) * (G + G.T)
            V = XtXi @ S @ XtXi
            for j in range(N):
                if j == i:
                    continue
                idx = owner == j
                bj = b[idx]; Vj = V[np.ix_(idx, idx)]
                wald = float(bj @ np.linalg.pinv(Vj) @ bj)
                if chi2.sf(wald, idx.sum()) < thr:
                    W_hat[i, j] = True
    return W_hat
