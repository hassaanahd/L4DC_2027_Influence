"""Scoring against the true network, and the power prediction of eq. (7) with the TRUE gains. Uses the truth, so it
belongs to scoring, never to the test."""
import numpy as np
from scipy.stats import norm
import community as C

# Quantities of eq. (7) fixed by the design of the simulation (Appendix C):
SIGMA_E = 0.5; SIGMA_XI = 0.15; LAMBDA = SIGMA_E**2 / (SIGMA_E**2 + SIGMA_XI**2); KURT = 9.0   # t_5: K = 9
# r = mu_q^2 / (mu_q^2 + sigma_q^2). The canal innovation enters with gain q_k = Q[k], AR(1) with coefficient 0.8
# and innovation s.d. 0.15, so mu_q = 1 and sigma_q^2 = 0.15^2 / (1 - 0.8^2); the pest gain is fixed (r = 1).
R_GAIN = {0: 1.0 / (1.0 + 0.15**2 / (1 - 0.8**2)), 1: 1.0}


def score(W_hat, W):
    """One community: any false link (for the FWER), false links (FP), precision, recall, F1 and FDR."""
    off = ~np.eye(len(W), dtype=bool); true = (W > 0) & off
    tp = int((W_hat & true).sum()); fp = int((W_hat & ~true & off).sum()); fn = int((~W_hat & true).sum())
    p = tp / (tp + fp) if tp + fp else 1.0; r = tp / (tp + fn) if tp + fn else 1.0
    f1 = 2 * p * r / (p + r) if p + r else 0.0
    return dict(any_false=fp > 0, FP=fp, precision=p, recall=r, F1=f1, FDR=1 - p)


def with_true_gains(infos, truth):
    """Add the TRUE impulse responses h = S_ij g at the tested lags to the info returned by certify_plan."""
    return {d: dict(f, h={o: C.impulse_response(truth, d, o, t) for o, t in f['taus'].items()}) for d, f in infos.items()}


def link_mu(fam, scale=1.0):
    """[eq. (7)] mu_ij for every pair, from the true impulse responses at the tested lags and the residual spreads
    observed by the test, with the outputs pooled as if their residuals were uncorrelated
    (g' Sigma^-1 g = sum_o g_o^2 / sigma_o^2). `scale` multiplies the event counts n_j (longer durations)."""
    out = {}
    for d, f in fam.items():
        N = len(f['n_events']); rho2 = np.zeros((N, N))
        for o, h in f['h'].items():
            sd = np.full(N, np.inf)
            for (oo, i), v in f['sd'].items():
                if oo == o and v > 0:
                    sd[i] = v
            rho2 += (h * SIGMA_E * np.sqrt(LAMBDA) / sd[:, None]) ** 2
        n_j = f['n_events'][None, :] * scale
        out[d] = (np.sqrt(n_j) * np.sqrt(rho2) / np.sqrt(1 + (KURT - 1) / R_GAIN[d] * rho2), f['c_alpha'])
    return out


def predicted_found(fam, scale=1.0):
    """Predicted probability that each pair is reported: 1 - prod_d (1 - P(|N(mu_d, 1)| >= c_alpha))."""
    q = None
    for d, (mu, c) in link_mu(fam, scale).items():
        p = 1 - (norm.cdf(mu - c) + norm.cdf(-mu - c)); q = p if q is None else q * p
    return 1 - q
