"""The simulated farming community of Section 4 and Appendix C: eight farms on one watercourse, one step per day.

Notation follows the paper (Appendix A). Each farm i has n = 8 channels, the entries of its state x_i:

    'sm'   soil moisture            x_i^sm
    'gw'   local groundwater level  x_i^gw
    'bel'  perceived scarcity       x_i^bel   (measured only by a weekly survey)
    'pump' pumping                  x_i^pump
    'spr'  spraying                 x_i^spr
    'talk' phone talk               x_i^talk  (never measured)
    'temp' canopy temperature       x_i^temp  (decoy: driven only by shared drivers)
    'pr'   price attention          x_i^pr    (decoy; measured only by a weekly survey)

Inputs u_i (index d of the paper): d = 0 canal water received, u_i^canal; d = 1 pest pressure, u_i^pest.
The excitations are the measured innovations m_{j,d} = e_{j,d} + xi_{j,d} in these inputs.
Shared drivers omega (the shared series Omega when measured): rain, et (evapotranspiration), temp, pr (price),
Q (head discharge of the canal), gw (regional groundwater), pest (regional pest pressure).
The social network W acts through H_i on bel, pump, spr and talk; the physical network P acts through E_i on gw
(local-aquifer scenario only).

All variables are deviations from a seasonal operating point. The test (certify.py) never sees anything in this
module except the data returned by simulate() and, optionally, the zero pattern returned by sketch()."""
import numpy as np

CH = ['sm', 'gw', 'bel', 'pump', 'spr', 'talk', 'temp', 'pr']          # channels of one farm, in state order
IX = {c: k for k, c in enumerate(CH)}
n = len(CH)
EXC = {0: 'canal', 1: 'pest'}                                          # input index d -> name
SHARED = ['rain', 'et', 'temp', 'pr', 'Q', 'gw', 'pest']               # columns of omega, in order

# Nominal coefficients, as printed in the equations of Appendix C. Names: phi_c persistence of channel c;
# theta_c copying weight of channel c (the diagonal of H_i); zeta, chi, zeta_h, eta, eta_h, eta_mu, delta_* couplings
# inside a farm; a_* input and driver gains. Each farm multiplies every one of these by its own draw (agent()); the
# fixed driver gains in F (0.02, 0.2, 0.5) and the gains of z_i (1, 0.5 in simulate()) are the same for all farms.
# The order of this dict fixes the order of the random draws in agent(); do not reorder it.
NOM = dict(phi_sm=0.75, zeta=0.20, a_c=1.0, a_r=0.8, a_et=0.6,
           phi_gw=0.95, chi=0.05, zeta_h=0.10,
           phi_bel=0.45, eta=0.20, eta_h=0.10, eta_mu=0.20, theta_bel=0.25,
           phi_pump=0.25, delta_s=0.50, delta_h=0.10, delta_b=0.20, delta_f=0.30, theta_pump=0.45,
           phi_spr=0.40, a_p=1.0, theta_spr=0.35,
           phi_talk=0.20, a_f=0.50, theta_talk=0.30,
           phi_temp=0.60, a_t=0.50, phi_pr=0.80, a_mu=0.50)


def agent(rng, het=0.25, drop=()):
    """One farm's blocks (Phi_i, H_i, Gamma_i, F_i) and its parameter values. Each nominal value is multiplied by
    the farm's own draw from U[1-het, 1+het]; persistences are capped at 0.95. `drop` sets named parameters to
    zero (used to build a sketch with a missing arrow)."""
    p = {k: v * (rng.uniform(1 - het, 1 + het) if het > 0 else 1.0) for k, v in NOM.items()}
    for k in drop:
        p[k] = 0.0
    for k in p:
        if k.startswith('phi_'):
            p[k] = min(p[k], 0.95)
    Phi = np.zeros((n, n)); H = np.zeros((n, n)); Gam = np.zeros((n, 2)); F = np.zeros((n, len(SHARED)))
    sm, gw, bel, pump, spr, talk, temp, pr = (IX[c] for c in CH)
    Phi[sm, sm] = p['phi_sm']; Phi[sm, pump] = p['zeta']
    Phi[gw, gw] = p['phi_gw']; Phi[gw, sm] = p['chi']; Phi[gw, pump] = -p['zeta_h']
    Phi[bel, bel] = p['phi_bel']; Phi[bel, sm] = -p['eta']; Phi[bel, gw] = -p['eta_h']; Phi[bel, pr] = p['eta_mu']
    Phi[pump, pump] = p['phi_pump']; Phi[pump, sm] = -p['delta_s']; Phi[pump, gw] = -p['delta_h']
    Phi[pump, bel] = p['delta_b']; Phi[pump, talk] = p['delta_f']
    Phi[spr, spr] = p['phi_spr']
    Phi[talk, talk] = p['phi_talk']
    Phi[temp, temp] = p['phi_temp']; Phi[pr, pr] = p['phi_pr']
    H[bel, bel] = p['theta_bel']; H[pump, pump] = p['theta_pump']; H[spr, spr] = p['theta_spr']; H[talk, talk] = p['theta_talk']
    Gam[sm, 0] = p['a_c']; Gam[talk, 0] = p['a_f']; Gam[spr, 1] = p['a_p']
    S = {c: k for k, c in enumerate(SHARED)}
    F[sm, S['rain']] = p['a_r']; F[sm, S['et']] = -p['a_et']; F[gw, S['gw']] = 0.02; F[pump, S['gw']] = 0.2
    F[temp, S['temp']] = p['a_t']; F[pr, S['pr']] = p['a_mu']; F[spr, S['pest']] = 0.5
    return Phi, H, Gam, F, p


# ---------------- networks ----------------
def social_network(N, rng, w_min=0.2):
    """W: each farm follows 2 or 3 others, an upstream or downstream neighbour with probability 0.7, otherwise a
    random farm; Dirichlet(2) weights of at least w_min; rows sum to one. W[i, j] > 0 means i follows j."""
    W = np.zeros((N, N))
    for i in range(N):
        k = rng.integers(2, 4); nb = set()
        while len(nb) < k:
            j = (i + rng.choice([-1, 1])) if rng.random() < 0.7 else rng.integers(N)
            if 0 <= j < N and j != i:
                nb.add(int(j))
        nb = sorted(nb)
        while True:
            w = rng.dirichlet(2 * np.ones(len(nb)))
            if w.min() >= w_min:
                break
        W[i, nb] = w
    return W


def physical_network(N):
    """P: adjacency of neighbouring farms along the watercourse."""
    P = np.zeros((N, N))
    for i in range(N):
        for j in (i - 1, i + 1):
            if 0 <= j < N:
                P[i, j] = 1.0
    return P


def build(N, seed, scenario='regional', het=0.25, beta_P=0.10, seepage=0.0):
    """One community: the full matrices of model (1)-(2). scenario='local' adds the physical network P on the
    groundwater channel (E_i has one entry, on gw); seepage > 0 lets canal water reach the aquifer the same day."""
    rng = np.random.default_rng(seed); W = social_network(N, rng)
    agents = [agent(rng, het) for _ in range(N)]
    P = physical_network(N) if scenario == 'local' else np.zeros((N, N))
    Phi = np.zeros((N * n, N * n)); Gamma = np.zeros((N * n, N * 2)); F = np.zeros((N * n, len(SHARED))); g = IX['gw']
    for i, (Phi_i, H_i, Gam_i, F_i, _) in enumerate(agents):
        Phi[i*n:(i+1)*n, i*n:(i+1)*n] = Phi_i; Gamma[i*n:(i+1)*n, 2*i:2*i+2] = Gam_i; F[i*n:(i+1)*n] = F_i
        if seepage > 0:
            Gamma[i*n+g, 2*i] = seepage
        for j in range(N):
            if W[i, j] > 0:
                Phi[i*n:(i+1)*n, j*n:(j+1)*n] += W[i, j] * H_i
            if P[i, j] > 0:
                Phi[i*n+g, j*n+g] += beta_P * P[i, j]; Phi[i*n+g, i*n+g] -= beta_P * P[i, j]
    return dict(W=W, P=P, Phi=Phi, Gamma=Gamma, F=F, agents=agents, N=N, beta_P=beta_P, seepage=seepage)


# ---------------- simulation ----------------
# sigma_e: innovation s.d. (Student-t with df degrees of freedom, scaled); sigma_xi: measurement error of the
# excitations; turn_period: days in the canal rotation (pi_canal = 1/7); sigma_w: process noise; sigma_z: noise of
# the unmeasured driver z_i; sigma_v: measurement error of each channel. The order of sigma_v fixes the order of
# the random draws; do not reorder it.
CFG = dict(sigma_e_canal=0.5, sigma_e_pest=0.5, df=5, sigma_xi_canal=0.15, sigma_xi_pest=0.15, turn_period=7,
           sigma_w=0.02, sigma_z=0.02,
           sigma_v=dict(sm=0.02, gw=0.01, bel=0.15, pump=0.02, spr=0.05, temp=0.02, pr=0.15))
WEEKLY = ('bel', 'pr')                                                 # measured by a weekly survey


def simulate(Cm, T, seed, cfg=CFG, burn=365):
    """Simulate T days after a burn-in. Returns the data a field team could measure, plus the true innovations
    under 'truth' (used only for checks, never by the test):
        Y[c]      T x N   measured channel c (zero where not measured)
        avail[c]  T x N   mask of the measured entries
        M[d]      T x N   excitation values m_{j,d} (zero off event days)
        Pi[d]     T x N   event days of excitation d
        sched     T x N   canal schedule: Q[k] on a farm's turn days, else 0
        Omega[s]  T       shared series s"""
    rng = np.random.default_rng(seed); N = Cm['N']; TT = T + burn; df = cfg['df']
    st = lambda size: rng.standard_t(df, size=size) / np.sqrt(df / (df - 2))     # unit-variance Student-t
    day = np.arange(TT); season = np.sin(2 * np.pi * day / 365)
    turn = np.zeros((TT, N), bool); slot = np.random.default_rng(seed + 7).permutation(N) % cfg['turn_period']
    for i in range(N):
        turn[:, i] = (day % cfg['turn_period']) == slot[i]
    e_canal = np.where(turn, cfg['sigma_e_canal'] * st((TT, N)), 0.0)
    e_pest = cfg['sigma_e_pest'] * st((TT, N))
    # shared drivers
    rain = np.where(rng.random(TT) < 0.08 + 0.12 * np.clip(season, 0, None), rng.gamma(2.0, 0.5, TT), 0.0)
    temp = np.zeros(TT); price = np.zeros(TT); q = np.zeros(TT); gw_reg = np.zeros(TT); pest_reg = np.zeros(TT)
    for k in range(1, TT):
        temp[k] = 0.7 * temp[k-1] + 0.3 * rng.normal(); price[k] = 0.98 * price[k-1] + 0.1 * rng.normal()
        q[k] = 0.8 * q[k-1] + 0.15 * rng.normal(); gw_reg[k] = 0.99 * gw_reg[k-1] + 0.02 * rng.normal()
        pest_reg[k] = 0.9 * pest_reg[k-1] + 0.2 * rng.normal()
    et = 0.5 * season + 0.3 * temp
    Q = 1 + q                                                          # head discharge, mean mu_Q = 1
    canal = np.where(turn, Q[:, None] * (1 + e_canal), 0.0)            # u^canal: water received on turn days
    pest = np.zeros((TT, N))
    for k in range(1, TT):
        pest[k] = 0.5 * pest[k-1] + e_pest[k]                          # u^pest, so psi_l = 0.5^l in A2
    omega = np.stack([rain - rain.mean(), et, temp, price, q, gw_reg, pest_reg], 1)   # columns as in SHARED
    z = np.zeros((TT, N)); x = np.zeros(N * n); X = np.zeros((TT, N, n))
    # u^canal is centred by its design mean mu_Q / 7 = 1/7, fixed in advance and independent of the innovations
    # (Appendix C). Centring by the realised mean would put a share of every innovation into every farm's input.
    c_mean = 1.0 / cfg['turn_period']
    for k in range(TT):
        X[k] = x.reshape(N, n)
        if k:
            z[k] = 0.95 * z[k-1] + cfg['sigma_z'] * rng.normal(size=N)   # unmeasured driver z_i
        u = np.stack([canal[k] - c_mean, pest[k]], 1).reshape(-1)
        dz = np.zeros((N, n)); dz[:, IX['sm']] = z[k]; dz[:, IX['pump']] = 0.5 * z[k]
        x = Cm['Phi'] @ x + Cm['Gamma'] @ u + Cm['F'] @ omega[k] + dz.reshape(-1) + cfg['sigma_w'] * rng.normal(size=N * n)
    s = slice(burn, TT)
    Y = {c: X[s][:, :, IX[c]] + cfg['sigma_v'][c] * rng.normal(size=(T, N)) for c in cfg['sigma_v']}
    survey = (np.arange(T) % 7) == 0
    avail = {c: np.ones((T, N), bool) for c in Y}
    for c in WEEKLY:
        avail[c] = np.repeat(survey[:, None], N, 1)
    m_canal = np.where(turn[s], e_canal[s] + cfg['sigma_xi_canal'] * rng.normal(size=(T, N)), 0.0)
    m_pest = e_pest[s] + cfg['sigma_xi_pest'] * rng.normal(size=(T, N))
    Omega = {nm: omega[s, k] for k, nm in enumerate(SHARED)}
    return dict(Y=Y, avail=avail, M={0: m_canal, 1: m_pest}, Pi={0: turn[s], 1: np.ones((T, N), bool)},
                sched=np.where(turn[s], Q[s][:, None], 0.0), Omega=Omega, truth=dict(e_canal=e_canal[s], e_pest=e_pest[s]))


# ---------------- the channel sketch and first lags (Definition 1, Lemma 1) ----------------
def sketch(drop=(), seepage=False):
    """The channel sketch: which entries of Phi_i, H_i, Gamma_i and E_i are non-zero, without values.
    `drop` removes named arrows (a sketch with a missing arrow); seepage=True adds the arrow from canal water to
    the groundwater level of the seepage variant (build(..., seepage > 0))."""
    Phi, H, Gam, _, _ = agent(np.random.default_rng(0), het=0.0, drop=drop)
    if seepage:
        Gam[IX['gw'], 0] = 1.0
    E = np.zeros((n, n)); E[IX['gw'], IX['gw']] = 1.0
    return dict(Phi=Phi != 0, H=H != 0, Gamma=Gam != 0, E=E != 0)


def first_lags(sk, d, out, physical=False, maxlag=15):
    """First lags of Definition 1 for excitation d and channel `out` of another agent, from the sketch alone:
        W    tau^W, one crossing through the social network (H_i)
        P    tau^P, one crossing through the physical network (E_i); None without a physical network
        tau2 tau_2, two or more crossings of any kind
        own  first lag at which the input reaches the agent's own channel `out`
    None means never reached within maxlag."""
    A = sk['Phi'].astype(int); XW = sk['H'].astype(int)
    XP = sk['E'].astype(int) if physical else np.zeros_like(XW); X = ((XW + XP) > 0).astype(int)
    g = sk['Gamma'][:, d].astype(int); pw = [np.eye(len(A), dtype=int)]
    for _ in range(maxlag):
        pw.append(((pw[-1] @ A) > 0).astype(int))

    def first(fun):
        for tau in range(1, maxlag):
            if fun(tau):
                return tau
        return None

    def one(Xc):
        return lambda tau: tau >= 2 and any(((pw[a] @ Xc @ pw[tau-2-a] @ g)[out]) > 0 for a in range(tau - 1))
    two = lambda tau: tau >= 3 and any(((pw[a] @ X @ pw[b] @ X @ pw[tau-3-a-b] @ g)[out]) > 0
                                       for a in range(tau - 2) for b in range(tau - 2 - a))
    own = first(lambda tau: ((pw[tau-1] @ g)[out]) > 0)
    return dict(W=first(one(XW)), P=first(one(XP)) if physical else None, tau2=first(two), own=own)


def impulse_response(Cm, d, o, tau):
    """h_{ij,d,o}[tau] of eq. (3) from the TRUE matrices, as an N x N array (receiver i, sender j)."""
    v = np.linalg.matrix_power(Cm['Phi'], tau - 1) @ Cm['Gamma'][:, d::2]
    return v[IX[o]::n, :]
