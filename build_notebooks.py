"""Writes notebooks 00-05. Run `python build_notebooks.py`, then execute them in order (see README.md)."""
import nbformat as nbf
from nbformat.v4 import new_notebook, new_markdown_cell as md, new_code_cell as code


def save(name, cells):
    nb = new_notebook(cells=cells)
    nb.metadata['kernelspec'] = dict(name='python3', display_name='Python 3', language='python')
    nbf.write(nb, name)


TAGS = """**How to read the tags.** A code cell that uses a result of the paper starts with a comment such as
`# [Thm 2] threshold c_alpha`. The tag names the result; the rest of the line says what the cell does. Numbering
follows the paper: Definition 1 and Lemma 1 (first lags), Theorem 2 (certified links), Proposition 3 (lag scan),
Corollary 4 (channel certificates), Algorithm 1 and the Procedure of Section 3, equations (3) (impulse response),
(4) (residuals), (5) (the statistic), (6) (the threshold), (7) (power) and (8) (measurement duration), and the
conditions C1 (measured innovations) and C2 (predictable regressors and weights) of Theorem 2."""

SETUP = """import os, sys, glob, pickle, warnings, numpy as np, pandas as pd
warnings.filterwarnings('ignore')            # scikit-learn convergence messages from the Lasso baseline
sys.path.insert(0, 'lib')
import certify as K, farmlib as FL
np.set_printoptions(precision=2, suppress=True, linewidth=150)
files = sorted(glob.glob('data/measurements/*.npz'))
SK = FL.load_sketch('data/sketch/channel_sketch.npz')
TAU, TAU0 = FL.lags(SK), FL.tau0(SK)          # lags tau_{d,o} and tau^0_c, from the sketch alone
ALPHA = 0.1"""

# =====================================================================================================
# 00 overview
# =====================================================================================================
save('00_overview.ipynb', [
md("""# 00 Overview: what the code does

The code follows the paper's claim that **the test runs on field data only**. Data generation is in a separate
notebook. It writes three kinds of file, and each later notebook reads only what it is allowed to:

![process](process.png)

| Folder | Contents | Who may read it |
|---|---|---|
| `data/measurements/` | what a field team measures: excitations $\\mathbf M_d$ and their event days $\\bm\\Pi_d$, outputs $\\mathbf Y_o$ with masks of the measured entries, the canal schedule, shared series $\\bm\\Omega$ | 02, 03, 04 |
| `data/sketch/` | the channel sketch: **which** entries of $\\mathbf\\Phi_i$, $\\mathbf H_i$, $\\mathbf\\Gamma_i$, $\\mathbf E_i$ are non-zero (no values) | 02, 03, 04 |
| `data/truth/` | the networks $\\mathbf W$, $\\mathbf P$ and the full matrices | **04 only** (scoring) |
| `results/` | what the test reported | 04 |
"""),
md("""## Notebooks and the paper's results

| Notebook | Task | Paper results used |
|---|---|---|
| `01_generate_data` | build random communities, simulate day by day, split into measurements / sketch / truth | model (1)-(2), Appendix C |
| `02_certify_links` | find links from the data | Lemma 1 (lags from the sketch); Step 1 and C2 (residuals, eq. 4); Steps 2-3 (eq. 5, by hand and in matrix form); Theorem 2 (threshold eq. 6, decisions, p-values); Proposition 3 (lag scan without the sketch) |
| `03_what_to_measure` | pilot on every candidate channel; decide what to measure | Corollary 4 (channel certificates); eq. (7) scaled from the pilot; the Procedure of Section 3 |
| `04_score_and_compare` | score against the truth; baselines; predicted vs observed power; wrong sketch | checks Theorem 2, eq. (7), Proposition 3 |
| `05_hand_example` | 3 farms, 2 channels, 50 days, end to end | all of the above, small |

## The code underneath (`lib/`)
* `community.py`: the simulator (farm model, networks, the channel sketch, first lags from the sketch).
* `certify.py`: the test (`residuals`, `test_excitation`, `certify_plan`, `threshold`, `p_values`), the lag scan (`lag_scan`), channel certificates (`channel_certificates`) and the baselines (`baselines`, `conditional_granger`).
* `plans.py`: the measurement plans of Table 3, `view()` (the data a plan provides) and the lags read from a sketch.
* `farmlib.py`: saving and loading the three kinds of file. `scoring.py` and `netplot.py`: scoring and drawing, used only where the truth may be read.

These are the same modules that produced the paper's numbers (`paper_runs/`).

## Run order and run time
Run 01, then 02, 03 and 04 (05 is independent). With the defaults (10 communities, 5 years) the set runs in a
few minutes. The paper's full runs (40 communities, up to 10 years) are in `paper_runs/`.

## One limit the code makes visible
The Procedure predicts each plan from a pilot. From pilot data alone one can estimate each pair's
$\\hat\\mu(T)=|Z^{\\rm pilot}|\\sqrt{T/T_{\\rm pilot}}$ and scale it to longer durations, which is what 03 does, but not
separate the link weight $S_{ij}$ from the gain $g$ (Appendix B, Status of the predictions). The paper's predicted
power, which uses the **true** gains, is therefore a check of eq. (7) (done in 04), not something a field team
can compute.
"""),
])

# =====================================================================================================
# 01 generate data
# =====================================================================================================
save('01_generate_data.ipynb', [
md("""# 01 Generate data (the simulator)

**Task.** Build random farming communities, simulate them day by day, and split what comes out into:
* `data/measurements/`: only what a field team measures;
* `data/sketch/`: the channel sketch, zero patterns only;
* `data/truth/`: the networks and the full matrices, read later **only** by notebook 04 for scoring.

Nothing in this notebook uses the test. Model (1)-(2) of the paper is implemented in `lib/community.py`:
$$x[k+1]=\\mathbf\\Phi x[k]+\\mathbf\\Gamma u[k]+\\mathbf F\\omega[k]+w[k],\\qquad y[k]=\\mathbf Cx[k]+v[k],$$
$$(\\mathbf\\Phi)_{ii}=\\mathbf\\Phi_i,\\qquad(\\mathbf\\Phi)_{ij}=W_{ij}\\mathbf H_i+P_{ij}\\mathbf E_i\\quad(i\\neq j).$$
"""),
code("""import sys, os, glob, numpy as np
sys.path.insert(0, 'lib')
import community as C, farmlib as FL
np.set_printoptions(precision=2, suppress=True, linewidth=150)

CFG = dict(N=8, communities=list(range(10)), years=5, scenario='regional', seepage=0.0)
# Local-aquifer runs (physical network P): scenario='local', seepage=0.0 or 0.5.
for d in ('data/measurements', 'data/truth', 'data/sketch', 'results'):
    os.makedirs(d, exist_ok=True)
CFG"""),
md("""## The farms and their channels
Each farm has 8 channels (Appendix C): soil moisture `sm`, local groundwater level `gw`, perceived scarcity `bel`,
pumping `pump`, spraying `spr`, phone talk `talk` (**never measured**), canopy temperature `temp` and price attention
`pr` (two decoys). The excitations are $d=0$, the canal-turn deviation $e_i^{\\rm canal}$ (on the farm's weekly turn
day), and $d=1$, the pest-pressure innovation $e_i^{\\rm pest}$ (daily)."""),
code("""# [Model (1)-(2)] build one community: random social network W, heterogeneous farms
Cm = C.build(CFG['N'], seed=0, scenario=CFG['scenario'], seepage=CFG['seepage'])
print('W (truth). Row i = whom farm i follows; rows sum to 1:')
print(Cm['W'])
print('row sums:', Cm['W'].sum(1))"""),
code("""# [A1, common structure] farms have different values but the same zeros
Phi_1, H_1, Gam_1, F_1, p_1 = Cm['agents'][0]
Phi_2, H_2, Gam_2, F_2, p_2 = Cm['agents'][1]
print('same zeros in Phi_1 and Phi_2 :', np.array_equal(Phi_1 != 0, Phi_2 != 0))
print('same values in Phi_1 and Phi_2:', np.allclose(Phi_1, Phi_2))
print('delta_b (pumping from perceived scarcity), farms 1 and 2:', round(p_1['delta_b'], 3), round(p_2['delta_b'], 3))"""),
code("""# [stability, truth side] the guarantee of Thm 2 does not need stability; the power prediction (7) does
rho = max(abs(np.linalg.eigvals(Cm['Phi'])))
print('spectral radius of the true Phi:', round(rho, 3))"""),
md("""## Simulate, then split
`community.simulate` returns everything, including the true innovations (`D['truth']`). `FL.save_measurements`
writes only the measurable parts and drops `D['truth']`. The networks and matrices go to a separate file."""),
code("""# [Model (1)] simulate one community and look at what comes out
D = C.simulate(Cm, CFG['years'] * 365, seed=1000)
print('keys returned by the simulator:', list(D.keys()))
print('measurable channels Y         :', list(D['Y'].keys()), '(no talk: phone talk is never measured)')
print('excitations M_d               :', {d: FL.EXCITATIONS[d] for d in D['M']})
print('shared series Omega           :', list(D['Omega'].keys()))
print('event days per farm per year  : canal', D['Pi'][0][:365].sum(0)[0], '| pest', D['Pi'][1][:365].sum(0)[0])"""),
code("""# write all communities: measurements / truth, plus the channel sketch
for s in CFG['communities']:
    Cm = C.build(CFG['N'], seed=s, scenario=CFG['scenario'], seepage=CFG['seepage'])
    D = C.simulate(Cm, CFG['years'] * 365, seed=1000 + s)
    FL.save_measurements(f'data/measurements/community_{s:02d}.npz', D)   # D['truth'] is NOT written
    FL.save_truth(f'data/truth/community_{s:02d}.npz', Cm)
FL.save_sketch('data/sketch/channel_sketch.npz')       # zero patterns of Phi_i, H_i, Gamma_i, E_i only
print(len(glob.glob('data/measurements/*.npz')), 'measurement files,', len(glob.glob('data/truth/*.npz')), 'truth files')"""),
code("""# check the contract: a measurement file contains no network and no parameters
z = np.load('data/measurements/community_00.npz')
print(sorted(z.keys()))
assert not any(k in z for k in ('W', 'P', 'Phi', 'Gamma')), 'truth leaked into the measurements'
sk = FL.load_sketch('data/sketch/channel_sketch.npz')
print('sketch arrays:', {k: v.dtype for k, v in sk.items()}, '(booleans: zeros only)')"""),
])

# =====================================================================================================
# 02 certify links
# =====================================================================================================
save('02_certify_links.ipynb', [
md("""# 02 Certify links (blind)

**Task.** From the data alone, report which farm follows which, with $\\Pr(\\text{any false link})\\le\\alpha$.

**Reads:** `data/measurements/`, `data/sketch/`. **Never reads:** `data/truth/`.

""" + TAGS),
code(SETUP + """
os.makedirs('results', exist_ok=True)
print(len(files), 'communities; reading only data/measurements and data/sketch')"""),
md("""## Where to look: lags from the channel sketch
The sketch says only which channel moves which. Definition 1 and Lemma 1 give the first lag $\\tau^W$ at which farm
$j$'s excitation $d$ can reach channel $o$ of another farm through one social crossing. The pair $(d,o)$ is tested
at $\\tau_{d,o}=\\tau^W$ only if the social network arrives first: before the physical network and before any chain
with two or more crossings ($\\tau_2$)."""),
code("""# [Def 1, Lemma 1] first lags from the zero pattern alone
_, detail = FL.lags(SK, detail=True)
pd.DataFrame([dict(excitation=FL.EXCITATIONS[d], output=FL.NAMES[o], tau_W=a['W'], tau_2=a['tau2'],
                   tested_lag=TAU.get((d, o), 'not reached')) for (d, o), a in detail.items()])"""),
md("""Canal $\\to$ pumping and perceived scarcity at lag 3; canal $\\to$ soil moisture and groundwater level at lag 4;
pest $\\to$ spraying at lag 2; nothing reaches the decoys. These are the paper's predictions *before any data*
(Section 4)."""),
md("""## Step 1: residuals that use only the past (eq. 4, condition C2)
For each receiver $i$ and output $o$, the predictable part is removed by a ridge regression fitted on earlier rows
only, refitted weekly after a 90-day warm-up. Row $k$ of the regressors $\\mathbf X_{i,o}$ holds:
* a constant;
* $i$'s own daily outputs at the two most recent lags allowed: output $o$ itself at lags 1, 2, and another channel
  $c$ only before $k-\\tau+\\tau^0_c$;
* $i$'s own excitations and canal schedule at lags $1,\\dots,\\tau+1$;
* the shared series at lags $0,\\dots,3$;
* the other farms' daily outputs at lag $\\tau$ and their excitations at lag $\\tau+1$.

None of these can depend on the sender's innovation on day $k-\\tau$ (Appendix B, Step 1 satisfies C2)."""),
code("""PLAN = 'Pump+Spray'; YEARS = 3; T = YEARS * 365
pl = FL.plan(PLAN)
R0 = FL.load_measurements(files[0])
Rv = FL.view(R0, T, pl['outputs'], pl['excitations'], pl['shared'], pl.get('farms_out'))

# [Step 1, eq. (4), C2] online residual of farm 1's spraying, tested at the pest -> spraying lag
d, o = 1, 'spr'; tau = TAU[(d, o)]; i = 0
t0_d = {c: v for (dd, c), v in TAU0.items() if dd == d}
res = K.residuals(Rv, o, i, tau, pl['excitations'], pl['shared'], tau0=t0_d)
raw = Rv['Y'][o][:, i]
print(f'lag tau={tau};  s.d. of spraying {raw[200:].std():.3f}  ->  s.d. of its residual {np.nanstd(res[200:]):.3f}')"""),
md("""## Steps 2-3: the statistic, by hand for one pair
For sender $j$ and receiver $i$, on $j$'s event days $k$:
$$D_k=\\bar{\\mathbf M}_d(k,j)\\,\\bar{\\mathbf R}_o(k,i)=\\frac{m_{j,d}[k]}{\\hat\\sigma_{m,j}[k]}\\cdot\\frac{\\tilde y_{i,o}[k+\\tau]}{\\hat\\sigma_{\\tilde y,i,o}[k]},\\qquad
\\mathbf A_{ij}=\\sum_kD_k,\\qquad \\mathbf V^2_{ij}=\\sum_kD_k^2,\\qquad \\mathbf Z_{ij}=\\mathbf A_{ij}/\\sqrt{\\mathbf V^2_{ij}}.$$
The scales $\\hat\\sigma$ are root-mean-squares over the days **before** $k$, which keeps C2. The cell recomputes
$Z$ in a few lines and checks it against the library."""),
code("""# [Steps 2-3, eq. (5)] Z for one pair, by hand, then compared with certify.test_excitation
j, i = 1, 0                                           # sender farm 2 -> receiver farm 1
Tn = Rv['Y'][o].shape[0]; ks = np.arange(90 + 14, Tn - tau)
res = K.residuals(Rv, o, i, tau, pl['excitations'], pl['shared'], tau0=t0_d)
f2 = np.where(np.isnan(res), 0, res)**2
s_y = np.sqrt(np.maximum(np.cumsum(f2)[ks - 1], 1e-12) / np.maximum(np.cumsum(~np.isnan(res))[ks - 1], 1))  # past only
yt = np.where(np.isnan(res[ks + tau]), 0, res[ks + tau])
m = Rv['M'][d][:, j]; ev = Rv['Pi'][d][:, j]
s_m = np.sqrt(np.maximum(np.cumsum(m**2)[ks - 1], 1e-12) / np.maximum(np.cumsum(ev)[ks - 1], 1))              # past only
Dk = (m[ks] / s_m) * (yt / s_y) * ev[ks]
A, V2 = Dk.sum(), (Dk**2).sum()
Z_hand = A / np.sqrt(V2)
Z_lib, _ = K.test_excitation(Rv, d, {o: tau}, ALPHA, pl['excitations'], pl['shared'], tau0=TAU0)
print(f'events n_j={int(ev[ks].sum())}  A={A:.2f}  V^2={V2:.1f}  Z(hand)={Z_hand:.4f}  Z(library)={Z_lib[i, j]:.4f}')
assert abs(Z_hand - Z_lib[i, j]) < 1e-9"""),
md("""## Step 3: all pairs at once (the matrix form of eq. 5)
With data matrices (days $\\times$ farms), the whole test for one excitation and one output is two matrix products:
$$\\mathbf A=\\bar{\\mathbf R}_o^\\top(\\bm\\Pi_d\\odot\\bar{\\mathbf M}_d),\\qquad
\\mathbf V^2=(\\bar{\\mathbf R}_o^{\\odot2})^\\top(\\bm\\Pi_d\\odot\\bar{\\mathbf M}_d)^{\\odot2},\\qquad
\\mathbf Z=\\mathbf A\\oslash\\sqrt{\\mathbf V^2},$$
where $\\bar{\\mathbf R}_o(k,i)$ is receiver $i$'s residual on day $k+\\tau$ and $\\bar{\\mathbf M}_d(k,j)$ is sender $j$'s
excitation, each divided by its root-mean-square over earlier days. Rows of $\\mathbf Z$ are receivers and columns
are senders."""),
code("""# [Step 3, eq. (5), matrix form] Z for all 56 pairs from two matrix products, checked against the library
N = Rv['Y'][o].shape[1]
Rbar = np.zeros((len(ks), N)); Mbar = np.zeros((len(ks), N)); Pi = Rv['Pi'][d][ks].astype(float)
for ii in range(N):                                    # receivers: residual on day k+tau, scaled by its past RMS
    r_i = K.residuals(Rv, o, ii, tau, pl['excitations'], pl['shared'], tau0=t0_d)
    f2 = np.where(np.isnan(r_i), 0, r_i)**2; cnt = np.cumsum(~np.isnan(r_i))
    s_y = np.sqrt(np.maximum(np.cumsum(f2)[ks - 1], 1e-12) / np.maximum(cnt[ks - 1], 1))
    Rbar[:, ii] = np.where(np.isnan(r_i[ks + tau]), 0, r_i[ks + tau]) / s_y
for jj in range(N):                                    # senders: excitation scaled by its past RMS
    m_j = Rv['M'][d][:, jj]; ev_j = Rv['Pi'][d][:, jj]
    s_m = np.sqrt(np.maximum(np.cumsum(m_j**2)[ks - 1], 1e-12) / np.maximum(np.cumsum(ev_j)[ks - 1], 1))
    Mbar[:, jj] = m_j[ks] / s_m
A_mat = Rbar.T @ (Pi * Mbar)
V2_mat = (Rbar**2).T @ (Pi * Mbar)**2
Z_mat = A_mat / np.sqrt(V2_mat); np.fill_diagonal(Z_mat, np.nan)
print('largest difference between the matrix form and the library over all pairs:', float(np.nanmax(np.abs(Z_mat - Z_lib))))
print('Z (rows = receivers, columns = senders):'); print(np.round(Z_mat, 1))"""),
md("""## The threshold and the decisions (Theorem 2)
With $n_p=N(N-1)=56$ ordered pairs, the level is split over the plan's excitations ($\\alpha/|\\mathcal D|$ each). The
threshold is
$$c_\\alpha=\\sqrt{2\\kappa\\log(2n_pn_\\kappa/\\alpha)},\\qquad n_\\kappa=\\lceil\\log_\\kappa(b_{\\rm hi}/b_{\\rm lo})\\rceil,$$
with $[b_{\\rm lo},b_{\\rm hi}]=[n_j/100,\\,100\\,n_j]$ and $\\kappa$ chosen to make $c_\\alpha$ smallest. A pair is
reported if $|\\mathbf Z_{ij}|\\ge c_\\alpha$ and $\\mathbf V^2_{ij}\\in[b_{\\rm lo},b_{\\rm hi}]$."""),
code("""# [Thm 2, eq. (6)] threshold c_alpha
n_p = N * (N - 1); alpha_d = ALPHA / len(pl['excitations'])
c, kappa, n_kappa = K.threshold(n_p, alpha_d)
print(f'n_p={n_p}, alpha per excitation={alpha_d}:  kappa={kappa:.2f}, n_kappa={n_kappa}, c_alpha={c:.3f}')
print('recomputed from the formula:', round(float(np.sqrt(2 * kappa * np.log(2 * n_p * n_kappa / alpha_d))), 3))"""),
code("""# [Thm 2, Algorithm 1] certify every community for several plans and durations, with adjusted p-values
RUN = [('Pump+Spray', y) for y in (1, 3, 5)] + [('All practical', 3), ('All practical', 5), ('Spray+Pest', 3)]
out = []
for f in files:
    name = os.path.splitext(os.path.basename(f))[0]; R = FL.load_measurements(f)
    for plan_name, yrs in RUN:
        p_ = FL.plan(plan_name)
        Rv_ = FL.view(R, yrs * 365, p_['outputs'], p_['excitations'], p_['shared'], p_.get('farms_out'))
        W_hat, Zs, infos = K.certify_plan(Rv_, p_, TAU, ALPHA, tau0=TAU0)
        out.append(dict(community=name, plan=plan_name, years=yrs, W_hat=W_hat, Z=Zs,
                        p=K.p_values(Zs, infos, ALPHA), info=infos))
pickle.dump(out, open('results/links.pkl', 'wb'))
pd.DataFrame([dict(community=r['community'], plan=r['plan'], years=r['years'], links_reported=int(r['W_hat'].sum()))
              for r in out]).pivot_table(index='community', columns=['plan', 'years'], values='links_reported')"""),
md("""The table shows **how many** links each run reported. Whether they are right is unknown here; that is
scored in notebook 04, the only notebook that opens the truth."""),
code("""# the reported links of one community, with family-wise adjusted p-values (blind)
r = [x for x in out if x['community'] == 'community_00' and x['plan'] == 'Pump+Spray' and x['years'] == 3][0]
rep = [(f'{j+1}->{i+1}', f"{float(r['p'][i, j]):.1e}") for i in range(N) for j in range(N) if r['W_hat'][i, j]]
print(len(rep), 'reported links (sender->receiver, adjusted p-value):'); print(rep)
assert all(float(p) <= ALPHA for _, p in rep)"""),
md("""## Without a sketch: the lag scan (Proposition 3)
Scan lags $\\tau=1,\\dots,\\bar L=6$ for each (excitation, output) pair at level $\\alpha/(\\bar L|\\mathcal D||\\mathcal O|)$,
and stop at the first lag with a report. Lags below the true lag $\\tau^\\ast$ are safe; the only risk is passing
$\\tau^\\ast$ without a report."""),
code("""# [Prop 3] lag scan, ignoring the sketch
scan = []
for f in files:
    name = os.path.splitext(os.path.basename(f))[0]; R = FL.load_measurements(f)
    for yrs in (3, 5):
        p_ = FL.plan('Pump+Spray')
        Rv_ = FL.view(R, yrs * 365, p_['outputs'], p_['excitations'], p_['shared'])
        W_hat, found = K.lag_scan(Rv_, p_, ALPHA)
        scan.append(dict(community=name, years=yrs, W_hat=W_hat, found=found))
pickle.dump(scan, open('results/scan.pkl', 'wb'))
pd.DataFrame([dict(community=s['community'], years=s['years'], links_reported=int(s['W_hat'].sum()),
                   **{f"{['canal', 'pest'][d]}->{FL.NAMES[o]}": ('-' if s['found'][(d, o)] is None else s['found'][(d, o)])
                      for (d, o) in s['found']}) for s in scan])"""),
md("""Where the scan finds something, it stops at the sketch's lag: lag 2 for pest $\\to$ spraying and, with enough
data, lag 3 for canal $\\to$ pumping. `-` means nothing was reported at any lag yet: inconclusive, never wrong."""),
])

# =====================================================================================================
# 03 what to measure
# =====================================================================================================
save('03_what_to_measure.ipynb', [
md("""# 03 What to measure (blind)

**Task.** Follow the Procedure of Section 3 on a **pilot**: measure every candidate channel for 2 years, certify
which (excitation, channel) pairs carry influence (Corollary 4), predict from the pilot how each plan would do if
it ran longer (eq. 7), and choose a plan. Then check the prediction against what the longer data report.

**Reads:** `data/measurements/` (the first 2 years as the pilot), `data/sketch/`. **Never reads:** `data/truth/`.

""" + TAGS),
code(SETUP + """
from scipy.stats import norm
PILOT_YEARS = 2"""),
md("""## Step 1 of the Procedure: pilot, then channel certificates (Corollary 4)
For each candidate pair $(d,o)$, the lag scan runs over all farm pairs at level $\\alpha/(\\bar L|\\mathcal D||\\mathcal O|)$.
The pair is certified if any farm pair is reported at any lag. The probability of certifying any pair that is
**not reached** is at most $\\alpha$, with no failure term."""),
code("""# [Cor 4] channel certificates from a 2-year pilot in which every candidate channel is measured
cert = []
for f in files:
    R = FL.load_measurements(f)
    Rp = FL.view(R, PILOT_YEARS * 365, FL.CANDIDATES, [0, 1])
    cert.append(K.channel_certificates(Rp, FL.CANDIDATES, [0, 1], FL.SHARED, ALPHA))
rows = []
for (d, o) in cert[0]:
    found = [c[(d, o)] for c in cert if c[(d, o)] is not None]
    rows.append(dict(excitation=FL.EXCITATIONS[d], channel=FL.NAMES[o],
                     share_certified=np.mean([c[(d, o)] is not None for c in cert]),
                     lag_found=sorted(set(found)) if found else '-', sketch_lag=TAU.get((d, o), 'not reached')))
pd.DataFrame(rows)"""),
md("""Read this as Corollary 4 says: a certified pair is **useful**; a pair that is not certified **may still carry
influence** that the pilot was too short to show. A pair can be certified useful but never useless. The decoys and
every pair that the sketch says is not reached should (almost) never be certified."""),
md("""## Step 2: predict each plan from the pilot (eq. 7)
For each farm pair, the pilot gives $Z^{\\rm pilot}_{ij}$. By eq. (7) the mean of $Z$ grows like $\\sqrt{T}$, so a plan
that runs for $T$ days is predicted to give $\\hat\\mu_{ij}(T)=|Z^{\\rm pilot}_{ij}|\\sqrt{T/T_{\\rm pilot}}$, and to report the
pair with probability $\\Pr(|\\mathcal N(\\hat\\mu,1)|\\ge c_\\alpha)$, combined over excitations. Summing over pairs
predicts how many links the plan will report.

*Limit.* This uses only the pilot. It cannot predict links that are invisible in the pilot, and it cannot separate
the link weight from the gain, so it cannot give the paper's "years to find 90% of links" (Table 3). Notebook 04
checks eq. (7) with the true gains."""),
code("""# [eq. (7), scaled from the pilot] predicted number of reports after longer durations, for each plan
PLANS_CMP = ['Spray+Pest', 'Pump+Spray', 'Pump', 'SM+Pump', 'SM', 'Survey', 'Decoys']
HORIZONS = [3, 5]
pred = []
for f in files:
    R = FL.load_measurements(f)
    for plan_name in PLANS_CMP:
        p_ = FL.plan(plan_name)
        Rp = FL.view(R, PILOT_YEARS * 365, p_['outputs'], p_['excitations'], p_['shared'], p_.get('farms_out'))
        W_hat, Zs, infos = K.certify_plan(Rp, p_, TAU, ALPHA, tau0=TAU0)
        row = dict(community=os.path.splitext(os.path.basename(f))[0][-2:], plan=plan_name, pilot_reports=int(W_hat.sum()))
        for Hy in HORIZONS:
            q = None
            for d, Zd in Zs.items():
                mu = np.abs(np.nan_to_num(Zd)) * np.sqrt(Hy / PILOT_YEARS)
                c = infos[d]['c_alpha']
                pr = norm.cdf(mu - c) + norm.cdf(-mu - c); np.fill_diagonal(pr, 0)
                q = (1 - pr) if q is None else q * (1 - pr)
            row[f'pred_{Hy}y'] = float((1 - q).sum()) if q is not None else 0.0
            # what Hy years of data actually report (still blind: counts only)
            Rh = FL.view(R, Hy * 365, p_['outputs'], p_['excitations'], p_['shared'], p_.get('farms_out'))
            row[f'obs_{Hy}y'] = int(K.certify_plan(Rh, p_, TAU, ALPHA, tau0=TAU0)[0].sum())
        pred.append(row)
P = pd.DataFrame(pred)
P.groupby('plan')[['pilot_reports'] + [f'{a}_{h}y' for h in HORIZONS for a in ('pred', 'obs')]].mean().round(1).loc[PLANS_CMP]"""),
code("""# [Procedure, step 3] choose the plan predicted to report the most links within the budget
BUDGET_YEARS = 3
best = P.groupby('plan')[f'pred_{BUDGET_YEARS}y'].mean().sort_values(ascending=False)
print(f'predicted links reported after {BUDGET_YEARS} years, per community:'); print(best.round(1))
print('\\nchosen plan:', best.index[0])"""),
md("""The pilot alone recommends measuring **spraying with pest scouting** (plus the shared series). It does
**not** certify soil moisture, groundwater level, the survey or the decoys, and it predicts that they would report
almost nothing within 5 years. That is not a certificate that they are useless (Corollary 4 never gives one); it is
a prediction that they are not worth the cost, which is the paper's conclusion in Section 4.

The predicted counts track what the longer data report. For weak plans (Pump) the pilot-scaled prediction runs
high, because $|Z^{\\rm pilot}|$ overstates a small $\\mu$: noise only adds to an absolute value."""),
])

# =====================================================================================================
# 04 score and compare
# =====================================================================================================
save('04_score_and_compare.ipynb', [
md("""# 04 Score and compare (the only notebook that opens the truth)

**Task.** Score what notebooks 02 and 03 reported against the true networks. Run DMDc, output Granger, conditional
Granger and Lasso-Granger on the **same data**. Compare the power predicted by eq. (7) with the true gains with the
power observed. Show what a wrong channel sketch does.

**Reads:** `results/`, `data/truth/`, and `data/measurements/` (for the baselines and the wrong-sketch run).

""" + TAGS),
code(SETUP + """
import scoring as SC, netplot as NP
truth = {os.path.splitext(os.path.basename(f))[0]: FL.load_truth(f) for f in sorted(glob.glob('data/truth/*.npz'))}
links = pickle.load(open('results/links.pkl', 'rb')); scan = pickle.load(open('results/scan.pkl', 'rb'))
print(len(truth), 'true networks;', 'mean number of links per community:',
      np.mean([((t['W'] > 0) & ~np.eye(len(t['W']), dtype=bool)).sum() for t in truth.values()]))"""),
md("""## Is the guarantee met? (checks Theorem 2)
FWER is the share of communities with **any** false link; Theorem 2 says it is at most $\\alpha=0.1$. FP is the
number of false links per community and FDR the false-discovery rate."""),
code("""# [checks Thm 2] score every run of notebook 02
S = pd.DataFrame([dict(plan=r['plan'], years=r['years'], **SC.score(r['W_hat'], truth[r['community']]['W'])) for r in links])
S.groupby(['plan', 'years']).agg(FWER=('any_false', 'mean'), FP=('FP', 'mean'), precision=('precision', 'mean'),
                                 recall=('recall', 'mean'), F1=('F1', 'mean'), FDR=('FDR', 'mean')).round(3)"""),
code("""# [checks Prop 3] the lag scan, scored
pd.DataFrame([dict(years=s['years'], **SC.score(s['W_hat'], truth[s['community']]['W'])) for s in scan]) \\
  .groupby('years').agg(FWER=('any_false', 'mean'), recall=('recall', 'mean')).round(3)"""),
md("""## Against standard identification, on the same data (Table 2)
All methods receive soil moisture, groundwater level, pumping and spraying (the weekly survey is not given to the
baselines), the measured excitations, the canal schedule and the shared series. DMDc is a one-step least-squares
regression on all of these; output Granger uses the outputs at lags 1-3; conditional Granger adds the excitations,
canal schedules and shared series as regressors, with four lags and HAC Wald tests; Lasso-Granger fits the DMDc
regressors at lags 1 and 2 with the penalty chosen by BIC (Appendix C)."""),
code("""BASE_OUTS = ['sm', 'gw', 'pump', 'spr']
base = []
for name, t in truth.items():
    R = FL.load_measurements(f'data/measurements/{name}.npz')
    Rb = FL.view(R, 3 * 365, BASE_OUTS, [0, 1])
    res = K.baselines(Rb, BASE_OUTS, ALPHA); res['conditional_granger'] = K.conditional_granger(Rb, BASE_OUTS, 4, ALPHA)
    for meth, W_hat in res.items():
        base.append(dict(method=meth, years=3, **SC.score(W_hat, t['W'])))
ours = S[(S.plan == 'All practical') & (S.years == 3)].assign(method='ours')
B = pd.concat([ours, pd.DataFrame(base)])
B.groupby('method').agg(FWER=('any_false', 'mean'), FP=('FP', 'mean'), precision=('precision', 'mean'),
                        recall=('recall', 'mean'), F1=('F1', 'mean'), FDR=('FDR', 'mean')).round(2)"""),
md("""## The true network and the recovered ones
Farms sit in order along the watercourse, from the head (farm 1) to the tail. An arrow $j\\to i$ means farm $i$
follows farm $j$, and its width shows the true weight $W_{ij}$. On the recovered panels:
* **blue** = found (true and reported);
* **grey dashed** = missed (true, not reported);
* **red** = false (reported, not true).

Change `COMMUNITY` to look at another community."""),
code("""# [Thm 2, visually] the true network and what the test recovers as the measurement duration grows
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif'})
COMMUNITY = 'community_00'
W = truth[COMMUNITY]['W']
get = lambda plan, yrs: [r for r in links if r['community'] == COMMUNITY and r['plan'] == plan and r['years'] == yrs][0]['W_hat']
fig, ax = plt.subplots(1, 4, figsize=(13, 3.9))
NP.draw_network(ax[0], W, None, 'true network')
for k, yrs in enumerate((1, 3, 5)):
    NP.draw_network(ax[k + 1], W, get('Pump+Spray', yrs), f'ours (pumping + spraying), {yrs} yr')
NP.network_legend(fig); fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(os.path.join('results', f'network_{COMMUNITY}_ours.png'), dpi=150); plt.show()"""),
code("""# the same community: ours against standard identification, all on the same 3 years of data (cf. Figure 3)
R = FL.load_measurements(os.path.join('data', 'measurements', f'{COMMUNITY}.npz'))
bd = K.baselines(FL.view(R, 3 * 365, BASE_OUTS, [0, 1]), BASE_OUTS, ALPHA)
fig, ax = plt.subplots(1, 4, figsize=(13, 3.9))
NP.draw_network(ax[0], W, get('All practical', 3), 'ours, 3 yr')
for k, (m, lab) in enumerate([('dmdc', 'DMDc'), ('output_granger', 'output Granger'), ('lasso_granger', 'Lasso-Granger')]):
    NP.draw_network(ax[k + 1], W, bd[m], f'{lab}, 3 yr')
NP.network_legend(fig); fig.tight_layout(rect=(0, 0.06, 1, 1))
fig.savefig(os.path.join('results', f'network_{COMMUNITY}_baselines.png'), dpi=150); plt.show()"""),
md("""Our test draws no red arrow; with more data the grey arrows turn blue. The baselines find nearly every true
link, but surround the true links with false ones. Many of the false arrows join farms that are two steps apart
along a path (indirect paths), or farms that share the same water and weather."""),
md("""## Does eq. (7) predict the power? (with the true gains)
Using the **true** impulse responses at the tested lags (eq. 3) and the residual spreads observed by the test, eq.
(7) predicts each pair's chance of being found. Compare the mean over true links with the observed share found."""),
code("""# [eq. (7), true gains] predicted against observed power
rows = []
for r in links:
    t = truth[r['community']]; true = (t['W'] > 0) & ~np.eye(len(t['W']), dtype=bool)
    pf = SC.predicted_found(SC.with_true_gains(r['info'], t))
    rows.append(dict(plan=r['plan'], years=r['years'], predicted=pf[true].mean(), observed=SC.score(r['W_hat'], t['W'])['recall']))
pd.DataFrame(rows).groupby(['plan', 'years'])[['predicted', 'observed']].mean().round(3)"""),
md("""## What a wrong channel sketch does (Proposition 3 against a fixed wrong lag)
A sketch with an extra arrow says that pest pressure acts on spraying through perceived scarcity, so lag 3, while
the true lag is 2. At lag 3 the indirect paths $j\\to l\\to i$ have arrived (Table 4)."""),
code("""# [Prop 3 vs a wrong lag] spraying only: lag 3 from a wrong sketch, lag 2 from the true sketch, and the lag scan
rows = []
for name, t in truth.items():
    R = FL.load_measurements(f'data/measurements/{name}.npz')
    for yrs in (3, 5):
        Rv = FL.view(R, yrs * 365, ['spr'], [1])
        _, W_wrong = K.test_excitation(Rv, 1, {'spr': 3}, ALPHA, [1], FL.SHARED, tau0=TAU0)
        _, W_true = K.test_excitation(Rv, 1, {'spr': 2}, ALPHA, [1], FL.SHARED, tau0=TAU0)
        W_scan, _ = K.lag_scan(Rv, FL.plan('Spray+Pest'), ALPHA)
        for lab, Wh in (('wrong sketch, lag 3', W_wrong), ('true sketch, lag 2', W_true), ('lag scan', W_scan)):
            rows.append(dict(method=lab, years=yrs, **SC.score(Wh, t['W'])))
pd.DataFrame(rows).groupby(['method', 'years']).agg(FWER=('any_false', 'mean'), FP=('FP', 'mean'),
                                                    recall=('recall', 'mean')).round(2)"""),
md("""## The paper's full runs (40 communities, up to 10 years)
The paper's numbers come from the scripts in `paper_runs/` (see `paper_runs/README.md`). Their saved results are
included, so the cells below only load them. `paper_runs/farm/farm_tables.py` writes Tables 2-4 from the same
file."""),
code("""import json
F = json.load(open(os.path.join('paper_runs', 'farm', 'results', 'farm_summary.json')))
print('farm: all 11 plans x 5 durations x 40 communities -> runs with a false link =',
      F['plans_total']['runs_with_false'], '/', F['plans_total']['runs'])
print('farm: placebo statistics, mean %.3f, s.d. %.3f over %d pairs' % (F['placebo']['mean'], F['placebo']['sd'], F['placebo']['n']))
pd.DataFrame([r for r in F['plans'] if r['yrs'] != 'T90']).pivot_table(index='plan', columns='yrs', values='recall').round(2)"""),
code("""# Table 2 (full run): every method on the same data
pd.DataFrame(F['baselines']).set_index(['method', 'yrs'])[['FWER', 'FP', 'precision', 'recall', 'F1', 'FDR']].round(2)"""),
code("""# generic benchmark (Appendix D): validity and power against measurement duration, 40 networks of 20 agents
G = json.load(open(os.path.join('paper_runs', 'generic', 'paper_summary.json')))
pd.DataFrame(G['E1']['table'])[['T', 'FWER_0.05', 'FWER_0.1', 'FWER_0.2', 'power_0.1', 'pred_0.1', 'naive_FWER']].round(3)"""),
])

# =====================================================================================================
# 05 hand example
# =====================================================================================================
save('05_hand_example.ipynb', [
md("""# 05 The whole method by hand: 3 farms, 2 channels, 50 days

A small worked example with the same separation as notebooks 01-04: data generation writes measurements and
truth; the test reads only the measurements; scoring reads the truth.

""" + TAGS),
code("""import os, numpy as np, pandas as pd
from math import log, sqrt, ceil
os.makedirs('data/hand', exist_ok=True)
np.set_printoptions(precision=3, suppress=True, linewidth=150)"""),
md("""## Data generation (truth side)
The channels are (perceived scarcity, pumping). The canal innovation enters perceived scarcity, which raises
pumping, and only pumping is copied ($\\mathbf H_i$ has one non-zero entry). Farm 2 follows farm 1, farm 3 follows
farm 2, and farm 1 follows farm 3."""),
code("""# [Model (1)-(2)] the true system
Phi = [np.array([[0.2, 0], [1.0, 0.1]]), np.array([[0.2, 0], [0.8, 0.1]]), np.array([[0.2, 0], [1.2, 0.1]])]
theta = [0.5, 0.6, 0.6]; H = [np.diag([0, x]) for x in theta]
W = np.zeros((3, 3)); W[1, 0] = 1; W[2, 1] = 1; W[0, 2] = 1
gam = np.array([1.0, 0.0]); c_pump = np.array([0.0, 1.0])
Big = np.zeros((6, 6))
for i in range(3):
    Big[2*i:2*i+2, 2*i:2*i+2] = Phi[i]
    for j in range(3):
        if i != j: Big[2*i:2*i+2, 2*j:2*j+2] = W[i, j] * H[i]
G = np.zeros((6, 3))
for j in range(3): G[2*j:2*j+2, j] = gam
print(Big)
print('spectral radius:', round(max(abs(np.linalg.eigvals(Big))), 3))"""),
code("""# simulate 50 event days; the residual is pumping minus the response to the farm's own excitation
def h(i, j, tau):
    \"\"\"[eq. (3)] impulse response from farm j's input to farm i's pumping at lag tau (truth side).\"\"\"
    return c_pump @ np.linalg.matrix_power(Big, tau - 1)[2*i:2*i+2, :] @ G[:, j]
rng = np.random.default_rng(20); n_ev = 50; burn = 60; T = burn + n_ev + 8
e = rng.choice([-1.0, 1.0], size=(T, 3)); x = np.zeros(6); Y = np.zeros((T, 3))
for k in range(T):
    Y[k] = [c_pump @ x[2*i:2*i+2] for i in range(3)]
    x = Big @ x + G @ e[k] + 0.1 * rng.standard_normal(6)
Y += 0.05 * rng.standard_normal(Y.shape)
Res = Y.copy()
for i in range(3):
    for s in range(1, 7): Res[s:, i] -= h(i, i, s) * e[:-s, i]
k0 = burn - 1
m = e[k0+1:k0+n_ev+1]; yt = np.round(Res[k0+1:k0+n_ev+5], 2)
np.savez('data/hand/measurements.npz', m=m, ytilde=yt)      # what the test sees
np.savez('data/hand/truth.npz', W=W, Big=Big)               # what only scoring sees
print('measurements:', m.shape, yt.shape)"""),
md("""## The test (reads only `data/hand/measurements.npz`)"""),
code("""z = np.load('data/hand/measurements.npz'); m, yt = z['m'], z['ytilde']; n = m.shape[0]
# [Def 1, Lemma 1] from the sketch: one step to enter perceived scarcity, one internal step to pumping, one to cross
TAU = 3
# [Thm 2, eq. (6)] threshold for n_p = 6 pairs, alpha = 0.1, b_hi / b_lo = 1e4
def threshold(n_p, alpha, ratio=1e4):
    best = None
    for kappa in np.linspace(1.02, 1.6, 59):
        n_kappa = ceil(log(ratio) / log(kappa)); c = sqrt(2 * kappa * log(2 * n_p * n_kappa / alpha))
        if best is None or c < best[0]: best = (c, kappa, n_kappa)
    return best
c, kappa, n_kappa = threshold(6, 0.1)
print(f'kappa={kappa:.2f}  n_kappa={n_kappa}  c_alpha={c:.2f};  fewest events that can be reported: c^2 = {c*c:.1f}')
def Z(j, i, tau, Y=yt):
    D = m[:, j] * Y[tau:tau + n, i]; return D.sum(), (D**2).sum(), D.sum() / np.sqrt((D**2).sum())
rows = []
for j in range(3):
    for i in range(3):
        if i == j: continue
        A, V2, z_ = Z(j, i, TAU)
        rows.append(dict(pair=f'{j+1}->{i+1}', A=round(A, 2), V2=round(V2, 4), Z=round(z_, 2), reported=abs(z_) >= c,
                         p_value=min(1, 2 * 6 * n_kappa * np.exp(-z_**2 / (2 * kappa)))))
pd.DataFrame(rows)"""),
code("""# [Prop 3] lag scan over lags 1..4, each at level alpha / 4
c4 = threshold(6, 0.1 / 4)[0]
tab = pd.DataFrame({f'{j+1}->{i+1}': [round(Z(j, i, t)[2], 2) for t in (1, 2, 3, 4)]
                    for j in range(3) for i in range(3) if i != j}, index=[f'tau={t}' for t in (1, 2, 3, 4)]).T
print(f'threshold per lag: {c4:.2f}'); tab"""),
md("""## Scoring (reads the truth)"""),
code("""t = np.load('data/hand/truth.npz')
reported = {r['pair'] for r in rows if r['reported']}
true = {f'{j+1}->{i+1}' for i in range(3) for j in range(3) if t['W'][i, j] > 0}
print('reported:', sorted(reported)); print('true    :', sorted(true)); print('exact match:', reported == true)
# [eqs. (7)-(8)] with innovations of +-1 (sigma_e = lambda = K = 1, r = 1): mu = sqrt(n) h / sigma_ytilde,
# estimated as h n / sqrt(V^2); measurement duration for 90% power: (c_alpha + z_0.1)^2 V^2 / (n h^2)
for (j, i) in [(0, 1), (1, 2), (2, 0)]:
    g = h(i, j, 3); A, V2, z_ = Z(j, i, 3)
    print(f'{j+1}->{i+1}: h={g:.2f}  predicted mu={g*n/np.sqrt(V2):.2f}  observed Z={z_:.2f}  '
          f'events for 90% power={(c+1.2816)**2*(V2/n)/g**2:.0f}')"""),
code("""# the true and the recovered network (3 farms)
import sys; sys.path.insert(0, 'lib'); import netplot as NP
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif'})
W_hat = np.zeros((3, 3), bool)
for r in rows:
    if r['reported']:
        j, i = [int(x) - 1 for x in r['pair'].split('->')]; W_hat[i, j] = True
fig, ax = plt.subplots(1, 2, figsize=(7, 3.6))
NP.draw_network(ax[0], t['W'], None, 'true network')
NP.draw_network(ax[1], t['W'], W_hat, 'recovered (lag 3)')
NP.network_legend(fig); fig.tight_layout(rect=(0, 0.1, 1, 1)); plt.show()"""),
])
print('notebooks written')
