# Learning Influence with Finite-Sample Guarantees in Socio-hydrological Networks from Farmer-Specific Excitations

Code for the paper by Hassaan Ahmed and Abubakr Muhammad (LUMS), submitted to L4DC 2027.

The test certifies links of an influence network from field data: a measured excitation that hits one agent only
(a farm's canal-turn deviation, a pest outbreak) is correlated with another agent's data at the first lag that the
agents' channel structure allows. The probability of reporting any false link is at most a chosen level α, for any
measurement duration and noise distribution. The same test certifies which measurements carry influence, and a
power calculation predicts how long each measurement plan must run.

## Quick start

```bash
pip install -r requirements.txt
python verify.py                       # reruns community 0 of every experiment and checks it against the saved results (~1 min)
cd paper_runs/farm && python farm_analyze.py && python farm_tables.py && python fig_farm.py && python fig_network.py
```
The last line rebuilds Tables 2-4, the numbers quoted in Section 4 (`tables/numbers.md`) and Figures 1 and 3 from the
saved results. To work through the method step by step, run the notebooks in order (below).

## Layout

| Path | Contents |
|---|---|
| `lib/community.py` | the simulated farming community of Section 4 and Appendix C: farm model, networks, simulation, channel sketch, first lags (Definition 1) |
| `lib/certify.py` | the test: residuals (Step 1, eq. 4), the statistic (Steps 2-3, eq. 5), threshold (eq. 6), Algorithm 1, p-values, lag scan (Proposition 3), channel certificates (Corollary 4), and the baselines |
| `lib/plans.py` | the measurement plans of Table 3, the data a plan provides (`view`), the lags τ_{d,o} and τ⁰_c read from a sketch |
| `lib/scoring.py`, `lib/netplot.py` | scoring against the truth, eq. (7) with the true gains, drawing networks (never used by the test) |
| `lib/farmlib.py` | saving and loading measurements, sketch and truth for the notebooks |
| `00_overview.ipynb` ... `05_hand_example.ipynb` | the method step by step on 10 simulated communities (built by `build_notebooks.py`) |
| `paper_runs/farm/` | the runs behind Section 4, Appendix C, Tables 2-4 and Figures 1 and 3, with their saved results |
| `paper_runs/generic/` | the generic benchmark of Appendix D and Table 5, with its saved summaries |
| `verify.py` | reproducibility check against the saved results |

## Paper → code

| In the paper | Script | Saved results |
|---|---|---|
| Table 2 (ours against standard identification) | `paper_runs/farm/farm_tables.py` → `tables/tab_base.tex` | `farm_checks.pkl`, `farm_cgranger.pkl`, `farm_ours_base_data.pkl` |
| Table 3 (measurement plans) | `paper_runs/farm/farm_tables.py` → `tables/tab_plans.tex` | `farm_plans.pkl` |
| Table 4 (certification checks) | `paper_runs/farm/farm_tables.py` → `tables/tab_farmcert.tex` | all `farm_*.pkl` |
| Numbers in Section 4 and Appendix C | `paper_runs/farm/farm_tables.py` → `tables/numbers.md` | `farm_summary.json`, `pool_corr.json` |
| Figure 1 (plans; channel certificates) | `paper_runs/farm/fig_farm.py` | `farm_summary.json` |
| Figure 3 in Appendix C (one community) | `paper_runs/farm/fig_network.py` | recomputed (community 0) |
| Table 5 (generic benchmark) | `paper_runs/generic/bench_table.py` → `tables/tab_bench.tex` | `paper_summary.json`, `extra_summary.json` |

Every `.pkl` and `.json` of the farm study is in `paper_runs/farm/results/`; `farm_analyze.py` turns the `.pkl`
files into `farm_summary.json`. How to rerun everything is in `paper_runs/README.md`.

## Notation: code → paper

| Code | Paper | Meaning |
|---|---|---|
| `W[i, j]`, `P[i, j]` | $W_{ij}$, $P_{ij}$ | social and physical networks; `W[i, j] > 0` means farm i follows farm j |
| `Phi`, `Gamma`, `F` (`community.build`) | $\mathbf\Phi$, $\mathbf\Gamma$, $\mathbf F$ | matrices of model (1)-(2) |
| `Phi_i, H_i, Gamma_i, F_i` (`community.agent`) | $\mathbf\Phi_i$, $\mathbf H_i$, $\mathbf\Gamma_i$ | one farm's blocks |
| `'sm' 'gw' 'bel' 'pump' 'spr' 'talk' 'temp' 'pr'` | $x^{\rm sm}, x^{\rm gw}, x^{\rm bel}, x^{\rm pump}, x^{\rm spr}, x^{\rm talk}, x^{\rm temp}, x^{\rm pr}$ | the eight channels of a farm (Appendix C) |
| `d = 0`, `d = 1` | $e^{\rm canal}$, $e^{\rm pest}$ | excitation (input) index d |
| `phi_sm`, `zeta`, `chi`, `theta_pump`, `delta_b`, `a_mu`, ... (`community.NOM`) | the nominal coefficients printed in the equations of Appendix C | `phi_c`: persistence of channel c; `theta_c`: copying weight of channel c (diagonal of $\mathbf H_i$); the others: couplings inside a farm and input/driver gains |
| `beta_P` | the coefficient 0.1 of $\sum_j P_{ij}(x_j^{\rm gw}-x_i^{\rm gw})$ | strength of the physical coupling (Appendix C) |
| `R['M'][d]`, `R['Pi'][d]` | $\mathbf M_d$, $\boldsymbol\Pi_d$ | excitation values and event days, T × N |
| `R['Y'][o]`, `R['avail'][o]` | $\mathbf Y_o$ and its mask | output o, T × N |
| `R['Omega'][s]`, `R['sched']` | columns of $\boldsymbol\Omega$; canal schedule | regressors |
| `sketch()` | channel sketch | zero patterns of $\mathbf\Phi_i$, $\mathbf H_i$, $\mathbf\Gamma_i$, $\mathbf E_i$ |
| `first_lags(...)['W'], ['P'], ['tau2']` | $\tau^W$, $\tau^P$, $\tau_2$ | first lags of Definition 1 |
| `TAU[(d, o)]` (`plans.lags`) | $\tau_{d,o}$ | lag tested for the pair (d, o) |
| `TAU0[(d, c)]` (`plans.tau0`) | $\tau^0_c$ | first lag at which channel c can respond without a link (Step 1) |
| `residuals(...)`, `ridge` | $\tilde y_{i,o}$, $\varrho_k$ | eq. (4) |
| `test_excitation(...)` → `Z`, `W_hat` | $\mathbf Z_d$, $\hat{\mathbf W}_d$ | eq. (5) and the decision of Step 3 |
| `threshold(n_p, alpha)` → `c_alpha, kappa, n_kappa` | $c_\alpha$, $\kappa$, $n_\kappa$ | eq. (6) |
| `ratio = 1e4` | $b_{\rm hi}/b_{\rm lo}$ | $[b_{\rm lo},b_{\rm hi}]=[n_j/100,\,100\,n_j]$ |
| `n_events` | $n_j$ | number of j's events |
| `p_values(...)` | $\hat\alpha_{ij}$ | family-wise adjusted p-values (Appendix B.4) |
| `lag_scan(..., Lbar=6)` | Proposition 3, $\bar L$ | the test without a sketch |
| `channel_certificates(...)` | Corollary 4 | which (excitation, output) pairs carry influence |
| `impulse_response(...)` | $h_{ij,d,o}[\tau]$ | eq. (3), from the true matrices |
| `scoring.link_mu`, `LAMBDA`, `KURT`, `R_GAIN` | $\mu_{ij}$, $\lambda$, $K$, $r$ | eq. (7) |

The generic benchmark (`paper_runs/generic/`) is a separate, smaller code base with its own names: `Yd`, `Yb`
(decision and belief outputs), `m` (excitations), `ush` (shared input), `wx` (regional weather).

## What the test sees
`certify.py` receives only the data dict of a plan (`plans.view`) and, optionally, lags read from the channel
sketch. It never receives W, P, parameter values, gains or noise variances; these are used only to simulate and
to score. In the notebooks this is enforced by files: only notebook 04 opens `data/truth/`.

## Notebooks
Run `01_generate_data` first, then `02_certify_links`, `03_what_to_measure` and `04_score_and_compare`;
`05_hand_example` is independent. With the defaults (10 communities, 5 years) the set runs in a few minutes.
`python build_notebooks.py` rewrites the notebooks from source; execute them with
`jupyter nbconvert --to notebook --execute --inplace <name>.ipynb`.

## Requirements
Python 3.10 or later with numpy, scipy, pandas, scikit-learn, matplotlib and jupyter (`requirements.txt`).
Tested with Python 3.13, numpy 2.5, scipy 1.18, scikit-learn 1.9, pandas 3.0 and matplotlib 3.11.
