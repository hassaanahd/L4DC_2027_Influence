"""Write the bodies of Tables 2, 3 and 4 and the numbers quoted in the text of Section 4 and Appendix C, all from
results/farm_summary.json (run farm_analyze.py first). Usage: python farm_tables.py
Outputs: tables/tab_base.tex (Table 2), tables/tab_plans.tex (Table 3), tables/tab_farmcert.tex (Table 4),
tables/numbers.md (numbers in the text)."""
import os, json
HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, 'results', 'farm_summary.json')))
OUT = os.path.join(HERE, 'tables'); os.makedirs(OUT, exist_ok=True)
f2 = lambda x: f'{x:.2f}'
f1 = lambda x: f'{x:.1f}'
pct = lambda x: f'{100 * x:.0f}\\%'
write = lambda name, rows: open(os.path.join(OUT, name), 'w').write('\n'.join(rows) + '\n')
by_plan = {(r['plan'], r['yrs']): r for r in S['plans']}
NCOM = by_plan[('Pump', 1)]['runs']

# ---- Table 2: our test against standard identification on the same data ('All practical' outputs, 5-year runs)
cells = lambda r: ' & '.join([f2(r['FWER']), f1(r['FP']), f2(r['precision']), f2(r['recall']), f2(r['F1']), f2(r['FDR'])])
base = {(r['method'], r['yrs']): r for r in S['baselines']}
rows = []
for m, lab in (('ours', 'ours'), ('dmdc', 'DMDc'), ('output_granger', 'output Granger'), ('conditional_granger', 'conditional Granger'),
               ('lasso_granger', 'Lasso-Granger')):
    rows.append(f'{lab} & ' + cells(base[(m, 3)]) + ' & ' + cells(base[(m, 5)]) + '\\\\')
write('tab_base.tex', rows)

# ---- Table 3: measurement plans
LABELS = {'SM': ('canal', 'soil moisture'), 'GW level': ('canal', 'GW level'), 'Pump': ('canal', 'pumping'),
          'SM+Pump': ('canal', 'soil moisture, pumping'), 'SM+Pump(4 farms)': ('canal', 'soil moisture; pumping at 4 farms'),
          'Survey': ('canal', 'weekly belief survey'), 'Spray+Pest': ('pest', 'spraying'),
          'Pump+Spray': ('canal, pest', 'pumping, spraying'), 'All practical': ('canal, pest', 'SM, GW, pumping, spraying, survey'),
          'Pump+Spray, no shared': ('canal, pest', 'pumping, spraying; no shared regressors'),
          'Decoys': ('canal, pest', 'canopy temp., price survey')}
rows = []
for p, (exc, outs) in LABELS.items():
    t90 = by_plan[(p, 'T90')]['predicted_years']
    if all(by_plan[(p, y)]['predicted'] < 0.005 and by_plan[(p, y)]['recall'] == 0 for y in (1, 2, 3, 5, 10)) and p == 'Decoys':
        t90 = 'never'                                              # nothing reaches the decoys (Lemma 1)
    elif t90 is None:
        t90 = '$>200$'
    else:
        t90 = f'{t90:.1f}' if t90 < 10 else f'{t90:.0f}'
    cells = ' & '.join(f"{f2(by_plan[(p, y)]['recall'])}\\,({f2(by_plan[(p, y)]['predicted'])})" for y in (1, 3, 10))
    rows.append(f'{exc} & {outs} & {cells} & {t90}\\\\')
write('tab_plans.tex', rows)

# ---- Table 4: certification checks
tot = S['plans_total']; pl = S['placebo']; sc = {r['yrs']: r for r in S['scan']}; st = S['scan_total']
wl = {r['yrs']: r for r in S['wrong_lag']}; wf = S['wrong_lag_false_links']; ws = S['wrong_lag_scan']
H = S['hidden']; P = {(r['seep'], r['yrs']): r for r in S['physical']}; SOC = [r for r in S['physical'] if r['social'] is not None]
frac = lambda k, n: f'{k}/{n}'
rows = [f"all {len(LABELS)} plans, 1--10 years & {frac(tot['runs_with_false'], tot['runs'])} & --\\\\",
        "placebo: statistics at lag 1 (all pairs null) & -- & " + f"{pl['n']:,}".replace(',', '{,}')
        + f" pairs: $\\hat\\mu_Z={pl['mean']:.3f}$, $\\hat\\sigma_Z={pl['sd']:.3f}$, max $|Z|$ {pl['max_abs']:.2f}\\\\",
        f"lag scan, no sketch & {frac(st['runs_with_false'], st['runs'])} & power {f2(sc[2]['recall'])} / {f2(sc[5]['recall'])} at 2 / 5 years\\\\",
        "lag from a wrong sketch (3 instead of 2) & " + ' / '.join(pct(wl[y]['wrong']['FWER']) for y in (3, 5, 10))
        + ' & at 3 / 5 / 10 years' + ('; all on indirect pairs' if wf['on_two_step_paths'] == wf['false_links'] else '') + '\\\\',
        f"lag scan on the same data & {frac(ws['runs_with_false'], ws['runs'])} & "
        + ('stopped at the true lag in every run' if ws['lags'] == ["{(1, 'spr'): 2}"] else 'lags found: ' + ', '.join(ws['lags'])) + '\\\\',
        f"two farms are not measured & {frac(sum(round(h['FWER'] * h['runs']) for h in H), sum(h['runs'] for h in H))} & "
        f"{H[0]['indirect_pairs_via_hidden']:.2f} indirect pairs through hidden farms\\\\",
        f"local aquifer: social links via pumping, no seepage & {frac(sum(round(r['social']['FWER'] * r['social']['runs']) for r in SOC), sum(r['social']['runs'] for r in SOC))} & "
        + ('no physical neighbour reported as a social link' if all(r['social_false_on_physical_neighbours'] == 0 for r in SOC) else 'physical neighbours reported')
        + ('; with seepage the two lags tie, so pumping is not tested' if all(P[(0.5, y)]['social'] is None for y in (3, 5)) else '') + '\\\\',
        f"local aquifer: physical links via GW level & {frac(sum(round(r['physical']['FWER'] * r['physical']['runs']) for r in P.values()), sum(r['physical']['runs'] for r in P.values()))} & "
        f"power {f2(P[(0.0, 5)]['physical']['recall'])} at lag {P[(0.0, 5)]['lag_physical']} (no seepage), "
        f"{f2(P[(0.5, 5)]['physical']['recall'])} at lag {P[(0.5, 5)]['lag_physical']} (seepage), at 5 years\\\\"]
write('tab_farmcert.tex', rows)

# ---- numbers quoted in the text
ch = {(r['excitation'], r['output'], r['yrs']): r for r in S['channels']}
unreached = [r for r in S['channels'] if not r['reached']]
never = [k for k in {(r['excitation'], r['output']) for r in S['channels'] if r['reached']}
         if all(ch[(k[0], k[1], y)]['certified'] == 0 for y in (1, 2, 3, 5))]
pc = S['pool_corr']; cg = S['cgranger_false_on_2_3_step_paths']; ap = lambda y: by_plan[('All practical', y)]
lines = [
    '# Numbers in the text (farming community)', '',
    '| Where | Quantity | Value |', '|---|---|---|',
    f"| Abstract | years for spraying with pest scouting to find 90% of links | {by_plan[('Spray+Pest', 'T90')]['predicted_years']} |",
    f"| Abstract | years for pump logging alone | {by_plan[('Pump', 'T90')]['predicted_years']} |",
    f"| Sec. 4, Setting | links per community (of {8 * 7} ordered pairs), mean over {NCOM} | {S['links_per_community']:.2f} |",
    f"| Sec. 4, theory | c_alpha with 56 pairs, alpha = 0.1 | {S['c_alpha_56_pairs']:.3f} |",
    f"| Sec. 4, Certified links | runs with a false link, all plans and durations | {tot['runs_with_false']}/{tot['runs']} |",
    f"| Sec. 4, Certified links | largest placebo statistic max abs Z | {pl['max_abs']:.3f} |",
    f"| Sec. 4, Against standard identification | our test without the weekly survey: identical reports | {S['ours_without_survey']['identical_reports']}/{S['ours_without_survey']['runs']} |",
    f"| Sec. 4, Against standard identification | conditional Granger false links on 2- or 3-step paths | {cg['on_paths']}/{cg['false_links']} = {100 * cg['share']:.1f}% |",
    f"| Sec. 4, What to measure | largest gap between predicted and observed power (points) | {100 * S['prediction_gap_max']:.1f} |",
    f"| Sec. 4, What to measure | All practical at 2 years: predicted / observed | {ap(2)['predicted']:.3f} / {ap(2)['recall']:.3f} |",
    f"| Sec. 4, What to measure | pooled residual correlation: median, 5% and 95% quantiles | {pc['median']:.3f}, {pc['q05']:.3f}, {pc['q95']:.3f} |",
    f"| Sec. 4, What to measure | pest -> spraying certified after 1 / 2 years | {ch[('pest', 'spr', 1)]['certified']:.3f} / {ch[('pest', 'spr', 2)]['certified']:.3f} |",
    f"| Sec. 4, What to measure | canal -> pumping certified after 5 years | {ch[('canal', 'pump', 5)]['certified']:.3f} |",
    f"| Sec. 4, What to measure, Fig. 1b | unreached pairs; largest certification rate among them | {len({(r['excitation'], r['output']) for r in unreached})}; {max(r['certified'] for r in unreached):.3f} |",
    f"| Fig. 1b caption | reached pairs never certified | {len(never)} ({', '.join(f'{a} -> {b}' for a, b in sorted(never))}) |",
    f"| Appendix C | largest spectral radius of Phi over 300 communities (regional; local aquifer) | {S['max_spectral_radius_300']:.4f}; {S['max_spectral_radius_300_local']:.4f} |",
]
open(os.path.join(OUT, 'numbers.md'), 'w').write('\n'.join(lines) + '\n')
print('wrote tables/tab_base.tex, tab_plans.tex, tab_farmcert.tex, numbers.md')
