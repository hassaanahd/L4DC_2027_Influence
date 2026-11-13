"""Write the body of Table 5 (Appendix D, generic benchmark) from paper_summary.json and extra_summary.json.
Usage: python bench_table.py  ->  tables/tab_bench.tex"""
import os, json, math
HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, 'paper_summary.json'))); X = json.load(open(os.path.join(HERE, 'extra_summary.json')))
OUT = os.path.join(HERE, 'tables'); os.makedirs(OUT, exist_ok=True)
pc = lambda x, d=0: f'{100 * x:.{d}f}\\%'
nfalse = lambda rows: sum(round(r['FWER'] * r['runs']) for r in rows)
runs = lambda rows: sum(r['runs'] for r in rows)

E1 = S['E1']['table']; R40 = 40
e1_false = max(round(r[f'FWER_{a}'] * R40) for r in E1 for a in (0.05, 0.1, 0.2))
e1_gap = max(abs(r['power_0.1'] - r['pred_0.1']) for r in E1)
naive = [round(r['naive_FWER'] * R40) for r in E1]
pw = {r['T']: r['power_0.1'] for r in E1}
st = {r['stress']: r for r in S['stress']}
S1, S2, S3, S4 = (S[k]['table'] for k in ('S1', 'S2', 'S3', 'S4'))
s1_gap = max(abs(r['power'] - r['pred']) for r in S1)
ring = [r for r in S2 if r['level'] == '1.0' and r['T'] == 4000][0]
s3 = {(r['level'], r['T']): r for r in S3}
s4 = {(r['level'], r['T']): r for r in S4}
s4_4000 = [r['power'] for r in S4 if r['T'] == 4000]
lag = {(r['T'], r['test']): r for r in X['lag']}
seq = [lag[(T, 'seq lag found')]['lags'] for T in (2000, 4000)]
agg = {r['Delta']: r for r in S['aggregation']}

rows = [
    f"validity and power, $T=1000$--$6000$ steps, $\\alpha\\in\\{{0.05,0.1,0.2\\}}$ & {e1_false}/{R40} at each & "
    f"power {pc(pw[1000])} $\\to$ {pc(pw[3000])} $\\to$ {pc(pw[6000], 1)}, predicted within {math.ceil(100 * e1_gap)} point; "
    f"a Gaussian threshold gives false links in {min(naive)}--{max(naive)} of {R40} runs\\\\",
    "Gaussian / $t_3$ / $t_{2.5}$ innovations & "
    + ('0/40 each' if all(st[k]['ours']['FWER'] == 0 for k in ('Gaussian innovations', 'heavy tails t(3)', 'heavier tails t(2.5)')) else '?')
    + f" & power {pc(st['Gaussian innovations']['ours']['power'])}, {pc(st['heavy tails t(3)']['ours']['power'])}, "
      f"{pc(st['heavier tails t(2.5)']['ours']['power'])}\\\\",
    f"shared input and weather $\\times3$ & {round(st['shared drivers x3']['ours']['FWER'] * 40)}/40 & "
    f"power {pc(st['shared drivers x3']['ours']['power'])}; Granger {st['shared drivers x3']['granger']['false_per_net']:.0f} false links per network\\\\",
    f"unmeasured drivers $\\times3$; heterogeneity $\\pm33\\%$ & {round(st['unmeasured drivers x3']['ours']['FWER'] * 40)}/40; "
    f"{round(st['heterogeneity 33%']['ours']['FWER'] * 40)}/40 & power {pc(st['unmeasured drivers x3']['ours']['power'])}; "
    f"{pc(st['heterogeneity 33%']['ours']['power'])}\\\\",
    f"contacts per agent from 1--2 to 6--8 & {nfalse(S1)}/{runs(S1)} & predicted within {math.ceil(100 * s1_gap)} points; "
    f"up to {max(r['twohop_pairs'] for r in S1):.0f} indirect pairs per network, "
    + ('none reported' if all(r['false_twohop'] == 0 for r in S1) else 'some reported') + '\\\\',
    f"links moved to non-neighbours (ring, $N=40$) & {nfalse(S2)}/{runs(S2)} & Granger and DMDc report "
    f"{ring['granger_false_geo']:.0f}--{ring['arx_false_geo']:.0f} false links among unlinked neighbours\\\\",
    f"channels per agent $n=3$ to 8 & {nfalse(S3)}/{runs(S3)} & power {pc(s3[('3', 2000)]['power'])} $\\to$ "
    f"{pc(s3[('8', 2000)]['power'])} at $T=2000$\\\\",
    f"agents $N=20$ to 500 ({500 * 499:,} pairs)".replace(',', '{,}') + f" & {nfalse(S4)}/{runs(S4)} & threshold "
    f"{s4[('20', 2000)]['threshold']:.2f} $\\to$ {s4[('500', 2000)]['threshold']:.2f}; power at $T=4000$ stays "
    f"{100 * min(s4_4000):.0f}--{100 * max(s4_4000):.0f}\\%\\\\",
    "lag too short / true / too long (two inputs) & "
    f"{max(round(lag[(T, 'tau0-1')]['FWER'] * 40) for T in (2000, 4000))} / {max(round(lag[(T, 'tau0 (true)')]['FWER'] * 40) for T in (2000, 4000))} / "
    f"{round(lag[(2000, 'tau0+1')]['FWER'] * 40)} and {round(lag[(4000, 'tau0+1')]['FWER'] * 40)} of 40 & too long: up to "
    f"{max(lag[(T, 'tau0+1')]['false_per_net'] for T in (2000, 4000)):.1f} false links per network; the lag scan stopped at the true lag in "
    f"{sum(s['2'] for s in seq)}/{sum(sum(s.values()) for s in seq)} runs\\\\",
    f"aggregation over $\\Delta=1,2,4$ periods (A4 fails for $\\Delta>1$) & {round(agg[1]['FWER'] * 20)} / {round(agg[2]['FWER'] * 20)} / "
    f"{round(agg[4]['FWER'] * 20)} of 20 & at $\\Delta=4$ "
    + ('every false link is indirect' if agg[4]['false_twohop'] == agg[4]['false_per_net'] else 'some false links are direct') + '\\\\',
]
open(os.path.join(OUT, 'tab_bench.tex'), 'w').write('\n'.join(rows) + '\n')
print('wrote tables/tab_bench.tex')
