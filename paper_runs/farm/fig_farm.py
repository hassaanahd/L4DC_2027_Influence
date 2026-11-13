"""Figure 1: (a) links found against measurement duration for several plans, with the prediction of eq. (7) using
the true gains; (b) share of communities in which each excitation-channel pair is certified by Corollary 4 after
a pilot of 1-5 years. Reads results/farm_summary.json. Usage: python fig_farm.py  ->  figures/fig_farm.pdf"""
import os, json
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
HERE = os.path.dirname(os.path.abspath(__file__))
S = json.load(open(os.path.join(HERE, 'results', 'farm_summary.json')))
INK = '#0b0b0b'; INK2 = '#52514e'; GRID = '#e6e5e1'
COL = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#8a5cd6', '#d1477a', '#52514e']
plt.rcParams.update({'font.size': 8.5, 'font.family': 'serif', 'axes.edgecolor': INK2, 'axes.labelcolor': INK,
                     'xtick.color': INK2, 'ytick.color': INK2, 'axes.spines.top': False, 'axes.spines.right': False,
                     'axes.grid': True, 'grid.color': GRID, 'grid.linewidth': 0.6, 'legend.frameon': False,
                     'figure.dpi': 200, 'mathtext.fontset': 'stix'})
fig, ax = plt.subplots(1, 2, figsize=(6.0, 3.1))
show = [('Spray+Pest', 'spraying + pest scouting'), ('Pump+Spray', 'pumping + spraying'), ('Pump', 'pump logger'),
        ('SM+Pump', 'soil moisture + pump'), ('SM', 'soil moisture'), ('Pump+Spray, no shared', 'pump + spray,\nno shared regressors'),
        ('Decoys', 'decoy channels')]
for k, (p, lab) in enumerate(show):
    rs = [r for r in S['plans'] if r['plan'] == p and r['yrs'] != 'T90']; y = [r['yrs'] for r in rs]
    ax[0].plot(y, [r['recall'] for r in rs], '-o', color=COL[k], lw=1.4, ms=3, label=lab)
    ax[0].plot(y, [r['predicted'] for r in rs], '--', color=COL[k], lw=0.8)
ax[0].set_xscale('log'); ax[0].set_xticks([1, 2, 3, 5, 10]); ax[0].set_xticklabels(['1', '2', '3', '5', '10']); ax[0].minorticks_off()
ax[0].set_ylim(0, 1.03); ax[0].set_xlabel('measurement duration (years)'); ax[0].set_ylabel('links found')
ax[0].set_title('(a) measurement plans (dashed: predicted)', fontsize=8.5, loc='left', color=INK)
ax[0].legend(fontsize=7.4, loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2, handlelength=1.2, columnspacing=0.7)
ch = S['channels']
pairs = [('canal', 'pump', 'canal $\\to$ pumping'), ('canal', 'bel', 'canal $\\to$ belief survey'),
         ('canal', 'sm', 'canal $\\to$ soil moisture'), ('canal', 'gw', 'canal $\\to$ GW level'), ('pest', 'spr', 'pest $\\to$ spraying')]
# Several pairs lie on the zero line, so hollow markers of decreasing size are drawn concentrically and every
# series stays visible where they coincide.
STY = [dict(ls='-', marker='o', ms=3.5, mfc=None), dict(ls='--', marker='s', ms=7.5, mfc='none'),
       dict(ls=':', marker='^', ms=6, mfc='none'), dict(ls='-.', marker='D', ms=4, mfc='none'), dict(ls='-', marker='o', ms=3.5, mfc=None)]
for k, (d, o, lab) in enumerate(pairs):
    rs = [r for r in ch if r['excitation'] == d and r['output'] == o]; st = STY[k]
    kw = dict(color=COL[k], lw=1.4, ls=st['ls'], marker=st['marker'], ms=st['ms'], label=lab, zorder=3 + k)
    if st['mfc'] == 'none':
        kw.update(mfc='none', mew=1.1)
    ax[1].plot([r['yrs'] for r in rs], [r['certified'] for r in rs], **kw)
yrs = sorted({r['yrs'] for r in ch})
unreached = [max(r['certified'] for r in ch if r['yrs'] == y and not r['reached']) for y in yrs]
n_unreached = len({(r['excitation'], r['output']) for r in ch if not r['reached']})
ax[1].plot(yrs, unreached, color=COL[6], lw=1.2, ls=(0, (1, 1.2)), marker='x', ms=5, mew=1.2,
           label=f'not reached (max of {n_unreached})', zorder=9)
ax[1].set_xticks(yrs); ax[1].set_ylim(-0.06, 1.04); ax[1].set_xlim(0.7, 5.3)
ax[1].set_xlabel('pilot duration (years)'); ax[1].set_ylabel('share of communities certified')
ax[1].set_title('(b) channel certificates', fontsize=8.5, loc='left', color=INK)
ax[1].legend(fontsize=7.4, loc='upper center', bbox_to_anchor=(0.5, -0.24), ncol=2, handlelength=1.2, columnspacing=0.7)
fig.subplots_adjust(left=0.085, right=0.985, top=0.91, bottom=0.37, wspace=0.32)
os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
out = os.path.join(HERE, 'figures', 'fig_farm.pdf')
fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('.pdf', '.png'), bbox_inches='tight')
print('saved', out)
