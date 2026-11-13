"""Figure 3 (Appendix C): one community (community 0), the true network and the links recovered from the same 3 years of data
of the 'All practical' plan by our test, DMDc, output Granger and Lasso-Granger. These are the same runs as
community 0 in Table 2. Usage: python fig_network.py  ->  figures/fig_network.pdf"""
import os, sys, warnings
warnings.filterwarnings('ignore')
HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, HERE); sys.path.insert(0, os.path.join(HERE, '..', '..', 'lib'))
import farm_run as F, certify as K, plans as PL, netplot as NP
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
plt.rcParams.update({'font.family': 'serif', 'font.serif': ['Times New Roman', 'Times', 'DejaVu Serif'], 'mathtext.fontset': 'stix'})

Cm, D = F.community(0, 5); T = 3 * 365
ours, _, _, _ = F.run_plan(Cm, D, T, 'All practical')
bd = K.baselines(PL.view(D, T, F.BASE_OUTS, [0, 1]), F.BASE_OUTS, F.ALPHA)
# Two rows of three panels at the full text width (6 in), so that the text prints at about 8 pt.
fig, axs = plt.subplots(2, 3, figsize=(6.0, 4.6)); ax = axs.ravel()
kw = dict(scale=0.55, node_fs=7, title_fs=8)
NP.draw_network(ax[0], Cm['W'], None, '(a) true network', **kw)
NP.draw_network(ax[1], Cm['W'], ours, '(b) ours', **kw)
for k, (m, lab) in enumerate([('dmdc', '(c) DMDc'), ('output_granger', '(d) output Granger'), ('lasso_granger', '(e) Lasso-Granger')]):
    NP.draw_network(ax[k + 2], Cm['W'], bd[m], lab, **kw)
ax[5].axis('off'); ax[5].legend(handles=NP.network_handles(), loc='center', frameon=False, fontsize=8, handlelength=2.2)
fig.tight_layout(h_pad=0.6, w_pad=0.2)
os.makedirs(os.path.join(HERE, 'figures'), exist_ok=True)
out = os.path.join(HERE, 'figures', 'fig_network.pdf')
fig.savefig(out, bbox_inches='tight'); fig.savefig(out.replace('.pdf', '.png'), dpi=130, bbox_inches='tight')
print('saved', out)
