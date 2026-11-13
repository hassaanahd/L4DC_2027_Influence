"""Drawing a true network and a recovered one (Figure 3 and the notebooks). Arrow j -> i means farm i follows
farm j (W[i, j] > 0). Uses the truth, so it belongs to scoring, never to the test."""
import numpy as np

COL = dict(found='#2a78d6', missed='#b9b8b3', false='#e05a2b', true='#3d3d3a', canal='#cfe3f5', node='#ffffff')


def farm_positions(N):
    """Farms in order along a horseshoe-shaped watercourse, head (farm 1) to tail (farm N)."""
    th = np.linspace(np.deg2rad(225), np.deg2rad(-45), N)
    return np.c_[np.cos(th), np.sin(th)]


def _canal(ax, scale=1.0):
    th = np.linspace(np.deg2rad(238), np.deg2rad(-58), 300)
    ax.plot(np.cos(th), np.sin(th), color=COL['canal'], lw=12 * scale, solid_capstyle='round', zorder=0)


def draw_network(ax, W, W_hat=None, title='', pos=None, show_weights=True, scale=1.0, node_fs=9, title_fs=9):
    """W_hat=None draws the true network. Otherwise each arrow is coloured: found (true and reported), missed
    (true, not reported; dashed), false (reported, not true). `scale` multiplies line widths, arrow heads and the
    canal band, for panels drawn small; node_fs and title_fs are font sizes in points."""
    from matplotlib.patches import FancyArrowPatch, Circle
    N = len(W); pos = farm_positions(N) if pos is None else pos
    _canal(ax, scale)
    true = (W > 0) & ~np.eye(N, dtype=bool)
    edges = []
    for i in range(N):
        for j in range(N):
            if i == j:
                continue
            if W_hat is None:
                if true[i, j]:
                    edges.append((j, i, 'true'))
            elif true[i, j] and W_hat[i, j]:
                edges.append((j, i, 'found'))
            elif true[i, j]:
                edges.append((j, i, 'missed'))
            elif W_hat[i, j]:
                edges.append((j, i, 'false'))
    order = {'missed': 0, 'true': 1, 'found': 2, 'false': 3}
    for j, i, kind in sorted(edges, key=lambda e: order[e[2]]):
        w = W[i, j] if W[i, j] > 0 else 0.3
        lw = (0.8 + 2.6 * w if show_weights else 1.6) * scale
        ax.add_patch(FancyArrowPatch(pos[j], pos[i], connectionstyle='arc3,rad=0.18', arrowstyle='-|>',
                                     mutation_scale=10 * scale, lw=lw, color=COL[kind], shrinkA=13 * scale, shrinkB=13 * scale,
                                     linestyle='--' if kind == 'missed' else '-', zorder=2 + order[kind], alpha=0.95))
    for k, (x, y) in enumerate(pos):
        ax.add_patch(Circle((x, y), 0.11, facecolor=COL['node'], edgecolor=COL['true'], lw=1.2 * scale, zorder=10))
        ax.text(x, y, str(k + 1), ha='center', va='center', fontsize=node_fs, zorder=11)
    ax.text(pos[0, 0] - 0.05, pos[0, 1] - 0.26, 'head', fontsize=node_fs - 1.5, color='#6b6b6b', ha='center')
    ax.text(pos[-1, 0] + 0.05, pos[-1, 1] - 0.26, 'tail', fontsize=node_fs - 1.5, color='#6b6b6b', ha='center')
    ax.set_xlim(-1.3, 1.3); ax.set_ylim(-1.3, 1.25); ax.set_aspect('equal'); ax.axis('off')
    if W_hat is not None:
        nf = sum(e[2] == 'found' for e in edges); nm = sum(e[2] == 'missed' for e in edges)
        nfa = sum(e[2] == 'false' for e in edges)
        title = f'{title}\nfound {nf}   missed {nm}   false {nfa}'
    else:
        title = f'{title}\n{int(true.sum())} links'
    ax.set_title(title, fontsize=title_fs)


def network_handles():
    from matplotlib.lines import Line2D
    h = [Line2D([0], [0], color=COL['true'], lw=2, label='true link'),
         Line2D([0], [0], color=COL['found'], lw=2, label='found (true and reported)'),
         Line2D([0], [0], color=COL['missed'], lw=2, ls='--', label='missed (true, not reported)'),
         Line2D([0], [0], color=COL['false'], lw=2, label='false (reported, not true)')]
    return h


def network_legend(fig, fontsize=8):
    fig.legend(handles=network_handles(), loc='lower center', ncol=4, frameon=False, fontsize=fontsize)
