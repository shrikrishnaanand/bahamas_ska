"""Shared plotting style of the needlet patch study (matplotlib, static figures)."""
import numpy as np
import healpy as hp
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt  # noqa: E402

# Categorical slots in fixed order: galaxy, then the patches in config order
COLORS = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#6250d6']
INK, MUTED = '#0b0b0b', '#52514e'
SEQ_CMAP = 'Blues'          # magnitude: one hue, light -> dark
DIV_CMAP = 'RdBu_r'         # polarity: two hues, neutral midpoint

plt.rcParams.update({
    'font.size': 9, 'axes.titlesize': 9, 'axes.labelsize': 9, 'legend.fontsize': 8,
    'axes.edgecolor': MUTED, 'axes.labelcolor': INK, 'xtick.color': MUTED, 'ytick.color': MUTED,
    'axes.grid': True, 'grid.color': '#e4e3df', 'grid.linewidth': 0.6, 'axes.spines.top': False,
    'axes.spines.right': False, 'lines.linewidth': 1.6, 'figure.dpi': 130, 'savefig.bbox': 'tight',
})


def mollview(m, sub, title, diverging=False, vmax=None, vmin=None, unit='', fig=None, cmap=None):
    """Mollweide map in galactic coordinates (maps are stored ecliptic)."""
    if diverging:
        vmax = np.max(np.abs(m)) if vmax is None else vmax
        kw = dict(cmap=cmap or DIV_CMAP, min=-vmax, max=vmax)
    else:
        kw = dict(cmap=cmap or SEQ_CMAP, min=0 if vmin is None else vmin, max=vmax)
    hp.mollview(m, coord=['E', 'G'], sub=sub, title=title, unit=unit, fig=fig, cbar=True, notext=True,
                bgcolor='white', **kw)
    hp.graticule(dpar=30, dmer=60, color='#9a998f', alpha=0.4)


def mark(l_deg, b_deg, label=None, color=INK):
    """Mark a galactic position on the current mollview."""
    hp.projscatter(l_deg, b_deg, lonlat=True, coord='G', s=40, facecolors='none', edgecolors=color, linewidths=1.2)
    if label:
        hp.projtext(l_deg, b_deg + 8, label, lonlat=True, coord='G', fontsize=7, color=color, ha='center')
