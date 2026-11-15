"""
Effect of the Galactic-envelope fixes F1 (cosh -> sinh in H_S;(3)) and F2 (exp(-2 s+^2), exp(-6 s+^2) damping on
the 3b and 5b harmonics) of reports/PAPER_BUGS.md on the modulation P_c(t) = M_c(t)^2 of the A and E channels.

Each figure toggles one fix with the other one on, at the pack-2 sky parameters (simulations/30_days/config/sources.yaml),
over one year (the 30-day analysis window is shaded). Run from the repo root:

    uv run python reports/figures/envelope_fix_impact.py
"""
from pathlib import Path

import numpy as np
import yaml
import matplotlib.pyplot as plt

from bahamas import fixes
from bahamas.psd_response.modulation import envelopes_gaussian
from bahamas.wavelet.simulate import YEAR

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
DAY = 86400.0

src = yaml.safe_load((ROOT / 'simulations/30_days/config/sources.yaml').read_text())['sources']
par = {p['name']: p['injected'] for p in src['galactic_DWD_time']}

t = np.linspace(0.0, YEAR, 4000)


def power(**switch):
    with fixes.override(**switch):
        A, E = envelopes_gaussian(par['lat'], par['long'], par['s1'], par['s2'], par['psi'], 1.0 / YEAR, t)
    return np.asarray(A) ** 2, np.asarray(E) ** 2


def figure(key, label_fixed, label_bug, fname):
    fixed = power(F1=True, F2=True)
    buggy = power(**{'F1': True, 'F2': True, key: False})
    days = t / DAY

    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    lines = []
    for c, name in enumerate('AE'):
        ax, axd = axes[0, c], axes[1, c]
        ax.plot(days, fixed[c], color='C0', lw=1.6, label=label_fixed)
        ax.plot(days, buggy[c], color='C3', lw=1.2, ls='--', label=label_bug)
        ax.set_title(f'channel {name}')
        ax.set_ylabel(f'$P_{name}(t) = M_{name}^2$')
        rel = 100 * (buggy[c] - fixed[c]) / fixed[c]
        axd.plot(days, rel, color='k', lw=1.2)
        axd.axhline(0, color='0.6', lw=0.8)
        axd.set_ylabel('(bug − fixed) / fixed  [%]')
        axd.set_xlabel('t [days]')
        for a in (ax, axd):
            a.axvspan(0, 30, color='0.9', zorder=0)
        win = days <= 30
        mean_rel = 100 * (buggy[c].mean() / fixed[c].mean() - 1)
        mean_rel30 = 100 * (buggy[c][win].mean() / fixed[c][win].mean() - 1)
        ax.text(0.02, 0.96,
                f'max |Δ|/P: {np.abs(rel).max():.1f}%\n'
                f'mean P, 1 yr: {mean_rel:+.1f}%\n'
                f'mean P, 30 d: {mean_rel30:+.1f}%',
                transform=ax.transAxes, va='top', fontsize=8.5,
                bbox=dict(facecolor='white', edgecolor='0.8', alpha=0.9))
        lines.append((name, np.abs(rel).max(), mean_rel, mean_rel30, rel.min(), rel.max()))
    axes[0, 1].legend(loc='upper right', fontsize=8.5)
    fig.suptitle(f'{key}: Galactic envelope with and without the fix (pack-2 sky parameters; shaded = 30-day window)')
    fig.tight_layout()
    fig.savefig(OUT / fname, dpi=150)
    for name, mx, m1, m30, lo, hi in lines:
        print(f'{key} {name}: rel diff range [{lo:+.2f}, {hi:+.2f}]%, max |.| {mx:.2f}%, '
              f'mean-P change 1 yr {m1:+.2f}%, 30 d {m30:+.2f}%')


figure('F2', 'fixed (3b, 5b damped)', 'bug: no $e^{-2\\sigma_+^2}$, $e^{-6\\sigma_+^2}$ damping', 'envelope_F2_damping.png')
figure('F1', 'fixed (sinh)', 'bug: cosh in $H_{S;(3)}$', 'envelope_F1_cosh.png')
