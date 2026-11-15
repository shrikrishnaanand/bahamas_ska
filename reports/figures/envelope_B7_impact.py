"""
Effect of fix B7 of reports/PAPER_BUGS.md (16 lbar instead of 4 lbar in the n = 4 A^2 - E^2 term of the point-source
chunk average, average_envelope_num.envelopes_time_average) on 30-day chunk averages of the A and E modulation.

Two cases, each against the exact time average of the instantaneous envelope (envelopes_gaussian at zero width,
divided by cos b, the convention of average_envelope_num):
  * a point source at the Galactic centre;
  * the galactic_DWD_num Galaxy: disk weights with R_d = 2.5 kpc, z_d = 0.3 kpc on the precomputed sky grid.

Chunks are 30 days long; their start slides through one year. Run from the repo root:

    uv run --python 3.12 python reports/figures/envelope_B7_impact.py
"""
from pathlib import Path

import numpy as np
import matplotlib.pyplot as plt

from bahamas.wavelet import wdm  # noqa: F401  (selects the JAX backend first)
import jax
import jax.numpy as jnp
from bahamas import fixes
from bahamas.psd_response.modulation import envelopes_gaussian
from bahamas.psd_response.average_envelope_num import envelopes_time_average
from bahamas.psd_strain.psd_galaxy import PRECOMPUTED, compute_weight_auto_components

OUT = Path(__file__).resolve().parent
YEAR, DAY = 31557600.0, 86400.0
CHUNK = 30 * DAY
starts = np.arange(0, 366, 3) * DAY
mids = (starts + CHUNK / 2) / DAY

# Fine time grid covering every chunk; chunk averages of the instantaneous envelope by trapezoid
t_fine = np.arange(0, starts[-1] + CHUNK + DAY, DAY / 8)


def chunk_mean(p):
    out = []
    for t1 in starts:
        m = (t_fine >= t1 - 1e-6) & (t_fine <= t1 + CHUNK + 1e-6)
        out.append(np.trapezoid(p[m], t_fine[m]) / CHUNK)
    return np.array(out)


def analytic(b, l, w=None):
    """Chunk averages (A, E) from envelopes_time_average, fixed and pre-fix, optionally weighted over pixels."""
    res = {}
    for on in (True, False):
        with fixes.override(B7=on):
            f = jax.jit(lambda t1, c: envelopes_time_average(b, l, 1 / YEAR, t1, t1 + CHUNK, c))
            vals = []
            for c in (0, 1):
                v = np.array([np.asarray(f(t1, c)) for t1 in starts])
                vals.append(v @ w if w is not None else v)
            res[on] = vals
    return res[True], res[False]


def numeric(b, l, w=None):
    """Exact chunk averages of the instantaneous envelope."""
    b, l = np.atleast_1d(b), np.atleast_1d(l)
    w = np.ones(1) if w is None else w
    keep = w > 0
    b, l, w = b[keep], l[keep], w[keep]
    PA, PE = np.zeros_like(t_fine), np.zeros_like(t_fine)
    env = jax.jit(lambda sb, ll, t: envelopes_gaussian(sb, ll, 0., 0., 0., 1 / YEAR, t))
    for i in range(0, len(t_fine), 500):
        t = jnp.asarray(t_fine[i:i + 500])[None, :]
        A, E = env(jnp.sin(b)[:, None], jnp.asarray(l)[:, None], t)
        scale = (w / np.cos(b))[:, None]
        PA[i:i + 500] = np.sum(np.asarray(A) ** 2 * scale, axis=0)
        PE[i:i + 500] = np.sum(np.asarray(E) ** 2 * scale, axis=0)
    return chunk_mean(PA), chunk_mean(PE)


cases = []
b_gc, l_gc = np.deg2rad(-5.6), np.deg2rad(266.8)
cases.append(('point source at the Galactic centre', analytic(b_gc, l_gc), numeric(b_gc, l_gc)))
w = np.asarray(compute_weight_auto_components({'R_d': 2.5, 'z_d': 0.3}))
bg, lg = np.asarray(PRECOMPUTED['Eclat']), np.asarray(PRECOMPUTED['Eclon'])
cases.append(('Galaxy (galactic_DWD_num, disk R_d = 2.5, z_d = 0.3 kpc)', analytic(bg, lg, w), numeric(bg, lg, w)))

fig, axes = plt.subplots(4, 2, figsize=(11, 10), sharex=True, gridspec_kw={'height_ratios': [2, 1, 2, 1]})
for k, (title, (fixed, buggy), num) in enumerate(cases):
    for c, name in enumerate('AE'):
        ax, axd = axes[2 * k, c], axes[2 * k + 1, c]
        ax.plot(mids, num[c], color='0.6', lw=4, alpha=0.6, label='exact time average')
        ax.plot(mids, fixed[c], color='C0', lw=1.5, label='fixed (4 $\\bar l$)')
        ax.plot(mids, buggy[c], color='C3', lw=1.2, ls='--', label='bug (16 $\\bar l$)')
        ax.set_ylabel(f'30-day mean $P_{name}$')
        ax.set_title(f'{title}: channel {name}', fontsize=9.5)
        rel_b = 100 * (buggy[c] / num[c] - 1)
        rel_f = 100 * (fixed[c] / num[c] - 1)
        axd.plot(mids, rel_b, color='C3', lw=1.2, label='bug')
        axd.plot(mids, rel_f, color='C0', lw=1.2, label='fixed')
        axd.axhline(0, color='0.6', lw=0.8)
        axd.set_ylabel('vs exact [%]')
        ax.text(0.02, 0.95, f'bug: {rel_b.min():+.1f}% to {rel_b.max():+.1f}%\nfixed: max |.| {np.abs(rel_f).max():.0e}%',
                transform=ax.transAxes, va='top', fontsize=8.5,
                bbox=dict(facecolor='white', edgecolor='0.8', alpha=0.9))
        print(f'{title} {name}: bug {rel_b.min():+.2f}% .. {rel_b.max():+.2f}%, fixed max {np.abs(rel_f).max():.1e}%')
axes[0, 1].legend(fontsize=8, loc='upper right')
axes[1, 1].legend(fontsize=8, loc='upper right')
for a in axes[-1]:
    a.set_xlabel('chunk midpoint [days]')
fig.suptitle('B7: 30-day chunk average of the point-source envelope, fixed vs bug vs exact')
fig.tight_layout()
fig.savefig(OUT / 'envelope_B7_phase.png', dpi=150)
