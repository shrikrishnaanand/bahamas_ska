"""
What the wavelet likelihood sees, in the time-frequency plane, before and after grouping, with and without a data gap.

Uses the run-1 data (seed 2026, wavelet-atom injection, so the model at the injection is exact) for channel A.

Columns: every analysed WDM pixel (no grouping) | the likelihood grid (time_bin x log-frequency groups).
Rows:
1. data / model at the injection: w^2 / sigma^2 per pixel, Y / (nu sigma^2) per group. The Gamma likelihood is
   a function of this ratio only (Section 6.2 of WAVELET_README.md), and it scatters around 1.
2. galactic fraction of the variance, r = P S_gal / (P S_gal + S_n): where the likelihood can learn about the
   galaxy. The Fisher information on ln(galactic amplitude) is r^2 / 2 per pixel, i.e. nu r^2 / 2 per group.
3. as row 1, after zeroing GAP_DAYS of the time series (a hard gap, no taper).

Saves ``outputs/checks/likelihood_grid.png`` and prints where the gap biases the pixels.
Run as ``python likelihood_grid.py``.
"""
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm

from bahamas.wavelet.model import WaveletData, DAY

import injection as I
import dof_check as D

CHANNEL = 0           # A
GAP_DAYS = (12.0, 13.0)
OUT = os.path.join(I.OUT, 'checks')


def pixel_edges(centres, step):
    return np.append(centres - step / 2, centres[-1] + step / 2)


def group_edges(like, data):
    dF = data.freqs[1]
    starts = np.flatnonzero(np.diff(np.append(-1, np.asarray(like.group_ids))))
    f = data.freqs[like.pixels]
    return np.append(f[starts] - dF / 2, f[-1] + dF / 2)


def block_edges(like, data):
    dT = data.times[1] - data.times[0]
    t = data.times[like.time_index][::like.time_bin]
    return np.append(t, t[-1] + like.time_bin * dT) - dT / 2


def grids(x, inj):
    data = I.to_wavelet(x, inj)
    like, post = I.build_posterior(data)
    sig2_group, sig2_pixel = D.expected(like, post, inj, data)
    w = data.w[CHANNEL][like.time_index][:, like.pixels]
    pix = w ** 2 / sig2_pixel[CHANNEL]
    grp = np.asarray(like.Y)[CHANNEL] / (np.asarray(like.nu) * sig2_group[CHANNEL])
    return data, like, post, pix, grp


def galactic_fraction(like, post, inj, data):
    import jax.numpy as jnp
    from bahamas.psd_strain import psd_galaxy as gal
    from bahamas.wavelet.sampler import _sampled
    theta = jnp.asarray([p.value for p in _sampled(post.psd_params)], dtype=float)
    _, sig2_pixel = D.expected(like, post, inj, data)
    S_gal, S_n, values = post.spectra(theta)
    f = like.eval_freqs
    Sg = np.asarray(jnp.reshape(gal.galactic_foreground(f, values[post.galaxy_source], gen2=post.gen2),
                                (like.npix, -1)) @ like._weights)
    Ppix = np.asarray(inj.P(data.times[like.time_index], CHANNEL))
    r_pix = Ppix[:, None] * Sg[None, :] / (2 * like.dt) / sig2_pixel[CHANNEL]
    P = np.asarray(inj.P(like.times, CHANNEL))
    r_grp = P[:, None] * np.asarray(S_gal)[None, :] / (P[:, None] * np.asarray(S_gal)[None, :] + np.asarray(S_n)[CHANNEL])
    return r_pix, r_grp


def main():
    x, dt, t0 = I.load_timeseries('run1')
    _, inj = I.simulate_series('wavelet', seed=0)          # only the Injection object (P(t) of run 1) is used
    data, like, post, pix, grp = grids(x, inj)
    r_pix, r_grp = galactic_fraction(like, post, inj, data)

    xg = x.copy()
    t = t0 + np.arange(x.shape[1]) * dt
    xg[:, (t >= GAP_DAYS[0] * DAY) & (t < GAP_DAYS[1] * DAY)] = 0.0
    _, _, _, pix_gap, grp_gap = grids(xg, inj)

    dT = data.times[1] - data.times[0]
    te_pix = pixel_edges(data.times[like.time_index], dT) / DAY
    fe_pix = pixel_edges(data.freqs[like.pixels], data.freqs[1])
    te_grp, fe_grp = block_edges(like, data) / DAY, group_edges(like, data)

    ratio_norm = TwoSlopeNorm(vcenter=0.0, vmin=-1.0, vmax=1.0)
    fig, axes = plt.subplots(3, 2, figsize=(13, 12), sharex=True, sharey=True, constrained_layout=True)
    rows = [(np.log10(pix), np.log10(grp), 'RdBu_r', ratio_norm, r'$\log_{10}$ (data / model)'),
            (r_pix, r_grp, 'viridis', None, r'galactic fraction $r = P S_{\rm gal} / (P S_{\rm gal} + S_n)$'),
            (np.log10(pix_gap), np.log10(grp_gap), 'RdBu_r', ratio_norm, r'$\log_{10}$ (data / model)')]
    titles = [('every pixel: $w^2_{nm}/\\sigma^2_{nm}$', 'likelihood grid: $Y_{sg}/(\\nu_g\\sigma^2_{sg})$'),
              ('every pixel', 'likelihood grid'),
              (f'every pixel, gap {GAP_DAYS[0]:g}-{GAP_DAYS[1]:g} d', 'likelihood grid, same gap')]
    for i, (a, b, cmap, norm, label) in enumerate(rows):
        for j, (z, te, fe) in enumerate([(a, te_pix, fe_pix), (b, te_grp, fe_grp)]):
            ax = axes[i, j]
            kw = dict(norm=norm) if norm is not None else dict(vmin=0, vmax=1)
            m = ax.pcolormesh(te, fe, z.T, cmap=cmap, shading='flat', rasterized=True, **kw)
            ax.set_yscale('log')
            ax.set_title(titles[i][j], fontsize=11)
            if i == 2:
                for edge in GAP_DAYS:
                    ax.axvline(edge, color='k', ls='--', lw=1.0)
        fig.colorbar(m, ax=axes[i], label=label, shrink=0.9)
    for ax in axes[:, 0]:
        ax.set_ylabel('frequency [Hz]')
    for ax in axes[-1]:
        ax.set_xlabel('time [days]')
    fig.suptitle(f'Channel A, run-1 data: {pix.shape[0]} x {pix.shape[1]} pixels (left) -> '
                 f'{grp.shape[0]} x {grp.shape[1]} groups (right)', fontsize=12)
    os.makedirs(OUT, exist_ok=True)
    path = os.path.join(OUT, 'likelihood_grid.png')
    fig.savefig(path, dpi=130)
    print('saved', path)

    # Where does the gap bias the pixels? Compare with the no-gap ratio, averaged over frequency bands
    tc = data.times[like.time_index] / DAY
    print(f"\nGap {GAP_DAYS} d: mean of w^2/sigma^2 per time pixel (no gap -> with gap), by band")
    fp = data.freqs[like.pixels]
    for lo, hi in [(1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 3e-2)]:
        s = (fp >= lo) & (fp < hi)
        near = (tc > GAP_DAYS[0] - 3) & (tc < GAP_DAYS[1] + 3)
        print(f"  {lo * 1e3:g}-{hi * 1e3:g} mHz:")
        for k in np.flatnonzero(near):
            print(f"    t = {tc[k]:5.2f} d   {pix[k, s].mean():.2f} -> {pix_gap[k, s].mean():.2f}")


if __name__ == '__main__':
    main()
