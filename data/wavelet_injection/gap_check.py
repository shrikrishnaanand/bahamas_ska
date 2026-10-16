"""
A data gap in the wavelet-domain likelihood: which WDM pixels it spoils, and which ones the likelihood must ignore.

Takes the run-1 setup (30 d, wavelet-atom injection), zeroes GAP_DAYS of the time series (a hard gap, no taper),
transforms to the WDM domain and compares every pixel with its expectation at the injection, w^2 / sigma^2
(expectation 1 without a gap). Over NSEEDS noise realisations, the mean of w^2 / sigma^2 per time pixel and frequency
band shows how far from the gap the pixels are biased. A time pixel is

* a **gap pixel** if its 10-h interval overlaps the gap, and
* a **buffer pixel** if it is next to the gap (or to another buffer pixel) and its mean w^2 / sigma^2 differs from 1
  by more than TOL *and* by more than 3 standard errors in some band (the buffer grows outward from the gap and stops
  at the first pixel on each side that is unbiased in every band);

both are removed from the likelihood. No sampling is done.

Saves ``outputs/checks/gap_check.png`` and ``outputs/checks/gap_check.npz``. Run as ``python gap_check.py``.
"""
import os

import numpy as np
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.colors import TwoSlopeNorm
from matplotlib.patches import Patch

from bahamas.wavelet.model import DAY

import injection as I
import dof_check as D

GAP_DAYS = (12.0, 13.0)
NSEEDS = 20
SEED0 = 1000
TOL = 0.02          # 2% bias in w^2 / sigma^2 is the most a pixel kept in the likelihood may carry
BANDS = [(1e-4, 1e-3), (1e-3, 1e-2), (1e-2, 2.9e-2)]
CHANNEL = 0         # channel shown in the maps (A)
OUT = os.path.join(I.OUT, 'checks')


def apply_gap(x, dt, t0, gap=GAP_DAYS):
    t = t0 + np.arange(x.shape[1]) * dt
    xg = x.copy()
    xg[:, (t >= gap[0] * DAY) & (t < gap[1] * DAY)] = 0.0
    return xg


def ratios(x, inj, like, sig2_pixel):
    data = I.to_wavelet(x, inj)
    w = data.w[:, like.time_index][:, :, like.pixels]
    return w ** 2 / sig2_pixel                          # (channels, time pixels, frequency pixels)


def run():
    cfg = I.CONFIG
    dt, t0 = cfg['dt'], cfg['t0']
    full, gapped = [], []
    for k in range(NSEEDS):
        x, inj = I.simulate_series('wavelet', seed=SEED0 + k)
        if k == 0:
            data0 = I.to_wavelet(x, inj)
            like, post = I.build_posterior(data0)
            _, sig2_pixel = D.expected(like, post, inj, data0)
            x0 = x
        full.append(ratios(x, inj, like, sig2_pixel))
        gapped.append(ratios(apply_gap(x, dt, t0), inj, like, sig2_pixel))
        print(f'realisation {k + 1}/{NSEEDS}', flush=True)
    full, gapped = np.array(full), np.array(gapped)     # (seeds, channels, n, m)

    dT = data0.Nf * dt
    tc = data0.times[like.time_index]                    # pixel centres
    fp = data0.freqs[like.pixels]
    in_gap = (tc + dT / 2 > GAP_DAYS[0] * DAY) & (tc - dT / 2 < GAP_DAYS[1] * DAY)

    # mean over realisations, channels and the pixels of each band; standard error from the scatter over seeds
    bias, err = [], []
    for lo, hi in BANDS:
        s = (fp >= lo) & (fp < hi)
        per_seed = gapped[..., s].mean(axis=(1, 3))      # (seeds, n)
        bias.append(per_seed.mean(0) - 1)
        err.append(per_seed.std(0, ddof=1) / np.sqrt(NSEEDS))
    bias, err = np.array(bias), np.array(err)
    bad = ((np.abs(bias) > TOL) & (np.abs(bias) > 3 * err)).any(axis=0)
    buffer = np.zeros_like(in_gap)
    idx = np.flatnonzero(in_gap)
    for start, step in ((idx[0] - 1, -1), (idx[-1] + 1, +1)):     # grow outward from each end of the gap
        k = start
        while 0 <= k < len(tc) and bad[k]:
            buffer[k] = True
            k += step
    keep = ~(in_gap | buffer)
    control = np.array([full[..., (fp >= lo) & (fp < hi)].mean(axis=(0, 1, 3)) - 1 for lo, hi in BANDS])

    res = dict(tc=tc, fp=fp, dT=dT, in_gap=in_gap, buffer=buffer, keep=keep, bias=bias, err=err, control=control,
               map_full=full[0, CHANNEL], map_gap=gapped[0, CHANNEL], gap=np.array(GAP_DAYS), tol=TOL,
               nseeds=NSEEDS, time_bin=like.time_bin)
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, 'gap_check.npz'), **res)
    report(res)
    plot(res, os.path.join(OUT, 'gap_check.png'))
    return res


def report(res):
    tc, gap = res['tc'] / DAY, res['gap']
    print(f"\ngap {gap[0]:g}-{gap[1]:g} d, {res['nseeds']} realisations x 2 channels; "
          f"mean w^2/sigma^2 - 1 per time pixel (+- standard error), by band")
    near = np.flatnonzero((tc > gap[0] - 2.5) & (tc < gap[1] + 2.5))
    head = '  '.join(f'{lo * 1e3:g}-{hi * 1e3:g} mHz'.center(17) for lo, hi in BANDS)
    print(f"  t centre [d]   status  {head}")
    for k in near:
        status = 'gap' if res['in_gap'][k] else ('buffer' if res['buffer'][k] else 'kept')
        cells = '  '.join(f"{100 * b:+7.1f} +- {100 * e:4.1f}%" for b, e in zip(res['bias'][:, k], res['err'][:, k]))
        print(f"  {tc[k]:10.2f}   {status:6s}  {cells}")
    far = res['keep'] & (np.abs(tc - np.mean(gap)) > 3)
    print(f"\npixels far from the gap (control): max |bias| {100 * np.abs(res['bias'][:, far]).max():.1f}%; "
          f"without the gap at the gap pixels: max |bias| {100 * np.abs(res['control'][:, res['in_gap']]).max():.1f}%")
    ng, nb = res['in_gap'].sum(), res['buffer'].sum()
    print(f"removed: {ng} gap + {nb} buffer time pixels = {(ng + nb) * res['dT'] / 3600:.0f} h of data for a "
          f"{24 * (gap[1] - gap[0]):.0f}-h gap")


def plot(res, path):
    tc, fp, dT = res['tc'] / DAY, res['fp'], res['dT'] / DAY
    te = np.append(tc - dT / 2, tc[-1] + dT / 2)
    dF = fp[1] - fp[0]
    fe = np.append(fp - dF / 2, fp[-1] + dF / 2)
    gap = res['gap']
    norm = TwoSlopeNorm(vcenter=0.0, vmin=-1.5, vmax=1.5)

    fig = plt.figure(figsize=(13, 9.5), constrained_layout=True)
    gs = fig.add_gridspec(2, 2, height_ratios=[1.25, 1])
    ax_a, ax_b = fig.add_subplot(gs[0, 0]), fig.add_subplot(gs[0, 1])
    ax_c, ax_d = fig.add_subplot(gs[1, 0]), fig.add_subplot(gs[1, 1])

    for ax, z, title in [(ax_a, res['map_full'], 'no gap'), (ax_b, res['map_gap'], f'gap {gap[0]:g}-{gap[1]:g} d')]:
        m = ax.pcolormesh(te, fe, np.log10(z).T, cmap='RdBu_r', norm=norm, shading='flat', rasterized=True)
        ax.set_yscale('log')
        ax.set_xlabel('time [days]')
        ax.set_title(f'channel A, one realisation, {title}: $\\log_{{10}}(w^2_{{nm}}/\\sigma^2_{{nm}})$', fontsize=10)
    ax_a.set_ylabel('frequency [Hz]')
    fig.colorbar(m, ax=[ax_a, ax_b], label=r'$\log_{10}$ (data / expectation)', shrink=0.85)
    for k in np.flatnonzero(~res['keep']):
        hatch, col = ('xx', 'k') if res['in_gap'][k] else ('///', '#52514e')
        ax_b.fill_between([te[k], te[k + 1]], fe[0], fe[-1], facecolor='none', edgecolor=col, hatch=hatch, lw=0)
    for edge in gap:
        ax_b.axvline(edge, color='k', ls='--', lw=1)
    ax_b.legend(handles=[Patch(facecolor='none', edgecolor='k', hatch='xx', label='gap pixels (ignored)'),
                         Patch(facecolor='none', edgecolor='#52514e', hatch='///', label='buffer pixels (ignored)')],
                loc='lower left', fontsize=8, framealpha=0.85)

    cols = ['#2a78d6', '#eb6834', '#1f9e6e']
    for (lo, hi), b, e, c in zip(BANDS, res['bias'], res['err'], cols):
        ax_c.errorbar(tc, 100 * b, yerr=100 * e, color=c, marker='o', ms=3, lw=1, capsize=2,
                      label=f'{lo * 1e3:g}–{hi * 1e3:g} mHz')
    ax_c.axhspan(-100 * res['tol'], 100 * res['tol'], color='#52514e', alpha=0.12, lw=0, label=f'±{100 * res["tol"]:g}%')
    for k in np.flatnonzero(~res['keep']):
        ax_c.axvspan(te[k], te[k + 1], color='k' if res['in_gap'][k] else '#52514e',
                     alpha=0.18 if res['in_gap'][k] else 0.08, lw=0)
    ax_c.set_xlim(gap[0] - 3, gap[1] + 3)
    ax_c.set_ylim(-100, 60)
    ax_c.set_xlabel('time pixel centre [days]')
    ax_c.set_ylabel(r'mean $w^2/\sigma^2 - 1$ [%]')
    ax_c.set_title(f'bias per time pixel, {res["nseeds"]} realisations × A, E (dark: gap, light: buffer)', fontsize=10)
    ax_c.legend(fontsize=8, ncol=2)

    zoom = (te[:-1] >= gap[0] - 3) & (te[1:] <= gap[1] + 3)
    kz = np.flatnonzero(zoom)
    ax_d.pcolormesh(te[kz[0]:kz[-1] + 2], fe, np.log10(res['map_gap'][kz]).T, cmap='RdBu_r', norm=norm,
                    shading='flat', rasterized=True)
    for k in np.flatnonzero(~res['keep']):
        ax_d.fill_between([te[k], te[k + 1]], fe[0], fe[-1], color='k' if res['in_gap'][k] else '#52514e',
                          alpha=0.55 if res['in_gap'][k] else 0.35, lw=0)
    for edge in gap:
        ax_d.axvline(edge, color='#eb6834', ls='--', lw=1.2)
    ax_d.set_yscale('log')
    ax_d.set_xlabel('time [days]')
    ax_d.set_title(f'zoom, channel A with gap: ignored pixels shaded '
                   f'({res["in_gap"].sum()} gap + {res["buffer"].sum()} buffer of {len(tc)} time pixels)', fontsize=10)
    fig.savefig(path, dpi=130)
    print('saved', path)


if __name__ == '__main__':
    run()


def plot_likelihood_mask(path=os.path.join(OUT, 'gap_likelihood_mask.png'), tmin=9.0, tmax=16.0):
    """
    One figure: the WDM pixels of channel A with the gap, the pixels the likelihood ignores (gap + buffer, from
    ``gap_check.npz``), and the likelihood groups (time blocks of ``time_bin`` pixels x log-frequency groups) with the
    degrees of freedom each block keeps.
    """
    from bahamas.wavelet.model import frequency_groups
    res = dict(np.load(os.path.join(OUT, 'gap_check.npz')))
    cfg = I.CONFIG
    x, inj = I.simulate_series('wavelet', seed=SEED0)
    data = I.to_wavelet(x, inj)
    like, post = I.build_posterior(data)
    _, sig2_pixel = D.expected(like, post, inj, data)
    z = ratios(apply_gap(x, cfg['dt'], cfg['t0']), inj, like, sig2_pixel)[CHANNEL]

    dT = res['dT'] / DAY
    tc = res['tc'] / DAY
    te = np.append(tc - dT / 2, tc[-1] + dT / 2)
    fp = res['fp']
    dF = fp[1] - fp[0]
    fe = np.append(fp - dF / 2, fp[-1] + dF / 2)
    _, starts = frequency_groups(data.freqs, cfg['f1'], cfg['f2'], cfg['wavelet']['freq_bins'])
    tb = like.time_bin
    sel = (tc >= tmin) & (tc <= tmax)
    k0, k1 = np.flatnonzero(sel)[[0, -1]]
    k0 -= k0 % tb                      # start on a block boundary (blocks are pairs of analysed pixels)
    k1 += tb - 1 - (k1 % tb)

    fig, ax = plt.subplots(figsize=(12, 7.8), constrained_layout=True)
    m = ax.pcolormesh(te[k0:k1 + 2], fe, np.log10(z[k0:k1 + 1]).T, cmap='RdBu_r',
                      norm=TwoSlopeNorm(vcenter=0.0, vmin=-1.5, vmax=1.5), shading='flat', rasterized=True)
    fig.colorbar(m, ax=ax, label=r'$\log_{10}\,(w^2_{nm}/\sigma^2_{nm})$, channel A', shrink=0.8)
    for k in range(k0, k1 + 1):
        if not res['keep'][k]:
            ingap = res['in_gap'][k]
            ax.fill_between([te[k], te[k + 1]], fe[0], fe[-1], color='#3d3d3a' if ingap else '#8a8985', lw=0,
                            alpha=0.92, zorder=3)
    for f in fe[starts]:                                   # frequency-group boundaries
        ax.axhline(f, color='k', lw=0.6, alpha=0.7, zorder=4)
    for k in range(k0, k1 + 2, tb):                        # time-block boundaries
        ax.axvline(te[k], color='k', lw=1.4, zorder=5)
    for edge in res['gap']:
        ax.axvline(edge, color='#e3342f', ls='--', lw=2, zorder=6)
    excl = np.flatnonzero(~res['keep'])
    # degrees of freedom kept per time block, as a multiple of n_g (frequency pixels in group g)
    for k in range(k0, k1 + 1, tb):
        kept = int(res['keep'][k:k + tb].sum())
        label = {tb: r'$\nu = 2n_g$', 0: r'$\nu = 0$'}.get(kept, rf'$\nu = {kept}n_g$')
        ax.text((te[k] + te[k + tb]) / 2, 1.012, label, transform=ax.get_xaxis_transform(), ha='center', va='bottom',
                fontsize=9, color='#e3342f' if kept < tb else 'k')
    ax.set_yscale('log')
    ax.set_xlim(te[k0], te[k1 + 1])
    ax.set_xlabel('time [days]')
    ax.set_ylabel('frequency [Hz]')
    t_lo, t_hi = te[excl[0]], te[excl[-1] + 1]
    ax.set_title(f'Gap {res["gap"][0]:g}–{res["gap"][1]:g} d (red dashed): dark = gap pixels, grey = buffer pixels, '
                 f'both ignored → {t_lo:.2f}–{t_hi:.2f} d ({24 * (t_hi - t_lo):.0f} h) unused\n'
                 r'black lines: time blocks of 2 pixels (20 h); thin lines: frequency groups; '
                 r'top: $\nu$ kept per block, $n_g$ = frequency pixels in group $g$', fontsize=10, pad=24)
    fig.savefig(path, dpi=130)
    print('saved', path)
    for k in range(k0, k1 + 1, tb):
        print(f"block {te[k]:.2f}-{te[k + tb]:.2f} d: pixels {k}..{k + tb - 1} (analysed index), "
              f"kept {int(res['keep'][k:k + tb].sum())}/{tb}")
