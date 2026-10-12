"""
Step-2 data checks: are the inputs of both pipelines unbiased for a realistic (non-periodic) 30-day series?

For ``NSEEDS`` fresh noise realisations of the run-2 injection (true sky envelope), compare the data with their
exact expectation at the injected parameters:

- OG: binned chunk periodogram / model PSD with the chunk-averaged modulation, versus frequency, without a taper
  and with the taper of og_config.yaml. Spectral leakage of the rectangular 15-day chunk shows up as a ratio above 1
  where the spectrum is steep.
- WDM: pixel power w^2 / expected variance (P_c(t) S_gal + S_n) / (2 dt), versus time pixel. The transform is
  periodic, so the first and last pixels mix the two ends of the series; this sets how many pixels the likelihood
  has to drop (``edge_bins``).

Both ratios have expectation 1. Saves ``outputs/checks/data_checks.png`` and ``outputs/checks/data_checks.npz``.
Run as ``python data_checks.py``.
"""
import os
import copy

import numpy as np
import yaml
import jax.numpy as jnp
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bahamas.wavelet.model import WaveletLikelihood, parametric_modulation, DAY
from bahamas.wavelet.sampler import WaveletPosterior, _sampled

import injection as I
import og

NSEEDS = 10
SEED0 = 100           # different from the run seed (2026)
FREQ_EDGES = np.logspace(-4, np.log10(0.029), 13)   # frequency bands for both checks
LOW_F = 1e-3          # WDM edge check also done for f < 1 mHz only
OUT = os.path.join(I.OUT, 'checks')


def og_ratio(x, dt, t0, cfg):
    """Count-weighted mean periodogram/model ratio per (chunk, channel, frequency band), and the counts."""
    sp = og.chunk_spectra(x, dt, t0, cfg)
    num = np.zeros((len(sp.f_av), sp.ntdi, len(FREQ_EDGES) - 1))
    cnt = np.zeros_like(num)
    for i, r in enumerate(og.binned_vs_model(sp)):
        b = np.digitize(r['f'], FREQ_EDGES) - 1
        ok = (b >= 0) & (b < len(FREQ_EDGES) - 1)
        for c in range(sp.ntdi):
            np.add.at(num[i, c], b[ok], (r['count'] * r['ratio'][c])[ok])
            np.add.at(cnt[i, c], b[ok], r['count'][ok])
    return num, cnt


def wdm_ratio(x, inj, sources):
    """sum(w^2 / var) and sum(dof) per (channel, time pixel), for all frequencies and for f < LOW_F."""
    data = I.to_wavelet(x, inj)
    like = WaveletLikelihood(data, I.CONFIG['f1'], I.CONFIG['f2'], freq_bins=I.CONFIG['wavelet']['freq_bins'],
                             edge_bins=0, nsub=I.CONFIG['wavelet']['band_samples'], time_bin=1)
    post = WaveletPosterior(like, sources, modulation='parametric', gen2=I.CONFIG['gen2'])
    theta = jnp.asarray([p.value for p in _sampled(post.psd_params)])
    S_gal, S_noise, _ = post.spectra(theta)
    P = parametric_modulation(like.times, data.channels, I.sky_params())
    var = (P[:, :, None] * S_gal[None, None, :] + S_noise[:, None, :]) / (2 * like.dt)
    z = np.asarray(like.Y / var)                       # (channels, times, groups), expectation nu per group
    nu = np.asarray(like.nu)
    low = like.group_freqs < LOW_F
    return z.sum(-1), nu.sum(), z[:, :, low].sum(-1), nu[low].sum()


def run():
    os.makedirs(OUT, exist_ok=True)
    with open(og.OG_CONFIG) as f:
        cfg = yaml.safe_load(f)
    sources = I.og_sources()
    cfg_raw = copy.deepcopy(cfg)
    cfg_raw['chunk']['taper'] = 0.0
    og_num, og_cnt, raw_num, w_all, w_low = [], [], [], [], []
    for k in range(NSEEDS):
        x, inj = I.simulate_series('sky', seed=SEED0 + k)
        n, c = og_ratio(x, I.CONFIG['dt'], I.CONFIG['t0'], cfg)
        og_num.append(n)
        og_cnt.append(c)
        raw_num.append(og_ratio(x, I.CONFIG['dt'], I.CONFIG['t0'], cfg_raw)[0])
        za, nu_a, zl, nu_l = wdm_ratio(x, inj, sources)
        w_all.append(za / nu_a)
        w_low.append(zl / nu_l)
        print(f'seed {SEED0 + k} done')
    og_num, og_cnt = np.array(og_num), np.array(og_cnt)
    og_mean = og_num.sum(0) / og_cnt.sum(0)                 # (chunk, channel, band)
    og_err = 1 / np.sqrt(og_cnt.sum(0))                       # Gamma: var(mean) = 1 / total count
    og_raw = np.array(raw_num).sum(0) / og_cnt.sum(0)
    w_all, w_low = np.array(w_all), np.array(w_low)          # (seed, channel, time)
    res = {'freq_edges': FREQ_EDGES, 'og_mean': og_mean, 'og_err': og_err, 'og_raw': og_raw,
           'taper': cfg['chunk']['taper'],
           'wdm_all': w_all.mean(0), 'wdm_all_err': w_all.std(0, ddof=1) / np.sqrt(NSEEDS),
           'wdm_low': w_low.mean(0), 'wdm_low_err': w_low.std(0, ddof=1) / np.sqrt(NSEEDS),
           'wdm_all_sigma_theory': np.sqrt(2 / nu_a / NSEEDS), 'wdm_low_sigma_theory': np.sqrt(2 / nu_l / NSEEDS)}
    np.savez(os.path.join(OUT, 'data_checks.npz'), **res)
    report(res)
    plot(res)
    return res


def report(res):
    fc = np.sqrt(FREQ_EDGES[1:] * FREQ_EDGES[:-1])
    print(f'\nOG: binned periodogram / model, mean over seeds (expect 1 +- err); no taper | taper {res["taper"]}')
    print('  f [mHz]  ' + '  '.join(f'ch{i + 1}{c}' for i in range(2) for c in 'AE') + '  | '
          + '  '.join(f'ch{i + 1}{c}' for i in range(2) for c in 'AE') + '    err')
    for b, f in enumerate(fc):
        print(f'  {1e3 * f:7.3f}  ' + '  '.join(f'{v:5.3f}' for v in res['og_raw'][:, :, b].ravel()) + '  | '
              + '  '.join(f'{v:5.3f}' for v in res['og_mean'][:, :, b].ravel()) + f'  {res["og_err"][0, 0, b]:.3f}')
    for key, label in [('all', 'all frequencies'), ('low', f'f < {1e3 * LOW_F:.0f} mHz')]:
        r, e = res[f'wdm_{key}'], res[f'wdm_{key}_err']
        print(f'\nWDM w^2/var per time pixel ({label}); expect 1 +- {res[f"wdm_{key}_sigma_theory"]:.4f}')
        for c, ch in enumerate('AE'):
            pulls = (r[c] - 1) / e[c]
            print(f'  {ch}: first 6 ' + ' '.join(f'{v:.3f}' for v in r[c, :6]) + '  | last 6 '
                  + ' '.join(f'{v:.3f}' for v in r[c, -6:]) + f'  | interior mean {r[c, 6:-6].mean():.4f}'
                  + f'  | pixels with |pull| > 3: {list(np.flatnonzero(np.abs(pulls) > 3))}')


def plot(res):
    fc = np.sqrt(FREQ_EDGES[1:] * FREQ_EDGES[:-1])
    fig, axes = plt.subplots(1, 3, figsize=(15, 4))
    ax = axes[0]
    ax.plot(fc, res['og_raw'].mean((0, 1)), color='#52514e', ls='--', marker='x', ms=5,
            label='no taper (mean of chunks, channels)')
    for i in range(2):
        for c, (ch, col) in enumerate(zip('AE', ['#2a78d6', '#eb6834'])):
            ax.errorbar(fc * (1 + 0.04 * (2 * i + c - 1.5)), res['og_mean'][i, c], res['og_err'][i, c], fmt='o' if i == 0 else 's',
                        ms=4, color=col, mfc=col if i == 0 else 'white', label=f'chunk {i + 1}, {ch}')
    ax.axhline(1, color='#52514e', lw=0.8)
    ax.set_xscale('log')
    ax.set_xlabel('frequency [Hz]')
    ax.set_ylabel('binned periodogram / model')
    ax.set_title(f'OG chunks, taper {res["taper"]}', loc='left', fontsize=11)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    dT = I.CONFIG['wavelet']['Nf'] * I.CONFIG['dt'] / DAY
    for ax, key, label in [(axes[1], 'all', 'all frequencies'), (axes[2], 'low', f'f < {1e3 * LOW_F:.0f} mHz')]:
        n = np.arange(res[f'wdm_{key}'].shape[1])
        for c, (ch, col) in enumerate(zip('AE', ['#2a78d6', '#eb6834'])):
            ax.errorbar(n * dT, res[f'wdm_{key}'][c], res[f'wdm_{key}_err'][c], fmt='o', ms=3, color=col, label=ch)
        s = res[f'wdm_{key}_sigma_theory']
        ax.axhspan(1 - 2 * s, 1 + 2 * s, color='#52514e', alpha=0.15, lw=0, label='expected ±2σ')
        ax.axhline(1, color='#52514e', lw=0.8)
        ax.set_xlabel('time pixel centre [days]')
        ax.set_ylabel(r'$w^2$ / expected variance')
        ax.set_title(f'WDM pixels, {label}', loc='left', fontsize=11)
        ax.legend(fontsize=8, frameon=False)
    for ax in axes:
        ax.grid(alpha=0.25)
        ax.spines[['top', 'right']].set_visible(False)
    fig.suptitle(f'Data checks: mean over {NSEEDS} noise realisations of the run-2 injection (expectation 1)')
    fig.tight_layout()
    fig.savefig(os.path.join(OUT, 'data_checks.png'), dpi=130, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    run()
