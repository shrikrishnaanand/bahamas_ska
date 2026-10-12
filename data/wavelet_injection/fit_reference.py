"""
Reference wavelet fit of the true GF envelope over the 30-day window (the run-1 injection).

Fits log P_c(t) = b_c + sum_k a_k psi((t - t_k) / tau_k) to the true envelope P_c(t) = M_c^2(t) of the sky
parameters in ``config/sources.yaml``, for K = 0, 1, ... atoms, and keeps the smallest K with an rms error below
``RMS_TARGET``. Same procedure as the archived stage-2 notebook (greedy start from the K-1 fit plus random
restarts, L-BFGS-B), but:

- the window is the span the WaveletPosterior uses, [t0, t0 + N dt] (``N`` from ``wdm_shape``), so the atoms need no
  remapping;
- the atom bounds are the run-1 inference priors (``injection.CONFIG['inference']``), with the centres kept
  ``EDGE_MARGIN`` away from the window ends, so every injected atom lies inside the prior and not on its boundary
  (a recovery test of a value on a prior bound would look biased);
- the levels b_c are unbounded: only level_E = b_E - b_A enters the inference, and it is checked against its prior;
- a weak penalty ``AMP_PENALTY * sum a_k^2`` breaks the level / wide-atom degeneracy (a wider atom with a larger
  amplitude and a higher level draws almost the same curve), which otherwise runs the E atom onto its amplitude bound.

The result is written in the inference parametrisation, where b_A = 0 is absorbed into the galactic amplitude::

    amp_eff = amp + b_A / ln 10,    level_E = b_E - b_A

Saves ``outputs/reference/atoms.npz`` (read by ``injection.load_reference``) and ``outputs/reference/fit.png``.
Run as ``python fit_reference.py``.
"""
import os

import numpy as np
import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from scipy.optimize import minimize
import matplotlib

matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bahamas.wavelet.model import WaveletModulation, DAY
from bahamas.wavelet.simulate import power_modulation
from bahamas.wavelet.pipeline import wdm_shape

import injection as I

RMS_TARGET = 0.01   # 1% rms in P, well below the ~10% width of the run-1 posterior band
K_MAX = 4
EDGE_MARGIN = 0.02  # atom centres in [0.02, 0.98] of the window (0.6 d from each end)
LEVEL_FIT = (-10., 10.)
AMP_PENALTY = 1e-6  # costs < 0.05% rms; picks the smallest-amplitude solution along the level / width degeneracy
RESTARTS = 20
CH = 'AE'


def window(config=I.CONFIG):
    _, N = wdm_shape(config['T'], config['dt'], config['wavelet']['Nf'])
    return config['t0'], N * config['dt']


def bounds(K, config=I.CONFIG):
    icfg = config['inference']
    return ([LEVEL_FIT] + [(EDGE_MARGIN, 1. - EDGE_MARGIN), tuple(np.log10(icfg['width_range_days'])),
                           tuple(icfg['amp_range'])] * K)


def fit(t, y, wm, K, prev=None, seed=0):
    """Best K-atom fit of y = log P(t). theta = (level, t_frac_1, log10_width_1, amp_1, ...)."""
    def loss(theta):
        return (jnp.mean((wm.log_modulation(t, theta[1:].reshape(K, 3), level=theta[0]) - y) ** 2)
                + AMP_PENALTY * jnp.sum(theta[3::3] ** 2))
    vg = jax.jit(jax.value_and_grad(loss))
    fun = lambda th: tuple(np.asarray(v, float) for v in vg(jnp.asarray(th)))
    bnds = bounds(K)
    (wlo, whi), (alo, ahi) = bnds[2] if K else (0, 1), bnds[3] if K else (-1, 1)

    rng = np.random.default_rng(seed)
    starts = []
    if prev is not None:   # previous fit plus one atom at the largest residual
        res = np.asarray(y - wm.log_modulation(t, prev[1:].reshape(K - 1, 3), level=prev[0]))
        i = np.argmax(np.abs(res))
        tf = np.clip((float(t[i]) - wm.t0) / wm.T, EDGE_MARGIN, 1 - EDGE_MARGIN)
        starts.append(np.concatenate([prev, [tf, np.log10(20.), np.clip(res[i], alo, ahi)]]))
    for _ in range(RESTARTS):
        starts.append(np.array([float(jnp.mean(y))] + [v for _ in range(K) for v in
                                                         (rng.uniform(EDGE_MARGIN, 1 - EDGE_MARGIN),
                                                          rng.uniform(wlo, whi), rng.uniform(-1, 1))]))
    best = min((minimize(fun, s, jac=True, method='L-BFGS-B', bounds=bnds, options={'maxiter': 3000})
                for s in starts), key=lambda r: r.fun)
    P_fit = np.exp(np.asarray(wm.log_modulation(t, best.x[1:].reshape(K, 3), level=best.x[0])))
    rel = P_fit / np.exp(np.asarray(y)) - 1
    return dict(theta=best.x, rms=float(np.sqrt(np.mean(rel ** 2))), max_rel=float(np.max(np.abs(rel))), P_fit=P_fit)


def run(out=os.path.join(I.OUT, 'reference')):
    os.makedirs(out, exist_ok=True)
    t0, T = window()
    t = jnp.arange(t0, t0 + T, 3600.)   # hourly
    wm = WaveletModulation(t0, T, I.CONFIG['inference']['wavelet_shape'])
    sky = I.sky_params()
    print(f"sky parameters: {sky}\nwindow: [{t0 / DAY:.2f}, {(t0 + T) / DAY:.2f}] d")

    best = {}
    for c, ch in enumerate(CH):
        P_true = np.asarray(power_modulation(sky, t, c))
        prev, scan = None, []
        for K in range(K_MAX + 1):
            r = fit(t, jnp.log(P_true), wm, K, prev)
            scan.append(r)
            prev = r['theta']
        K_sel = next((K for K, r in enumerate(scan) if r['rms'] < RMS_TARGET), K_MAX)
        print(f"{ch}: rms by K = " + '  '.join(f"{K}:{r['rms']:.4f}" for K, r in enumerate(scan))
              + f"  -> K = {K_sel} (max |dP/P| {scan[K_sel]['max_rel']:.4f})")
        best[ch] = dict(scan[K_sel], K=K_sel, P_true=P_true, scan=scan)

    bA, bE = best['A']['theta'][0], best['E']['theta'][0]
    amp = I.true_amp()
    ref = {'atoms_A': best['A']['theta'][1:].reshape(-1, 3), 'atoms_E': best['E']['theta'][1:].reshape(-1, 3),
           'amp_eff': amp + bA / np.log(10), 'level_E': bE - bA, 'b_A': bA, 'b_E': bE, 'amp_true': amp,
           't0': t0, 'T': T, 'rms_A': best['A']['rms'], 'rms_E': best['E']['rms'],
           'rms_scan': np.array([[r['rms'] for r in best[ch]['scan']] for ch in CH])}
    np.savez(os.path.join(out, 'atoms.npz'), **ref)

    print(f"\namp_eff = {amp:.4f} + {bA:+.4f}/ln10 = {ref['amp_eff']:.4f};  level_E = {ref['level_E']:+.4f}")
    check_prior(ref)
    for ch in CH:
        print(f"{ch} atoms:   centre[d]  width[d]     amp")
        for tf, lw, a in ref[f'atoms_{ch}']:
            print(f"          {(t0 + tf * T) / DAY:9.2f} {10 ** lw:9.2f} {a:+7.3f}")
    plot(t, wm, best, ref, os.path.join(out, 'fit.png'))
    return ref


def check_prior(ref, config=I.CONFIG, tol=1e-3):
    """Warn about injected values on (or outside) a boundary of the inference prior."""
    icfg = config['inference']
    vals = [('level_E', ref['level_E'], icfg['level_range'])]
    for ch in CH:
        for k, (tf, lw, a) in enumerate(ref[f'atoms_{ch}']):
            vals += [(f'{ch}{k + 1} t_frac', tf, (0., 1.)), (f'{ch}{k + 1} log10 width', lw,
                     tuple(np.log10(icfg['width_range_days']))), (f'{ch}{k + 1} amp', a, icfg['amp_range'])]
    bad = [f'{n} = {v:.4f} (prior {lo:.3g}..{hi:.3g})' for n, v, (lo, hi) in vals
           if not lo + tol * (hi - lo) < v < hi - tol * (hi - lo)]
    print('all injected values inside the prior' if not bad else 'ON/OUTSIDE PRIOR BOUND: ' + '; '.join(bad))


def plot(t, wm, best, ref, path):
    td = np.asarray(t) / DAY
    fig, axes = plt.subplots(3, 2, figsize=(11, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1, 1.2]})
    for c, ch in enumerate(CH):
        r = best[ch]
        atoms = ref[f'atoms_{ch}']
        ax0, ax1, ax2 = axes[:, c]
        ax0.plot(td, r['P_true'], color='#0b0b0b', lw=2.5, alpha=0.35, label=r'true $M_c^2(t)$, sin$\psi$ = −0.83')
        ax0.plot(td, r['P_fit'], color='#2a78d6', ls='--', label=f"wavelet fit, K = {r['K']}")
        ax0.set_title(f'channel {ch}', loc='left')
        ax1.plot(td, 100 * (r['P_fit'] / r['P_true'] - 1), color='#2a78d6')
        ax1.axhline(0, color='#52514e', lw=0.6)
        ax2.axhline(r['theta'][0], color='#52514e', lw=1, ls=':', label=f"level b = {r['theta'][0]:+.3f}")
        for k in range(len(atoms)):
            ax2.plot(td, np.asarray(wm.log_modulation(t, atoms[k:k + 1], level=0.)), lw=1.2,
                     label=f'atom {k + 1}: centre {(wm.t0 + atoms[k, 0] * wm.T) / DAY:.1f} d')
        ax2.set_xlabel('time [days]')
        for ax in axes[:, c]:
            ax.grid(alpha=0.25)
            ax.spines[['top', 'right']].set_visible(False)
        ax0.legend(fontsize=8, frameon=False)
        ax2.legend(fontsize=8, frameon=False)
    axes[0, 0].set_ylabel(r'$P_c(t)$')
    axes[1, 0].set_ylabel(r'$\Delta P/P$ [%]')
    axes[2, 0].set_ylabel(r'atoms in $\log P$')
    fig.suptitle('Run-1 reference: wavelet fit of the true envelope (30-day window)')
    fig.tight_layout()
    fig.savefig(path, dpi=130, bbox_inches='tight')
    plt.close(fig)


if __name__ == '__main__':
    run()
