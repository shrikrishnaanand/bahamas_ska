"""
Plots of a finished wavelet run from its saved outputs: the plots of the run notebooks, without Jupyter.

Reads ``<out>/<tag>/chains.npz`` (``injection.py``), ``<out>/reference/atoms.npz`` (``fit_reference.py``, wavelet
injections) and, for the convergence table, ``<out>/<tag>/data.h5``. It never reruns the fit, the simulation or the
sampler: a missing file is an error. Writes to ``<out>/<tag>/plots/``:

    true_modulation.png       true sky envelope M_c^2(t) over the window
    reference_fit.png         the wavelet fit to it, i.e. the injection (wavelet injections only)
    K_posterior.png           p(K) per channel, true K in orange (wavelet injections only)
    modulation_recovery.png   recovered P_c(t): median and 90% band vs the injection
    corner_psd.png            spectral parameters, b_A and b_E, injected values marked
    K_trace_burnin.png        only with --burnin-plot: K of the cold walkers per step, burn-in included. Needs
                              k_trace.npz, from a second sampler run: injection.py ... --k-trace
    summary.txt               the printed numbers (fit rms, acceptance, coverage, convergence table)

Run from the repository root, e.g.
    python simulations/common/plot_run.py --config simulations/1_year/config/wavelet.yaml --tag run1
"""
import argparse
import contextlib
import io
import os
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use('Agg')                    # no display on a cluster node
import matplotlib.pyplot as plt

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import injection as I
import fit_reference as FR
import diagnostics as D
from bahamas.wavelet.model import WaveletModulation, DAY
from bahamas.wavelet.simulate import power_modulation

DIAG_TIMES_DAYS = [60., 180., 300.]      # P_c(t) monitored in the convergence table (as the 1-year notebooks)


class Log:
    """print() to the terminal and to summary.txt."""
    def __init__(self, path):
        self.f = open(path, 'w')

    def __call__(self, *args):
        s = ' '.join(str(a) for a in args)
        print(s)
        self.f.write(s + '\n')
        self.f.flush()


def need(path, how):
    if not Path(path).exists():
        raise SystemExit(f'missing {path}\n  make it with: {how}')
    return path


def save(fig, path, log, dpi=150):
    fig.savefig(path, dpi=dpi, bbox_inches='tight')
    plt.close(fig)
    log(f'  wrote {path}')


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument('--config', required=True, help='study config, e.g. simulations/1_year/config/wavelet.yaml')
    ap.add_argument('--tag', required=True, help='run name, e.g. run1 (outputs/<tag>/)')
    ap.add_argument('--burnin-plot', action='store_true',
                    help='also plot the K trace including the burn-in (needs outputs/<tag>/k_trace.npz)')
    ap.add_argument('--no-diagnostics', action='store_true',
                    help='skip the convergence table (it loads data.h5 and evaluates P_c(t) for every sample)')
    a = ap.parse_args()

    cfg = I.load_config(a.config)
    out = Path(I.out_dir(cfg))
    run = out / a.tag
    plots = run / 'plots'
    plots.mkdir(parents=True, exist_ok=True)
    log = Log(plots / 'summary.txt')
    sampler_cmd = f'python simulations/common/injection.py --config {a.config} --tag {a.tag} --injection ...'
    r = dict(np.load(need(run / 'chains.npz', sampler_cmd), allow_pickle=True))
    kind = str(r.get('injection', 'wavelet'))
    CH = sorted(k[len('mod_'):-len('_nleaves')] for k in r if k.startswith('mod_') and k.endswith('_nleaves'))
    log(f'{a.tag} ({kind} injection), config {a.config}')
    log(f'outputs {run}\n')

    # 1. true modulation
    t0, T = FR.window(cfg)
    t = np.linspace(t0, t0 + T, 3000)
    P_sky = np.stack([np.asarray(power_modulation(I.sky_params(), t, c)) for c in range(len(CH))])
    fig, ax = plt.subplots(figsize=(11, 3))
    for i, c in enumerate(CH):
        ax.plot(t / DAY, P_sky[i], label=c)
    ax.set_xlabel('time [days]'); ax.set_ylabel(r'$M_c^2(t)$'); ax.legend(); ax.set_title('true sky envelope')
    save(fig, plots / 'true_modulation.png', log)

    # 2. reference fit (the injection of wavelet runs)
    ref_path = Path(I.reference_path(cfg))
    if kind == 'wavelet':
        ref = np.load(need(ref_path, f'python simulations/common/fit_reference.py --config {a.config} --k-max ...'))
        wm = WaveletModulation(t0, T, cfg['inference']['wavelet_shape'])
        fig, axes = plt.subplots(1, len(CH), figsize=(12, 3), sharey=True)
        for i, c in enumerate(CH):
            atoms, b = ref[f'atoms_{c}'], float(ref[f'b_{c}'])
            axes[i].plot(t / DAY, P_sky[i], color='0.6', lw=3, label='true')
            axes[i].plot(t / DAY, np.exp(np.asarray(wm.log_modulation(t, atoms, level=b))), 'k--',
                         label=f'fit, K = {len(atoms)}')
            axes[i].set_title(f"{c}: rms {100 * float(ref[f'rms_{c}']):.2f}%"); axes[i].set_xlabel('time [days]')
            axes[i].legend()
        save(fig, plots / 'reference_fit.png', log)
        log(f"reference fit: true amp {float(ref['amp_true']):.3f}, b_A = {float(ref['b_A']):.3f}, "
            f"b_E = {float(ref['b_E']):.3f} (amp + b_A/ln10 = {float(ref['amp_eff']):.3f}); "
            + ', '.join(f"{c}: K = {len(ref[f'atoms_{c}'])}, rms {100 * float(ref[f'rms_{c}']):.2f}%" for c in CH))

    # 3. sampler summary and convergence
    log(f"\nsampler: {int(r['nsteps'])} steps x {int(r['nwalkers'])} walkers, runtime {float(r['runtime_s']) / 60:.0f} min, "
        f"acceptance {np.mean(r['acceptance_fraction']):.2f}, RJ acceptance {np.mean(r['rj_acceptance_fraction']):.2f}")
    if 'loglike_true' in r:
        log(f"median log L {np.median(r['log_like']):.2f}, log L(true) {float(r['loglike_true']):.2f}")
    post = None
    if not a.no_diagnostics:
        data = I.WaveletData.load(need(run / 'data.h5', sampler_cmd))
        _, post = I.build_posterior(data, cfg)
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        D.report(r, post, None if post is None else np.array(DIAG_TIMES_DAYS) * DAY)
    log('\nconvergence' + (' (P_c(t) rows skipped: --no-diagnostics)' if post is None else '') + ':')
    log(buf.getvalue().rstrip())

    # 4. number of atoms
    if kind == 'wavelet':
        fig, axes = plt.subplots(1, len(CH), figsize=(12, 2.8))
        log('')
        for ax, c in zip(axes, CH):
            K, p, K_true = r[f'mod_{c}_bayes_K'], r[f'mod_{c}_bayes_posterior'], len(r[f'atoms_true_{c}'])
            ax.bar(K, p, color=['C1' if k == K_true else 'C0' for k in K])
            ax.set_title(f'{c}: p(K), true K = {K_true} (orange)'); ax.set_xlabel('K')
            log(f'{c}: p(K = true {K_true}) = {float(np.sum(p[K == K_true])):.3f}, most probable K = {int(K[np.argmax(p)])}')
        save(fig, plots / 'K_posterior.png', log)

    # 5. modulation recovery: 10^(amp - amp_inj) P_c(t), 90% band vs the injection
    td = r['modulation_times'] / DAY
    q = np.percentile(r['modulation_draws'], [5, 50, 95], axis=0)
    fig, axes = plt.subplots(1, len(CH), figsize=(12, 3.2), sharey=True)
    log('')
    for i, c in enumerate(CH):
        axes[i].fill_between(td, q[0, i], q[2, i], alpha=0.4, label='90%')
        axes[i].plot(td, q[1, i], label='median')
        axes[i].plot(td, r['P_true'][i], 'k--', label='injected')
        axes[i].set_title(c); axes[i].set_xlabel('time [days]'); axes[i].legend()
        inside = np.mean((q[0, i] <= r['P_true'][i]) & (r['P_true'][i] <= q[2, i]))
        log(f'{c}: injection inside the 90% band at {100 * inside:.0f}% of time pixels')
    axes[0].set_ylabel(r'$10^{\,amp - amp_{inj}}\,P_c(t)$')
    save(fig, plots / 'modulation_recovery.png', log)

    # 6. spectral parameters, b_A and b_E
    import corner
    names = [str(n).split(':')[-1] for n in r['psd_names']]
    fig = corner.corner(r['psd_chain'], labels=names, truths=r['psd_injected'], show_titles=True, title_fmt='.3f')
    save(fig, plots / 'corner_psd.png', log, dpi=110)

    # 7. (optional) K trace including the burn-in
    if a.burnin_plot:
        kt = dict(np.load(need(I.k_trace_path(a.tag, cfg),
                               f'python simulations/common/injection.py --config {a.config} --tag {a.tag} '
                               f'--injection {kind} --k-trace   (a second sampler run, as long as the first)')))
        plot_k_trace(kt, CH, plots / 'K_trace_burnin.png', log)

    log(f'\ndone: plots in {plots}')


def plot_k_trace(kt, CH, path, log):
    """Min-max and 25-75% range of K over the cold walkers per step, burn-in shaded; right: the first 500 steps."""
    burn = int(kt['burnin'])
    K_true = kt['K_true'] if len(kt['K_true']) else [None] * len(CH)
    log(f"\nK trace: {int(kt['nwalkers'])} walkers x {int(kt['ntemps'])} temperatures, cold walkers shown; "
        f"{burn} burn-in + {int(kt['nsteps'])} stored steps; rerun reproduces chains.npz: {bool(kt.get('matches_chains'))}")
    fig, axes = plt.subplots(len(CH), 2, figsize=(11, 2.8 * len(CH)), sharey='row', squeeze=False,
                             gridspec_kw={'width_ratios': [2.2, 1]})
    for row, (c, k_true) in enumerate(zip(CH, K_true)):
        K = kt[f'K_{c}'][:, 0].astype(float)                       # cold chain: (steps, walkers)
        step = np.arange(1, len(K) + 1)
        lo, hi = K.min(1), K.max(1)
        q25, q75 = np.percentile(K, [25, 75], axis=1)
        zoom = min(500, len(K))
        for ax, sl, xmax in zip(axes[row], [slice(None), slice(0, zoom)], [len(K), zoom]):
            ax.fill_between(step[sl], lo[sl], hi[sl], step='mid', color='C0', alpha=0.25, lw=0,
                            label='min-max over cold walkers')
            ax.fill_between(step[sl], q25[sl], q75[sl], step='mid', color='C0', alpha=0.45, lw=0, label='25-75%')
            ax.plot(step[sl], K[sl].mean(1), color='C0', lw=1.2, label='mean')
            if k_true is not None:
                ax.axhline(k_true, color='C1', ls='--', lw=1.2, label='true K')
            ax.axvspan(0, min(burn, xmax), color='0.5', alpha=0.10, lw=0)
            ax.set_xlim(0, xmax)
        axes[row, 0].text(burn / 2, 0.95, 'burn-in', transform=axes[row, 0].get_xaxis_transform(), ha='center',
                          va='top', fontsize=8, color='0.4')
        axes[row, 0].set_ylabel(f'K, channel {c}')
        log(f'{c}: mean K, last 500 burn-in steps {K[max(burn - 500, 0):burn].mean():.2f}, stored {K[burn:].mean():.2f}')
    for ax in axes[-1]:
        ax.set_xlabel('step (burn-in included)')
    axes[0, 1].set_title(f'first {min(500, len(K))} steps', fontsize=9)
    fig.legend(*axes[0, 0].get_legend_handles_labels(), fontsize=8, ncol=4, loc='lower center',
               bbox_to_anchor=(0.5, -0.04))
    fig.tight_layout(rect=(0, 0.04, 1, 1))
    save(fig, path, log)


if __name__ == '__main__':
    main()
