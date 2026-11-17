"""
Effect of the code bugs B1, B2, B3, B3b and B6 of reports/PAPER_BUGS.md, one figure each: the affected quantity with
the fix on (blue, solid) and off (red, dashed), and their relative difference. F1, F2 and B7 are in
envelope_fix_impact.py and envelope_B7_impact.py; the gap bugs B4 and B5 have no figure.

Every curve comes from the bahamas code itself with one fix switched off through ``bahamas.fixes.override`` (the
other fixes on); B3 is an option (``paper_eqs``), not a switch:

  B1  ERYN Gamma log-likelihood of one segment versus the model level S        setting_eryn.gamma_lik
  B2  Gamma likelihood of S with dof = 2 n_b (fixed) or n_b (paper)            setting_eryn.gamma_lik
  B3  A and T instrument-noise PSD, bahamas form vs Pozzoli et al. Eqs. 16-17  psd_noise.noise (paper_eqs option)
  B3b A and T instrument-noise PSD with and without the Nam et al. A/E fix     psd_noise.noise
  B6  Galactic modulation P_c(t) = M_c^2 at alpha0 = 0.4, beta0 = -1.1         modulation.envelopes_gaussian

Settings follow the runs: pack-2 sources (simulations/30_days/config/sources.yaml), gen2 = False. Run from the repo
root:

    .venv/bin/python reports/figures/paper_bugs_impact.py
"""
from pathlib import Path

import numpy as np
import yaml
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

from bahamas.wavelet import wdm  # noqa: F401  (selects the JAX backend first)
import jax.numpy as jnp
from bahamas import fixes
from bahamas.method import setting_eryn
from bahamas.psd_strain import psd_noise
from bahamas.psd_response.modulation import envelopes_gaussian

ROOT = Path(__file__).resolve().parents[2]
OUT = Path(__file__).resolve().parent
YEAR, DAY = 31557600.0, 86400.0
FIXED = dict(color='C0', lw=1.6)
BUG = dict(color='C3', lw=1.3, ls='--')
BOX = dict(facecolor='white', edgecolor='0.8', alpha=0.9)

src = yaml.safe_load((ROOT / 'simulations/30_days/config/sources.yaml').read_text())['sources']
GAL = {p['name']: p['injected'] for p in src['galactic_DWD_time']}
NOISE = {p['name']: p['injected'] for p in src['instr_noise']}


def info(ax, text, loc=(0.02, 0.96)):
    ax.text(*loc, text, transform=ax.transAxes, va='top', fontsize=8.5, bbox=BOX)


def rel_axis(ax, num='bug', den='fixed', unit='rel. diff [%]'):
    """Short y label; the definition of the plotted difference as the panel title."""
    ax.set_ylabel(unit)
    ax.set_title(f'({num} − {den}) / {den}' if den else num, fontsize=9, loc='left')


def finish(fig, title, fname):
    fig.suptitle(title, fontsize=11)
    fig.tight_layout()
    fig.align_ylabels()
    fig.savefig(OUT / fname, dpi=150)
    plt.close(fig)
    print('wrote', fname)


# ---------------------------------------------------------------- B1, B2: likelihoods
NB, NAMP, S_TRUE = 30, 30, 1e-40          # 30 coarse bins of n_b = 30 complex amplitudes, true level S


def eryn_gamma(P_hat, S, dof):
    """setting_eryn.gamma_lik for one segment with a constant model S (the PSD model is replaced by S)."""
    model_psd, to_sources = setting_eryn.psd.model_psd, setting_eryn._eryn_coords_to_sources
    setting_eryn.psd.model_psd = lambda **kw: S * jnp.ones(NB)
    setting_eryn._eryn_coords_to_sources = lambda *a, **k: {}
    try:
        return float(setting_eryn.gamma_lik({}, [P_hat[None]], [np.ones(NB)], [[np.ones(NB)]], {}, 1., [0], [0],
                                            [dof], False))
    finally:
        setting_eryn.psd.model_psd, setting_eryn._eryn_coords_to_sources = model_psd, to_sources


rng = np.random.default_rng(1)
P_HAT = S_TRUE * rng.gamma(NAMP, 1 / NAMP, NB)           # averaged periodogram: Gamma with shape n_b, mean S


def figure_B1():
    x = np.linspace(-3, 3, 241)                           # ln(S / S_true)
    dof = 2 * NAMP * np.ones(NB)
    ll = {}
    for on in (True, False):
        with fixes.override(B1=on):
            ll[on] = np.array([eryn_gamma(P_HAT, S_TRUE * np.exp(v), dof) for v in x])
    fixed, bug = ll[True] - ll[True].max(), ll[False] - ll[False].max()

    fig, (ax, axd) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    ax.plot(x, fixed, label=r'fixed: $(\frac{n}{2}-1)\ln\hat P - \frac{n}{2}\ln S - \frac{n}{2}\hat P/S$', **FIXED)
    ax.plot(x, bug, label=r'bug: $(\frac{n}{2}-1)\ln\hat P^2 - \frac{n}{2}\ln S - \hat P^2/2S$', **BUG)
    ax.axvline(0, color='0.6', lw=0.8)
    ax.set_ylabel(r'$\ln\mathcal{L} - \max\ln\mathcal{L}$')
    ax.legend(loc='lower center', fontsize=8.5)
    info(ax, f'maximum at S / S_true:\n  fixed {np.exp(x[np.argmax(fixed)]):.2f}\n'
             f'  bug   {np.exp(x[np.argmax(bug)]):.3f} (grid edge)')
    axd.plot(x, bug - fixed, color='k', lw=1.2)
    axd.axhline(0, color='0.6', lw=0.8)
    rel_axis(axd, num='ln L(bug) − ln L(fixed), each from its maximum; a ratio of ln L is not meaningful '
                      '(only differences of ln L matter)', den=None, unit=r'$\Delta\ln\mathcal{L}$')
    axd.set_xlabel(r'$\ln(S / S_{\rm true})$')
    finish(fig, f'B1: ERYN Gamma log-likelihood of one segment ({NB} bins, $n_b$ = {NAMP})', 'likelihood_B1_eryn_gamma.png')


def figure_B2():
    x = np.linspace(-0.25, 0.25, 201)
    ll = {k: np.array([eryn_gamma(P_HAT, S_TRUE * np.exp(v), d * NAMP * np.ones(NB)) for v in x])
          for k, d in (('fixed', 2), ('bug', 1))}
    d = {k: v - v.max() for k, v in ll.items()}
    width = {k: 1 / np.sqrt(-2 * np.polyfit(x, v, 2)[0]) for k, v in d.items()}

    fig, (ax, axd) = plt.subplots(2, 1, figsize=(8, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    ax.plot(x, np.exp(d['fixed']), label=r'fixed: shape $n_b$ (dof = 2 $n_b$)', **FIXED)
    ax.plot(x, np.exp(d['bug']), label=r'paper / bug: shape $n_b/2$ (dof = $n_b$)', **BUG)
    ax.set_ylabel(r'$\mathcal{L}(S) / \mathcal{L}_{\max}$')
    ax.legend(loc='upper right', fontsize=8.5)
    info(ax, f'1σ width of ln S:\n  fixed {width["fixed"]:.4f}\n  bug   {width["bug"]:.4f}\n'
             f'  ratio {width["bug"] / width["fixed"]:.3f}  (√2 = 1.414)')
    axd.plot(x, 100 * (width['bug'] / width['fixed'] - 1) * np.ones_like(x), color='k', lw=1.2,
             label='posterior width')
    mask = d['fixed'] < -1e-3
    axd.plot(x[mask], 100 * (d['bug'][mask] / d['fixed'][mask] - 1), color='0.45', lw=1.2, ls=':',
             label=r'$\Delta\ln\mathcal{L}$ from the maximum')
    axd.axhline(0, color='0.6', lw=0.8)
    axd.set_ylim(-60, 60)
    rel_axis(axd)
    axd.set_xlabel(r'$\ln(S / S_{\rm true})$')
    axd.legend(fontsize=8, loc='lower right')
    finish(fig, f'B2: Gamma likelihood of the PSD level ({NB} bins, $n_b$ = {NAMP}, data drawn with shape $n_b$)',
           'likelihood_B2_dof.png')


# ---------------------------------------------------------------- B3, B3b: instrument noise
f_noise = np.logspace(-4, np.log10(0.029), 600)


def noise(tdi, **par):
    return np.asarray(psd_noise.noise(jnp.asarray(f_noise), {**NOISE, **par}, tdi=tdi, gen2=False))


def noise_figure(key, curves, labels, title, fname):
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    for c, (tdi, name) in enumerate(((0, 'A (= E)'), (2, 'T'))):
        right, wrong = curves(tdi)
        ax, axd = axes[0, c], axes[1, c]
        ax.loglog(f_noise, right, label=labels[0], **FIXED)
        ax.loglog(f_noise, wrong, label=labels[1], **BUG)
        ax.set_title(f'channel {name}', fontsize=10)
        ax.set_ylabel('noise PSD [1/Hz]')
        rel = 100 * (wrong / right - 1)
        axd.semilogx(f_noise, rel, color='k', lw=1.2)
        axd.axhline(0, color='0.6', lw=0.8)
        rel_axis(axd, labels[1].split(':')[0], labels[0].split(':')[0])
        axd.set_xlabel('f [Hz]')
        info(ax, f'rel. diff {rel.min():+.2f}% to {rel.max():+.2f}%', loc=(0.40, 0.96))
        print(f'{key} {name}: rel diff {rel.min():+.3f}% .. {rel.max():+.3f}%')
    axes[0, 0].legend(fontsize=8.5, loc='lower left')
    finish(fig, title, fname)


def figure_B3():
    noise_figure('B3', lambda tdi: (noise(tdi, paper_eqs=1), noise(tdi)),
                 ('paper: Pozzoli et al. Eqs. 16–17', 'bahamas: default form'),
                 f'B3 (open, option paper_eqs): instrument noise, A = {NOISE["A"]}, P = {NOISE["P"]}, TDI 1',
                 'noise_B3_paper_eqs.png')


def figure_B3b():
    def curves(tdi):
        right = noise(tdi)
        with fixes.override(B3b=False):
            wrong = noise(tdi)
        return right, wrong
    noise_figure('B3b', curves, ('fixed: Nam et al. Table II', 'bug: 2 cos w'),
                 f'B3b: instrument noise with and without the A/E test-mass transfer-function fix, TDI 1',
                 'noise_B3b_transfer.png')


# ---------------------------------------------------------------- B6: envelope rotation
def figure_B6(alpha0=0.4, beta0=-1.1):
    t = np.linspace(0.0, YEAR, 2000)
    env = {}
    for on in (True, False):
        with fixes.override(B6=on):
            A, E = envelopes_gaussian(GAL['lat'], GAL['long'], GAL['s1'], GAL['s2'], GAL['psi'], 1 / YEAR,
                                      jnp.asarray(t), alpha0, beta0)
        env[on] = (np.asarray(A) ** 2, np.asarray(E) ** 2)
    days = t / DAY
    fig, axes = plt.subplots(2, 2, figsize=(11, 6), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    for c, name in enumerate('AE'):
        ax, axd = axes[0, c], axes[1, c]
        ax.plot(days, env[True][c], label='fixed: phase shift of $A^2-E^2$', **FIXED)
        ax.plot(days, env[False][c], label='bug: rotation of the envelopes', **BUG)
        ax.set_title(f'channel {name}', fontsize=10)
        ax.set_ylabel(f'$P_{name}(t) = M_{name}^2$')
        rel = 100 * (env[False][c] / env[True][c] - 1)
        axd.plot(days, rel, color='k', lw=1.2)
        axd.axhline(0, color='0.6', lw=0.8)
        rel_axis(axd)
        axd.set_xlabel('t [days]')
        info(ax, f'rel. diff {rel.min():+.0f}% to {rel.max():+.0f}%')
        print(f'B6 {name}: rel diff {rel.min():+.1f}% .. {rel.max():+.1f}%')
    axes[0, 1].legend(fontsize=8.5, loc='upper right')
    finish(fig, rf'B6: Galactic modulation at $\alpha_0$ = {alpha0}, $\beta_0$ = {beta0} (pack-2 sky; '
                r'identical at $\alpha_0 = \beta_0 = 0$)', 'envelope_B6_rotation.png')


if __name__ == '__main__':
    figure_B1()
    figure_B2()
    figure_B3()
    figure_B3b()
    figure_B6()
