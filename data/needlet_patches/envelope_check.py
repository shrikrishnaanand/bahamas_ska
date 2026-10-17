"""
Check of the analytic Galactic envelope ``bahamas.psd_response.modulation.envelopes_gaussian`` against
first principles and against the printed formula of Buscicchio et al. 2024 (arXiv:2410.08263, App. A,
Eqs. 105-125).

The formula is the low-frequency A/E power of a sky that is a bivariate Gaussian in ecliptic (beta, lambda),
normalised in the (beta, lambda) plane, integrated with the sky measure cos(beta) d beta d lambda. Its point-source
limit (Sigma1 = Sigma2 = 0) is therefore ``R_c(n) cos(beta)``, not ``R_c(n)``.

Up to commit 4db1cc9 the bahamas transcription differed from the paper in two places (``average_envelope.py`` had
the same two; both are fixed since):

1. One ``cosh`` that should be ``sinh``: the -864 term of H_S;(3) (paper Eq. 117), ``cosh3HalvesSigmaSqSin * s3DphiL
   * cbM``. At zero width it leaves a parity-odd term.
2. The factors exp(-2 sigma_+^2) and exp(-6 sigma_+^2) on the 3 beta and 5 beta harmonics are kept in the pure-beta
   group (H_S;(0), H_D;(4)) but dropped in every Delta-phi group (24 + 24 terms in Sum and Diff). The paper has them
   (e.g. "-1296 e^{-2 sigma_+^2} (...sin 3b_M cos 3 Delta phi_L...)" in Eq. 117).

``code_version()`` is the function before the fix (from git), ``paper_version()`` the current, fixed one. Checks (orbit: bahamas
``SpacecraftOrbit``; first-principles response: ``sky.point_response``):

* point source: formula / cos(beta) vs first principles, and parity, for the code and the corrected version;
* finite width (pack-2 values): formula vs the exact Gaussian average of ``R_c cos(beta)`` (Gauss-Hermite);
* 30-day pack-2 envelope: code vs corrected, and the sky parameters with which the code's model best mimics the
  corrected envelope (the bias one would get fitting correctly modelled data with the code), compared with the
  run-2 OG posterior widths.

    python envelope_check.py   -> outputs/envelope_check/{envelope_check.json, envelope_check.png}
"""
import os
import json
import inspect
import subprocess

import numpy as np
import healpy as hp
from scipy.optimize import least_squares

import sky  # noqa: F401  (sets the JAX backend first)
import bahamas.psd_response.modulation as modulation
from bahamas.wavelet.simulate import YEAR
from plotting import plt, COLORS, MUTED

OUT = os.path.join(sky.HERE, 'outputs', 'envelope_check')
OG_RUN2 = os.path.join(sky.HERE, '..', 'wavelet_injection', 'outputs', 'run2', 'og', 'result.hdf5')
TYPO = 'cosh3HalvesSigmaSqSin * s3DphiL * cbM'
PRE_FIX_COMMIT = '4db1cc9'      # last commit with the two transcription errors
PACK2 = {'lat': -0.09664565, 'long': -1.6259238, 's1': 0.04156049812924856, 's2': 0.13842963137295858,
         'psi': -0.83}


def _compile(source, name):
    namespace = {'jnp': np}
    exec(source.replace('def envelopes_gaussian', f'def {name}'), namespace)
    return namespace[name]


def code_version(source=False):
    """The function as it was before the fix (commit PRE_FIX_COMMIT), compiled with NumPy."""
    src = subprocess.run(['git', 'show', f'{PRE_FIX_COMMIT}:bahamas/psd_response/modulation.py'], cwd=sky.HERE,
                         capture_output=True, text=True, check=True).stdout
    src = src[src.index('def envelopes_gaussian'):]
    return src if source else _compile(src, 'code')


def paper_version(source=False):
    """The current (fixed) function, compiled with NumPy (in-place updates need full-shape inputs)."""
    src = inspect.getsource(modulation.envelopes_gaussian)
    assert TYPO not in src, 'modulation.py still has the cosh/sinh typo'
    return src if source else _compile(src, 'paper')


def power(f, p, t):
    return np.stack([a ** 2 for a in f(p['lat'], p['long'], p['s1'], p['s2'], p['psi'], 1 / YEAR, t)])


def best_scale_residual(ref, model):
    """Max |ref - k model| / max(ref) per channel, with one common best-fit scale k."""
    k = np.sum(ref * model) / np.sum(model * model)
    return (np.max(np.abs(ref - k * model), axis=-1) / np.max(ref, axis=-1)).max(axis=-1), float(k)


def point_source(fs, nside=16):
    n = sky.pixel_vectors(nside)
    theta, lam = hp.pix2ang(nside, np.arange(n.shape[1]))
    beta = np.pi / 2 - theta
    t = np.linspace(0, YEAR, 25)
    R = sky.point_response(t, n)
    anti = hp.vec2pix(nside, *(-n))
    B, L = np.broadcast_arrays(np.sin(beta)[None], lam[None] + 0 * t[:, None])
    out = {}
    for name, f in fs.items():
        A, E = f(B, L, 0., 0., 0., 1 / YEAR, t[:, None] + 0 * B)
        F = np.stack([A ** 2, E ** 2])
        res, _ = best_scale_residual(R, F / np.cos(beta))
        res_raw, _ = best_scale_residual(R[0] + R[1], (F[0] + F[1]))
        out[name] = {'max_rel_residual_vs_first_principles_A_E (formula / cos beta)': res.tolist(),
                     'max_rel_residual_A2+E2_without_cos_beta': float(res_raw),
                     'parity_violation': float(np.max(np.abs(F / np.cos(beta) - (F / np.cos(beta))[:, :, anti]))
                                               / np.max(F / np.cos(beta)))}
    return out


def gaussian_average(p, t, order=40):
    """Exact average of R_c cos(beta) over the bivariate Gaussian in (beta, lambda) of the formula."""
    S, D = p['s1'] + p['s2'], p['s1'] - p['s2']
    cospsi = np.sqrt(1 - p['psi'] ** 2)
    C = np.array([[(S + D * cospsi) / 2, D * p['psi'] / 2], [D * p['psi'] / 2, (S - D * cospsi) / 2]])
    x, w = np.polynomial.hermite_e.hermegauss(order)
    X, Y = np.meshgrid(x, x)
    W = np.outer(w, w).ravel()
    W /= W.sum()
    z = np.stack([X.ravel(), Y.ravel()], 1) @ np.linalg.cholesky(C).T
    b, l = np.arcsin(p['lat']) + z[:, 0], p['long'] + z[:, 1]
    R = sky.point_response(t, np.array([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)]))
    return (R * np.cos(b) * W).sum(-1)


def fit_bias(code, P_target, t):
    """Sky parameters (and a log10 amplitude) with which the code's model best matches ``P_target``."""
    names = list(PACK2)
    x0 = np.array([PACK2[n] for n in names] + [0.0])
    lo, hi = [-1, -2, 1e-4, 0.1, -1, -1], [0.1, -1, 0.1, 0.3, -0.5, 1]       # run-2 prior bounds (and amp)

    def residual(x):
        return ((10 ** x[5] * power(code, dict(zip(names, x[:5])), t) - P_target) / P_target).ravel()
    r = least_squares(residual, x0, bounds=(lo, hi))
    return dict(zip(names + ['log10_amp'], r.x.tolist())), float(np.abs(r.fun).max())


def og_widths():
    try:
        import h5py
        with h5py.File(OG_RUN2, 'r') as f:
            ps = f['posterior_samples'][()]
        return {n: float(ps[n].std()) for n in PACK2}
    except (OSError, KeyError):
        return None


def main():
    os.makedirs(OUT, exist_ok=True)
    fs = {'before_fix': code_version(), 'fixed': paper_version()}
    result = {'point_source': point_source(fs)}

    t_year = np.linspace(0, YEAR, 120)
    exact = gaussian_average(PACK2, t_year)
    result['finite_width_pack2_one_year'] = {
        name: {'max_rel_residual_vs_exact_gaussian_A_E': best_scale_residual(exact, power(f, PACK2, t_year))[0]
               .tolist()} for name, f in fs.items()}

    t = np.linspace(0, 30 * 86400, 300)
    Pc, Pp = power(fs['before_fix'], PACK2, t), power(fs['fixed'], PACK2, t)
    k = np.sum(Pc * Pp) / np.sum(Pc * Pc)
    fitted, fit_res = fit_bias(fs['before_fix'], Pp, t)
    widths = og_widths()
    shift = {n: fitted[n] - PACK2[n] for n in PACK2}
    result['pack2_30_days'] = {
        'mean_ratio_paper_over_code_A_E': (Pp.mean(1) / Pc.mean(1)).tolist(),
        'shape_max_rel_difference_A_E': (np.abs(Pp - k * Pc) / Pp).max(1).tolist(),
        'code_model_fit_to_corrected_envelope': fitted,
        'code_model_fit_max_rel_residual': fit_res,
        'parameter_shift': shift,
        'run2_og_posterior_sd': widths,
        'shift_over_og_sd': None if widths is None else {n: shift[n] / widths[n] for n in PACK2},
    }

    fig, axes = plt.subplots(1, 2, figsize=(10, 3.4), sharex=True)
    for c, ch in enumerate('AE'):
        ax = axes[c]
        ax.plot(t / 86400, Pc[c], color=COLORS[0], label='before the fix (runs 1-2)')
        ax.plot(t / 86400, Pp[c], color=COLORS[1], label='fixed (= Buscicchio+24)')
        ax.plot(t / 86400, 10 ** fitted['log10_amp'] * power(fs['before_fix'], fitted, t)[c], '--', color=MUTED,
                label='code with shifted sky parameters')
        ax.set(title=f'pack-2 envelope, channel {ch}', xlabel='time [d]', ylabel='P(t)')
    axes[0].legend(frameon=False, fontsize=7)
    fig.savefig(os.path.join(OUT, 'envelope_check.png'))
    plt.close(fig)

    with open(os.path.join(OUT, 'envelope_check.json'), 'w') as f:
        json.dump(result, f, indent=2)
    print(json.dumps(result, indent=2))


if __name__ == '__main__':
    main()
