"""
Stage 0: the sky -> modulation forward model, and checks of it.

1. Response checks: the first-principles A/E power response of ``sky.point_response`` against the closed form
   R_A + R_E ~ 1 + 6 c^2 + c^4 (c = n . constellation normal), parity, harmonic content, isotropic normalisation.
2. Comparison with the analytic envelope ``envelopes_gaussian`` of bahamas at zero width (a point source), divided
   by cos(beta) (its sky-measure convention; details in envelope_check.py).
3. The sky model (Galaxy + patches) and what an A/E auto-power analysis can see of it (l = 0, 2, 4).
4. The 30-day modulation P_c(t) with and without the patches.

    python stage0_forward.py        -> outputs/stage0/{checks.json, response_checks.png, sky.png, modulation.png}
"""
import os
import json

import numpy as np
import healpy as hp
import jax

jax.config.update('jax_enable_x64', True)
import bahamas.wavelet  # noqa: F401,E402  (initialises the JAX backend of the response modules)
from bahamas.psd_response.modulation import envelopes_gaussian  # noqa: E402
from bahamas.psd_response import orbits as orb  # noqa: E402

import sky  # noqa: E402
import plotting  # noqa: E402
from plotting import plt, COLORS, MUTED  # noqa: E402

OUT = os.path.join(sky.HERE, 'outputs', 'stage0')
YEAR = 2 * np.pi / orb.twoPiPerYear
T_OBS = 30 * sky.DAY


def constellation_normal(t, orbit):
    p = np.asarray(orbit.SC_positions(t))
    z = np.cross(p[1] - p[0], p[2] - p[0])
    return z / np.linalg.norm(z)


def response_checks(nside=16):
    """Checks of the first-principles response over one year; returns a dict of numbers."""
    orbit = orb.SpacecraftOrbit()
    n = sky.pixel_vectors(nside)
    times = np.linspace(0, YEAR, 13)
    R = sky.point_response(times, n, orbit) / sky.isotropic_mean(nside)
    S = R[0] + R[1]

    closed = []
    for i, t in enumerate(times):
        c = constellation_normal(t, orbit) @ n
        C = 1 + 6 * c ** 2 + c ** 4
        k = np.sum(S[i] * C) / np.sum(C * C)
        closed.append(np.max(np.abs(S[i] - k * C)) / S[i].max())

    anti = hp.vec2pix(nside, *(-n))
    parity = np.max(np.abs(R - R[:, :, anti])) / R.max()

    cl = np.array([[hp.anafast(R[c, i]) for i in range(len(times))] for c in (0, 1)])   # (2, nt, 3 nside)
    cl_frac = cl / cl.sum(axis=-1, keepdims=True)
    ell = np.arange(cl.shape[-1])
    outside = cl_frac[..., (ell % 2 == 1) | (ell > 4)].sum(axis=-1).max()

    iso = R.mean(axis=-1)                                                                  # (2, nt)
    return {
        'closed_form_max_rel_residual': float(np.max(closed)),
        'parity_max_rel_violation': float(parity),
        'power_outside_l024_max_fraction': float(outside),
        'cl_fraction_A_t0': {int(l): float(cl_frac[0, 0, l]) for l in range(6)},
        'isotropic_P_range': [float(iso.min()), float(iso.max())],
    }, (times, R, cl_frac)


def formula_comparison(nside=16, ntimes=25):
    """
    ``envelopes_gaussian`` at Sigma1 = Sigma2 = 0 (a point source) against the first-principles response.
    Returns the best-scale shape residual of A^2 + E^2 and the formula's parity violation and odd-l power.
    """
    n = sky.pixel_vectors(nside)
    theta, lam = hp.pix2ang(nside, np.arange(n.shape[1]))
    sinb = np.cos(theta)
    times = np.linspace(0, YEAR, ntimes)
    R = sky.point_response(times, n)
    A, E = envelopes_gaussian(sinb[None], lam[None], 0., 0., 0., 1 / YEAR, times[:, None])
    # The formula integrates a Gaussian in (beta, lambda) with the measure cos(beta): its point-source limit is
    # R cos(beta) (see envelope_check.py, which also covers the two transcription errors)
    F = np.stack([np.asarray(A) ** 2, np.asarray(E) ** 2]) / np.cos(np.arcsin(sinb))
    Sr, Sf = R[0] + R[1], F[0] + F[1]
    k = np.sum(Sr * Sf) / np.sum(Sf * Sf)
    anti = hp.vec2pix(nside, *(-n))
    cl = hp.anafast(F[0, 0])
    return {
        'sum_shape_max_rel_residual': float(np.max(np.abs(Sr - k * Sf)) / Sr.max()),
        'parity_max_rel_violation': float(np.max(np.abs(F - F[:, :, anti])) / F.max()),
        'cl_fraction_A_t0': {int(l): float(cl[l] / cl.sum()) for l in range(8)},
    }, (times, Sr, k * Sf)


def plot_response(checks_data, formula_data, path):
    times, R, cl_frac = checks_data
    ftimes, Sr, Sf = formula_data
    fig, axes = plt.subplots(1, 2, figsize=(9, 3.2))
    ax = axes[0]
    for c, name in enumerate('AE'):
        ax.semilogy(np.arange(9), np.maximum(cl_frac[c, 0, :9], 1e-18), 'o-', color=COLORS[c], ms=4, label=name)
    ax.set(xlabel='multipole l', ylabel='fraction of response power', title='point-source response: harmonic content',
           ylim=(1e-18, 2))
    ax.legend(frameon=False)
    ax = axes[1]
    pix = [0, Sr.shape[1] // 3, 2 * Sr.shape[1] // 3]
    for i, p in enumerate(pix):
        ax.plot(ftimes / sky.DAY, Sr[:, p], color=COLORS[i], label=f'first principles, pixel {p}')
        ax.plot(ftimes / sky.DAY, Sf[:, p], '--', color=COLORS[i], lw=1.2)
    ax.set(xlabel='time [d]', ylabel='R_A + R_E', title='first principles (solid) vs envelopes_gaussian / cos β, Σ=0 (dashed)')
    ax.legend(frameon=False, fontsize=7)
    fig.savefig(path)
    plt.close(fig)


def plot_sky(maps, cfg, path):
    nside = hp.npix2nside(len(maps['galaxy']))
    area = 4 * np.pi / hp.nside2npix(nside)
    names = [p['name'] for p in cfg['patches']]
    total = sum(maps.values())
    excess = sum(maps[k] for k in names)
    fig = plt.figure(figsize=(11, 6.4))
    logI = np.log10(total / area)
    plotting.mollview(logI, (2, 2, 1), 'log10 intensity: Galaxy + patches (flux / sr)', vmin=logI.max() - 4,
                      vmax=logI.max(), fig=fig.number)
    for p, c in zip(cfg['patches'], COLORS[1:]):
        plotting.mark(p['l'], p['b'], p['name'], c)
    plotting.mollview(excess / area, (2, 2, 2), 'patches alone (flux / sr)', fig=fig.number)
    vis_gal = sky.bandlimit(maps['galaxy']) / area
    vis_exc = sky.bandlimit(excess) / area
    plotting.mollview(vis_gal, (2, 2, 3), 'Galaxy as LISA sees it: l = 0, 2, 4 only', diverging=True, fig=fig.number)
    plotting.mollview(vis_exc, (2, 2, 4), 'patches as LISA sees them: l = 0, 2, 4 only', diverging=True,
                      fig=fig.number)
    for p, c in zip(cfg['patches'], COLORS[1:]):
        plotting.mark(p['l'], p['b'], None, c)
    fig.savefig(path)
    plt.close(fig)


def plot_modulation(times, P, cfg, path):
    names = [p['name'] for p in cfg['patches']]
    fig, axes = plt.subplots(2, 2, figsize=(10, 5.6), sharex=True)
    td = times / sky.DAY
    for c, ch in enumerate('AE'):
        ax = axes[0, c]
        ax.plot(td, P['galaxy'][c], color=COLORS[0], label='Galaxy')
        ax.plot(td, sum(P[k][c] for k in P), color=MUTED, ls='--', label='Galaxy + patches')
        ax.set(title=f'channel {ch}: power modulation P(t)', ylabel='P (isotropic sky = 1)')
        ax.legend(frameon=False)
        ax = axes[1, c]
        for k, col in zip(names, COLORS[1:]):
            ax.plot(td, P[k][c] / P['galaxy'][c], color=col, label=k)
        ax.set(title=f'channel {ch}: patch contribution relative to the Galaxy', xlabel='time [d]',
               ylabel='ΔP / P_galaxy')
        ax.legend(frameon=False)
    fig.savefig(path)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    cfg = sky.load_config()

    checks, checks_data = response_checks()
    formula, formula_data = formula_comparison()
    plot_response(checks_data, formula_data, os.path.join(OUT, 'response_checks.png'))

    maps = sky.sky_maps(cfg)
    plot_sky(maps, cfg, os.path.join(OUT, 'sky.png'))

    times = np.linspace(0, T_OBS, 721)
    resp = sky.SkyResponse(times, cfg['maps']['nside'])
    P = {k: resp.modulation(m) for k, m in maps.items()}
    plot_modulation(times, P, cfg, os.path.join(OUT, 'modulation.png'))

    names = [p['name'] for p in cfg['patches']]
    excess = sum(maps[k] for k in names)
    visible = {k: float(np.linalg.norm(sky.bandlimit(m)) / np.linalg.norm(m)) for k, m in maps.items()}
    summary = {
        'response': checks,
        'envelopes_gaussian_point_source': formula,
        'sky': {
            'galaxy_flux_within_10deg_of_plane': float(maps['galaxy'][np.abs(
                sky.ecliptic_to_galactic(sky.pixel_vectors(cfg['maps']['nside']))[2]) < np.sin(np.radians(10))].sum()),
            'visible_norm_fraction': visible,
            'patch_excess_total_flux': float(excess.sum()),
        },
        'modulation_30d': {
            k: {'mean_A': float(P[k][0].mean()), 'mean_E': float(P[k][1].mean()),
                'peak_to_peak_A': float(np.ptp(P[k][0])), 'peak_to_peak_E': float(np.ptp(P[k][1]))}
            for k in P},
    }
    with open(os.path.join(OUT, 'checks.json'), 'w') as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == '__main__':
    main()
