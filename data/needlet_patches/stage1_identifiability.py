"""
Stage 1: what can 30 days of A/E auto-power data tell us about bright patches? Fisher analysis, no MCMC.

All sky parametrisations are linear in the map, so with ``W_c = fisher_P`` (information on P_c per time group)
and the Jacobian ``J`` of P with respect to the parameters, the Fisher matrix is ``J^T W J``. Spectra are fixed at
their true values; the Galaxy nuisances are its flux and shape (R_d, z_d, r_b, bulge_fraction).

1. Harmonic modes: the response only contains l = 0, 2, 4, so 15 real Y_lm span everything the data can see.
   Eigen-decomposition of their Fisher matrix: which sky patterns are measured, and how well.
2. Needlets (j = -1..3): information per band; j = 3 (l >= 5) must be blind.
3. Patch detection: SNR with the Galaxy known, and marginalised over the Galaxy nuisances.
4. Patch localisation: Fisher errors on (flux, l, b) of each patch, marginalised over the Galaxy and the other
   patches.
5. Needlet reconstruction of the excess from the noise-free data (Wiener filter, Gaussian prior per band set by
   the Galaxy's own needlet coefficients), with its posterior uncertainty: the best a needlet fit can do.

    python stage1_identifiability.py   -> outputs/stage1/{fisher.json, modes.png, needlet_bands.png,
                                                          localisation.png, reconstruction.png}
"""
import os
import json
import copy

import numpy as np
import healpy as hp

import sky
import needlets
import patches
import plotting
from plotting import plt, COLORS, MUTED

OUT = os.path.join(sky.HERE, 'outputs', 'stage1')
GALAXY_NUISANCE = ['R_d', 'z_d', 'r_b', 'bulge_fraction']


class Setup:
    """Truth, likelihood times, Fisher weights and the response at the likelihood times."""

    def __init__(self, tag='sim'):
        self.truth = patches.Truth()
        self.cfg = self.truth.cfg
        self.nside = self.cfg['maps']['nside']
        self.area = 4 * np.pi / hp.nside2npix(self.nside)
        data = patches.load_or_simulate(tag, self.truth)
        self.like = patches.likelihood(data)
        self.times = np.asarray(self.like.times)
        self.response = sky.SkyResponse(self.times, self.nside)
        S_gal, S_noise = patches.spectra(self.like, self.truth.sources)
        self.P_true = self.response.modulation(self.truth.total)
        self.W = patches.fisher_P(self.like, S_gal, S_noise, self.P_true).reshape(-1)     # (2 nt,)
        self.names = [p['name'] for p in self.cfg['patches']]

    def jac(self, flux_maps):
        """Jacobian (2 nt, K) of P for flux-map columns (npix, K)."""
        return self.response.modulation(flux_maps).reshape(-1, flux_maps.shape[1])

    def fisher(self, J):
        return J.T @ (self.W[:, None] * J)

    def galaxy_jacobian(self, rel_step=0.02):
        """Columns: Galaxy flux scale, then d(map)/d(log shape parameter) for GALAXY_NUISANCE."""
        g = self.truth.maps['galaxy']
        cols = [g]
        for name in GALAXY_NUISANCE:
            maps = []
            for s in (1 + rel_step, 1 - rel_step):
                c = copy.deepcopy(self.cfg['galaxy'])
                c[name] *= s
                maps.append(sky.galaxy_map(self.nside, c))
            cols.append((maps[0] - maps[1]) / (2 * np.log(1 + rel_step)))
        return self.jac(np.stack(cols, axis=1))


def real_harmonics(nside, lmax=4, even=True):
    """Orthonormal real Y_lm as intensity maps (npix, M) with labels; even l only by default."""
    npix = hp.nside2npix(nside)
    area = 4 * np.pi / npix
    maps, labels = [], []
    for ell in range(0, lmax + 1, 2 if even else 1):
        for m in range(ell + 1):
            for part in ((1.0,) if m == 0 else (1.0, 1j)):
                alm = np.zeros(hp.Alm.getsize(lmax), dtype=complex)
                alm[hp.Alm.getidx(lmax, ell, m)] = part
                y = hp.alm2map(alm, nside, lmax=lmax)
                maps.append(y / np.sqrt(np.sum(y ** 2) * area))
                labels.append(f'l={ell},m={m}' + ('' if m == 0 else ('c' if part == 1.0 else 's')))
    return np.stack(maps, axis=1), labels


def harmonic_modes(s):
    """Fisher eigenmodes of the 15 visible harmonics; SNR of each sky component in each mode."""
    Y, labels = real_harmonics(s.nside)
    J = s.jac(Y * s.area)
    F = s.fisher(J)
    lam, V = np.linalg.eigh(F)
    order = np.argsort(lam)[::-1]
    lam, V = lam[order], V[:, order]
    E = Y @ V                                                        # eigenmode intensity patterns, orthonormal
    proj = {k: (E.T @ m) for k, m in s.truth.maps.items()}           # <I_k, e> = sum flux * e
    sigma = 1 / np.sqrt(np.clip(lam, 1e-300, None))
    return {'eigenvalues': lam, 'sigma': sigma, 'modes': E, 'proj': proj, 'labels': labels}


def patch_jacobian(s, p, step_deg=0.5):
    """Jacobian of P for a patch's (log flux, l, b) (l, b in degrees)."""
    def m(l, b):
        return sky.patch_map(s.nside, l, b, p['sigma'], p['flux'])
    base = m(p['l'], p['b'])
    cols = [base]
    for dl, db in ((step_deg, 0), (0, step_deg)):
        cols.append((m(p['l'] + dl, p['b'] + db) - m(p['l'] - dl, p['b'] - db)) / (2 * step_deg))
    return s.jac(np.stack(cols, axis=1))


def detection_and_localisation(s):
    Jg = s.galaxy_jacobian()
    out = {}
    Jp = {p['name']: patch_jacobian(s, p) for p in s.cfg['patches']}
    for p in s.cfg['patches']:
        name = p['name']
        d = Jp[name][:, :1]                                           # the patch's own P contribution
        snr_fixed = float(np.sqrt(s.fisher(d)[0, 0]))
        # Profile out the Galaxy (flux only, then flux + shape)
        res = {}
        for label, cols in (('galaxy_flux', Jg[:, :1]), ('galaxy_flux_shape', Jg)):
            J = np.concatenate([d, cols], axis=1)
            cov = np.linalg.inv(s.fisher(J))
            res[label] = float(1 / np.sqrt(cov[0, 0]))
        # Localisation: (log flux, l, b) of this patch + Galaxy + the other patches' fluxes
        others = [Jp[q['name']][:, :1] for q in s.cfg['patches'] if q['name'] != name]
        J = np.concatenate([Jp[name], Jg] + others, axis=1)
        cov = np.linalg.inv(s.fisher(J))
        J0 = np.concatenate([Jp[name], Jg[:, :1]], axis=1)
        cov0 = np.linalg.inv(s.fisher(J0))
        out[name] = {
            'snr_galaxy_known': snr_fixed,
            'snr_marg_galaxy_flux': res['galaxy_flux'],
            'snr_marg_galaxy_flux_shape': res['galaxy_flux_shape'],
            'sigma_l_deg_galaxy_flux_only': float(np.sqrt(cov0[1, 1])),
            'sigma_b_deg_galaxy_flux_only': float(np.sqrt(cov0[2, 2])),
            'sigma_l_deg_full': float(np.sqrt(cov[1, 1])),
            'sigma_b_deg_full': float(np.sqrt(cov[2, 2])),
            'sigma_lnflux_full': float(np.sqrt(cov[0, 0])),
            'cov_lb_galaxy_flux_only': cov0[1:3, 1:3].tolist(),
        }
    return out


def chi2_position_map(s, p, nside_scan=16):
    """
    Delta chi^2 of moving one patch (flux re-fitted, Galaxy flux re-fitted) to every position on an nside_scan
    grid, noise-free data: the full (non-Gaussian) localisation surface, including the antipodal image.
    """
    true_d = s.jac(sky.patch_map(s.nside, p['l'], p['b'], p['sigma'], p['flux'])[:, None])[:, 0]
    g = s.jac(s.truth.maps['galaxy'][:, None])[:, 0]
    n = sky.pixel_vectors(nside_scan)
    ng = sky.ecliptic_to_galactic(n)
    out = np.empty(n.shape[1])
    for i in range(n.shape[1]):
        l, b = np.degrees(np.arctan2(ng[1, i], ng[0, i])), np.degrees(np.arcsin(ng[2, i]))
        t = s.jac(sky.patch_map(s.nside, l, b, p['sigma'], 1.0)[:, None])[:, 0]
        A = np.stack([t, g], axis=1)                      # fit patch flux + Galaxy flux (Galaxy shape at truth)
        target = true_d + g                               # residual relative to the Galaxy-only model is true_d
        F = A.T @ (s.W[:, None] * A)
        x = np.linalg.solve(F, A.T @ (s.W * target))
        r = target - A @ x
        out[i] = np.sum(s.W * r ** 2)
    return out


def wiener_reconstruction(s, frame, data_P=None):
    """
    Posterior of the excess map in the needlet frame, P = P_galaxy(theta) + T beta, with the Galaxy nuisances
    linearised around the truth (flat prior) and beta_jk ~ N(0, s_j^2), s_j = rms of the Galaxy's own needlet
    coefficients in band j. Noise-free data (``data_P = P_true``) unless given.

    Returns: mean excess map, its per-pixel std, and the coefficient posterior (mean, cov).
    """
    data_P = s.P_true if data_P is None else data_P
    T = s.jac(frame.Psi * frame.pixel_area)
    Jg = s.galaxy_jacobian()
    beta_gal = frame.analysis(s.truth.maps['galaxy'])
    prior_sd = np.array([np.sqrt(np.mean(beta_gal[frame.band == j] ** 2)) for j in frame.band])
    J = np.concatenate([T, Jg], axis=1)
    prec = s.fisher(J)
    prec[np.arange(frame.K), np.arange(frame.K)] += 1 / prior_sd ** 2
    prec[frame.K:, frame.K:] += np.eye(Jg.shape[1]) * 1e-8
    resid = (data_P - s.response.modulation(s.truth.maps['galaxy'])).reshape(-1)
    cov = np.linalg.inv(prec)
    mean = cov @ (J.T @ (s.W * resid))
    bm, bc = mean[:frame.K], cov[:frame.K, :frame.K]
    S = frame.Psi * frame.pixel_area                              # flux-map synthesis matrix
    mean_map = S @ bm
    sd_map = np.sqrt(np.einsum('pk,kl,pl->p', S, bc, S))
    return mean_map, sd_map, bm, bc, prior_sd


# ---------------------------------------------------------------------------------------------------------------

def plot_modes(h, s, path, nshow=8):
    fig = plt.figure(figsize=(12, 9.5))
    for k in range(nshow):
        snr = {name: abs(h['proj'][name][k]) / h['sigma'][k] for name in ['galaxy'] + s.names}
        title = (f"mode {k + 1}: σ = {h['sigma'][k]:.3g}\nSNR gal {snr['galaxy']:.1f} | "
                 + ' '.join(f"{n[:4]} {snr[n]:.1f}" for n in s.names))
        plotting.mollview(h['modes'][:, k], (3, 3, k + 1), title, diverging=True, fig=fig.number)
        for p, c in zip(s.cfg['patches'], COLORS[1:]):
            plotting.mark(p['l'], p['b'], None, c)
    ax = fig.add_subplot(3, 3, 9)
    ax.semilogy(np.arange(1, 16), h['sigma'], 'o-', color=COLORS[0], ms=4)
    ax.set(xlabel='eigenmode', ylabel='σ (intensity coefficient)', title='Fisher eigenmodes of the 15 visible Y_lm')
    fig.savefig(path)
    plt.close(fig)


def plot_bands(frame, s, path):
    T = s.jac(frame.Psi * frame.pixel_area)
    F = s.fisher(T)
    lam, V = np.linalg.eigh(F)
    lam = lam[::-1]
    fig, axes = plt.subplots(1, 2, figsize=(9.5, 3.2))
    js = list(range(-1, frame.jmax + 1))
    info = [np.trace(F[np.ix_(frame.band == j, frame.band == j)]) for j in js]
    axes[0].bar([str(j) for j in js], np.maximum(np.array(info) / sum(info), 1e-18), color=COLORS[0], width=0.6)
    axes[0].set(yscale='log', xlabel='needlet band j', ylabel='share of Fisher trace',
                title='information per needlet band')
    axes[1].semilogy(np.arange(1, 31), np.clip(lam[:30] / lam[0], 1e-18, None), 'o-', color=COLORS[0], ms=4)
    axes[1].set(xlabel='eigenvalue index', ylabel='λ / λ_max', title=f'needlet Fisher spectrum (K = {frame.K})')
    fig.savefig(path)
    plt.close(fig)
    return {str(j): float(i / sum(info)) for j, i in zip(js, info)}, (lam / lam[0])[:20].tolist()


def normal_track(s, n=7):
    """Galactic (l, b) in degrees of the constellation normal over the observation (start to end)."""
    from bahamas.psd_response import orbits as orb
    orbit = orb.SpacecraftOrbit()
    z = []
    for t in np.linspace(s.times[0], s.times[-1], n):
        p = np.asarray(orbit.SC_positions(t))
        v = np.cross(p[1] - p[0], p[2] - p[0])
        z.append(v / np.linalg.norm(v))
    zg = sky.ecliptic_to_galactic(np.array(z).T)
    return np.degrees(np.arctan2(zg[1], zg[0])), np.degrees(np.arcsin(zg[2])), np.array(z).T


def plot_localisation(s, chi2maps, path):
    l_n, b_n, _ = normal_track(s)
    fig = plt.figure(figsize=(12, 3.4))
    for i, (p, c) in enumerate(zip(s.cfg['patches'], COLORS[1:])):
        m = hp.ud_grade(np.sqrt(np.clip(chi2maps[p['name']], 0, None)), s.nside)
        plotting.mollview(m, (1, 3, i + 1), f"{p['name']}: sqrt Δχ² of moving the patch", fig=fig.number,
                          cmap='Blues_r')
        plotting.mark(p['l'], p['b'], 'true', c)
        hp.projplot(l_n, b_n, lonlat=True, coord='G', color=MUTED, lw=2)
        hp.projplot(l_n + 180, -b_n, lonlat=True, coord='G', color=MUTED, lw=2, ls=':')
    fig.savefig(path)
    plt.close(fig)


def ring_test(s, chi2maps, nside_scan=16):
    """
    Are the near-degenerate positions a cone around the constellation normal? For positions with
    Delta chi^2 < 1: spread of |cos(angle to the mean normal)| compared with the whole sky.
    """
    _, _, z = normal_track(s)
    zbar = z.mean(axis=1)
    zbar /= np.linalg.norm(zbar)
    n = sky.pixel_vectors(nside_scan)
    cosang = np.abs(zbar @ n)
    out = {}
    for p in s.cfg['patches']:
        c = chi2maps[p['name']]
        true_cos = float(abs(zbar @ sky.galactic_direction(p['l'], p['b'])))
        sel = c < 1.0
        out[p['name']] = {'true_abs_cos_to_normal': true_cos,
                          'dchi2_lt_1_abs_cos_mean': float(cosang[sel].mean()),
                          'dchi2_lt_1_abs_cos_std': float(cosang[sel].std()),
                          'all_sky_abs_cos_std': float(cosang.std())}
    return out


def plot_reconstruction(s, rec, path):
    mean_map, sd_map = rec[:2]
    excess = sum(s.truth.maps[k] for k in s.names)
    a = s.area
    vis = sky.bandlimit(excess) / a
    vmax = np.max(np.abs(vis))
    panels = [(excess / a, 'true excess (patches)', {}),
              (vis, 'true excess, l = 0, 2, 4 part (the best possible)', dict(diverging=True, vmax=vmax)),
              (mean_map / a, 'needlet reconstruction, noise-free data (posterior mean)',
               dict(diverging=True, vmax=vmax)),
              (mean_map / sd_map, 'posterior mean / std', dict(diverging=True, vmax=5))]
    fig = plt.figure(figsize=(11, 6.4))
    for k, (m, title, kw) in enumerate(panels):
        plotting.mollview(m, (2, 2, k + 1), title, fig=fig.number, **kw)
        for p, c in zip(s.cfg['patches'], COLORS[1:]):
            plotting.mark(p['l'], p['b'], None, c)
    fig.savefig(path)
    plt.close(fig)


def main():
    os.makedirs(OUT, exist_ok=True)
    s = Setup()
    h = harmonic_modes(s)
    plot_modes(h, s, os.path.join(OUT, 'modes.png'))

    frame = needlets.NeedletFrame(s.nside, jmax=3)
    band_info, needlet_spectrum = plot_bands(frame, s, os.path.join(OUT, 'needlet_bands.png'))

    det = detection_and_localisation(s)
    chi2maps = {p['name']: chi2_position_map(s, p) for p in s.cfg['patches']}
    plot_localisation(s, chi2maps, os.path.join(OUT, 'localisation.png'))

    frame2 = needlets.NeedletFrame(s.nside, jmax=2)
    rec = wiener_reconstruction(s, frame2)
    plot_reconstruction(s, rec, os.path.join(OUT, 'reconstruction.png'))
    excess = sum(s.truth.maps[k] for k in s.names)
    vis = sky.bandlimit(excess)
    corr = float(np.corrcoef(rec[0], vis)[0, 1])

    ng = sky.ecliptic_to_galactic(sky.pixel_vectors(16))
    minima = {}
    for p in s.cfg['patches']:
        c = chi2maps[p['name']]
        within = np.where(c < 1.0)[0]                    # positions indistinguishable at 1 sigma
        minima[p['name']] = {
            'positions_within_dchi2_1_fraction_of_sky': float(len(within) / len(c)),
            'positions_within_dchi2_4_fraction_of_sky': float(np.mean(c < 4.0)),
        }

    np.savez(os.path.join(OUT, 'chi2_position_maps.npz'), **chi2maps)
    out = {
        'ring_test': ring_test(s, chi2maps),
        'fisher_P_sigma_per_group_mean': float(np.mean(1 / np.sqrt(s.W))),
        'harmonic_modes': {
            'sigma': h['sigma'].tolist(),
            'snr_per_mode': {k: (np.abs(v) / h['sigma']).tolist() for k, v in h['proj'].items()},
            'n_modes_sigma_below_galaxy_monopole_10pct': int(np.sum(h['sigma'] < 0.1 * h['proj']['galaxy'][0])),
        },
        'needlet_band_information_share': band_info,
        'needlet_fisher_spectrum_rel': needlet_spectrum,
        'patches': det,
        'localisation_scan': minima,
        'wiener_reconstruction_corr_with_visible_truth': corr,
    }
    with open(os.path.join(OUT, 'fisher.json'), 'w') as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == '__main__':
    main()
