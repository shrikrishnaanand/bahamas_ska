"""
Run 5 sky: run 2's smooth Galaxy plus a few compact patches of excess power, as HEALPix flux maps, and the power
modulation ``P_c(t)`` they produce in A and E.

All Galactic power comes from unresolved DWDs with one common spectrum ``S_gal(f)``; the components differ only in
where the power sits. The modulation is linear in the sky map (``data/needlet_patches/sky.py``, first-principles
low-frequency response ``R_c(t, n)``)::

    P_c(t) = k * sum_pix I(pix) R_c(t, pix) / Rbar

- **Smooth Galaxy**: the bivariate Gaussian in ecliptic (beta, lambda) of ``envelopes_gaussian`` with the pack-2 sky
  parameters of run 2 (``config/sources.yaml``). The formula integrates the Gaussian density p(beta, lambda) with the
  sky measure cos(beta) d beta d lambda, so the flux per steradian is p itself and the flux per pixel p * dOmega.
  Normalised to total flux 1.
- **Patches** (``config/run5_sky.yaml``): elliptical Gaussians on the sphere at real positions (LMC, SMC, Sgr dSph,
  near end of the Galactic bar), flux in units of the smooth Galaxy's flux.
- ``k`` is fitted once so that the smooth Galaxy's map-based P_c(t) reproduces run 2's analytic M_c^2(t) over the
  30 days (shape agreement checked: ``Run5Sky.galaxy_match``). Run 5's smooth part is therefore run 2's injection,
  at the same ``amp``, and the patches come on top.

Maps are stored in ecliptic coordinates (like the orbit); positions in the config are galactic.
"""
import os
import sys

import numpy as np
import healpy as hp
import yaml

HERE = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, os.path.join(HERE, '..', 'needlet_patches'))
import sky as skyresp  # noqa: E402  (first-principles A/E response, coordinates; sets the JAX backend)

from bahamas.wavelet.simulate import power_modulation  # noqa: E402

SKY_CONFIG = os.path.join(HERE, 'config', 'run5_sky.yaml')
DAY = 86400.0


def load_config(path=SKY_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


def gaussian_galaxy_map(nside, par):
    """Run 2's smooth Galaxy: the (beta, lambda) bivariate Gaussian of ``envelopes_gaussian`` as flux per pixel.

    ``par``: ``lat`` (sine of the centre latitude), ``long``, ``s1``, ``s2`` (variances, rad^2), ``psi`` (sine of
    the rotation angle), with the covariance convention of ``needlet_patches/envelope_check.gaussian_average``.
    """
    S, D = par['s1'] + par['s2'], par['s1'] - par['s2']
    cospsi = np.sqrt(1 - par['psi'] ** 2)
    C = np.array([[(S + D * cospsi) / 2, D * par['psi'] / 2], [D * par['psi'] / 2, (S - D * cospsi) / 2]])
    theta, lam = hp.pix2ang(nside, np.arange(hp.nside2npix(nside)))
    d = np.stack([np.pi / 2 - theta - np.arcsin(par['lat']),
                  np.angle(np.exp(1j * (lam - par['long'])))])          # longitude offset wrapped to (-pi, pi]
    p = np.exp(-0.5 * np.einsum('in,ij,jn->n', d, np.linalg.inv(C), d))
    return p / p.sum()


def patch_map(nside, l, b, sigma_major, sigma_minor, pa, flux, **_):
    """Elliptical Gaussian on the sphere at galactic (l, b), widths in degrees, total flux ``flux``.

    Offsets are azimuthal-equidistant about the centre; ``pa`` is the major axis' angle from galactic north towards
    increasing l.
    """
    lr, br = np.radians(l), np.radians(b)
    c = np.array([np.cos(br) * np.cos(lr), np.cos(br) * np.sin(lr), np.sin(br)])
    e_l = np.array([-np.sin(lr), np.cos(lr), 0.0])
    e_b = np.cross(c, e_l)
    n = skyresp.ecliptic_to_galactic(skyresp.pixel_vectors(nside))
    theta = np.arccos(np.clip(c @ n, -1, 1))
    phi = np.arctan2(e_l @ n, e_b @ n)                                    # from north towards +l
    a = phi - np.radians(pa)
    major, minor = theta * np.cos(a), theta * np.sin(a)
    m = np.exp(-0.5 * ((major / np.radians(sigma_major)) ** 2 + (minor / np.radians(sigma_minor)) ** 2))
    return flux * m / m.sum()


def sky_maps(cfg=None, par=None, nside=None):
    """``{'galaxy': map, <patch name>: map, ...}``, flux per pixel, galaxy normalised to 1."""
    cfg = load_config() if cfg is None else cfg
    par = sky_params() if par is None else par
    nside = cfg['maps']['nside'] if nside is None else nside
    maps = {'galaxy': gaussian_galaxy_map(nside, par)}
    for p in cfg['patches']:
        maps[p['name']] = patch_map(nside, **p)
    return maps


def sky_params():
    """Run 2's (pack-2) sky parameters."""
    import injection
    return injection.sky_params()


def realistic_flux(cfg=None, n=1_000_000, seed=0):
    """Flux fraction M / d^2 of each patch with a mass and distance, relative to the Milky Way's sum of M / s^2.

    The Milky Way term is a Monte Carlo average of 1 / s^2 over its stellar mass (exponential disk + Gaussian
    bulge, seen from the Sun). Assumes the same DWD rate per stellar mass everywhere.
    """
    cfg = load_config() if cfg is None else cfg
    mw = cfg['milky_way']
    rng = np.random.default_rng(seed)
    nb = int(n * mw['bulge_fraction'])
    R = rng.gamma(2.0, mw['R_d'], n - nb)
    ph = rng.uniform(0, 2 * np.pi, n - nb)
    disk = np.stack([R * np.cos(ph), R * np.sin(ph), rng.laplace(0, mw['z_d'], n - nb)])
    bulge = rng.normal(0, mw['r_b'] / np.sqrt(2), (3, nb))
    pos = np.concatenate([disk, bulge], axis=1) - np.array([[-mw['R_sun']], [0.0], [mw['z_sun']]])
    mw_flux = float(mw['mass_msun']) * np.mean(1.0 / np.sum(pos ** 2, axis=0))
    return {p['name']: float(p['mass_msun']) / p['dist_kpc'] ** 2 / mw_flux
            for p in cfg['patches'] if 'mass_msun' in p}


class Run5Sky:
    """
    Sky maps, response and modulation of run 5 on a time grid.

    Args:
        t0, T (float): Analysis window in seconds (run 2: 0 and 30 d).
        pad_days (float): Extra days on each side (the series is simulated padded, as in run 2).
        step_hours (float): Time grid of P(t); ``P`` interpolates it linearly (P is smooth on day scales).
    """

    def __init__(self, t0=0.0, T=30 * DAY, pad_days=2.0, step_hours=1.0, cfg=None):
        self.cfg = load_config() if cfg is None else cfg
        self.par = sky_params()
        self.t0, self.T = t0, T
        self.maps = sky_maps(self.cfg, self.par)
        self.names = list(self.maps)
        self.times = np.arange(t0 - pad_days * DAY, t0 + T + pad_days * DAY + 1, step_hours * 3600.0)
        nside_r = self.cfg['maps']['response_nside']
        self.response = skyresp.SkyResponse(self.times, nside_r)
        lowres = np.stack([hp.ud_grade(self.maps[k], nside_r, power=-2) for k in self.names], axis=1)
        self.P_unit = self.response.modulation(lowres)                     # (2, nt, ncomp), before k
        # k: map-based smooth Galaxy -> run 2's analytic envelope, least squares over the 30 days, A and E jointly
        self.P_run2 = np.stack([np.asarray(power_modulation(self.par, self.times, c)) for c in (0, 1)])
        w = (self.times >= t0) & (self.times <= t0 + T)
        g = self.P_unit[:, w, 0]
        self.k = float(np.sum(g * self.P_run2[:, w]) / np.sum(g * g))
        self.P_components = self.k * self.P_unit                           # (2, nt, ncomp)

    @property
    def total_map(self):
        return sum(self.maps.values())

    @property
    def excess_map(self):
        return sum(m for k, m in self.maps.items() if k != 'galaxy')

    def galaxy_match(self):
        """Max relative difference (per channel) of the map-based smooth Galaxy from run 2's M_c^2 over the 30 d."""
        w = (self.times >= self.t0) & (self.times <= self.t0 + self.T)
        d = self.P_components[:, w, 0] - self.P_run2[:, w]
        return np.max(np.abs(d), axis=1) / np.max(self.P_run2[:, w], axis=1)

    def P(self, t, channel):
        """Total modulation ``P(t, channel)`` (smooth Galaxy + patches), for ``simulate_channels``."""
        import jax.numpy as jnp
        return jnp.interp(jnp.asarray(t), self.times, self.P_components[channel].sum(-1))
