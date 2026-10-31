"""
Sky model and sky-to-modulation forward model of the needlet patch study.

The Galactic foreground is a sum of unresolved sources that all share the spectrum ``S_gal(f)``; only their sky
distribution ``I(n)`` (flux per pixel, ``sum_pix I = total flux``) differs. In the low-frequency limit the power a
point source at ``n`` puts in TDI channel ``c`` at time ``t`` is ``R_c(t, n) = F+^2 + Fx^2`` (polarisation-averaged
antenna pattern of the channel), so the power modulation is linear in the map::

    P_c(t) = sum_pix I(pix) R_c(t, pix) / Rbar

``Rbar`` is the all-sky mean of ``R_A`` (equal to that of ``R_E`` and constant in time), so an isotropic sky of unit
flux has ``P_A = P_E = 1``. ``S_gal`` is thus the PSD the same sources would give if spread isotropically.

``R_c`` is computed from first principles: equal-arm Michelson X, Y, Z from the spacecraft positions of
``bahamas.psd_response.orbits`` (arms are fixed during a light travel time at low frequency), then
A = (Z - X)/sqrt2, E = (X - 2Y + Z)/sqrt6. As a function of ``n`` it is parity-even and contains only l = 0, 2, 4
(checked in ``stage0_forward.py``): that is all the sky information an A/E auto-power analysis can access.

Coordinates: maps are HEALPix (RING) in ecliptic coordinates, like the orbit; galactic positions are rotated.
"""
import os

import numpy as np
import healpy as hp
import yaml

import bahamas.wavelet  # noqa: F401  (sets the JAX backend before any bahamas.psd_* module is imported)
from bahamas.psd_response import orbits as orb

HERE = os.path.dirname(os.path.abspath(__file__))
SKY_CONFIG = os.path.join(HERE, 'config', 'sky.yaml')
DAY = 86400.0


def load_config(path=SKY_CONFIG):
    with open(path) as f:
        return yaml.safe_load(f)


# ---------------------------------------------------------------------------------------------------------------
# Coordinates
# ---------------------------------------------------------------------------------------------------------------

def pixel_vectors(nside):
    """Unit vectors (3, npix) of the ecliptic HEALPix pixel centres."""
    return np.array(hp.pix2vec(nside, np.arange(hp.nside2npix(nside))))


def ecliptic_to_galactic(vec):
    """Rotate ecliptic unit vectors (3, N) to galactic ones."""
    return np.asarray(hp.Rotator(coord=['E', 'G'])(vec))


def galactic_to_ecliptic(vec):
    return np.asarray(hp.Rotator(coord=['G', 'E'])(vec))


def galactic_direction(l_deg, b_deg):
    """Ecliptic unit vector (3,) of a galactic direction (l, b) in degrees."""
    l, b = np.radians(l_deg), np.radians(b_deg)
    return galactic_to_ecliptic(np.array([np.cos(b) * np.cos(l), np.cos(b) * np.sin(l), np.sin(b)]))


# ---------------------------------------------------------------------------------------------------------------
# Response
# ---------------------------------------------------------------------------------------------------------------

def _polarisation_basis(n):
    """Orthonormal transverse vectors (u, v) for directions n (3, N); any basis gives the same F+^2 + Fx^2."""
    theta = np.arccos(np.clip(n[2], -1, 1))
    phi = np.arctan2(n[1], n[0])
    u = np.array([np.cos(theta) * np.cos(phi), np.cos(theta) * np.sin(phi), -np.sin(theta)])
    v = np.array([-np.sin(phi), np.cos(phi), np.zeros_like(phi)])
    return u, v


def point_response(times, n, orbit=None):
    """
    Low-frequency A and E power response ``R_c(t, n) = F+^2 + Fx^2`` of a point source.

    Args:
        times (array): Times in seconds, shape (nt,).
        n (array): Source directions, unit vectors (3, N), ecliptic.
        orbit (SpacecraftOrbit, optional): Orbit model (bahamas default).

    Returns:
        np.ndarray: Shape (2, nt, N), channels A, E.
    """
    orbit = orb.SpacecraftOrbit() if orbit is None else orbit
    u, v = _polarisation_basis(n)
    times = np.atleast_1d(times)
    out = np.empty((2, len(times), n.shape[1]))
    for i, t in enumerate(times):
        p = np.asarray(orbit.SC_positions(t))          # rows are spacecraft

        def arm(a, b):
            d = p[b] - p[a]
            return d / np.linalg.norm(d)

        # Michelson X, Y, Z: half the difference of the two arm responses r.e.r at each vertex
        def michelson(a, b, c):
            ra, rb = arm(a, b), arm(a, c)
            ua, va, ub, vb = u.T @ ra, v.T @ ra, u.T @ rb, v.T @ rb
            return 0.5 * ((ua ** 2 - va ** 2) - (ub ** 2 - vb ** 2)), (ua * va - ub * vb)

        (Xp, Xc), (Yp, Yc), (Zp, Zc) = michelson(0, 1, 2), michelson(1, 2, 0), michelson(2, 0, 1)
        Ap, Ac = (Zp - Xp) / np.sqrt(2), (Zc - Xc) / np.sqrt(2)
        Ep, Ec = (Xp - 2 * Yp + Zp) / np.sqrt(6), (Xc - 2 * Yc + Zc) / np.sqrt(6)
        out[0, i] = Ap ** 2 + Ac ** 2
        out[1, i] = Ep ** 2 + Ec ** 2
    return out


def isotropic_mean(nside=16):
    """All-sky mean of ``R_A`` (= that of ``R_E``, constant in time): the normalisation ``Rbar``."""
    return float(point_response([0.0], pixel_vectors(nside))[0, 0].mean())


class SkyResponse:
    """
    Response matrix ``R_c(t, pix) / Rbar`` on a HEALPix grid and a time grid, so that ``P = R @ I``.

    Args:
        times (array): Times in seconds.
        nside (int): HEALPix resolution of the maps.
    """

    def __init__(self, times, nside):
        self.times = np.asarray(times, dtype=float)
        self.nside = nside
        self.Rbar = isotropic_mean()
        self.R = point_response(self.times, pixel_vectors(nside)) / self.Rbar   # (2, nt, npix)

    def modulation(self, flux_map):
        """``P_c(t)`` of a flux map (npix,) or of several maps (npix, K) -> (2, nt) or (2, nt, K)."""
        return np.einsum('ctp,p...->ct...', self.R, flux_map)

    def interpolator(self, flux_map):
        """``P(t, channel)`` callable (for ``simulate_channels``), linear interpolation of the grid."""
        P = self.modulation(flux_map)

        def P_of_t(t, channel):
            import jax.numpy as jnp
            return jnp.interp(jnp.asarray(t), self.times, P[channel])
        return P_of_t


# ---------------------------------------------------------------------------------------------------------------
# Sky maps (flux per pixel)
# ---------------------------------------------------------------------------------------------------------------

def galaxy_map(nside, cfg, nsteps=1500, chunk=2048):
    """
    Disk + bulge Galaxy: flux per pixel proportional to the line-of-sight integral of the density
    (number of sources per solid angle ~ int rho s^2 ds, flux of each ~ 1/s^2). Normalised to total flux 1.
    """
    n_gal = ecliptic_to_galactic(pixel_vectors(nside))
    s = np.linspace(0.0, cfg['s_max'], nsteps)
    ds = s[1] - s[0]
    rho_d0 = 1.0 / (4 * np.pi * cfg['R_d'] ** 2 * cfg['z_d'])              # unit-mass disk
    rho_b0 = 1.0 / (np.pi ** 1.5 * cfg['r_b'] ** 3)                          # unit-mass bulge
    fb = cfg['bulge_fraction']
    out = np.empty(n_gal.shape[1])
    for i in range(0, n_gal.shape[1], chunk):
        n = n_gal[:, i:i + chunk]
        # Galactocentric position along the line of sight; the Sun at (-R_sun, 0, z_sun)
        x = -cfg['R_sun'] + s[:, None] * n[0]
        y = s[:, None] * n[1]
        z = cfg['z_sun'] + s[:, None] * n[2]
        R = np.hypot(x, y)
        rho = ((1 - fb) * rho_d0 * np.exp(-R / cfg['R_d'] - np.abs(z) / cfg['z_d'])
               + fb * rho_b0 * np.exp(-(R ** 2 + z ** 2) / cfg['r_b'] ** 2))
        out[i:i + chunk] = rho.sum(axis=0) * ds
    return out / out.sum()


def patch_map(nside, l, b, sigma, flux):
    """Gaussian disc of width ``sigma`` (deg) at galactic (l, b), total flux ``flux``."""
    centre = galactic_direction(l, b)
    theta = np.arccos(np.clip(centre @ pixel_vectors(nside), -1, 1))
    m = np.exp(-0.5 * (theta / np.radians(sigma)) ** 2)
    return flux * m / m.sum()


def sky_maps(cfg=None, nside=None):
    """``{'galaxy': map, <patch name>: map, ...}`` (flux per pixel) from the sky configuration."""
    cfg = load_config() if cfg is None else cfg
    nside = cfg['maps']['nside'] if nside is None else nside
    maps = {'galaxy': galaxy_map(nside, cfg['galaxy'])}
    for p in cfg['patches']:
        maps[p['name']] = patch_map(nside, p['l'], p['b'], p['sigma'], p['flux'])
    return maps


def bandlimit(flux_map, lmax=4, even=True):
    """The part of a map an A/E auto-power analysis can see: harmonics l <= lmax (even l only)."""
    nside = hp.npix2nside(len(flux_map))
    alm = hp.map2alm(flux_map, lmax=3 * nside - 1, iter=3)
    ell = hp.Alm.getlm(3 * nside - 1)[0]
    keep = (ell <= lmax) & ((ell % 2 == 0) | (not even))
    return hp.alm2map(np.where(keep, alm, 0), nside)
