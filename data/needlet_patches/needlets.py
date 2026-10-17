"""
Spherical needlets (Narcowich, Petrushev & Ward 2006; Marinucci et al. 2008) on HEALPix.

A needlet of band ``j`` centred on the cubature point ``xi_jk`` is

    psi_jk(n) = sqrt(lambda_jk) sum_l b(l / B^j) (2l + 1) / (4 pi) P_l(xi_jk . n),

with ``b`` the standard smooth window supported on l in (B^(j-1), B^(j+1)), the points ``xi_jk`` the pixel centres
of a HEALPix grid of resolution ``nside_j`` and weights ``lambda_jk = 4 pi / N_j``. Band ``j = -1`` is the scaling
function ``sqrt(phi(l))``, which for B = 2 is the monopole alone. Since ``phi(l) + sum_j b^2(l / B^j) = 1``, the
needlets form a (nearly, the HEALPix cubature is not exact) tight frame::

    f(n) = sum_jk beta_jk psi_jk(n),   beta_jk = int f psi_jk dOmega.

Each needlet is localised in space (width ~ pi / B^j, quasi-exponential tails) and in harmonic space (a few l).
"""
import numpy as np
import healpy as hp
from scipy.integrate import quad
from scipy.special import eval_legendre


def _bump(t):
    t = np.asarray(t, dtype=float)
    out = np.zeros_like(t)
    inside = np.abs(t) < 1
    out[inside] = np.exp(-1.0 / (1.0 - t[inside] ** 2))
    return out


_BUMP_NORM = quad(lambda t: float(_bump(t)), -1, 1)[0]


def _psi(u):
    """Smooth step from 0 (u <= -1) to 1 (u >= 1)."""
    u = np.atleast_1d(np.asarray(u, dtype=float))
    return np.array([quad(lambda t: float(_bump(t)), -1, min(max(x, -1), 1))[0] for x in u]) / _BUMP_NORM


def _phi(t, B):
    """Low-pass function: 1 on [0, 1/B], smooth decrease to 0 at 1."""
    t = np.atleast_1d(np.asarray(t, dtype=float))
    out = np.zeros_like(t)
    out[t <= 1.0 / B] = 1.0
    mid = (t > 1.0 / B) & (t < 1.0)
    out[mid] = _psi(1.0 - 2.0 * B / (B - 1.0) * (t[mid] - 1.0 / B))
    return out


def window(j, lmax, B=2.0):
    """Harmonic window ``b_j(l)``, l = 0..lmax; ``j = -1`` is the scaling function."""
    ell = np.arange(lmax + 1)
    if j < 0:
        return np.sqrt(np.clip(_phi(ell, B), 0, None))
    return np.sqrt(np.clip(_phi(ell / B ** (j + 1), B) - _phi(ell / B ** j, B), 0, None))


def band_lmax(j, B=2.0):
    """Largest l with ``b_j(l) > 0``."""
    return 0 if j < 0 else int(np.ceil(B ** (j + 1))) - 1


def default_nside(j):
    """Cubature grid of band j: nside_j = 2^j (12, 48, 192, 768 points for j = 0..3); one point for j = -1."""
    return None if j < 0 else 2 ** j


class NeedletFrame:
    """
    Needlets of bands ``j = -1 .. jmax`` evaluated on a HEALPix map grid.

    Attributes:
        Psi (np.ndarray): Needlet maps (npix, K), intensity units (per steradian).
        band (np.ndarray): Band ``j`` of each column.
        centres (np.ndarray): Unit vectors (3, K) of the needlet centres (the north pole for j = -1).
        windows (dict): ``b_j(l)`` per band.
    """

    def __init__(self, nside, jmax, B=2.0, nsides=None):
        self.nside, self.jmax, self.B = nside, jmax, B
        npix = hp.nside2npix(nside)
        n = np.array(hp.pix2vec(nside, np.arange(npix)))
        self.pixel_area = 4 * np.pi / npix
        nsides = {} if nsides is None else nsides
        cols, bands, centres, self.windows = [], [], [], {}
        for j in range(-1, jmax + 1):
            lmax = band_lmax(j, B)
            b = window(j, lmax, B)
            self.windows[j] = b
            if j < 0:
                xi = np.array([[0.0], [0.0], [1.0]])
                lam = 4 * np.pi
            else:
                ns = nsides.get(j, default_nside(j))
                xi = np.array(hp.pix2vec(ns, np.arange(hp.nside2npix(ns))))
                lam = 4 * np.pi / xi.shape[1]
            cosang = np.clip(n.T @ xi, -1, 1)                                     # (npix, K_j)
            psi = np.zeros_like(cosang)
            for ell in range(lmax + 1):
                if b[ell] > 0:
                    psi += b[ell] * (2 * ell + 1) / (4 * np.pi) * eval_legendre(ell, cosang)
            cols.append(np.sqrt(lam) * psi)
            bands.append(np.full(xi.shape[1], j))
            centres.append(xi)
        self.Psi = np.concatenate(cols, axis=1)
        self.band = np.concatenate(bands)
        self.centres = np.concatenate(centres, axis=1)

    @property
    def K(self):
        return self.Psi.shape[1]

    def analysis(self, flux_map):
        """Needlet coefficients ``beta = int f psi dOmega`` of a flux map (flux per pixel, f = flux / area)."""
        return self.Psi.T @ flux_map

    def synthesis(self, beta):
        """Flux map ``sum beta_jk psi_jk * pixel_area`` from coefficients."""
        return (self.Psi @ beta) * self.pixel_area

    def band_maps(self, beta):
        """Flux map of each band separately: ``{j: map}``."""
        return {j: (self.Psi[:, self.band == j] @ beta[self.band == j]) * self.pixel_area
                for j in range(-1, self.jmax + 1)}
