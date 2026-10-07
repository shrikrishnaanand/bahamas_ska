"""
Wilson-Daubechies-Meyer (WDM) wavelet transform in JAX.

The WDM basis tiles the time-frequency plane with a uniform grid of
``Nt`` time bins of width ``dT = Nf * dt`` and ``Nf`` frequency bins of
width ``dF = 1 / (2 * Nf * dt)``. It is orthonormal, so for Gaussian noise
with a slowly varying (evolutionary) PSD ``S(f, t)`` the coefficients are
approximately independent with variance ``S(f_m, t_n) / (2 dt)``.

References:
    - Necula, Klimenko & Mitselmakher (2012), arXiv:1202.5246
    - Cornish (2020), "Time-frequency analysis of gravitational wave data",
      arXiv:2009.00043 (Eq. 11 for the window, Eq. 13 for the transform)
    - Digman & Cornish (2022), "LISA gravitational wave sources in a
      time-varying galactic stochastic background", arXiv:2212.04600

The transforms are jit-compiled versions of the frequency-domain algorithm
of Cornish (2020), with the same packing convention as ``pywavelet`` (row
``m = 0`` holds the DC bin on even time indices and the Nyquist bin on odd
time indices). The normalisation makes the transform orthonormal on real
time series: ``sum(w**2) == sum(x**2)``.

All arrays use the convention ``w[n, m]``: time index first, frequency second.
"""
from functools import lru_cache, partial

import numpy as np
import jax
import jax.numpy as jnp
from scipy.special import betainc

jax.config.update('jax_enable_x64', True)


def _check_shape(Nf, Nt):
    if Nf < 2 or Nf % 2:
        raise ValueError(f"Nf must be even and >= 2, got {Nf}")
    if Nt < 2 or Nt % 2:
        raise ValueError(f"Nt must be even and >= 2, got {Nt}")


def phitilde(omega, Nf, nx=4.0):
    """
    Fourier-domain Meyer window of the WDM basis (Cornish 2020, Eq. 11).

    Angular frequency ``omega`` is in units of the sampling rate (dt = 1).
    The window is flat for ``|omega| < A`` and rolls off smoothly to zero at
    ``|omega| = A + B``, with ``A = dOmega / 4`` and ``B = dOmega / 2``.
    It only depends on the grid, so it is computed once in NumPy and
    enters the compiled transforms as a constant.

    Args:
        omega (array): Angular frequencies (dt = 1 units).
        Nf (int): Number of frequency bins.
        nx (float): Steepness of the roll-off (order of the incomplete beta function).

    Returns:
        array: Window values.
    """
    omega = np.abs(np.asarray(omega, dtype=float))
    d_omega = 2 * np.pi / (2 * Nf)
    A = d_omega / 4
    B = d_omega / 2

    phi = np.zeros_like(omega)
    phi[omega < A] = 1.0
    taper = (omega >= A) & (omega < A + B)
    phi[taper] = np.cos(0.5 * np.pi * betainc(nx, nx, (omega[taper] - A) / B))
    return phi / np.sqrt(d_omega)


@lru_cache(maxsize=None)
def _phif(Nf, Nt, nx):
    """Window sampled on the rfft grid of a series with Nf*Nt samples."""
    omega = 2 * np.pi / (Nf * Nt) * np.arange(Nt // 2 + 1)
    return phitilde(omega, Nf, nx) * np.sqrt(np.pi)


def _norm(Nf):
    """Forward-transform normalisation that makes the WDM basis orthonormal."""
    return 2.0 / Nf


def wdm_grid(Nf, Nt, dt):
    """
    Time and frequency centres of the WDM pixels.

    Args:
        Nf (int): Number of frequency bins.
        Nt (int): Number of time bins.
        dt (float): Sampling cadence in seconds.

    Returns:
        tuple: (times, freqs) with shapes (Nt,) and (Nf,).
    """
    dT = Nf * dt
    dF = 1.0 / (2 * dT)
    return np.arange(Nt) * dT, np.arange(Nf) * dF


@partial(jax.jit, static_argnums=(1, 2, 3))
def _forward_freq(X, Nf, Nt, nx):
    N = Nf * Nt
    half = Nt // 2
    d = jnp.arange(-half + 1, half)
    window = jnp.asarray(_phif(Nf, Nt, nx))[jnp.abs(d)]

    # DX[m, half + d] = phi(d) X[m*half + d] for |d| < half
    m_all = jnp.arange(Nf + 1)
    jj = m_all[:, None] * half + d[None, :]
    DX = jnp.zeros((Nf + 1, Nt), dtype=jnp.complex128)
    DX = DX.at[:, 1:].set(jnp.where((jj >= 0) & (jj <= N // 2), window * X[jnp.clip(jj, 0, N // 2)], 0.0))
    DX = DX.at[0, half].multiply(0.5).at[Nf, half].multiply(0.5)
    DXt = jnp.fft.ifft(DX, axis=1)

    n = jnp.arange(Nt)[:, None]
    m = jnp.arange(1, Nf)[None, :]
    inner = DXt[1:Nf].T
    # Real part on (n + m) even; on (n + m) odd use -Im for odd m and +Im for even m
    sign = jnp.where(m % 2 == 1, -1.0, 1.0)
    w_inner = jnp.where((n + m) % 2 == 1, sign * inner.imag, inner.real)
    col0 = jnp.zeros(Nt).at[0::2].set(jnp.sqrt(2.0) * DXt[0, 0::2].real)
    col0 = col0.at[1::2].set(jnp.sqrt(2.0) * DXt[Nf, 0::2].real)
    return _norm(Nf) * jnp.concatenate([col0[:, None], w_inner], axis=1)


@partial(jax.jit, static_argnums=(1,))
def _inverse_freq(w, nx):
    Nt, Nf = w.shape
    N = Nf * Nt
    half = Nt // 2
    phif = jnp.asarray(_phif(Nf, Nt, nx))

    n = jnp.arange(Nt)
    pre = jnp.zeros((Nf + 1, Nt), dtype=jnp.complex128)
    pre = pre.at[0].set(w[(2 * n) % Nt, 0] / jnp.sqrt(2.0))
    pre = pre.at[Nf].set(w[(2 * n) % Nt + 1, 0] / jnp.sqrt(2.0))
    m = jnp.arange(1, Nf)[:, None]
    pre = pre.at[1:Nf].set(jnp.where((n[None, :] + m) % 2 == 1, -1j, 1.0) * w[:, 1:].T)
    F = jnp.fft.fft(pre, axis=1)

    res = jnp.zeros(N // 2 + 1, dtype=jnp.complex128)

    # Interior frequency bins: res[m*half + d] += F[m, (m*half + d) % Nt] phi(|d|)
    d = jnp.arange(-half + 1, half)
    idx = m * half + d[None, :]
    res = res.at[idx.ravel()].add((F[m, idx % Nt] * phif[jnp.abs(d)][None, :]).ravel())

    # DC and Nyquist bins
    d0 = jnp.arange(half)
    res = res.at[d0].add(F[0, (2 * d0) % Nt] * phif[d0])
    dN = jnp.arange(half + 1)
    res = res.at[N // 2 - dN].add(F[Nf, (-2 * dN) % Nt] * phif[dN])
    # With the orthonormal forward normalisation the adjoint needs no rescaling
    return res


def forward_freq(X, Nf, Nt, nx=4.0):
    """
    WDM transform of a frequency series.

    Args:
        X (array): ``rfft`` of a real series of length ``Nf * Nt``.
        Nf (int): Number of frequency bins (even).
        Nt (int): Number of time bins (even).
        nx (float): Window steepness.

    Returns:
        jax.Array: Wavelet coefficients, shape (Nt, Nf).
    """
    _check_shape(Nf, Nt)
    X = jnp.asarray(X, dtype=jnp.complex128)
    if X.shape[-1] != Nf * Nt // 2 + 1:
        raise ValueError(f"X must have length Nf*Nt/2+1 = {Nf * Nt // 2 + 1}, got {X.shape[-1]}")
    return _forward_freq(X, int(Nf), int(Nt), float(nx))


def inverse_freq(w, nx=4.0):
    """
    Inverse WDM transform to the frequency domain.

    Args:
        w (array): Wavelet coefficients, shape (Nt, Nf).
        nx (float): Window steepness (must match the forward transform).

    Returns:
        jax.Array: ``rfft`` of the reconstructed series, length ``Nf*Nt/2 + 1``.
    """
    w = jnp.asarray(w, dtype=jnp.float64)
    _check_shape(w.shape[1], w.shape[0])
    return _inverse_freq(w, float(nx))


def forward_time(x, Nf, nx=4.0):
    """
    WDM transform of a real time series.

    Args:
        x (array): Time series. Its length must be a multiple of ``2 * Nf``.
        Nf (int): Number of frequency bins.
        nx (float): Window steepness.

    Returns:
        jax.Array: Wavelet coefficients, shape (Nt, Nf) with ``Nt = len(x) // Nf``.
    """
    x = jnp.asarray(x, dtype=jnp.float64)
    if x.shape[0] % (2 * Nf):
        raise ValueError(f"len(x) = {x.shape[0]} must be a multiple of 2*Nf = {2 * Nf}")
    return forward_freq(jnp.fft.rfft(x), Nf, x.shape[0] // Nf, nx)


def inverse_time(w, nx=4.0):
    """
    Inverse WDM transform to the time domain.

    Args:
        w (array): Wavelet coefficients, shape (Nt, Nf).
        nx (float): Window steepness.

    Returns:
        jax.Array: Real time series of length ``Nt * Nf``.
    """
    Nt, Nf = np.shape(w)
    return jnp.fft.irfft(inverse_freq(w, nx), n=Nt * Nf)


def band_weights(Nf, nx=4.0, nsub=8):
    """
    Spectral window of a WDM pixel, for band-averaging a PSD.

    For stationary noise with one-sided PSD ``S(f)``, the variance of pixel
    ``m`` is ``sum_j W_j S(f_m + offsets_j * dF) / (2 dt)``, i.e. the PSD
    averaged over the squared window ``|phitilde|^2``. This matters when the
    PSD changes appreciably across one pixel (steep spectra, low frequencies).

    Args:
        Nf (int): Number of frequency bins.
        nx (float): Window steepness.
        nsub (int): Sub-samples per frequency bin.

    Returns:
        tuple: (offsets, weights) as NumPy arrays. Offsets are in units of ``dF``,
        weights sum to one.
    """
    # Window support is |f - f_m| < 3/4 dF
    offsets = np.arange(-nsub, nsub + 1) * (0.75 / nsub)
    omega = 2 * np.pi * offsets / (2 * Nf)
    weights = phitilde(omega, Nf, nx) ** 2
    weights[0] = weights[-1] = 0.0
    weights /= weights.sum()
    keep = weights > 0
    return offsets[keep], weights[keep]
