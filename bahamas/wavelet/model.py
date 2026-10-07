"""
Wavelet-domain likelihood for a cyclostationary galactic foreground (JAX).

Model, for TDI channel ``c`` and WDM pixel ``(n, m)``::

    sigma^2_c[n, m] = ( P_c(t_n) <S_gal>_m + <S_n,c>_m ) / (2 dt)

``<.>_m`` is the PSD averaged over the spectral window of pixel ``m`` (see
``wdm.band_weights``), and ``P_c(t)`` is the power modulation, given by one of:

- ``stationary_modulation``: ``P = 1`` (no modulation, the null model).
- ``parametric_modulation``: the sky-envelope model of the chunked analysis
  (lat, long, s1, s2, psi), evaluated at the pixel times.
- ``WaveletModulation``: ``log P_c(t) = b_c + sum_k a_k psi((t - t_k) / tau_k)``
  with a variable number of wavelet atoms, sampled by reversible jump.

Coefficients are treated as independent Gaussians, the standard diagonal
approximation for locally stationary noise in the WDM basis (Cornish 2020).
Optionally, adjacent frequency pixels are grouped in log-spaced bins, and
consecutive time pixels in blocks of ``time_bin``. The group sums of ``w^2``
are then sufficient statistics with a Gamma likelihood,
the wavelet analogue of the coarse-grained Gamma likelihood of the chunked
pipeline. With one pixel per group this is exactly the Gaussian likelihood
up to a data-only constant.

Everything evaluated during sampling is a pure ``jax.numpy`` function, so it
can be jit-compiled, vectorised over walkers with ``vmap`` and differentiated
(for NUTS).
"""
import numpy as np
import jax
import jax.numpy as jnp
import h5py
from scipy.special import gammaln

from bahamas.wavelet import wdm
from bahamas.wavelet.simulate import power_modulation, CHANNEL_NAMES

DAY = 86400.0


class WaveletData:
    """
    Container for WDM coefficients of one or more TDI channels.

    The coefficients are stored as a NumPy array (they are data, read once);
    the likelihood moves what it needs to the JAX device.

    Attributes:
        w (np.ndarray): Coefficients, shape (nchannels, Nt, Nf).
        dt (float): Sampling cadence of the underlying time series.
        nx (float): WDM window steepness.
        channels (list): TDI channel indices.
        t0 (float): Start time of the series in seconds.
        injection (dict): Optional injected curves for plotting.
    """

    def __init__(self, w, dt, nx=4.0, channels=(0, 1), t0=0.0, injection=None):
        self.w = np.asarray(w, dtype=float)
        if self.w.ndim == 2:
            self.w = self.w[None]
        self.dt = float(dt)
        self.nx = float(nx)
        self.channels = [int(c) for c in channels]
        self.t0 = float(t0)
        self.injection = {k: np.asarray(v) for k, v in (injection or {}).items()}
        if len(self.channels) != self.w.shape[0]:
            raise ValueError("Number of channels does not match the data shape")

    @property
    def Nt(self):
        return self.w.shape[1]

    @property
    def Nf(self):
        return self.w.shape[2]

    @property
    def times(self):
        """Pixel-centre times in seconds (including ``t0``)."""
        return self.t0 + wdm.wdm_grid(self.Nf, self.Nt, self.dt)[0]

    @property
    def freqs(self):
        """Pixel-centre frequencies in Hz."""
        return wdm.wdm_grid(self.Nf, self.Nt, self.dt)[1]

    @property
    def channel_names(self):
        return [CHANNEL_NAMES.get(c, str(c)) for c in self.channels]

    @classmethod
    def from_time_series(cls, x, dt, Nf, nx=4.0, channels=(0, 1), t0=0.0, injection=None):
        """Transform time series of shape (nchannels, N) to the WDM domain."""
        x = jnp.atleast_2d(jnp.asarray(x))
        w = np.stack([np.asarray(wdm.forward_time(xi, Nf, nx)) for xi in x])
        return cls(w, dt, nx, channels, t0, injection)

    def save(self, path):
        """Save to an HDF5 file."""
        with h5py.File(path, 'w') as f:
            f.create_dataset('w', data=self.w)
            f.attrs['dt'] = self.dt
            f.attrs['nx'] = self.nx
            f.attrs['t0'] = self.t0
            f.attrs['channels'] = np.array(self.channels)
            group = f.create_group('injection')
            for key, value in self.injection.items():
                group.create_dataset(key, data=np.asarray(value))

    @classmethod
    def load(cls, path):
        """Load from an HDF5 file written by ``save``."""
        with h5py.File(path, 'r') as f:
            injection = {key: np.array(f['injection'][key]) for key in f.get('injection', {})}
            return cls(np.array(f['w']), f.attrs['dt'], f.attrs['nx'], list(f.attrs['channels']),
                       f.attrs['t0'], injection)


def frequency_groups(freqs, f1, f2, nbins=None):
    """
    Partition the WDM frequency pixels in ``[f1, f2]`` into contiguous groups.

    Args:
        freqs (array): Pixel-centre frequencies.
        f1, f2 (float): Frequency band. Pixel ``m = 0`` (DC/Nyquist) is always excluded.
        nbins (int, optional): Number of log-spaced bins. ``None`` keeps every pixel separate.

    Returns:
        tuple: ``(pixels, starts)`` as NumPy arrays. ``pixels`` holds the selected
        pixel indices in increasing order, and ``starts`` the start of each group
        within ``pixels``.
    """
    freqs = np.asarray(freqs)
    pixels = np.where((np.arange(len(freqs)) > 0) & (freqs >= f1) & (freqs <= f2))[0]
    if len(pixels) == 0:
        raise ValueError(f"No WDM pixels in [{f1}, {f2}] Hz")
    if nbins is None or nbins >= len(pixels):
        return pixels, np.arange(len(pixels))

    edges = np.logspace(np.log10(freqs[pixels[0]]), np.log10(freqs[pixels[-1]]), nbins + 1)
    labels = np.clip(np.searchsorted(edges, freqs[pixels], side='right') - 1, 0, nbins - 1)
    starts = np.concatenate([[0], np.where(np.diff(labels))[0] + 1])
    return pixels, starts


class WaveletLikelihood:
    """
    Gaussian (or binned Gamma) likelihood of WDM coefficients.

    The data-dependent pieces are precomputed once; ``band_average`` and
    ``log_likelihood`` are pure JAX functions of the model.

    Args:
        data (WaveletData): Wavelet coefficients.
        f1, f2 (float): Frequency band in Hz.
        freq_bins (int, optional): Number of log frequency bins (``None`` = per pixel).
        edge_bins (int): Time pixels dropped at each end, where the periodic
            transform wraps the end of the series onto its start.
        nsub (int): Sub-samples per pixel for band-averaging the PSD.
        time_bin (int): Number of consecutive time pixels summed into one group
            (the modulation is evaluated at the group's mean time). Use it when
            the modulation is slow compared with ``time_bin * Nf * dt``.
    """

    def __init__(self, data, f1, f2, freq_bins=None, edge_bins=4, nsub=8, time_bin=1):
        self.data = data
        self.dt = data.dt
        time_bin = int(time_bin)
        if 2 * edge_bins + time_bin > data.Nt:
            raise ValueError(f"edge_bins={edge_bins}, time_bin={time_bin} leave no time pixels (Nt={data.Nt})")

        time_index = np.arange(edge_bins, data.Nt - edge_bins)
        ntimes = len(time_index) // time_bin
        time_index = time_index[:ntimes * time_bin]
        self.time_index = time_index
        self.time_bin = time_bin
        self.times = jnp.asarray(data.times[time_index].reshape(ntimes, time_bin).mean(axis=1))

        freqs = data.freqs
        pixels, starts = frequency_groups(freqs, f1, f2, freq_bins)
        self.pixels = pixels
        self.npix = len(pixels)
        counts = np.diff(np.append(starts, len(pixels))).astype(float)
        self.ngroups = len(counts)
        self._counts = jnp.asarray(counts)
        # Degrees of freedom of each (time, frequency) group
        nu = counts * time_bin
        self.nu = jnp.asarray(nu)
        self.group_freqs = np.add.reduceat(freqs[pixels], starts) / counts
        self.group_ids = jnp.asarray(np.repeat(np.arange(self.ngroups), counts.astype(int)))

        # Frequencies at which PSDs are evaluated: nsub-point window around each pixel
        dF = freqs[1] - freqs[0]
        offsets, weights = wdm.band_weights(data.Nf, data.nx, nsub)
        self._weights = jnp.asarray(weights)
        self.eval_freqs = jnp.asarray((freqs[pixels][:, None] + offsets[None, :] * dF).ravel())

        # Sufficient statistics Y[c, n, g] = sum over the group of w^2
        w = data.w[:, time_index][:, :, pixels]
        Y = np.add.reduceat(w ** 2, starts, axis=2)
        Y = Y.reshape(Y.shape[0], ntimes, time_bin, -1).sum(axis=2)
        self.Y = jnp.asarray(Y)

        # Data-only part of the Gamma log-density, kept so the value is a proper likelihood
        nu_b = nu[None, None, :]
        self._const = float(np.sum((nu_b / 2 - 1) * np.log(Y) - (nu_b / 2) * np.log(2.0) - gammaln(nu_b / 2)))
        self.npixels = int(self.npix * len(time_index) * len(data.channels))

    def band_average(self, psd_values):
        """
        Average PSD values from ``eval_freqs`` over the pixel windows and the groups.

        Args:
            psd_values (array): PSD on ``eval_freqs``, shape (..., len(eval_freqs)).

        Returns:
            jax.Array: Group-averaged PSD, shape (..., ngroups).
        """
        psd_values = jnp.asarray(psd_values)
        lead = psd_values.shape[:-1]
        per_pixel = psd_values.reshape(lead + (self.npix, -1)) @ self._weights
        summed = jax.ops.segment_sum(jnp.moveaxis(per_pixel, -1, 0), self.group_ids,
                                     num_segments=self.ngroups, indices_are_sorted=True)
        return jnp.moveaxis(summed, 0, -1) / self._counts

    def log_likelihood(self, S_gal, S_noise, P):
        """
        Log-likelihood for band-averaged spectra and a power modulation.

        Args:
            S_gal (array): Galactic PSD per group, shape (ngroups,) or (nchannels, ngroups).
            S_noise (array): Stationary PSD per channel and group, shape (nchannels, ngroups).
            P (array): Power modulation, shape (nchannels, ntimes).

        Returns:
            jax.Array: Scalar log-likelihood (``-inf`` for non-positive variances).
        """
        S_gal = jnp.broadcast_to(S_gal, S_noise.shape)
        var = (P[:, :, None] * S_gal[:, None, :] + S_noise[:, None, :]) / (2 * self.dt)
        ok = jnp.all(var > 0) & jnp.all(jnp.isfinite(var))
        safe = jnp.where(ok, var, 1.0)
        value = self._const - jnp.sum(self.nu * jnp.log(safe) / 2 + self.Y / (2 * safe))
        return jnp.where(ok, value, -jnp.inf)


def stationary_modulation(times, channels):
    """No modulation: ``P_c(t) = 1``."""
    return jnp.ones((len(channels), len(times)))


def parametric_modulation(times, channels, params):
    """Sky-envelope modulation of the chunked analysis, evaluated at the pixel times."""
    return jnp.stack([power_modulation(params, times, c) for c in channels])


def _gaussian(x):
    return jnp.exp(-0.5 * x ** 2)


def _ricker(x):
    return (1 - x ** 2) * jnp.exp(-0.5 * x ** 2)


WAVELET_SHAPES = {'gaussian': _gaussian, 'ricker': _ricker}


class WaveletModulation:
    """
    Log power modulation as a sum of wavelet atoms per channel::

        log P_c(t) = b_c + sum_k a_k psi((t - t_k) / tau_k)

    Each atom has parameters ``(t_frac, log10_width_days, amp)``, with
    ``t_k = t0 + t_frac * T``. The atom shape ``psi`` is ``gaussian`` (default)
    or ``ricker`` (Mexican hat). The level ``b_c`` is zero for the first
    channel (it is degenerate with the galactic amplitude) and free for the others.

    Atoms are passed padded to a fixed maximum number with a boolean mask, so
    that array shapes stay fixed under ``jit`` and ``vmap``.

    Args:
        t0 (float): Start time.
        T (float): Duration covered by ``t_frac`` in [0, 1].
        shape (str): Atom shape.
    """

    def __init__(self, t0, T, shape='gaussian'):
        if shape not in WAVELET_SHAPES:
            raise ValueError(f"Unknown wavelet shape '{shape}', choose from {list(WAVELET_SHAPES)}")
        self.t0 = float(t0)
        self.T = float(T)
        self.shape = shape
        self._psi = WAVELET_SHAPES[shape]

    def log_modulation(self, times, atoms, mask=None, level=0.0):
        """
        ``log P(t)`` for one channel.

        Args:
            times (array): Times in seconds.
            atoms (array or None): Shape (K, 3) with columns (t_frac, log10_width_days, amp).
            mask (array, optional): Shape (K,), which atoms are active (default: all).
            level (float): Constant ``b_c``.

        Returns:
            jax.Array: ``log P`` at ``times``.
        """
        times = jnp.asarray(times)
        if atoms is None:
            return jnp.full(times.shape, level, dtype=jnp.float64)
        atoms = jnp.reshape(jnp.asarray(atoms, dtype=jnp.float64), (-1, 3))
        if atoms.shape[0] == 0:
            return jnp.full(times.shape, level, dtype=jnp.float64)
        mask = jnp.ones(atoms.shape[0], dtype=bool) if mask is None else jnp.asarray(mask)
        # Inactive slots may hold anything (eryn stores NaN there): replace them by a
        # zero-amplitude atom before any arithmetic, since 0 * NaN is still NaN
        atoms = jnp.where(mask[:, None], atoms, jnp.array([0.5, 1.0, 0.0]))
        centres = self.t0 + atoms[:, 0] * self.T
        widths = 10 ** atoms[:, 1] * DAY
        x = (times[None, :] - centres[:, None]) / widths[:, None]
        return level + jnp.sum(atoms[:, 2:3] * self._psi(x), axis=0)

    def __call__(self, times, atoms_per_channel, masks_per_channel, levels):
        return jnp.exp(jnp.stack([self.log_modulation(times, a, m, b)
                                  for a, m, b in zip(atoms_per_channel, masks_per_channel, levels)]))
