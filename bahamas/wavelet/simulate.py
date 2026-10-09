"""
Time-domain simulation of cyclostationary LISA data for the wavelet-domain analysis (JAX).

The chunked (STFT) pipeline in ``bahamas_data`` draws each chunk independently
in the frequency domain. A wavelet analysis instead needs one continuous time
series, so here the galactic foreground is simulated as a stationary Gaussian
process with PSD ``S_gal(f)`` multiplied by the square root of the
instantaneous power modulation ``P_c(t)`` of TDI channel ``c``::

    x_c(t) = n_c(t) + sqrt(P_c(t)) g_c(t)

For a modulation that is slow compared with the WDM time resolution, the
evolutionary PSD is ``S_c(f, t) = S_n(f) + P_c(t) S_gal(f)``. Averaged over a
chunk ``[t1, t2]`` this reproduces the chunk PSD of ``galactic_DWD_time``,
because ``average_envelopes_gaussian`` is the time average of ``P_c(t)``.
"""
import numpy as np
import jax
import jax.numpy as jnp

from bahamas.psd_response.modulation import envelopes_gaussian
from bahamas.logger_config import logger

YEAR = 31557600.0

# Sky/shape parameters of galactic_DWD_time that set the time modulation
ENVELOPE_PARAMS = ['lat', 'long', 's1', 's2', 'psi']

CHANNEL_NAMES = {0: 'A', 1: 'E', 2: 'T'}


def as_key(seed):
    """A ``jax.random`` key from an integer seed, a key, or ``None`` (fresh entropy)."""
    if seed is None:
        seed = int(np.random.SeedSequence().entropy % 2 ** 32)
    if isinstance(seed, (int, np.integer)):
        return jax.random.PRNGKey(int(seed))
    return seed


def stationary_gaussian(psd, N, dt, key):
    """
    Draw a real stationary Gaussian time series with a given one-sided PSD.

    Args:
        psd (callable or array): One-sided PSD, either a function of frequency
            or its values on ``rfftfreq(N, dt)``.
        N (int): Number of samples (even).
        dt (float): Sampling cadence in seconds.
        key: ``jax.random`` key or integer seed.

    Returns:
        jax.Array: Time series of length ``N``.
    """
    freqs = jnp.fft.rfftfreq(N, dt)
    S = jnp.asarray(psd(freqs) if callable(psd) else psd, dtype=jnp.float64)
    S = S.at[0].set(0.0)

    # E|X_k|^2 = N S(f_k) / (2 dt) so that the periodogram 2 dt |X|^2 / N is unbiased
    scale = jnp.sqrt(N * S / (4 * dt))
    k_re, k_im, k_nyq = jax.random.split(as_key(key), 3)
    X = scale * (jax.random.normal(k_re, scale.shape) + 1j * jax.random.normal(k_im, scale.shape))
    if N % 2 == 0:
        X = X.at[-1].set(jnp.sqrt(2.0) * scale[-1] * jax.random.normal(k_nyq))
    return jnp.fft.irfft(X, n=N)


def power_modulation(par, t, channel):
    """
    Instantaneous power modulation of the galactic foreground in one TDI channel.

    Args:
        par (dict): Must contain ``lat`` (sine of latitude), ``long``, ``s1``,
            ``s2`` and ``psi`` (sine of the rotation angle). Values may be traced.
        t (array): Times in seconds.
        channel (int): TDI channel (0 = A, 1 = E).

    Returns:
        jax.Array: ``P_c(t)``, whose time average over a chunk equals
        ``average_envelopes_gaussian`` for that chunk.
    """
    if channel not in (0, 1):
        raise ValueError(f"Galactic modulation is only defined for channels 0 (A) and 1 (E), got {channel}")
    A, E = envelopes_gaussian(par['lat'], par['long'], par['s1'], par['s2'], par['psi'],
                              1.0 / YEAR, jnp.asarray(t, dtype=jnp.float64))
    return jnp.asarray([A, E][channel]) ** 2


def injected_parameters(sources):
    """Convert a sources configuration into ``{source: {param: injected}}``."""
    return {name: {p['name']: p['injected'] for p in params} for name, params in sources.items()}


def source_spectra(sources, gen2):
    """
    Split the injected sources into stationary and modulated spectral components.

    Supported sources are ``instr_noise``, ``galactic_DWD`` (stationary) and
    ``galactic_DWD_time`` (modulated).

    Args:
        sources (dict): Sources configuration with ``injected`` values.
        gen2 (bool): Second-generation TDI.

    Returns:
        tuple: ``(stationary, galaxy, envelope)``. ``stationary(f, channel)``
        returns the summed stationary PSD; ``galaxy(f)`` returns the unmodulated
        galactic PSD or is ``None``; ``envelope`` holds the modulation
        parameters or is ``None``.
    """
    from bahamas.psd_strain import psd_galaxy as gal
    from bahamas.psd_strain import psd_noise as noise

    supported = {'instr_noise', 'galactic_DWD', 'galactic_DWD_time'}
    unsupported = set(sources) - supported
    if unsupported:
        raise NotImplementedError(
            f"Wavelet-domain simulation supports {sorted(supported)}; got unsupported sources {sorted(unsupported)}")

    inj = injected_parameters(sources)

    def stationary(f, channel):
        S = jnp.zeros_like(f)
        if 'instr_noise' in inj:
            S = S + noise.noise(f, inj['instr_noise'], tdi=channel, gen2=gen2)
        if 'galactic_DWD' in inj:
            S = S + gal.galactic_foreground(f, inj['galactic_DWD'], gen2=gen2)
        return S

    galaxy, envelope = None, None
    if 'galactic_DWD_time' in inj:
        par = inj['galactic_DWD_time']
        galaxy = lambda f: gal.galactic_foreground(f, par, gen2=gen2)
        envelope = {k: par[k] for k in ENVELOPE_PARAMS}

    return stationary, galaxy, envelope


def _positive(func):
    """Evaluate a PSD only at f > 0 (the models diverge at f = 0) and return 0 at DC."""
    return lambda f: jnp.where(f > 0, func(jnp.where(f > 0, f, 1.0)), 0.0)


def simulate_channels(sources, N, dt, channels, gen2, t0=0.0, seed=None, modulation=None):
    """
    Simulate continuous time series for several TDI channels.

    Args:
        sources (dict): Sources configuration with ``injected`` values.
        N (int): Number of samples (even).
        dt (float): Sampling cadence.
        channels (list): TDI channel indices.
        gen2 (bool): Second-generation TDI.
        t0 (float): Start time in seconds (sets the orbital phase of the modulation).
        seed (int or key, optional): Random seed.
        modulation (callable, optional): ``modulation(t, channel)`` returning ``P_c(t)``,
            used instead of the sky envelope of ``galactic_DWD_time`` (e.g. to inject
            a known wavelet modulation).

    Returns:
        jax.Array: Time series of shape (len(channels), N).
    """
    key = as_key(seed)
    stationary, galaxy, envelope = source_spectra(sources, gen2)
    t = t0 + jnp.arange(N) * dt

    # The modulation varies over months: evaluate it hourly and interpolate, since the
    # envelope formula holds dozens of full-length temporaries at the sample rate
    t_coarse = jnp.linspace(t[0], t[-1], max(1024, int((N - 1) * dt / 3600) + 2))

    def channel(k_noise, k_gal, c):
        x = stationary_gaussian(_positive(lambda f: stationary(f, c)), N, dt, k_noise)
        if galaxy is not None:
            g = stationary_gaussian(_positive(galaxy), N, dt, k_gal)
            P_coarse = power_modulation(envelope, t_coarse, c) if modulation is None else modulation(t_coarse, c)
            P = jnp.interp(t, t_coarse, P_coarse)
            x = x + jnp.sqrt(P) * g
        return x

    # Compiled per channel so that XLA fuses the full-length spectral evaluations
    channel = jax.jit(channel, static_argnums=2)
    out = []
    for c in channels:
        key, k_noise, k_gal = jax.random.split(key, 3)
        x = channel(k_noise, k_gal, c)
        logger.info(f"Simulated channel {CHANNEL_NAMES.get(c, c)}: std = {float(jnp.std(x)):.3e}")
        out.append(x)
    return jnp.stack(out)
