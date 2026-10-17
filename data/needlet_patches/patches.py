"""
Simulation and wavelet-domain likelihood of the needlet patch study (30 days, A and E independent).

The data are one continuous series per channel, as in data/wavelet_injection::

    x_c(t) = n_c(t) + sqrt(P_c(t)) g_c(t),   P_c(t) = sum_pix I(pix) R_c(t, pix) / Rbar      (sky.py)

with ``I`` the Galaxy + patches map and ``g`` stationary with the galactic DWD spectrum. Data settings are those
of run 2 (30.00 d, dt = 10 s, TDI gen 1, 0.1-29 mHz, Nf = 3600, 2 edge pixels dropped, 2-pixel time groups, 200
log-frequency bins, Gamma likelihood); only the modulation differs.

Because the modulation is linear in the sky, the Fisher information of any linear sky parametrisation follows
from the Fisher information of ``P_c`` at the likelihood's time groups (``fisher_P``).
"""
import os
import copy

import numpy as np
import yaml
import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp  # noqa: E402

from bahamas.wavelet.model import WaveletData, WaveletLikelihood, DAY  # noqa: E402
from bahamas.wavelet.simulate import simulate_channels, power_modulation, source_spectra, _positive  # noqa: E402
from bahamas.wavelet.pipeline import wdm_shape  # noqa: E402

import sky  # noqa: E402

HERE = sky.HERE
OUT = os.path.join(HERE, 'outputs')
SOURCES = os.path.join(HERE, 'config', 'sources.yaml')

CONFIG = {
    'T': 2592000.0, 'dt': 10.0, 't0': 0.0, 'gen2': False, 'f1': 1e-4, 'f2': 0.029, 'pad_days': 2.0,
    'wavelet': {'Nf': 3600, 'nx': 4.0, 'channels': [0, 1], 'seed': 2026,
                'freq_bins': 200, 'edge_bins': 2, 'time_bin': 2, 'band_samples': 8},
}


def load_sources():
    with open(SOURCES) as f:
        return yaml.safe_load(f)['sources']


def amp_offset(response, galaxy, t0=0.0, T=CONFIG['T']):
    """log10 factor that gives the smooth Galaxy the same 30-day mean A + E galactic power as run 2."""
    src = {p['name']: p['injected'] for p in load_sources()['galactic_DWD_time']}
    t = np.linspace(t0, t0 + T, 721)
    run2 = sum(float(np.mean(power_modulation(src, t, c))) for c in (0, 1))
    P = response.modulation(galaxy)
    inside = (response.times >= t0) & (response.times <= t0 + T)
    ours = float(P[:, inside].mean(axis=1).sum())
    return float(np.log10(run2 / ours))


def sources_with_amp(offset):
    src = copy.deepcopy(load_sources())
    for p in src['galactic_DWD_time']:
        if p['name'] == 'amp':
            p['injected'] += offset
    return src


class Truth:
    """Sky maps, the response on an hourly grid covering the padded series, and the injected modulation."""

    def __init__(self, cfg=None, config=CONFIG, maps=None):
        self.cfg = sky.load_config() if cfg is None else cfg
        self.config = config
        self.maps = sky.sky_maps(self.cfg) if maps is None else maps
        pad = config['pad_days'] * DAY
        t = np.arange(config['t0'] - pad - DAY, config['t0'] + config['T'] + pad + DAY, 3600.0)
        self.response = sky.SkyResponse(t, self.cfg['maps']['nside'])
        self.total = sum(self.maps.values())
        self.amp_offset = amp_offset(self.response, self.maps['galaxy'], config['t0'], config['T'])
        self.sources = sources_with_amp(self.amp_offset)

    def P(self, t, channel):
        return self.response.interpolator(self.total)(t, channel)

    def components(self, times):
        """Injected P of each map component at ``times``: ``{name: (2, nt)}``."""
        return {k: np.stack([np.interp(times, self.response.times, P[c]) for c in (0, 1)])
                for k, P in ((k, self.response.modulation(m)) for k, m in self.maps.items())}


def simulate(truth, config=CONFIG, seed=None, path=None):
    """Simulate the 30-day series with the sky modulation; returns WaveletData (saved to ``path`` if given)."""
    wcfg, dt = config['wavelet'], config['dt']
    Nt, N = wdm_shape(config['T'], dt, wcfg['Nf'])
    pad = int(round(config['pad_days'] * DAY / dt))
    x = simulate_channels(truth.sources, N + 2 * pad, dt, wcfg['channels'], config['gen2'],
                          t0=config['t0'] - pad * dt, seed=wcfg['seed'] if seed is None else seed,
                          modulation=truth.P)[:, pad:pad + N]
    times = config['t0'] + np.arange(Nt) * wcfg['Nf'] * dt
    P = np.stack([np.asarray(truth.P(times, c)) for c in wcfg['channels']])
    data = WaveletData.from_time_series(np.asarray(x), dt, wcfg['Nf'], wcfg['nx'], wcfg['channels'], config['t0'],
                                        {'times': times, 'P': P})
    if path:
        data.save(path)
    return data


def likelihood(data, config=CONFIG):
    wcfg = config['wavelet']
    return WaveletLikelihood(data, config['f1'], config['f2'], freq_bins=wcfg['freq_bins'],
                             edge_bins=wcfg['edge_bins'], nsub=wcfg['band_samples'], time_bin=wcfg['time_bin'])


def spectra(like, sources, config=CONFIG):
    """Band-averaged galactic and stationary PSDs of the likelihood's groups: (S_gal (ng,), S_noise (nc, ng))."""
    stationary, galaxy, _ = source_spectra(sources, config['gen2'])
    f = like.eval_freqs
    S_gal = like.band_average(_positive(galaxy)(f))
    S_noise = jnp.stack([like.band_average(_positive(lambda x: stationary(x, c))(f)) for c in like.data.channels])
    return np.asarray(S_gal), np.asarray(S_noise)


def fisher_P(like, S_gal, S_noise, P):
    """
    Fisher information of P_c at each time group (diagonal, groups and channels independent)::

        F_c[n] = sum_g nu_g / 2 * (S_gal_g / (P_c[n] S_gal_g + S_noise_cg))^2

    Args:
        P (array): Modulation at the likelihood times, shape (nc, ntimes).

    Returns:
        np.ndarray: Shape (nc, ntimes).
    """
    nu = np.asarray(like.nu)
    ratio = S_gal[None, None, :] / (P[:, :, None] * S_gal[None, None, :] + S_noise[:, None, :])
    return np.sum(0.5 * nu * ratio ** 2, axis=-1)


def load_or_simulate(tag, truth, config=CONFIG, rerun=False):
    path = os.path.join(OUT, tag, 'data.h5')
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if rerun or not os.path.exists(path):
        return simulate(truth, config, path=path)
    return WaveletData.load(path)
