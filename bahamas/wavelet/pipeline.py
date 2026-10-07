"""
Config-driven wavelet-domain pipeline, selected with ``domain: 'wavelet'`` in the config.

- ``generate_data``: simulate continuous LISA data with a modulated galactic
  foreground, transform it to the WDM domain and save it (``bahamas_data``).
- ``run_inference``: sample the wavelet-domain posterior (``bahamas_inference``):
  with eryn, using reversible jump over the number of wavelet atoms when
  ``inference.modulation`` is ``'wavelets'``, or with NUTS for the
  fixed-dimension models (``'parametric'``, ``'none'``).

Everything that is evaluated repeatedly (transform, simulation, likelihood)
is written in JAX, like the chunked pipeline; the likelihood is jit-compiled
and vectorised over walkers.

See ``template/config_wavelet.yaml`` for all options.
"""
import os

import numpy as np

from bahamas.backend_context import get_backend_components
from bahamas.wavelet import plotting
from bahamas.wavelet.model import WaveletData, WaveletLikelihood
from bahamas.wavelet.sampler import WaveletPosterior, WaveletSampler, run_nuts, modulation_draws
from bahamas.wavelet.simulate import simulate_channels, power_modulation, injected_parameters
from bahamas.logger_config import logger


def is_wavelet(config):
    """True if the config selects the wavelet-domain pipeline."""
    return config.get('domain', 'stft') == 'wavelet'


def _require_jax():
    """The wavelet pipeline is written in JAX; the spectral models must use the JAX backend."""
    if get_backend_components()[2] is None:
        raise RuntimeError("The wavelet-domain pipeline needs the JAX backend (run with --use_jax True)")


def _name(path):
    return os.path.basename(path)


def _plot_path(config, kind, name):
    folder = config.get('folder_plot', os.path.dirname(config['wavelet']['file']) or '.')
    return os.path.join(folder, f'wavelet_{kind}_{name}.png')


def wdm_shape(T, dt, Nf):
    """Number of time bins (even) and samples for a duration ``T``."""
    Nt = int(round(T / dt)) // Nf
    Nt -= Nt % 2
    if Nt < 2:
        raise ValueError(f"T/dt = {T / dt:.0f} samples is too short for Nf = {Nf}")
    return Nt, Nf * Nt


def generate_data(config, sources):
    """
    Simulate data, transform it to the WDM domain and save ``<wavelet.file>.h5``.

    Args:
        config (dict): Configuration (``T``, ``dt``, ``gen2``, ``wavelet`` section).
        sources (dict): Sources with injected values.

    Returns:
        WaveletData: The saved data.
    """
    _require_jax()
    wcfg = config['wavelet']
    dt, T, gen2 = float(config['dt']), float(config['T']), bool(config.get('gen2', False))
    t0 = float(config.get('t0', 0.0))
    Nf = int(wcfg['Nf'])
    nx = float(wcfg.get('nx', 4.0))
    channels = list(wcfg.get('channels', [0, 1]))

    Nt, N = wdm_shape(T, dt, Nf)
    if N * dt < T:
        logger.info(f"Using {N * dt:.0f} s of data so that Nt = {Nt} is even (requested {T:.0f} s)")
    logger.info(f"WDM grid: Nf = {Nf} (dF = {1 / (2 * Nf * dt):.3e} Hz), Nt = {Nt} (dT = {Nf * dt / 3600:.2f} h)")

    x = simulate_channels(sources, N, dt, channels, gen2, t0=t0, seed=wcfg.get('seed'))

    times = t0 + np.arange(Nt) * Nf * dt
    injection = {'times': times}
    inj = injected_parameters(sources)
    if 'galactic_DWD_time' in inj:
        injection['P'] = np.stack([np.asarray(power_modulation(inj['galactic_DWD_time'], times, c))
                                   for c in channels])

    data = WaveletData.from_time_series(x, dt, Nf, nx, channels, t0, injection)
    data.save(wcfg['file'] + '.h5')
    logger.info(f"Wavelet data saved in {wcfg['file']}.h5")

    plotting.plot_spectrogram(data, config.get('f1', 1e-4), config.get('f2', 0.029),
                              _plot_path(config, 'data', _name(wcfg['file'])))
    return data


def _flatten(results):
    """Turn nested result dictionaries into ``np.savez``-friendly keys."""
    out = {}
    for key, value in results.items():
        if isinstance(value, dict):
            for k, v in value.items():
                out[f'{key}_{k}'] = np.asarray(v)
        else:
            out[key] = np.asarray(value)
    return out


def run_inference(config, sources):
    """
    Run the wavelet-domain inference and save results and plots.

    Args:
        config (dict): Configuration (``wavelet`` and ``inference`` sections).
        sources (dict): Sources with bounds (sampled) and injected values.

    Returns:
        dict: Sampler results.
    """
    _require_jax()
    wcfg, icfg = config['wavelet'], config['inference']
    data_file = icfg.get('file', wcfg['file'])
    data = WaveletData.load(data_file + '.h5')

    like = WaveletLikelihood(data, float(config.get('f1', 1e-4)), float(config.get('f2', 0.029)),
                             freq_bins=wcfg.get('freq_bins'), edge_bins=int(wcfg.get('edge_bins', 4)),
                             nsub=int(wcfg.get('band_samples', 8)), time_bin=int(wcfg.get('time_bin', 1)))
    logger.info(f"Wavelet likelihood: {like.npixels} pixels in {len(like.nu)} frequency groups "
                f"x {len(like.times)} time bins x {len(data.channels)} channels")

    modulation = icfg.get('modulation', 'wavelets')
    max_wavelets = int(icfg.get('max_wavelets', 8))
    posterior = WaveletPosterior(like, sources, modulation=modulation, gen2=bool(config.get('gen2', False)),
                                 wavelet_shape=icfg.get('wavelet_shape', 'gaussian'),
                                 level_range=tuple(icfg.get('level_range', (-2.0, 2.0))),
                                 max_wavelets=max_wavelets)

    sampler_name = icfg.get('sampler', 'eryn')
    if sampler_name == 'eryn':
        sampler = WaveletSampler(
            posterior,
            nwalkers=int(icfg.get('nwalkers', 32)),
            ntemps=int(icfg.get('ntemps', 1)),
            min_wavelets=int(icfg.get('min_wavelets', 0)),
            max_wavelets=max_wavelets,
            width_range_days=tuple(icfg.get('width_range_days', (5.0, 180.0))),
            amp_range=tuple(icfg.get('amp_range', (-3.0, 3.0))),
            atom_step=float(icfg.get('atom_step', 0.02)),
            init=icfg.get('init', 'injected'),
            seed=icfg.get('seed'),
        )
        results = sampler.run(int(icfg.get('nsteps', 1000)), burnin=int(icfg.get('burnin', 0)),
                              thin=int(icfg.get('thin', 1)), progress=icfg.get('progress', True))
    elif sampler_name == 'NUTS':
        results = run_nuts(posterior, warmup=int(icfg.get('warmup', 1000)), samples=int(icfg.get('samples', 2000)),
                           chains=int(icfg.get('chains', 1)), chain_method=icfg.get('chain_method', 'parallel'),
                           adapt_matrix=bool(icfg.get('adapt_matrix', True)), init=icfg.get('init', 'injected'),
                           seed=int(icfg.get('seed') or 0), progress=icfg.get('progress', True))
    else:
        raise ValueError(f"Unknown sampler '{sampler_name}' for the wavelet pipeline (use 'eryn' or 'NUTS')")
    results['modulation'] = modulation

    name = _name(data_file)
    if posterior.rj_branches:
        plotting.plot_bayes_factors(results, posterior.rj_branches, _plot_path(config, 'bayes_factor', name))
        plotting.plot_k_trace(results, posterior.rj_branches, _plot_path(config, 'k_trace', name))

    for b in ('psd', 'envelope'):
        if f'{b}_chain' in results and results[f'{b}_chain'].shape[1] > 1:
            truths = results[f'{b}_injected']
            plotting.plot_corner(results[f'{b}_chain'], results[f'{b}_names'], _plot_path(config, f'corner_{b}', name),
                                 truths=truths if np.all(np.isfinite(truths)) else None)

    if posterior.galaxy_source == 'galactic_DWD_time':
        times = data.times
        draws = modulation_draws(posterior, results, times, nsamples=int(icfg.get('modulation_draws', 500)),
                                 seed=icfg.get('seed'))
        results['modulation_times'] = times
        results['modulation_quantiles'] = np.percentile(draws, [5, 25, 50, 75, 95], axis=0)
        plotting.plot_modulation(times, draws, data.channel_names, _plot_path(config, 'modulation', name),
                                 truth=data.injection.get('P'))

    np.savez(icfg['file_post'], **_flatten(results))
    logger.info(f"Results saved to {icfg['file_post']}")
    return results
