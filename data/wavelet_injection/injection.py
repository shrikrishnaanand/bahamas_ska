"""
Stage 3: injection and recovery with the wavelet-domain RJMCMC pipeline on the cyclo pack-2 setup (30 d).

Two injections, same noise realisation (same seed), same data settings as Riccardo's pack-2 run:

- ``run1`` (``injection='wavelet'``): the stage-2 best-fit wavelet modulation of the true GF envelope
  (``wavelet_fit_reference.npz``), written in the inference parametrisation (b_A = 0 absorbed into amp)::

      log P_A(t) = sum_k a_k psi((t - t_k) / tau_k),   log P_E(t) = level_E + sum_k a_k psi(...),   amp = amp_eff

  The wavelet model can represent this exactly: a pure recovery test.
- ``run2`` (``injection='sky'``): the true GF envelope P_c(t) = M_c^2(t) of the pack-2 sky parameters with the
  true amplitude, i.e. the signal Riccardo's chunked (OG) analysis was run on. Used for the OG comparison.

The data are one continuous time series x_c(t) = n_c(t) + sqrt(P_c(t)) g_c(t), transformed to the WDM domain.
Both runs are analysed with the same wavelet model and priors. Used by the notebooks in this folder; also a script::

    python injection.py --tag run1 --injection wavelet
    python injection.py --tag run2 --injection sky
"""
import os
import time
import copy

import numpy as np
import yaml
import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

from bahamas.wavelet.model import WaveletData, WaveletLikelihood, WaveletModulation, DAY
from bahamas.wavelet.sampler import WaveletPosterior, WaveletSampler, modulation_draws, _sampled
from bahamas.wavelet.simulate import simulate_channels, power_modulation, ENVELOPE_PARAMS
from bahamas.wavelet.pipeline import wdm_shape

HERE = os.path.dirname(os.path.abspath(__file__))
ROOT = os.path.dirname(os.path.dirname(HERE))
REFERENCE = os.path.join(ROOT, 'data', 'cyclo_fig7_outputs', 'wavelet_fit_reference.npz')
SOURCES_OG = os.path.join(ROOT, 'cyclo_riccardo', 'cyclo', 'pe_pack2_cyclo.yaml')
OG_RESULT = os.path.join(ROOT, 'cyclo_riccardo', 'cyclo', 'result.hdf5')
OUT = os.path.join(HERE, 'outputs')

# Same data settings as Riccardo's pack-2 analysis (T = 30 d, dt = 10 s, 0.1-29 mHz, TDI gen 1)
CONFIG = {
    'T': 2592000.0, 'dt': 10.0, 't0': 0.0, 'gen2': False, 'f1': 1e-4, 'f2': 0.029,
    'wavelet': {'Nf': 4096, 'nx': 4.0, 'channels': [0, 1], 'seed': 2026,
                'freq_bins': 200, 'edge_bins': 4, 'time_bin': 2, 'band_samples': 8},
    'inference': {'wavelet_shape': 'gaussian', 'min_wavelets': 0, 'max_wavelets': 8,
                  'width_range_days': [5.0, 60.0], 'amp_range': [-3.0, 3.0], 'level_range': [-3.0, 3.0],
                  'atom_step': 0.02, 'nwalkers': 32, 'ntemps': 4, 'nsteps': 6000, 'burnin': 2000,
                  'seed': 1},
}


def load_reference(path=REFERENCE):
    """Inference-parametrisation values of the stage-2 fit over the 30-day window."""
    r = np.load(path)
    return {'atoms': {'A': r['pack2_A_atoms'], 'E': r['pack2_E_atoms']},
            'amp': float(r['pack2_amp_eff']), 'level_E': float(r['pack2_level_E']),
            't0': float(r['pack2_t0']), 'T': float(r['pack2_T'])}


def og_sources():
    """Sources of Riccardo's pack-2 run (true amplitude and sky parameters)."""
    with open(SOURCES_OG) as f:
        return yaml.safe_load(f)['sources']


def true_amp():
    return next(p['injected'] for p in og_sources()['galactic_DWD_time'] if p['name'] == 'amp')


def sky_params():
    return {p['name']: p['injected'] for p in og_sources()['galactic_DWD_time'] if p['name'] in ENVELOPE_PARAMS}


def remap_atoms(atoms, t0_fit, T_fit, t0, T):
    """Express atoms fitted on [t0_fit, t0_fit + T_fit] in the ``t_frac`` convention of [t0, t0 + T]."""
    atoms = np.array(atoms, dtype=float)
    atoms[:, 0] = (t0_fit + atoms[:, 0] * T_fit - t0) / T
    return atoms


def inference_sources(ref):
    """Sampled sources: OG spectral parameters without the sky parameters; amp referenced to ``amp_eff``.

    The wavelet model fixes b_A = 0, so the amplitude it measures is amp_eff = amp + b_A / ln 10 (stage 2).
    Using amp_eff as the injected value only sets the initial ball and the reference of ``galactic_power``.
    """
    src = copy.deepcopy(og_sources())
    src['galactic_DWD_time'] = [p for p in src['galactic_DWD_time'] if p['name'] not in ENVELOPE_PARAMS]
    for p in src['galactic_DWD_time']:
        if p['name'] == 'amp':
            p['injected'] = ref['amp']
    return src


class Injection:
    """
    Injected modulation ``P(t, channel)`` and its galactic amplitude.

    ``kind='wavelet'``: stage-2 atoms (posterior ``t_frac`` convention), amp = amp_eff.
    ``kind='sky'``: M_c^2(t) of the pack-2 sky parameters, amp = true amp.
    ``P_rel(t, c)`` is the injected galactic power relative to ``amp_ref`` (the posterior reference, amp_eff),
    the quantity returned by ``modulation_draws``.
    """

    def __init__(self, kind, ref, t0, T, shape='gaussian'):
        if kind not in ('wavelet', 'sky'):
            raise ValueError(f"Unknown injection '{kind}'")
        self.kind = kind
        self.wm = WaveletModulation(t0, T, shape)
        self.amp_ref = ref['amp']
        if kind == 'wavelet':
            self.atoms = {c: remap_atoms(a, ref['t0'], ref['T'], t0, T) for c, a in ref['atoms'].items()}
            self.levels = {'A': 0.0, 'E': ref['level_E']}
            self.amp = ref['amp']
        else:
            self.sky = sky_params()
            self.amp = true_amp()

    def P(self, t, channel):
        if self.kind == 'sky':
            return power_modulation(self.sky, t, channel)
        name = 'AE'[channel]
        return jnp.exp(self.wm.log_modulation(t, self.atoms[name], level=self.levels[name]))

    def P_rel(self, t, channel):
        return 10 ** (self.amp - self.amp_ref) * np.asarray(self.P(t, channel))


def injection_for(kind, data, config=CONFIG, ref=None):
    ref = load_reference() if ref is None else ref
    return Injection(kind, ref, data.t0, data.Nt * data.Nf * data.dt, config['inference']['wavelet_shape'])


def simulate(kind, config=CONFIG, path=None):
    """Simulate the 30-day series with the chosen injection and save the WDM coefficients."""
    ref = load_reference()
    wcfg = config['wavelet']
    dt, Nf = config['dt'], wcfg['Nf']
    Nt, N = wdm_shape(config['T'], dt, Nf)
    # WaveletPosterior spans [times[0], times[-1] + dT] = [t0, t0 + N dt]
    inj = Injection(kind, ref, config['t0'], N * dt, config['inference']['wavelet_shape'])

    src = copy.deepcopy(og_sources())
    for p in src['galactic_DWD_time']:
        if p['name'] == 'amp':
            p['injected'] = inj.amp
    # kind='sky': the simulator builds M_c^2 from the sky parameters itself (modulation=None)
    modulation = inj.P if kind == 'wavelet' else None
    x = simulate_channels(src, N, dt, wcfg['channels'], config['gen2'], t0=config['t0'],
                          seed=wcfg['seed'], modulation=modulation)

    times = config['t0'] + np.arange(Nt) * Nf * dt
    P = np.stack([np.asarray(inj.P(times, c)) for c in wcfg['channels']])
    data = WaveletData.from_time_series(x, dt, Nf, wcfg['nx'], wcfg['channels'], config['t0'],
                                        {'times': times, 'P': P})
    if path:
        data.save(path)
    return data, inj


def build_posterior(data, config=CONFIG):
    ref = load_reference()
    wcfg, icfg = config['wavelet'], config['inference']
    like = WaveletLikelihood(data, config['f1'], config['f2'], freq_bins=wcfg['freq_bins'],
                             edge_bins=wcfg['edge_bins'], nsub=wcfg['band_samples'], time_bin=wcfg['time_bin'])
    post = WaveletPosterior(like, inference_sources(ref), modulation='wavelets', gen2=config['gen2'],
                            wavelet_shape=icfg['wavelet_shape'], level_range=tuple(icfg['level_range']),
                            max_wavelets=icfg['max_wavelets'])
    # level_E is appended by WaveletPosterior with injected value 0: use the stage-2 value
    # (exact for run 1, the best wavelet approximation for run 2)
    for p in post.psd_params:
        if p.name == 'level_E':
            p.value = ref['level_E']
    return like, post


def make_sampler(post, config=CONFIG, ntemps=None, nwalkers=None):
    icfg = config['inference']
    return WaveletSampler(post, nwalkers=nwalkers or icfg['nwalkers'], ntemps=ntemps or icfg['ntemps'],
                          min_wavelets=icfg['min_wavelets'], max_wavelets=icfg['max_wavelets'],
                          width_range_days=tuple(icfg['width_range_days']), amp_range=tuple(icfg['amp_range']),
                          atom_step=icfg['atom_step'], init='injected', seed=icfg['seed'])


def true_loglike(post, inj):
    """Log-likelihood at the injected point (wavelet injection only)."""
    theta = jnp.asarray(np.array([p.value for p in _sampled(post.psd_params)], dtype=float))[None]
    atoms, masks = [], []
    for name in post.channel_names:
        a = np.zeros((1, post.max_wavelets, 3))
        m = np.zeros((1, post.max_wavelets), bool)
        k = len(inj.atoms[name])
        a[0, :k], m[0, :k] = inj.atoms[name], True
        atoms.append(a)
        masks.append(m)
    pad = post.batch_size or 1
    rep = lambda x: jnp.repeat(jnp.asarray(x), pad, axis=0)
    ll = post.batch(rep(theta), jnp.zeros((pad, 0)), tuple(map(rep, atoms)), tuple(map(rep, masks)))
    return float(ll[0])


def run_dir(tag):
    path = os.path.join(OUT, tag)
    os.makedirs(path, exist_ok=True)
    return path


def load_or_simulate(kind, tag, config=CONFIG, rerun=False):
    path = os.path.join(run_dir(tag), 'data.h5')
    if rerun or not os.path.exists(path):
        return simulate(kind, config, path=path)
    data = WaveletData.load(path)
    return data, injection_for(kind, data, config)


def run(kind, tag, config=CONFIG, nsteps=None, burnin=None, ntemps=None, progress=True, rerun_data=False):
    """Simulate (or reuse) the data, run eryn from K = 0 and save ``outputs/<tag>/chains.npz``."""
    data, inj = load_or_simulate(kind, tag, config, rerun=rerun_data)
    like, post = build_posterior(data, config)
    sampler = make_sampler(post, config, ntemps=ntemps)
    icfg = config['inference']
    t = time.time()
    res = sampler.run(nsteps or icfg['nsteps'], burnin=icfg['burnin'] if burnin is None else burnin, progress=progress)
    res['runtime_s'] = time.time() - t
    res['injection'] = kind
    if kind == 'wavelet':
        res['loglike_true'] = true_loglike(post, inj)
        for name in post.channel_names:
            res[f'atoms_true_{name}'] = inj.atoms[name]

    # galactic power relative to amp_ref (= amp_eff), at the pixel times
    res['modulation_times'] = data.times
    res['modulation_draws'] = modulation_draws(post, res, data.times, nsamples=1000, seed=0)
    res['P_true'] = np.stack([inj.P_rel(data.times, c) for c in data.channels])
    res['amp_ref'] = inj.amp_ref
    res['analysed_times'] = np.asarray(like.times)
    flat = {}
    for k, v in res.items():
        if isinstance(v, dict):
            for kk, vv in v.items():
                flat[f'{k}_{kk}'] = np.asarray(vv)
        else:
            flat[k] = np.asarray(v)
    np.savez(os.path.join(run_dir(tag), 'chains.npz'), **flat)
    return res, post, data, inj


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--injection', choices=['wavelet', 'sky'], required=True)
    ap.add_argument('--nsteps', type=int)
    ap.add_argument('--burnin', type=int)
    ap.add_argument('--ntemps', type=int)
    a = ap.parse_args()
    res, *_ = run(a.injection, a.tag, nsteps=a.nsteps, burnin=a.burnin, ntemps=a.ntemps)
    print(f"done in {res['runtime_s'] / 60:.1f} min; median log L = {np.median(res['log_like']):.2f}"
          + (f", log L(true) = {res['loglike_true']:.2f}" if 'loglike_true' in res else ''))
