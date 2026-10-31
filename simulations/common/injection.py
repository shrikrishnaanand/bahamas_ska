"""
Stage 3: injection and recovery with the wavelet-domain RJMCMC pipeline on the cyclo pack-2 setup (30 d).

Two injections, same noise realisation (same seed), same data settings as the pack-2 run of paper 2:

- ``run1`` (``injection='wavelet'``): the best-fit wavelet modulation of the true GF envelope
  (``outputs/reference/atoms.npz``, made by ``fit_reference.py``), in the inference parametrisation
  (b_A = 0 absorbed into amp)::

      log P_A(t) = sum_k a_k psi((t - t_k) / tau_k),   log P_E(t) = level_E + sum_k a_k psi(...),   amp = amp_eff

  The wavelet model can represent this exactly: a pure recovery test.
- ``run2`` (``injection='sky'``): the true GF envelope P_c(t) = M_c^2(t) of the pack-2 sky parameters with the
  true amplitude (``config/sources.yaml``). Used for the OG comparison.
- ``run5`` (``injection='patches'``): run 2's Galaxy as a sky map plus four bright patches (``sky_patches.py``,
  ``config/run5_sky.yaml``), true amplitude. Same seed as run 2, so the same noise realisation.

The data are one continuous time series x_c(t) = n_c(t) + sqrt(P_c(t)) g_c(t), transformed to the WDM domain.
Both runs are analysed with the same wavelet model and priors. Used by the notebooks in this folder; also a script::

    python injection.py --tag run1 --injection wavelet
    python injection.py --tag run2 --injection sky
    python injection.py --tag run5 --injection patches
    python injection.py --tag run3 --injection sky --gap 12 13      (run 2's series with a 1-day gap)

A gap (``config['gap']``: ``days`` = [start, end], ``buffer_pixels``) zeroes the series in [start, end) and removes
from the likelihood every time pixel that overlaps it plus ``buffer_pixels`` on each side, whose WDM coefficients the
gap edges bias (``gap_check.py``: ~10% one pixel away, <~2% two pixels away).
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
from eryn.state import State

HERE = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), '30_days')  # default study: simulations/30_days
ROOT = os.path.dirname(os.path.dirname(HERE))
OUT = os.path.join(HERE, 'outputs')
REFERENCE = os.path.join(OUT, 'reference', 'atoms.npz')
SOURCES = os.path.join(HERE, 'config', 'sources.yaml')

WAVELET_CONFIG = os.path.join(HERE, 'config', 'wavelet.yaml')


def load_config(path=WAVELET_CONFIG):
    """
    Wavelet-pipeline settings from a YAML file (``config/wavelet.yaml``: data, WDM grid, priors, sampler).
    Relative ``out`` and ``og_config`` paths are taken relative to the study folder (the parent of the file's
    ``config/`` folder).
    """
    with open(path) as f:
        config = yaml.safe_load(f)
    study = os.path.dirname(os.path.dirname(os.path.abspath(path)))
    for key in ('out', 'og_config'):
        if key in config and not os.path.isabs(config[key]):
            config[key] = os.path.join(study, config[key])
    return config


CONFIG = load_config()


def out_dir(config=CONFIG):
    """Outputs folder of ``config``: ``config['out']`` if set (e.g. the 1-year study), else ``outputs/`` here."""
    return config.get('out', OUT)


def reference_path(config=CONFIG):
    return os.path.join(out_dir(config), 'reference', 'atoms.npz')


def with_gap(gap_days, buffer_pixels=1, config=CONFIG):
    """``config`` with a data gap of ``gap_days`` = (start, end) in days."""
    cfg = copy.deepcopy(config)
    cfg['gap'] = {'days': [float(gap_days[0]), float(gap_days[1])], 'buffer_pixels': int(buffer_pixels)}
    return cfg


def apply_gap(x, config=CONFIG):
    """Zero the series x (channels, N) inside the gap of ``config`` (if any)."""
    if 'gap' not in config:
        return x
    g0, g1 = (d * DAY for d in config['gap']['days'])
    t = config['t0'] + np.arange(x.shape[1]) * config['dt']
    x = np.array(x, copy=True)
    x[:, (t >= g0) & (t < g1)] = 0.0
    return x


def gap_time_mask(data, config=CONFIG):
    """``WaveletLikelihood`` time mask: False for the pixels overlapping the gap and ``buffer_pixels`` on each side."""
    if 'gap' not in config:
        return None
    g0, g1 = (d * DAY for d in config['gap']['days'])
    dT = data.Nf * data.dt
    t = data.times
    hit = np.flatnonzero((t + dT / 2 > g0) & (t - dT / 2 < g1))
    b = config['gap']['buffer_pixels']
    keep = np.ones(data.Nt, bool)
    keep[max(hit[0] - b, 0):min(hit[-1] + b + 1, data.Nt)] = False
    return keep


def load_reference(path=REFERENCE):
    """Inference-parametrisation values of the wavelet fit to the true envelope (``fit_reference.py``)."""
    r = np.load(path)
    return {'atoms': {'A': r['atoms_A'], 'E': r['atoms_E']},
            'amp': float(r['amp_eff']), 'level_E': float(r['level_E']),
            't0': float(r['t0']), 'T': float(r['T'])}


def og_sources():
    """Injected values and priors (``config/sources.yaml``): true amplitude and sky parameters."""
    with open(SOURCES) as f:
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

    The wavelet model fixes b_A = 0, so the amplitude it measures is amp_eff = amp + b_A / ln 10 (fit_reference.py).
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

    ``kind='wavelet'``: reference-fit atoms (posterior ``t_frac`` convention), amp = amp_eff.
    ``kind='sky'``: M_c^2(t) of the pack-2 sky parameters, amp = true amp.
    ``kind='patches'``: run 5's sky (smooth Galaxy + patches, ``sky_patches.Run5Sky``), amp = true amp.
    ``P_rel(t, c)`` is the injected galactic power relative to ``amp_ref`` (the posterior reference, amp_eff),
    the quantity returned by ``modulation_draws``.
    """

    def __init__(self, kind, ref, t0, T, shape='gaussian'):
        if kind not in ('wavelet', 'sky', 'patches'):
            raise ValueError(f"Unknown injection '{kind}'")
        self.kind = kind
        self.wm = WaveletModulation(t0, T, shape)
        self.amp_ref = ref['amp']
        if kind == 'wavelet':
            self.atoms = {c: remap_atoms(a, ref['t0'], ref['T'], t0, T) for c, a in ref['atoms'].items()}
            self.levels = {'A': 0.0, 'E': ref['level_E']}
            self.amp = ref['amp']
        elif kind == 'sky':
            self.sky = sky_params()
            self.amp = true_amp()
        else:
            import sky_patches
            self.sky5 = sky_patches.Run5Sky(t0=t0, T=T, pad_days=CONFIG['pad_days'] + 0.5)
            self.amp = true_amp()

    def P(self, t, channel):
        if self.kind == 'sky':
            return power_modulation(self.sky, t, channel)
        if self.kind == 'patches':
            return self.sky5.P(t, channel)
        name = 'AE'[channel]
        return jnp.exp(self.wm.log_modulation(t, self.atoms[name], level=self.levels[name]))

    def P_rel(self, t, channel):
        return 10 ** (self.amp - self.amp_ref) * np.asarray(self.P(t, channel))


def injection_for(kind, data, config=CONFIG, ref=None):
    ref = load_reference(reference_path(config)) if ref is None else ref
    return Injection(kind, ref, data.t0, data.Nt * data.Nf * data.dt, config['inference']['wavelet_shape'])


def simulate_series(kind, config=CONFIG, seed=None):
    """The series x (channels, N) with the chosen injection, and the Injection. ``seed`` overrides the config."""
    ref = load_reference(reference_path(config))
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
    modulation = None if kind == 'sky' else inj.P
    pad = int(round(config['pad_days'] * DAY / dt))
    x = simulate_channels(src, N + 2 * pad, dt, wcfg['channels'], config['gen2'], t0=config['t0'] - pad * dt,
                          seed=wcfg['seed'] if seed is None else seed, modulation=modulation)[:, pad:pad + N]
    return apply_gap(np.asarray(x), config), inj


def to_wavelet(x, inj, config=CONFIG):
    """WDM transform of the series, with the injected P at the pixel times."""
    wcfg = config['wavelet']
    Nt = x.shape[1] // wcfg['Nf']
    times = config['t0'] + np.arange(Nt) * wcfg['Nf'] * config['dt']
    P = np.stack([np.asarray(inj.P(times, c)) for c in wcfg['channels']])
    return WaveletData.from_time_series(x, config['dt'], wcfg['Nf'], wcfg['nx'], wcfg['channels'], config['t0'],
                                        {'times': times, 'P': P})


def simulate(kind, config=CONFIG, path=None):
    """
    Simulate the series with the chosen injection and save the WDM coefficients (``path``) and the time
    series itself (``timeseries.npz`` next to it; the OG pipeline reads it).
    """
    wcfg, dt = config['wavelet'], config['dt']
    x, inj = simulate_series(kind, config)
    data = to_wavelet(x, inj, config)
    if path:
        data.save(path)
        np.savez(timeseries_path(os.path.dirname(path)), x=np.asarray(x), dt=dt, t0=config['t0'],
                 channels=np.asarray(wcfg['channels']))
    return data, inj


def timeseries_path(run_path):
    return os.path.join(run_path, 'timeseries.npz')


def load_timeseries(tag, config=CONFIG):
    """The simulated series of run ``tag``: (x of shape (channels, N), dt, t0)."""
    r = np.load(timeseries_path(os.path.join(out_dir(config), tag)))
    return r['x'], float(r['dt']), float(r['t0'])


def build_posterior(data, config=CONFIG):
    ref = load_reference(reference_path(config))
    wcfg, icfg = config['wavelet'], config['inference']
    like = WaveletLikelihood(data, config['f1'], config['f2'], freq_bins=wcfg['freq_bins'],
                             edge_bins=wcfg['edge_bins'], nsub=wcfg['band_samples'], time_bin=wcfg['time_bin'],
                             time_mask=gap_time_mask(data, config))
    post = WaveletPosterior(like, inference_sources(ref), modulation='wavelets', gen2=config['gen2'],
                            wavelet_shape=icfg['wavelet_shape'], level_range=tuple(icfg['level_range']),
                            max_wavelets=icfg['max_wavelets'])
    # level_E is appended by WaveletPosterior with injected value 0: use the reference-fit value
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


def run_dir(tag, config=CONFIG):
    path = os.path.join(out_dir(config), tag)
    os.makedirs(path, exist_ok=True)
    return path


def load_or_simulate(kind, tag, config=CONFIG, rerun=False):
    path = os.path.join(run_dir(tag, config), 'data.h5')
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
    np.savez(os.path.join(run_dir(tag, config), 'chains.npz'), **flat)
    return res, post, data, inj


def k_trace_path(tag, config=CONFIG):
    return os.path.join(out_dir(config), tag, 'k_trace.npz')


def run_k_trace(kind, tag, config=CONFIG, progress=True):
    """
    Rerun the sampler of ``run`` storing the burn-in too, and save K of every walker and temperature at every step
    (``outputs/<tag>/k_trace.npz``). ``chains.npz`` is left untouched.

    ``run`` passes ``burn`` to eryn, which runs the burn-in with ``store=False`` and keeps no record of it. Here the
    burn-in and the production steps are two stored ``run_mcmc`` calls instead. Storing draws no random numbers, so
    with the same seeds this is the same chain, and the saved ``matches_chains`` flag checks that against ``chains.npz``.
    """
    data, inj = load_or_simulate(kind, tag, config)
    like, post = build_posterior(data, config)
    sampler = make_sampler(post, config)
    icfg = config['inference']
    burnin, nsteps = icfg['burnin'], icfg['nsteps']
    t = time.time()
    coords, inds = sampler.initial_state()
    es = sampler.sampler
    log_prior = es.compute_log_prior(coords, inds=inds)
    log_like = es.compute_log_like(coords, inds=inds, logp=log_prior)[0]
    state = State(coords, inds=inds, log_like=log_like, log_prior=log_prior)
    state = es.run_mcmc(state, burnin, progress=progress)
    es.run_mcmc(state, nsteps, progress=progress)

    out = {'burnin': burnin, 'nsteps': nsteps, 'nwalkers': sampler.nwalkers, 'ntemps': sampler.ntemps,
           'betas': es.get_betas(), 'log_like': es.get_log_like(), 'runtime_s': time.time() - t,
           'K_true': np.array([len(inj.atoms[c]) for c in post.channel_names]) if kind == 'wavelet' else np.array([])}
    all_inds = es.get_inds()
    for name in post.channel_names:
        out[f'K_{name}'] = all_inds[f'mod_{name}'].sum(axis=-1)       # (burnin + nsteps, ntemps, nwalkers)
    chains = os.path.join(out_dir(config), tag, 'chains.npz')
    if os.path.exists(chains):
        ref = np.load(chains)
        out['matches_chains'] = all(np.array_equal(out[f'K_{c}'][burnin:, 0].ravel(), ref[f'mod_{c}_nleaves'])
                                    for c in post.channel_names)
    np.savez(k_trace_path(tag, config), **out)
    return out


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--config', default=WAVELET_CONFIG,
                    help='wavelet settings (default: config/wavelet.yaml here; e.g. ../1_year/config/wavelet.yaml)')
    ap.add_argument('--tag', required=True)
    ap.add_argument('--injection', choices=['wavelet', 'sky', 'patches'], required=True)
    ap.add_argument('--nsteps', type=int)
    ap.add_argument('--burnin', type=int)
    ap.add_argument('--ntemps', type=int)
    ap.add_argument('--rerun-data', action='store_true', help='simulate again even if outputs/<tag>/data.h5 exists')
    ap.add_argument('--gap', type=float, nargs=2, metavar=('START', 'END'), help='data gap in days, e.g. 12 13')
    ap.add_argument('--gap-buffer', type=int, default=1, help='time pixels removed on each side of the gap')
    ap.add_argument('--k-trace', action='store_true',
                    help='rerun storing the burn-in and save K of every walker per step (outputs/<tag>/k_trace.npz)')
    a = ap.parse_args()
    base = load_config(a.config)
    cfg = with_gap(a.gap, a.gap_buffer, config=base) if a.gap else base
    if a.k_trace:
        out = run_k_trace(a.injection, a.tag, config=cfg)
        print(f"done in {out['runtime_s'] / 60:.1f} min; reproduces chains.npz: {out.get('matches_chains')}")
        raise SystemExit
    res, *_ = run(a.injection, a.tag, config=cfg, nsteps=a.nsteps, burnin=a.burnin, ntemps=a.ntemps,
                  rerun_data=a.rerun_data)
    print(f"done in {res['runtime_s'] / 60:.1f} min; median log L = {np.median(res['log_like']):.2f}"
          + (f", log L(true) = {res['loglike_true']:.2f}" if 'loglike_true' in res else ''))
