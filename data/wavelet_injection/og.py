"""
The OG (chunked) pipeline on our own simulated time series.

The original pipeline (``bahamas.bahamas_data.SignalProcessor``) never builds a time series: it draws each 15-day chunk
as stationary Fourier coefficients with the chunk-averaged modulation. Here the chunks are cut from the continuous
series that the wavelet pipeline analyses (``outputs/<tag>/timeseries.npz``, written by ``injection.simulate``), so
both pipelines see the same noise and the same smooth modulation. Everything else is the original code path:

    x_c(t), 30 d ──cut──▶ chunk i = [i T_ch, (i+1) T_ch), T_ch = 15 d, N = T_ch / dt samples
          d_c,i(f_k) = sqrt(2 dt / N) rfft(w · chunk / rms(w))_k,   f_k = k / T_ch, k < N/2
                       (normalisation as SignalProcessor.simulate_data, so that E|d|^2 = one-sided PSD)
          ──average_log_chunks (nseg = 1000)──▶ binned |d|^2, counts                 (setting_data)
          ──SignalProcessor.save_data──▶ og/data.h5, og/data_bin.h5, og/time_interval.txt
          ──BayesianInference (Gamma likelihood, nessai)──▶ og/result.hdf5

Degrees of freedom: bahamas' Gamma likelihood (``setting_nessai.gamma_lik``) uses shape dof/2 with dof = ``count``,
the number of Fourier frequencies in the bin, i.e. one degree of freedom per |d_k|^2. A complex Fourier coefficient
has two (|d_k|^2 is exponential; checked on the data: count * var(binned/model) = 1.03-1.06, ``data_checks``-style
test in the run-2 report), so the original likelihood assumes twice the variance of the binned periodogram and its
posteriors are about sqrt(2) too wide. ``dof_factor = 2`` multiplies dof by 2 at run time (the library is unchanged)
and writes to ``og_dof2x/`` instead of ``og/``.

w is a Tukey window (``chunk: taper`` in og_config.yaml, the tapered fraction). The original chunks were simulated
directly in the frequency domain, i.e. periodic; a chunk cut from a real series is not, and without a taper the
rectangular window leaks the steep low-frequency spectrum into the band (+40-70% at 0.13 mHz, +8% at 0.5 mHz;
``data_checks.py``). A 5% taper removes this.

Settings: ``config/og_config.yaml`` (the file paths there are set here) and ``config/sources.yaml``.
Script: ``python og.py --tag run2 [--nlive 1000] [--data-only]``. As a script it uses bahamas' NumPy backend (as
``run_pe.py --use_jax False``): nessai's worker pool forks the process, and forked workers deadlock once JAX is
initialised. Imported after a JAX-backend module (e.g. in a notebook), it keeps that backend: fine for everything
except ``run`` with ``n_pool``.
"""
import os
import copy

if __name__ == '__main__':
    from bahamas.backend_context import initialize_backend
    initialize_backend(use_jax=False)

import numpy as np
import yaml

from scipy.signal.windows import tukey

from bahamas.bahamas_data import SignalProcessor
from bahamas.method import setting_data
from bahamas.psd_strain import psd_function as psd

import injection as I

OG_CONFIG = os.path.join(I.HERE, 'config', 'og_config.yaml')


def og_name(dof_factor=1):
    return 'og' if dof_factor == 1 else f'og_dof{dof_factor:g}x'


def og_dir(tag, dof_factor=1):
    path = os.path.join(I.OUT, tag, og_name(dof_factor))
    os.makedirs(path, exist_ok=True)
    return path


def og_config(tag, nlive=None, n_pool=None, dof_factor=1):
    """``config/og_config.yaml`` with the file paths of ``outputs/<tag>/og/`` (or ``og_dof<f>x/``)."""
    with open(OG_CONFIG) as f:
        cfg = yaml.safe_load(f)
    d = og_dir(tag, dof_factor)
    cfg['file'] = os.path.join(d, 'data')
    cfg['fileAV'] = os.path.join(d, 'data_bin')
    cfg['folder_plot'] = d + '/'
    cfg['inference']['file'] = cfg['fileAV']
    cfg['inference']['file_post'] = os.path.join(d, 'result.npz')
    if nlive:
        cfg['inference']['nlive'] = nlive
    if n_pool:
        cfg['inference']['n_pool'] = n_pool
    return cfg


def check_consistency(cfg, x, dt, t0):
    """The OG settings must describe the simulated series."""
    if cfg['dt'] != dt or cfg['gen2'] != I.CONFIG['gen2'] or t0 != cfg.get('t0', 0.0):
        raise ValueError('og_config.yaml (dt, gen2, t0) does not match the simulated series')
    if x.shape[1] * dt != cfg['T']:
        raise ValueError(f"series covers {x.shape[1] * dt:.0f} s, og_config.yaml has T = {cfg['T']}")


def make_data(tag, cfg=None):
    """Cut ``outputs/<tag>/timeseries.npz`` into chunks and write the OG data files. Returns the SignalProcessor."""
    cfg = og_config(tag) if cfg is None else cfg
    sp = chunk_spectra(*I.load_timeseries(tag), cfg)
    sp.save_data()
    np.savetxt(os.path.join(os.path.dirname(cfg['fileAV']), 'time_interval.txt'), np.column_stack([sp.T1, sp.T2]))
    return sp


def chunk_spectra(x, dt, t0, cfg):
    """Chunk FFTs and log-binned periodograms of the series x (channels, N), in the SignalProcessor's attributes."""
    check_consistency(cfg, x, dt, t0)
    sp = SignalProcessor(cfg, copy.deepcopy(I.og_sources()))
    sp.handle_series()

    N = int(sp.TOBS // dt)
    taper = cfg['chunk'].get('taper', 0.0)
    window = tukey(N, taper) if taper else np.ones(N)
    window /= np.sqrt(np.mean(window ** 2))   # keeps E|d|^2 = PSD
    factor = np.sqrt(2 * dt / N)
    sp.data, sp.response, sp.dataAV, sp.responseAV, sp.f_av, sp.count = [], [], [], [], [], []
    for i, freqs in enumerate(sp.freq_tot):
        start = int(round((sp.T1[i] - t0) / dt))
        response = sp.compute_response(freqs)
        d_tdi, dav_tdi, rav_tdi = [], [], []
        for c in range(sp.ntdi):
            d = factor * np.fft.rfft(window * x[c, start:start + N])[:len(freqs)]
            f, d_av, r_av, count = setting_data.average_log_chunks(freqs, np.abs(d) ** 2, response[c], cfg['nseg'])
            d_tdi.append(d)
            dav_tdi.append(d_av)
            rav_tdi.append(r_av)
        sp.data.append(d_tdi)
        sp.response.append(response)
        sp.dataAV.append(dav_tdi)
        sp.responseAV.append(rav_tdi)
        sp.f_av.append(f)
        sp.count.append(count)
    return sp


def binned_vs_model(sp, f1=None, f2=None):
    """
    Binned periodogram / injected model PSD (chunk-averaged modulation), per chunk and channel.

    Returns a list over chunks of dicts with ``f``, ``count`` and ``ratio`` (channels × bins), restricted to
    [f1, f2]. If the chunk FFTs follow the OG convention, E[ratio] = 1 with a standard deviation of 1/sqrt(count).
    """
    f1 = sp.config['f1'] if f1 is None else f1
    f2 = sp.config['f2'] if f2 is None else f2
    out = []
    for i, f in enumerate(sp.f_av):
        sel = (f > f1) & (f < f2)
        ratio = []
        for c in range(sp.ntdi):
            model, _ = psd.model_psd(f[sel], sources=sp.sources, response=sp.responseAV[i][c][sel], injected=True,
                                     t1=sp.T1[i], t2=sp.T2[i], tdi=c, gen2=sp.gen2)
            ratio.append(np.asarray(sp.dataAV[i][c])[sel] / np.asarray(model))
        out.append({'f': f[sel], 'count': np.asarray(sp.count[i])[sel], 'ratio': np.array(ratio)})
    return out


def run(tag, nlive=None, n_pool=None, dof_factor=1):
    """Write the OG data of run ``tag`` and run nessai. The result is ``outputs/<tag>/og[_dof<f>x]/result.hdf5``."""
    from bahamas.bahamas_inference import BayesianInference
    cfg = og_config(tag, nlive, n_pool, dof_factor)
    make_data(tag, cfg)
    d = og_dir(tag, dof_factor)
    path = os.path.join(d, 'og_config_resolved.yaml')
    with open(path, 'w') as f:
        yaml.safe_dump(cfg, f, sort_keys=False)
    inference = BayesianInference(path, I.SOURCES)
    inference.dof = [dof_factor * np.asarray(c) for c in inference.dof]
    inference.run()
    for f in ('nested_sampler_resume.pkl', 'nested_sampler_resume.pkl.old'):   # only needed to resume a run
        if os.path.exists(os.path.join(d, f)):
            os.remove(os.path.join(d, f))
    return os.path.join(d, 'result.hdf5')


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--tag', required=True)
    ap.add_argument('--nlive', type=int)
    ap.add_argument('--n-pool', type=int)
    ap.add_argument('--dof-factor', type=float, default=1, help='multiply the Gamma dof (2 = correct for |d|^2)')
    ap.add_argument('--data-only', action='store_true', help='only write the chunk data files')
    a = ap.parse_args()
    if a.data_only:
        make_data(a.tag, og_config(a.tag, dof_factor=a.dof_factor))
    else:
        print('result:', run(a.tag, a.nlive, a.n_pool, a.dof_factor))
