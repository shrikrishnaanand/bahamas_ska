"""Legendre-series modulation (``modulation: legendre``) of the wavelet pipeline."""
import numpy as np
import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp
from numpy.polynomial import legendre as npleg

from bahamas.wavelet.model import WaveletData, WaveletLikelihood, LegendreModulation, legendre_basis
from bahamas.wavelet.sampler import WaveletPosterior, run_nuts

SOURCES = {
    'galactic_DWD_time': [
        {'name': 'alpha', 'bounds': None, 'injected': 1.8},
        {'name': 'amp', 'bounds': None, 'injected': -42.5},
        {'name': 'fknee', 'bounds': None, 'injected': -2.595679532778269},
        {'name': 'fr1', 'bounds': None, 'injected': -2.856},
        {'name': 'fr2', 'bounds': None, 'injected': -3.487854923626168},
    ],
    'instr_noise': [
        {'name': 'A', 'bounds': [1, 5], 'injected': 2.4},
        {'name': 'P', 'bounds': None, 'injected': 7.9},
    ],
}


def test_basis_matches_numpy():
    x = np.linspace(-1, 1, 101)
    B = np.asarray(legendre_basis(x, 6))
    for n in range(1, 7):
        assert np.allclose(B[n - 1], npleg.legval(x, np.eye(7)[n]))


def test_log_modulation_maps_window_to_unit_interval():
    lm = LegendreModulation(t0=10.0, T=100.0, degree=2)
    t = np.array([10.0, 60.0, 110.0])
    assert np.allclose(lm.x(t), [-1, 0, 1])
    c = np.array([0.3, -0.2])
    assert np.allclose(lm.log_modulation(t, c, level=0.5), 0.5 + npleg.legval(lm.x(t), [0, *c]))


def test_nuts_recovers_legendre_coefficients():
    """Data drawn from the model variance with known coefficients; the posterior covers them."""
    dt, Nf, Nt = 1000.0, 64, 492
    blank = WaveletData(np.ones((1, Nt, Nf)), dt, channels=[0])
    like = WaveletLikelihood(blank, 1e-4, 4.9e-4, edge_bins=0)
    post = WaveletPosterior(like, SOURCES, modulation='legendre', gen2=True, legendre_degree=2,
                            fix_first_level=True)
    names = [p.name for p in post.psd_params if p.bounds is not None]
    assert names == ['A', 'c1_A', 'c2_A']
    truth = jnp.array([2.4, 0.6, -0.4])
    S_gal, S_n, values = post.spectra(truth)
    P = np.asarray(post.power_modulation(None, (), (), values, blank.times))
    var = np.ones((Nt, Nf))
    var[:, like.pixels] = (P[0][:, None] * np.asarray(S_gal)[None, :] + np.asarray(S_n)[0][None, :]) / (2 * dt)
    w = np.random.default_rng(3).normal(size=(Nt, Nf)) * np.sqrt(var)
    data = WaveletData(w[None], dt, channels=[0])
    post = WaveletPosterior(WaveletLikelihood(data, 1e-4, 4.9e-4, edge_bins=0), SOURCES, modulation='legendre',
                            gen2=True, legendre_degree=2, fix_first_level=True)

    res = run_nuts(post, warmup=300, samples=400, init='prior', seed=0, progress=False)
    chain = res['psd_chain']
    mean, std = chain.mean(axis=0), chain.std(axis=0)
    assert np.all(np.abs(mean - np.asarray(truth)) < 4 * std)
