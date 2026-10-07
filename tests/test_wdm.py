import importlib.util  # noqa: F401  (pywavelet 0.2.x uses importlib.util without importing it)

import numpy as np
import pytest

from bahamas.wavelet import wdm
from bahamas.wavelet.simulate import stationary_gaussian


@pytest.mark.parametrize("Nf, Nt", [(2, 4), (8, 10), (16, 32), (64, 64), (32, 128)])
def test_orthonormal_and_invertible(Nf, Nt):
    rng = np.random.default_rng(0)
    x = rng.normal(size=Nf * Nt)
    w = wdm.forward_time(x, Nf)
    assert w.shape == (Nt, Nf)
    # Parseval: an orthonormal basis preserves the energy
    assert np.isclose(np.sum(w ** 2), np.sum(x ** 2), rtol=1e-12)
    # Perfect reconstruction
    assert np.allclose(wdm.inverse_time(w), x, atol=1e-12)


def test_frequency_domain_inverse():
    rng = np.random.default_rng(1)
    x = rng.normal(size=32 * 64)
    X = np.fft.rfft(x)
    assert np.allclose(wdm.inverse_freq(wdm.forward_freq(X, 32, 64)), X, atol=1e-10)


@pytest.mark.parametrize("Nf, Nt", [(7, 10), (8, 9), (1, 4)])
def test_rejects_invalid_shapes(Nf, Nt):
    with pytest.raises(ValueError):
        wdm.forward_freq(np.zeros(Nf * Nt // 2 + 1, dtype=complex), Nf, Nt)


def test_matches_pywavelet():
    """Coefficients agree with pywavelet, whose convention is sqrt(2) times the orthonormal one."""
    pytest.importorskip("pywavelet")
    from pywavelet.types import TimeSeries
    from pywavelet.transforms.numpy import from_freq_to_wavelet, from_time_to_wavelet, from_wavelet_to_time

    Nf, Nt, dt = 32, 64, 5.0
    x = np.random.default_rng(2).normal(size=Nf * Nt)
    mine = wdm.forward_time(x, Nf)

    ts = TimeSeries(x, np.arange(len(x)) * dt)
    ref_freq = np.asarray(from_freq_to_wavelet(ts.to_frequencyseries(), Nf=Nf, Nt=Nt).data).T
    ref_time = np.asarray(from_time_to_wavelet(ts, Nf=Nf, Nt=Nt, mult=Nt // 2).data).T
    assert np.allclose(ref_freq, np.sqrt(2) * mine, atol=1e-10)
    assert np.allclose(ref_time, np.sqrt(2) * mine, atol=1e-10)

    # And our inverse undoes theirs
    wave = from_freq_to_wavelet(ts.to_frequencyseries(), Nf=Nf, Nt=Nt)
    assert np.allclose(wdm.inverse_time(np.asarray(wave.data).T / np.sqrt(2)), x, atol=1e-10)
    assert np.allclose(np.asarray(from_wavelet_to_time(wave, dt=dt, mult=Nt // 2).data), x, atol=1e-6)


def test_white_noise_variance():
    """White noise with variance s2 gives coefficients with variance s2 in every pixel."""
    rng = np.random.default_rng(3)
    Nf, Nt, s2 = 32, 2048, 4.0
    w = wdm.forward_time(rng.normal(scale=np.sqrt(s2), size=Nf * Nt), Nf)
    per_freq = np.mean(w[:, 1:] ** 2, axis=0) / s2
    tol = 5 * np.sqrt(2 / Nt)
    assert np.all(np.abs(per_freq - 1) < tol)
    assert abs(np.mean(w ** 2) / s2 - 1) < 5 * np.sqrt(2 / w.size)


def test_localisation():
    """A sine-Gaussian lands in the pixel at its central time and frequency."""
    Nf, Nt, dt = 64, 128, 1.0
    t = np.arange(Nf * Nt) * dt
    n0, m0 = 50, 20
    t0, f0 = n0 * Nf * dt, m0 / (2 * Nf * dt)
    x = np.exp(-0.5 * ((t - t0) / (3 * Nf * dt)) ** 2) * np.cos(2 * np.pi * f0 * t)
    w = wdm.forward_time(x, Nf)
    power = w[:, 1:] ** 2
    # Average out the oscillation between neighbouring time pixels
    profile_t = power.sum(axis=1)
    profile_f = power.sum(axis=0)
    assert abs(np.argmax(np.convolve(profile_t, np.ones(3), 'same')) - n0) <= 1
    assert np.argmax(profile_f) + 1 == m0
    assert profile_f[m0 - 1] / profile_f.sum() > 0.9


def test_band_average_matches_coloured_noise():
    """For a steep spectrum the pixel variance is the window-weighted PSD, not the PSD at the centre."""
    Nf, Nt, dt = 64, 16384, 1.0
    psd = lambda f: np.where(np.asarray(f) > 0, (np.maximum(f, 1e-6) / 0.05) ** -4, 0.0)
    x = stationary_gaussian(psd, Nf * Nt, dt, 4)
    w = np.asarray(wdm.forward_time(x, Nf))

    freqs = wdm.wdm_grid(Nf, Nt, dt)[1]
    dF = freqs[1]
    offsets, weights = wdm.band_weights(Nf)
    m = np.arange(3, 12)
    predicted = np.array([np.sum(weights * psd(freqs[k] + offsets * dF)) for k in m]) / (2 * dt)
    centre = psd(freqs[m]) / (2 * dt)
    measured = np.mean(w[:, m] ** 2, axis=0)

    # Each measured variance has relative error sqrt(2 / Nt)
    chi2 = lambda model: np.sum((measured / model - 1) ** 2) * Nt / 2
    assert chi2(predicted) < len(m) + 5 * np.sqrt(2 * len(m))
    # Evaluating the PSD at the pixel centre is a much worse fit
    assert chi2(centre) > 3 * chi2(predicted) + 20


def test_band_weights_normalised():
    offsets, weights = wdm.band_weights(64, nsub=8)
    assert np.isclose(weights.sum(), 1.0)
    assert np.allclose(weights, weights[::-1])
    assert np.all(np.abs(offsets) < 0.75)
