import numpy as np
import jax
import jax.numpy as jnp
import pytest
from scipy.integrate import trapezoid
from scipy.signal import welch

# Import the wavelet package first: it selects the JAX backend before psd_response is loaded
from bahamas.wavelet import wdm
from bahamas.psd_response.average_envelope import average_envelopes_gaussian
from bahamas.wavelet.model import WaveletData, WaveletLikelihood, WaveletModulation, frequency_groups, DAY
from bahamas.wavelet.sampler import WaveletPosterior, _injected, _pad_leaves, run_nuts
from bahamas.wavelet.simulate import YEAR, power_modulation, simulate_channels, stationary_gaussian

ENVELOPE = dict(lat=-0.09664565, long=-1.6259238, s1=0.04156049812924856, s2=0.13842963137295858,
                psi=-0.9900146151015816)

SOURCES = {
    'galactic_DWD_time': [
        {'name': 'alpha', 'bounds': [1., 2.], 'injected': 1.8},
        {'name': 'amp', 'bounds': [-47, -40], 'injected': -43.94309514866353},
        {'name': 'fknee', 'bounds': [-4, -1.8], 'injected': -2.595679532778269},
        {'name': 'fr1', 'bounds': [-4, -1.8], 'injected': -2.856},
        {'name': 'fr2', 'bounds': [-4, -1.8], 'injected': -3.487854923626168},
    ] + [{'name': k, 'bounds': None, 'injected': v} for k, v in ENVELOPE.items()],
    'instr_noise': [
        {'name': 'A', 'bounds': [1, 5], 'injected': 2.4},
        {'name': 'P', 'bounds': [3, 10], 'injected': 7.9},
    ],
}

# Laptop-sized year: dt = 1000 s, band where the galaxy dominates
DT, NF, NT, F1, F2 = 1000.0, 64, 492, 1e-4, 4.9e-4


def _sources_with(**bounds):
    """Copy of SOURCES with some parameters sampled ({name: bounds}) and the rest fixed."""
    out = {}
    for source, params in SOURCES.items():
        out[source] = [dict(p, bounds=bounds.get(p['name'])) for p in params]
    return out


@pytest.fixture(scope='module')
def year_data():
    x = simulate_channels(SOURCES, NF * NT, DT, [0, 1], gen2=True, seed=11)
    return WaveletData.from_time_series(x, DT, NF, channels=[0, 1])


@pytest.mark.parametrize("t1, t2", [(0, 1296000), (5e6, 5e6 + 1296000), (1e7, 2e7), (0, YEAR)])
@pytest.mark.parametrize("tdi", [0, 1])
def test_power_modulation_averages_to_chunk_envelope(t1, t2, tdi):
    """The chunk envelope of galactic_DWD_time is the time average of the instantaneous P(t)."""
    t = np.linspace(t1, t2, 20001)
    mean = trapezoid(np.asarray(power_modulation(ENVELOPE, t, tdi)), t) / (t2 - t1)
    chunk = average_envelopes_gaussian(ENVELOPE['lat'], ENVELOPE['long'], ENVELOPE['s1'], ENVELOPE['s2'],
                                       ENVELOPE['psi'], t1, t2, 1 / YEAR, tdi=tdi)
    assert np.isclose(mean, float(chunk), rtol=1e-7)


def test_stationary_gaussian_matches_psd():
    """Welch estimate of the simulated series recovers the input PSD."""
    N, dt = 2 ** 18, 2.0
    psd = lambda f: 1e-3 * (1 + (jnp.maximum(f, 1e-6) / 0.02) ** -2)
    x = np.asarray(stationary_gaussian(psd, N, dt, 5))
    f, P = welch(x, fs=1 / dt, nperseg=4096)
    band = (f > 0.01) & (f < 0.2)
    ratio = P[band] / np.asarray(psd(f[band]))
    assert abs(np.mean(ratio) - 1) < 0.02


def test_simulation_is_reproducible():
    a = simulate_channels(SOURCES, NF * 16, DT, [0], gen2=True, seed=3)
    b = simulate_channels(SOURCES, NF * 16, DT, [0], gen2=True, seed=3)
    c = simulate_channels(SOURCES, NF * 16, DT, [0], gen2=True, seed=4)
    assert np.array_equal(a, b) and not np.allclose(a, c, atol=0)


def test_modulated_noise_variance_tracks_modulation():
    """White noise times sqrt(P(t)) gives pixel variances proportional to P(t_n)."""
    rng = np.random.default_rng(6)
    Nf, Nt, dt = 32, 1024, 1.0
    t = np.arange(Nf * Nt) * dt
    P = 1.5 + np.sin(2 * np.pi * t / t[-1]) ** 2
    w = np.asarray(wdm.forward_time(np.sqrt(P) * rng.normal(size=t.size), Nf))
    times = wdm.wdm_grid(Nf, Nt, dt)[0]
    measured = np.mean(w[4:-4, 1:] ** 2, axis=1)
    expected = 1.5 + np.sin(2 * np.pi * times[4:-4] / t[-1]) ** 2
    resid = (measured - expected) / (expected * np.sqrt(2 / (Nf - 1)))
    assert abs(np.mean(resid)) < 5 / np.sqrt(len(resid))
    assert np.std(resid) < 1.2


def test_frequency_groups_partition():
    freqs = wdm.wdm_grid(64, 10, 1.0)[1]
    pixels, starts = frequency_groups(freqs, 0.02, 0.4, nbins=6)
    assert pixels[0] > 0 and np.all(np.diff(pixels) == 1)
    assert starts[0] == 0 and np.all(np.diff(starts) > 0)
    assert len(starts) <= 6
    pixels1, starts1 = frequency_groups(freqs, 0.02, 0.4, nbins=None)
    assert np.array_equal(pixels, pixels1) and np.array_equal(starts1, np.arange(len(pixels1)))


def test_band_average_matches_numpy(year_data):
    like = WaveletLikelihood(year_data, F1, F2, freq_bins=7)
    psd = np.random.default_rng(0).uniform(1, 2, size=(2, len(like.eval_freqs)))
    per_pixel = psd.reshape(2, like.npix, -1) @ np.asarray(like._weights)
    starts = np.concatenate([[0], np.cumsum(np.asarray(like.nu))[:-1]]).astype(int)
    expected = np.add.reduceat(per_pixel, starts, axis=1) / np.asarray(like.nu)
    assert np.allclose(like.band_average(psd), expected)


def test_whitened_coefficients(year_data):
    """At the injected parameters the data divided by the model variance is chi^2 with one dof."""
    like = WaveletLikelihood(year_data, F1, F2)
    post = WaveletPosterior(like, SOURCES, modulation='parametric', gen2=True)
    S_gal, S_n, values = post.spectra(jnp.asarray(_injected(post.psd_params)))
    P = post.power_modulation(jnp.zeros(0), (), (), values, like.times)
    z = np.asarray(like.Y / ((P[:, :, None] * S_gal + S_n[:, None, :]) / (2 * DT)))
    assert abs(z.mean() - 1) < 4 * np.sqrt(2 / z.size)
    assert abs(z.var() - 2) < 0.1


def test_binned_likelihood_matches_pixel_likelihood_differences(year_data):
    """One pixel per group: the Gamma likelihood differs from the Gaussian one by a data-only constant."""
    like = WaveletLikelihood(year_data, F1, F2, freq_bins=None)
    w = year_data.w[:, like.time_index][:, :, like.pixels]
    rng = np.random.default_rng(7)

    def gaussian(var):
        return -0.5 * np.sum(w ** 2 / var + np.log(2 * np.pi * var))

    diffs = []
    for _ in range(3):
        S_n = 10 ** rng.uniform(-42, -40, size=(2, like.ngroups))
        S_gal = 10 ** rng.uniform(-42, -40, size=like.ngroups)
        P = rng.uniform(0.5, 2.0, size=(2, len(like.times)))
        var = (P[:, :, None] * S_gal + S_n[:, None, :]) / (2 * DT)
        diffs.append(float(like.log_likelihood(S_gal, S_n, P)) - gaussian(var))
    assert np.allclose(diffs, diffs[0], rtol=0, atol=1e-6 * abs(diffs[0]))


def test_time_binning(year_data):
    """Summing consecutive time pixels keeps the sufficient statistics and adds degrees of freedom."""
    fine = WaveletLikelihood(year_data, F1, F2, freq_bins=10)
    coarse = WaveletLikelihood(year_data, F1, F2, freq_bins=10, time_bin=4)
    n = len(coarse.times)
    assert np.allclose(coarse.Y, np.asarray(fine.Y)[:, :4 * n].reshape(2, n, 4, -1).sum(axis=2))
    assert np.allclose(coarse.nu, 4 * np.asarray(fine.nu))
    assert np.allclose(coarse.times, np.asarray(fine.times)[:4 * n].reshape(n, 4).mean(axis=1))
    # Band averages do not depend on the time binning
    psd = jnp.ones(len(fine.eval_freqs))
    assert np.allclose(coarse.band_average(psd), 1.0)

    # The binned likelihood still peaks at the injected amplitude
    post = WaveletPosterior(coarse, SOURCES, modulation='parametric', gen2=True)
    names = [p.name for p in post.psd_params if p.bounds is not None]
    x0 = _injected(post.psd_params)
    grid = x0[names.index('amp')] + np.linspace(-0.1, 0.1, 41)
    thetas = np.tile(x0, (len(grid), 1))
    thetas[:, names.index('amp')] = grid
    profile = post.batch(jnp.asarray(thetas), jnp.zeros((len(grid), 0)), (), ())
    assert abs(grid[int(jnp.argmax(profile))] - x0[names.index('amp')]) < 0.03


def test_invalid_variance_gives_minus_infinity(year_data):
    like = WaveletLikelihood(year_data, F1, F2, freq_bins=10)
    S = jnp.ones((2, like.ngroups))
    assert float(like.log_likelihood(S[0], -S, jnp.zeros((2, len(like.times))))) == -np.inf


def test_likelihood_prefers_injected_modulation(year_data):
    """The injected envelope beats a stationary galaxy and a wrong sky position."""
    like = WaveletLikelihood(year_data, F1, F2, freq_bins=30)
    sources = _sources_with(alpha=[1, 2], amp=[-47, -40], fknee=[-4, -1.8], fr1=[-4, -1.8], fr2=[-4, -1.8],
                            A=[1, 5], P=[3, 10], long=[-3.14, 3.14])
    post = WaveletPosterior(like, sources, modulation='parametric', gen2=True)
    stationary = WaveletPosterior(like, sources, modulation='none', gen2=True)
    theta = jnp.asarray(_injected(post.psd_params))
    true_long = ENVELOPE['long']
    L_true = post.single(theta, jnp.array([true_long]), (), ())
    assert L_true > stationary.single(theta, jnp.zeros(0), (), ()) + 100
    assert L_true > post.single(theta, jnp.array([true_long + 1.0]), (), ()) + 100


def test_jit_vmap_and_grad_agree(year_data):
    """The compiled batch matches single evaluations, and the gradient matches finite differences."""
    like = WaveletLikelihood(year_data, F1, F2, freq_bins=30)
    post = WaveletPosterior(like, SOURCES, modulation='parametric', gen2=True)
    theta0 = jnp.asarray(_injected(post.psd_params))
    thetas = theta0 + 1e-3 * jax.random.normal(jax.random.PRNGKey(0), (4, len(theta0)))
    env = jnp.zeros((4, 0))
    batch = post.batch(thetas, env, (), ())
    single = jnp.array([post.single(t, jnp.zeros(0), (), ()) for t in thetas])
    assert np.allclose(batch, single, rtol=0, atol=1e-6)

    f = lambda t: post.single(t, jnp.zeros(0), (), ())
    grad = jax.grad(f)(theta0)
    assert np.all(np.isfinite(grad))
    i = [p.name for p in post.psd_params if p.bounds is not None].index('amp')
    h = 1e-5
    e = jnp.zeros_like(theta0).at[i].set(h)
    fd = (f(theta0 + e) - f(theta0 - e)) / (2 * h)
    assert np.isclose(grad[i], fd, rtol=1e-4)


def test_amplitude_profile_peaks_at_injection(year_data):
    like = WaveletLikelihood(year_data, F1, F2, freq_bins=30)
    post = WaveletPosterior(like, SOURCES, modulation='parametric', gen2=True)
    names = [p.name for p in post.psd_params if p.bounds is not None]
    x0 = _injected(post.psd_params)
    grid = x0[names.index('amp')] + np.linspace(-0.1, 0.1, 41)
    thetas = np.tile(x0, (len(grid), 1))
    thetas[:, names.index('amp')] = grid
    profile = post.batch(jnp.asarray(thetas), jnp.zeros((len(grid), 0)), (), ())
    assert abs(grid[int(jnp.argmax(profile))] - x0[names.index('amp')]) < 0.03


def test_eryn_vectorised_interface(year_data):
    """Eryn's flat leaves + group indices give the same values as evaluating each walker on its own."""
    like = WaveletLikelihood(year_data, F1, F2, freq_bins=20)
    post = WaveletPosterior(like, SOURCES, modulation='wavelets', gen2=True, max_wavelets=3)
    post.batch_size = 8
    rng = np.random.default_rng(1)
    x0 = _injected(post.psd_params)
    X_psd = x0 + 1e-3 * rng.normal(size=(3, len(x0)))
    # Walker 0: two atoms in A, none in E; walker 1: none in A, one in E; walker 2: one in each
    X_A = np.array([[0.2, 1.3, 0.5], [0.6, 1.6, -0.4], [0.8, 1.1, 0.7]])
    g_A = np.array([0, 0, 2])
    X_E = np.array([[0.4, 1.5, 0.3], [0.1, 1.2, -0.2]])
    g_E = np.array([1, 2])
    ll = post([X_psd, X_A, X_E], [np.arange(3), g_A, g_E])

    for walker in range(3):
        aA, mA = _pad_leaves(X_A[g_A == walker], np.zeros((g_A == walker).sum(), int), 1, 3)
        aE, mE = _pad_leaves(X_E[g_E == walker], np.zeros((g_E == walker).sum(), int), 1, 3)
        expected = post.single(jnp.asarray(X_psd[walker]), jnp.zeros(0), (aA[0], aE[0]), (mA[0], mE[0]))
        assert np.isclose(ll[walker], expected, rtol=0, atol=1e-6)


def test_pad_leaves():
    X = np.arange(12.0).reshape(4, 3)
    atoms, mask = _pad_leaves(X, np.array([2, 0, 2, 2]), 4, 3)
    assert mask.sum(axis=1).tolist() == [1, 0, 3, 0]
    assert np.array_equal(atoms[0, 0], X[1])
    assert np.array_equal(atoms[2], X[[0, 2, 3]])


def test_nuts_recovers_amplitude(year_data):
    """A short NUTS run on the parametric wavelet model brackets the injected values."""
    like = WaveletLikelihood(year_data, F1, F2, freq_bins=20)
    post = WaveletPosterior(like, _sources_with(amp=[-45, -43], A=[1, 5]), modulation='parametric', gen2=True)
    res = run_nuts(post, warmup=150, samples=200, seed=1, progress=False)
    for i, name in enumerate(res['psd_names']):
        chain = res['psd_chain'][:, i]
        lo, hi = np.percentile(chain, [0.5, 99.5])
        assert lo < res['psd_injected'][i] < hi, name
    assert np.all(np.isfinite(res['log_like']))


def test_wavelet_modulation_atoms():
    times = np.linspace(0, YEAR, 1000)
    mod = WaveletModulation(0.0, YEAR, 'gaussian')
    assert np.allclose(mod.log_modulation(times, None, level=0.3), 0.3)
    atoms = np.array([[0.5, np.log10(20.0), 1.2], [0.2, 1.0, 5.0]])
    logP = np.asarray(mod.log_modulation(times, atoms, mask=np.array([True, False])))
    assert np.isclose(logP.max(), 1.2, atol=1e-3)
    assert abs(times[np.argmax(logP)] - 0.5 * YEAR) < 2 * (times[1] - times[0])
    # Half maximum at one width times sqrt(2 ln 2)
    half = np.abs(times - 0.5 * YEAR) < 20.0 * DAY * np.sqrt(2 * np.log(2))
    assert np.all(logP[half] >= 0.6 - 1e-2)

    # Inactive slots are ignored even if they hold NaN (as eryn stores them)
    padded = np.vstack([atoms[:1], np.full((3, 3), np.nan)])
    logP_padded = np.asarray(mod.log_modulation(times, padded, mask=np.array([True, False, False, False])))
    assert np.allclose(logP_padded, np.asarray(mod.log_modulation(times, atoms[:1])))

    ricker = WaveletModulation(0.0, YEAR, 'ricker')
    # Zero-mean wavelet: integrates to ~0 when well inside the window
    assert abs(trapezoid(np.asarray(ricker.log_modulation(times, atoms[:1])), times)) < 1e-3 * YEAR

    with pytest.raises(ValueError):
        WaveletModulation(0.0, YEAR, 'haar')


def test_save_and_load_roundtrip(tmp_path, year_data):
    path = tmp_path / 'w.h5'
    data = WaveletData(year_data.w, DT, 4.0, [0, 1], 10.0, {'P': np.ones((2, NT))})
    data.save(path)
    loaded = WaveletData.load(path)
    assert np.array_equal(loaded.w, data.w)
    assert loaded.dt == DT and loaded.channels == [0, 1] and loaded.t0 == 10.0
    assert np.array_equal(loaded.injection['P'], data.injection['P'])
    assert np.allclose(loaded.times, 10.0 + np.arange(NT) * NF * DT)
