"""Regression tests for the code bugs B1-B6 of reports/PAPER_BUGS.md."""
import numpy as np
import pytest

# Import the wavelet package first: it selects the JAX backend before psd_response is loaded
from bahamas.wavelet import wdm  # noqa: F401
import jax.numpy as jnp
from bahamas.method import gaps, setting_data, setting_eryn, setting_hmc, setting_nessai
from bahamas.psd_response.average_envelope import average_envelopes_gaussian
from bahamas.psd_response.modulation import envelopes_gaussian
from bahamas.psd_strain import psd_noise

YEAR = 31557600.0
DAY = 86400.0
NB = 30
P_HAT = 1e-40


@pytest.fixture
def constant_psd(monkeypatch):
    """Replace the PSD model by a constant S and the priors by an empty sample, to test the likelihood formulas."""
    def set_S(S):
        for mod in (setting_hmc, setting_nessai, setting_eryn):
            monkeypatch.setattr(mod.psd, 'model_psd', lambda **kw: S * jnp.ones(NB))
    monkeypatch.setattr(setting_eryn, '_eryn_coords_to_sources', lambda *a, **k: {})
    monkeypatch.setattr(setting_hmc, 'prior_sample', lambda *a, **k: {})
    return set_S


def _log_like(kind, fn, data, dof):
    from numpyro import handlers
    args = ([data], [np.ones(NB)], [[np.ones(NB)]])
    if kind == 'hmc':
        with handlers.trace() as tr:
            fn(*args, {}, 1., [0], [0], [dof], False)
        return float(tr['log_likelihood']['fn'].log_factor)
    if kind == 'nessai':
        return float(fn({}, *args, 1., [0], [0], [dof], False))
    return float(fn({}, *args, {}, 1., [0], [0], [dof], False))


GAMMA = [('hmc', setting_hmc.gamma_lik), ('nessai', setting_nessai.gamma_lik), ('eryn', setting_eryn.gamma_lik)]
WHITTLE = [('hmc', setting_hmc.whittle_lik), ('nessai', setting_nessai.whittle_lik), ('eryn', setting_eryn.whittle_lik)]


@pytest.mark.parametrize('kind, fn', GAMMA)
def test_gamma_peaks_at_periodogram(constant_psd, kind, fn):
    """B1: every Gamma likelihood (ERYN included) peaks at S = P_hat; B2: curvature -n_b per bin."""
    data, dof = np.full((1, NB), P_HAT), 2 * NB * np.ones(NB)
    grid = P_HAT * np.exp(np.linspace(-0.05, 0.05, 11))
    ll = []
    for S in grid:
        constant_psd(S)
        ll.append(_log_like(kind, fn, data, dof))
    ll = np.array(ll)
    assert np.argmax(ll) == 5
    curvature = 2 * np.polyfit(np.log(grid / P_HAT), ll, 2)[0]
    assert curvature == pytest.approx(-NB * NB, rel=1e-3)


@pytest.mark.parametrize('kind, fn', WHITTLE)
def test_whittle_is_complex_gaussian(constant_psd, kind, fn):
    """B2: with 2 dof per amplitude the Whittle likelihood is the circular complex Gaussian density; with 1 dof
    (legacy_dof) it is Pozzoli et al. Eq. (3) as printed."""
    rng = np.random.default_rng(0)
    d = (rng.normal(size=NB) + 1j * rng.normal(size=NB)) * np.sqrt(P_HAT / 2)
    constant_psd(P_HAT)
    exact = np.sum(-np.log(np.pi * P_HAT) - np.abs(d) ** 2 / P_HAT)
    printed = np.sum(-0.5 * (np.abs(d) ** 2 / P_HAT + np.log(P_HAT) + np.log(2 * np.pi)))
    assert _log_like(kind, fn, d[None, :], 2 * np.ones(NB)) == pytest.approx(exact, rel=1e-12)
    assert _log_like(kind, fn, d[None, :], np.ones(NB)) == pytest.approx(printed, rel=1e-12)


def test_noise_transfer_functions_nam():
    """B3b: TDI-2 transfer functions equal Nam et al. (2023) Table II (uncorrelated noises)."""
    L = 8.3
    f = np.logspace(-4, np.log10(0.035), 50)
    w = 2 * np.pi * f * L
    cxx = 16 * np.sin(w) ** 2 * np.sin(2 * w) ** 2
    nam = {('acc', 0): 4 * cxx * (3 + 2 * np.cos(w) + np.cos(2 * w)), ('oms', 0): 2 * cxx * (2 + np.cos(w)),
           ('acc', 2): 32 * cxx * np.sin(w / 2) ** 4, ('oms', 2): 4 * cxx * (1 - np.cos(w))}
    for (noise, tdi), expected in nam.items():
        tf = getattr(psd_noise, f'transfer_functions_{noise}')(jnp.array(f), L, tdi=tdi, gen2=True)
        np.testing.assert_allclose(np.array(tf), expected, rtol=1e-10)


def test_noise_paper_eqs():
    """B3: with paper_eqs the noise is Pozzoli et al. Eqs. (16)-(17) times the transfer functions."""
    L, A, P = 8.3, 2.4, 7.9
    f = np.logspace(-4, np.log10(0.029), 20)
    c = psd_noise.clight
    s_tm = (A * 1e-15) ** 2 / (2 * np.pi * c * f) ** 2 * (1 + (4e-4 / f) ** 2) * (1 + (f / 8e-3) ** 4)
    s_oms = (P * 1e-12) ** 2 * (2 * np.pi * f / c) ** 2 * (1 + (2e-3 / f) ** 4)
    for tdi in (0, 1, 2):
        expected = (s_tm * np.array(psd_noise.transfer_functions_acc(jnp.array(f), L, tdi=tdi, gen2=True))
                    + s_oms * np.array(psd_noise.transfer_functions_oms(jnp.array(f), L, tdi=tdi, gen2=True)))
        got = psd_noise.noise(jnp.array(f), {'A': A, 'P': P, 'paper_eqs': 1}, tdi=tdi, gen2=True)
        np.testing.assert_allclose(np.array(got), expected, rtol=1e-10)
        default = psd_noise.noise(jnp.array(f), {'A': A, 'P': P}, tdi=tdi, gen2=True)
        off = psd_noise.noise(jnp.array(f), {'A': A, 'P': P, 'paper_eqs': 0}, tdi=tdi, gen2=True)
        np.testing.assert_array_equal(np.array(off), np.array(default))


def test_average_chunks_counts():
    """B4: linear binning returns the number of amplitudes actually averaged in each bin."""
    freqs = np.arange(1, 1001) * 1e-5
    f, d, r, count = setting_data.average_chunks(freqs, np.ones(1000), np.ones(1000), 1000 // 50)
    assert len(f) == 50 and count.sum() == 1000


@pytest.mark.parametrize('unscheduled_gap', [0., 3 * 3600.])
def test_gap_segments_stay_between_gaps(unscheduled_gap):
    """B5: segments never overlap, never pass T_obs and are at most sched_period long."""
    for seed in range(5):
        np.random.seed(seed)
        start, end = gaps.generate_gaps(YEAR, scheduled_gap=7 * 3600, scheduled_period=14 * DAY,
                                        unscheduled_gap=unscheduled_gap, exp_scale=10 * DAY,
                                        merge_threshold=DAY, duty_cycle=0.75)
        assert np.all(end <= YEAR)
        assert np.all(start[1:] >= end[:-1])
        assert np.all(end - start <= 14 * DAY + 1e-6)
        assert np.all(end - start > 0.9 * 14 * DAY)


ENVELOPE = dict(lat=-0.0966, long=-1.626, s1=0.0416, s2=0.1384, sin_psi=-0.83)


@pytest.mark.parametrize('alpha0, beta0', [(0.4, -1.1), (2.0, 0.7)])
def test_envelope_rotation(alpha0, beta0):
    """B6: with alpha0, beta0 != 0, A^2 + E^2 depends on alpha0 only through l - alpha0, and the chunk average
    equals the time average of the instantaneous envelope."""
    e = ENVELOPE
    t1, t2 = 0.1 * YEAR, 0.1 * YEAR + 30 * DAY
    ts = jnp.linspace(t1, t2, 4001)
    A, E = envelopes_gaussian(e['lat'], e['long'], e['s1'], e['s2'], e['sin_psi'], 1 / YEAR, ts, alpha0, beta0)
    A_ref, E_ref = envelopes_gaussian(e['lat'], e['long'] - alpha0, e['s1'], e['s2'], e['sin_psi'], 1 / YEAR,
                                      ts, 0., beta0 - alpha0)
    np.testing.assert_allclose(np.array(A ** 2 + E ** 2), np.array(A_ref ** 2 + E_ref ** 2), rtol=1e-10)
    for tdi, inst in ((0, A), (1, E)):
        chunk = average_envelopes_gaussian(e['lat'], e['long'], e['s1'], e['s2'], e['sin_psi'], t1, t2, 1 / YEAR,
                                           alpha0, beta0, tdi)
        time_avg = np.trapezoid(np.array(inst) ** 2, np.array(ts)) / (t2 - t1)
        assert float(chunk) == pytest.approx(time_avg, rel=1e-6)


def test_point_source_time_average():
    """B7: the point-source chunk average of average_envelope_num equals the time average of the point-source
    envelope (envelopes_gaussian at zero width, divided by cos b, convention C1)."""
    from bahamas.psd_response.average_envelope_num import envelopes_time_average
    rng = np.random.default_rng(2)
    for _ in range(4):
        b, l = rng.uniform(-1.2, 1.2), rng.uniform(0, 2 * np.pi)
        t1 = rng.uniform(0, 300) * DAY
        t2 = t1 + 30 * DAY
        ts = jnp.linspace(t1, t2, 4001)
        A, E = envelopes_gaussian(np.sin(b), l, 0., 0., 0., 1 / YEAR, ts)
        for tdi, inst in ((0, A), (1, E)):
            ref = np.trapezoid(np.array(inst) ** 2, np.array(ts)) / (t2 - t1) / np.cos(b)
            assert float(envelopes_time_average(b, l, 1 / YEAR, t1, t2, tdi)) == pytest.approx(ref, rel=1e-6)
