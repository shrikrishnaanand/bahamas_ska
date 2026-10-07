import numpy as np
import jax.numpy as jnp
import pytest

from bahamas.wavelet.model import WaveletData, WaveletLikelihood
from bahamas.wavelet.sampler import Parameter, WaveletPosterior, WaveletSampler, bayes_factors, _pad_leaves


class FlatPosterior:
    """Constant likelihood: the reversible-jump chain must sample the prior on K."""
    rj_branches = ['mod_A']
    branch_names = ['psd', 'mod_A']
    psd_params = [Parameter('instr_noise', 'A', (1, 5), 2.4)]
    envelope_params = []
    max_wavelets = 4
    batch_size = None

    def __call__(self, params, groups):
        return np.zeros(int(np.max(groups[0])) + 1)


def test_bayes_factors_from_counts():
    nleaves = np.repeat([[0], [1], [1], [2], [2], [2], [2]], 10, axis=1).T  # (10 steps, 7 walkers)
    bf = bayes_factors(nleaves, 0, 3)
    assert np.allclose(bf['posterior'], [1 / 7, 2 / 7, 4 / 7, 0])
    assert bf['K_ref'] == 2
    assert np.allclose(bf['lnB'][:3], np.log([1 / 4, 2 / 4, 1]))
    assert np.isnan(bf['lnB'][3])
    assert bf['lnB_upper'] < bf['lnB'][0]


def test_prior_only_k_is_uniform():
    """With a flat likelihood every K, including 0 and the maximum, is equally likely."""
    s = WaveletSampler(FlatPosterior(), nwalkers=16, ntemps=1, min_wavelets=0, max_wavelets=4, init='prior', seed=3)
    res = s.run(2000, burnin=100, progress=False)
    bf = res['mod_A_bayes']
    assert np.allclose(bf['posterior'], 0.2, atol=0.03)
    assert np.all(np.abs(bf['lnB']) < 0.2)


class FactorisedPosterior(FlatPosterior):
    """L = prod_k g(t_k): the exact posterior on K is proportional to c^K with c = int g."""
    sigma = 0.3

    def __call__(self, params, groups):
        n = int(np.max(groups[0])) + 1
        atoms, mask = _pad_leaves(params[1], np.asarray(groups[1]), n, self.max_wavelets)
        return np.sum(np.where(mask, -0.5 * ((atoms[..., 0] - 0.5) / self.sigma) ** 2, 0.0), axis=1)


def test_rj_matches_analytic_k_posterior():
    """Reversible jump with a likelihood reproduces a K posterior known in closed form."""
    from scipy.special import erf
    sigma = FactorisedPosterior.sigma
    c = sigma * np.sqrt(2 * np.pi) * erf(0.5 / (sigma * np.sqrt(2)))
    expected = c ** np.arange(5)
    expected /= expected.sum()

    s = WaveletSampler(FactorisedPosterior(), nwalkers=16, ntemps=1, min_wavelets=0, max_wavelets=4,
                       init='prior', seed=5)
    res = s.run(3000, burnin=200, progress=False)
    assert np.allclose(res['mod_A_bayes']['posterior'], expected, atol=0.025)


KMAX = 5


def _known_k_data(atoms, seed, amp=-42.5):
    """
    WDM coefficients drawn directly from the model variance with known wavelet atoms.

    The default galactic amplitude makes the galaxy several times louder than the
    noise in the band, so that no atom can hide (see the note above ATOMS).
    """
    sources = {
        'galactic_DWD_time': [
            {'name': 'alpha', 'bounds': None, 'injected': 1.8},
            {'name': 'amp', 'bounds': [-44, -41], 'injected': amp},
            {'name': 'fknee', 'bounds': None, 'injected': -2.595679532778269},
            {'name': 'fr1', 'bounds': None, 'injected': -2.856},
            {'name': 'fr2', 'bounds': None, 'injected': -3.487854923626168},
        ],
        'instr_noise': [
            {'name': 'A', 'bounds': [1, 5], 'injected': 2.4},
            {'name': 'P', 'bounds': None, 'injected': 7.9},
        ],
    }
    dt, Nf, Nt = 1000.0, 64, 492
    blank = WaveletData(np.ones((1, Nt, Nf)), dt, channels=[0])
    like = WaveletLikelihood(blank, 1e-4, 4.9e-4, edge_bins=0)
    post = WaveletPosterior(like, sources, modulation='wavelets', gen2=True, max_wavelets=KMAX)
    S_gal, S_n, values = post.spectra(jnp.array([amp, 2.4]))
    a, m = _pad_leaves(atoms, np.zeros(len(atoms), int), 1, KMAX)
    P = np.asarray(post.power_modulation(None, (a[0],), (m[0],), values, blank.times))

    # Pixel variances on the full grid (pixels outside the band keep variance 1)
    var = np.ones((Nt, Nf))
    var[:, like.pixels] = (P[0][:, None] * np.asarray(S_gal)[None, :] + np.asarray(S_n)[0][None, :]) / (2 * dt)
    w = np.random.default_rng(seed).normal(size=(Nt, Nf)) * np.sqrt(var)
    data = WaveletData(w[None], dt, channels=[0])
    return sources, WaveletLikelihood(data, 1e-4, 4.9e-4, edge_bins=0)


# With amp = -42.5 the galaxy is several times louder than the noise in this band, so every
# atom changes the likelihood. When the galaxy is below the noise, negative or narrow atoms
# barely change the total variance and K is genuinely poorly constrained.
ATOMS = np.array([[0.3, np.log10(40.0), 1.0], [0.7, np.log10(30.0), -1.0]])


def _sampler(post, seed=4):
    return WaveletSampler(post, nwalkers=16, ntemps=2, min_wavelets=0, max_wavelets=KMAX, seed=seed)


@pytest.mark.parametrize("atoms", [np.empty((0, 3)), ATOMS])
def test_posterior_prefers_true_number_of_wavelets(atoms):
    """Started at the truth, fewer atoms are excluded and more atoms are disfavoured."""
    k_true = len(atoms)
    sources, like = _known_k_data(atoms, seed=10 + k_true)
    post = WaveletPosterior(like, sources, modulation='wavelets', gen2=True, max_wavelets=KMAX)
    s = _sampler(post)
    coords, inds = s.initial_state()
    if k_true:
        noise = 1e-3 * np.random.default_rng(0).normal(size=coords['mod_A'][..., :k_true, :].shape)
        coords['mod_A'][..., :k_true, :] = atoms + noise
        inds['mod_A'][..., :k_true] = True
    p = s.run_from(coords, inds, 600, progress=False)['mod_A_bayes']['posterior']
    assert np.sum(p[:k_true]) < 0.01
    assert np.argmax(p) == k_true
    assert np.all(p[k_true + 1:] < p[k_true])


def test_rj_detects_modulation_from_scratch():
    """Starting with no atoms, the sampler adds atoms until both features are modelled."""
    sources, like = _known_k_data(ATOMS, seed=12)
    post = WaveletPosterior(like, sources, modulation='wavelets', gen2=True, max_wavelets=KMAX)
    p = _sampler(post).run(500, burnin=300, progress=False)['mod_A_bayes']['posterior']
    assert np.sum(p[:2]) < 0.01


def test_max_wavelets_must_match_posterior():
    sources, like = _known_k_data(np.empty((0, 3)), seed=1)
    post = WaveletPosterior(like, sources, modulation='wavelets', gen2=True, max_wavelets=KMAX)
    with pytest.raises(ValueError):
        WaveletSampler(post, nwalkers=16, max_wavelets=KMAX + 1)
