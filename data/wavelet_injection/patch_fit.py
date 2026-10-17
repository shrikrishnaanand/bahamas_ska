"""
Run 5a: fit the fluxes of patches at known positions to run 5's data (smooth Galaxy + four bright patches).

Same data and WDM likelihood as the wavelet run on run 5 (``injection.py --tag run5 --injection patches``); only the
model of the power modulation differs. Instead of wavelet atoms in time, P_c(t) is built from the sky::

    P_c(t) = T_c,gal(t) + sum_i F_i T_c,i(t)

``T_c,gal`` is the smooth Galaxy (run 2's Gaussian, shape fixed at the truth) and ``T_c,i`` the modulation of patch i
per unit flux, at its true position and shape (``sky_patches.Run5Sky``). ``F_i`` (flux in units of the Galaxy's) and
the 7 spectral parameters (alpha, amp, fknee, fr1, fr2, A, P) are sampled with NUTS. Priors: uniform within the
bounds of ``config/sources.yaml`` for the spectral parameters (as in runs 1-4), F_i ~ U(0, 1).

The model is linear in F at fixed spectra, and the templates are evaluated once at the likelihood's group times.

    python patch_fit.py        -> outputs/run5/5a/chains.npz   (a few minutes)
"""
import os
import time
import copy

import numpy as np
import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import injection as I
import sky_patches as SP
from bahamas.wavelet.model import WaveletLikelihood
from bahamas.wavelet.sampler import WaveletPosterior, _sampled
from bahamas.wavelet.simulate import ENVELOPE_PARAMS

OUT = os.path.join(I.OUT, 'run5', '5a')
FLUX_PRIOR = (0.0, 1.0)


def spectral_sources():
    """Spectral parameters of ``config/sources.yaml`` (sky parameters removed), amp at its true value."""
    src = copy.deepcopy(I.og_sources())
    src['galactic_DWD_time'] = [p for p in src['galactic_DWD_time'] if p['name'] not in ENVELOPE_PARAMS]
    return src


class PatchFit:
    """
    Likelihood of run 5a: spectral parameters + patch fluxes, Galaxy shape and patch positions fixed.

    Args:
        data (WaveletData): Run 5's WDM data.
        config (dict): Data settings (``injection.CONFIG``).
    """

    def __init__(self, data, config=I.CONFIG):
        wcfg = config['wavelet']
        self.like = WaveletLikelihood(data, config['f1'], config['f2'], freq_bins=wcfg['freq_bins'],
                                      edge_bins=wcfg['edge_bins'], nsub=wcfg['band_samples'],
                                      time_bin=wcfg['time_bin'])
        # modulation='none' only to reuse the spectra; P comes from the templates below
        self.post = WaveletPosterior(self.like, spectral_sources(), modulation='none', gen2=config['gen2'])
        self.psd = _sampled(self.post.psd_params)
        self.psd_names = [p.name for p in self.psd]
        self.sky = SP.Run5Sky(t0=data.t0, T=data.Nt * data.Nf * data.dt)
        self.patch_names = self.sky.names[1:]
        self.flux_true = np.array([p['flux'] for p in self.sky.cfg['patches']])
        self.T = jnp.asarray(self.templates(np.asarray(self.like.times)))          # (2, ntimes, 1 + npatch)

    def templates(self, times):
        """``[T_gal, T_1, ...]`` per channel at ``times``: shape (2, len(times), 1 + npatch); patches per unit flux.

        ``Run5Sky.P_components`` are built from maps that already carry each patch's flux, so divide it out.
        """
        per_unit = np.r_[1.0, self.flux_true]
        return np.stack([np.stack([np.interp(times, self.sky.times, self.sky.P_components[c, :, i]) / per_unit[i]
                                   for i in range(len(self.sky.names))], axis=-1) for c in (0, 1)])

    def P(self, flux, T=None):
        T = self.T if T is None else T
        return T[..., 0] + T[..., 1:] @ flux

    def log_likelihood(self, theta_psd, flux):
        S_gal, S_noise, _ = self.post.spectra(theta_psd)
        return self.like.log_likelihood(S_gal, S_noise, self.P(flux))

    def log_var(self, theta_psd, flux):
        """log of the model variance per channel, time group and frequency group (as in ``log_likelihood``)."""
        S_gal, S_noise, _ = self.post.spectra(theta_psd)
        S_gal = jnp.broadcast_to(S_gal, S_noise.shape)
        return jnp.log(self.P(flux)[:, :, None] * S_gal[:, None, :] + S_noise[:, None, :])

    def fisher(self, theta_psd, flux):
        """Expected (Fisher) information of the Gamma likelihood: sum nu/2 dlog var/dx_i dlog var/dx_j."""
        n = len(theta_psd)
        J = jax.jacfwd(lambda x: self.log_var(x[:n], x[n:]))(jnp.asarray(np.r_[theta_psd, flux]))
        J = np.asarray(J).reshape(-1, J.shape[-1])
        w = np.broadcast_to(np.asarray(self.like.nu_sg)[None] / 2, (2,) + np.asarray(self.like.nu_sg).shape).ravel()
        return J.T @ (w[:, None] * J)

    def truth(self):
        return np.array([p.value for p in self.psd], float), self.flux_true


def run_nuts(fit, warmup=1000, samples=2000, chains=4, seed=0, progress=True, chain_method='vectorized'):
    """NUTS on (spectral parameters, fluxes), started at the truth (as the wavelet runs).

    The parameters differ in posterior width by orders of magnitude (amp ~0.01, fluxes ~0.1-1, noise levels ~0.1)
    and are strongly correlated (the four patch templates are 96-99% parallel over 30 days), which makes plain NUTS
    take maximal trees. So it samples whitened logit coordinates: x = lo + (hi - lo) sigmoid(u) maps each parameter
    to its uniform prior range (the log-Jacobian keeps the prior uniform in x, and the walls are smooth), and
    u = u0 + L z with L L^T the inverse of the curvature in u at the truth (Fisher matrix in u plus the Jacobian
    term). z has an improper flat prior.
    """
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, init_to_value

    numpyro.enable_x64()
    theta0, flux0 = fit.truth()
    npsd = len(theta0)
    x0 = np.r_[theta0, flux0]
    lo = np.array([p.bounds[0] for p in fit.psd] + [FLUX_PRIOR[0]] * len(flux0), float)
    hi = np.array([p.bounds[1] for p in fit.psd] + [FLUX_PRIOR[1]] * len(flux0), float)
    s0 = (x0 - lo) / (hi - lo)
    u0 = np.log(s0 / (1 - s0))
    J = (hi - lo) * s0 * (1 - s0)                                         # dx/du at the truth
    fisher = fit.fisher(theta0, flux0)
    cov = np.linalg.inv(fisher)
    curv_u = J[:, None] * fisher * J[None, :] + np.diag(2 * s0 * (1 - s0))
    L = jnp.asarray(np.linalg.cholesky(np.linalg.inv(curv_u)))
    u0, lo_j, hi_j = jnp.asarray(u0), jnp.asarray(lo), jnp.asarray(hi)
    names = fit.psd_names + [f'F_{n}' for n in fit.patch_names]

    def model():
        z = numpyro.sample('z', dist.ImproperUniform(dist.constraints.real_vector, (), event_shape=(len(names),)))
        u = u0 + L @ z
        s = jax.nn.sigmoid(u)
        x = lo_j + (hi_j - lo_j) * s
        for i, n in enumerate(names):
            numpyro.deterministic(n, x[i])
        numpyro.factor('log_jacobian', jnp.sum(jax.nn.log_sigmoid(u) + jax.nn.log_sigmoid(-u)))
        numpyro.factor('log_likelihood', fit.log_likelihood(x[:npsd], x[npsd:]))

    kernel = NUTS(model, init_strategy=init_to_value(values={'z': jnp.zeros(len(names))}), dense_mass=True)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=chains, chain_method=chain_method,
                progress_bar=progress)
    t = time.time()
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=('diverging', 'num_steps'))
    runtime = time.time() - t
    draws = mcmc.get_samples(group_by_chain=True)
    extra = mcmc.get_extra_fields(group_by_chain=True)
    from numpyro.diagnostics import summary
    diag = summary(draws, group_by_chain=True)

    flat = {k: np.asarray(v).reshape(-1) for k, v in draws.items()}
    psd_chain = np.column_stack([flat[n] for n in fit.psd_names])
    flux_chain = np.column_stack([flat[f'F_{n}'] for n in fit.patch_names])
    ll = jax.jit(jax.vmap(fit.log_likelihood))(jnp.asarray(psd_chain), jnp.asarray(flux_chain))
    names = fit.psd_names + [f'F_{n}' for n in fit.patch_names]
    return {'psd_chain': psd_chain, 'psd_names': np.array(fit.psd_names), 'psd_true': theta0,
            'flux_chain': flux_chain, 'patch_names': np.array(fit.patch_names), 'flux_true': flux0,
            'log_like': np.asarray(ll), 'runtime_s': runtime, 'chains': chains, 'samples': samples,
            'warmup': warmup, 'divergences': int(np.asarray(extra['diverging']).sum()),
            'num_steps_mean': float(np.asarray(extra['num_steps']).mean()),
            'diag_names': np.array(names), 'r_hat': np.array([float(diag[n]['r_hat']) for n in
                                                              [*fit.psd_names, *[f'F_{n}' for n in fit.patch_names]]]),
            'n_eff': np.array([float(diag[n]['n_eff']) for n in
                               [*fit.psd_names, *[f'F_{n}' for n in fit.patch_names]]]),
            'loglike_true': float(fit.log_likelihood(jnp.asarray(theta0), jnp.asarray(flux0))),
            'analysed_times': np.asarray(fit.like.times),
            'fisher_sigma': np.sqrt(np.diag(cov))}


def run(tag='run5', **kw):
    data, _ = I.load_or_simulate('patches', tag)
    fit = PatchFit(data)
    res = run_nuts(fit, **kw)
    os.makedirs(OUT, exist_ok=True)
    np.savez(os.path.join(OUT, 'chains.npz'), **res)
    return res, fit


if __name__ == '__main__':
    res, fit = run()
    print(f"done in {res['runtime_s'] / 60:.1f} min, {res['divergences']} divergences, "
          f"mean leapfrog steps {res['num_steps_mean']:.0f}")
    for n, rh, ne in zip(res['diag_names'], res['r_hat'], res['n_eff']):
        print(f"  {n:12s} r_hat {rh:.3f}  n_eff {ne:7.0f}")
    for n, ch, tr in zip(res['patch_names'], res['flux_chain'].T, res['flux_true']):
        print(f"  {n:9s} flux {np.median(ch):.3f} +- {ch.std():.3f}  (true {tr:.2f}, {np.median(ch) / ch.std():.1f} sigma)")
