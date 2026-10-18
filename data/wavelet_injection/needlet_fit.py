"""
Run 5c: fit the excess sky in needlets to run 5's data (smooth Galaxy + four bright patches), no positions assumed.

Same data and WDM likelihood as runs 5 (atoms) and 5a (patch fluxes); only the model of the power modulation
differs. The smooth Galaxy is fixed at the truth (shape and unit flux), as in 5a, and the excess is a needlet map::

    P_c(t) = T_c,gal(t) + sum_jk beta_jk T_c,jk(t),    T_c,jk(t) = k (1 / Rbar) int psi_jk R_c(t, n) dOmega

with needlets of bands j = -1, 1, 2 on the response grid (nside 32; ``needlet_patches/needlets.py``). Band j = 0
(l = 1) and j >= 3 (l >= 5) are left out: the A/E response has only l = 0, 2, 4, so their T vanish. That leaves
1 + 48 + 192 = 241 coefficients.

Prior: beta_jk ~ N(0, s_j^2), s_j = rms of the smooth Galaxy's own needlet coefficients in band j (as stage 1 of
``needlet_patches``); nothing about the patches.

Sampling. With z = beta / s ~ N(0, I) the modulation is P = T_gal + M z, M = T diag(s). The likelihood depends on z
only through M z, and M has rank r <= 15 (the l = 0, 2, 4 harmonics). With the SVD M = U S V^T truncated to the r
non-zero singular values, NUTS samples the r data-visible coordinates y = V_r^T z (prior N(0, I)) together with the
7 spectral parameters; the remaining K - r directions of z are independent of the data and are drawn exactly from
their prior. Each posterior draw of beta is s * (V_r y + (I - V_r V_r^T) xi), xi ~ N(0, I). This is the exact
posterior of the 241-coefficient model, at the cost of a 7 + r dimensional NUTS run.

    python needlet_fit.py                     -> outputs/run5/5c/chains.npz   (a few minutes)
    python needlet_fit.py --prior-scale 3     -> outputs/run5/5c/chains_prior3x.npz   (prior sd x 3)
"""
import os
import sys
import time

import numpy as np
import healpy as hp
import jax

jax.config.update('jax_enable_x64', True)
import jax.numpy as jnp

import injection as I
import patch_fit as PF

sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), '..', 'needlet_patches'))
import needlets as NL  # noqa: E402

OUT = os.path.join(I.OUT, 'run5', '5c')
BANDS = (-1, 1, 2)
SV_REL_CUT = 1e-6


class NeedletFit(PF.PatchFit):
    """
    Likelihood of run 5c: spectral parameters + needlet coefficients of the excess, Galaxy fixed at the truth.

    Attributes of note: ``frame`` (needlets on the response grid, bands ``BANDS``), ``s`` (prior sd per
    coefficient), ``Tn`` (2, ntimes, K) needlet templates at the likelihood's times, ``V`` (K, r) the data-visible
    directions of z = beta / s, ``sv`` their singular values, ``beta_true`` the needlet coefficients of the true
    excess.
    """

    def __init__(self, data, config=I.CONFIG, prior_scale=1.0):
        super().__init__(data, config)
        self.prior_scale = prior_scale
        nside = self.sky.cfg['maps']['response_nside']
        frame = NL.NeedletFrame(nside, jmax=max(BANDS))
        keep = np.isin(frame.band, BANDS)
        self.frame = frame
        self.keep = keep
        self.band = frame.band[keep]
        self.Psi = frame.Psi[:, keep]
        self.pixel_area = frame.pixel_area
        self.K = int(keep.sum())
        maps = self.sky.response.modulation(self.Psi * self.pixel_area)                # (2, nt, K), before k
        self.Tn_grid = self.sky.k * maps
        self.Tn = jnp.asarray(self.needlet_templates(np.asarray(self.like.times)))
        self.T_gal = self.T[..., 0]

        lowres = {n: hp.ud_grade(m, nside, power=-2) for n, m in self.sky.maps.items()}
        self.galaxy_map = lowres['galaxy']
        self.excess_map = sum(m for n, m in lowres.items() if n != 'galaxy')
        beta_gal = self.Psi.T @ self.galaxy_map
        self.s = prior_scale * np.array([np.sqrt(np.mean(beta_gal[self.band == j] ** 2)) for j in self.band])
        self.beta_true = self.Psi.T @ self.excess_map

        M = np.asarray(self.Tn).reshape(-1, self.K) * self.s[None, :]
        _, sv, Vt = np.linalg.svd(M, full_matrices=False)
        r = int(np.sum(sv > SV_REL_CUT * sv[0]))
        self.sv, self.sv_all, self.r = sv[:r], sv, r
        self.V = Vt[:r].T                                                                  # (K, r)
        self.MV = jnp.asarray(np.asarray(self.Tn) * self.s[None, None, :] @ self.V)       # (2, ntimes, r)
        self.y_true = self.V.T @ (self.beta_true / self.s)

    def needlet_templates(self, times):
        """Needlet templates ``T_c,jk`` per channel at ``times``: shape (2, len(times), K)."""
        return np.stack([np.stack([np.interp(times, self.sky.times, self.Tn_grid[c, :, k]) for k in range(self.K)],
                                  axis=-1) for c in (0, 1)])

    def P(self, y, MV=None, T_gal=None):
        MV = self.MV if MV is None else MV
        T_gal = self.T_gal if T_gal is None else T_gal
        return T_gal + MV @ y

    def MV_at(self, times):
        """``M V`` at arbitrary times (for plotting P(t) on a fine grid)."""
        return self.needlet_templates(times) * self.s[None, None, :] @ self.V

    def log_likelihood(self, theta_psd, y):
        S_gal, S_noise, _ = self.post.spectra(theta_psd)
        return self.like.log_likelihood(S_gal, S_noise, self.P(y))

    def log_var(self, theta_psd, y):
        S_gal, S_noise, _ = self.post.spectra(theta_psd)
        S_gal = jnp.broadcast_to(S_gal, S_noise.shape)
        return jnp.log(self.P(y)[:, :, None] * S_gal[:, None, :] + S_noise[:, None, :])

    def truth(self):
        return np.array([p.value for p in self.psd], float), self.y_true

    def beta_draws(self, y_chain, seed=0):
        """Exact posterior draws of all K coefficients: data-visible part from ``y``, the rest from the prior."""
        rng = np.random.default_rng(seed)
        xi = rng.standard_normal((len(y_chain), self.K))
        z = y_chain @ self.V.T + xi - (xi @ self.V) @ self.V.T
        return z * self.s[None, :]

    def flux_map(self, beta):
        """Flux map(s) of coefficients ``beta`` (..., K) on the response grid."""
        return beta @ (self.Psi * self.pixel_area).T


def visible_part(flux_map, lmax=4):
    """The l = 0, 2, 4 part of a flux map (what the A/E response sees)."""
    nside = hp.npix2nside(flux_map.shape[-1])
    alm = hp.map2alm(flux_map, lmax=lmax, iter=3)
    ell, _ = hp.Alm.getlm(lmax)
    alm[ell % 2 == 1] = 0
    return hp.alm2map(alm, nside, lmax=lmax)


def run_nuts(fit, warmup=1000, samples=2000, chains=4, seed=0, progress=True, chain_method='vectorized',
             target_accept=0.95):
    """NUTS on (spectral parameters, q), started at the truth, in whitened coordinates.

    Spectral parameters: logit map to their uniform prior range, as in 5a. The excess is sampled as
    q = g y with g = 10^(amp - amp0): the galactic power is 10^amp (T_gal + M V y) = 10^amp0 (g T_gal + M V q), so q
    enters independently of amp. In (amp, y) the two trade along a curved ridge (corr(amp, y_0) = 0.985), which
    gave 7% divergent transitions; in (amp, q) the ridge is straight. The prior y ~ N(0, I) becomes
    log p(q | amp) = -|q / g|^2 / 2 - r log g (Jacobian of y = q / g). Then (u, q) = (u0, q0) + L z with L L^T the
    inverse curvature at the truth (Fisher + prior), and NUTS samples z. The chain is stored as y = q / g.
    """
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, init_to_value

    numpyro.enable_x64()
    theta0, y0 = fit.truth()
    npsd, r = len(theta0), fit.r
    lo = np.array([p.bounds[0] for p in fit.psd], float)
    hi = np.array([p.bounds[1] for p in fit.psd], float)
    s0 = (theta0 - lo) / (hi - lo)
    u0 = np.log(s0 / (1 - s0))
    Jx = np.r_[(hi - lo) * s0 * (1 - s0), np.ones(r)]                                # dx/du at the truth
    ia = fit.psd_names.index('amp')
    amp0 = theta0[ia]

    def log_var_q(x):
        g = 10 ** (x[ia] - amp0)
        return fit.log_var(x[:npsd], x[npsd:] / g)

    Jq = np.asarray(jax.jacfwd(log_var_q)(jnp.asarray(np.r_[theta0, y0])))
    Jq = Jq.reshape(-1, Jq.shape[-1])
    w = np.broadcast_to(np.asarray(fit.like.nu_sg)[None] / 2, (2,) + np.asarray(fit.like.nu_sg).shape).ravel()
    fisher = Jq.T @ (w[:, None] * Jq)
    curv = Jx[:, None] * fisher * Jx[None, :]
    curv[:npsd, :npsd] += np.diag(2 * s0 * (1 - s0))
    curv[npsd:, npsd:] += np.eye(r)
    L = jnp.asarray(np.linalg.cholesky(np.linalg.inv(curv)))
    c0 = jnp.asarray(np.r_[u0, y0])
    lo_j, hi_j = jnp.asarray(lo), jnp.asarray(hi)
    ynames = [f'y{i}' for i in range(r)]

    def model():
        z = numpyro.sample('z', dist.ImproperUniform(dist.constraints.real_vector, (), event_shape=(npsd + r,)))
        c = c0 + L @ z
        u, q = c[:npsd], c[npsd:]
        x = lo_j + (hi_j - lo_j) * jax.nn.sigmoid(u)
        log_g = (x[ia] - amp0) * jnp.log(10.0)
        y = q * jnp.exp(-log_g)
        for i, n in enumerate(fit.psd_names):
            numpyro.deterministic(n, x[i])
        numpyro.deterministic('y', y)
        numpyro.factor('log_jacobian', jnp.sum(jax.nn.log_sigmoid(u) + jax.nn.log_sigmoid(-u)))
        numpyro.factor('log_prior_q', -0.5 * jnp.sum(y ** 2) - r * log_g)
        numpyro.factor('log_likelihood', fit.log_likelihood(x, y))

    kernel = NUTS(model, init_strategy=init_to_value(values={'z': jnp.zeros(npsd + r)}), dense_mass=True,
                  target_accept_prob=target_accept)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=chains, chain_method=chain_method,
                progress_bar=progress)
    t = time.time()
    mcmc.run(jax.random.PRNGKey(seed), extra_fields=('diverging', 'num_steps'))
    runtime = time.time() - t
    draws = mcmc.get_samples(group_by_chain=True)
    extra = mcmc.get_extra_fields(group_by_chain=True)
    from numpyro.diagnostics import summary
    diag_in = {n: draws[n] for n in fit.psd_names}
    diag_in.update({yn: draws['y'][..., i] for i, yn in enumerate(ynames)})
    diag = summary(diag_in, group_by_chain=True)

    psd_chain = np.column_stack([np.asarray(draws[n]).reshape(-1) for n in fit.psd_names])
    y_chain = np.asarray(draws['y']).reshape(-1, r)
    ll = jax.jit(jax.vmap(fit.log_likelihood))(jnp.asarray(psd_chain), jnp.asarray(y_chain))
    names = fit.psd_names + ynames
    return {'psd_chain': psd_chain, 'psd_names': np.array(fit.psd_names), 'psd_true': theta0,
            'y_chain': y_chain, 'y_true': y0, 'sv': fit.sv, 'sv_all': fit.sv_all, 'V': fit.V, 's': fit.s,
            'band': fit.band, 'beta_true': fit.beta_true, 'beta_draws': fit.beta_draws(y_chain[::4]),
            'log_like': np.asarray(ll), 'runtime_s': runtime, 'chains': chains, 'samples': samples,
            'warmup': warmup, 'divergences': int(np.asarray(extra['diverging']).sum()), 'target_accept': target_accept,
            'diverging': np.asarray(extra['diverging']).reshape(-1),
            'num_steps_mean': float(np.asarray(extra['num_steps']).mean()),
            'diag_names': np.array(names), 'r_hat': np.array([float(diag[n]['r_hat']) for n in names]),
            'n_eff': np.array([float(diag[n]['n_eff']) for n in names]),
            'loglike_true': float(fit.log_likelihood(jnp.asarray(theta0), jnp.asarray(y0))),
            'analysed_times': np.asarray(fit.like.times)}


def run(tag='run5', prior_scale=1.0, **kw):
    data, _ = I.load_or_simulate('patches', tag)
    fit = NeedletFit(data, prior_scale=prior_scale)
    res = run_nuts(fit, **kw)
    res['prior_scale'] = prior_scale
    os.makedirs(OUT, exist_ok=True)
    name = 'chains.npz' if prior_scale == 1.0 else f'chains_prior{prior_scale:g}x.npz'
    np.savez(os.path.join(OUT, name), **res)
    return res, fit


if __name__ == '__main__':
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument('--prior-scale', type=float, default=1.0, help='multiply the prior sd s_j (sensitivity check)')
    res, fit = run(prior_scale=ap.parse_args().prior_scale)
    print(f"K = {fit.K} needlets, r = {fit.r} data-visible directions; "
          f"done in {res['runtime_s'] / 60:.1f} min, {res['divergences']} divergences, "
          f"mean leapfrog steps {res['num_steps_mean']:.0f}")
    for n, rh, ne in zip(res['diag_names'], res['r_hat'], res['n_eff']):
        print(f"  {n:8s} r_hat {rh:.3f}  n_eff {ne:7.0f}")
    sd = res['y_chain'].std(0)
    print('posterior sd of the visible directions (prior 1):', np.round(sd, 3))
