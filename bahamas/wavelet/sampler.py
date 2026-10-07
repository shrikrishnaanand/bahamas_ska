"""
Sampling of the wavelet-domain galactic foreground model.

The likelihood is a single pure JAX function, jit-compiled and vectorised over
walkers with ``vmap``. Two samplers use it:

- **Eryn** (any modulation; required for ``modulation: wavelets``), with
  reversible jump over the number of wavelet atoms. Eryn runs in
  vectorised mode, so each likelihood request from the sampler is one
  compiled batch call.
- **NUTS** from NumPyro (fixed-dimension models: ``parametric`` and ``none``),
  like the chunked pipeline. It uses gradients of the same likelihood.

Eryn branches:

- ``psd``: spectral parameters of the galaxy (``alpha``, ``amp``, ``fknee``,
  ``fr1``, ``fr2``), the instrument noise (``A``, ``P``) and, for wavelet
  modulation, the log levels of channels after the first. Always one leaf.
- ``envelope`` (``modulation: parametric`` with sampled sky parameters):
  ``lat``, ``long``, ``s1``, ``s2``, ``psi``. Always one leaf.
- ``mod_<channel>`` (``modulation: wavelets``): one reversible-jump branch per
  TDI channel whose leaves are wavelet atoms ``(t_frac, log10_width_days, amp)``.

The posterior over the number of atoms ``K`` gives the Bayes factor between
models with different ``K`` directly, without a separate evidence computation
for each ``K``.
"""
import warnings

import numpy as np
import jax
import jax.numpy as jnp
from eryn.ensemble import EnsembleSampler
from eryn.prior import ProbDistContainer, uniform_dist
from eryn.state import State
from eryn.moves import StretchMove, GaussianMove, DistributionGenerateRJ

from bahamas.wavelet.model import stationary_modulation, parametric_modulation, WaveletModulation
from bahamas.wavelet.simulate import ENVELOPE_PARAMS
from bahamas.logger_config import logger

# Public from jax 0.4.31; it only steers XLA's fusion, so older versions run without it
_optimization_barrier = getattr(jax.lax, 'optimization_barrier', lambda x: x)

GALAXY_SOURCES = ('galactic_DWD_time', 'galactic_DWD')
ATOM_PARAMS = ['t_frac', 'log10_width_days', 'amp']


class Parameter:
    """A model parameter: sampled if ``bounds`` is set, otherwise fixed at ``value``."""

    def __init__(self, source, name, bounds=None, value=None):
        self.source = source
        self.name = name
        self.bounds = None if bounds is None else (float(bounds[0]), float(bounds[1]))
        self.value = value

    @property
    def label(self):
        return f"{self.source}:{self.name}"


def _sampled(params):
    return [p for p in params if p.bounds is not None]


def _injected(params):
    """Injected values of sampled parameters (NaN where unknown)."""
    return np.array([np.nan if p.value is None else p.value for p in _sampled(params)], dtype=float)


def _fill(params, x):
    """``{source: {name: value}}`` from fixed values and the sampled vector ``x`` (may be traced)."""
    out, i = {}, 0
    for p in params:
        if p.bounds is not None:
            value = x[i]
            i += 1
        else:
            value = p.value
        out.setdefault(p.source, {})[p.name] = value
    return out


def _pad_leaves(X, groups, n, kmax):
    """Scatter eryn's flat leaves into padded (n, kmax, 3) arrays plus a mask."""
    atoms = np.zeros((n, kmax, 3))
    mask = np.zeros((n, kmax), dtype=bool)
    if len(groups):
        order = np.argsort(groups, kind='stable')
        g = np.asarray(groups)[order]
        slot = np.arange(len(g)) - np.searchsorted(g, g, side='left')
        atoms[g, slot] = np.asarray(X)[order]
        mask[g, slot] = True
    return atoms, mask


class WaveletPosterior:
    """
    Wavelet-domain likelihood as a pure JAX function of the model parameters.

    ``single(theta_psd, theta_env, atoms, masks)`` evaluates one point:
    ``theta_psd`` and ``theta_env`` are the sampled parameter vectors of the
    ``psd`` and ``envelope`` branches, and ``atoms``/``masks`` are tuples with
    one padded (Kmax, 3) array and (Kmax,) mask per channel (empty tuples
    unless ``modulation == 'wavelets'``). ``batch`` is its jit-compiled ``vmap``.
    Calling the object directly is the vectorised eryn interface.

    Args:
        like (WaveletLikelihood): Wavelet-domain likelihood.
        sources (dict): Sources configuration (bounds define the sampled parameters).
        modulation (str): ``'wavelets'``, ``'parametric'`` or ``'none'``.
        gen2 (bool): Second-generation TDI.
        wavelet_shape (str): Atom shape for ``'wavelets'``.
        level_range (tuple): Prior range of the per-channel log levels (``'wavelets'`` only).
        max_wavelets (int): Padded number of atoms per channel (``'wavelets'`` only).
    """

    def __init__(self, like, sources, modulation='wavelets', gen2=True, wavelet_shape='gaussian',
                 level_range=(-2.0, 2.0), max_wavelets=8):
        if modulation not in ('wavelets', 'parametric', 'none'):
            raise ValueError(f"Unknown modulation '{modulation}'")
        self.like = like
        self.gen2 = bool(gen2)
        self.modulation = modulation
        self.channels = like.data.channels
        self.channel_names = like.data.channel_names
        self.max_wavelets = int(max_wavelets)
        self.batch_size = None  # pad eryn batches to this size so jit compiles once

        self.galaxy_source = next((s for s in GALAXY_SOURCES if s in sources), None)
        if 'instr_noise' not in sources:
            raise ValueError("The wavelet model requires an 'instr_noise' source")
        unsupported = set(sources) - {'instr_noise', *GALAXY_SOURCES}
        if unsupported:
            raise NotImplementedError(f"Unsupported sources for the wavelet model: {sorted(unsupported)}")
        if modulation != 'none' and self.galaxy_source != 'galactic_DWD_time':
            raise ValueError("Modulated models need a 'galactic_DWD_time' source")

        self.psd_params, self.envelope_params = [], []
        for source, params in sources.items():
            for p in params:
                param = Parameter(source, p['name'], p.get('bounds'), p.get('injected'))
                if source == 'galactic_DWD_time' and p['name'] in ENVELOPE_PARAMS:
                    if modulation == 'parametric':
                        self.envelope_params.append(param)
                else:
                    self.psd_params.append(param)

        if modulation == 'wavelets':
            for name in self.channel_names[1:]:
                self.psd_params.append(Parameter('modulation', f'level_{name}', level_range, 0.0))
            times = np.asarray(like.data.times)
            self.wavelets = WaveletModulation(times[0], times[-1] - times[0] + (times[1] - times[0]), wavelet_shape)

        if not _sampled(self.psd_params):
            raise ValueError("The 'psd' branch needs at least one sampled parameter")

        self.batch = jax.jit(jax.vmap(self.single))
        self._power_batch = jax.jit(jax.vmap(self.galactic_power, in_axes=(0, 0, 0, 0, None)))

    # ------------------------------------------------------------------ structure
    @property
    def rj_branches(self):
        return [f'mod_{n}' for n in self.channel_names] if self.modulation == 'wavelets' else []

    @property
    def branch_names(self):
        names = ['psd']
        if _sampled(self.envelope_params):
            names.append('envelope')
        return names + self.rj_branches

    @property
    def n_psd(self):
        return len(_sampled(self.psd_params))

    @property
    def n_env(self):
        return len(_sampled(self.envelope_params))

    # ------------------------------------------------------------------ model (pure JAX)
    def spectra(self, theta_psd):
        """
        Band-averaged galactic and stationary PSDs.

        Returns:
            tuple: ``(S_gal, S_noise, values)`` with shapes (ngroups,) and
            (nchannels, ngroups), and the parameter dictionary.
        """
        from bahamas.psd_strain import psd_galaxy as gal
        from bahamas.psd_strain import psd_noise as noise

        values = _fill(self.psd_params, theta_psd)
        f = self.like.eval_freqs
        S_noise = jnp.stack([noise.noise(f, values['instr_noise'], tdi=c, gen2=self.gen2) for c in self.channels])
        S_gal = jnp.zeros_like(f)
        if self.galaxy_source is not None:
            galaxy = gal.galactic_foreground(f, values[self.galaxy_source], gen2=self.gen2)
            if self.galaxy_source == 'galactic_DWD':
                S_noise = S_noise + galaxy[None, :]
            else:
                S_gal = galaxy
        return self.like.band_average(S_gal), self.like.band_average(S_noise), values

    def power_modulation(self, theta_env, atoms, masks, values, times):
        """Power modulation ``P_c(t)``, shape (nchannels, ntimes)."""
        if self.modulation == 'parametric':
            envelope = _fill(self.envelope_params, theta_env)['galactic_DWD_time']
            return parametric_modulation(times, self.channels, envelope)
        if self.modulation == 'wavelets':
            levels = [0.0] + [values['modulation'][f'level_{n}'] for n in self.channel_names[1:]]
            return self.wavelets(times, atoms, masks, levels)
        return stationary_modulation(times, self.channels)

    def single(self, theta_psd, theta_env, atoms, masks):
        """Log-likelihood at one point (pure JAX, differentiable)."""
        S_gal, S_noise, values = self.spectra(theta_psd)
        P = self.power_modulation(theta_env, atoms, masks, values, self.like.times)
        # Keep XLA from fusing the (expensive) spectra into the per-pixel loop,
        # which would recompute them for every time bin
        S_gal, S_noise, P = _optimization_barrier((S_gal, S_noise, P))
        return self.like.log_likelihood(S_gal, S_noise, P)

    def galactic_power(self, theta_psd, theta_env, atoms, masks, times):
        """``10^(amp - amp_injected) * P_c(t)``: galactic power relative to the injected amplitude."""
        values = _fill(self.psd_params, theta_psd)
        amp = values[self.galaxy_source]['amp']
        amp_ref = next(p.value for p in self.psd_params if p.source == self.galaxy_source and p.name == 'amp')
        return 10 ** (amp - amp_ref) * self.power_modulation(theta_env, atoms, masks, values, times)

    # ------------------------------------------------------------------ eryn interface
    def _empty_atoms(self, n):
        k = len(self.rj_branches)
        return (tuple(jnp.zeros((n, self.max_wavelets, 3)) for _ in range(k)),
                tuple(jnp.zeros((n, self.max_wavelets), dtype=bool) for _ in range(k)))

    def __call__(self, params, groups):
        """
        Vectorised eryn likelihood.

        Args:
            params (list): Per branch, the flat leaf coordinates of all walkers.
            groups (list): Per branch, the walker (group) index of each leaf.

        Returns:
            np.ndarray: Log-likelihood per group.
        """
        if not isinstance(params, list):
            params, groups = [params], [groups]
        by_branch = dict(zip(self.branch_names, zip(params, groups)))

        X, g = by_branch['psd']
        g = np.asarray(g)
        n = int(g.max()) + 1
        size = max(n, self.batch_size or 0)

        theta_psd = np.zeros((size, self.n_psd))
        theta_psd[g] = X
        theta_psd[n:] = theta_psd[0]
        theta_env = np.zeros((size, self.n_env))
        if 'envelope' in by_branch:
            Xe, ge = by_branch['envelope']
            theta_env[np.asarray(ge)] = Xe
            theta_env[n:] = theta_env[0]

        atoms, masks = [], []
        for b in self.rj_branches:
            Xb, gb = by_branch[b]
            a, m = _pad_leaves(Xb, np.asarray(gb), size, self.max_wavelets)
            atoms.append(a)
            masks.append(m)

        ll = np.asarray(self.batch(jnp.asarray(theta_psd), jnp.asarray(theta_env), tuple(atoms), tuple(masks)))[:n]
        # eryn expects finite values; -inf breaks its acceptance bookkeeping
        return np.where(np.isfinite(ll), ll, -1e300)


def bayes_factors(nleaves, kmin, kmax, nblocks=10):
    """
    Log Bayes factors between numbers of atoms from a reversible-jump chain.

    The prior on ``K`` is uniform (eryn's birth/death move includes the
    edge corrections at ``kmin`` and ``kmax``), so the Bayes factor is the
    posterior odds: ``ln B(K, K_ref) = ln[p(K|d) / p(K_ref|d)]`` with ``K_ref``
    the most probable ``K``. Errors come from splitting the chain into
    ``nblocks`` consecutive blocks.

    Args:
        nleaves (array): Number of leaves, shape (nsteps, nwalkers) in step order.
        kmin, kmax (int): Allowed range of ``K``.
        nblocks (int): Number of blocks for the error estimate.

    Returns:
        dict: ``K``, ``posterior``, ``lnB`` (NaN where ``K`` was never
        visited), ``lnB_err``, ``lnB_upper`` (bound for unvisited ``K``) and ``K_ref``.
    """
    nleaves = np.asarray(nleaves)
    K = np.arange(kmin, kmax + 1)

    def odds(samples):
        counts = np.array([(samples == k).sum() for k in K], dtype=float)
        return counts / max(counts.sum(), 1.0)

    post = odds(nleaves.ravel())
    ref = int(np.argmax(post))
    with np.errstate(divide='ignore', invalid='ignore'):
        lnB = np.log(post) - np.log(post[ref])
        lnB[post == 0] = np.nan

        block_lnB = []
        for b in np.array_split(nleaves, nblocks, axis=0):
            p = odds(b.ravel())
            block_lnB.append(np.log(p) - np.log(p[ref]))
        block_lnB = np.array(block_lnB)
        block_lnB[~np.isfinite(block_lnB)] = np.nan
        nvalid = np.sum(np.isfinite(block_lnB), axis=0)
        lnB_err = np.full(len(K), np.nan)
        ok = nvalid > 1
        lnB_err[ok] = np.nanstd(block_lnB[:, ok], axis=0, ddof=1) / np.sqrt(nvalid[ok])

    # A single visit to an unvisited K would give this value: an upper limit
    upper = np.log(1.0 / nleaves.size) - np.log(post[ref])
    return {'K': K, 'posterior': post, 'lnB': lnB, 'lnB_err': lnB_err,
            'lnB_upper': upper, 'K_ref': int(K[ref])}


class WaveletSampler:
    """
    Set up and run eryn for a ``WaveletPosterior``.

    Args:
        posterior (WaveletPosterior): Likelihood.
        nwalkers (int): Walkers per temperature.
        ntemps (int): Number of temperatures (parallel tempering helps RJ mixing).
        min_wavelets, max_wavelets (int): Range of the number of atoms per channel.
        width_range_days (tuple): Prior range of atom widths in days (log-uniform).
        amp_range (tuple): Prior range of atom amplitudes (in ``log P``).
        atom_step (float): Random-walk step for the atoms as a fraction of the prior width.
        init (str): ``'injected'`` starts the fixed-dimension branches in a small
            ball around the injected values, ``'prior'`` draws them from the prior.
        init_scale (float): Size of the injected ball as a fraction of the prior width.
        seed (int, optional): Random seed.
    """

    def __init__(self, posterior, nwalkers=32, ntemps=1, min_wavelets=0, max_wavelets=8,
                 width_range_days=(5.0, 180.0), amp_range=(-3.0, 3.0), atom_step=0.02,
                 init='injected', init_scale=1e-3, seed=None):
        self.posterior = posterior
        self.nwalkers = nwalkers
        self.ntemps = ntemps
        self.kmin, self.kmax = int(min_wavelets), int(max_wavelets)
        if posterior.rj_branches and self.kmax != posterior.max_wavelets:
            raise ValueError(f"max_wavelets ({self.kmax}) must match the posterior ({posterior.max_wavelets})")
        self.init = init
        self.init_scale = init_scale
        self.rng = np.random.default_rng(seed)
        if seed is not None:
            # Eryn and its prior draws use NumPy's global random state
            np.random.seed(seed)
        posterior.batch_size = ntemps * nwalkers

        self.branch_names = posterior.branch_names
        self.branch_params = {'psd': _sampled(posterior.psd_params),
                              'envelope': _sampled(posterior.envelope_params)}

        atom_bounds = [(0.0, 1.0), tuple(np.log10(width_range_days)), tuple(amp_range)]
        self.bounds = {b: [p.bounds for p in self.branch_params[b]] for b in self.branch_params}
        for b in posterior.rj_branches:
            self.bounds[b] = atom_bounds

        self.ndims = {b: len(self.bounds[b]) for b in self.branch_names}
        self.nleaves_max = {b: (self.kmax if b in posterior.rj_branches else 1) for b in self.branch_names}
        self.nleaves_min = {b: (self.kmin if b in posterior.rj_branches else 1) for b in self.branch_names}
        self.priors = {b: ProbDistContainer({i: uniform_dist(lo, hi) for i, (lo, hi) in enumerate(self.bounds[b])})
                       for b in self.branch_names}

        # Stretch moves for the fixed-dimension branches; a Gaussian random walk for
        # the atoms, since eryn's stretch move pairs leaves badly when K varies
        # (eryn's walker-count check counts every branch at its maximum leaf count,
        # so it is done here for the one branch each stretch move updates)
        moves = []
        for b in self.branch_names:
            if b in posterior.rj_branches:
                continue
            if nwalkers < 2 * self.ndims[b]:
                raise ValueError(f"Need nwalkers >= {2 * self.ndims[b]} for the '{b}' branch")
            moves.append((StretchMove(gibbs_sampling_setup=b, live_dangerously=True), 1.0))
        for b in posterior.rj_branches:
            widths = np.diff(np.array(self.bounds[b]), axis=1).ravel()
            moves.append((GaussianMove({b: np.diag((atom_step * widths) ** 2)}, gibbs_sampling_setup=b), 1.0))
        rj_moves = None
        if posterior.rj_branches:
            # 'psd' is listed with each RJ branch only so that K = 0 stays reachable:
            # eryn sets logp = -inf when a Gibbs RJ move empties every branch it runs on.
            # 'psd' has a fixed leaf count, so the move never changes it.
            rj_moves = [(DistributionGenerateRJ(self.priors, nleaves_max=self.nleaves_max,
                                                nleaves_min=self.nleaves_min, gibbs_sampling_setup={b: None, 'psd': None}), 1.0)
                        for b in posterior.rj_branches]

        with warnings.catch_warnings():
            # Eryn warns about any stretch move in an RJ run; ours only touch fixed-K branches
            warnings.filterwarnings('ignore', message='If using revisible jump')
            self.sampler = EnsembleSampler(
                nwalkers, self.ndims, posterior, self.priors,
                tempering_kwargs=dict(ntemps=ntemps),
                nbranches=len(self.branch_names), branch_names=self.branch_names,
                nleaves_max=self.nleaves_max, nleaves_min=self.nleaves_min,
                moves=moves, rj_moves=rj_moves, vectorize=True, provide_groups=True,
            )
        logger.info(f"Wavelet sampler: branches {self.branch_names}, ndims {self.ndims}, "
                    f"K in [{self.kmin}, {self.kmax}], {nwalkers} walkers x {ntemps} temperatures")

    def initial_state(self):
        """Starting coordinates: fixed branches from ``init``, RJ branches at ``K = min_wavelets``."""
        coords, inds = {}, {}
        shape = (self.ntemps, self.nwalkers)
        for b in self.branch_names:
            nmax, ndim = self.nleaves_max[b], self.ndims[b]
            coords[b] = self.priors[b].rvs(size=shape + (nmax,)).reshape(shape + (nmax, ndim))
            inds[b] = np.zeros(shape + (nmax,), dtype=bool)
            if b in self.posterior.rj_branches:
                inds[b][..., :self.kmin] = True
                continue
            inds[b][..., 0] = True
            if self.init == 'injected':
                lo, hi = np.array(self.bounds[b]).T
                centre = np.array([np.nan if p.value is None else p.value for p in self.branch_params[b]], dtype=float)
                ball = centre + self.init_scale * (hi - lo) * self.rng.normal(size=shape + (ndim,))
                ball = np.clip(ball, lo + 1e-12 * (hi - lo), hi - 1e-12 * (hi - lo))
                # Parameters without an injected value keep their prior draws
                known = np.isfinite(centre)
                coords[b][..., 0, known] = ball[..., known]
            elif self.init != 'prior':
                raise ValueError(f"Unknown init '{self.init}'")
        return coords, inds

    def run_from(self, coords, inds, nsteps, burnin=0, thin=1, progress=True):
        """Run from given coordinates and leaf indicators (see ``initial_state`` for shapes)."""
        log_prior = self.sampler.compute_log_prior(coords, inds=inds)
        log_like = self.sampler.compute_log_like(coords, inds=inds, logp=log_prior)[0]
        state = State(coords, inds=inds, log_like=log_like, log_prior=log_prior)
        self.sampler.run_mcmc(state, nsteps, burn=burnin, thin_by=thin, progress=progress)
        return self.results()

    def run(self, nsteps, burnin=0, thin=1, progress=True):
        """
        Run the sampler from ``initial_state``.

        Returns:
            dict: Chains and summaries (see ``results``).
        """
        coords, inds = self.initial_state()
        return self.run_from(coords, inds, nsteps, burnin, thin, progress)

    def results(self):
        """
        Cold-chain samples and reversible-jump summaries.

        Returns:
            dict: ``psd_chain`` (nsamples, ndim) with ``psd_names`` and
            ``psd_injected``; the same for ``envelope`` when sampled; for each RJ
            branch, ``<branch>_atoms`` (nsamples, Kmax, 3), ``<branch>_inds`` and
            ``<branch>_nleaves``, and ``<branch>_bayes`` from ``bayes_factors``;
            plus ``log_like`` and the acceptance fractions.
        """
        chain = self.sampler.get_chain()
        inds = self.sampler.get_inds()
        log_like = self.sampler.get_log_like()[:, 0]
        nsteps = log_like.shape[0]

        out = {'sampler': 'eryn', 'log_like': log_like.reshape(-1), 'nsteps': nsteps, 'nwalkers': self.nwalkers,
               'branch_names': self.branch_names, 'acceptance_fraction': self.sampler.acceptance_fraction[0]}
        if self.posterior.rj_branches:
            out['rj_acceptance_fraction'] = self.sampler.rj_acceptance_fraction[0]

        for b in ('psd', 'envelope'):
            if b in self.branch_names:
                out[f'{b}_chain'] = chain[b][:, 0, :, 0, :].reshape(-1, self.ndims[b])
                out[f'{b}_names'] = [p.label for p in self.branch_params[b]]
                out[f'{b}_injected'] = _injected(self.branch_params[b])

        for b in self.posterior.rj_branches:
            nleaves = inds[b][:, 0].sum(axis=-1)  # (nsteps, nwalkers)
            out[f'{b}_atoms'] = chain[b][:, 0].reshape(-1, self.kmax, 3)
            out[f'{b}_inds'] = inds[b][:, 0].reshape(-1, self.kmax)
            out[f'{b}_nleaves'] = nleaves.reshape(-1)
            out[f'{b}_bayes'] = bayes_factors(nleaves, self.kmin, self.kmax)
            post = out[f'{b}_bayes']['posterior']
            logger.info(f"{b}: posterior on K = " +
                        ", ".join(f"{k}:{p:.3f}" for k, p in zip(out[f'{b}_bayes']['K'], post) if p > 0))
        return out


def run_nuts(posterior, warmup=1000, samples=2000, chains=1, chain_method='parallel', adapt_matrix=True,
             init='injected', seed=0, progress=True):
    """
    Sample a fixed-dimension wavelet model (``parametric`` or ``none``) with NumPyro's NUTS.

    Priors are uniform within the source bounds, as in the chunked pipeline.

    Args:
        posterior (WaveletPosterior): Likelihood (``modulation`` must not be ``'wavelets'``).
        warmup, samples, chains (int): NUTS settings.
        chain_method (str): ``'parallel'``, ``'vectorized'`` or ``'sequential'``.
        adapt_matrix (bool): Adapt the mass matrix.
        init (str): ``'injected'`` starts at the injected values, ``'prior'`` at a uniform draw.
        seed (int): Random seed.
        progress (bool): Show the progress bar.

    Returns:
        dict: Same keys as ``WaveletSampler.results`` for the fixed-dimension branches.
    """
    import numpyro
    import numpyro.distributions as dist
    from numpyro.infer import MCMC, NUTS, init_to_uniform, init_to_value

    if posterior.modulation == 'wavelets':
        raise ValueError("A variable number of wavelets needs reversible jump: use sampler 'eryn'")
    numpyro.enable_x64()

    psd = _sampled(posterior.psd_params)
    env = _sampled(posterior.envelope_params)

    def model():
        theta_psd = jnp.stack([numpyro.sample(p.label, dist.Uniform(*p.bounds)) for p in psd])
        theta_env = jnp.stack([numpyro.sample(p.label, dist.Uniform(*p.bounds)) for p in env]) if env else jnp.zeros(0)
        numpyro.factor('log_likelihood', posterior.single(theta_psd, theta_env, (), ()))

    if init == 'injected':
        values = {p.label: p.value for p in psd + env if p.value is not None}
        strategy = init_to_value(values=values)
    elif init == 'prior':
        strategy = init_to_uniform
    else:
        raise ValueError(f"Unknown init '{init}'")

    kernel = NUTS(model, adapt_mass_matrix=adapt_matrix, init_strategy=strategy)
    mcmc = MCMC(kernel, num_warmup=warmup, num_samples=samples, num_chains=chains,
                chain_method=chain_method, progress_bar=progress)
    mcmc.run(jax.random.PRNGKey(seed))
    draws = mcmc.get_samples()

    out = {'sampler': 'NUTS', 'nsteps': samples, 'nwalkers': chains, 'branch_names': posterior.branch_names}
    out['psd_chain'] = np.column_stack([np.asarray(draws[p.label]) for p in psd])
    out['psd_names'] = [p.label for p in psd]
    out['psd_injected'] = _injected(posterior.psd_params)
    theta_env = np.zeros((len(out['psd_chain']), 0))
    if env:
        theta_env = out['envelope_chain'] = np.column_stack([np.asarray(draws[p.label]) for p in env])
        out['envelope_names'] = [p.label for p in env]
        out['envelope_injected'] = _injected(posterior.envelope_params)

    empty = posterior._empty_atoms(len(theta_env))
    out['log_like'] = np.asarray(posterior.batch(jnp.asarray(out['psd_chain']), jnp.asarray(theta_env), *empty))
    return out


def modulation_draws(posterior, results, times, nsamples=500, seed=None):
    """
    Posterior draws of ``10^(amp - amp_injected) * P_c(t)``, the galactic power
    relative to the injected amplitude, for eryn or NUTS results.

    Args:
        posterior (WaveletPosterior): Likelihood used for the run.
        results (dict): Sampler results.
        times (array): Times at which to evaluate the modulation.
        nsamples (int): Number of posterior draws.
        seed (int, optional): Random seed for choosing the draws.

    Returns:
        np.ndarray: Shape (nsamples, nchannels, ntimes).
    """
    if posterior.galaxy_source != 'galactic_DWD_time':
        raise ValueError("No modulated galactic source in the model")
    n = len(results['psd_chain'])
    idx = np.random.default_rng(seed).choice(n, size=min(nsamples, n), replace=False)

    theta_psd = jnp.asarray(results['psd_chain'][idx])
    theta_env = jnp.asarray(results['envelope_chain'][idx]) if 'envelope_chain' in results \
        else jnp.zeros((len(idx), 0))
    if posterior.rj_branches:
        atoms = tuple(jnp.asarray(results[f'{b}_atoms'][idx]) for b in posterior.rj_branches)
        masks = tuple(jnp.asarray(results[f'{b}_inds'][idx]) for b in posterior.rj_branches)
    else:
        atoms, masks = (), ()
    return np.asarray(posterior._power_batch(theta_psd, theta_env, atoms, masks, jnp.asarray(times)))
