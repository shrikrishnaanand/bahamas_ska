"""
Convergence diagnostics of a wavelet RJMCMC run (``outputs/<tag>/chains.npz``).

The cold-chain samples are stored flattened as (step, walker). For each monitored quantity (the spectral parameters,
K per channel, log L and P_c(t) at a few times) this estimates the integrated autocorrelation time tau (in steps) in
two ways and keeps the larger:

- the walker-averaged autocorrelation function with Sokal's automatic window (c = 5, as in emcee). Parallel-tempering
  swaps make the autocorrelation drop fast at first, and the window can then cut off a weak but long tail;
- batch means: tau = L var(means of batches of L steps) / var(x), with ``NBATCH`` batches per walker, which
  includes any slow wander longer than a batch.

ESS = nsteps * nwalkers / tau. A run is usable when nsteps > 50 tau for every quantity.

It also flags stuck walkers: walkers whose spectral parameters move in less than a third of the median fraction of
steps, or whose mean log L is more than 5 below the median walker's.
"""
import numpy as np

STUCK_MOVE_FRACTION = 1 / 3
STUCK_DLOGL = 5.0
MIN_STEPS_PER_TAU = 50
NBATCH = 20


def by_walker(flat, nwalkers):
    """(nsteps * nwalkers, ...) -> (nsteps, nwalkers, ...)."""
    return flat.reshape((-1, nwalkers) + flat.shape[1:])


def autocorr_time(x, c=5.0):
    """Integrated autocorrelation time of x (nsteps, nwalkers), walker-averaged ACF and Sokal window."""
    x = np.asarray(x, float)
    n = x.shape[0]
    if np.allclose(x.std(0), 0):
        return np.inf
    f = np.fft.rfft(x - x.mean(0), n=2 * n, axis=0)
    acf = np.fft.irfft(f * np.conj(f), axis=0)[:n]
    acf = acf / np.where(acf[0] > 0, acf[0], 1.0)
    rho = acf[:, x.std(0) > 0].mean(1)
    taus = 2 * np.cumsum(rho) - 1
    m = np.arange(n) < c * taus
    window = np.argmin(m) if not m.all() else n - 1
    return float(taus[window])


def batch_means_time(x, nbatch=NBATCH):
    """tau from the variance of batch means, pooled over walkers (x: nsteps, nwalkers)."""
    x = np.asarray(x, float)
    L = x.shape[0] // nbatch
    var = x.var()
    if L < 2 or var == 0:
        return np.inf
    means = x[:L * nbatch].reshape(nbatch, L, -1).mean(1)
    return float(L * means.var(ddof=1) / var)


def quantities(res, post=None, times=None):
    """Monitored quantities as {name: (nsteps, nwalkers)}; P_c(t) at ``times`` needs the posterior."""
    nw = int(res['nwalkers'])
    q = {name: by_walker(res['psd_chain'][:, i], nw) for i, name in enumerate(res['psd_names'])}
    for b in [k[:-len('_nleaves')] for k in res if k.endswith('_nleaves')]:
        q[f'K_{b[4:]}'] = by_walker(res[f'{b}_nleaves'].astype(float), nw)
    q['log L'] = by_walker(res['log_like'], nw)
    if post is not None and times is not None:
        P = galactic_power_all(post, res, times)          # (nsamples, nchannels, ntimes)
        for c, ch in enumerate(post.channel_names):
            for j, t in enumerate(times):
                q[f'P_{ch}({t / 86400:.0f} d)'] = by_walker(P[:, c, j], nw)
    return q


def galactic_power_all(post, res, times, batch=4096):
    """10^(amp - amp_ref) P_c(t) for every stored sample, in batches."""
    import jax.numpy as jnp
    n = len(res['psd_chain'])
    out = []
    for s in range(0, n, batch):
        sl = slice(s, min(s + batch, n))
        atoms = tuple(jnp.asarray(res[f'{b}_atoms'][sl]) for b in post.rj_branches)
        masks = tuple(jnp.asarray(res[f'{b}_inds'][sl]) for b in post.rj_branches)
        out.append(np.asarray(post._power_batch(jnp.asarray(res['psd_chain'][sl]), jnp.zeros((sl.stop - sl.start, 0)),
                                                atoms, masks, jnp.asarray(times))))
    return np.concatenate(out)


def summary(q):
    """{name: (tau_acf, tau_batch, ESS, nsteps / tau)}, with tau = max(tau_acf, tau_batch, 1)."""
    out = {}
    for name, x in q.items():
        t_acf, t_bm = autocorr_time(x), batch_means_time(x)
        tau = max(t_acf, t_bm, 1.0)
        out[name] = (t_acf, t_bm, x.size / tau, x.shape[0] / tau)
    return out


def stuck_walkers(res):
    """Indices of walkers that rarely move or sit at low log L, with the per-walker statistics."""
    nw = int(res['nwalkers'])
    chain = by_walker(res['psd_chain'], nw)                      # (nsteps, nwalkers, ndim)
    moved = np.any(np.diff(chain, axis=0) != 0, axis=-1).mean(0)  # fraction of steps with a psd move
    mean_ll = by_walker(res['log_like'], nw).mean(0)
    flag = (moved < STUCK_MOVE_FRACTION * np.median(moved)) | (mean_ll < np.median(mean_ll) - STUCK_DLOGL)
    return np.flatnonzero(flag), moved, mean_ll


def report(res, post=None, times=None):
    """Print the diagnostics table and the stuck walkers; returns (summary, stuck indices)."""
    s = summary(quantities(res, post, times))
    nsteps = int(res['nsteps'])
    print(f"{'quantity':>16s} {'tau_acf':>8s} {'tau_batch':>9s} {'ESS':>8s} {'nsteps/tau':>11s}")
    for name, (t_acf, t_bm, ess, ratio) in s.items():
        flag = '' if ratio >= MIN_STEPS_PER_TAU else f'  < {MIN_STEPS_PER_TAU}: not converged'
        print(f'{name.split(":")[-1]:>16s} {t_acf:8.1f} {t_bm:9.1f} {ess:8.0f} {ratio:11.1f}{flag}')
    stuck, moved, mean_ll = stuck_walkers(res)
    print(f'\n{nsteps} steps x {int(res["nwalkers"])} walkers; psd moves per walker: median {np.median(moved):.3f}, '
          f'min {moved.min():.3f}; stuck walkers: {list(stuck) if len(stuck) else "none"}')
    return s, stuck
