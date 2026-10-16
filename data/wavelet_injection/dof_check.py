"""
Are WDM pixels independent? Test of the diagonal approximation behind the wavelet Gamma likelihood.

The likelihood treats the nu pixels of a group (``time_bin`` time pixels x one log-frequency group) as independent
Gaussians, so that Y = sum w^2 is Gamma(nu/2, 2 sigma^2) and nu * Var(Y / E[Y]) = 2. Correlated pixels would give
a value above 2, i.e. fewer effective degrees of freedom than nu and posteriors that are too narrow.

For ``NSEEDS`` fresh noise realisations of the run-1 injection (wavelet atoms, so E[Y] at the injection is exact),
with the likelihood settings of runs 1-2:

- per group, nu * Var(Y / E[Y]) (expect 2) and the mean of Y / E[Y] (expect 1);
- per pixel row, the correlation of whitened pixels w / sigma with their neighbour in frequency (n, m+1) and in
  time (n+1, m) (expect 0).

Saves ``outputs/checks/dof_check.npz``. Run as ``python dof_check.py`` (about 4 s per realisation).
"""
import os

import numpy as np
import jax.numpy as jnp

from bahamas.psd_strain import psd_galaxy as gal
from bahamas.psd_strain import psd_noise as noise
from bahamas.wavelet.sampler import _sampled

import injection as I

NSEEDS = 100
SEED0 = 1000          # different from the run seed (2026) and the data_checks seeds
BANDS = [1e-4, 3e-4, 1e-3, 3e-3, 1e-2, 3e-2]
OUT = os.path.join(I.OUT, 'checks')


def expected(like, post, inj, data):
    """Model variance per group (c, s, g) and per analysed pixel (c, n, m) at the injection."""
    theta = jnp.asarray([p.value for p in _sampled(post.psd_params)], dtype=float)
    S_gal, S_n, values = post.spectra(theta)
    P = jnp.stack([inj.P(like.times, c) for c in data.channels])
    sig2_group = (P[:, :, None] * S_gal[None, None, :] + S_n[:, None, :]) / (2 * like.dt)

    # Per pixel: band-averaged PSD of each pixel (no grouping), P at the pixel times
    f = like.eval_freqs
    per_pixel = lambda s: np.asarray(jnp.reshape(s, (like.npix, -1)) @ like._weights)
    Sg = per_pixel(gal.galactic_foreground(f, values[post.galaxy_source], gen2=post.gen2))
    Sn = np.stack([per_pixel(noise.noise(f, values['instr_noise'], tdi=c, gen2=post.gen2)) for c in data.channels])
    Ppix = np.stack([np.asarray(inj.P(data.times[like.time_index], c)) for c in data.channels])
    sig2_pixel = (Ppix[:, :, None] * Sg[None, None, :] + Sn[:, None, :]) / (2 * like.dt)
    return np.asarray(sig2_group), sig2_pixel


def run():
    ratio, rho_f, rho_t = [], [], []
    for k in range(NSEEDS):
        x, inj = I.simulate_series('wavelet', seed=SEED0 + k)
        data = I.to_wavelet(x, inj)
        like, post = I.build_posterior(data)
        if k == 0:
            sig2_group, sig2_pixel = expected(like, post, inj, data)
            nu = np.asarray(like.nu)
        ratio.append(np.asarray(like.Y) / (nu * sig2_group))
        z = data.w[:, like.time_index][:, :, like.pixels] / np.sqrt(sig2_pixel)
        rho_f.append((z[:, :, :-1] * z[:, :, 1:]).mean(axis=(0, 1)))
        rho_t.append((z[:, :-1] * z[:, 1:]).mean(axis=(0, 1)))
        if k % 10 == 0:
            print(f"realisation {k + 1}/{NSEEDS}", flush=True)
    return {'ratio': np.array(ratio), 'nu': nu, 'group_freqs': like.group_freqs,
            'pixel_freqs': data.freqs[like.pixels], 'rho_f': np.array(rho_f), 'rho_t': np.array(rho_t)}


def report(res):
    r, nu, gf, fp = res['ratio'], res['nu'], res['group_freqs'], res['pixel_freqs']
    n = r.shape[0] * r.shape[1] * r.shape[2]
    mean = r.mean(axis=(0, 1, 2))
    q = nu * r.var(axis=(0, 1, 2), ddof=1)
    rho_f, rho_t = res['rho_f'].mean(0), res['rho_t'].mean(0)
    print(f"{r.shape[0]} realisations, {n} values per group, {len(nu)} groups (nu {nu.min():.0f}-{nu.max():.0f})")
    print(f"all groups: mean Y/E[Y] = {r.mean():.4f}, nu*Var = {q.mean():.4f}")
    print("band (mHz)    groups  nu   mean Y/E[Y]  nu*Var (2)       rho freq  rho time")
    for lo, hi in zip(BANDS[:-1], BANDS[1:]):
        g = (gf >= lo) & (gf < hi)
        p = (fp >= lo) & (fp < hi)
        err = q[g].std(ddof=1) / np.sqrt(g.sum())
        print(f"{lo * 1e3:5.2f}-{hi * 1e3:5.2f}   {g.sum():4d}  {np.median(nu[g]):4.0f}   {mean[g].mean():.4f}"
              f"      {q[g].mean():.3f} ± {err:.3f}   {rho_f[p[:-1]].mean():+.4f}   {rho_t[p].mean():+.4f}")


if __name__ == '__main__':
    os.makedirs(OUT, exist_ok=True)
    res = run()
    np.savez(os.path.join(OUT, 'dof_check.npz'), **res)
    report(res)
