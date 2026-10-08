# Wavelet-domain analysis of the galactic foreground

This document describes the wavelet-domain (WDM) analysis added to BAHAMAS as an
**optional** alternative to the chunked short-time Fourier transform (STFT) analysis of
[Pozzoli et al. 2025 (arXiv:2506.22542)](https://arxiv.org/abs/2506.22542). Nothing in the
existing chunked pipeline was replaced. The wavelet pipeline runs only when a config sets
`domain: 'wavelet'`.

The goal is to compare two ways of describing the yearly amplitude modulation of the
galactic white-dwarf foreground:

1. the **parametric sky-envelope model** used by the chunked analysis
   (`lat`, `long`, `s1`, `s2`, `psi`), and
2. a **data-driven model**: the log of the modulation is a sum of an unknown number of
   wavelet atoms, sampled with reversible-jump MCMC (RJMCMC). The posterior on the
   number of atoms gives the Bayes factor versus the number of wavelets.

Both are evaluated in the same wavelet-domain likelihood, so the comparison isolates the
modulation model. A third option (no modulation) is included as the null model.

Like the rest of BAHAMAS, everything evaluated repeatedly (the transform, the simulation
and the likelihood) is written in JAX.

---

## 1. Overview

| Step | What happens | Code |
|---|---|---|
| Simulate | One continuous time series per TDI channel: stationary noise plus a galaxy multiplied by $\sqrt{P_c(t)}$ | [simulate.py](../bahamas/wavelet/simulate.py) |
| Transform | Wilson-Daubechies-Meyer (WDM) transform to an $N_t \times N_f$ time-frequency grid | [wdm.py](../bahamas/wavelet/wdm.py) |
| Likelihood | Independent Gaussian (or binned Gamma) coefficients with variance from the model | [model.py](../bahamas/wavelet/model.py) |
| Sampling | Eryn with reversible jump over the number of atoms, or NUTS for the fixed-dimension models | [sampler.py](../bahamas/wavelet/sampler.py) |
| Pipeline | Config-driven data generation and inference, results and plots | [pipeline.py](../bahamas/wavelet/pipeline.py), [plotting.py](../bahamas/wavelet/plotting.py) |
| CLI hooks | `bahamas_data` / `bahamas_inference` dispatch on `domain: 'wavelet'` | [run_data.py:28-33](../bahamas/utilities/run_data.py#L28-L33), [run_pe.py:35-46](../bahamas/utilities/run_pe.py#L35-L46) |

The whole subpackage is [bahamas/wavelet/](../bahamas/wavelet/).

---

## 2. Physical model

For TDI channel $c$ (A or E) the data are modelled as a locally stationary process with
evolutionary power spectral density

$$
S_c(f, t) = S_{n,c}(f) + P_c(t)\, S_{\rm gal}(f),
$$

where

- $S_{n,c}(f)$ is the instrument noise, taken unchanged from the existing
  `psd_noise.noise` (parameters `A`, `P`);
- $S_{\rm gal}(f)$ is the unmodulated galactic spectrum, taken unchanged from the
  existing `psd_galaxy.galactic_foreground` (parameters `alpha`, `amp`, `fknee`, `fr1`, `fr2`);
- $P_c(t)$ is the **instantaneous power modulation** of channel $c$.

**Link to the chunked model.** The chunked analysis uses the chunk-averaged envelope
`average_envelopes_gaussian(t1, t2)`. The instantaneous power modulation is the square of
the amplitude envelope returned by `psd_response.modulation.envelopes_gaussian`:

$$
P_c(t) = \big[\text{envelopes\_gaussian}(t)_c\big]^2,
\qquad
\frac{1}{t_2 - t_1}\int_{t_1}^{t_2} P_c(t)\,dt = \text{average\_envelopes\_gaussian}(t_1, t_2).
$$

The second identity was checked numerically to a relative precision of $10^{-7}$ for several
chunks and both channels
([test_power_modulation_averages_to_chunk_envelope](../tests/test_wavelet_model.py#L52)). The
wavelet model with the parametric envelope therefore describes **the same physics** as the
chunked model, just without chunk averaging. Code:
[simulate.power_modulation](../bahamas/wavelet/simulate.py#L68).

### Simulated data

The chunked pipeline draws each chunk independently in the frequency domain. A wavelet
analysis needs one continuous series, so
[simulate_channels](../bahamas/wavelet/simulate.py#L144) draws

$$
x_c(t) = n_c(t) + \sqrt{P_c(t)}\; g_c(t),
$$

with $n_c$ and $g_c$ independent stationary Gaussian processes with PSDs $S_{n,c}$ and
$S_{\rm gal}$ ([stationary_gaussian](../bahamas/wavelet/simulate.py#L41)). The modulation is
evaluated on an hourly grid and interpolated (error $\lesssim 10^{-6}$), because the
envelope formula is expensive at the 10 s sampling rate. Random numbers use `jax.random`
keys, so simulations are reproducible from a seed.

Supported sources are `instr_noise`, `galactic_DWD_time` (modulated) and `galactic_DWD`
(stationary). The template `template/pe_galaxy_cyclo.yaml` works as is.

---

## 3. The WDM transform

**Code:** [wdm.py](../bahamas/wavelet/wdm.py).

The WDM basis (Necula et al. 2012; Cornish 2020,
[arXiv:2009.00043](https://arxiv.org/abs/2009.00043)) tiles the time-frequency plane with
$N_t$ time bins of width $\Delta T = N_f\,\Delta t$ and $N_f$ frequency bins of width
$\Delta F = 1/(2 N_f \Delta t)$. It is the representation used for the LISA galactic
foreground by Digman & Cornish 2022 ([arXiv:2212.04600](https://arxiv.org/abs/2212.04600)).

- The Meyer window $\tilde\phi(\omega)$ (Cornish 2020, Eq. 11) is
  [phitilde](../bahamas/wavelet/wdm.py#L42). It depends only on the grid, so it is computed
  once ([_phif](../bahamas/wavelet/wdm.py#L73)) and enters the compiled transform as a constant.
- The forward transform from the frequency domain (Cornish 2020, Eq. 13) is
  [_forward_freq](../bahamas/wavelet/wdm.py#L102), jit-compiled with the grid shape as static
  arguments. The inverse is [_inverse_freq](../bahamas/wavelet/wdm.py#L128). Public wrappers:
  [forward_time](../bahamas/wavelet/wdm.py#L194), [inverse_time](../bahamas/wavelet/wdm.py#L212),
  [forward_freq](../bahamas/wavelet/wdm.py#L158), [inverse_freq](../bahamas/wavelet/wdm.py#L178).
- The normalisation makes the transform **orthonormal**: $\sum w^2 = \sum x^2$. Then white
  noise of variance $\sigma^2$ gives coefficients of variance $\sigma^2$, and a stationary
  one-sided PSD $S(f)$ gives coefficient variance $S(f_m)/(2\Delta t)$.
- Both $N_f$ and $N_t$ must be even (the DC and Nyquist bins share row $m = 0$).

The transform was implemented here rather than taken from a package, and validated against
[`pywavelet`](https://pypi.org/project/pywavelet/), the WDM implementation used in the LISA
community (Section 7).

### Band-averaging the PSD

When the PSD changes appreciably across one pixel (steep spectra, low frequencies), the
variance of pixel $m$ is not $S(f_m)$ but the PSD averaged over the squared spectral window
of the pixel:

$$
\sigma^2_m = \frac{1}{2\Delta t}\sum_j W_j\, S(f_m + \delta_j \Delta F),
\qquad W_j \propto |\tilde\phi(\delta_j \Delta F)|^2,\quad \sum_j W_j = 1 .
$$

Code: [band_weights](../bahamas/wavelet/wdm.py#L227). The test
[test_band_average_matches_coloured_noise](../tests/test_wdm.py#L85) shows this is needed: for
an $f^{-4}$ spectrum the band-averaged prediction fits simulated data with
$\chi^2 \approx N_{\rm dof}$, while evaluating the PSD at the pixel centre is rejected.

---

## 4. Wavelet-domain likelihood

**Code:** [WaveletLikelihood](../bahamas/wavelet/model.py#L147).

For locally stationary noise the WDM coefficients are, to a good approximation,
independent Gaussians (the standard diagonal approximation, Cornish 2020):

$$
w_{c,nm} \sim \mathcal N\!\left(0,\ \sigma^2_{c,nm}\right),\qquad
\sigma^2_{c,nm} = \frac{P_c(t_n)\,\langle S_{\rm gal}\rangle_m + \langle S_{n,c}\rangle_m}{2\Delta t}.
$$

**Coarse graining (optional).** Adjacent frequency pixels can be grouped in log-spaced bins
([frequency_groups](../bahamas/wavelet/model.py#L120), config `freq_bins`), and consecutive
time pixels in blocks (`time_bin`). Within a group $g$ of $\nu_g$ pixels with a common
variance, $Y_g = \sum w^2$ is a sufficient statistic and $Y_g/\sigma_g^2 \sim \chi^2_{\nu_g}$,
which gives a Gamma likelihood:

$$
\ln \mathcal L = \sum_{c,n,g}\Big[\big(\tfrac{\nu_g}{2}-1\big)\ln Y - \tfrac{\nu_g}{2}\ln(2\sigma^2) - \ln\Gamma\big(\tfrac{\nu_g}{2}\big) - \frac{Y}{2\sigma^2}\Big].
$$

This is the wavelet analogue of the coarse-grained Gamma likelihood of the chunked pipeline.
With one pixel per group it equals the exact Gaussian likelihood up to a data-only constant
([test_binned_likelihood_matches_pixel_likelihood_differences](../tests/test_wavelet_model.py#L124)).
Time binning is justified because the modulation varies over months while one time pixel is
hours long ([test_time_binning](../tests/test_wavelet_model.py#L143)). Code:
[log_likelihood](../bahamas/wavelet/model.py#L227) and
[band_average](../bahamas/wavelet/model.py#L210) (pure JAX).

**Edges.** The FFT-based transform is periodic, so the first and last time pixels mix the
end of the series with its start. `edge_bins` pixels are dropped at each end (default 4).

---

## 5. Modulation models

Selected with `inference.modulation`. Code: [model.py:247-326](../bahamas/wavelet/model.py#L247-L326)
and [WaveletPosterior.power_modulation](../bahamas/wavelet/sampler.py#L206).

| `modulation` | $P_c(t)$ | Parameters |
|---|---|---|
| `none` | $1$ (stationary galaxy, null model) | spectral only |
| `parametric` | sky envelope of the chunked analysis, [parametric_modulation](../bahamas/wavelet/model.py#L252) | `lat`, `long`, `s1`, `s2`, `psi` (those with `bounds` in the sources file) |
| `wavelets` | $\ln P_c(t) = b_c + \sum_{k=1}^{K_c} a_k\, \psi\!\big((t - t_k)/\tau_k\big)$, [WaveletModulation](../bahamas/wavelet/model.py#L268) | $K_c$ atoms per channel, each $(t_k, \log_{10}\tau_k, a_k)$ |

Details of the wavelet model:

- Each atom has a centre $t_k = t_0 + t_{\rm frac}\,T$ with $t_{\rm frac} \in [0, 1]$, a
  width $\tau_k$ with a log-uniform prior (`width_range_days`, default 5–180 days) and an
  amplitude $a_k$ in $\ln P$ (`amp_range`, default $[-3, 3]$).
- The atom shape $\psi$ is `gaussian` (default) or `ricker` (Mexican-hat wavelet, zero mean).
- Working in $\ln P$ keeps $P_c(t) > 0$.
- The constant level $b_c$ is fixed to 0 in the first channel, because it is degenerate with
  the galactic amplitude `amp`, and free in the others (`level_range`).
- **Only the product $10^{\rm amp}P_c(t)$ is identifiable** in the wavelet model, since atoms
  can trade off against `amp`. Plots and comparisons therefore use
  $10^{\,\rm amp - amp_{inj}}\,P_c(t)$
  ([galactic_power](../bahamas/wavelet/sampler.py#L225)).
- The atoms are passed to JAX padded to `max_wavelets` with a boolean mask, so array shapes
  are fixed under `jit`/`vmap`. Inactive slots are replaced by a zero-amplitude atom before
  any arithmetic ([model.py:318](../bahamas/wavelet/model.py#L318)), because eryn stores NaN in
  inactive slots.

---

## 6. Sampling

**Code:** [sampler.py](../bahamas/wavelet/sampler.py).

### The likelihood as one JAX function

[WaveletPosterior](../bahamas/wavelet/sampler.py#L98) maps the sources file to parameters and
builds a single pure function
[single(theta_psd, theta_env, atoms, masks)](../bahamas/wavelet/sampler.py#L216). It computes the
band-averaged spectra ([spectra](../bahamas/wavelet/sampler.py#L183)) using the existing
BAHAMAS spectral functions, then the modulation, then the likelihood. It is jit-compiled and
vectorised over walkers with `vmap` (`WaveletPosterior.batch`), and it is differentiable
(gradient checked against finite differences in
[test_jit_vmap_and_grad_agree](../tests/test_wavelet_model.py#L186)).

An `optimization_barrier` between the spectra and the per-pixel loop
([sampler.py:222](../bahamas/wavelet/sampler.py#L222)) stops XLA from fusing the expensive
spectral evaluation into the pixel loop, which otherwise recomputes it for every time bin.
On jax 0.4.30, the version pinned in `requirements.txt`, the function is not public and the
barrier is skipped ([sampler.py:43](../bahamas/wavelet/sampler.py#L43)). Results are identical,
just slower.

### Eryn with reversible jump (`sampler: 'eryn'`)

[WaveletSampler](../bahamas/wavelet/sampler.py#L328). Eryn branches:

- `psd`: galaxy spectral parameters, noise parameters and the channel levels $b_c$ (one leaf);
- `envelope`: sky-envelope parameters, for `parametric` with sampled sky parameters (one leaf);
- `mod_A`, `mod_E`: one reversible-jump branch per channel whose leaves are wavelet atoms,
  $K_c \in$ [`min_wavelets`, `max_wavelets`].

Moves:

- stretch moves on the fixed-dimension branches;
- a Gaussian random walk on the atoms;
- birth/death moves (`DistributionGenerateRJ`) per channel, proposing new atoms from the prior.

Eryn runs in vectorised mode ([sampler.py:409](../bahamas/wavelet/sampler.py#L409)): each
likelihood request becomes one compiled batch call. The leaves of all walkers are scattered
into padded arrays by [_pad_leaves](../bahamas/wavelet/sampler.py#L85), and the batch is padded
to a fixed size so it compiles once ([WaveletPosterior.\_\_call\_\_](../bahamas/wavelet/sampler.py#L238)).

Two eryn details were needed to make the reversible jump correct, both found by testing:

1. **$K = 0$ must be reachable.** Eryn sets the log prior to $-\infty$ when a Gibbs-restricted
   RJ move empties every branch it acts on. Each RJ move is therefore grouped with the `psd`
   branch, which has a fixed leaf count and is never changed by it
   ([sampler.py:398](../bahamas/wavelet/sampler.py#L398)). Without this, $K = 0$ was never
   visited and every Bayes factor against it was wrong.
2. **The prior on $K$ is uniform.** Eryn's birth/death move includes the edge corrections at
   $K_{\min}$ and $K_{\max}$. This was checked with a flat likelihood
   ([test_prior_only_k_is_uniform](../tests/test_wavelet_rj.py#L32)).

### Bayes factor versus number of wavelets

With a uniform prior on $K$, the Bayes factor between two values of $K$ is their posterior
odds, read directly from the RJ chain:

$$
\ln \mathcal B(K, K_{\rm ref}) = \ln\frac{p(K\mid d)}{p(K_{\rm ref}\mid d)} ,
$$

with $K_{\rm ref}$ the most probable value. Errors come from splitting the chain into
consecutive blocks; values of $K$ never visited get an upper limit. Code:
[bayes_factors](../bahamas/wavelet/sampler.py#L279), plotted by
[plot_bayes_factors](../bahamas/wavelet/plotting.py#L41).

### NUTS (`sampler: 'NUTS'`)

[run_nuts](../bahamas/wavelet/sampler.py#L495) samples the fixed-dimension models (`parametric`,
`none`) with NumPyro's NUTS, as in the chunked pipeline (uniform priors within the source
bounds). It uses gradients of the same JAX likelihood. NUTS cannot change the number of
atoms, so `wavelets` requires eryn.

---

## 7. Validation

The tests are in [tests/test_wdm.py](../tests/test_wdm.py),
[tests/test_wavelet_model.py](../tests/test_wavelet_model.py) and
[tests/test_wavelet_rj.py](../tests/test_wavelet_rj.py) (46 tests, about 2 minutes). They pass
with the latest packages (jax 0.6.2) and with the versions pinned in `requirements.txt`
(jax 0.4.30, numpy 1.26, scipy 1.13).

**Transform**

| Check | Test |
|---|---|
| Orthonormal ($\sum w^2 = \sum x^2$) and exactly invertible (error $\sim 10^{-15}$), several grid shapes | [test_orthonormal_and_invertible](../tests/test_wdm.py#L11) |
| Agrees with `pywavelet` coefficient by coefficient (theirs is $\sqrt 2\times$ ours, a normalisation convention); checked with `pywavelet` 0.2.6, 0.2.7 and 0.2.10 | [test_matches_pywavelet](../tests/test_wdm.py#L35) |
| White noise gives unit-variance coefficients in every frequency row | [test_white_noise_variance](../tests/test_wdm.py#L57) |
| A sine-Gaussian lands in the right time-frequency pixel | [test_localisation](../tests/test_wdm.py#L68) |
| Band-averaged PSD predicts coloured-noise variances ($\chi^2$ test) | [test_band_average_matches_coloured_noise](../tests/test_wdm.py#L85) |

**Physics and likelihood**

| Check | Test |
|---|---|
| Time average of $P_c(t)$ equals the chunked-analysis envelope ($10^{-7}$) | [test_power_modulation_averages_to_chunk_envelope](../tests/test_wavelet_model.py#L52) |
| Simulated noise has the input PSD (Welch estimate, `scipy`) | [test_stationary_gaussian_matches_psd](../tests/test_wavelet_model.py#L61) |
| Modulated noise has pixel variances that follow $P(t)$ | [test_modulated_noise_variance_tracks_modulation](../tests/test_wavelet_model.py#L79) |
| At the injected parameters, $w^2/\sigma^2$ has mean 1 and variance 2 ($\chi^2_1$) on a full simulated year | [test_whitened_coefficients](../tests/test_wavelet_model.py#L113) |
| Gamma (binned) likelihood equals the Gaussian one up to a constant | [test_binned_likelihood_matches_pixel_likelihood_differences](../tests/test_wavelet_model.py#L124) |
| The injected envelope beats a stationary galaxy and a wrong sky position by $\Delta\ln\mathcal L > 100$ | [test_likelihood_prefers_injected_modulation](../tests/test_wavelet_model.py#L172) |
| `jit`, `vmap` and `grad` agree with plain evaluation and finite differences | [test_jit_vmap_and_grad_agree](../tests/test_wavelet_model.py#L186) |
| Eryn's batched interface gives the same values as single-walker evaluation | [test_eryn_vectorised_interface](../tests/test_wavelet_model.py#L219) |
| A short NUTS run brackets the injected values | [test_nuts_recovers_amplitude](../tests/test_wavelet_model.py#L249) |

**Reversible jump**

| Check | Test |
|---|---|
| Flat likelihood: $p(K)$ uniform over $0..4$, $\ln\mathcal B \approx 0$ | [test_prior_only_k_is_uniform](../tests/test_wavelet_rj.py#L32) |
| Factorised likelihood with a closed-form answer $p(K) \propto c^K$: reproduced to $\sim 1\%$ | [test_rj_matches_analytic_k_posterior](../tests/test_wavelet_rj.py#L51) |
| Data generated with known $K$ (0 and 2): fewer atoms excluded, more atoms disfavoured | [test_posterior_prefers_true_number_of_wavelets](../tests/test_wavelet_rj.py#L115) |
| Started with no atoms, the sampler finds both injected features | [test_rj_detects_modulation_from_scratch](../tests/test_wavelet_rj.py#L132) |

**End to end.** The quick config (one year, $\Delta t = 1000$ s, A and E) runs through the
command-line tools in about 1.5 minutes on a laptop. The reconstructed modulation contains
the injected one within the 90% band over the whole year in both channels, and $K \le 1$ is
excluded. Recomputing $\ln\mathcal L$ from the stored chain reproduces eryn's stored values
exactly.

---

## 8. How to run

```bash
uv sync --extra test          # or: pip install -e ".[test]"
pytest tests                  # about 2 minutes

cd template
bahamas_data      --config config_wavelet_quick.yaml --sources pe_galaxy_cyclo.yaml
bahamas_inference --config config_wavelet_quick.yaml --sources pe_galaxy_cyclo.yaml
```

Templates:

- [template/config_wavelet_quick.yaml](../template/config_wavelet_quick.yaml): laptop-sized, a
  few minutes. One year at $\Delta t = 1000$ s analysed in $[10^{-4}, 4.9\times10^{-4}]$ Hz.
- [template/config_wavelet.yaml](../template/config_wavelet.yaml): realistic. One year at
  $\Delta t = 10$ s, $[10^{-4}, 0.029]$ Hz, $N_f = 4096$ ($\Delta F = 1.2\times10^{-5}$ Hz,
  $\Delta T = 11.4$ h), 200 log frequency bins, `time_bin: 4`. Every option is commented.

Outputs (in `folder_plot` and `inference.file_post`):

| File | Content |
|---|---|
| `result_*.npz` | chains (`psd_chain`, `envelope_chain`, `mod_<ch>_atoms/_inds/_nleaves`), `log_like`, Bayes factors (`mod_<ch>_bayes_*`), modulation quantiles |
| `wavelet_bayes_factor_*.png` | posterior on $K$ and $\ln\mathcal B$ versus $K$, per channel |
| `wavelet_modulation_*.png` | reconstructed $10^{\Delta\rm amp}P_c(t)$ with 50%/90% bands and the injection |
| `wavelet_k_trace_*.png` | mean $K$ versus step, to check convergence |
| `wavelet_corner_*.png` | corner plots of the spectral (and envelope) parameters |
| `wavelet_data_*.png` | wavelet power of the data |

Main options, in the `wavelet:` and `inference:` sections of the config:

| Option | Meaning |
|---|---|
| `domain: 'wavelet'` | selects this pipeline (default `'stft'`, the chunked one) |
| `wavelet.Nf` | number of WDM frequency bins (must be even) |
| `wavelet.freq_bins`, `wavelet.time_bin` | coarse graining (`null` / `1` = per pixel) |
| `wavelet.edge_bins` | time pixels dropped at each end |
| `inference.modulation` | `wavelets`, `parametric` or `none` |
| `inference.sampler` | `eryn` (any model) or `NUTS` (`parametric`, `none`) |
| `inference.min_wavelets`, `max_wavelets` | range of $K$ per channel |
| `inference.width_range_days`, `amp_range`, `level_range` | priors of the wavelet model |
| `inference.ntemps` | parallel-tempering temperatures (helps the RJ mix) |

---

## 9. Performance

Measured on a laptop CPU (WSL):

- Realistic config, data generation: about 15 s, about 1.1 GB peak memory (the cost of JAX's
  random numbers, FFTs and interpolation on a $3\times10^6$-sample series).
- Realistic config, likelihood with `time_bin: 4`: about 1.3–1.5 ms per walker in a batch of
  64 walkers, compared with 4–11 ms per walker for the first (NumPy) version. A full eryn run
  (32 walkers, 2 temperatures, 5000 steps) is therefore roughly 45 minutes.
- On CPU, JAX is not much faster than NumPy for the per-pixel arithmetic itself. The gains are
  from compiling the whole likelihood, from batching walkers, from gradients (NUTS), and from
  running on a GPU.

---

## 10. Limitations and open points

1. **RJ mixing from scratch.** New atoms are proposed from the prior, and there is no
   split/merge move. Started from $K = 0$, walkers can build one feature out of several
   partial atoms and stay at too-high $K$ for a long time. Started from the truth, the
   posterior correctly prefers the true $K$ (tests above). Use tempering (`ntemps`), long
   runs, and check `wavelet_k_trace_*.png`. Data-driven birth proposals (as in BayesWave) are
   the natural improvement.
2. **$K$ is only well defined where the galaxy dominates.** Where the galaxy is below the
   noise, negative or narrow atoms barely change the total variance and cost almost nothing,
   so the posterior on $K$ flattens. This is the correct posterior, not a sampler problem. In
   the quick config's band the galaxy is at most about 0.3 of the noise. In the realistic band
   it dominates between roughly $3\times10^{-4}$ and $3\times10^{-3}$ Hz.
3. **Spectral shape in the quick config.** Its band lies below the galaxy's knee
   ($\sim 2.5\times10^{-3}$ Hz), so `alpha`, `fknee`, `fr1` and `fr2` are essentially
   unconstrained there. NUTS then takes very deep trees (about 10 minutes for 800
   iterations). Fix those parameters (no `bounds`) for quick runs.
4. **Diagonal likelihood.** WDM coefficients are treated as independent. This is standard
   for locally stationary noise and is supported by the whitening test, but correlations
   between neighbouring pixels are neglected.
5. **The wavelet pipeline needs the JAX backend.** With `--use_jax False` it stops with a
   clear error. The chunked pipeline is unaffected.

## 11. Next step: comparison with the chunked analysis

The paper's chunked results can be compared directly with three runs on the same injection:

1. `modulation: parametric` in the wavelet domain: same physics as the paper, different data
   representation;
2. `modulation: wavelets`: data-driven modulation, with the Bayes factor versus $K$;
3. `modulation: none`: the stationary null model.

A useful addition would be to generate the chunked (STFT) data from the **same continuous
time series** as the wavelet data, so that both analyses see the same noise realisation.

---

## 12. Other changes outside `bahamas/wavelet/`

| File | Change |
|---|---|
| [bahamas/utilities/run_data.py](../bahamas/utilities/run_data.py#L28-L33) | dispatch to the wavelet data generation when `domain: 'wavelet'` |
| [bahamas/utilities/run_pe.py](../bahamas/utilities/run_pe.py#L35-L46) | same for inference; the wavelet package is only imported for wavelet configs |
| [bahamas/psd_response/modulation.py](../bahamas/psd_response/modulation.py#L8-L12) | `envelopes_gaussian` uses the backend `jnp` (JAX or NumPy, like `average_envelope.py`) so it can be traced; results unchanged |
| [pyproject.toml](../pyproject.toml), [setup.py](../setup.py) | register `bahamas.wavelet`; optional `test` extra (`pytest`, `pywavelet`); `requires-python >= 3.10` (needed by `eryn`, which was already a dependency, and by jax; with `>= 3.8` `uv lock` fails, also before these changes) |
| [requirements.txt](../requirements.txt) | add `eryn` and `pywavelet==0.2.7` (0.2.8+ needs a newer astropy than the one pinned) |
| [README.md](../README.md) | short section pointing to this option |
| [template/config_wavelet.yaml](../template/config_wavelet.yaml), [template/config_wavelet_quick.yaml](../template/config_wavelet_quick.yaml) | new templates |

## References

- F. Pozzoli et al., *Bahamas: BAyesian inference with HAmiltonian Montecarlo for
  Astrophysical Stochastic background*, [arXiv:2506.22542](https://arxiv.org/abs/2506.22542)
- N. J. Cornish, *Time-frequency analysis of gravitational wave data*,
  Phys. Rev. D 102, 124038 (2020), [arXiv:2009.00043](https://arxiv.org/abs/2009.00043)
- M. C. Digman, N. J. Cornish, *LISA gravitational wave sources in a time-varying galactic
  stochastic background*, [arXiv:2212.04600](https://arxiv.org/abs/2212.04600)
- V. Necula, S. Klimenko, G. Mitselmakher, *Transient analysis with fast Wilson-Daubechies
  time-frequency transform*, J. Phys. Conf. Ser. 363, 012032 (2012)
- M. L. Katz et al., Eryn: [github.com/mikekatz04/Eryn](https://github.com/mikekatz04/Eryn)
- `pywavelet`: [pypi.org/project/pywavelet](https://pypi.org/project/pywavelet/)
