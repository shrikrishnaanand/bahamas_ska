# Run 2: true sky envelope, wavelet RJMCMC vs the OG chunked pipeline on the same data

**Run 2.** Simulate one 30-day time series with the Galactic foreground modulated by the true sky envelope
$M_c^2(t)$, and analyse **that same series** with both pipelines: the wavelet RJMCMC (free modulation) and the OG
chunked model (5 sky parameters), both run by us. The question: **are the wavelet posteriors broader?**

* Notebook (reproduces every number and figure): [data/wavelet_injection/run2_sky_vs_og.ipynb](../data/wavelet_injection/run2_sky_vs_og.ipynb)
* Code: [injection.py](../data/wavelet_injection/injection.py) (simulation, wavelet run),
  [og.py](../data/wavelet_injection/og.py) (OG run), [data_checks.py](../data/wavelet_injection/data_checks.py)
  (input checks), [diagnostics.py](../data/wavelet_injection/diagnostics.py) (convergence)
* Outputs: `data/wavelet_injection/outputs/run2/` (`og/` and `og_dof2x/` for the two OG runs), `outputs/checks/`

**Supersedes** the first version of this run (Oct 9), which compared our wavelet posterior with Riccardo's OG posterior
on a *different* noise realisation and with sin ψ = −0.99 in the earlier stages. That OG run could not be reproduced:
the data files we received did not give back its stored likelihood values (archived, see
[data/archive/stage1-2_psi099/](../data/archive/stage1-2_psi099/)).

## Setup

Injected values: [config/sources.yaml](../data/wavelet_injection/config/sources.yaml), the paper-2 pack-2 values
([arXiv:2511.03604](https://arxiv.org/abs/2511.03604)): lat −0.0966, long −1.626, **sin ψ −0.83**, s1 0.0416,
s2 0.1384, amp −43.943, alpha 1.8, fknee −2.177, fr1 −2.429, fr2 −3.488, noise A 2.4, P 7.9.

| | OG (chunked) | wavelet |
|---|---|---|
| data | **the same series**: 30.00 d, Δt = 10 s, TDI gen 1, seed 2026, simulated with 2 extra days at each end and cut (non-periodic, like real data) | same |
| representation | two 15-day chunks, 5% Tukey taper, FFT, 1000 log bins, 0.1–29 mHz | WDM, $N_f$ = 3600 (72 pixels of 10 h), 174 log-frequency × 34 time groups, 0.1–29 mHz; 2 edge pixels (20 h) dropped at each end |
| likelihood | Gamma (bahamas `gamma_lik`); run twice, see below | Gamma |
| modulation | sky envelope, 5 parameters | free Gaussian atoms per channel, K ∈ [0, 8] by reversible jump, plus `level_E` |
| sampler | nessai, 2000 live points, 8 cores, 15 min | eryn, 32 walkers × 4 temperatures, 2000 + 6000 steps from K = 0, 40 min |

Compared quantity: the galactic power relative to the true amplitude, $10^{\log_{10}A-\log_{10}A_{\rm true}}\,P_c(t)$
(neither pipeline separates the amplitude from the modulation's mean level). Its truth is $M_c^2(t)$.

## Code flow

```
config/sources.yaml ─┐
                     ▼
injection.simulate_series('sky')  [injection.py]
   simulate_channels(N + 2·pad, t0 = −2 d, seed 2026) → cut [0, 30 d]        (non-periodic ends)
   saved: outputs/run2/timeseries.npz, data.h5 (WDM)
        │                                         │
        ▼ wavelet                                 ▼ OG
injection.run('sky', 'run2')                      og.run('run2', dof_factor = 1 | 2)   [og.py]
   WaveletLikelihood (edge_bins = 2, time_bin = 2)   chunk_spectra: 2 × 15 d, Tukey 5%, sqrt(2dt/N)·rfft
   WaveletPosterior (atoms) + eryn RJMCMC            average_log_chunks → SignalProcessor.save_data
   → outputs/run2/chains.npz                          BayesianInference (Gamma, nessai), NumPy backend
                                                      → outputs/run2/og[_dof2x]/result.hdf5
        └──────────────────┬──────────────────────────┘
                           ▼
run2_sky_vs_og.ipynb: diagnostics, p(K), OG sky posterior, spectral corner, P_c(t) bands, shape
```

## Checks before the runs (`data_checks.py`, `outputs/checks/data_checks.png`)

Ten fresh noise realisations of this injection, each compared with its exact expectation at the true parameters:

* **OG chunks need a taper.** Riccardo's chunks were simulated directly in the frequency domain, i.e. periodic. A
  15-day chunk cut from a real series is not, and the rectangular window leaks the steep low-frequency spectrum into
  the band: the binned periodogram is **36–72% too high at 0.13 mHz**, 20–36% at 0.33 mHz and ~10% at 0.5 mHz (fine
  above 0.8 mHz). A 5% Tukey taper (renormalised) brings every band back to 1 within its error; stronger tapers gain
  nothing.
* **WDM edges: 2 pixels.** The periodic WDM transform biases the first and last pixels: +10–12% at pixel 0 over the
  whole band, and up to +270% (pixel 0) and +90% (pixel 1) below 1 mHz. From the third pixel on, every pixel is
  within its error. So `edge_bins` went from 4 (1.9 d at each end, old setting) to **2 (20 h)**: the analysed span is
  now 0.8–29.2 d instead of 1.9–28.1 d.
* **The OG likelihood is consistent with our data.** The maximum-likelihood point is only 2.9 log L above the truth
  (expected ~6 for 12 parameters), with every parameter next to its injected value.

## A finding about the OG likelihood: one degree of freedom too few

bahamas' Gamma likelihood (`setting_nessai.gamma_lik`, the same in `setting_eryn` and `setting_hmc`) has shape
$\nu/2$ with $\nu$ = `count`, the number of Fourier frequencies averaged in the bin. That treats each periodogram value
$|d_k|^2$ as one degree of freedom. A complex Fourier coefficient has **two** ($|d_k|^2$ is exponential), so the bin
average has shape `count`, not `count/2`. Measured on the data: count × var(binned / model) = **1.04 ± 0.04** on the
run-2 data and 1.03–1.06 over five more realisations (1 for two degrees of freedom, 2 for one).

So the original OG likelihood assumes **twice the variance** of the binned periodogram, and its posteriors are about
**√2 too wide**. This is a property of the library, not of our setup, so it applies to the paper-2 OG widths too. We
have **not** changed the library: `og.py --dof-factor 2` doubles $\nu$ at run time. Both OG runs are shown below.

## Results

### Convergence (wavelet)

Every monitored quantity (8 spectral parameters, K per channel, log L, $P_c(t)$ at days 5, 15, 25) has an
autocorrelation time ≤ 13 steps, i.e. more than 460 τ in the 6000 stored steps and an effective sample size ≥ 15 000.
No stuck walkers. Cold acceptance 6.5%, RJ acceptance 0.7%. (τ is the larger of a Sokal-windowed ACF estimate and a
batch-means estimate; the batch means catch the slower correlations, e.g. alpha–fr1.)

p(K): A = {1: 0.26, **2: 0.40**, 3: 0.22, 4: 0.07, ≥5: 0.05}; E = {**1: 0.36**, 2: 0.28, 3: 0.23, 4: 0.09,
≥5: 0.04}. K = 0 is never visited.

### Galactic power $P_c(t)$ (`outputs/run2/modulation_og_vs_wavelet.png`)

Over the analysed span (0.8–29.2 d), 90% width relative to the truth, median offset, and how much of the span has the
truth inside the 90% band:

| channel | | OG, bahamas dof | OG, 2 dof | wavelet | wavelet / OG (bahamas) | wavelet / OG (2 dof) |
|---|---|---|---|---|---|---|
| A | width | 13.3% | 9.7% | 10.1% | 0.76 | **1.05** |
| | offset | −1.1% | −1.2% | −1.7% | | |
| | truth inside | 100% | 100% | 100% | | |
| E | width | 12.6% | 9.5% | 9.1% | 0.72 | **0.95** |
| | offset | −3.1% | −3.3% | −4.2% | | |
| | truth inside | 100% | 72% | 71% | | |

* **With the correct degrees of freedom, the two pipelines measure the galactic power equally well** (width ratio
  0.95–1.05). Against the OG likelihood as it is in bahamas, the wavelet bands look 25% narrower, but that is the √2.
* **The width differs only at the ends.** The wavelet band widens over the last ~3 days and the first ~2 (up to 17–19%
  at the very edges), where it has fewer pixels and the atoms extrapolate; the OG envelope extrapolates with the
  physical model and stays at 10–14%.
* **The centres now agree, because the data are shared.** Both pipelines put E about 3–4% low over the whole window,
  and A about 4% low after day 25. These offsets are in the data (this noise realisation), not in either method. That
  settles the old question from the first version of runs 1 and 2, where the wavelet E band sat ~4% low and we could
  not tell noise from bias. The one wavelet-only feature is A at the start: +6% at day 0, back to 0 by day 5, inside
  its band.
* E's truth lies inside the 90% band over 71–72% of the span for **both** pipelines (with correct dof): one coherent
  ~1σ fluctuation of the shared data, not a coverage problem of one method.

### Time dependence only (`outputs/run2/shape_og_vs_wavelet.png`)

Each draw divided by its mean over the analysed span, which removes the common amplitude:

| channel | OG, bahamas dof | OG, 2 dof | wavelet | wavelet / OG (2 dof) |
|---|---|---|---|---|
| A | 4.0% | 3.0% | 4.9% | **1.64** |
| E | 4.2% | 3.0% | 4.3% | **1.42** |

**Here the physical model wins.** The shape of the envelope is tied to 5 sky parameters, and both channels constrain
them together, whereas the free atoms fit each channel's shape on its own. The model-agnostic description costs
40–65% in the width of the time dependence over 30 days. The truth is inside every band except OG (2 dof) in A,
which covers it over 88% of the span.

### Spectral parameters (`outputs/run2/corner_spectral_og_vs_wavelet.png`)

| param | truth | OG (bahamas dof) median, 90% width | OG (2 dof) median, 90% width | wavelet median, 90% width | wavelet / OG (2 dof) |
|---|---|---|---|---|---|
| alpha | 1.800 | 1.817, 0.266 | 1.819, 0.197 | 1.819, 0.193 | 0.98 |
| fknee | −2.177 | −2.178, 0.0081 | −2.178, 0.0056 | −2.179, 0.0064 | 1.14 |
| fr1 | −2.429 | −2.421, 0.034 | −2.421, 0.025 | −2.420, 0.024 | 0.96 |
| fr2 | −3.488 | −3.548, 0.316 | −3.545, 0.224 | −3.543, 0.236 | 1.05 |
| A (noise) | 2.400 | 2.441, 0.158 | 2.442, 0.112 | 2.448, 0.124 | 1.11 |
| P (noise) | 7.900 | 7.910, 0.055 | 7.910, 0.038 | 7.905, 0.045 | 1.17 |

All |z| < 1.3. The medians of the three runs agree to a fraction of a σ (same data). The OG widths shrink by
0.70–0.73 ≈ 1/√2 when the dof are corrected; the wavelet widths are then within 0.96–1.17 of OG. The slightly wider
fknee, A and P are expected: the wavelet likelihood drops 40 h of edge data, and a free modulation absorbs a little of
the information that the physical envelope puts into the spectrum.

### OG sky parameters (`outputs/run2/og_sky_corner.png`)

All five within 1σ of the truth (|z| ≤ 0.9). They are broad (e.g. sin ψ ∈ [−0.96, −0.54] at 90%) and barely tighten
with the corrected dof: 30 days constrain the envelope's shape, not each sky parameter separately.

## Verdict

* **Over the data, the wavelet posterior on the galactic power is as narrow as the physical model's** (0.95–1.05 of
  OG with the correct likelihood), and the spectral parameters are within 0.96–1.17.
* **The time dependence alone is 1.4–1.6× wider** with free atoms than with the 5-parameter sky model. That is the
  real price of not assuming the sky distribution.
* **At the edges (first ~2, last ~3 days) the wavelet band is wider**, up to ~18% against ~10–14%.
* The earlier conclusion ("wavelet 6–16% broader") came from different noise realisations and an OG likelihood with
  half the degrees of freedom; it is replaced by the numbers above.

## Caveats

* One noise realisation (shared): the widths are good to about 10%. E's −3 to −4% offset is in the data.
* The dof factor of 2 assumes independent frequencies. The 5% taper correlates neighbouring frequencies slightly;
  measured count × var = 1.04–1.06, i.e. the corrected OG may be ~3% too narrow.
* 30 days only. Over a year, the sky model's rigidity (5 parameters for the whole year) and the wavelet model's freedom
  will pull further apart.

## Next steps (suggested)

1. Decide on the dof in bahamas' Gamma likelihood (fix in the library, or report OG widths with the factor).
2. Let wavelet atom centres extend beyond the window (see run 1: the +7% at the start of A).
3. Year-long run.
