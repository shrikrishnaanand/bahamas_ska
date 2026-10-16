# Run 2: true sky envelope, wavelet RJMCMC vs the OG chunked pipeline on the same data

**Run 2.** Simulate one 30-day time series with the Galactic foreground modulated by the true sky envelope
$M_c^2(t)$, and analyse **that same series** with both pipelines, both run by us:

* the wavelet RJMCMC, with a free modulation;
* the OG chunked model, with 5 sky parameters.

The question: **are the wavelet posteriors broader?**

**Redone on Oct 16, 2026** with the fixed code ([PAPER_BUGS.md](PAPER_BUGS.md)). The fixes that matter here:

* **F1, F2:** the Galactic envelope, i.e. the injected $M_c^2(t)$ and the OG model.
* **B2:** the OG Gamma likelihood now has the correct degrees of freedom by default. The paper's version is run as
  well.
* **B3b:** the A/E test-mass transfer function, which enters the noise PSD by ≤0.3% in band.

The previous version (Oct 11, pre-fix envelope) is archived in `data/archive/wavelet_runs_prefix_oct11/run2/`.

* Notebook (reproduces every number and figure): [data/wavelet_injection/run2_sky_vs_og.ipynb](../data/wavelet_injection/run2_sky_vs_og.ipynb)
* Code: [injection.py](../data/wavelet_injection/injection.py) (simulation, wavelet run),
  [og.py](../data/wavelet_injection/og.py) (OG run), [data_checks.py](../data/wavelet_injection/data_checks.py)
  (input checks), [diagnostics.py](../data/wavelet_injection/diagnostics.py) (convergence)
* Outputs: `data/wavelet_injection/outputs/run2/` (`og/` and `og_legacy/` for the two OG runs), `outputs/checks/`
* Run 1 (recovery of a modulation the atoms represent exactly): [WAVELET_INJECTION_RUN1.md](WAVELET_INJECTION_RUN1.md)

## Setup

Injected values: [config/sources.yaml](../data/wavelet_injection/config/sources.yaml), the paper-2 pack-2 values
([arXiv:2511.03604](https://arxiv.org/abs/2511.03604)):

| lat | long | sin ψ | s1 | s2 | amp | alpha | fknee | fr1 | fr2 | noise A | noise P |
|---|---|---|---|---|---|---|---|---|---|---|---|
| −0.0966 | −1.626 | **−0.83** | 0.0416 | 0.1384 | −43.943 | 1.8 | −2.177 | −2.429 | −3.488 | 2.4 | 7.9 |

| | OG (chunked) | wavelet |
|---|---|---|
| data | **the same series**: 30.00 d, Δt = 10 s, TDI gen 1, seed 2026, simulated with 2 extra days at each end and cut (non-periodic, like real data) | same |
| representation | two 15-day chunks, 5% Tukey taper, FFT, 1000 log bins, 0.1–29 mHz | WDM, $N_f$ = 3600 (72 pixels of 10 h), 174 log-frequency × 34 time groups, 0.1–29 mHz; 2 edge pixels (20 h) dropped at each end |
| likelihood | Gamma, run twice: **2 dof** per Fourier coefficient (bahamas default since B2) and the **paper's** 1 dof (`og.py --legacy-dof`, Pozzoli et al. Eq. 9) | Gamma (exact for real WDM pixels) |
| modulation | sky envelope, 5 parameters | free Gaussian atoms per channel, K ∈ [0, 8] by reversible jump, plus `level_E` |
| sampler | nessai, 2000 live points, 8 cores, 15 min | eryn, 32 walkers × 4 temperatures, 2000 + 6000 steps from K = 0, 40 min |

## Code flow

```
config/sources.yaml ─┐
                     ▼
injection.simulate_series('sky')  [injection.py]
   simulate_channels(N + 2·pad, t0 = −2 d, seed 2026) → cut [0, 30 d]        (non-periodic ends)
   saved: outputs/run2/timeseries.npz, data.h5 (WDM)
        │                                         │
        ▼ wavelet                                 ▼ OG
injection.run('sky', 'run2')                      og.run('run2', legacy = False | True)   [og.py]
   WaveletLikelihood (edge_bins = 2, time_bin = 2)   chunk_spectra: 2 × 15 d, Tukey 5%, sqrt(2dt/N)·rfft
   WaveletPosterior (atoms) + eryn RJMCMC            average_log_chunks → SignalProcessor.save_data
   → outputs/run2/chains.npz                          BayesianInference (Gamma, nessai), NumPy backend
                                                      → outputs/run2/og[_legacy]/result.hdf5
        └──────────────────┬──────────────────────────┘
                           ▼
run2_sky_vs_og.ipynb: diagnostics, p(K), OG sky posterior, spectral corner, galactic-power bands, shape
```

## Input checks

The data checks behind the settings ([data_checks.py](../data/wavelet_injection/data_checks.py),
`outputs/checks/data_checks.png`) were not rerun: they test the transforms and the windowing, which the fixes do not
touch. Their conclusions:

* **OG taper:** a 15-day chunk cut from a real series needs a taper. Without one, the binned periodogram is 36–72% too
  high at 0.13 mHz; a 5% Tukey taper fixes this.
* **WDM edges:** the WDM transform biases the first and last two 10-h pixels, which are dropped.

On the run-2 OG data, count × (binned / model − 1)² above 1 mHz is **1.039 ± 0.042**. That is 1 for two degrees of
freedom per Fourier coefficient and 2 for one, which confirms B2 on this data.

## Results

### Convergence and K (wavelet)

**Convergence.** I monitored the 8 spectral parameters, K per channel, log L, and $P_c(t)$ at days 5, 15 and 25.
Every one has an autocorrelation time ≤ 7.6 steps: more than 790 τ in the 6000 stored steps, ESS ≥ 25 000. No walkers
are stuck. Cold acceptance is 7.1% and RJ acceptance 0.7%.

**K.** K = 0 is never visited:

| channel | p(K=1) | p(K=2) | p(K=3) | p(K=4) | p(K≥5) |
|---|---|---|---|---|---|
| A | 0.39 | **0.44** | 0.14 | 0.03 | <0.01 |
| E | **0.44** | 0.34 | 0.16 | 0.05 | 0.01 |

**Exploration of K, including the burn-in** (`K_trace_burnin.png`). eryn does not store the burn-in, so the run was
repeated with the same seeds and the burn-in stored (`python injection.py --tag run2 --injection sky --k-trace`,
`outputs/run2/k_trace.npz`). The rerun reproduces `chains.npz` exactly. The true envelope is not a finite sum of
atoms, so there is no true K; the plot marks the posterior mode instead.

| channel | every walker off K = 0 from step | K range, burn-in | K range, stored | mean K, last 500 burn-in steps / stored |
|---|---|---|---|---|
| A | 44 | 0–5 | 1–6 | 1.80 / 1.81 |
| E | 3 | 0–6 | 1–6 | 1.89 / 1.86 |

Starting from K = 0, every walker has found the modulation within 44 steps. The mean K is flat from a few hundred
steps on, well inside the 2000-step burn-in, and the same at the end of the burn-in as in the stored part.

### OG sky parameters (`og_sky_corner.png`)

All five are within 1.4σ of the truth. They are broad (e.g. sin ψ ∈ [−0.95, −0.52] at 90%) and barely tighten with
the correct dof: 30 days constrain the envelope's shape, not each sky parameter separately. The largest pulls are s2
(+1.3σ) and long (−1.1σ), the same in both OG runs. They are this noise realisation, and they shape the OG
modulation bands below.

| param | truth | OG (2 dof), median [90%] | z |
|---|---|---|---|
| lat | −0.097 | −0.122 [−0.213, −0.063] | −0.55 |
| long | −1.626 | −1.725 [−1.898, −1.614] | −1.12 |
| sin ψ | −0.830 | −0.707 [−0.947, −0.521] | +0.92 |
| s1 | 0.042 | 0.030 [0.003, 0.079] | −0.48 |
| s2 | 0.138 | 0.206 [0.117, 0.278] | +1.33 |

### Spectral parameters (`corner_spectral_og_vs_wavelet.png`)

The corner plot overlays all three posteriors on the six spectral parameters the models share:

* OG with the paper's dof (`og_legacy/`, grey dashed);
* OG with 2 dof (`og/`, orange);
* wavelet (blue).

`amp` is left out because the wavelet `amp` absorbs the level $b_A$. Median and 90% width:

| param | truth | OG (paper dof) | OG (2 dof) | wavelet | wavelet / OG (2 dof) |
|---|---|---|---|---|---|
| alpha | 1.800 | 1.822, 0.257 | 1.822, 0.192 | 1.832, 0.200 | 1.04 |
| fknee | −2.177 | −2.178, 0.0077 | −2.178, 0.0054 | −2.179, 0.0061 | 1.12 |
| fr1 | −2.429 | −2.421, 0.033 | −2.421, 0.025 | −2.419, 0.025 | 1.00 |
| fr2 | −3.488 | −3.545, 0.302 | −3.543, 0.216 | −3.545, 0.236 | 1.09 |
| A (noise) | 2.400 | 2.444, 0.159 | 2.443, 0.115 | 2.443, 0.118 | 1.02 |
| P (noise) | 7.900 | 7.910, 0.054 | 7.910, 0.038 | 7.904, 0.043 | 1.13 |

* All |z| ≤ 1.35, and the three runs agree on the medians (same data).
* Going from the paper's dof to the correct one shrinks the OG widths by a factor 1.34–1.43 ≈ √2, as B2 predicts.
* The wavelet widths are within 1.00–1.13 of the correct OG widths. The slightly wider fknee and P are expected: the
  wavelet likelihood drops 40 h of edge data, and a free modulation absorbs a little information that the physical
  envelope puts into the spectrum.

### The modulation: two ways to normalise it

Run 1 showed that `amp` trades off against the spectral shape (α, fknee, fr1, fr2): the data constrain the galactic
**PSD** in the band, not `amp` on its own. Both pipelines share the spectral model, so both are affected. Two
quantities are compared, both relative to the truth $M_c^2(t)$, over the analysed span (time-group centres 1.0–28.5 d):

1. **`amp` as the normalisation** (`modulation_og_vs_wavelet.png`): $10^{\log_{10}A-\log_{10}A_{\rm true}}\thinspace P_c(t)$.
   This is the quantity of the previous version of this report.
2. **The galactic PSD × $P_c(t)$** (`modulation_power_og_vs_wavelet.png`, notebook §5b). Each draw is multiplied by its
   galactic spectral shape relative to the injected one, at a common `amp`. The shape is averaged over frequency with
   the Fisher weight of a galactic-power scale, $w_g \propto \nu_g\thinspace [S_{\rm gal}/(S_{\rm gal}+S_n)]^2$, so it is
   the galactic power where the data constrain it.

**1. `amp` as the normalisation**

| channel | | OG (paper dof) | OG (2 dof) | wavelet | wavelet / OG (2 dof) |
|---|---|---|---|---|---|
| A | 90% width | 13.0% | 9.7% | 10.1% | **1.04** |
| | median offset | −1.1% | −0.9% | −1.9% | |
| | truth inside | 100% | 100% | 97% | |
| E | 90% width | 12.5% | 9.2% | 9.6% | **1.05** |
| | median offset | −3.1% | −3.3% | −4.5% | |
| | truth inside | 94% | 73% | 72% | |

**2. Galactic PSD × $P_c(t)$**

| channel | | OG (paper dof) | OG (2 dof) | wavelet | wavelet / OG (2 dof) |
|---|---|---|---|---|---|
| A | 90% width | 6.3% | 4.8% | 5.8% | **1.20** |
| | median offset | +2.2% | +2.2% | +1.5% | |
| | truth inside | 100% | 66% | 90% | |
| E | 90% width | 6.6% | 4.6% | 5.2% | **1.12** |
| | median offset | +0.1% | −0.3% | −1.3% | |
| | truth inside | 100% | 74% | 100% | |

What the two tables show:

* **E's −3 to −5% offset comes from the `amp` normalisation.** In the first quantity both pipelines put E 3–5% low over
  the whole window. The previous report took this as the noise realisation. In the galactic PSD × $P_c$, E is at
  −0.3% (OG) and −1.3% (wavelet). The offset is the `amp`–shape trade-off, common to both pipelines because they share
  the spectral model. This agrees with run 1, where a likelihood profile put E's galactic power at 0.994 ± 0.010 of the
  injection.
* **About half of the band width in the first quantity is `amp`–shape scatter.** It cancels in the PSD, and the bands
  shrink from 9–10% to 5–6%. The two quantities measure different things. The second is the one the data constrain.
* **The `amp` metric hides some of the physical model's advantage.** On the measured power, the wavelet band is
  **12–20% wider** than OG's with the correct dof, where the `amp`-normalised comparison said 4–5%. The common
  `amp`–shape scatter dilutes the difference.
* **The widths differ mostly at the ends.** The wavelet band widens over the first ~3 and last ~4 days, up to ±10%
  at the edges, where it has fewer pixels and the atoms extrapolate. The OG envelope extrapolates with the physical
  model and stays at ±3–5%, apart from E's first days. In the middle of the window the two widths are close.
* **Coverage.** On the measured power, OG (2 dof) has the truth inside its 90% band over 66% (A) and 74% (E) of the
  span. It is excluded where its median runs +3% (A, days 3–12) and −4 to −6% (E, days 0–4). These are smooth,
  correlated excursions, produced by the sky-parameter pulls above (s2 +1.3σ, long −1.1σ), i.e. one ~1σ fluctuation
  of the shared data. The fraction of time inside the band is not a coverage probability. The wavelet band contains
  the truth over 90% (A) and 100% (E).

### Time dependence only (`shape_og_vs_wavelet.png`)

Each draw is divided by its mean over the analysed span. This removes the common amplitude, so `amp` and the spectral
shape drop out:

| channel | OG (paper dof) | OG (2 dof) | wavelet | wavelet / OG (2 dof) | truth inside: OG (2 dof) / wavelet |
|---|---|---|---|---|---|
| A | 3.8% | 2.9% | 4.3% | **1.49** | 65% / 100% |
| E | 4.2% | 3.1% | 3.9% | **1.28** | 69% / 100% |

**Here the physical model wins.** The shape of the envelope is tied to 5 sky parameters, and both channels constrain
them together, whereas the free atoms fit each channel's shape on its own. Over 30 days, the model-agnostic
description costs 30–50% in the width of the time dependence. The OG bands are also the ones that miss the truth over
a third of the span, through the same sky-parameter pulls.

## Comparison with the previous run 2 (Oct 11, pre-fix envelope)

| | Oct 11 | Oct 16 |
|---|---|---|
| injected envelope, OG model | pre-fix (F1, F2) | fixed |
| OG likelihoods | bahamas (1 dof) and `--dof-factor 2` | bahamas default = 2 dof (B2 fixed), and `--legacy-dof` = paper |
| wavelet / OG (2 dof), `amp`-normalised power, A / E | 1.05 / 0.95 | 1.04 / 1.05 |
| wavelet / OG (2 dof), shape only, A / E | 1.64 / 1.42 | 1.49 / 1.28 |
| wavelet / OG (2 dof), spectral widths | 0.96–1.17 | 1.00–1.13 |
| OG width shrink with correct dof | 0.70–0.73 | 0.70–0.75 |
| E offset, both pipelines (`amp`-normalised) | −3 to −4%, read as the noise realisation | −3 to −5%; shown to be the `amp`–shape trade-off |
| galactic PSD × $P_c$ | not computed | wavelet / OG (2 dof) 1.20 / 1.12 |

The conclusions of the previous run hold, with the fixed code:

* equal widths on the `amp`-normalised power;
* a wider time dependence with free atoms;
* the √2 from the dof.

The new metric adds two things. On the power the data actually measure, the wavelet band is 12–20% wider. And the
shared E offset is a normalisation artefact.

## Verdict

* **Galactic power, as the data measure it:** the wavelet posterior is **12–20% wider** than the physical model's
  with the correct likelihood (5.2–5.8% against 4.6–4.8%). Most of the difference is at the edges of the window.
* **Time dependence alone:** **1.3–1.5× wider** with free atoms than with the 5-parameter sky model. That is the price
  of not assuming the sky distribution over 30 days.
* **Spectral parameters:** within 1.00–1.13 of OG.
* **With the paper's likelihood (1 dof),** OG would look ≈√2 wider than it should, and the wavelet would appear to
  beat it on the power. That was the conclusion of the first comparison with Riccardo's OG run.
* **`amp`-normalised curves** should not be used to compare centres or widths. Their E offset and half of their width
  are `amp`–shape scatter.

## Caveats

* **One noise realisation (shared).** The widths are good to about 10%. The OG sky-parameter pulls (s2, long) are part
  of this realisation, and they set where the OG bands miss the truth.
* **Taper correlations.** The dof factor of 2 assumes independent frequencies. The 5% taper correlates neighbouring
  frequencies slightly (measured count × var = 1.04 ± 0.04), so the corrected OG may be ~2–4% too narrow.
* **The Fisher-weighted power is one choice of weighting.** Other weightings over the band would give somewhat
  different numbers. The weighting is fixed at the injection and is the same for both pipelines.
* **30 days only.** Over a year, the sky model's rigidity (5 parameters for the whole year) and the wavelet model's
  freedom will pull further apart.

## Next steps (suggested)

1. Let wavelet atom centres extend beyond the window (runs 1 and 2: the excess at the start of A, and the wider edges).
2. Report the galactic PSD × $P_c$ (or the shape) in place of the `amp`-normalised power.
3. Year-long run.
