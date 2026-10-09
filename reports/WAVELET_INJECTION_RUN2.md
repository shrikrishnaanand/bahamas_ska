# Run 2: true sky envelope, wavelet RJMCMC vs Riccardo's chunked (OG) posterior

**Stage 3, run 2** of the cyclo / Fig. 7 project. Inject the signal that Riccardo's pack-2 chunked run analysed (the GF
modulated by the true sky envelope $M_c^2(t)$, true amplitude) and analyse it with the wavelet RJMCMC exactly as in
[run 1](WAVELET_INJECTION_RUN1.md). Then overlay the wavelet posterior on $P_c(t)$ with the OG posterior and ask:
**are the wavelet posteriors broader?**

* Notebook (reproduces every number and figure): [data/wavelet_injection/run2_sky_vs_og.ipynb](../data/wavelet_injection/run2_sky_vs_og.ipynb)
* Driver code: [data/wavelet_injection/injection.py](../data/wavelet_injection/injection.py) (`python injection.py --tag run2 --injection sky`)
* Outputs: `data/wavelet_injection/outputs/run2/` (data, chains, log, diagnostics) and `outputs/comparison/` (OG vs wavelet figures)
* Folder layout: [data/wavelet_injection/README.md](../data/wavelet_injection/README.md)

## Setup

| | OG (Riccardo) | wavelet (run 2) |
|---|---|---|
| data | 30 d, Δt = 10 s, 0.1–29 mHz, TDI gen 1 | same |
| representation | two 15-day chunks, 1000 log-binned periodogram frequencies | WDM, ΔT = 11.4 h, 178 log-frequency groups × 27 time groups (2.1–26.8 d) |
| likelihood | Gamma (coarse-grained) | Gamma (coarse-grained WDM power) |
| modulation model | sky envelope $M_c^2(t)$: 5 parameters (lat, long, psi, s1, s2) | free wavelet atoms per channel, K ∈ [0, 8] by reversible jump, plus `level_E` |
| sampler | nessai, 2000 live points | eryn, 32 walkers × 4 temperatures, 2000 + 6000 steps, start at K = 0 |
| noise realisation | Riccardo's | new: same seed as run 1 (the raw OG time series isn't available, and the wavelet pipeline needs one) |
| run | **reused** from `cyclo_riccardo/cyclo/result.hdf5` | 29.5 min on 8 CPU cores |

**Compared quantity.** Neither pipeline can separate the amplitude from the modulation's overall level, so both are
shown as the galactic power relative to the true amplitude, $10^{\log_{10}A-\log_{10}A_{\rm true}}\,P_c(t)$. Its truth is
$M_c^2(t)$.
* OG: $10^{\rm amp-amp_{true}}\,M_c^2(t;\text{sky})$ for 1000 nessai samples.
* Wavelet: `modulation_draws` (power relative to `amp_eff` = −43.267) × $10^{\rm amp_{eff}-amp_{true}}$, 1000 samples.

## Code flow

```
pe_pack2_cyclo.yaml ── og_sources / sky_params / true_amp [injection.py:64-76]
        │
        ▼  simulate('sky') [:138] (via load_or_simulate [:212])
Injection('sky') [:99]: P = power_modulation(sky, t, c) = M_c^2(t), amp = −43.943
simulate_channels(..., modulation=None)  → the simulator builds M_c^2 from the sky parameters itself
   same seed (2026) as run 1 ⇒ identical noise and galaxy draws; the injections differ by < 0.9%
WaveletData → outputs/run2/data.h5
        │
        ▼  build_posterior [:165] / make_sampler [:181] / run [:220]   (identical to run 1)
outputs/run2/chains.npz
        │
        ▼  run2_sky_vs_og.ipynb
§2 convergence + p(K)                                     → outputs/run2/convergence_K.png
§3 OG: result.hdf5 posterior_samples → 10^(amp−amp_true)·M_c^2(t)        (1000 draws, 300 times)
   wavelet: modulation_draws(post, chains, t) × 10^(amp_eff − amp_true)  (1000 draws, 300 times)
§4 shared spectral parameters, OG vs wavelet               → comparison/corner_spectral_og_vs_wavelet.png
§5 P_c(t) bands on top of each other + 90% width / truth  → comparison/modulation_og_vs_wavelet.png
   OG offset diagnostic (chunk means, sin ψ test, alpha at bound)
   shape only: each draw / its mean over the analysed span → comparison/shape_og_vs_wavelet.png
```

## Results

### Wavelet recovery (run 2 on its own)

As in run 1. K = 0 is never visited. p(K): A = {1: 0.29, **2: 0.39**, 3: 0.21, 4: 0.08, ≥5: 0.03}; E = {**1: 0.56**,
2: 0.29, 3: 0.11, 4: 0.04}. The chains are stationary (cold acceptance 7.8%, RJ acceptance 0.7%). The injected
$M_c^2(t)$ lies inside the wavelet 90% band at 100% (A) and 98% (E) of the analysed span.

### Which posterior is broader? (`comparison/modulation_og_vs_wavelet.png`)

Median 90% width over the analysed span (2.1–26.8 d), relative to the truth:

| quantity | channel | OG | wavelet | wavelet / OG |
|---|---|---|---|---|
| galactic power $10^{\Delta A}P_c(t)$ | A | 10.9% | 11.5% | **1.06** |
| | E | 10.3% | 10.5% | **1.01** |
| shape only $P_c(t)/\langle P_c\rangle$ | A | 3.8% | 5.1% | **1.35** |
| | E | 3.9% | 3.6% | **0.94** |

* **Galactic power: the same width.** The wavelet band is only 1–6% wider than the OG band. Most of each band is the
  common ~10% uncertainty on the overall power level (amplitude × mean modulation). Both pipelines see the same
  30 days in the same band, so this part is the same.
* **Time dependence: comparable.** Once the level is divided out, the wavelet constraint is 35% wider in A and 6%
  narrower in E. The A envelope rises steeply over the 30 days (×3). The 5 sky parameters tie its curvature to E's,
  but the free atoms have to fit A's rise on their own. E is a shallow bowl, and here the wavelet time resolution
  (27 time groups vs 2 chunks) compensates for the extra freedom.
* **The edges differ.** Outside the analysed span (the 1.9-d blocks the WDM likelihood drops at each end), the wavelet
  band widens to 18–24%. The OG model extrapolates with the physical envelope and stays at about 11–12%. Within the
  data, the cost of the model-agnostic description is small.

**Answer:** over the data the wavelet posteriors are **not noticeably broader** for the galactic power (+1 to +6%).
For the time dependence alone they are up to 35% broader in A. They are broader only where the wavelet model has no
data (the dropped edges).

### Spectral parameters (`comparison/corner_spectral_og_vs_wavelet.png`)

90% widths, wavelet / OG: alpha 1.17, fknee 0.88, fr1 1.22, fr2 0.74, A 0.87, P 0.72. They are similar, and part of
the scatter comes from the different noise realisations. The wavelet medians are all close to the truth. **The OG
`alpha` posterior is pushed against its prior bound of 2.0** (median 1.927, 35% of samples above 1.95; truth 1.80),
so the OG alpha width is truncated by the prior.

### The OG band misses the truth

The OG band contains the injected $M_c^2(t)$ at **0%** of the analysed span: it sits 6–10% high everywhere, while its
shape matches the truth. The chunk averages, which are what a chunked run measures directly, show the same thing:

| | OG $10^{\Delta A}\bar M_c$ / truth [5%, median, 95%] | if sin ψ = −0.83 instead of −0.99 |
|---|---|---|
| A, chunk 1 | [1.046, 1.102, 1.164] | 1.176 |
| A, chunk 2 | [1.015, 1.064, 1.119] | 1.100 |
| E, chunk 1 | [1.034, 1.087, 1.146] | 1.095 |
| E, chunk 2 | [1.042, 1.090, 1.144] | 1.070 |

Every chunk excludes the truth at 90%. This is a coherent ~2–3σ excess, plus `alpha` at its bound. Two explanations
fit, and these runs cannot tell them apart:
1. **A noise fluctuation** in Riccardo's realisation. A and E share the amplitude, so this would be a single fluctuation.
2. **His data were generated with sin ψ = −0.83** (the value in paper 2) rather than the yaml's −0.990. Stage 1 already
   flagged this. That change would raise the chunk powers by 7–18% in the same pattern, and the OG ψ posterior
   (median −0.842) leans towards it, though it is broad ([−0.975, −0.554]).

**This doesn't change the width comparison**, but Riccardo should confirm which ψ generated the pack-2 data, or share
the raw data. If it was −0.83, run 2 should be repeated with that value to compare centres too.

## Caveats

* One noise realisation per pipeline, and they are different realisations: the widths are reliable to roughly 10–20%,
  the centres are not comparable.
* 30 days only. For longer windows the sky model's rigidity (5 parameters for the whole year) and the wavelet model's
  freedom (more atoms) will pull further apart. The year run from the earlier plan is the place to test that.

## Next steps (suggested)

1. Ask Riccardo about ψ (and the pack-2 raw data).
2. Optionally run the OG chunked model on our run-2 realisation (same data for both) to remove the
   realisation scatter from the comparison. This needs the chunked pipeline on our time series.
3. Year-long run.
