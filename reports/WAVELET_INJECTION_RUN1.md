# Run 1: injecting the best-fit wavelet modulation and recovering it with RJMCMC

**Run 1.** When the data contain a modulation that the wavelet model can represent **exactly**, does the
reversible-jump sampler recover it? "It" means the number of atoms K, the atoms themselves, the spectral parameters
and, most importantly, $P_c(t)$. This is the recovery test behind [run 2](WAVELET_INJECTION_RUN2.md), which analyses
the true sky envelope with the wavelet and OG pipelines on the same data.

* Notebook (reproduces every number and figure): [data/wavelet_injection/run1_inject_recover.ipynb](../data/wavelet_injection/run1_inject_recover.ipynb)
* Code: [fit_reference.py](../data/wavelet_injection/fit_reference.py) (the injected atoms),
  [injection.py](../data/wavelet_injection/injection.py) (`python injection.py --tag run1 --injection wavelet`),
  [diagnostics.py](../data/wavelet_injection/diagnostics.py)
* Outputs: `data/wavelet_injection/outputs/run1/`, `outputs/reference/`; layout in
  [data/wavelet_injection/README.md](../data/wavelet_injection/README.md)

**Supersedes** the first version of this run (Oct 9), which injected atoms fitted to the sin ψ = −0.99 envelope,
used $N_f$ = 4096 (29.4 d), dropped 4 edge pixels (1.9 d) at each end and analysed a periodic simulated series.

## What was injected

The least-squares wavelet fit of the true GF envelope (sin ψ = −0.83, [config/sources.yaml](../data/wavelet_injection/config/sources.yaml))
over the 30 days (`fit_reference.py`, `outputs/reference/fit.png`), in the inference parametrisation ($b_A \equiv 0$
is absorbed into the galactic amplitude, and `level_E` $= b_E - b_A$):

| | value |
|---|---|
| galactic `amp` | −43.171 (effective; the true GF amp is −43.943, shifted by $b_A/\ln 10$) |
| `level_E` | +0.734 |
| A atoms (K = 2) | centre 0.60 d, width 23.6 d, amp −1.81; centre 15.19 d, width 17.5 d, amp +0.49 |
| E atom (K = 1) | centre 10.35 d, width 35.5 d, amp −1.49 |
| other spectral parameters | the injected values of `config/sources.yaml` |

The atoms are drawn in the time domain under the injected $P_c(t)$ in `outputs/run1/injection.png`. Each one is
$a_k\,\psi\big((t-t_k)/\tau_k\big)$ with $\psi(x) = e^{-x^2/2}$, in $\ln P$ units, over $\pm 2.5\,\tau_k$ around its
centre. All three are wider than half the 30-day span, so the data see only part of each Gaussian. A is a deep wide dip
centred at the start plus a shallower positive bump at day 15; E is one dip at day 10 on top of the level $b_E$. The
model is defined in [WAVELET_README.md](WAVELET_README.md), Section 4.

This $P_c(t)$ matches the true envelope to 0.2% rms in A and 0.24% in E. Two constraints of the fit make it a fair
recovery test: atom centres at least 0.6 d from the ends, so no injected value sits on a prior boundary (A's first atom
would otherwise sit at exactly t = 0); and a weak penalty on atom amplitudes, so E's atom does not run onto its
amplitude bound along the level / width degeneracy. Every injected value is inside the inference prior.

Data: one continuous 30.00-day series, Δt = 10 s, TDI gen 1, seed 2026, simulated with 2 extra days at each end and
cut (non-periodic ends); WDM with $N_f$ = 3600 (72 pixels of 10 h). **Same seed as run 2**, so the noise and galaxy
draws are the same; only the modulation differs.

## Code flow

```
fit_reference.py: true M_c^2(t) (sin psi = -0.83) ──least squares──▶ outputs/reference/atoms.npz
        │ load_reference [injection.py]
        ▼
Injection('wavelet').P(t, c) = exp( level_c + Σ_k a_k ψ((t − t_k)/τ_k) )
        ▼  simulate_series: simulate_channels(N + 2·pad, t0 = −2 d, modulation = P) → cut [0, 30 d]
WaveletData (Nf = 3600, Nt = 72) → outputs/run1/data.h5, timeseries.npz
        ▼  build_posterior
WaveletLikelihood: 174 log-frequency groups, edge_bins = 2 (20 h dropped at each end), time_bin = 2
                   → 34 time groups, analysed span 0.8–29.2 d, Gamma likelihood
WaveletPosterior(modulation='wavelets', level_range [−3, 3], max_wavelets = 8)
        ▼  make_sampler / run
eryn: psd (8 params, stretch), mod_A, mod_E (atoms, Gaussian moves + reversible jump)
   priors: t_frac U[0,1], width log-U[5, 60] d, amp U[−3, 3], K U{0..8}, spectral bounds of sources.yaml
   start: psd in a 1e-3 ball around the truth, every walker at K = 0
   32 walkers × 4 temperatures, 2000 burn-in + 6000 stored steps, 42 min on 8 cores
        ▼
outputs/run1/chains.npz → run1_inject_recover.ipynb
        (+ run_k_trace: same run with the burn-in stored → outputs/run1/k_trace.npz)
```

## Results

**Walkers.** 32 walkers per temperature × 4 temperatures = 128 walkers. The posterior comes from the 32 walkers of
the cold chain ($\beta = 1$); the three hotter chains only help mixing through swaps.

**Exploration of K, including the burn-in** (`K_trace_burnin.png`). eryn does not store the burn-in, so the run was
repeated with the same seeds and the burn-in stored (`python injection.py --tag run1 --injection wavelet --k-trace`,
`outputs/run1/k_trace.npz`). The rerun reproduces `chains.npz` exactly: K is identical at every stored step for every
walker. The plot shows, per step, the band from the minimum to the maximum K over the 32 cold walkers, the 25–75%
range and the mean.

| channel | true K | some walker at the true K from step | every walker off K = 0 from step | K range, burn-in | K range, stored | mean K, last 500 burn-in steps / stored |
|---|---|---|---|---|---|---|
| A | 2 | 11 | 15 | 0–8 | 1–8 | 2.20 / 2.27 |
| E | 1 | 1 | 3 | 0–7 | 1–8 | 2.07 / 2.01 |

Starting from K = 0, the ensemble finds the modulation within the first ~15 steps and then spans the whole relevant
range: the true K is inside the [min, max] band at every stored step in both channels, and the band reaches up to 7–8
atoms, so the walkers are not confined to a narrow range of K. The mean K is flat from about step 100, well inside the
2000-step burn-in, and the same at the end of the burn-in as in the stored part. E's mean K ≈ 2 against a true K = 1
is not a bias: it is the mean of the broad posterior below (p(K = 1) = 0.41, p(K = 2) = 0.33, p(K ≥ 3) = 0.26).

**Convergence: good.** Every monitored quantity (8 spectral parameters, K per channel, log L, $P_c(t)$ at days 5, 15,
25) has an autocorrelation time ≤ 10 steps: more than 570 τ in 6000 steps, ESS ≥ 18 000. No stuck walkers. Cold
acceptance 6.5%, RJ acceptance 0.7%. The median log L is 0.9 above the log L at the injection (maximum 6.9 above),
typical for ~17 free parameters.

**Modulation: recovered** (`modulation_recovery.png`). The plotted quantity is
$10^{\Delta\log_{10}A}\,P_c(t)$ with $\Delta\log_{10}A = {\rm amp} - {\rm amp}_{\rm inj}$: the galactic power of each
posterior sample relative to the injected amplitude ($\rm amp = \log_{10}A_{\rm gal}$, ${\rm amp}_{\rm inj} = -43.171$).
Only the product $A_{\rm gal}P_c(t)$ is measured, since a very wide atom is a constant in $\ln P_c$ that `amp` can
cancel; dividing by the injected amplitude puts the product on the scale of the injected $P_c(t)$
([WAVELET_README.md](WAVELET_README.md), Section 4.2).

| channel | 90% band width (median) | injection inside the 90% band | median offset |
|---|---|---|---|
| A | 11.0% of P | 100% of the analysed time bins | about −3% over days 7–29, +7% at day 0 |
| E | 10.2% of P | 87% | about −5% over the whole window |

Both offsets are the noise realisation, not the method: run 2 analyses the same noise with both the wavelet and the
OG pipeline, and both put E 3–4% low and A ~4% low towards the end. The one wavelet-only feature is A at the start
(+7% at day 0, back to 0 by day 5, inside the band). It also appears in run 2's wavelet posterior but not in the OG
one, so it comes from the model near the edge: the posterior on the first A atom's centre piles up against
$t_{\rm frac} = 0$ (`atoms.png`), i.e. the data would like the atom at or before the start, which the prior forbids, and
the curve flattens there.

**Spectral parameters: recovered.** All 8 inside their 90% intervals, every |z| ≤ 1.3:

| param | truth | median | 90% interval |
|---|---|---|---|
| alpha | 1.800 | 1.833 | [1.732, 1.936] |
| amp | −43.171 | −43.249 | [−43.353, −43.101] |
| fknee | −2.177 | −2.179 | [−2.182, −2.176] |
| fr1 | −2.429 | −2.419 | [−2.433, −2.406] |
| fr2 | −3.488 | −3.549 | [−3.665, −3.441] |
| A (noise) | 2.400 | 2.449 | [2.385, 2.515] |
| P (noise) | 7.900 | 7.903 | [7.884, 7.923] |
| level_E | 0.734 | 0.757 | [0.472, 1.027] |

**Number of atoms: the true K is the mode in both channels, but only just.**

| channel | true K | p(K=0) | p(K=1) | p(K=2) | p(K=3) | p(K=4) | p(K≥5) |
|---|---|---|---|---|---|---|---|
| A | 2 | 0 | 0.240 | **0.419** | 0.217 | 0.097 | 0.028 |
| E | 1 | 0 | **0.407** | 0.331 | 0.160 | 0.061 | 0.040 |

K = 0 is never visited, so the modulation is detected decisively. The log Bayes factor of the true K over the next
best is 0.56 (A) and 0.21 (E): over 30 days both envelopes are smooth, and with ~10% bands one, two or three atoms draw
the same curve. **K is not identifiable here and is not the recovery criterion.**

**Atoms** (samples with K = K_true, `atoms.png`):
* **E:** the single atom is recovered cleanly: centre 10.35 d inside [7.6, 12.5], log10 width 1.55 inside
  [1.39, 1.62], amplitude −1.49 inside [−1.73, −1.10].
* **A:** individually not identifiable, as expected. The first atom's centre piles up at the start (90% interval
  [0.1, 22.7] d, mode at 0) and its amplitude −1.81 is inside [−2.54, −0.54]; the second atom's centre spreads over
  the whole window. Many atom configurations draw the same monotonic rise.

## Comparison with the old run 1

| | old (Oct 9) | new |
|---|---|---|
| injection | atoms of the sin ψ = −0.99 envelope; first A atom at 0.2 d | sin ψ = −0.83; first A atom at 0.6 d |
| data | 29.4 d, periodic series, 4 edge pixels (1.9 d) dropped | 30.0 d, non-periodic, 2 edge pixels (20 h) dropped |
| analysed span | 2.1–26.8 d | 0.8–29.2 d |
| 90% band width A / E | 11% / 10% | 11.0% / 10.2% |
| truth inside the band A / E | 100% / 100% | 100% / 87% |
| E offset | −4% (cause unknown) | −5%; shown by run 2 to be the noise realisation |
| convergence | τ ≤ 8 steps (checked afterwards) | τ ≤ 10 steps |

The band widths are unchanged; the analysed span grew by 3.6 days.

## Verdict

The sampler recovers what the data constrain: the modulation $P_c(t)$ (band ≈ 10–11%, truth inside it over the whole
span in A and 87% in E, where the shared noise pulls both pipelines low) and all spectral parameters. Individual atoms
and K are not unique for 30 days of a smooth envelope, a property of the model and these data. The run is well
converged.

## Possible improvement

Let atom centres extend beyond the window (e.g. $t_{\rm frac} \in [-0.2, 1.2]$): the data want A's first atom at or
before the start, and the prior boundary at 0 is the likely cause of the +7% at the start of A.
