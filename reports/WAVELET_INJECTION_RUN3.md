# Run 3: a one-day data gap in the wavelet analysis

**Run 3.** Take [run 2](WAVELET_INJECTION_RUN2.md)'s 30-day series (true sky envelope, seed 2026) and cut a hard
1-day gap into it, from day 12.0 to 13.0. Run the same wavelet RJMCMC with the gap pixels left out of the likelihood.
**The only difference from run 2's wavelet run is the gap**, so the two posteriors can be compared directly. The
questions: is the modulation still recovered, and how much do the posteriors widen? Wavelet pipeline only (no OG).

Done on Oct 16, 2026, with the code of runs 1–2 plus the likelihood mask described below.

* Notebook (reproduces every number and figure): [data/wavelet_injection/run3_gap.ipynb](../data/wavelet_injection/run3_gap.ipynb)
* Code:
  * [injection.py](../data/wavelet_injection/injection.py): `python injection.py --tag run3 --injection sky --gap 12 13`
    (`--k-trace` for the burn-in);
  * [gap_check.py](../data/wavelet_injection/gap_check.py): which pixels a gap biases;
  * `WaveletLikelihood(time_mask=...)` in [bahamas/wavelet/model.py](../bahamas/wavelet/model.py).
* Outputs: `data/wavelet_injection/outputs/run3/`; gap checks in `outputs/checks/gap_check.png`,
  `gap_likelihood_mask.png`

## How the gap is handled

1. **The data.** The time series is set to zero for 12.0 ≤ t < 13.0 d, i.e. 8640 samples per channel. It is identical
   to run 2's series everywhere else (checked in the notebook).
2. **Which pixels are biased.** [gap_check.py](../data/wavelet_injection/gap_check.py) measured this over 20 noise
   realisations, with the WDM power of each pixel compared with its expectation:
   * the 3 time pixels that overlap the gap are 60–91% low;
   * the pixel just outside on each side is about 10% low above 1 mHz, and 25–40% high below 1 mHz, where the hard
     gap edge leaks the steep low-frequency spectrum;
   * two pixels away, the bias is below about 2%.
3. **The likelihood leaves those 5 time pixels out.** That is the 3 gap pixels plus 1 buffer pixel on each side,
   **11.46–13.54 d (50 h)** for a 24-h gap. The pixels are left out, not set to zero: a zero in the Gamma likelihood
   would be a measured power of zero.
4. **Grouping.** The likelihood sums pixels into blocks of 2 time pixels (20 h) × $n_g$ frequency pixels, with
   $\nu = 2n_g$. A gap removes whole time columns, so the frequency groups are untouched, and each block keeps
   $\nu = ({\rm kept\ time\ pixels}) \times n_g$:
   * the two blocks covering 11.46–13.13 d lose both pixels and are dropped;
   * the block 13.13–13.96 d keeps one pixel ($\nu = n_g$), and its modulation is evaluated at that pixel's time;
   * every other block is unchanged.

   The likelihood uses 32 blocks instead of 34, i.e. 7.4% fewer pixels (`outputs/run3/data_gap_mask.png`,
   `outputs/checks/gap_likelihood_mask.png`).

The mask is a new option of `WaveletLikelihood` (`time_mask`). Without a mask, the likelihood is bit-for-bit the same
as before (checked on the run-1 and run-2 data), and `tests/test_wavelet_model.py::test_time_mask` checks the block
sums, the reduced ν, and that masked pixel values do not enter the likelihood.

## Results

### Sampling

32 walkers × 4 temperatures, 2000 burn-in + 6000 stored steps from K = 0, the same seeds as run 2, 30 min on 8 cores.

* **Convergence.** I monitored the 8 spectral parameters, K, log L, and $P_c(t)$ at days 5, 12 (inside the gap), 15
  and 25. Every one has an autocorrelation time ≤ 8 steps: more than 750 τ, ESS ≥ 24 000. No walkers are stuck.
  Acceptance is 7.9% and RJ acceptance 0.8%, as in run 2.
* **K, including the burn-in** (`K_trace_burnin.png`; the rerun reproduces `chains.npz` exactly). Every walker has
  found the modulation by step 29 (A) and step 3 (E). The mean K is flat well inside the burn-in: 1.73 → 1.80 (A) and
  1.95 → 1.97 (E), from the last 500 burn-in steps to the stored part.

  | channel | run | p(K=1) | p(K=2) | p(K=3) | p(K≥4) |
  |---|---|---|---|---|---|
  | A | no gap | 0.39 | **0.44** | 0.14 | 0.03 |
  | A | gap | **0.47** | 0.35 | 0.11 | 0.07 |
  | E | no gap | **0.44** | 0.34 | 0.16 | 0.06 |
  | E | gap | **0.41** | 0.35 | 0.13 | 0.10 |

  As in runs 1–2, K is not identifiable (one or two wide atoms draw the same curve). The gap shifts p(K) by less
  than this ambiguity.

### Spectral parameters (`corner_gap_vs_nogap.png`)

| param | truth | no gap: median, 90% width | gap: median, 90% width | width gap / no gap | shift / σ(no gap) |
|---|---|---|---|---|---|
| alpha | 1.800 | 1.832, 0.200 | 1.845, 0.204 | 1.02 | +0.21 |
| fknee | −2.177 | −2.1785, 0.0061 | −2.1788, 0.0059 | 0.96 | −0.16 |
| fr1 | −2.429 | −2.419, 0.025 | −2.416, 0.026 | 1.06 | +0.35 |
| fr2 | −3.488 | −3.545, 0.236 | −3.548, 0.236 | 1.00 | −0.03 |
| A (noise) | 2.400 | 2.443, 0.118 | 2.467, 0.128 | 1.08 | +0.66 |
| P (noise) | 7.900 | 7.904, 0.043 | 7.900, 0.043 | 1.00 | −0.30 |
| amp | (−43.417 ref.) | −43.401, 0.247 | −43.392, 0.219 | 0.88 | +0.11 |
| level_E | (+1.172 ref.) | 1.143, 0.500 | 1.031, 0.635 | 1.27 | −0.75 |

* **The six spectral parameters change by 0–8% in width**, and all medians move by less than 0.7σ. Losing 7.4% of the
  pixels would widen a uniformly constrained parameter by about √(1/0.926) ≈ 4%, so this is as expected.
* **`amp` and `level_E` are not physical normalisations here.** `amp` trades against the spectral shape, and
  `level_E` against the E atoms (run 1). Their marginals are irregular in both runs (`amp` has two bumps), so a
  0.88× or 1.27× width is a change of shape along these degeneracies, not a loss of information.

### The modulation (`modulation_gap_vs_nogap.png`)

The plotted quantity is the galactic brightness recovered / injected: each posterior sample's amplitude × spectral
shape × $P_c(t)$, divided by the injected one. The spectral shape is averaged over frequency with the weight of where
the galaxy dominates (run-2 notebook §5b). 0 on the plot means recovered = injected. There are 1000 posterior
samples, and the median and the 90% band are taken at each time.

| channel | run | 90% width, away from the gap | 90% width, within 1 d of the gap | median offset | truth inside 90%, whole span / near the gap |
|---|---|---|---|---|---|
| A | no gap | 6.1% | 5.9% | +1.5% | 90% / 100% |
| A | gap | 6.1% | 6.2% | +1.2% | 91% / 100% |
| E | no gap | 5.4% | 5.4% | −1.3% | 100% / 100% |
| E | gap | 5.5% | 5.8% | −1.4% | 100% / 100% |

* **The modulation is recovered through the gap.** The run-3 median follows run 2's within about 0.5%, and the truth
  stays inside the band across the gap in both channels.
* **The band widens only locally, and only by about 0.3–0.4 percentage points** (5–7%) around the gap. The envelope
  changes on a scale of weeks, so atoms 20–35 days wide interpolate across 50 h with little extra freedom. Away from
  the gap the widths are unchanged.
* A's 91% coverage over the whole span comes from the window ends (the excess at day 0 and the drop after day 27, as
  in run 2), not from the gap.

### Shape only (`shape_gap_vs_nogap.png`)

$P_c(t)$ divided by its own mean over the span, so amplitude and spectrum drop out.

| channel | 90% width, no gap | 90% width, gap | truth inside, no gap / gap |
|---|---|---|---|
| A | 4.3% | 4.5% | 100% / 96% |
| E | 3.9% | 4.0% | 100% / 100% |

A's 96% is at the end of the window (days 27–30), where both runs drop below the injection; the gap region is covered.

## Verdict

**A 1-day gap, handled by leaving out 50 h of WDM pixels, costs almost nothing over 30 days.**

* The spectral parameters widen by 0–8% (about 4% expected from the lost data).
* The galactic brightness band widens by about 0.4 points near the gap and not at all elsewhere.
* The recovered modulation is the same as without the gap, and the truth stays inside the band across the gap.

The masking itself is exact: the remaining pixels are independent, and partial blocks only lose degrees of freedom.

## Caveats

* **One realisation, one gap in the middle of the window.** A gap near an end, where the atoms already extrapolate,
  would cost more.
* **A smooth envelope.** A modulation feature on the scale of the gap (≲ 2 d) could not be recovered inside it. Over
  30 days the Galactic envelope has no such feature.
* **The buffer of 1 pixel per side** keeps the residual bias of the kept neighbours at the ~2% level that 20
  realisations can resolve. A buffer of 2, or a tapered gap edge, would push it lower at the cost of 20 h more data.
* **The hard gap leaks low-frequency power into the neighbouring pixels** (+25–40% below 1 mHz). This is removed by
  the buffer here, but it is a reason to taper the gap edges in real data.

## Next steps (suggested)

1. A realistic gap pattern: the scheduled 7-h gaps every 14 days plus random unscheduled gaps (`bahamas.method.gaps`),
   over 30 days and later a year.
2. A gap near the start or end of the window.
3. Tapered gap edges: does the buffer shrink below 1 pixel at low frequency?
4. The OG chunked pipeline on the same gapped data (it has to drop or shorten chunks).
