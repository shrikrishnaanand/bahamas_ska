# Run 4: a ten-day data gap in the wavelet analysis

**Run 4.** As [run 3](WAVELET_INJECTION_RUN3.md), but with a **10-day gap**, from day 10.0 to 20.0: a third of the
30-day window. The input is [run 2](WAVELET_INJECTION_RUN2.md)'s series (true sky envelope, seed 2026), with the gap
pixels left out of the wavelet likelihood. **The only difference from run 2's wavelet run is the gap.** The question:
can the atoms bridge 10 days of missing data, and what does it cost? Wavelet pipeline only; no burn-in K trace.

Done on Oct 17, 2026.

* Notebook (reproduces every number and figure): [data/wavelet_injection/run4_gap.ipynb](../data/wavelet_injection/run4_gap.ipynb)
* Command: `python injection.py --tag run4 --injection sky --gap 10 20` ([injection.py](../data/wavelet_injection/injection.py))
* Outputs: `data/wavelet_injection/outputs/run4/`

## Setup

Same handling as run 3:

* the series is set to zero for 10.0 ≤ t < 20.0 d, and is identical to run 2's elsewhere;
* the likelihood leaves out every 10-h time pixel that overlaps the gap, plus one buffer pixel on each side.

That is 27 time pixels, **9.38–20.62 d (270 h)** (`outputs/run4/data_gap_mask.png`). The likelihood keeps 21 of the 34
time blocks and uses **39.7% fewer pixels** than run 2. Sampler, priors and seeds are as in run 2: 32 walkers × 4
temperatures, 2000 + 6000 steps from K = 0, 26 min on 8 cores.

## Results

### Sampling

* **Convergence.** Every monitored quantity has an autocorrelation time ≤ 5.5 steps: more than 1000 τ, ESS ≥ 35 000.
  The monitored quantities are the 8 spectral parameters, K, log L, and $P_c(t)$ at days 5, 15 (inside the gap) and
  25. No walkers are stuck. Acceptance is 9.2% and RJ acceptance 1.0%.
* **K** is not identifiable, as in all runs, and the gap changes little:

  | channel | run | p(K=1) | p(K=2) | p(K=3) | p(K≥4) |
  |---|---|---|---|---|---|
  | A | no gap | 0.39 | **0.44** | 0.14 | 0.03 |
  | A | 10-day gap | **0.43** | 0.37 | 0.14 | 0.06 |
  | E | no gap | **0.44** | 0.34 | 0.16 | 0.06 |
  | E | 10-day gap | **0.38** | 0.34 | 0.18 | 0.10 |

### Spectral parameters (`corner_gap_vs_nogap.png`)

| param | truth | no gap: median, 90% width | gap: median, 90% width | width gap / no gap | shift / σ(no gap) |
|---|---|---|---|---|---|
| alpha | 1.800 | 1.832, 0.200 | 1.824, 0.254 | 1.27 | −0.14 |
| fknee | −2.177 | −2.1785, 0.0061 | −2.1790, 0.0074 | 1.21 | −0.25 |
| fr1 | −2.429 | −2.419, 0.025 | −2.420, 0.032 | 1.32 | −0.20 |
| fr2 | −3.488 | −3.545, 0.236 | −3.560, 0.263 | 1.11 | −0.21 |
| A (noise) | 2.400 | 2.443, 0.118 | 2.460, 0.155 | 1.31 | +0.48 |
| P (noise) | 7.900 | 7.904, 0.043 | 7.906, 0.054 | 1.25 | +0.12 |
| amp | (−43.417 ref.) | −43.401, 0.247 | −43.402, 0.263 | 1.06 | −0.01 |
| level_E | (+1.172 ref.) | 1.143, 0.500 | 1.097, 0.735 | 1.47 | −0.31 |

* **The spectral parameters widen by 11–32%.** That is what losing 39.7% of the pixels predicts:
  √(1/0.603) ≈ 1.29. Every median stays within 0.5σ of run 2's.
* `amp` and `level_E` trade against the spectral shape and the E atoms (runs 1–3), so their widths are not
  information measures.

### The modulation (`modulation_gap_vs_nogap.png`)

The plotted quantity is the galactic brightness recovered / injected: each posterior sample's amplitude × spectral
shape × $P_c(t)$, divided by the injected one. The median and the 90% band are taken over 1000 samples at each time.

90% band width at fixed days, and over spans:

| channel | run | 5 d | 10 d | 15 d | 20 d | 25 d | in the gap: mean / max | kept data, 3–27 d |
|---|---|---|---|---|---|---|---|---|
| A | no gap | 6.5% | 5.5% | 5.8% | 5.5% | 6.2% | 5.7% / 5.9% | 6.1% |
| A | 10-day gap | 6.9% | 10.5% | **14.5%** | 10.9% | 6.2% | 12.9% / 14.6% | 7.5% |
| E | no gap | 5.8% | 5.0% | 5.3% | 4.7% | 5.6% | 5.1% / 5.4% | 5.5% |
| E | 10-day gap | 6.8% | 9.9% | **12.9%** | 8.8% | 6.5% | 11.3% / 13.1% | 7.2% |

* **The atoms bridge the 10 days.** Inside the gap, the run-4 median follows run 2's within about 1% (offset +2.0%
  against +1.2% in A, −1.9% against −1.6% in E), and the truth stays inside the 90% band throughout the gap in both
  channels.
* **The price is in the width.** The band grows steadily into the gap and peaks in the middle: 14.5% (A) and 12.9%
  (E) at day 15, about 2.5× run 2's. This is an honest interpolation uncertainty: the posterior admits every smooth
  curve that joins the two sides.
* **The data on either side widen slightly too** (6.1 → 7.5% in A, 5.5 → 7.2% in E over 3–27 d). Over 30 days the
  spectral parameters are shared by all times, and they are less constrained now. At the window ends (days 0–3 and
  27–30) the bands are as wide as in run 2.

### Shape only (`shape_gap_vs_nogap.png`)

$P_c(t)$ divided by its own mean over the span, so amplitude and spectrum drop out. The median 90% width over the span
goes from 4.3 to 6.9% in A and from 3.9 to 6.8% in E. The truth is inside the band over the whole span in both runs
and both channels.

## Run 3 vs run 4

| | 1-day gap (run 3) | 10-day gap (run 4) |
|---|---|---|
| left out | 50 h, 7.4% of the pixels | 270 h, 39.7% of the pixels |
| spectral widths vs no gap | +0–8% | +11–32% (≈ √ of the data loss) |
| brightness band, middle of the gap (A / E) | 6.2% / 5.8% (no gap: 5.9% / 5.4%) | 14.5% / 12.9% (no gap: 5.8% / 5.3%) |
| truth inside the band in the gap | yes | yes |
| median vs no gap | within 0.5% | within 1% |

## Verdict

**The wavelet analysis handles a 10-day gap without bias.** The recovered modulation goes through the truth inside
the gap, and the spectral parameters lose only what the missing data account for. The band across the gap is about
2.5× wider in the middle, the honest uncertainty of interpolating a third of the window. That this works so well
depends on the envelope being smooth on 10-day scales, and the atoms (20–35 d wide) building that smoothness in.

## Caveats

* **One realisation; the gap in the middle of the window, with data on both sides.** A 10-day gap at the start or the
  end has no data on one side, and the atoms would extrapolate rather than interpolate.
* **The atom widths are 5–60 d (log-uniform prior).** A modulation with structure shorter than the gap would not be
  constrained inside it. The width prior sets how smooth the bridge is.
* **No OG comparison.** The chunked pipeline would lose most of one 15-day chunk here.

## Next steps (suggested)

1. A long gap at the start or end of the window (extrapolation instead of interpolation).
2. The realistic LISA gap pattern (scheduled + unscheduled gaps, 75% duty cycle) over 30 days and a year.
3. The OG pipeline on the gapped data, for the same comparison as run 2.
