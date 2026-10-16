# Run 1: injecting the best-fit wavelet modulation and recovering it with RJMCMC

**Run 1.** When the data contain a modulation that the wavelet model can represent **exactly**, does the
reversible-jump sampler recover it? "It" means the number of atoms K, the atoms themselves, the spectral parameters
and, most importantly, the galactic power $A_{\rm gal}P_c(t)$. Run 2 (to be redone) analyses the true sky envelope
with the wavelet and OG pipelines on the same data.

**Redone on Oct 16, 2026** with the fixed code ([PAPER_BUGS.md](PAPER_BUGS.md)). The fixes that matter here:

* **F1, F2:** the Galactic envelope. The injected atoms are a fit to it, so they changed.
* **B3b:** the A/E test-mass transfer function. It enters the noise PSD by ≤0.3% in band, in both the data and the
  model.

The wavelet likelihood was never affected by the dof issue B2: WDM coefficients are real, so shape ν/2 is exact.
The previous version of this run (Oct 11, pre-fix code) is archived in
`data/archive/wavelet_runs_prefix_oct11/`.

* Notebook (reproduces every number and figure): [data/wavelet_injection/run1_inject_recover.ipynb](../data/wavelet_injection/run1_inject_recover.ipynb)
* Code: [fit_reference.py](../data/wavelet_injection/fit_reference.py) (the injected atoms),
  [injection.py](../data/wavelet_injection/injection.py) (`python injection.py --tag run1 --injection wavelet --rerun-data`),
  [diagnostics.py](../data/wavelet_injection/diagnostics.py)
* Outputs: `data/wavelet_injection/outputs/run1/`, `outputs/reference/`; layout in
  [data/wavelet_injection/README.md](../data/wavelet_injection/README.md)
* Method: [GF_TIME_FREQUENCY_NOTE.pdf](GF_TIME_FREQUENCY_NOTE.pdf), Part II

## What was injected

`fit_reference.py` fits the true GF envelope over the 30 days by least squares. The envelope is computed with the
fixed `envelopes_gaussian`, for the sky parameters with sin ψ = −0.83 in
[config/sources.yaml](../data/wavelet_injection/config/sources.yaml). The fit keeps the smallest K with an rms error
below 1%:

| K | 0 | 1 | 2 | 3 |
|---|---|---|---|---|
| rms error, A | 23.9% | 1.15% | **0.19%** | 0.05% |
| rms error, E | 7.7% | **0.11%** | 0.04% | 0.04% |

The result is written in the inference parametrisation, where $b_A \equiv 0$ is absorbed into the galactic amplitude
and `level_E` $= b_E - b_A$:

| | value |
|---|---|
| galactic `amp` | −43.417 (effective; the true GF amp is −43.943, shifted by $b_A/\ln 10$ with $b_A$ = 1.211) |
| `level_E` | +1.172 |
| A atoms (K = 2) | centre 1.10 d, width 23.7 d, amp −0.951; centre 29.36 d, width 26.9 d, amp +0.592 |
| E atom (K = 1) | centre 7.40 d, width 33.7 d, amp −1.349 |
| other spectral parameters | the injected values of `config/sources.yaml` |

`outputs/run1/injection.png` draws the atoms in the time domain under the injected $P_c(t)$. Each atom is
$a_k\thinspace \psi\big((t-t_k)/\tau_k\big)$ with $\psi(x) = e^{-x^2/2}$, in $\ln P$ units.

* **A** rises monotonically by a factor 2.1 over the window. The fit builds this from a wide dip centred near the start
  and a wide bump centred near the end.
* **E** has a shallow minimum at day 7 and rises by 30% towards the end.

All three atoms are wider than 23 days, so the data see only part of each Gaussian. Every injected value is inside the
inference prior. Atom centres are kept at least 0.6 d from the ends, so A's second atom (29.36 d) is close to that
margin but still inside the prior.

Data: one continuous 30.00-day series, Δt = 10 s, TDI gen 1, seed 2026, simulated with 2 extra days at each end and
cut, so the ends are not periodic. The WDM transform uses $N_f$ = 3600, i.e. 72 pixels of 10 h.

## Code flow

```
fit_reference.py: true M_c^2(t) (sin psi = -0.83, fixed envelope) ──least squares──▶ outputs/reference/atoms.npz
        │ load_reference [injection.py]
        ▼
Injection('wavelet').P(t, c) = exp( level_c + Σ_k a_k ψ((t − t_k)/τ_k) )
        ▼  simulate_series: simulate_channels(N + 2·pad, t0 = −2 d, modulation = P) → cut [0, 30 d]
WaveletData (Nf = 3600, Nt = 72) → outputs/run1/data.h5, timeseries.npz
        ▼  build_posterior
WaveletLikelihood: 174 log-frequency groups, edge_bins = 2 (20 h dropped at each end), time_bin = 2
                   → 34 time groups (centres 1.0–28.5 d), Gamma likelihood
WaveletPosterior(modulation='wavelets', level_range [−3, 3], max_wavelets = 8)
        ▼  make_sampler / run
eryn: psd (8 params, stretch), mod_A, mod_E (atoms, Gaussian moves + reversible jump)
   priors: t_frac U[0,1], width log-U[5, 60] d, amp U[−3, 3], K U{0..8}, spectral bounds of sources.yaml
   start: psd in a 1e-3 ball around the truth, every walker at K = 0
   32 walkers × 4 temperatures, 2000 burn-in + 6000 stored steps, 25 min on 8 cores
        ▼
outputs/run1/chains.npz → run1_inject_recover.ipynb
        (+ run_k_trace: same run with the burn-in stored → outputs/run1/k_trace.npz)
```

## Results

**Walkers.** 32 walkers per temperature × 4 temperatures = 128 walkers. The posterior comes from the 32 walkers of
the cold chain ($\beta = 1$); the three hotter chains only help mixing through swaps.

**Convergence: good.** I monitored the 8 spectral parameters, K per channel, log L, and $P_c(t)$ at days 5, 15 and 25.
Every one has an autocorrelation time ≤ 6.2 steps: more than 960 τ in 6000 steps, ESS ≥ 30 000. No walkers are stuck.
Cold acceptance is 7.4% and RJ acceptance 0.7%. The median log L is 1.2 above the log L at the injection (maximum
6.8 above).

**Exploration of K, including the burn-in** (`K_trace_burnin.png`). eryn does not store the burn-in, so the run was
repeated with the same seeds and the burn-in stored (`--k-trace`, `outputs/run1/k_trace.npz`). The rerun reproduces
`chains.npz` exactly.

| channel | true K | some walker at the true K from step | every walker off K = 0 from step | K range, burn-in | K range, stored | mean K, last 500 burn-in steps / stored |
|---|---|---|---|---|---|---|
| A | 2 | 8 | 45 | 0–5 | 1–5 | 1.60 / 1.70 |
| E | 1 | 1 | 3 | 0–8 | 1–8 | 2.02 / 2.01 |

Starting from K = 0, the ensemble finds the modulation within the first ~50 steps. The mean K is flat from about step
100, well inside the 2000-step burn-in. The true K is inside the [min, max] band over the cold walkers at every stored
step in both channels.

**Spectral parameters: recovered.** All 8 are inside their 90% intervals, with every |z| ≤ 1.3:

| param | truth | median | 90% interval |
|---|---|---|---|
| alpha | 1.800 | 1.829 | [1.736, 1.938] |
| amp | −43.417 | −43.382 | [−43.510, −43.304] |
| fknee | −2.177 | −2.179 | [−2.182, −2.175] |
| fr1 | −2.429 | −2.419 | [−2.432, −2.407] |
| fr2 | −3.488 | −3.553 | [−3.684, −3.450] |
| A (noise) | 2.400 | 2.447 | [2.389, 2.510] |
| P (noise) | 7.900 | 7.904 | [7.884, 7.926] |
| level_E | 1.172 | 1.025 | [0.739, 1.347] |

**Number of atoms: not identifiable.**

| channel | true K | p(K=0) | p(K=1) | p(K=2) | p(K=3) | p(K=4) | p(K≥5) |
|---|---|---|---|---|---|---|---|
| A | 2 | 0 | 0.434 | **0.454** | 0.094 | 0.019 | <0.001 |
| E | 1 | 0 | **0.373** | 0.391 | 0.147 | 0.043 | 0.046 |

K = 0 is never visited, so the modulation is detected decisively. The data cannot tell one atom from two, though:
the log Bayes factor of the true K over the best alternative is +0.05 (A) and −0.05 (E). Over 30 days both envelopes
are smooth, and one or two wide atoms draw the same curve. **K is not the recovery criterion.**

**Atoms** (samples with K = K_true, `atoms.png`).

* **E:** the single atom is recovered:

  | | injected | 90% interval |
  |---|---|---|
  | centre | 7.40 d | [3.6, 10.6] d |
  | log10 width | 1.53 | [1.39, 1.60] |
  | amplitude | −1.35 | [−1.58, −1.06] |

* **A:** individually not identifiable, as expected. Both injected atoms are inside the intervals, but the intervals
  are wide:

  | | first atom (injected) | 90% interval | second atom (injected) | 90% interval |
  |---|---|---|---|---|
  | centre | 1.10 d | [0.5, 22.0] d | 29.36 d | [7.1, 29.6] d |
  | amplitude | −0.95 | [−1.36, −0.36] | +0.59 | [−0.08, 1.16] |

  Many atom configurations draw the same monotonic rise.

### The recovered modulation

**What the data measure.** The galactic part of the likelihood is $A_{\rm gal}\thinspace S_{\rm shape}(f)\thinspace P_c(t)$,
where $S_{\rm shape}$ depends on α, fknee, fr1 and fr2. Neither $P_c(t)$ nor `amp` is identified on its own:

* a very wide atom is a constant in $\ln P_c$, so it trades against `amp`;
* `amp` trades against the spectral shape where the galaxy dominates.

Two quantities are shown, both relative to the injection.

**1. `amp` as the normalisation** (`modulation_recovery.png`). This is $10^{\Delta\log_{10}A}\thinspace P_c(t)$ with
$`\Delta\log_{10}A = {\rm amp} - {\rm amp}_{\rm inj}`$ and $`{\rm amp}_{\rm inj} = -43.417`$. It is the curve the
notebook plots and the one the previous version of this report used.

**2. The full galactic PSD** (`modulation_power.png`, notebook §7b). Each sample's galactic PSD relative to the
injected one replaces $10^{\Delta\log_{10}A}$. It is averaged over frequency with the Fisher weight of a
galactic-power scale, $w_g \propto \nu_g\thinspace [S_{\rm gal}/(S_{\rm gal}+S_n)]^2$, so it is the galactic power where
the data constrain it.

| channel | quantity | 90% band width (median) | injection inside the 90% band | median offset |
|---|---|---|---|---|
| A | $10^{\Delta\log_{10}A}P_c$ | 9.6% | 100% of analysed time bins | −1.8% (−6% to +4%) |
| A | galactic PSD × $P_c$ | **5.7%** | 88% | +1.5% (+8% at day 0) |
| E | $10^{\Delta\log_{10}A}P_c$ | 9.4% | 68% | −4.6% |
| E | galactic PSD × $P_c$ | **5.5%** | 100% | −1.4% |

**The E offset comes from the `amp` normalisation, not from the data.** In the first quantity the whole E posterior
sits 4.6% low, but there is no offset in the data. A profile of the likelihood over one galactic-power scale per
channel, with every other parameter at the injection, gives

| channel | best-fit scale of the galactic power |
|---|---|
| A | 1.020 ± 0.010 |
| E | 0.994 ± 0.010 |

What happens instead is a trade-off in the posterior. Its median `amp` is 0.035 dex (+8%) above the injection, the
spectral shape (α, fr2) compensates in band, and `level_E` drops (1.03 against 1.17). In the full galactic PSD these
cancel, and E is recovered to −1.4% with the truth inside the band throughout.

**The first quantity is also misleading in width.** With the spectrum included, the bands narrow from about 9.5% to
5.6%, because part of the `amp` scatter is not a change in power. For comparing pipelines on $P_c(t)$, the galactic
PSD × $P_c$ is the better quantity.

**A at the start.** The one feature the second quantity shows is A at the start: +8% at day 0, back to +2% by day 5.
The posterior on the first A atom's centre reaches the start (90% interval from 0.5 d), and the data would like the
atom at or before t = 0, which the prior $t_{\rm frac} \geq 0$ forbids. The previous run showed the same +7% at day 0.

## Comparison with the previous run 1 (Oct 11, pre-fix envelope)

| | Oct 11 | Oct 16 |
|---|---|---|
| injected envelope | pre-fix `envelopes_gaussian` (F1, F2) | fixed |
| atoms A | 0.60 d / 23.6 d / −1.81; 15.19 d / 17.5 d / +0.49 | 1.10 d / 23.7 d / −0.95; 29.36 d / 26.9 d / +0.59 |
| atom E, level_E | 10.35 d / 35.5 d / −1.49; level_E +0.73 | 7.40 d / 33.7 d / −1.35; level_E +1.17 |
| amp_eff | −43.171 | −43.417 |
| p(K_true), A / E | 0.42 / 0.41 | 0.45 / 0.37 |
| τ (max), runtime | 10 steps, 42 min | 6.2 steps, 25 min |
| band A / E, $10^{\Delta\log_{10}A}P_c$ | 11.0% / 10.2% | 9.6% / 9.4% |
| truth inside, A / E | 100% / 87% | 100% / 68% |
| E offset | −5%, attributed to the noise realisation | −4.6%; shown above to come from the `amp` normalisation |

The data use the same seed, so the noise realisation is the same. The old report said E's −5% offset was the noise
realisation, because run 2's OG pipeline was also low on the same noise. The profile likelihood above contradicts
that for this run: the data put E's galactic power at 0.994 ± 0.010 of the injection. Run 2 should therefore compare
the pipelines on the galactic PSD × $P_c$.

## Verdict

The sampler recovers what the data constrain:

* **Galactic power:** recovered to within ±1.5% in both channels, with a 5.5–5.7% band, apart from A's first day at
  the prior edge.
* **Spectral parameters:** all recovered.

Individual atoms and K are not unique for 30 days of a smooth envelope, which is a property of the model and of these
data. The run is well converged.

## Possible improvements

* **Let atom centres extend beyond the window** (e.g. $t_{\rm frac} \in [-0.2, 1.2]$). The data want A's first atom at
  or before the start, and the prior boundary at 0 is the likely cause of the +8% at day 0.
* **Report the galactic PSD × $P_c$** (notebook §7b) in place of, or next to, $10^{\Delta\log_{10}A}P_c$, in run 2 and
  later runs.
