# Wavelet fit to the GF modulation: code workflow

**Stage 2** of the cyclo / Fig. 7 project. The question: can the wavelet modulation model of
`bahamas.wavelet` represent the true Galactic-foreground (GF) modulation $P_c(t)=M_c^2(t)$ built in
stage 1? This is a **deterministic least-squares fit to the noiseless truth**, not an inference.
It tells us how many atoms are needed and gives reference atoms for stage 3 (injection and recovery).

Notebook: [data/wavelet_modulation_fit.ipynb](../data/wavelet_modulation_fit.ipynb).
Stage 1 (how the truth is produced): [GF_MODULATION_CODE_FLOW.md](GF_MODULATION_CODE_FLOW.md).

## The model being fitted

[`WaveletModulation.log_modulation`](../bahamas/wavelet/model.py#L296), for one channel $c$:

$$\log P_c(t) = b_c + \sum_{k=1}^{K} a_k\,\psi\!\left(\frac{t-t_k}{\tau_k}\right),\qquad
t_k = t_0 + t_{\rm frac,k}\,T \;\;[\text{model.py:319}],\qquad \tau_k = 10^{\log_{10}w_k}\,\text{day} \;\;[\text{model.py:320}]$$

* Each atom is a row `(t_frac, log10_width_days, amp)`. `psi` is `gaussian` (default) or `ricker`.
* `t0` and `T` define the window that `t_frac ∈ [0, 1]` spans, so atom centres stay inside the window.
* The same function is used inside the likelihood during sampling, so the fit uses exactly the inference model.

## Flowchart

```
data/cyclo_fig7_outputs/gf_modulation_truth.npz         (written by stage 1, cyclo_fig7_modulation.ipynb §7)
   t_hour, P_A, P_E  = envelopes_gaussian(cyclo pack-2 sky params, t)²   on an hourly grid, 4 yr
   Pbar_*, Pbar_err_*  15 d chunk averages and their time-domain estimates (used as the precision yardstick)
            │
            ▼
for shape  in {gaussian, ricker}
 for window in {year: [0, 1 yr],  pack2: [0, 30 d]}
  for channel c in {A, E}
   for K = 0, 1, 2, ...                                                (notebook §1, fit_window)
     ┌────────────────────────────────────────────────────────────────────────────────────┐
     │ wm   = WaveletModulation(t0, T, shape)                          [model.py:268]       │
     │ θ    = [b_c, (t_frac, log10_width_days, amp) × K]                                   │
     │ loss = mean over the window of ( wm.log_modulation(t, atoms, level=b_c) − log P_c(t) )² │
     │ (value, grad) = jax.jit(jax.value_and_grad(loss))                                   │
     │ bounds = the inference prior of template/config_wavelet.yaml [:41-43]               │
     │          t_frac ∈ [0,1], width ∈ [5,180] d (log), amp ∈ [−3,3], level ∈ [−2,2]      │
     │ starts = (i) best K−1 fit + new atom at the largest |residual| (greedy)             │
     │          (ii) 20 random draws inside the bounds                                     │
     │ scipy.optimize.minimize(L-BFGS-B, jac=True) from every start → keep the lowest loss │
     └────────────────────────────────────────────────────────────────────────────────────┘
            │  rms(log P), max |ΔP/P| for every (shape, window, channel, K)
            ▼
§2  rms vs K plot  →  K_SEL = smallest K with rms(log P) < 1%
            │          (1% is below the ~1.5% precision of one 15 d chunk estimate of the GF level)
            ▼
§3  best-fit plots: truth vs fit, ΔP/P in % with the chunk-precision band, individual atoms
    atom tables (centre, width, amplitude; flags atoms sitting on a prior bound)
            │
            ▼
§4  map to inference parameters                                        [sampler.py:152, :212]
      WaveletPosterior fixes b_A = 0 (degenerate with the galactic amplitude) and samples level_E
      ⇒ effective injected amp = log10 A + b_A / ln 10,   level_E = b_E − b_A
    save data/cyclo_fig7_outputs/wavelet_fit_reference.npz
      {window}_{A,E}_atoms, {window}_{A,E}_level, {window}_amp_eff, {window}_level_E, {window}_t0/T, rms_scan
```

## Where each piece lives

| Step | Code |
|---|---|
| Truth $P_c(t)$ | [modulation.envelopes_gaussian](../bahamas/psd_response/modulation.py#L15), squared; same as [simulate.power_modulation](../bahamas/wavelet/simulate.py#L68) |
| Model | [model.WaveletModulation](../bahamas/wavelet/model.py#L268), [log_modulation](../bahamas/wavelet/model.py#L296) |
| Prior ranges used as bounds | [template/config_wavelet.yaml](../template/config_wavelet.yaml#L40-L43), defaults in [pipeline.py:149-150](../bahamas/wavelet/pipeline.py#L149-L150), [sampler.py:367](../bahamas/wavelet/sampler.py#L367) |
| A level fixed at 0, `level_E` sampled | [sampler.py:152](../bahamas/wavelet/sampler.py#L152), [sampler.py:212](../bahamas/wavelet/sampler.py#L212) |
| Fit loop, plots, saving | [data/wavelet_modulation_fit.ipynb](../data/wavelet_modulation_fit.ipynb) |

## Results

rms error of $\log P$ (≈ rms relative error of $P$) against the number of atoms $K$, Gaussian atoms:

| window | ch | K=0 | 1 | 2 | 3 | 4 | 5 | 6 | 7 | 8 |
|---|---|---|---|---|---|---|---|---|---|---|
| year | A | 0.64 | 0.51 | 0.13 | 0.071 | 0.038 | 0.019 | **0.0053** | 0.0040 | 0.0022 |
| year | E | 0.44 | 0.34 | 0.059 | 0.030 | 0.011 | **0.0054** | 0.0034 | 0.0025 | 0.0019 |
| pack2 (30 d) | A | 0.34 | 0.016 | **0.0023** | 0.0002 | | | | | |
| pack2 (30 d) | E | 0.071 | **0.0031** | 0.0001 | 0.0001 | | | | | |

Bold: the smallest $K$ with rms < 1%. For comparison, one 15-day chunk estimate of the GF level has a
precision of about 2%.

* **The wavelet model matches the envelope.** Over the full year, **6 atoms (A) and 5 (E)** give 0.5% rms.
  The worst local error is 3–4%, at the window edges where the Gaussian atoms are cut off. Over the
  30-day pack-2 window, **2 atoms (A) and 1 (E)** give 0.2–0.3% rms.
* **Gaussian vs Ricker:** similar. Ricker needs one more atom for A over the year and does slightly better on the
  short window. Gaussian (the default) is fine.
* **Inference parametrisation:** with $b_A$ absorbed into the amplitude, the year fit implies
  `amp` = −43.979 (injected −43.943) and `level_E` = +1.98. The pack-2 fit implies `amp` = −43.267 and
  `level_E` = +0.44. These are the "true" values to compare against in stage 3 (`wavelet_fit_reference.npz`).

**Points to handle in stage 3**

1. **Level/atom degeneracy.** An atom much wider than the window is close to constant over it, so it trades
   off against the level. In the pack-2 fit the E level sits on its +2 bound, offset by a broad −1.07 atom.
   $P(t)$ is well determined, but the individual atoms are not. Judge recovery on $P(t)$, not on the atoms.
   For short windows, also consider capping `width_range_days` at about the window length.
2. **`level_E` prior.** The year fit needs `level_E` ≈ +1.98, at the edge of the template's `level_range: [-2, 2]`.
   Widen it to e.g. `[-3, 3]`.
3. **Atom count.** 5–6 atoms per channel per year is well inside `max_wavelets: 15`. If the data are informative
   enough, the reversible-jump posterior on $K$ should land near these values.

Figures: `data/cyclo_fig7_outputs/wavelet_fit_rms_vs_K.png`, `wavelet_fit_year_{gaussian,ricker}.png`,
`wavelet_fit_pack2_{gaussian,ricker}.png`.
