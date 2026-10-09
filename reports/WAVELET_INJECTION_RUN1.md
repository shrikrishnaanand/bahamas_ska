# Run 1: injecting the best-fit wavelet modulation and recovering it with RJMCMC

**Stage 3, run 1** of the cyclo / Fig. 7 project. The question: when the data contain a modulation that the wavelet
model can represent **exactly**, does the reversible-jump sampler recover it? Here "it" means the number of atoms K,
the atoms themselves, the spectral parameters and, most importantly, $P_c(t)$. This test comes before run 2, which
injects the true sky envelope and compares the result with Riccardo's chunked (OG) posterior.

* Notebook (reproduces every number and figure): [data/wavelet_injection/run1_inject_recover.ipynb](../data/wavelet_injection/run1_inject_recover.ipynb)
* Driver code: [data/wavelet_injection/injection.py](../data/wavelet_injection/injection.py) (also runs as a script: `python injection.py --tag run1 --injection wavelet`)
* Outputs: `data/wavelet_injection/outputs/run1/` (`data.h5`, `chains.npz`, figures, `run.log`); folder layout in [data/wavelet_injection/README.md](../data/wavelet_injection/README.md)
* Earlier stages: [GF_MODULATION_CODE_FLOW.md](GF_MODULATION_CODE_FLOW.md) (stage 1), [WAVELET_FIT_WORKFLOW.md](WAVELET_FIT_WORKFLOW.md) (stage 2)

## What was injected

The stage-2 fit over the 30-day pack-2 window (`wavelet_fit_reference.npz`, `pack2_*`), in the inference
parametrisation ($b_A \equiv 0$ is absorbed into the galactic amplitude, and `level_E` $= b_E - b_A$):

| | value |
|---|---|
| galactic `amp` | −43.267 (effective; the true GF amp is −43.943, shifted by $b_A/\ln 10$) |
| `level_E` | +0.443 |
| A atoms (K = 2) | centre 0.20 d, width 17.3 d, amp −2.41; centre 4.01 d, width 14.7 d, amp +1.00 |
| E atoms (K = 1) | centre 9.62 d, width 27.9 d, amp −1.07 |
| other spectral parameters | the injected values of `cyclo_riccardo/cyclo/pe_pack2_cyclo.yaml` |

This $P_c(t)$ matches the true GF envelope $M_c^2(t)$ to about 0.3% rms (`injection.png`). Data settings follow
Riccardo's pack-2 run: T = 30 d, Δt = 10 s, 0.1–29 mHz, **TDI gen 1** (`gen2: False`, as in `config_pack2_Gamma_cyclo.yaml`).

## Code flow

```
wavelet_fit_reference.npz (stage 2)          pe_pack2_cyclo.yaml (OG sources)
        │ load_reference [injection.py:56]             │ inference_sources [:85]: drop lat/long/psi/s1/s2, amp ← amp_eff
        ▼                                              ▼
remap_atoms [:78]: t_frac on [0, 30 d] → t_frac on the WDM span [0, N·Δt] = [0, 29.4 d]
Injection('wavelet').P(t, c) [:123] = exp( level_c + Σ_k a_k ψ((t − t_k)/τ_k) )      (WaveletModulation.log_modulation)
        │
        ▼  simulate [:138] (via load_or_simulate [:212])
simulate_channels(..., modulation=Injection.P)    [bahamas/wavelet/simulate.py, new optional argument]
   x_c(t) = n_c(t) + sqrt(P_c(t)) g_c(t)          one continuous 30-d series per channel, seed 2026
WaveletData.from_time_series → WDM, Nf = 4096 (ΔT = 11.4 h), Nt = 62
   saved: outputs/run1/data.h5 (with the injected P at the pixel times)
        │
        ▼  build_posterior [:165]
WaveletLikelihood: 200 log-frequency bins (178 non-empty), edge_bins = 4, time_bin = 2 → 27 time groups,
                   analysed span 2.1–26.8 d (group centres), Gamma likelihood
WaveletPosterior(modulation='wavelets', gen2=False, level_range=[-3, 3], max_wavelets=8)
   level_E injected value set to +0.443 (the class default is 0)
        │
        ▼  make_sampler [:181] / run [:220]
WaveletSampler (eryn): branches psd (8 params), mod_A, mod_E (atoms, reversible jump)
   priors: t_frac U[0,1], width log-U[5, 60] d, amp U[−3, 3], K U{0..8}, spectral bounds of pe_pack2_cyclo.yaml
   init: psd in a 1e-3 ball around the truth, every walker at K = 0 (atoms have to be born)
   32 walkers × 4 temperatures, 2000 burn-in + 6000 stored steps, 35 min on 8 CPU cores
        │
        ▼
outputs/run1/chains.npz: chains, K posteriors, log L, log L at the injection (true_loglike [:189]),
                  1000 posterior draws of 10^(amp − amp_inj)·P_c(t) (modulation_draws)
        │
        ▼  run1_inject_recover.ipynb
§3 convergence · §4 p(K) · §5 corner + table · §6 atoms at K = K_true · §7 P_c(t) bands vs injection
```

Library change: [`simulate_channels`](../bahamas/wavelet/simulate.py) has a new optional `modulation(t, channel)`
argument that replaces the sky envelope. It defaults to `None`, so existing behaviour is unchanged
(`tests/test_wavelet_model.py`: 25 passed).

## Results

**Modulation: recovered.** The injected $P_c(t)$ lies inside the 90% band at every analysed time bin in both channels.
The 90% band is about 11% of $P$ wide in A and 10% in E (`modulation_recovery.png`). A traces the injection almost
exactly. In E, the median is about 4% low over the whole window: a coherent level offset, compatible with the
`amp`/`level_E` uncertainty (still inside the 90% band). Outside the analysed span (the grey edges), the band widens
as expected.

**Spectral parameters: recovered.** All 8 lie inside their 90% intervals, and every |z| < 1:

| param | truth | median | 90% interval |
|---|---|---|---|
| alpha | 1.800 | 1.820 | [1.721, 1.932] |
| amp | −43.267 | −43.320 | [−43.439, −43.184] |
| fknee | −2.177 | −2.175 | [−2.179, −2.171] |
| fr1 | −2.429 | −2.422 | [−2.436, −2.408] |
| fr2 | −3.488 | −3.505 | [−3.632, −3.400] |
| A (noise) | 2.400 | 2.428 | [2.368, 2.492] |
| P (noise) | 7.900 | 7.900 | [7.878, 7.922] |
| level_E | 0.443 | 0.456 | [0.070, 0.775] |

**Number of atoms: the true K is the mode, but the data barely prefer it.**

| channel | true K | p(K=0) | p(K=1) | p(K=2) | p(K=3) | p(K=4) | p(K≥5) |
|---|---|---|---|---|---|---|---|
| A | 2 | 0 | 0.246 | **0.359** | 0.227 | 0.113 | 0.055 |
| E | 1 | 0 | **0.494** | 0.352 | 0.123 | 0.028 | 0.004 |

K = 0 is never visited after burn-in, so the modulation is detected decisively. Among K ≥ 1, the log Bayes factor of
the true K over the next best is only ≈ 0.35–0.4. Over 30 days, both envelopes are smooth (A is a monotonic rise; E is
a shallow bowl). With 10% posterior bands, one, two or three atoms can draw the same curve. Stage 2 found that K = 1 for
A already has 1.6% rms, well below the band width. So **K is not identifiable here, and should not be the recovery
criterion**.

**Atoms (samples with K = K_true, `atoms.png`):**
* **E:** the single atom is recovered cleanly (centre 9.6 d inside [3.0, 12.0], width 28 d, amp −1.07 inside [−1.32, −0.65]).
* **A:** not identifiable individually. The first injected atom is centred at day 0.2, inside the dropped edge, so
  the data see only its right-hand tail. Its amplitude (−2.41) lies below the 90% interval [−1.98, −0.52], traded off
  against its width and centre. The second atom's posterior is spread over the whole window. Many atom configurations
  produce the same monotonic rise, which is the degeneracy flagged in stage 2.

**Convergence.** The mean K over walkers is stationary over the 6000 stored steps (A ≈ 2.4, E ≈ 1.6–1.8; E has a slow
wander). The median log L is 1.4 below the log L at the injection, and the maximum is 4.3 above it: typical for a
posterior with about 17 free parameters. Cold-chain acceptance is 7% for in-model moves and 0.7% for RJ moves. The
RJ rate is low, but K still moves often enough to visit K = 1–5 many times.

## Verdict

The sampler recovers what the data constrain: the modulation $P_c(t)$ (inside the 90% band everywhere, band ≈ 10%) and
all spectral parameters. Individual atoms and K are not unique for 30 days of a smooth envelope. That is a property
of the model and these data, not a sampler failure. **For run 2, compare the $P_c(t)$ bands, not atoms.**

## Notes for run 2 (true sky envelope vs OG chunked posterior)

Run 2 is done: see [WAVELET_INJECTION_RUN2.md](WAVELET_INJECTION_RUN2.md). The notes below are what it was planned from.

1. Reuse `injection.py`: call `simulate_channels(..., modulation=None)` with the pack-2 sky parameters restored, and
   use the true `amp` (−43.943) with `level_E` free (no injected value).
2. The OG posterior is Riccardo's nessai run (`cyclo_riccardo/cyclo/result.hdf5`). It constrains $M_c^2(t)$ through 5
   sky parameters and only two 15-day chunk averages. Put both on the same scale,
   $10^{\Delta\log_{10}A}\,P_c(t)$, before overlaying.
3. Our data are a new noise realisation (Riccardo's raw time series isn't available), so compare band **widths**; the
   centres will scatter differently.
4. The data volume is identical (30 d, same band, gen 1), so the width difference reflects the model: 5 sky
   parameters versus free atoms.
5. `chains.npz` is 95 MB (atoms are stored for every walker and step). Thin, or keep it out of git.
