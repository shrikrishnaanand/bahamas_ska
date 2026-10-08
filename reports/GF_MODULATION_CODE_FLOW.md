# Galactic-foreground modulation: code flow

How the Fig. 7-type time-domain modulated Galactic foreground (GF) is produced in
[data/cyclo_fig7_modulation.ipynb](../data/cyclo_fig7_modulation.ipynb), and which function in the
package each piece comes from. All links are clickable from VS Code.

## What "noise" means here

There are two different Gaussian processes, and only one of them is in Fig. 7:

| Symbol | What it is | Where it comes from | In Fig. 7? |
|---|---|---|---|
| $n_c(t)$ | **Stationary GF "carrier"**: Gaussian noise with the *unmodulated* galactic PSD $S_{\rm GF}(f)$ | spectrum: [psd_galaxy.galactic_foreground](../bahamas/psd_strain/psd_galaxy.py#L41); random draw: [setting_data.GP_freq](../bahamas/method/setting_data.py#L9) | **yes**, this is the "noise" that gets modulated |
| $M_c(t)$ | Amplitude envelope from the bivariate-Gaussian sky model (Eq. 32 of arXiv:2410.08263) | [modulation.envelopes_gaussian](../bahamas/psd_response/modulation.py#L15) | **yes**, the black curve |
| $S_{n,c}(f)$ | **Instrument noise** (TM + OMS, parameters `A`, `P`) | [psd_noise.noise](../bahamas/psd_strain/psd_noise.py#L216) | no; it only enters the chunk spectra (§4a of the notebook) and the inference |

The Fig. 7 signal is therefore $x_c(t) = M_c(t)\,n_c(t)$: GF only, no instrument noise.

## Flowchart

```
 cyclo_riccardo/cyclo/pe_pack2_cyclo.yaml          cyclo_riccardo/cyclo/config_pack2_Gamma_cyclo.yaml
   galactic_DWD_time: alpha amp fknee fr1 fr2         T = 30 d, chunk = 15 d, dt = 10 s
                      lat long psi s1 s2             (the notebook uses dt = 30 s for the 4-yr series)
   instr_noise:       A P
            │
            ├──────────── spectral params (alpha, amp, fknee, fr1, fr2) ────────────┐
            │                                                                       ▼
            │                                       psd_galaxy.galactic_foreground(f, par)       [psd_galaxy.py:41]
            │                                         S_GF(f) = 10^amp f^-7/3 exp[-(f/f1)^alpha]
            │                                                   · ½[1 + tanh((fknee - f)/f2)]
            │                                                   · (2πfL)² sin²(2πfL)       (TDI factor, L = 8.3 s)
            │                                                                       │
            │                                                                       ▼
            │                                       setting_data.GP_freq(freqs, dt, S_GF, time=True)  [setting_data.py:9]
            │                                         random complex Fourier coeffs, var ∝ S_GF   (np.random, seeded)
            │                                         → np.fft.irfft → n_c(t)   with std(n) = sqrt(∫S_GF df)
            │                                         (called once per channel → independent n_A, n_E)
            │                                                                       │
            └──────────── sky params (lat=sinβ, long=λ, psi=sinψ, s1=σ1², s2=σ2²) ──┐   │
                                                                                │   │
                                                                                ▼   │
                         modulation.envelopes_gaussian(lat, long, s1, s2, psi,         │   [modulation.py:15]
                                                       f_orb = 1/yr, t)              │
                           phase  φ_L = 2π t / yr                    [modulation.py:80]  │
                           A²+E² ("Sum") and A²-E² ("Diff") harmonic series in φ_L    │
                           M_A = sqrt(½|Sum+Diff|),  M_E = sqrt(½|Sum-Diff|)  [:204]  │
                           evaluated on an hourly grid, then np.interp to the samples │
                                                                                │   │
                                                                                ▼   ▼
                                                     x_c(t) = M_c(t) · n_c(t)          (notebook §3)
                                                                │
                              ┌─────────────────────────────────┼──────────────────────────────┐
                              ▼                                 ▼                              ▼
                   Fig. 7: hourly min/max of x_c      15 d periodograms of x_c / S_GF   1-day running <x²>/σ²
                   with ±3σ M_c(t)                    → recovered chunk average M̄^c    → recovered M²(t)
                   (fig7_reproduction.png)            (modulation_recovery_td.png)
```

## The chunked (OG) path, for comparison

The original pipeline **never builds** $x_c(t)$. It replaces $M_c^2(t)$ by its chunk average and
draws each chunk as stationary data:

```
bahamas_data.SignalProcessor                                          [bahamas_data.py]
  handle_series()  chunk edges T1, T2 (t0 + k·chunk)                    [:218, :237]
  simulate_data()  for each chunk i, channel c:                         [:265]
      psd.model_psd(freqs, sources, t1=T1[i], t2=T2[i], tdi=c)          [psd_function.py:46]
        └─ 'galactic_DWD_time' branch                                   [psd_function.py:146]
             └─ psd_galaxy.galactic_foreground_time                     [psd_galaxy.py:98]
                  M̄^c = average_envelope.average_envelopes_gaussian(t1, t2)  [average_envelope.py:23, called at psd_galaxy.py:144]
                  S = M̄^c · S_GF(f)   (+ instr_noise from psd_noise.noise)
      setting_data.GP_freq(freqs, dt, S)  → chunk Fourier coefficients  [bahamas_data.py:282]
      log-binning (average_log_chunks) → Gamma-likelihood input         [setting_data.py:64]
```

Link between the two: $\bar M^c = \frac{1}{t_2-t_1}\int_{t_1}^{t_2} M_c^2(t)\,dt$. The notebook checks this
numerically (§2), and recovers it from $x_c(t)$ within the statistical error (§4b).

## The wavelet-pipeline path (same physics)

The wavelet subpackage builds the same signal as the notebook, plus instrument noise, using
`jax.random` instead of `np.random`:

```
wavelet.pipeline.generate_data                                        [pipeline.py:58]
  └─ simulate.simulate_channels                                       [simulate.py:144]
       n_c = stationary_gaussian(S_noise)          (instrument noise)  [simulate.py:41]
       g_c = stationary_gaussian(S_GF)             (GF carrier)        uses psd_galaxy.galactic_foreground
       P_c = power_modulation(envelope, t) = envelopes_gaussian(...)²  [simulate.py:68]
       x_c = n_c + sqrt(P_c) · g_c                                     [simulate.py:173]
```

So the notebook's $n_c(t)$ corresponds to `g_c` in [simulate.py](../bahamas/wavelet/simulate.py#L168),
and the notebook's $M_c^2(t)$ is `P_c`, the quantity that the wavelet model
`log P_c(t) = b_c + Σ_k a_k ψ((t - t_k)/τ_k)` ([model.py WaveletModulation](../bahamas/wavelet/model.py#L268)) is meant to represent.
