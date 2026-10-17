# Wavelet injection and recovery, and the comparison with the OG chunked pipeline (30 days)

Wavelet-domain RJMCMC (`bahamas.wavelet`) on 30 days of simulated LISA data with the settings of the paper-2 pack-2
run, and the OG chunked pipeline run by us on **the same time series**. Everything is simulated and analysed here;
no external data or posteriors are used.

| | injection | purpose | status | notebook | report |
|---|---|---|---|---|---|
| run 1 | best-fit wavelet atoms of the true envelope (`fit_reference.py`; A: 2, E: 1) | can the RJMCMC recover a modulation the model represents exactly? | **done** (redone Oct 16, fixed code) | [run1_inject_recover.ipynb](run1_inject_recover.ipynb) | [WAVELET_INJECTION_RUN1.md](../../reports/WAVELET_INJECTION_RUN1.md) |
| run 2 | true sky envelope $M_c^2(t)$, true amplitude | wavelet vs OG on the same data: which posterior on $P_c(t)$ is broader? | **done** (redone Oct 16, fixed code) | [run2_sky_vs_og.ipynb](run2_sky_vs_og.ipynb) | [WAVELET_INJECTION_RUN2.md](../../reports/WAVELET_INJECTION_RUN2.md) |
| run 3 | run 2's series with a 1-day gap (12–13 d) | wavelet with the gap pixels left out vs wavelet without the gap | **done** (Oct 16) | [run3_gap.ipynb](run3_gap.ipynb) | [WAVELET_INJECTION_RUN3.md](../../reports/WAVELET_INJECTION_RUN3.md) |
| run 4 | run 2's series with a 10-day gap (10–20 d) | as run 3, for a gap a third of the window long | **done** (Oct 17) | [run4_gap.ipynb](run4_gap.ipynb) | [WAVELET_INJECTION_RUN4.md](../../reports/WAVELET_INJECTION_RUN4.md) |
| run 5 | run 2's smooth Galaxy (as a sky map) + four bright patches (LMC, SMC, Sgr dSph, bar end); same noise as run 2 | can a sky model recover patches of excess power? | sky + atom run **done** (Oct 17) | [run5_needlet_sky.ipynb](run5_needlet_sky.ipynb) | |
| run 5a | run 5's series | patch fluxes at known positions (NUTS) vs the atom model | **done** (Oct 17) | [run5a_known_patches.ipynb](run5a_known_patches.ipynb) | |

Settings (both runs): sin ψ = −0.83 (paper 2); 30.00 d at Δt = 10 s, TDI gen 1, seed 2026, simulated with 2 extra
days at each end and cut (non-periodic ends). Wavelet: $N_f$ = 3600 (72 pixels of 10 h), 2 edge pixels dropped at each
end, eryn with 32 walkers × 4 temperatures, 2000 + 6000 steps from K = 0 (25 min on 8 cores). OG: two 15-day chunks,
5% Tukey taper, Gamma likelihood (correct dof, and the paper's with `--legacy-dof`), nessai with 2000 live points
(15 min on 8 cores).

```
config/sources.yaml          injected values and priors (sin psi = -0.83, paper 2), shared by both pipelines
config/og_config.yaml        OG chunked-pipeline settings (chunks, taper, binning, Gamma likelihood, nessai)
fit_reference.py             wavelet fit of the true envelope -> outputs/reference/ (atoms.npz = run-1 injection, fit.png)
injection.py                 simulation (one series for both pipelines), wavelet posterior and sampler, run()
                             script: python injection.py --tag run1 --injection wavelet [--rerun-data]
                                     python injection.py --tag run3 --injection sky --gap 12 13 [--gap-buffer 1]
                                     python injection.py --tag run2 --injection sky
                                     python injection.py --tag run5 --injection patches
                                     python injection.py --tag run1 --injection wavelet --k-trace
                                       (rerun with the burn-in stored -> k_trace.npz; chains.npz untouched)
og.py                        OG pipeline on the saved series: chunk FFTs, binning, nessai
                             script: python og.py --tag run2 [--legacy-dof]   (run from a terminal, not a notebook)
data_checks.py               input checks over 10 noise realisations (OG leakage vs taper, WDM edge pixels)
dof_check.py                 WDM pixel independence over 100 realisations: nu * Var(Y / E[Y]) = 2 per group,
                             neighbour correlations (-> outputs/checks/dof_check.npz)
gap_check.py                 which WDM pixels a data gap biases (20 realisations) and which the likelihood leaves out
                             (-> outputs/checks/gap_check.png, gap_likelihood_mask.png)
sky_patches.py               run 5 sky: run 2's Gaussian Galaxy as a HEALPix map + patches (config/run5_sky.yaml),
                             P_c(t) from the first-principles response of ../needlet_patches/sky.py
patch_fit.py                 run 5a: patch fluxes at known positions + spectra, NUTS in whitened logit coordinates
                             script: python patch_fit.py   (-> outputs/run5/5a/chains.npz, ~1 h on 8 cores)
likelihood_grid.py           time-frequency map of data/model per pixel and per likelihood group, galactic fraction,
                             and the effect of a 1-day gap (-> outputs/checks/likelihood_grid.png)
diagnostics.py               convergence of a wavelet run: autocorrelation times, ESS, stuck walkers
run1_inject_recover.ipynb    run 1 analysis (loads outputs/run1)
run2_sky_vs_og.ipynb         run 2 analysis and the OG comparison (loads outputs/run2)
run3_gap.ipynb               run 3 analysis: the gap, and the comparison with run 2's wavelet run (loads outputs/run2, run3)
run4_gap.ipynb               run 4 analysis: as run 3 for the 10-day gap (loads outputs/run2, run4)
run5_needlet_sky.ipynb       run 5: the patchy sky and its P_c(t) (-> outputs/run5/sky.png, modulation.png)
run5a_known_patches.ipynb    run 5a: patch fit vs atoms, corners, P_c(t), flux degeneracy (loads outputs/run5, run5/5a)
outputs/
  reference/                 atoms.npz, fit.png
  checks/                    data_checks.png, data_checks.npz, dof_check.npz, likelihood_grid.png
  run1/                      timeseries.npz, data.h5, chains.npz, run.log, injection.png, data_spectrogram.png,
                             convergence.png, K_posterior.png, corner_psd.png, atoms.png, modulation_recovery.png,
                             k_trace.npz (K of all 128 walkers per step, burn-in included), K_trace_burnin.png,
                             wdm_timefreq.png (WDM pixels and the likelihood grid in the time-frequency plane),
                             modulation_power.png (galactic PSD x P_c(t), notebook section 7b)
  run2/                      timeseries.npz (the shared series), data.h5 (WDM), chains.npz, run.log,
                             convergence_K.png, k_trace.npz, K_trace_burnin.png, modulation_og_vs_wavelet.png,
                             modulation_power_og_vs_wavelet.png, shape_og_vs_wavelet.png,
                             corner_spectral_og_vs_wavelet.png, og_sky_corner.png
    og/                      OG run, correct Gamma dof (bahamas default): data.h5, data_bin.h5, result.hdf5, nessai plots
    og_legacy/               OG run with the paper's Gamma dof (Pozzoli et al. Eq. 9; --legacy-dof)
  run3/                      timeseries.npz (run 2's series with the gap zeroed), data.h5, chains.npz, run.log, k_trace.npz,
                             data_gap_mask.png, K_trace_burnin.png, corner_gap_vs_nogap.png, modulation_gap_vs_nogap.png,
                             shape_gap_vs_nogap.png
  run4/                      as run3/ for the 10-day gap (no k_trace.npz, no K_trace_burnin.png)
```

`chains.npz` files are about 95 MB each (atoms of every walker and step); they are not meant for git.
