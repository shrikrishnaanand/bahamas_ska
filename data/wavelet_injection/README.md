# Wavelet injection and recovery, and the comparison with the OG chunked pipeline (30 days)

Wavelet-domain RJMCMC (`bahamas.wavelet`) on 30 days of simulated LISA data with the settings of the paper-2 pack-2
run, and the OG chunked pipeline run by us on **the same time series**. Everything is simulated and analysed here;
no external data or posteriors are used.

| | injection | purpose | status | notebook | report |
|---|---|---|---|---|---|
| run 1 | best-fit wavelet atoms of the true envelope (`fit_reference.py`; A: 2, E: 1) | can the RJMCMC recover a modulation the model represents exactly? | **done** (Oct 11) | [run1_inject_recover.ipynb](run1_inject_recover.ipynb) | [WAVELET_INJECTION_RUN1.md](../../reports/WAVELET_INJECTION_RUN1.md) |
| run 2 | true sky envelope $M_c^2(t)$, true amplitude | wavelet vs OG on the same data: which posterior on $P_c(t)$ is broader? | **done** (Oct 11) | [run2_sky_vs_og.ipynb](run2_sky_vs_og.ipynb) | [WAVELET_INJECTION_RUN2.md](../../reports/WAVELET_INJECTION_RUN2.md) |

Settings (both runs): sin ψ = −0.83 (paper 2); 30.00 d at Δt = 10 s, TDI gen 1, seed 2026, simulated with 2 extra
days at each end and cut (non-periodic ends). Wavelet: $N_f$ = 3600 (72 pixels of 10 h), 2 edge pixels dropped at each
end, eryn with 32 walkers × 4 temperatures, 2000 + 6000 steps from K = 0 (40 min on 8 cores). OG: two 15-day chunks,
5% Tukey taper, Gamma likelihood, nessai with 2000 live points (15 min on 8 cores).

```
config/sources.yaml          injected values and priors (sin psi = -0.83, paper 2), shared by both pipelines
config/og_config.yaml        OG chunked-pipeline settings (chunks, taper, binning, Gamma likelihood, nessai)
fit_reference.py             wavelet fit of the true envelope -> outputs/reference/ (atoms.npz = run-1 injection, fit.png)
injection.py                 simulation (one series for both pipelines), wavelet posterior and sampler, run()
                             script: python injection.py --tag run1 --injection wavelet
                                     python injection.py --tag run2 --injection sky
                                     python injection.py --tag run1 --injection wavelet --k-trace
                                       (rerun with the burn-in stored -> k_trace.npz; chains.npz untouched)
og.py                        OG pipeline on the saved series: chunk FFTs, binning, nessai
                             script: python og.py --tag run2 [--dof-factor 2]   (run from a terminal, not a notebook)
data_checks.py               input checks over 10 noise realisations (OG leakage vs taper, WDM edge pixels)
dof_check.py                 WDM pixel independence over 100 realisations: nu * Var(Y / E[Y]) = 2 per group,
                             neighbour correlations (-> outputs/checks/dof_check.npz)
likelihood_grid.py           time-frequency map of data/model per pixel and per likelihood group, galactic fraction,
                             and the effect of a 1-day gap (-> outputs/checks/likelihood_grid.png)
diagnostics.py               convergence of a wavelet run: autocorrelation times, ESS, stuck walkers
run1_inject_recover.ipynb    run 1 analysis (loads outputs/run1)
run2_sky_vs_og.ipynb         run 2 analysis and the OG comparison (loads outputs/run2)
outputs/
  reference/                 atoms.npz, fit.png
  checks/                    data_checks.png, data_checks.npz, dof_check.npz, likelihood_grid.png
  run1/                      timeseries.npz, data.h5, chains.npz, run.log, injection.png, data_spectrogram.png,
                             convergence.png, K_posterior.png, corner_psd.png, atoms.png, modulation_recovery.png,
                             k_trace.npz (K of all 128 walkers per step, burn-in included), K_trace_burnin.png,
                             wdm_timefreq.png (WDM pixels and the likelihood grid in the time-frequency plane)
  run2/                      timeseries.npz (the shared series), data.h5 (WDM), chains.npz, run.log,
                             convergence_K.png, modulation_og_vs_wavelet.png, shape_og_vs_wavelet.png,
                             corner_spectral_og_vs_wavelet.png, og_sky_corner.png
    og/                      OG run as in bahamas: data.h5, data_bin.h5, result.hdf5, nessai plots
    og_dof2x/                OG run with the Gamma dof doubled (see the run-2 report)
```

`chains.npz` files are about 95 MB each (atoms of every walker and step); they are not meant for git.
