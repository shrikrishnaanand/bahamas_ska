# Stage 3: wavelet injection and recovery on the cyclo pack-2 setup

Wavelet-domain RJMCMC (`bahamas.wavelet`) on 30 days of simulated LISA data with the same settings as Riccardo's
pack-2 chunked run (`cyclo_riccardo/cyclo/`), and a comparison with that run's posterior.

| | injection | purpose | notebook | report |
|---|---|---|---|---|
| run 1 | stage-2 best-fit wavelet atoms (A: 2, E: 1) | can the RJMCMC recover a modulation the model represents exactly? | [run1_inject_recover.ipynb](run1_inject_recover.ipynb) | [WAVELET_INJECTION_RUN1.md](../../reports/WAVELET_INJECTION_RUN1.md) |
| run 2 | true sky envelope $M_c^2(t)$, true amplitude | wavelet vs OG chunked posterior on $P_c(t)$: which is broader? | [run2_sky_vs_og.ipynb](run2_sky_vs_og.ipynb) | [WAVELET_INJECTION_RUN2.md](../../reports/WAVELET_INJECTION_RUN2.md) |

Both runs use the same noise seed, priors and sampler settings (32 walkers × 4 temperatures, 2000 + 6000 steps, start
at K = 0; about 35 min each on 8 CPU cores).

```
injection.py                 all non-plotting code: config, injections, simulation, posterior, sampler, run()
                             script: python injection.py --tag run1 --injection wavelet
                                     python injection.py --tag run2 --injection sky
run1_inject_recover.ipynb    run 1 analysis (loads outputs/run1; RERUN = True redoes everything)
run2_sky_vs_og.ipynb         run 2 analysis and the OG comparison (loads outputs/run2 and cyclo_riccardo/cyclo/result.hdf5)
outputs/
  run1/                      data.h5 (WDM coefficients + injected P), chains.npz, run.log, figures
  run2/                      same for run 2
  comparison/                OG vs wavelet figures: modulation_og_vs_wavelet.png, shape_og_vs_wavelet.png,
                             corner_spectral_og_vs_wavelet.png
```

`chains.npz` files are about 95 MB each (atoms of every walker and step); they are not meant for git.
