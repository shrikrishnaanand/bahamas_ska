# Reports

| Document | What it covers |
|---|---|
| [WAVELET_README.md](WAVELET_README.md) | The wavelet-domain (WDM) pipeline: model, transform, likelihood, sampling, configs |
| [GF_MODULATION_CODE_FLOW.md](GF_MODULATION_CODE_FLOW.md) | Stage 1: how the Fig. 7-type modulated GF is built from the cyclo pack-2 parameters, and which function each piece comes from |
| [WAVELET_FIT_WORKFLOW.md](WAVELET_FIT_WORKFLOW.md) | Stage 2: fitting the wavelet modulation model to the true GF envelope |
| [WAVELET_INJECTION_RUN1.md](WAVELET_INJECTION_RUN1.md) | Stage 3, run 1: injecting the best-fit wavelet modulation (pack 2) and recovering it with RJMCMC |
| [WAVELET_INJECTION_RUN2.md](WAVELET_INJECTION_RUN2.md) | Stage 3, run 2: the true sky envelope analysed with the wavelet RJMCMC, compared with Riccardo's chunked (OG) posterior |

Notebooks: [data/cyclo_fig7_modulation.ipynb](../data/cyclo_fig7_modulation.ipynb) (stage 1),
[data/wavelet_modulation_fit.ipynb](../data/wavelet_modulation_fit.ipynb) (stage 2),
[data/wavelet_injection/run1_inject_recover.ipynb](../data/wavelet_injection/run1_inject_recover.ipynb) (stage 3, run 1),
[data/wavelet_injection/run2_sky_vs_og.ipynb](../data/wavelet_injection/run2_sky_vs_og.ipynb) (stage 3, run 2).
