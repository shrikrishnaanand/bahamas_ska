# Reports

| Document | What it covers |
|---|---|
| [WAVELET_README.md](WAVELET_README.md) | The wavelet-domain (WDM) pipeline: the math end to end (transform, modulation atoms, Gamma likelihood and its equivalence with Eq. 9 of arXiv:2511.03604, RJMCMC), code, validation, configs |
| [WAVELET_INJECTION_RUN1.md](WAVELET_INJECTION_RUN1.md) | Run 1: injecting the best-fit wavelet modulation (30 d) and recovering it with RJMCMC (done Oct 11) |
| [WAVELET_INJECTION_RUN2.md](WAVELET_INJECTION_RUN2.md) | Run 2: one 30-day series with the true sky envelope, analysed with the wavelet RJMCMC and with the OG chunked model (done Oct 11) |

Code and notebooks: [data/wavelet_injection/](../data/wavelet_injection/) (see its README).

**Redone from scratch (Oct 2026)** with sin ψ = −0.83 and with the OG pipeline run by us on the same time series as
the wavelet pipeline. Both runs are done (Oct 11).

Archived (sin ψ = −0.99, Riccardo's OG run; not rerunnable): [archive/GF_MODULATION_CODE_FLOW.md](archive/GF_MODULATION_CODE_FLOW.md)
(stage 1), [archive/WAVELET_FIT_WORKFLOW.md](archive/WAVELET_FIT_WORKFLOW.md) (stage 2); notebooks and outputs in
[data/archive/stage1-2_psi099/](../data/archive/stage1-2_psi099/).
