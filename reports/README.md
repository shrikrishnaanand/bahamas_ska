# Reports

| Document | What it covers |
|---|---|
| [WAVELET_README.md](WAVELET_README.md) | The wavelet-domain (WDM) pipeline: the math end to end (transform, modulation atoms, Gamma likelihood and its equivalence with Eq. 9 of arXiv:2511.03604, RJMCMC), code, validation, configs |
| [WAVELET_INJECTION_RUN1.md](WAVELET_INJECTION_RUN1.md) | Run 1: injecting the best-fit wavelet modulation (30 d) and recovering it with RJMCMC (done Oct 11) |
| [WAVELET_INJECTION_RUN2.md](WAVELET_INJECTION_RUN2.md) | Run 2: one 30-day series with the true sky envelope, analysed with the wavelet RJMCMC and with the OG chunked model (done Oct 11) |
| [GF_TIME_FREQUENCY_NOTE.pdf](GF_TIME_FREQUENCY_NOTE.pdf) ([.tex](GF_TIME_FREQUENCY_NOTE.tex)) | Step-by-step guide for a master student: Part I the chunked Fourier analysis (Pozzoli et al.), Part II the wavelet (WDM) analysis: transform, atoms, pixel statistics, grouping, Gamma proof, equivalence with Eq. 9 and its dof error, RJMCMC, gaps |
| [PAPER_BUGS.md](PAPER_BUGS.md) | The Galactic envelope (`envelopes_gaussian`, `average_envelopes_gaussian`) checked term by term against Buscicchio et al. 2024: two code errors (fixed Oct 12), three misprints in the paper where the code is right, conventions |

Code and notebooks: [data/wavelet_injection/](../data/wavelet_injection/) (see its README).

Separate study (Oct 12, stages 0–1 done): bright patches of excess power on the Galaxy and what a needlet sky model
can recover in 30 days. Write-up and code in [data/needlet_patches/](../data/needlet_patches/README.md).

**Envelope fix (Oct 12).** `envelopes_gaussian` and `average_envelopes_gaussian` differed from the printed formula
of Buscicchio et al. 2024 in two places (one cosh/sinh typo, missing 3β/5β damping factors); both are fixed and
checked term by term against the paper. Runs 1–2 used the pre-fix code: they are internally consistent, but their
injected envelope is not that of a Gaussian Galaxy with the stated parameters. See [PAPER_BUGS.md](PAPER_BUGS.md).

**Redone from scratch (Oct 2026)** with sin ψ = −0.83 and with the OG pipeline run by us on the same time series as
the wavelet pipeline. Both runs are done (Oct 11).

Archived (sin ψ = −0.99, Riccardo's OG run; not rerunnable): [archive/GF_MODULATION_CODE_FLOW.md](archive/GF_MODULATION_CODE_FLOW.md)
(stage 1), [archive/WAVELET_FIT_WORKFLOW.md](archive/WAVELET_FIT_WORKFLOW.md) (stage 2); notebooks and outputs in
[data/archive/stage1-2_psi099/](../data/archive/stage1-2_psi099/).
