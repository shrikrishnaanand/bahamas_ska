# Reports

| Document | What it covers |
|---|---|
| [WAVELET_INJECTION_RUN1.md](WAVELET_INJECTION_RUN1.md) | Run 1: injecting the best-fit wavelet modulation (30 d) and recovering it with RJMCMC (redone Oct 16 with the fixed code) |
| [WAVELET_INJECTION_RUN2.md](WAVELET_INJECTION_RUN2.md) | Run 2: one 30-day series with the true sky envelope, analysed with the wavelet RJMCMC and with the OG chunked model (redone Oct 16 with the fixed code) |
| [WAVELET_INJECTION_RUN3.md](WAVELET_INJECTION_RUN3.md) | Run 3: run 2's series with a 1-day gap, wavelet RJMCMC with the gap pixels left out of the likelihood, compared with run 2 (Oct 16) |
| [WAVELET_INJECTION_RUN4.md](WAVELET_INJECTION_RUN4.md) | Run 4: as run 3 with a 10-day gap (10–20 d), compared with run 2 (Oct 17) |
| [GF_TIME_FREQUENCY_NOTE.pdf](GF_TIME_FREQUENCY_NOTE.pdf) ([.tex](GF_TIME_FREQUENCY_NOTE.tex)) | Step-by-step guide for a master student: Part I the chunked Fourier analysis (Pozzoli et al.), Part II the wavelet (WDM) analysis: transform, atoms, pixel statistics, grouping, Gamma proof, equivalence with Eq. 9 and its dof error, RJMCMC, gaps |
| [PAPER_BUGS.md](PAPER_BUGS.md) | bahamas checked against Klein et al. and Pozzoli et al.: envelope errors F1–F2 and code bugs B1–B7 (all fixed, with status and checks), misprints in the papers, open questions, conventions |

Code and notebooks: [data/wavelet_injection/](../data/wavelet_injection/) (see its README).

Separate study (Oct 12, stages 0–1 done): bright patches of excess power on the Galaxy and what a needlet sky model
can recover in 30 days. Write-up and code in [data/needlet_patches/](../data/needlet_patches/README.md).

**Runs redone with the fixed code (Oct 2026).** The first runs 1 and 2 (Oct 11) used the envelope before the F1/F2
fix; their reports were removed and their outputs archived in `data/archive/wavelet_runs_prefix_oct11/`. Both runs were
redone on Oct 16. The pipeline itself is
described in [GF_TIME_FREQUENCY_NOTE.pdf](GF_TIME_FREQUENCY_NOTE.pdf), Part II.

Archived (sin ψ = −0.99, Riccardo's OG run; not rerunnable): [archive/GF_MODULATION_CODE_FLOW.md](archive/GF_MODULATION_CODE_FLOW.md)
(stage 1), [archive/WAVELET_FIT_WORKFLOW.md](archive/WAVELET_FIT_WORKFLOW.md) (stage 2); notebooks and outputs in
[data/archive/stage1-2_psi099/](../data/archive/stage1-2_psi099/).
