# Cheatsheet: wavelet fit, original pipeline, runs 1 and 2

Every command is run from the **repository root** with the project environment active (`source .venv/bin/activate`).
Each study has its own config; pick one and use it in every command:

| study | `--config` |
|---|---|
| 30 days | `simulations/30_days/config/wavelet.yaml` |
| 1 year | `simulations/1_year/config/wavelet.yaml` |

Outputs go to `simulations/<study>/outputs/`. Below, `CFG` stands for the config path, e.g.
`CFG=simulations/1_year/config/wavelet.yaml`.

> **Known issue: NumPy ≥ 2.4.** eryn (the wavelet sampler, used in sections 3 and 4) calls `np.in1d`, which NumPy
> 2.4 removed, so every wavelet run stops with `AttributeError: module 'numpy' has no attribute 'in1d'`. Use an
> environment with `numpy<2.4`, or define the alias before eryn runs: `np.in1d = np.isin` (same result for eryn's 1-D
> arrays). Sections 1 and 2 are not affected.

## Slurm

The 1-year runs have ready-made scripts (`simulations/1_year/runN/runN.slurm`). For everything else, either wrap the
command:

```bash
mkdir -p logs
sbatch --job-name=wavelet --cpus-per-task=8 --mem=32G --time=24:00:00 --output=logs/%x_%j.out \
       --wrap "source .venv/bin/activate && <command>"
```

or copy this template to `job.slurm`, put the commands at the bottom and run `sbatch job.slurm`:

```bash
#!/bin/bash
#SBATCH --job-name=wavelet
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=logs/%x_%j.out
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"              # submit from the repository root (mkdir -p logs first)
source .venv/bin/activate           # edit: your environment with bahamas installed

# commands here
```

Check jobs with `squeue -u $USER`; the log is `logs/<job-name>_<jobid>.out`.

---

## 1. Wavelet fit of the modulation alone (no data, no sampler)

Fits $\ln P_c(t) = b_c + \sum_k a_k e^{-(t-t_k)^2/2\tau_k^2}$ to the true envelope $P_c(t) = M_c^2(t)$ of the sky
parameters in `simulations/30_days/config/sources.yaml`, for K = 0, 1, … atoms, and keeps the smallest K with < 1% rms
error. Takes seconds to a few minutes.

```bash
python simulations/common/fit_reference.py --config simulations/30_days/config/wavelet.yaml            # K up to 4
python simulations/common/fit_reference.py --config simulations/1_year/config/wavelet.yaml --k-max 12  # 1 year
```

| output | what |
|---|---|
| `outputs/reference/atoms.npz` | fitted atoms and levels $b_A$, $b_E$, true amp (the run-1 injection) |
| `outputs/reference/fit.png` | **the plot**: true envelope vs fit, relative error, atoms |

To compare the wavelet fit with a Legendre series (time, FFT and WDM views; writes only to `<study>/fit_curve/`,
never to `outputs/reference/`, ~20 s):

```bash
python simulations/common/fit_curve.py --config $CFG [--k-max 12] [--n-max 20]
```

Plots: `<study>/fit_curve/compare.png` and `<study>/fit_curve/{wavelet_atoms,legendre}/{fit,fft,wdm}.png`.

Slurm (not needed, it is fast): `sbatch ... --wrap "source .venv/bin/activate && python simulations/common/fit_reference.py --config $CFG"`.

## 2. The original (OG) pipeline alone, to get the modulation

The original chunked bahamas pipeline: it simulates each chunk in the frequency domain with the chunk-averaged sky
envelope and samples the spectral parameters and the sky parameters (lat, long, s1, s2, psi) with nessai. The
modulation is $P_c(t) = M_c^2(t)$ of the sky-parameter posterior.

**Config.** Copy the study's OG config and set the five file paths (they are `null` there because `og.py` fills
them in):

```bash
mkdir -p my_og
cp simulations/30_days/config/og_config.yaml my_og/og.yaml     # or simulations/1_year/config/og_config.yaml
# in my_og/og.yaml set:
#   file: 'my_og/data_whittle'          fileAV: 'my_og/data_gamma'          folder_plot: 'my_og/'
#   inference: file: 'my_og/data_gamma'   inference: file_post: 'my_og/result.npz'
```

**Run** (data: seconds; inference: ~15 min on 8 cores for 30 days, hours for a year):

```bash
bahamas-data      --config my_og/og.yaml --sources simulations/30_days/config/sources.yaml
bahamas-inference --config my_og/og.yaml --sources simulations/30_days/config/sources.yaml --use_jax False
```

`--use_jax False` is needed with `n_pool > 1` (nessai forks worker processes, which deadlock under JAX).

| output | what |
|---|---|
| `my_og/mod_gal.png`, `my_og/data_data_gamma.png` | injected modulation and binned data (from `bahamas-data`) |
| `my_og/result.hdf5` | nessai posterior (`posterior_samples`), plus nessai's own plots in `my_og/` |
| `my_og/result.npz` | the same samples as a plain array |

**Plot the posterior modulation:**

```python
import h5py, numpy as np, matplotlib.pyplot as plt
from bahamas.wavelet.simulate import power_modulation, ENVELOPE_PARAMS

with h5py.File('my_og/result.hdf5') as f:
    s = f['posterior_samples'][()]
t = np.linspace(0, 30 * 86400, 500)                      # T of the config
draws = np.array([[np.asarray(power_modulation({k: float(x[k]) for k in ENVELOPE_PARAMS}, t, c)) for c in (0, 1)]
                  for x in s[np.random.default_rng(0).choice(len(s), 300, replace=False)]])
for c, name in enumerate('AE'):
    q = np.percentile(draws[:, c], [5, 50, 95], axis=0)
    plt.fill_between(t / 86400, q[0], q[2], alpha=0.4, label=f'{name} 90%')
    plt.plot(t / 86400, q[1])
plt.xlabel('time [days]'); plt.ylabel('$P_c(t) = M_c^2(t)$'); plt.legend(); plt.savefig('my_og/modulation.png')
```

Slurm: put the two `bahamas-*` lines in the template above (8 cores; `n_pool` in the config should match).

## 3. Run 1: inject the wavelet fit, recover it with the wavelet RJMCMC

Data: a continuous series whose galactic modulation is the section-1 wavelet fit (true amp, fitted levels), noise
seed from the config, transformed to the WDM domain. Inference: eryn with reversible jump over the number of atoms
per channel; amp, b_A, b_E start from their priors (`inference: init_from_prior`). A recovery test: the model can
represent the injection exactly.

**Needs:** the reference fit (`outputs/reference/atoms.npz`), i.e. the first command below.

```bash
python simulations/common/fit_reference.py --config $CFG [--k-max 12]        # skip if outputs/reference/ exists
python simulations/common/injection.py     --config $CFG --tag run1 --injection wavelet
```

Options: `--rerun-data` (simulate again even if `outputs/run1/data.h5` exists), `--nsteps`, `--burnin`, `--ntemps`
(override the config), `--k-trace` (rerun storing the burn-in: `outputs/run1/k_trace.npz`, K of every walker per step).
Time: ~25 min on 8 cores for 30 days, hours for a year.

| output (`outputs/run1/`) | what |
|---|---|
| `timeseries.npz`, `data.h5` | the series and its WDM coefficients |
| `chains.npz` | samples: spectral parameters and levels, atoms, K, log L, `modulation_draws`, `P_true` |

**Plots:** open `simulations/<study>/run1/run1_inject_recover.ipynb` with `RERUN = False` (it loads the outputs;
with `RERUN = True` it runs the two commands itself). It writes the injection, convergence, K posterior, corner,
atoms and modulation-recovery figures to `outputs/run1/`.

**Slurm:** 1 year: `mkdir -p simulations/1_year/logs && sbatch simulations/1_year/run1/run1.slurm`
(it runs both commands with `--k-max 12`). 30 days: the template with the two commands above.

## 4. Run 2: data from the original model's envelope, wavelet posterior vs OG posterior

Data: one continuous series whose modulation is the true sky envelope $M_c^2(t)$ of the original (OG) model, true
amplitude. The **wavelet** RJMCMC analyses it in the WDM domain; the **OG** pipeline is run on the same series (cut into
its chunks, Gamma likelihood, nessai), so both see the same noise. The wavelet model can only approximate this
envelope.

**Needs:** the reference fit (section 1; it gives the truth for b_A, b_E). The OG step needs the series written by the
wavelet step, so keep the order.

```bash
python simulations/common/injection.py --config $CFG --tag run2 --injection sky      # simulate + wavelet RJMCMC
python simulations/common/og.py        --config $CFG --tag run2                      # OG, correct dof
python simulations/common/og.py        --config $CFG --tag run2 --legacy-dof         # OG, paper dof (optional)
```

`og.py` options: `--nlive`, `--n-pool`, `--data-only` (write the chunk data only). Run `og.py` from a terminal, not a
notebook (nessai's worker pool forks the process). Time: ~25 + 15 min on 8 cores for 30 days, hours for a year.

| output (`outputs/run2/`) | what |
|---|---|
| `timeseries.npz`, `data.h5`, `chains.npz` | the shared series, WDM data, wavelet samples |
| `og/result.hdf5`, `og_legacy/result.hdf5` | OG posteriors (and the chunk data `og/data.h5`, `og/data_bin.h5`) |

**Plots:** open `simulations/<study>/run2/run2_sky_vs_og.ipynb` with `RERUN = False`: wavelet vs OG modulation,
spectral-parameter corners, shape comparison.

**Slurm:** 1 year: `mkdir -p simulations/1_year/logs && sbatch simulations/1_year/run2/run2.slurm` (all three
commands). 30 days: the template with the three commands above.

---

Runs 3–5 (gaps, patchy sky) are described in the study READMEs: [30 days](30_days/README.md),
[1 year](1_year/README.md).
