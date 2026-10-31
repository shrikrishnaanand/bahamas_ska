# Wavelet injection and recovery over 1 year (runs 1-4)

Runs 1-4 of the [30-day study](../30_days/README.md) over one year of simulated LISA data. The code is the shared
[simulations/common/](../common/); this folder holds only its configs, notebooks and outputs. Injected values and
priors are the 30-day [sources.yaml](../30_days/config/sources.yaml).

| | injection | notebook |
|---|---|---|
| run 1 | best-fit wavelet atoms of the true envelope over the year | [run1/run1_inject_recover.ipynb](run1/run1_inject_recover.ipynb) |
| run 2 | true sky envelope $M_c^2(t)$; wavelet vs OG on the same series | [run2/run2_sky_vs_og.ipynb](run2/run2_sky_vs_og.ipynb) |
| run 3 | run 2's series with a 1-day gap (12-13 d) | [run3/run3_gap.ipynb](run3/run3_gap.ipynb) |
| run 4 | run 2's series with a 10-day gap (10-20 d) | [run4/run4_gap.ipynb](run4/run4_gap.ipynb) |

Run them in order: run 1 makes the reference fit (`outputs/reference/`) that runs 2-4 load, and runs 3 and 4 compare
with run 2's chains.

Settings, differing from the 30-day study only where the window needs it:

```
config/wavelet.yaml      T = 365.25 d (876 pixels of 10 h = 365.0 d analysed), atom widths 5-120 d, up to 16 atoms,
                         out: outputs/, og_config: config/og_config.yaml
config/og_config.yaml    OG pipeline: T = 365.0 d in 24 chunks of 15.21 d (the chunks must tile the series)
```

From the repository root (the notebooks do the same with `RERUN = True`, except the OG run):

```
python simulations/common/fit_reference.py --config simulations/1_year/config/wavelet.yaml --k-max 12
python simulations/common/injection.py --config simulations/1_year/config/wavelet.yaml --tag run1 --injection wavelet
python simulations/common/injection.py --config simulations/1_year/config/wavelet.yaml --tag run2 --injection sky
python simulations/common/og.py        --config simulations/1_year/config/wavelet.yaml --tag run2 [--legacy-dof]
python simulations/common/injection.py --config simulations/1_year/config/wavelet.yaml --tag run3 --injection sky --gap 12 13
python simulations/common/injection.py --config simulations/1_year/config/wavelet.yaml --tag run4 --injection sky --gap 10 20
```

Each run folder has a README with its commands and a Slurm script (`sbatch simulations/1_year/runN/runN.slurm`,
after `mkdir -p simulations/1_year/logs`). Every sampler sees about 12 times more data than over 30 days, so expect
hours per run on 8 cores.
