# Run 1 (1 year): commands

Needs: nothing. Run from the repository root.

## Slurm (everything: fit, sampler, plots)

```
sbatch simulations/1_year/run1/run1.slurm
```

The inputs are at the top of [run1.slurm](run1.slurm): `CONFIG`, `TAG`, `K_MAX` (12), `RUN_SAMPLER` (1; 0 only plots
the existing outputs), `BURNIN_PLOT` (0; 1 adds the K trace including the burn-in, which reruns the sampler storing the
burn-in unless `k_trace.npz` exists) and `DIAGNOSTICS` (1). Override one without editing the file:

```
sbatch --export=ALL,RUN_SAMPLER=0 simulations/1_year/run1/run1.slurm                 # plots only
sbatch --export=ALL,RUN_SAMPLER=0,BURNIN_PLOT=1 simulations/1_year/run1/run1.slurm   # plots + burn-in K trace
```

Log: `simulations/1_year/logs/run1_<jobid>.out`. Check the job with `squeue -u $USER`.

## Python

```
python simulations/common/fit_reference.py --config simulations/1_year/config/wavelet.yaml --k-max 12
python simulations/common/injection.py --config simulations/1_year/config/wavelet.yaml --tag run1 --injection wavelet
python simulations/common/plot_run.py  --config simulations/1_year/config/wavelet.yaml --tag run1 [--burnin-plot] [--no-diagnostics]
```

[plot_run.py](../../common/plot_run.py) only reads the saved outputs and writes the plots and `summary.txt` (the printed
numbers) to `simulations/1_year/outputs/run1/plots/`. `--burnin-plot` needs `k_trace.npz` first:
`python simulations/common/injection.py --config simulations/1_year/config/wavelet.yaml --tag run1 --injection wavelet --k-trace`.

The notebook [run1_inject_recover.ipynb](run1_inject_recover.ipynb) (`RERUN = False`, `K_MAX = 12`) shows the same
plots interactively.
