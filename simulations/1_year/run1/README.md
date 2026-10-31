# Run 1 (1 year): commands

Needs: nothing. Run everything from the repository root.

## Python

```
python simulations/common/fit_reference.py --config simulations/1_year/config/wavelet.yaml --k-max 12
python simulations/common/injection.py --config simulations/1_year/config/wavelet.yaml --tag run1 --injection wavelet
```

Optional (burn-in K trace, as long again as the wavelet run):

```
python simulations/common/injection.py --config simulations/1_year/config/wavelet.yaml --tag run1 --injection wavelet --k-trace
```

Then open [run1_inject_recover.ipynb](run1_inject_recover.ipynb) with `RERUN = False`: it loads the outputs from `simulations/1_year/outputs/run1/`.

## Slurm

```
mkdir -p simulations/1_year/logs
sbatch simulations/1_year/run1/run1.slurm
```

Edit the `#SBATCH` lines and the environment line in [run1.slurm](run1.slurm) for your cluster first.
Log: `simulations/1_year/logs/run1_<jobid>.out`. Check the job with `squeue -u $USER`.
