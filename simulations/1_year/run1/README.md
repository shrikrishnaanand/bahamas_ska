# Run 1 (1 year): commands

Needs: nothing. Run from the repository root. These are the commands the notebook runs with `RERUN = True`
(it uses `K_MAX = 12`); change both together.

## Python

```
python simulations/common/fit_reference.py --config simulations/1_year/config/wavelet.yaml --k-max 12
python simulations/common/injection.py --config simulations/1_year/config/wavelet.yaml --tag run1 --injection wavelet
```

Then open [run1_inject_recover.ipynb](run1_inject_recover.ipynb) with `RERUN = False` to load the outputs (`simulations/1_year/outputs/run1/`).

## Slurm

```
mkdir -p simulations/1_year/logs
sbatch simulations/1_year/run1/run1.slurm
```

Edit the `#SBATCH` lines and the environment line in [run1.slurm](run1.slurm) for your cluster first.
Log: `simulations/1_year/logs/run1_<jobid>.out`. Check the job with `squeue -u $USER`.
