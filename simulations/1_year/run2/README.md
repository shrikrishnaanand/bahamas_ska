# Run 2 (1 year): commands

Needs: run 1. Run from the repository root. These are the commands the notebook runs with `RERUN = True`
(it uses `TAG = run2`); change both together.

## Python

```
python simulations/common/injection.py --config simulations/1_year/config/wavelet.yaml --tag run2 --injection sky
python simulations/common/og.py --config simulations/1_year/config/wavelet.yaml --tag run2
python simulations/common/og.py --config simulations/1_year/config/wavelet.yaml --tag run2 --legacy-dof
```

Then open [run2_sky_vs_og.ipynb](run2_sky_vs_og.ipynb) with `RERUN = False` to load the outputs (`simulations/1_year/outputs/run2/`).

## Slurm

```
mkdir -p simulations/1_year/logs
sbatch simulations/1_year/run2/run2.slurm
```

Edit the `#SBATCH` lines and the environment line in [run2.slurm](run2.slurm) for your cluster first.
Log: `simulations/1_year/logs/run2_<jobid>.out`. Check the job with `squeue -u $USER`.
