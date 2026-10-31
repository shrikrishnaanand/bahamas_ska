# Run 2 (1 year): commands

Needs: run 1 (outputs/reference/). Run everything from the repository root.

## Python

```
python simulations/common/injection.py --config simulations/1_year/config/wavelet.yaml --tag run2 --injection sky
python simulations/common/og.py --config simulations/1_year/config/wavelet.yaml --tag run2
python simulations/common/og.py --config simulations/1_year/config/wavelet.yaml --tag run2 --legacy-dof
```

Optional (burn-in K trace, as long again as the wavelet run):

```
python simulations/common/injection.py --config simulations/1_year/config/wavelet.yaml --tag run2 --injection sky --k-trace
```

Then open [run2_sky_vs_og.ipynb](run2_sky_vs_og.ipynb) with `RERUN = False`: it loads the outputs from `simulations/1_year/outputs/run2/`.

## Slurm

```
mkdir -p simulations/1_year/logs
sbatch simulations/1_year/run2/run2.slurm
```

Edit the `#SBATCH` lines and the environment line in [run2.slurm](run2.slurm) for your cluster first.
Log: `simulations/1_year/logs/run2_<jobid>.out`. Check the job with `squeue -u $USER`.
