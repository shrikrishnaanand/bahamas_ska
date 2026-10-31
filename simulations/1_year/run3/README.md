# Run 3 (1 year): commands

Needs: runs 1 and 2. Run everything from the repository root.

## Python

```
python simulations/common/injection.py --config simulations/1_year/config/wavelet.yaml --tag run3 --injection sky --gap 12 13
```

Optional (burn-in K trace, as long again as the wavelet run):

```
python simulations/common/injection.py --config simulations/1_year/config/wavelet.yaml --tag run3 --injection sky --gap 12 13 --k-trace
```

Then open [run3_gap.ipynb](run3_gap.ipynb) with `RERUN = False`: it loads the outputs from `simulations/1_year/outputs/run3/`.

## Slurm

```
mkdir -p simulations/1_year/logs
sbatch simulations/1_year/run3/run3.slurm
```

Edit the `#SBATCH` lines and the environment line in [run3.slurm](run3.slurm) for your cluster first.
Log: `simulations/1_year/logs/run3_<jobid>.out`. Check the job with `squeue -u $USER`.
