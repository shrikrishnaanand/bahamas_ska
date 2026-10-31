# Run 4 (1 year): commands

Needs: runs 1 and 2. Run everything from the repository root.

## Python

```
python simulations/common/injection.py --config simulations/1_year/config/wavelet.yaml --tag run4 --injection sky --gap 10 20
```

Then open [run4_gap.ipynb](run4_gap.ipynb) with `RERUN = False`: it loads the outputs from `simulations/1_year/outputs/run4/`.

## Slurm

```
mkdir -p simulations/1_year/logs
sbatch simulations/1_year/run4/run4.slurm
```

Edit the `#SBATCH` lines and the environment line in [run4.slurm](run4.slurm) for your cluster first.
Log: `simulations/1_year/logs/run4_<jobid>.out`. Check the job with `squeue -u $USER`.
