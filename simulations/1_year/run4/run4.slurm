#!/bin/bash
#SBATCH --job-name=1yr_run4
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=simulations/1_year/logs/run4_%j.out

# 1-year run 4: the commands of run4_gap.ipynb (notebook settings: GAP = (10.0, 20.0), buffer_pixels=1); keep the two in sync.
# Submit from the repository root: sbatch simulations/1_year/run4/run4.slurm
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate          # edit: your environment with bahamas installed

python simulations/common/injection.py --config simulations/1_year/config/wavelet.yaml --tag run4 --injection sky --gap 10 20 --gap-buffer 1
