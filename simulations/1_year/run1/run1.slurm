#!/bin/bash
#SBATCH --job-name=1yr_run1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=simulations/1_year/logs/run1_%j.out

# 1-year run 1: the commands of run1_inject_recover.ipynb (notebook settings: K_MAX = 12); keep the two in sync.
# Submit from the repository root: sbatch simulations/1_year/run1/run1.slurm
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate          # edit: your environment with bahamas installed

python simulations/common/fit_reference.py --config simulations/1_year/config/wavelet.yaml --k-max 12
python simulations/common/injection.py --config simulations/1_year/config/wavelet.yaml --tag run1 --injection wavelet
