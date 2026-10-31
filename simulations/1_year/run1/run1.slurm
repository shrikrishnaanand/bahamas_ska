#!/bin/bash
#SBATCH --job-name=1yr_run1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=simulations/1_year/logs/run1_%j.out

# Run 1-year run 1. Submit from the repository root: sbatch simulations/1_year/run1/run1.slurm
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate          # edit: your environment with bahamas installed

python simulations/common/fit_reference.py --config simulations/1_year/config/wavelet.yaml --k-max 12
python simulations/common/injection.py --config simulations/1_year/config/wavelet.yaml --tag run1 --injection wavelet

# optional: burn-in K trace
# python simulations/common/injection.py --config simulations/1_year/config/wavelet.yaml --tag run1 --injection wavelet --k-trace
