#!/bin/bash
#SBATCH --job-name=1yr_run2
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=48:00:00
#SBATCH --output=simulations/1_year/logs/run2_%j.out

# 1-year run 2: the commands of run2_sky_vs_og.ipynb (notebook settings: TAG = run2); keep the two in sync.
# Submit from the repository root: sbatch simulations/1_year/run2/run2.slurm
set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
source .venv/bin/activate          # edit: your environment with bahamas installed

python simulations/common/injection.py --config simulations/1_year/config/wavelet.yaml --tag run2 --injection sky
python simulations/common/og.py --config simulations/1_year/config/wavelet.yaml --tag run2
python simulations/common/og.py --config simulations/1_year/config/wavelet.yaml --tag run2 --legacy-dof
