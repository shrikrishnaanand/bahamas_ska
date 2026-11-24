#!/bin/bash
#SBATCH --job-name=1yr_run1
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=24:00:00
#SBATCH --output=simulations/1_year/logs/run1_%j.out

# 1-year run 1: reference fit -> simulation + sampler -> plots (outputs/run1/plots/).
# Submit from the repository root:   sbatch simulations/1_year/run1/run1.slurm
# Override a setting without editing: sbatch --export=ALL,RUN_SAMPLER=0 simulations/1_year/run1/run1.slurm

# ---- inputs -------------------------------------------------------------------------------------------------------
CONFIG=${CONFIG:-simulations/1_year/config/wavelet.yaml}   # study settings (data, priors, nsteps, burnin, ...)
TAG=${TAG:-run1}                    # outputs go to simulations/1_year/outputs/$TAG/
K_MAX=${K_MAX:-12}                  # most atoms per channel the reference fit tries
RUN_SAMPLER=${RUN_SAMPLER:-1}       # 1: fit + simulate + sample, then plot.  0: only plot the existing outputs
BURNIN_PLOT=${BURNIN_PLOT:-0}       # 1: also plot K including the burn-in. With RUN_SAMPLER=1 the one sampler run
                                    #    stores the burn-in (no extra time). With RUN_SAMPLER=0 it reruns the sampler
                                    #    (k_trace.npz, about as long as the main run) unless k_trace.npz exists
DIAGNOSTICS=${DIAGNOSTICS:-1}       # 1: convergence table in plots/summary.txt (loads data.h5; a few minutes)
# -------------------------------------------------------------------------------------------------------------------

set -euo pipefail
cd "$SLURM_SUBMIT_DIR"
[ -f pyproject.toml ] || { echo "submit from the repository root (cd bahamas_ska; sbatch simulations/1_year/...)" >&2; exit 1; }
source .venv/bin/activate          # edit: your environment with bahamas installed
echo "inputs: CONFIG=$CONFIG TAG=$TAG K_MAX=$K_MAX RUN_SAMPLER=$RUN_SAMPLER BURNIN_PLOT=$BURNIN_PLOT DIAGNOSTICS=$DIAGNOSTICS"

if [ "$RUN_SAMPLER" = 1 ]; then
  python simulations/common/fit_reference.py --config "$CONFIG" --k-max "$K_MAX"
  store=(); [ "$BURNIN_PLOT" = 1 ] && store=(--store-burnin)   # chains.npz + k_trace.npz from one run
  python simulations/common/injection.py --config "$CONFIG" --tag "$TAG" --injection wavelet "${store[@]}"
fi

plot_args=(--config "$CONFIG" --tag "$TAG")
if [ "$BURNIN_PLOT" = 1 ]; then
  k_trace=$(python -c "import sys; sys.path.insert(0, 'simulations/common'); import injection as I; \
print(I.k_trace_path('$TAG', I.load_config('$CONFIG')))")
  if [ "$RUN_SAMPLER" = 0 ] && [ ! -f "$k_trace" ]; then
    python simulations/common/injection.py --config "$CONFIG" --tag "$TAG" --injection wavelet --k-trace
  fi
  plot_args+=(--burnin-plot)
fi
[ "$DIAGNOSTICS" = 1 ] || plot_args+=(--no-diagnostics)
python simulations/common/plot_run.py "${plot_args[@]}"
