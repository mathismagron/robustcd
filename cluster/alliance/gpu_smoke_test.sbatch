#!/bin/bash
#SBATCH --account=aip-hervete
#SBATCH --job-name=gpu-smoke
#SBATCH --time=00:20:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --output=%x-%j.out
#
# Validate the GPU env on one L40S, and time the staging of SECOND to node-local disk.
#   cd $SCRATCH/robustcd/logs && sbatch ~/robustcd/cluster/alliance/gpu_smoke_test.sbatch
set -euo pipefail

REPO="${ROBUSTCD_REPO:-$HOME/robustcd}"
ENV_DIR="${ROBUSTCD_GPU_ENV:-$HOME/envs/robustcd-gpu}"
PY_MODULE="${ROBUSTCD_PY_MODULE:-python/3.11}"

module load StdEnv/2023 "$PY_MODULE"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
export PYTHONDONTWRITEBYTECODE=1

nvidia-smi --query-gpu=name,driver_version,memory.total --format=csv

t0=$(date +%s)
tar -xf "$SCRATCH/robustcd/data/SECOND.tar" -C "$SLURM_TMPDIR"
echo "staging SECOND.tar -> \$SLURM_TMPDIR: $(( $(date +%s) - t0 )) s"

python "$REPO/scripts/gpu_smoke_test.py" --data "$SLURM_TMPDIR/SECOND" \
    --out "$SCRATCH/robustcd/checks/gpu_smoke_${SLURM_JOB_ID}.json"
