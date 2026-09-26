#!/bin/bash
#SBATCH --account=aip-hervete
#SBATCH --job-name=ding-train
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=%x-%j.out
#
# Train a Ding-codebase model under the robustcd protocol. Resubmits itself when
# the wall-clock guard fires; the run resumes from latest.pth with the same data order.
#
#   cd $SCRATCH/robustcd/logs
#   sbatch --export=ALL,MODEL=scannet,SEED=0 ~/robustcd/cluster/models/ding/train.sbatch
#   # pilot (separate run: the poly schedule depends on its total length)
#   sbatch --export=ALL,MODEL=scannet,SEED=0,MAX_ITERS=75000,RUN_NAME=pilot75k ~/robustcd/cluster/models/ding/train.sbatch
set -uo pipefail

MODEL="${MODEL:?set MODEL (scannet|ted|bisrnet|hrscd4|sscdl)}"
SEED="${SEED:?set SEED}"
MAX_ITERS="${MAX_ITERS:-50000}"
BUDGET="${BUDGET:-50000}"
ACCUM="${ACCUM:-1}"
GUARD_MIN="${GUARD_MIN:-1380}"          # 23 h of a 24 h allocation
REPO="${ROBUSTCD_REPO:-$HOME/robustcd}"
ENV_DIR="${ROBUSTCD_GPU_ENV:-$HOME/envs/robustcd-gpu}"
EXT="${ROBUSTCD_EXT:-$HOME/ext}"
export TORCH_HOME="${TORCH_HOME:-$HOME/weights/torch}"
case "$MODEL" in
    scannet|ted) UP="$EXT/SCanNet" ;;
    bisrnet|hrscd4|sscdl) UP="$EXT/Bi-SRNet" ;;
    *) echo "unknown MODEL $MODEL"; exit 1 ;;
esac
RUN_NAME="${RUN_NAME:-seed${SEED}}"
RUN="${RUN_DIR:-$SCRATCH/robustcd/runs/$MODEL/$RUN_NAME}"
SELF="$REPO/cluster/models/ding/train.sbatch"

module load StdEnv/2023 python/3.11 cuda/12.2
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RUN"
echo "$(date -Is) job $SLURM_JOB_ID model $MODEL seed $SEED max_iters $MAX_ITERS budget $BUDGET accum $ACCUM -> $RUN" \
    | tee -a "$RUN/slurm_jobs.txt"

tar -xf "$SCRATCH/robustcd/data/SECOND.tar" -C "$SLURM_TMPDIR" || exit 1

cd "$REPO"
python -W ignore::UserWarning -m robustcd.adapters.ding.train --model "$MODEL" --repo "$UP" \
    --second-root "$SLURM_TMPDIR/SECOND" --splits "$REPO/splits/SECOND" --out "$RUN" --seed "$SEED" \
    --max-iters "$MAX_ITERS" --budget-iters "$BUDGET" --batch-size 16 --accum "$ACCUM" \
    --eval-interval 2000 --workers "${SLURM_CPUS_PER_TASK:-8}" --stop-after-min "$GUARD_MIN"
rc=$?
if [ "$rc" = 3 ]; then
    echo "time guard hit, resubmitting"
    sbatch --export=ALL,MODEL="$MODEL",SEED="$SEED",MAX_ITERS="$MAX_ITERS",BUDGET="$BUDGET",ACCUM="$ACCUM",RUN_NAME="$RUN_NAME" "$SELF"
    exit 0
fi
exit "$rc"
