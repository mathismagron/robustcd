#!/bin/bash
#SBATCH --account=aip-hervete
#SBATCH --job-name=fcs-train
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=%x-%j.out
#
# Train Mamba-FCS under the robustcd protocol. Resubmits itself when the
# wall-clock guard fires; the run resumes from latest.pth with the same data order.
# StepLR is step-wise, so, as for ChangeMamba, the first 50k iterations of a
# 75k pilot are exactly the seed-0 campaign run.
#
#   cd $SCRATCH/robustcd/logs
#   sbatch --export=ALL,SEED=0,MAX_ITERS=75000 ~/robustcd/cluster/models/mambafcs/train.sbatch   # pilot = seed 0
#   sbatch --export=ALL,SEED=1 ~/robustcd/cluster/models/mambafcs/train.sbatch
set -uo pipefail

SEED="${SEED:?set SEED}"
MAX_ITERS="${MAX_ITERS:-50000}"
BUDGET="${BUDGET:-50000}"
ACCUM="${ACCUM:-4}"                      # set from validate.sbatch timing (micro-batch 16/ACCUM)
GUARD_MIN="${GUARD_MIN:-1380}"          # 23 h of a 24 h allocation
REPO="${ROBUSTCD_REPO:-$HOME/robustcd}"
ENV_DIR="${ROBUSTCD_GPU_ENV:-$HOME/envs/robustcd-gpu}"
FCS="${ROBUSTCD_EXT:-$HOME/ext}/MambaFCS"
W="${FCS_WEIGHTS:-$HOME/weights/mambafcs}"
RUN="${RUN_DIR:-$SCRATCH/robustcd/runs/mambafcs/seed${SEED}}"
SELF="$REPO/cluster/models/mambafcs/train.sbatch"

module load StdEnv/2023 python/3.11 cuda/12.2
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RUN"
echo "$(date -Is) job $SLURM_JOB_ID seed $SEED max_iters $MAX_ITERS budget $BUDGET accum $ACCUM -> $RUN" | tee -a "$RUN/slurm_jobs.txt"

tar -xf "$SCRATCH/robustcd/data/SECOND.tar" -C "$SLURM_TMPDIR" || exit 1
python -m robustcd.adapters.changemamba.prepare --second-root "$SLURM_TMPDIR/SECOND" \
    --splits "$REPO/splits/SECOND" --out "$SLURM_TMPDIR/cm_SECOND" || exit 1

cd "$REPO"
python -W ignore::UserWarning -m robustcd.adapters.mambafcs.train --fcs-root "$FCS" \
    --backbone "$W/vssm_base_0229_ckpt_epoch_237.pth" \
    --second-root "$SLURM_TMPDIR/SECOND" --cm-data "$SLURM_TMPDIR/cm_SECOND/train" \
    --splits "$REPO/splits/SECOND" --out "$RUN" --seed "$SEED" \
    --max-iters "$MAX_ITERS" --budget-iters "$BUDGET" --batch-size 16 --accum "$ACCUM" \
    --eval-interval 2000 --workers "${SLURM_CPUS_PER_TASK:-8}" --stop-after-min "$GUARD_MIN"
rc=$?
if [ "$rc" = 3 ]; then
    echo "time guard hit, resubmitting"
    sbatch --export=ALL,SEED="$SEED",MAX_ITERS="$MAX_ITERS",BUDGET="$BUDGET",ACCUM="$ACCUM" "$SELF"
    exit 0
fi
exit "$rc"
