#!/bin/bash
#SBATCH --account=aip-hervete
#SBATCH --job-name=cm-train
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=%x-%j.out
#
# Train MambaSCD-Tiny under the robustcd protocol. Resubmits itself when the
# wall-clock guard fires; the run resumes from latest.pth with the same data order.
#
#   cd $SCRATCH/robustcd/logs
#   # pilot (1.5 x budget; its first 50k iterations are exactly the seed-0 campaign run)
#   sbatch --export=ALL,SEED=0,MAX_ITERS=75000 ~/robustcd/cluster/models/changemamba/train.sbatch
#   # campaign seeds
#   sbatch --export=ALL,SEED=1 ~/robustcd/cluster/models/changemamba/train.sbatch
set -uo pipefail

SEED="${SEED:?set SEED}"
MAX_ITERS="${MAX_ITERS:-50000}"
BUDGET="${BUDGET:-50000}"
ACCUM="${ACCUM:-1}"
GUARD_MIN="${GUARD_MIN:-1380}"          # 23 h of a 24 h allocation
REPO="${ROBUSTCD_REPO:-$HOME/robustcd}"
ENV_DIR="${ROBUSTCD_GPU_ENV:-$HOME/envs/robustcd-gpu}"
CM_ROOT="${CM_ROOT:-$HOME/ext/ChangeMamba}"
W="${CM_WEIGHTS:-$HOME/weights/changemamba}"
CFG="$CM_ROOT/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml"
RUN="${RUN_DIR:-$SCRATCH/robustcd/runs/changemamba_tiny/seed${SEED}}"
SELF="$REPO/cluster/models/changemamba/train.sbatch"

module load StdEnv/2023 python/3.11 cuda/12.2
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
export PYTHONDONTWRITEBYTECODE=1
mkdir -p "$RUN"
echo "job $SLURM_JOB_ID seed $SEED max_iters $MAX_ITERS budget $BUDGET -> $RUN" | tee -a "$RUN/slurm_jobs.txt"

tar -xf "$SCRATCH/robustcd/data/SECOND.tar" -C "$SLURM_TMPDIR" || exit 1
python -m robustcd.adapters.changemamba.prepare --second-root "$SLURM_TMPDIR/SECOND" \
    --splits "$REPO/splits/SECOND" --out "$SLURM_TMPDIR/cm_SECOND" || exit 1

cd "$REPO"
python -m robustcd.adapters.changemamba.train \
    --cm-root "$CM_ROOT" --cfg "$CFG" \
    --encoder-pretrained "$W/vssm_tiny_0230_ckpt_epoch_262.pth" \
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
