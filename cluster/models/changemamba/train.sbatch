#!/bin/bash
#SBATCH --account=aip-hervete
#SBATCH --job-name=cm-train
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=%x-%j.out
#
# Train MambaSCD-Tiny (VARIANT=tiny, default) or MambaSCD-Base (VARIANT=base) under the robustcd protocol. Resubmits itself when the
# wall-clock guard fires; the run resumes from latest.pth with the same data order.
#
#   cd $SCRATCH/robustcd/logs
#   # pilot (1.5 x budget; its first 50k iterations are exactly the seed-0 campaign run)
#   sbatch --export=ALL,SEED=0,MAX_ITERS=75000 ~/robustcd/cluster/models/changemamba/train.sbatch
#   # campaign seeds
#   sbatch --export=ALL,SEED=1 ~/robustcd/cluster/models/changemamba/train.sbatch
#   # MambaSCD-Base (ACCUM from validate_base.sbatch timing)
#   sbatch --export=ALL,VARIANT=base,SEED=0,MAX_ITERS=75000,ACCUM=4 ~/robustcd/cluster/models/changemamba/train.sbatch
set -uo pipefail

SEED="${SEED:?set SEED}"
MAX_ITERS="${MAX_ITERS:-50000}"
BUDGET="${BUDGET:-50000}"
ACCUM="${ACCUM:-2}"                     # batch 16 in one pass OOMs on the 44 GiB L40S (41.7 GiB at the first forward);
                                        # 2 x 8 keeps the effective batch at 16 (BatchNorm then sees 8 per pass)
GUARD_MIN="${GUARD_MIN:-1380}"          # 23 h of a 24 h allocation
REPO="${ROBUSTCD_REPO:-$HOME/robustcd}"
ENV_DIR="${ROBUSTCD_GPU_ENV:-$HOME/envs/robustcd-gpu}"
CM_ROOT="${CM_ROOT:-$HOME/ext/ChangeMamba}"
W="${CM_WEIGHTS:-$HOME/weights/changemamba}"
VARIANT="${VARIANT:-tiny}"
case "$VARIANT" in
    tiny) CFG="$CM_ROOT/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml"; ENC="$W/vssm_tiny_0230_ckpt_epoch_262.pth" ;;
    base) CFG="$CM_ROOT/changedetection/configs/vssm1/vssm_base_224.yaml"; ENC="$W/vssm_base_0229_ckpt_epoch_237.pth" ;;
    *) echo "unknown VARIANT $VARIANT"; exit 1 ;;
esac
RUN="${RUN_DIR:-$SCRATCH/robustcd/runs/changemamba_${VARIANT}/seed${SEED}}"
SELF="$REPO/cluster/models/changemamba/train.sbatch"

module load StdEnv/2023 python/3.11 cuda/12.2
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
mkdir -p "$RUN"
echo "$(date -Is) job $SLURM_JOB_ID variant $VARIANT seed $SEED max_iters $MAX_ITERS budget $BUDGET accum $ACCUM -> $RUN" | tee -a "$RUN/slurm_jobs.txt"

tar -xf "$SCRATCH/robustcd/data/SECOND.tar" -C "$SLURM_TMPDIR" || exit 1
python -m robustcd.adapters.changemamba.prepare --second-root "$SLURM_TMPDIR/SECOND" \
    --splits "$REPO/splits/SECOND" --out "$SLURM_TMPDIR/cm_SECOND" || exit 1

cd "$REPO"
python -m robustcd.adapters.changemamba.train \
    --cm-root "$CM_ROOT" --cfg "$CFG" \
    --encoder-pretrained "$ENC" \
    --second-root "$SLURM_TMPDIR/SECOND" --cm-data "$SLURM_TMPDIR/cm_SECOND/train" \
    --splits "$REPO/splits/SECOND" --out "$RUN" --seed "$SEED" \
    --max-iters "$MAX_ITERS" --budget-iters "$BUDGET" --batch-size 16 --accum "$ACCUM" \
    --eval-interval 2000 --workers "${SLURM_CPUS_PER_TASK:-8}" --stop-after-min "$GUARD_MIN"
rc=$?
if [ "$rc" = 3 ]; then
    echo "time guard hit, resubmitting"
    sbatch --export=ALL,VARIANT="$VARIANT",SEED="$SEED",MAX_ITERS="$MAX_ITERS",BUDGET="$BUDGET",ACCUM="$ACCUM" "$SELF"
    exit 0
fi
exit "$rc"
