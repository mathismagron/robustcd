#!/bin/bash
#SBATCH --account=aip-hervete
#SBATCH --job-name=perascd-train
#SBATCH --time=1-00:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=%x-%j.out
#
# Train PerASCD under the robustcd protocol. Resubmits itself when the wall-clock
# guard fires; the run resumes from latest.pth with the same data order.
# The warm-up/poly schedule depends on its total length, so the 75k pilot is a
# separate run (as for the Ding models), not an extension of seed 0.
#
#   cd $SCRATCH/robustcd/logs
#   sbatch --export=ALL,SEED=0 ~/robustcd/cluster/models/perascd/train.sbatch
#   sbatch --export=ALL,SEED=0,MAX_ITERS=75000,RUN_NAME=pilot75k ~/robustcd/cluster/models/perascd/train.sbatch
#   sbatch --export=ALL,SEED=1 ~/robustcd/cluster/models/perascd/train.sbatch
set -uo pipefail

SEED="${SEED:?set SEED}"
MAX_ITERS="${MAX_ITERS:-50000}"
BUDGET="${BUDGET:-50000}"
ACCUM="${ACCUM:-4}"                      # set from validate.sbatch timing (micro-batch 16/ACCUM)
CKPT="${CKPT:-none}"                     # activation checkpointing: none | adapter | full
GUARD_MIN="${GUARD_MIN:-1350}"           # 22.5 h of a 24 h allocation (a val eval + 4.4 GB save must fit)
RUN_NAME="${RUN_NAME:-seed${SEED}}"
REPO="${ROBUSTCD_REPO:-$HOME/robustcd}"
ENV_DIR="${ROBUSTCD_GPU_ENV:-$HOME/envs/robustcd-gpu}"
PERASCD="${ROBUSTCD_EXT:-$HOME/ext}/PerASCD"
W="${PERASCD_WEIGHTS:-$HOME/weights/perascd}"
MSDA_SITE="${MSDA_SITE:-$HOME/ext/perascd_msda_site}"
RUN="${RUN_DIR:-$SCRATCH/robustcd/runs/perascd/$RUN_NAME}"
SELF="$REPO/cluster/models/perascd/train.sbatch"

module load StdEnv/2023 python/3.11 cuda/12.2
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$MSDA_SITE${PYTHONPATH:+:$PYTHONPATH}"
mkdir -p "$RUN"
echo "$(date -Is) job $SLURM_JOB_ID seed $SEED max_iters $MAX_ITERS budget $BUDGET accum $ACCUM ckpt $CKPT -> $RUN" \
    | tee -a "$RUN/slurm_jobs.txt"

tar -xf "$SCRATCH/robustcd/data/SECOND.tar" -C "$SLURM_TMPDIR" || exit 1
cd "$REPO"
python -W ignore::UserWarning -m robustcd.adapters.perascd.train --repo "$PERASCD" --msda cuda \
    --pretrained "$W/pera_ViTG161024.params" \
    --second-root "$SLURM_TMPDIR/SECOND" --splits "$REPO/splits/SECOND" --out "$RUN" --seed "$SEED" \
    --max-iters "$MAX_ITERS" --budget-iters "$BUDGET" --batch-size 16 --accum "$ACCUM" --checkpointing "$CKPT" \
    --eval-interval 2000 --workers "${SLURM_CPUS_PER_TASK:-8}" --stop-after-min "$GUARD_MIN"
rc=$?
if [ "$rc" = 3 ]; then
    echo "time guard hit, resubmitting"
    sbatch --export=ALL,SEED="$SEED",MAX_ITERS="$MAX_ITERS",BUDGET="$BUDGET",ACCUM="$ACCUM",CKPT="$CKPT",RUN_NAME="$RUN_NAME" "$SELF"
    exit 0
fi
exit "$rc"
