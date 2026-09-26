#!/bin/bash
#SBATCH --account=aip-hervete
#SBATCH --job-name=ding-validate
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=%x-%j.out
#
# Acceptance and cost check for the Ding-codebase adapter:
# 1. released SCanNet checkpoint on the SECOND test split (upstream and protocol decodings);
# 2. 100-iteration timing runs at effective batch 16 on full 512 tiles for every
#    campaign model (SCanNet and TED also with the pseudo-label teacher forced on).
#    Timing runs write to $SLURM_TMPDIR and are discarded; only the log matters.
#
#   cd $SCRATCH/robustcd/logs && sbatch ~/robustcd/cluster/models/ding/validate.sbatch
set -uo pipefail

REPO="${ROBUSTCD_REPO:-$HOME/robustcd}"
ENV_DIR="${ROBUSTCD_GPU_ENV:-$HOME/envs/robustcd-gpu}"
EXT="${ROBUSTCD_EXT:-$HOME/ext}"
W="${DING_WEIGHTS:-$HOME/weights/scannet}"
ACCUM="${ACCUM:-1}"
export TORCH_HOME="${TORCH_HOME:-$HOME/weights/torch}"

module load StdEnv/2023 python/3.11 cuda/12.2
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

tar -xf "$SCRATCH/robustcd/data/SECOND.tar" -C "$SLURM_TMPDIR" || exit 1

echo "== 1. released SCanNet checkpoint on SECOND test"
python -W ignore -m robustcd.adapters.ding.validate_release --model scannet --repo "$EXT/SCanNet" \
    --ckpt "$W/SCanNet_32e_mIoU73.37_Sek23.94_Fscd63.66_OA87.86.pth" \
    --source "$SLURM_TMPDIR/SECOND/test" --json "$SLURM_TMPDIR/scannet_released.json" || echo "VALIDATION FAILED"
cat "$SLURM_TMPDIR/scannet_released.json"

echo "== 2. timing (100 iterations, effective batch 16, accum $ACCUM)"
timing() {  # label model repo [extra args]
    local label=$1 model=$2 repo=$3; shift 3
    echo "-- $label"
    python -W ignore -m robustcd.adapters.ding.train --model "$model" --repo "$repo" \
        --second-root "$SLURM_TMPDIR/SECOND" --splits "$REPO/splits/SECOND" \
        --out "$SLURM_TMPDIR/timing/$label" --seed 0 --max-iters 100 --eval-interval 100 \
        --log-interval 25 --batch-size 16 --accum "$ACCUM" --workers "${SLURM_CPUS_PER_TASK:-8}" "$@" 2>&1 \
        | grep -E "^it (50|75|100)/|^VAL|Error|error|Traceback" | tail -5
    python - "$SLURM_TMPDIR/timing/$label" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1])
recs = [json.loads(l) for l in (d / "train_log.jsonl").read_text().splitlines()] if (d / "train_log.jsonl").exists() else []
recs = [r for r in recs if r["iteration"] > 25]          # skip warm-up / cudnn autotune
val = [json.loads(l) for l in (d / "val_log.jsonl").read_text().splitlines()] if (d / "val_log.jsonl").exists() else []
if recs:
    s = sum(r["s_per_iter"] for r in recs) / len(recs)
    print(f"TIMING {d.name}: {s:.3f} s/it, max mem {max(r['max_mem_gb'] for r in recs):.1f} GiB, "
          f"50k iterations = {s * 50000 / 3600:.1f} h, val eval {val[-1]['eval_seconds'] if val else 'n/a'} s")
else:
    print(f"TIMING {d.name}: no log (failed)")
EOF
}
timing hrscd4      hrscd4  "$EXT/Bi-SRNet"
timing bisrnet     bisrnet "$EXT/Bi-SRNet"
timing ted         ted     "$EXT/SCanNet"
timing ted_psd     ted     "$EXT/SCanNet" --psd-force
timing scannet     scannet "$EXT/SCanNet"
timing scannet_psd scannet "$EXT/SCanNet" --psd-force
echo "== done"
