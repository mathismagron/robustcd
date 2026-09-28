#!/bin/bash
#SBATCH --account=aip-hervete
#SBATCH --job-name=fcs-validate
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=%x-%j.out
#
# Acceptance and cost check for the Mamba-FCS adapter:
# 1. released checkpoint on the SECOND test split. Upstream selects on test and
#    reports the SECOND SeK, so upstream decoding should give 0.2550;
# 2. 60-iteration timing at effective batch 16 on full 512 tiles, with
#    micro-batches of 4 (ACCUM 4) and 8 (ACCUM 2; may run out of memory).
#
#   cd $SCRATCH/robustcd/logs && sbatch ~/robustcd/cluster/models/mambafcs/validate.sbatch
set -uo pipefail

REPO="${ROBUSTCD_REPO:-$HOME/robustcd}"
ENV_DIR="${ROBUSTCD_GPU_ENV:-$HOME/envs/robustcd-gpu}"
FCS="${ROBUSTCD_EXT:-$HOME/ext}/MambaFCS"
W="${FCS_WEIGHTS:-$HOME/weights/mambafcs}"

module load StdEnv/2023 python/3.11 cuda/12.2
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

tar -xf "$SCRATCH/robustcd/data/SECOND.tar" -C "$SLURM_TMPDIR" || exit 1
python -m robustcd.adapters.changemamba.prepare --second-root "$SLURM_TMPDIR/SECOND" \
    --splits "$REPO/splits/SECOND" --out "$SLURM_TMPDIR/cm_SECOND" || exit 1

echo "== 1. released Mamba-FCS checkpoint on SECOND test"
python -W ignore -m robustcd.adapters.mambafcs.validate_release --fcs-root "$FCS" \
    --ckpt "$W/SECOND_SeK_0.255.pth" --source "$SLURM_TMPDIR/SECOND/test" \
    --json "$SLURM_TMPDIR/mambafcs_released.json" 2>&1 | grep -vE "^(True|False)$|merge config" || echo "VALIDATION FAILED"
cat "$SLURM_TMPDIR/mambafcs_released.json"; echo

echo "== 2. timing (60 iterations, effective batch 16)"
for ACC in 4 2; do
    echo "-- accum $ACC (micro-batch $((16 / ACC)))"
    python -W ignore -m robustcd.adapters.mambafcs.train --fcs-root "$FCS" \
        --backbone "$W/vssm_base_0229_ckpt_epoch_237.pth" \
        --second-root "$SLURM_TMPDIR/SECOND" --cm-data "$SLURM_TMPDIR/cm_SECOND/train" \
        --splits "$REPO/splits/SECOND" --out "$SLURM_TMPDIR/timing/acc$ACC" --seed 0 \
        --max-iters 60 --eval-interval 60 --log-interval 20 --batch-size 16 --accum "$ACC" \
        --workers "${SLURM_CPUS_PER_TASK:-8}" 2>&1 \
        | grep -E "^backbone|^it (20|40|60)/|^VAL|OutOfMemory|out of memory|non-finite|Error|Traceback" | tail -6
    python - "$SLURM_TMPDIR/timing/acc$ACC" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1]); f = d / "train_log.jsonl"
recs = [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []
recs = [r for r in recs if r["iteration"] > 20]
if recs:
    s = sum(r["s_per_iter"] for r in recs) / len(recs)
    print(f"TIMING {d.name}: {s:.3f} s/it, max mem {max(r['max_mem_gb'] for r in recs):.1f} GiB, "
          f"50k iterations = {s * 50000 / 3600:.1f} h, SeK term skipped in "
          f"{sum(r.get('sek_skipped', 0) for r in recs):.2f} of {len(recs)} logged iterations")
else:
    print(f"TIMING {d.name}: no log (failed, probably out of memory)")
EOF
done
echo "== done"
