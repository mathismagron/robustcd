#!/bin/bash
#SBATCH --account=aip-hervete
#SBATCH --job-name=cmb-validate
#SBATCH --time=03:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --output=%x-%j.out
#
# Acceptance and cost check for MambaSCD-Base (protocol amendment A3). The kernel
# is already built by build_and_validate.sbatch; nothing is compiled here.
# 1. released MambaSCD-Base checkpoint on SECOND test. Upstream decoding must
#    reproduce the published from-to SeK 0.2292, as Tiny reproduced 0.2208.
#    The robustcd (SECOND-definition) scores are reported too;
# 2. 60-iteration timing at effective batch 16 on full 512 tiles, micro-batch 4 and 8.
#
#   cd $SCRATCH/robustcd/logs && sbatch ~/robustcd/cluster/models/changemamba/validate_base.sbatch
set -uo pipefail

REPO="${ROBUSTCD_REPO:-$HOME/robustcd}"
ENV_DIR="${ROBUSTCD_GPU_ENV:-$HOME/envs/robustcd-gpu}"
CM_ROOT="${CM_ROOT:-$HOME/ext/ChangeMamba}"
W="${CM_WEIGHTS:-$HOME/weights/changemamba}"
CFG="$CM_ROOT/changedetection/configs/vssm1/vssm_base_224.yaml"

module load StdEnv/2023 python/3.11 cuda/12.2
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
python -c "import torch, selective_scan_cuda_oflex" || { echo "kernel missing: run build_and_validate.sbatch"; exit 1; }

tar -xf "$SCRATCH/robustcd/data/SECOND.tar" -C "$SLURM_TMPDIR" || exit 1
python -m robustcd.adapters.changemamba.prepare --second-root "$SLURM_TMPDIR/SECOND" \
    --splits "$REPO/splits/SECOND" --out "$SLURM_TMPDIR/cm_SECOND" || exit 1

echo "== 1. released MambaSCD-Base, upstream decoding (published SeK_fromto 0.2292)"
python -W ignore -m robustcd.adapters.changemamba.predict --cm-root "$CM_ROOT" --cfg "$CFG" \
    --ckpt "$W/MambaSCD_Base_SECOND_SeK_0.2292.pth" --source "$SLURM_TMPDIR/SECOND/test" \
    --out "$SLURM_TMPDIR/pred_full" --decode full --evaluate 2>&1 | tail -2
echo "== 1b. protocol decoding"
python -W ignore -m robustcd.adapters.changemamba.predict --cm-root "$CM_ROOT" --cfg "$CFG" \
    --ckpt "$W/MambaSCD_Base_SECOND_SeK_0.2292.pth" --source "$SLURM_TMPDIR/SECOND/test" \
    --out "$SLURM_TMPDIR/pred_restricted" --decode restricted 2>&1 | tail -1
python "$REPO/scripts/evaluate.py" scd --gt "$SLURM_TMPDIR/SECOND/test" \
    --pred "$SLURM_TMPDIR/pred_restricted" --out "$SLURM_TMPDIR/eval_restricted" \
    --tag "MambaSCD-Base released ckpt, protocol decoding" > /dev/null
python - "$SLURM_TMPDIR" <<'EOF'
import json, sys
t = sys.argv[1]
full = json.load(open(f"{t}/pred_full/predict_info.json"))
d = json.load(open(f"{t}/eval_restricted/metrics.json"))
s, ci = d["scores"], d["bootstrap"]["ci"]
print(json.dumps({"upstream_decoding": full.get("scores"), "loaded_keys": full.get("loaded_keys"),
                  "unexpected_keys": full.get("unexpected_keys"), "images_per_s": full.get("images_per_s"),
                  "protocol_decoding": {k: round(s[k], 4) for k in ("SeK", "mIoU", "Fscd", "kappa_n0", "SeK_fromto")},
                  "SeK_ci95": [round(ci["SeK"]["lo"], 4), round(ci["SeK"]["hi"], 4)]}, indent=1))
EOF

echo "== 2. timing (60 iterations, effective batch 16)"
for ACC in 4 2; do
    echo "-- accum $ACC (micro-batch $((16 / ACC)))"
    python -W ignore -m robustcd.adapters.changemamba.train --cm-root "$CM_ROOT" --cfg "$CFG" \
        --encoder-pretrained "$W/vssm_base_0229_ckpt_epoch_237.pth" \
        --second-root "$SLURM_TMPDIR/SECOND" --cm-data "$SLURM_TMPDIR/cm_SECOND/train" \
        --splits "$REPO/splits/SECOND" --out "$SLURM_TMPDIR/timing/acc$ACC" --seed 0 \
        --max-iters 60 --eval-interval 60 --log-interval 20 --batch-size 16 --accum "$ACC" \
        --workers "${SLURM_CPUS_PER_TASK:-8}" 2>&1 \
        | grep -iE "load|^it (20|40|60)/|^VAL|OutOfMemory|out of memory|non-finite|Error|Traceback" | tail -8
    python - "$SLURM_TMPDIR/timing/acc$ACC" <<'EOF'
import json, sys
from pathlib import Path
d = Path(sys.argv[1]); f = d / "train_log.jsonl"
recs = [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []
recs = [r for r in recs if r["iteration"] > 20]
if recs:
    s = sum(r["s_per_iter"] for r in recs) / len(recs)
    print(f"TIMING {d.name}: {s:.3f} s/it, max mem {max(r['max_mem_gb'] for r in recs):.1f} GiB, "
          f"50k iterations = {s * 50000 / 3600:.1f} h")
else:
    print(f"TIMING {d.name}: no log (failed, probably out of memory)")
EOF
done
echo "== done"
