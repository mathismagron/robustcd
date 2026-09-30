#!/bin/bash
#SBATCH --account=aip-hervete
#SBATCH --job-name=perascd-validate
#SBATCH --time=04:00:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=80G
#SBATCH --output=%x-%j.out
#
# Acceptance and cost check for the PerASCD adapter:
# 0. compiles the MultiScaleDeformableAttention op for this GPU (sm_89) into
#    $MSDA_SITE, once (skipped when it already imports);
# 1. released checkpoint on the SECOND test split with the robustcd adapter (SDPA
#    attention, compiled op). Upstream decoding should give SeK 0.2611 / Fscd 0.6641;
# 2. 30-iteration timing at effective batch 16 on full 512 tiles, from the PerA
#    weights, for several micro-batch / activation-checkpointing settings.
#    Out-of-memory settings just print "no log".
#
#   cd $SCRATCH/robustcd/logs && sbatch ~/robustcd/cluster/models/perascd/validate.sbatch
set -uo pipefail

REPO="${ROBUSTCD_REPO:-$HOME/robustcd}"
ENV_DIR="${ROBUSTCD_GPU_ENV:-$HOME/envs/robustcd-gpu}"
PERASCD="${ROBUSTCD_EXT:-$HOME/ext}/PerASCD"
W="${PERASCD_WEIGHTS:-$HOME/weights/perascd}"
MSDA_SITE="${MSDA_SITE:-$HOME/ext/perascd_msda_site}"
SETTINGS="${SETTINGS:-4:none 4:full 8:none 2:full}"     # accum:checkpointing

module load StdEnv/2023 python/3.11 cuda/12.2
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
export PYTHONDONTWRITEBYTECODE=1 PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
export PYTHONPATH="$MSDA_SITE${PYTHONPATH:+:$PYTHONPATH}"
cd "$REPO"
nvidia-smi --query-gpu=name,memory.total --format=csv,noheader

echo "== 0. MultiScaleDeformableAttention op"
if ! python -c "import torch, MultiScaleDeformableAttention" 2>/dev/null; then
    t0=$(date +%s)
    B="$SLURM_TMPDIR/msda_src"; S="$SLURM_TMPDIR/msda_site"
    cp -r "$PERASCD/models/pera_layers/vit_adapter_layers/ops" "$B"
    rm -rf "$B/build" "$B/dist" "$B"/*.egg-info
    ( cd "$B" && TORCH_CUDA_ARCH_LIST="8.9" MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}" \
        pip install -q --no-index --no-build-isolation --no-deps --target "$S" . ) || { echo "MSDA BUILD FAILED"; exit 1; }
    echo "PerASCD $(git -C "$PERASCD" rev-parse HEAD) job $SLURM_JOB_ID $(date -Is) sm_89" > "$S/BUILD_INFO.txt"
    rm -rf "$MSDA_SITE.tmp"; cp -r "$S" "$MSDA_SITE.tmp"; rm -rf "$MSDA_SITE"; mv "$MSDA_SITE.tmp" "$MSDA_SITE"
    echo "   built in $(( $(date +%s) - t0 )) s"
fi
python -c "import torch, MultiScaleDeformableAttention as m; print('   ok', m.__file__)" || exit 1

tar -xf "$SCRATCH/robustcd/data/SECOND.tar" -C "$SLURM_TMPDIR" || exit 1

echo "== 1. released PerASCD checkpoint on SECOND test"
python -W ignore -m robustcd.adapters.perascd.validate_release --repo "$PERASCD" --msda cuda \
    --ckpt "$W/PerAChain_40e_mIoU74.33_Sek26.11_Fscd66.41_OA88.70.pth" --source "$SLURM_TMPDIR/SECOND/test" \
    --json "$SLURM_TMPDIR/perascd_released.json" 2>&1 | grep -vE "xFormers|warnings.warn" || echo "VALIDATION FAILED"
cat "$SLURM_TMPDIR/perascd_released.json"; echo

echo "== 2. timing (30 iterations, effective batch 16, PerA initialisation)"
for S in $SETTINGS; do
    ACC=${S%%:*}; CKPT=${S##*:}; L="acc${ACC}_${CKPT}"
    echo "-- accum $ACC (micro-batch $((16 / ACC))), checkpointing $CKPT"
    python -W ignore -m robustcd.adapters.perascd.train --repo "$PERASCD" --msda cuda \
        --pretrained "$W/pera_ViTG161024.params" \
        --second-root "$SLURM_TMPDIR/SECOND" --splits "$REPO/splits/SECOND" --out "$SLURM_TMPDIR/timing/$L" \
        --seed 0 --max-iters 30 --eval-interval 30 --val-limit 32 --log-interval 10 --batch-size 16 \
        --accum "$ACC" --checkpointing "$CKPT" --workers "${SLURM_CPUS_PER_TASK:-8}" 2>&1 \
        | grep -E "^built|^pretrained|^it (10|20|30)/|^VAL|OutOfMemory|out of memory|non-finite|Error|Traceback" | tail -7
    python - "$SLURM_TMPDIR/timing/$L" <<'PY'
import json, sys
from pathlib import Path
d = Path(sys.argv[1]); f = d / "train_log.jsonl"
recs = [json.loads(l) for l in f.read_text().splitlines()] if f.exists() else []
val = [json.loads(l) for l in (d / "val_log.jsonl").read_text().splitlines()] if (d / "val_log.jsonl").exists() else []
recs = [r for r in recs if r["iteration"] > 10]            # skip warm-up / cudnn autotune
if recs:
    s = sum(r["s_per_iter"] for r in recs) / len(recs)
    ev = val[-1]["eval_seconds"] * 297 / 32 if val else float("nan")
    print(f"TIMING {d.name}: {s:.3f} s/it, max mem {max(r['max_mem_gb'] for r in recs):.1f} GiB, "
          f"50k iterations = {s * 50000 / 3600:.1f} h (+ {25 * ev / 3600:.1f} h for 25 val evals of ~{ev:.0f} s), "
          f"grad norm {recs[-1].get('grad_norm', float('nan')):.2f}")
else:
    print(f"TIMING {d.name}: no log (failed, probably out of memory)")
PY
    rm -f "$SLURM_TMPDIR/timing/$L"/*.pth             # 2.2 GB model + 4.4 GB latest.pth per setting
done
echo "== done"
