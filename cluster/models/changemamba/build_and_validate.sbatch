#!/bin/bash
#SBATCH --account=aip-hervete
#SBATCH --job-name=cm-build
#SBATCH --time=01:30:00
#SBATCH --gres=gpu:l40s:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --output=%x-%j.out
#
# 1. compile VMamba's selective-scan CUDA extension for the L40S (adds sm_89)
# 2. check selective_scan_cuda_oflex numerically against a pure-PyTorch scan
# 3. validate the adapter end to end on the released MambaSCD-Tiny checkpoint:
#    upstream decoding on SECOND test must reproduce the published from-to SeK
#    (0.2208); then report the robustcd scores (SECOND definition, protocol decoding)
#
#   cd $SCRATCH/robustcd/logs && sbatch ~/robustcd/cluster/models/changemamba/build_and_validate.sbatch
set -euo pipefail

REPO="${ROBUSTCD_REPO:-$HOME/robustcd}"
ENV_DIR="${ROBUSTCD_GPU_ENV:-$HOME/envs/robustcd-gpu}"
CM_ROOT="${CM_ROOT:-$HOME/ext/ChangeMamba}"
W="${CM_WEIGHTS:-$HOME/weights/changemamba}"
CFG="$CM_ROOT/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml"
OUT="$SCRATCH/robustcd/checks/changemamba_${SLURM_JOB_ID}"
mkdir -p "$OUT"

module load StdEnv/2023 python/3.11 cuda/12.2   # cuda/12.2 = torch 2.5.1 build
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
export PYTHONDONTWRITEBYTECODE=1 MAX_JOBS="${SLURM_CPUS_PER_TASK:-8}"
nvidia-smi --query-gpu=name,driver_version --format=csv,noheader
nvcc --version | tail -1

echo "== 1. build selective_scan (core, ndstate, oflex) with sm_89"
cd "$CM_ROOT/kernels/selective_scan"
if ! grep -q "compute_89" setup.py; then
    sed -i 's|    cc_flag.append("arch=compute_80,code=sm_80")|    cc_flag.append("arch=compute_80,code=sm_80")\n    cc_flag.append("-gencode")\n    cc_flag.append("arch=compute_89,code=sm_89")|' setup.py
fi
echo "sm_89 gencode lines: $(grep -c 'compute_89' setup.py)"
# the extensions link against torch's libc10.so, which is only on the loader path once torch is imported
KCHECK='import torch, selective_scan_cuda_oflex, selective_scan_cuda_core; print("kernels import ok")'
if [ "${CM_REBUILD:-0}" = 1 ]; then
    pip install --no-index --no-build-isolation --force-reinstall --no-deps . 2>&1 | tail -3
elif python -c "$KCHECK" 2>/dev/null; then
    echo "kernels already built, skipping compilation (set CM_REBUILD=1 to force)"
else
    pip install --no-index --no-build-isolation . 2>&1 | tail -3
fi
python -c "$KCHECK"
cd "$REPO"

echo "== 2. numeric check of selective_scan_cuda_oflex"
python "$REPO/cluster/models/changemamba/check_kernel.py"

echo "== 3. stage SECOND"
tar -xf "$SCRATCH/robustcd/data/SECOND.tar" -C "$SLURM_TMPDIR"

echo "== 4. released checkpoint, upstream decoding (must reproduce SeK_fromto ~ 0.2208)"
python -m robustcd.adapters.changemamba.predict --cm-root "$CM_ROOT" --cfg "$CFG" \
    --ckpt "$W/MambaSCD_Tiny_SECOND_SeK_0.2208.pth" --source "$SLURM_TMPDIR/SECOND/test" \
    --out "$SLURM_TMPDIR/pred_full" --decode full --evaluate

echo "== 5. released checkpoint, protocol decoding, scored by scripts/evaluate.py"
python -m robustcd.adapters.changemamba.predict --cm-root "$CM_ROOT" --cfg "$CFG" \
    --ckpt "$W/MambaSCD_Tiny_SECOND_SeK_0.2208.pth" --source "$SLURM_TMPDIR/SECOND/test" \
    --out "$SLURM_TMPDIR/pred_restricted" --decode restricted
python "$REPO/scripts/evaluate.py" scd --gt "$SLURM_TMPDIR/SECOND/test" \
    --pred "$SLURM_TMPDIR/pred_restricted" --out "$OUT/eval_released_restricted" \
    --tag "MambaSCD-Tiny released ckpt, protocol decoding"
cp "$SLURM_TMPDIR"/pred_full/predict_info.json "$OUT/predict_info_full.json"
cp "$SLURM_TMPDIR"/pred_restricted/predict_info.json "$OUT/predict_info_restricted.json"
python - "$OUT" <<'EOF'
import json, sys
d = json.load(open(f"{sys.argv[1]}/eval_released_restricted/metrics.json"))
s, ci = d["scores"], d["bootstrap"]["ci"]
print("robustcd SeK %.4f [%.4f, %.4f]  mIoU %.4f  Fscd %.4f  |  SeK_fromto %.4f" % (
    s["SeK"], ci["SeK"]["lo"], ci["SeK"]["hi"], s["mIoU"], s["Fscd"], s["SeK_fromto"]))
EOF
echo "ALL DONE -> $OUT"
