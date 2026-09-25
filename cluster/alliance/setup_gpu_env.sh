#!/bin/bash
# Create the GPU training environment (torch + mamba_ssm + causal_conv1d) on an
# Alliance cluster. Run once, on a login node:
#     bash cluster/alliance/setup_gpu_env.sh
# then validate on a GPU node with cluster/alliance/gpu_smoke_test.sbatch.
#
# Overrides:  ROBUSTCD_GPU_ENV=<dir>   ROBUSTCD_PY_MODULE=python/3.11
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_DIR="${ROBUSTCD_GPU_ENV:-$HOME/envs/robustcd-gpu}"
PY_MODULE="${ROBUSTCD_PY_MODULE:-python/3.11}"

module load StdEnv/2023 "$PY_MODULE"
echo "python: $(python --version 2>&1)"

if [ ! -x "$ENV_DIR/bin/python" ]; then
    virtualenv --no-download "$ENV_DIR"
fi
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
pip install --no-index --upgrade pip
# one resolver call, so wheel metadata can reject an incompatible torch / mamba_ssm pair
pip install --no-index -r "$REPO/cluster/alliance/requirements-gpu.txt"
pip install --no-index --no-build-isolation -e "$REPO" \
    || echo "WARNING: editable install failed; scripts still run (they add the repo to sys.path)"

# record the exact resolved stack next to the env (commit a copy for the paper)
pip freeze > "$ENV_DIR/freeze.txt"
echo "--- resolved key packages"
grep -iE '^(torch|torchvision|triton|mamba.ssm|causal.conv1d|timm|einops|numpy)==' "$ENV_DIR/freeze.txt" || true

grep -iE '^transformers==' "$ENV_DIR/freeze.txt" || true

# import check (the CUDA extensions load without a GPU; kernels are exercised by the smoke test).
# An import failure here is a real packaging problem, so stop.
if ! python - <<'EOF'
import torch
print("torch", torch.__version__, "built for CUDA", torch.version.cuda)
import mamba_ssm, causal_conv1d
from mamba_ssm.ops.selective_scan_interface import selective_scan_fn  # noqa: F401
print("mamba_ssm", mamba_ssm.__version__, " causal_conv1d", causal_conv1d.__version__)
EOF
then
    echo "FAIL: import check failed; do not run the GPU smoke test until this is fixed"
    exit 1
fi

echo
echo "GPU env ready. Activate with:"
echo "  module load StdEnv/2023 $PY_MODULE && source $ENV_DIR/bin/activate"
