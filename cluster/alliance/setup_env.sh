#!/bin/bash
# Create the robustcd tooling environment on an Alliance cluster (Vulcan, Narval, ...).
# Run once, on a login node:      bash cluster/alliance/setup_env.sh
#
# Alliance rules: module-provided Python, `virtualenv --no-download`, `pip --no-index`
# (wheels come from the Alliance wheelhouse; no conda, no uv).
# This env holds the degradation / metric / data tooling only (numpy, scipy, pillow).
# The GPU training stack (torch, mamba_ssm, ...) gets its own env later.
#
# Overrides:  ROBUSTCD_ENV=<dir>   ROBUSTCD_PY_MODULE=python/3.12
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
ENV_DIR="${ROBUSTCD_ENV:-$HOME/envs/robustcd}"
PY_MODULE="${ROBUSTCD_PY_MODULE:-python/3.11}"

module purge
module load StdEnv/2023 "$PY_MODULE"
echo "python: $(python --version 2>&1)  ($(command -v python))"

if [ ! -x "$ENV_DIR/bin/python" ]; then
    virtualenv --no-download "$ENV_DIR"
fi
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
pip install --no-index --upgrade pip
pip install --no-index numpy scipy pillow

# editable install so `import robustcd` works from anywhere; scripts also work without it
pip install --no-index --no-build-isolation -e "$REPO" \
    || echo "WARNING: editable install failed; scripts still run (they add the repo to sys.path)"

python - <<'EOF'
import numpy, scipy, PIL, robustcd
print("ok  numpy", numpy.__version__, " scipy", scipy.__version__, " pillow", PIL.__version__,
      " robustcd", robustcd.__file__)
EOF
cd "$REPO" && python tests/test_metrics.py

echo
echo "Environment ready. Activate with:"
echo "  module load StdEnv/2023 $PY_MODULE && source $ENV_DIR/bin/activate"
