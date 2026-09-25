#!/bin/bash
# ChangeMamba setup on an Alliance login node (internet access needed). Run once:
#     bash ~/robustcd/cluster/models/changemamba/setup.sh
# 1. clones ChangeMamba at the pinned commit into ~/ext/ChangeMamba
# 2. installs its extra Python deps into the robustcd GPU env (--no-index)
# 3. downloads the VMamba-Tiny ImageNet backbone and the released
#    MambaSCD-Tiny SECOND checkpoint from Zenodo, verified by md5
# The CUDA kernel is compiled on a GPU node by build_and_validate.sbatch.
set -euo pipefail

REPO="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
ENV_DIR="${ROBUSTCD_GPU_ENV:-$HOME/envs/robustcd-gpu}"
CM_ROOT="${CM_ROOT:-$HOME/ext/ChangeMamba}"
W="${CM_WEIGHTS:-$HOME/weights/changemamba}"
COMMIT=9ce9cec13f9ea14bc0ad91f071577ec9b3a97983

module load StdEnv/2023 python/3.11
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

echo "== 1. code @ $COMMIT"
if [ ! -d "$CM_ROOT/.git" ]; then
    git clone https://github.com/ChenHongruixuan/ChangeMamba.git "$CM_ROOT"
fi
git -C "$CM_ROOT" fetch --quiet origin
git -C "$CM_ROOT" checkout --quiet "$COMMIT"
echo "   HEAD $(git -C "$CM_ROOT" rev-parse HEAD)"

echo "== 2. extra python deps"
pip install --no-index -r "$REPO/cluster/models/changemamba/requirements.txt"
pip freeze > "$ENV_DIR/freeze.txt"

echo "== 3. weights (md5-verified)"
mkdir -p "$W"
fetch() {  # name md5
    local f="$W/$1"
    if [ ! -f "$f" ] || [ "$(md5sum "$f" | cut -d' ' -f1)" != "$2" ]; then
        curl -L --fail --retry 3 -o "$f.part" "https://zenodo.org/records/15479555/files/$1?download=1"
        mv "$f.part" "$f"
    fi
    local got; got=$(md5sum "$f" | cut -d' ' -f1)
    if [ "$got" != "$2" ]; then echo "FAIL md5 $1: $got != $2"; exit 1; fi
    echo "   ok $1"
}
fetch vssm_tiny_0230_ckpt_epoch_262.pth d64653bba8f6e5c0d6f4ac6275e1be61
fetch MambaSCD_Tiny_SECOND_SeK_0.2208.pth d1c1d50267da8063b9caa56b63ff1396

echo "setup done. Next: sbatch ~/robustcd/cluster/models/changemamba/build_and_validate.sbatch"
