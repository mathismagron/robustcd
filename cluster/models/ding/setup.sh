#!/bin/bash
# Setup for the Ding-codebase models (SCanNet, TED, Bi-SRNet, SSCD-l, HRSCD-str4)
# on an Alliance login node (internet access needed). Run once:
#     bash ~/robustcd/cluster/models/ding/setup.sh
# 1. clones DingLei14/SCanNet and DingLei14/Bi-SRNet at the pinned commits into ~/ext
# 2. downloads torchvision's ImageNet ResNet-34 into $TORCH_HOME (compute nodes are offline)
# 3. downloads the released SCanNet SECOND checkpoint (Google Drive), md5-verified
# No extra Python packages: the models need torch, torchvision, timm and einops,
# all already in the robustcd GPU env.
set -euo pipefail

ENV_DIR="${ROBUSTCD_GPU_ENV:-$HOME/envs/robustcd-gpu}"
EXT="${ROBUSTCD_EXT:-$HOME/ext}"
W="${DING_WEIGHTS:-$HOME/weights/scannet}"
export TORCH_HOME="${TORCH_HOME:-$HOME/weights/torch}"
SCANNET_COMMIT=9c80d463bd0ca8cacca68b71c8b9adb7efd9f7a3
BISRNET_COMMIT=012f35aa9742e468b568a71107028fc3aa1e08b8

module load StdEnv/2023 python/3.11
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

clone() {  # name commit
    local dst="$EXT/$1"
    if [ ! -d "$dst/.git" ]; then git clone "https://github.com/DingLei14/$1.git" "$dst"; fi
    git -C "$dst" fetch --quiet origin
    git -C "$dst" checkout --quiet "$2"
    echo "   $1 HEAD $(git -C "$dst" rev-parse HEAD)"
}
echo "== 1. code"
mkdir -p "$EXT"
clone SCanNet "$SCANNET_COMMIT"
clone Bi-SRNet "$BISRNET_COMMIT"

echo "== 2. ImageNet ResNet-34 -> $TORCH_HOME"
R34="$TORCH_HOME/hub/checkpoints/resnet34-b627a593.pth"
mkdir -p "$(dirname "$R34")"
if [ ! -f "$R34" ]; then
    curl -L --fail --retry 3 -o "$R34.part" https://download.pytorch.org/models/resnet34-b627a593.pth
    mv "$R34.part" "$R34"
fi
sha=$(sha256sum "$R34" | cut -c1-8)
if [ "$sha" != "b627a593" ]; then echo "FAIL sha256 prefix $sha != b627a593"; exit 1; fi
echo "   ok resnet34-b627a593.pth"

echo "== 3. released SCanNet checkpoint (md5-verified)"
mkdir -p "$W"
F="$W/SCanNet_32e_mIoU73.37_Sek23.94_Fscd63.66_OA87.86.pth"
MD5=1e00a960a67662e694aec8d515f83108
if [ ! -f "$F" ] || [ "$(md5sum "$F" | cut -d' ' -f1)" != "$MD5" ]; then
    curl -L --fail --retry 3 -o "$F.part" \
        "https://drive.usercontent.google.com/download?id=1KfA_s3UVqK645WVYPdQ8aIlQkpnuPaPY&export=download&confirm=t"
    mv "$F.part" "$F"
fi
got=$(md5sum "$F" | cut -d' ' -f1)
if [ "$got" != "$MD5" ]; then echo "FAIL md5 $got != $MD5 (Drive may have served an HTML page)"; exit 1; fi
echo "   ok $(basename "$F")"

python -c "import timm, einops, torchvision; print('   timm', timm.__version__, 'einops', einops.__version__, 'torchvision', torchvision.__version__)"
echo "setup done. Next: sbatch ~/robustcd/cluster/models/ding/validate.sbatch"
