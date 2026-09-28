#!/bin/bash
# Mamba-FCS setup on an Alliance login node (internet access needed). Run once:
#     bash ~/robustcd/cluster/models/mambafcs/setup.sh
# 1. clones Buddhi19/MambaFCS at the pinned commit into ~/ext/MambaFCS
#    (the directory must be named MambaFCS: upstream imports `MambaFCS.changedetection...`)
# 2. downloads the ImageNet VMamba-Base backbone (Zenodo, md5) and the released
#    SECOND checkpoint (Hugging Face, sha256)
# 3. checks that the selective-scan extension built for ChangeMamba is importable:
#    Mamba-FCS ships the same kernel sources, so nothing is compiled here.
# Extra Python deps are those of ChangeMamba (cluster/models/changemamba/requirements.txt).
set -euo pipefail

ENV_DIR="${ROBUSTCD_GPU_ENV:-$HOME/envs/robustcd-gpu}"
EXT="${ROBUSTCD_EXT:-$HOME/ext}"
W="${FCS_WEIGHTS:-$HOME/weights/mambafcs}"
COMMIT=5c01f1ac86adff89d6dc5eb6e02d83ee8c16f25c

module load StdEnv/2023 python/3.11
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

echo "== 1. code @ $COMMIT"
mkdir -p "$EXT"
if [ ! -d "$EXT/MambaFCS/.git" ]; then git clone https://github.com/Buddhi19/MambaFCS.git "$EXT/MambaFCS"; fi
git -C "$EXT/MambaFCS" fetch --quiet origin
git -C "$EXT/MambaFCS" checkout --quiet "$COMMIT"
echo "   HEAD $(git -C "$EXT/MambaFCS" rev-parse HEAD)"

echo "== 2. weights"
mkdir -p "$W"
B="$W/vssm_base_0229_ckpt_epoch_237.pth"; B_MD5=3e4110259f482f552dc70abcf3381e71
if [ ! -f "$B" ] || [ "$(md5sum "$B" | cut -d' ' -f1)" != "$B_MD5" ]; then
    curl -L --fail --retry 3 -o "$B.part" "https://zenodo.org/records/15479555/files/vssm_base_0229_ckpt_epoch_237.pth?download=1"
    mv "$B.part" "$B"
fi
[ "$(md5sum "$B" | cut -d' ' -f1)" = "$B_MD5" ] || { echo "FAIL md5 $(basename "$B")"; exit 1; }
echo "   ok $(basename "$B")"
R="$W/SECOND_SeK_0.255.pth"; R_SHA=b1f3252f2761fd63c0b6076ee0abdc5c95a079716f0ff29733581d6d327a1522
if [ ! -f "$R" ] || [ "$(sha256sum "$R" | cut -d' ' -f1)" != "$R_SHA" ]; then
    curl -L --fail --retry 3 -o "$R.part" "https://huggingface.co/buddhi19/MambaFCS/resolve/main/SECOND_SeK_0.255.pth"
    mv "$R.part" "$R"
fi
[ "$(sha256sum "$R" | cut -d' ' -f1)" = "$R_SHA" ] || { echo "FAIL sha256 $(basename "$R")"; exit 1; }
echo "   ok $(basename "$R")"

echo "== 3. selective-scan extension (built for ChangeMamba)"
python -c "import torch, selective_scan_cuda_oflex; print('   ok', selective_scan_cuda_oflex.__file__)" \
    || { echo "   missing: run cluster/models/changemamba/build_and_validate.sbatch first"; exit 1; }
diff -rq "$EXT/MambaFCS/kernels/selective_scan/csrc" "$EXT/ChangeMamba/kernels/selective_scan/csrc" \
    | sed 's/^/   kernel source diff: /' || true
echo "setup done. Next: sbatch ~/robustcd/cluster/models/mambafcs/validate.sbatch"
