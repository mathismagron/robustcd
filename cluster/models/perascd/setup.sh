#!/bin/bash
# PerASCD setup on an Alliance login node (internet access needed). Run once:
#     bash ~/robustcd/cluster/models/perascd/setup.sh
# 1. clones SathShen/PerASCD, branch legacy, at the pinned commit into ~/ext/PerASCD
# 2. downloads the PerA ViT-G/16-1024 foundation weights (Google Drive, 9.1 GB; the
#    initialisation used by upstream train.py) and the released SECOND checkpoint
#    (Hugging Face zip, 4.1 GB; reused from the csf-mamba teacher directory if present),
#    and records their sha256 in ~/weights/perascd/SHA256SUMS
# The MultiScaleDeformableAttention CUDA op is compiled on a GPU node by validate.sbatch
# (its setup.py refuses to build without a visible GPU). No extra Python package is needed:
# the model code uses torch, timm and numpy only (xformers is optional upstream, replaced by SDPA).
set -euo pipefail

ENV_DIR="${ROBUSTCD_GPU_ENV:-$HOME/envs/robustcd-gpu}"
EXT="${ROBUSTCD_EXT:-$HOME/ext}"
W="${PERASCD_WEIGHTS:-$HOME/weights/perascd}"
COMMIT=a4d808a6cfb5df7efeee186730ac26b4504c9ed6
PRE=pera_ViTG161024.params; PRE_ID=1YhmDNLyyqbIfRFCfwZTGyyqYMQncamPE; PRE_BYTES=9097269032
ZIP=PerASCD_260128115444_vitg01min0Clip15LsscTau001.zip
PTH=PerAChain_40e_mIoU74.33_Sek26.11_Fscd66.41_OA88.70.pth

module load StdEnv/2023 python/3.11
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"

echo "== 1. code (legacy @ ${COMMIT:0:7})"
mkdir -p "$EXT"
if [ ! -d "$EXT/PerASCD/.git" ]; then git clone --branch legacy https://github.com/SathShen/PerASCD.git "$EXT/PerASCD"; fi
git -C "$EXT/PerASCD" fetch --quiet origin legacy
git -C "$EXT/PerASCD" -c advice.detachedHead=false checkout --quiet "$COMMIT"
echo "   HEAD $(git -C "$EXT/PerASCD" rev-parse HEAD)"

echo "== 2. weights"
mkdir -p "$W"
if [ ! -f "$W/$PRE" ] || [ "$(stat -c %s "$W/$PRE")" != "$PRE_BYTES" ]; then
    curl -L --fail --retry 3 -C - -o "$W/$PRE.part" \
        "https://drive.usercontent.google.com/download?id=$PRE_ID&export=download&confirm=t"
    mv "$W/$PRE.part" "$W/$PRE"
fi
[ "$(stat -c %s "$W/$PRE")" = "$PRE_BYTES" ] || { echo "FAIL size $PRE"; exit 1; }
python - "$W/$PRE" <<'PY'
import sys, zipfile
names = zipfile.ZipFile(sys.argv[1]).namelist()
arch = names[0].split("/")[0]
assert arch == "pera_ViTGall22601_ep42_auto", arch     # the file named in upstream train.py
print(f"   ok {sys.argv[1].rsplit('/', 1)[-1]} (archive {arch}, {len(names)} entries)")
PY
if [ ! -f "$W/$PTH" ]; then
    CSF="$SCRATCH/csf-distill/teacher/ckpt/$PTH"
    if [ -f "$CSF" ]; then
        echo "   copying $PTH from the csf-mamba teacher directory"
        cp "$CSF" "$W/$PTH.part" && mv "$W/$PTH.part" "$W/$PTH"
    else
        curl -L --fail --retry 3 -C - -o "$W/$ZIP.part" \
            "https://huggingface.co/SathShen/PerASCD-Checkpoint/resolve/main/$ZIP"
        mv "$W/$ZIP.part" "$W/$ZIP"
        unzip -o -j -q "$W/$ZIP" "*$PTH" -d "$W"
        rm -f "$W/$ZIP"
    fi
fi
echo "   ok $PTH ($(du -h "$W/$PTH" | cut -f1))"
if [ ! -f "$W/SHA256SUMS" ]; then (cd "$W" && sha256sum "$PRE" "$PTH" > SHA256SUMS); fi
cat "$W/SHA256SUMS"
echo "setup done. Next: cd \$SCRATCH/robustcd/logs && sbatch ~/robustcd/cluster/models/perascd/validate.sbatch"
