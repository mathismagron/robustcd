#!/bin/bash
#SBATCH --account=def-hervete
#SBATCH --job-name=prep-second
#SBATCH --time=01:00:00
#SBATCH --cpus-per-task=4
#SBATCH --mem=8G
#SBATCH --output=%x-%j.out
#
# Extract the SECOND archives in $SCRATCH, verify every sample against the
# reference computed locally, and build an uncompressed tar for fast staging
# to $SLURM_TMPDIR in training jobs.
#
#   mkdir -p $SCRATCH/robustcd/logs && cd $SCRATCH/robustcd/logs
#   sbatch ~/robustcd/cluster/alliance/prepare_second.sbatch
#
# Layout produced:
#   $SCRATCH/robustcd/data/SECOND/{train,test}/{im1,im2,label1,label2}/<id>.png
#   $SCRATCH/robustcd/data/SECOND.tar
#   $SCRATCH/robustcd/checks/second_{train,test}_check.json
set -euo pipefail

REPO="${ROBUSTCD_REPO:-$HOME/robustcd}"
ENV_DIR="${ROBUSTCD_ENV:-$HOME/envs/robustcd}"
PY_MODULE="${ROBUSTCD_PY_MODULE:-python/3.11}"
RAW="$SCRATCH/robustcd/raw/SECOND"
DATA="$SCRATCH/robustcd/data/SECOND"
CHECKS="$SCRATCH/robustcd/checks"

module purge
module load StdEnv/2023 "$PY_MODULE"
# shellcheck disable=SC1091
source "$ENV_DIR/bin/activate"
export PYTHONDONTWRITEBYTECODE=1

echo "== 1. archive sizes"
declare -A EXPECTED=( [SECOND_total_test.zip]=1378977505 [SECOND_train_set.rar]=2406111691 )
for f in "${!EXPECTED[@]}"; do
    got=$(stat -c %s "$RAW/$f")
    if [ "$got" != "${EXPECTED[$f]}" ]; then
        echo "FAIL $f: $got bytes, expected ${EXPECTED[$f]} (incomplete transfer?)"; exit 1
    fi
    echo "ok   $f  $got bytes"
done

echo "== 2. extract test (zip, contains a test/ prefix)"
mkdir -p "$DATA"
if [ ! -d "$DATA/test/im1" ]; then
    python -m zipfile -e "$RAW/SECOND_total_test.zip" "$DATA/"
fi
rm -rf "$DATA/__MACOSX"

echo "== 3. extract train (RAR5, entries at the archive root)"
if [ ! -d "$DATA/train/im1" ]; then
    mkdir -p "$DATA/train"
    if command -v bsdtar >/dev/null; then
        bsdtar -xf "$RAW/SECOND_train_set.rar" -C "$DATA/train"
    elif command -v unrar >/dev/null; then
        (cd "$DATA/train" && unrar x -o+ -idq "$RAW/SECOND_train_set.rar")
    elif command -v 7z >/dev/null || command -v 7za >/dev/null; then
        "$(command -v 7z || command -v 7za)" x -y -o"$DATA/train" "$RAW/SECOND_train_set.rar" >/dev/null
    elif command -v unar >/dev/null; then
        unar -q -f -D -o "$DATA/train" "$RAW/SECOND_train_set.rar"
    else
        echo "FAIL no RAR extractor (bsdtar/unrar/7z/unar) on this node."
        echo "     Extract locally instead, see cluster/README.md 'No RAR tool on the cluster'."
        rmdir "$DATA/train" 2>/dev/null || true
        exit 2
    fi
fi
rm -rf "$DATA/train/__MACOSX"

echo "== 4. verify every sample against the local reference"
status=0
for split in train test; do
    python "$REPO/scripts/check_dataset.py" --root "$DATA/$split" \
        --out "$CHECKS/second_${split}_check.json" --jobs "${SLURM_CPUS_PER_TASK:-4}" \
        --expect "$REPO/results/data_checks/second_${split}_reference.json" || status=1
done
if [ "$status" != 0 ]; then echo "FAIL verification, see $CHECKS"; exit 1; fi

echo "== 5. staging tar (uncompressed; training jobs untar it into \$SLURM_TMPDIR)"
if [ ! -f "$SCRATCH/robustcd/data/SECOND.tar" ]; then
    tar -cf "$SCRATCH/robustcd/data/SECOND.tar.part" -C "$SCRATCH/robustcd/data" SECOND
    mv "$SCRATCH/robustcd/data/SECOND.tar.part" "$SCRATCH/robustcd/data/SECOND.tar"
fi
ls -l "$SCRATCH/robustcd/data/SECOND.tar"
du -sh "$DATA"/train "$DATA"/test
echo "DONE"
