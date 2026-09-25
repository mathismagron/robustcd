"""Build the ChangeMamba folder layout from the verified robustcd SECOND copy.

    python -m robustcd.adapters.changemamba.prepare --second-root $SLURM_TMPDIR/SECOND \\
        --splits ~/robustcd/splits/SECOND --out $SLURM_TMPDIR/cm_SECOND

Writes <out>/train/{T1,T2,GT_T1,GT_T2,GT_CD} for the train + val ids (images
are symlinks; labels are converted to single-channel ChangeMamba class
order). Only the train split is used by the ChangeMamba loader: validation
and test go through robustcd readers on the original data.
"""

from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--second-root", required=True)
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args(argv)
    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    from robustcd.adapters.changemamba import prepare_split
    from robustcd.adapters.changemamba.infer import read_ids

    t0 = time.time()
    ids = read_ids(str(Path(args.splits) / "train.txt"), [])
    n = prepare_split(Path(args.second_root) / "train", Path(args.out) / "train", ids)
    print(f"prepared {n} train samples in {time.time() - t0:.0f}s -> {args.out}/train", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
