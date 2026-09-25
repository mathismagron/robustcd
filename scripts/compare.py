#!/usr/bin/env python3
"""Paired comparison of two evaluated runs (clean vs degraded, or model A vs B).

Reads the ``confusion.npz`` files written by ``scripts/evaluate.py``, pairs them
image-by-image and reports the difference and relative drop with a paired
image-level bootstrap interval.

    python scripts/compare.py runs/cnn/eval_clean runs/cnn/eval_misreg-shift_int-s2 --key SeK
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robustcd.metrics import binary_scores, paired_bootstrap, scd_scores  # noqa: E402
from robustcd.metrics.io import align, load_confusion  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("a", help="reference run directory (e.g. clean)")
    ap.add_argument("b", help="compared run directory (e.g. degraded)")
    ap.add_argument("--key", default=None, help="metric; default SeK (scd) or F1 (bcd)")
    ap.add_argument("--bootstrap", type=int, default=2000)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--json", default=None, help="optional output path")
    args = ap.parse_args()

    ida, psa, ta = load_confusion(Path(args.a) / "confusion.npz")
    idb, psb, tb = load_confusion(Path(args.b) / "confusion.npz")
    if ta != tb:
        raise SystemExit(f"task mismatch: {ta} vs {tb}")
    fn = scd_scores if ta == "scd" else binary_scores
    key = args.key or ("SeK" if ta == "scd" else "F1")
    common, psa, psb = align(ida, psa, idb, psb)
    if len(common) < len(ida) or len(common) < len(idb):
        print(f"warning: paired on {len(common)} common images ({len(ida)} vs {len(idb)})", file=sys.stderr)

    out = {
        "a": args.a, "b": args.b, "key": key, "n_images": len(common),
        "a_value": fn(psa.sum(0))[key], "b_value": fn(psb.sum(0))[key],
        "difference": paired_bootstrap(psa, psb, fn, key, "difference", args.bootstrap, seed=args.seed),
        "relative_drop": paired_bootstrap(psa, psb, fn, key, "relative_drop", args.bootstrap, seed=args.seed),
    }
    d, r = out["difference"], out["relative_drop"]
    print(f"{key}: {out['a_value']:.4f} -> {out['b_value']:.4f}   "
          f"diff {d['value']:+.4f} [{d['lo']:+.4f}, {d['hi']:+.4f}]   "
          f"rel. drop {100*r['value']:.1f}% [{100*r['lo']:.1f}, {100*r['hi']:.1f}]  (n={len(common)})")
    if args.json:
        Path(args.json).write_text(json.dumps(out, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
