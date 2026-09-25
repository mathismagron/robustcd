#!/usr/bin/env python3
"""Verify a SECOND-style split before any training or evaluation touches it.

Checks every sample, not a subset:

* the four streams (im1, im2, label1, label2) have the same ids;
* images are HxWx3 uint8 and all four streams share one shape;
* every label pixel decodes with the strict SECOND colour table;
* "no change" is identical in label1 and label2 (a SECOND invariant);

and records per-class pixel counts, the change ratio and a content hash of
the sorted id list.  Run it locally and on the cluster: the two JSON files must
agree, which is what "the data on the cluster is the data we validated" means.

    python scripts/check_dataset.py --root /path/to/SECOND/test --out second_test_check.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robustcd.datasets.second import SecondLike  # noqa: E402
from robustcd.metrics.labels import SECOND_CLASSES, rgb_to_index  # noqa: E402

_DS: Optional[SecondLike] = None


def _init(root: str) -> None:
    global _DS
    _DS = SecondLike(root)


def _check(sid: str) -> dict:
    ds = _DS
    assert ds is not None
    out = {"id": sid, "errors": []}
    try:
        s = ds.get(sid)
    except Exception as e:  # noqa: BLE001
        out["errors"].append(f"read: {type(e).__name__}: {e}")
        return out
    missing = [k for k in ("im1", "im2", "label1", "label2") if k not in s]
    if missing:
        out["errors"].append(f"missing streams {missing}")
        return out
    shapes = {k: s[k].shape for k in s}
    if len({v[:2] for v in shapes.values()}) != 1:
        out["errors"].append(f"shape mismatch {shapes}")
    for k in ("im1", "im2"):
        if s[k].ndim != 3 or s[k].shape[2] != 3 or s[k].dtype != np.uint8:
            out["errors"].append(f"{k} not HxWx3 uint8: {s[k].shape} {s[k].dtype}")
    counts = np.zeros(len(SECOND_CLASSES), np.int64)
    idx = []
    for k in ("label1", "label2"):
        try:
            i = rgb_to_index(s[k])
        except ValueError as e:
            out["errors"].append(f"{k}: {e}")
            return out
        idx.append(i)
        counts += np.bincount(i.ravel(), minlength=len(SECOND_CLASSES))
    if ((idx[0] == 0) != (idx[1] == 0)).any():
        out["errors"].append(f"no-change mask differs between dates: {int(((idx[0] == 0) != (idx[1] == 0)).sum())} px")
    out["shape"] = list(s["im1"].shape)
    out["class_counts"] = counts.tolist()
    out["changed_px"] = int((idx[0] != 0).sum())
    out["px"] = int(idx[0].size)
    return out


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--root", required=True, help="split root with im1/ im2/ label1/ label2/ (dir or .zip)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--expect", default=None, help="reference JSON to compare against (exit 1 on mismatch)")
    args = ap.parse_args()

    t0 = time.time()
    ds = SecondLike(args.root)
    ids = ds.ids
    counts_per_stream = {s: len(ds._members.get(s, {})) for s in ("im1", "im2", "label1", "label2")}
    if args.jobs > 1 and not str(args.root).endswith(".zip"):
        with mp.Pool(args.jobs, initializer=_init, initargs=(args.root,)) as pool:
            rows = pool.map(_check, ids, chunksize=16)
    else:
        _init(args.root)
        rows = [_check(s) for s in ids]

    bad = [r for r in rows if r["errors"]]
    ok = [r for r in rows if not r["errors"]]
    cls = np.sum([r["class_counts"] for r in ok], axis=0) if ok else np.zeros(len(SECOND_CLASSES), int)
    report = {
        "root": str(Path(args.root).resolve()),
        "n_samples": len(ids),
        "files_per_stream": counts_per_stream,
        "id_list_sha256": hashlib.sha256("\n".join(ids).encode()).hexdigest(),
        "shapes": sorted({tuple(r["shape"]) for r in ok}),
        "n_errors": len(bad),
        "errors": [{"id": r["id"], "errors": r["errors"]} for r in bad[:50]],
        "class_pixels_both_dates": dict(zip(SECOND_CLASSES, cls.tolist())),
        "change_ratio": float(sum(r["changed_px"] for r in ok) / max(1, sum(r["px"] for r in ok))),
        "samples_without_change": int(sum(1 for r in ok if r["changed_px"] == 0)),
        "seconds": round(time.time() - t0, 1),
    }
    mismatches = []
    if args.expect:
        ref = json.loads(Path(args.expect).read_text())
        for k in ("n_samples", "files_per_stream", "id_list_sha256", "class_pixels_both_dates",
                  "samples_without_change"):
            if ref.get(k) != report.get(k):
                mismatches.append({"field": k, "expected": ref.get(k), "found": report.get(k)})
        report["expect"] = {"reference": str(Path(args.expect).resolve()), "mismatches": mismatches}

    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(report, indent=2))
    print(f"{report['n_samples']} samples, {report['n_errors']} with errors, "
          f"change ratio {report['change_ratio']:.4f}, ids sha256 {report['id_list_sha256'][:12]}  ({report['seconds']}s)")
    if args.expect:
        print("reference: MATCH" if not mismatches else f"reference: {len(mismatches)} MISMATCH(ES): "
              + ", ".join(m["field"] for m in mismatches))
    return 1 if bad or mismatches or len(set(counts_per_stream.values())) != 1 else 0


if __name__ == "__main__":
    raise SystemExit(main())
