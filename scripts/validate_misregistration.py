#!/usr/bin/env python3
"""Verify that the misregistration pipeline applies the displacement it claims.

For each (mode, severity) and each sampled image, the applied warp is measured
back off the rendered pixels by phase correlation and compared with the analytic
displacement field:

* ``global_*``  : whole-image phase correlation vs. the field mean -- the right
  summary for the translation modes.
* ``block_*``   : block-wise phase correlation vs. the analytic field sampled at
  the same block centres -- the summary that also covers ``affine`` and
  ``local_warp``, where displacement varies across the scene.

Writes a per-(sample, config) CSV.  A systematic gap between nominal and
measured means the renderer and the manifest disagree, and the benchmark's
x-axis would be wrong.

    python scripts/validate_misregistration.py \
        --source /data/SECOND/SECOND_total_test.zip --limit 16 --jobs 4 \
        --out results/misregistration_validation.csv
"""

from __future__ import annotations

import argparse
import csv
import multiprocessing as mp
import sys
from pathlib import Path
from typing import Optional

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robustcd.datasets.second import SecondLike  # noqa: E402
from robustcd.degradations import misregistration as mis  # noqa: E402
from robustcd.degradations.measure import estimate_shift, estimate_shift_field  # noqa: E402

_DS: Optional[SecondLike] = None
FIELDS = [
    "sample_id", "mode", "severity", "nominal_px", "warp_kind",
    "analytic_mean_px", "analytic_peak_px",
    "global_measured_px", "global_err_px",
    "block_rmse_px", "block_corr", "block_n",
    "valid_fraction",
]


def _init(source: str, dataset: str, inner_prefix: str) -> None:
    global _DS
    _DS = SecondLike(source, inner_prefix=inner_prefix, name=dataset)


def _one(job) -> dict:
    sid, mode, sev, block, stride = job
    ds = _DS
    assert ds is not None
    spec = mis.MisregSpec(mode=mode, severity=sev)
    data = ds.get(sid)
    res = mis.degrade_pair(data["im1"], data["im2"], spec, sample_id=sid, dataset=ds.name)
    warp = res["warp_t2"]
    shape = data["im2"].shape[:2]

    fld = mis.displacement_field(warp, shape)
    mag = np.hypot(fld[0], fld[1])

    gy, gx = estimate_shift(data["im2"], res["im2"])
    mean_dy, mean_dx = float(fld[0].mean()), float(fld[1].mean())
    global_err = float(np.hypot(gy - mean_dy, gx - mean_dx))

    centres, disp = estimate_shift_field(data["im2"], res["im2"], block=block, stride=stride)
    ci = centres.astype(int)
    truth = np.stack([fld[0][ci[:, 0], ci[:, 1]], fld[1][ci[:, 0], ci[:, 1]]], axis=1)
    resid = disp - truth
    rmse = float(np.sqrt((resid**2).sum(1).mean()))
    flat_m, flat_t = disp.ravel(), truth.ravel()
    corr = float(np.corrcoef(flat_m, flat_t)[0, 1]) if flat_t.std() > 1e-9 else float("nan")

    return {
        "sample_id": sid, "mode": mode, "severity": sev, "nominal_px": spec.displacement_px,
        "warp_kind": warp.kind,
        "analytic_mean_px": float(mag.mean()), "analytic_peak_px": float(mag.max()),
        "global_measured_px": float(np.hypot(gy, gx)), "global_err_px": global_err,
        "block_rmse_px": rmse, "block_corr": corr, "block_n": int(len(centres)),
        "valid_fraction": res["record"]["valid_fraction"],
    }


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True)
    p.add_argument("--dataset", default="SECOND")
    p.add_argument("--inner-prefix", default="")
    p.add_argument("--limit", type=int, default=16)
    p.add_argument("--stride-ids", type=int, default=1, help="take every k-th id, to spread over the split")
    p.add_argument("--modes", nargs="+", default=list(mis.MODES))
    p.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3])
    p.add_argument("--block", type=int, default=96)
    p.add_argument("--stride", type=int, default=64)
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--out", default="results/misregistration_validation.csv")
    args = p.parse_args()

    ds = SecondLike(args.source, inner_prefix=args.inner_prefix, name=args.dataset)
    ids = ds.ids[:: args.stride_ids][: args.limit]
    jobs = [(sid, m, s, args.block, args.stride) for m in args.modes for s in args.severities for sid in ids]
    print(f"{len(ids)} samples x {len(jobs)//len(ids)} configs = {len(jobs)} checks", flush=True)

    if args.jobs > 1:
        with mp.Pool(args.jobs, initializer=_init, initargs=(args.source, args.dataset, ds.inner_prefix)) as pool:
            rows = list(pool.imap_unordered(_one, jobs, chunksize=4))
    else:
        _init(args.source, args.dataset, ds.inner_prefix)
        rows = [_one(j) for j in jobs]

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    rows.sort(key=lambda r: (r["mode"], r["severity"], r["sample_id"]))
    with out.open("w", newline="") as fh:
        wr = csv.DictWriter(fh, fieldnames=FIELDS)
        wr.writeheader()
        wr.writerows(rows)

    print(f"{'config':26s} {'nominal':>8s} {'analytic_pk':>12s} {'global_err':>11s} {'block_rmse':>11s} {'corr':>6s}")
    for m in args.modes:
        for s in args.severities:
            sub = [r for r in rows if r["mode"] == m and r["severity"] == s]
            if not sub:
                continue
            print(f"{m + '-s' + str(s):26s} {sub[0]['nominal_px']:8.2f} "
                  f"{np.mean([r['analytic_peak_px'] for r in sub]):12.3f} "
                  f"{np.mean([r['global_err_px'] for r in sub]):11.3f} "
                  f"{np.mean([r['block_rmse_px'] for r in sub]):11.3f} "
                  f"{np.nanmean([r['block_corr'] for r in sub]):6.3f}")
    print(f"wrote {out}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
