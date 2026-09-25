#!/usr/bin/env python3
"""Render misregistration-degraded copies of a SECOND-style split.

One rendered set per (mode, severity).  Streams that the degradation does not
touch (date 1 and the labels, in the default ``--apply-to t2`` configuration)
are *not* duplicated: ``dataset_spec.json`` in each set records where to read
them from, so a 12-point grid costs ~1/4 of the naive disk footprint.  Pass
``--materialize-all`` for self-contained directories instead.

Example (full SECOND test grid)::

    python scripts/render_misregistration.py \
        --source /data/SECOND/SECOND_total_test.zip \
        --out    /scratch/$USER/robustcd/degraded \
        --dataset SECOND --split test --jobs 8

Re-running is idempotent: existing outputs are skipped unless ``--overwrite``.
"""

from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import sys
import time
from pathlib import Path
from typing import List, Optional

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robustcd.datasets.second import SecondLike  # noqa: E402
from robustcd.degradations import misregistration as mis  # noqa: E402

_DS: Optional[SecondLike] = None
_ARGS: Optional[argparse.Namespace] = None


def _init(source: str, dataset: str, inner_prefix: str, args: argparse.Namespace) -> None:
    global _DS, _ARGS
    _DS = SecondLike(source, inner_prefix=inner_prefix, name=dataset)
    _ARGS = args


def _save_png(arr: np.ndarray, path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    mode = "L" if arr.ndim == 2 else None
    Image.fromarray(arr, mode=mode).save(path, format="PNG", compress_level=6)


def _spec_of(tag: str, specs: List[mis.MisregSpec]) -> mis.MisregSpec:
    return next(s for s in specs if s.tag == tag)


def _render_one(job) -> dict:
    sample_id, tag, out_dir = job
    ds, args = _DS, _ARGS
    assert ds is not None and args is not None
    spec = _spec_of(tag, mis.grid(
        apply_to=args.apply_to, border=args.border, interp_order=args.interp_order,
        warp_labels=args.warp_labels, global_seed=args.seed,
    ))
    out_dir = Path(out_dir)
    im2_path = out_dir / "im2" / f"{sample_id}.png"
    if im2_path.exists() and not args.overwrite:
        return {"sample_id": sample_id, "skipped": True}

    data = ds.get(sample_id)
    res = mis.degrade_pair(
        data["im1"], data["im2"], spec, sample_id=sample_id, dataset=ds.name,
        label1=data.get("label1"), label2=data.get("label2"),
    )
    _save_png(res["im2"], im2_path)
    _save_png((res["valid_mask"].astype(np.uint8) * 255), out_dir / "valid_mask" / f"{sample_id}.png")

    warped_t1 = spec.apply_to == "split"
    if args.materialize_all or warped_t1:
        _save_png(res["im1"], out_dir / "im1" / f"{sample_id}.png")
    for lab in ("label1", "label2"):
        if lab in res and (args.materialize_all or spec.warp_labels):
            _save_png(res[lab], out_dir / lab / f"{sample_id}.png")
    if args.materialize_all and not warped_t1:
        pass  # im1 already written above
    return res["record"]


def main() -> int:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--source", required=True, help="split root: directory or .zip")
    p.add_argument("--out", required=True, help="output root")
    p.add_argument("--dataset", default="SECOND")
    p.add_argument("--split", default="test")
    p.add_argument("--inner-prefix", default="", help="path inside the zip above im1/ (auto-detected)")
    p.add_argument("--modes", nargs="+", default=list(mis.MODES), choices=list(mis.MODES))
    p.add_argument("--severities", nargs="+", type=int, default=[1, 2, 3], choices=[1, 2, 3])
    p.add_argument("--apply-to", default="t2", choices=["t2", "split"])
    p.add_argument("--border", default="reflect", choices=["reflect", "constant", "nearest"])
    p.add_argument("--interp-order", type=int, default=3)
    p.add_argument("--warp-labels", action="store_true", help="ablation: warp labels with date 2")
    p.add_argument("--seed", type=int, default=20260101)
    p.add_argument("--limit", type=int, default=None, help="first N samples (smoke tests)")
    p.add_argument("--ids", default=None, help="file with one sample id per line")
    p.add_argument("--jobs", type=int, default=1)
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--materialize-all", action="store_true", help="also copy pass-through streams")
    args = p.parse_args()

    ds = SecondLike(args.source, inner_prefix=args.inner_prefix, name=args.dataset)
    ids = ds.ids
    if args.ids:
        wanted = [l.strip() for l in Path(args.ids).read_text().splitlines() if l.strip()]
        missing = sorted(set(wanted) - set(ids))
        if missing:
            raise SystemExit(f"{len(missing)} requested ids absent, e.g. {missing[:5]}")
        ids = wanted
    if args.limit:
        ids = ids[: args.limit]
    specs = [
        s for s in mis.grid(
            modes=args.modes, severities=args.severities, apply_to=args.apply_to,
            border=args.border, interp_order=args.interp_order,
            warp_labels=args.warp_labels, global_seed=args.seed,
        )
    ]
    out_root = Path(args.out) / args.dataset / args.split
    print(f"{len(ids)} samples x {len(specs)} configs -> {out_root}", flush=True)

    jobs = []
    for spec in specs:
        for sid in ids:
            jobs.append((sid, spec.tag, str(out_root / spec.tag)))

    t0 = time.time()
    records = []
    if args.jobs > 1:
        with mp.Pool(args.jobs, initializer=_init, initargs=(args.source, args.dataset, ds.inner_prefix, args)) as pool:
            for i, rec in enumerate(pool.imap_unordered(_render_one, jobs, chunksize=8), 1):
                records.append(rec)
                if i % 200 == 0:
                    print(f"  {i}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)
    else:
        _init(args.source, args.dataset, ds.inner_prefix, args)
        for i, job in enumerate(jobs, 1):
            records.append(_render_one(job))
            if i % 200 == 0:
                print(f"  {i}/{len(jobs)}  {time.time()-t0:.0f}s", flush=True)

    src_root = str(Path(args.source).resolve())
    for spec in specs:
        d = out_root / spec.tag
        d.mkdir(parents=True, exist_ok=True)
        recs = [r for r in records if not r.get("skipped") and r.get("mode") == spec.mode
                and r.get("severity") == spec.severity]
        if recs:
            with (d / "manifest.jsonl").open("w") as fh:
                for r in sorted(recs, key=lambda r: r["sample_id"]):
                    fh.write(json.dumps(r) + "\n")
        materialized = {"im2", "valid_mask"} | ({"im1"} if (args.materialize_all or spec.apply_to == "split") else set())
        if args.materialize_all or spec.warp_labels:
            materialized |= {s for s in ("label1", "label2") if s in ds.available_streams}
        streams = {
            s: {"root": str(d.resolve()) if s in materialized else src_root,
                "subdir": s, "degraded": s in materialized}
            for s in ds.available_streams + ["valid_mask"]
        }
        (d / "dataset_spec.json").write_text(json.dumps({
            "dataset": args.dataset, "split": args.split, "source_root": src_root,
            "n_samples": len(ids), "spec": spec.to_dict(), "streams": streams,
            "renderer": "scripts/render_misregistration.py",
        }, indent=2))

    n_new = sum(1 for r in records if not r.get("skipped"))
    print(f"done: {n_new} rendered, {len(records)-n_new} skipped, {time.time()-t0:.0f}s", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
