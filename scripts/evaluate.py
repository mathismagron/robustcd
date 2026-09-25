#!/usr/bin/env python3
"""Score predictions written to disk, with the benchmark's single metric code.

Protocol: every model repo only *writes predictions*; this script scores all of
them.  No model is ever scored by its own repo's metric code.

Semantic change detection (SECOND-style)::

    <pred>/im1/<id>.png   date-1 semantic map, 0 / white = no change
    <pred>/im2/<id>.png   date-2 semantic map
    <pred>/change/<id>.png  optional binary change map; if present, im1/im2 are
                            masked with it (robustcd.metrics.compose_prediction)

  PNGs may be index maps (HxW) or SECOND colour maps (HxWx3).

    python scripts/evaluate.py scd --gt /data/SECOND/test --pred runs/cnn/pred_clean \
        --out runs/cnn/eval_clean

Binary change detection (LEVIR-CD-style)::

    python scripts/evaluate.py bcd --gt-dir /data/LEVIR-CD/test/label \
        --pred runs/cnn/levir_pred --out runs/cnn/levir_eval

``--valid-from <rendered set dir>`` restricts scoring to the valid mask of a
degraded set (interior-only protocol); omit it to score the full tile.

Outputs in ``--out``: ``metrics.json`` (point scores + bootstrap CIs) and
``confusion.npz`` (per-image matrices, for later paired comparisons with
``scripts/compare.py``).  Missing prediction files are an error.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robustcd.datasets.second import SecondLike  # noqa: E402
from robustcd.metrics import BCDMeter, SCDMeter, bootstrap_ci, compose_prediction, to_index  # noqa: E402
from robustcd.metrics.io import save_confusion  # noqa: E402


def _read(path: Path) -> np.ndarray:
    with Image.open(path) as im:
        if im.mode == "P":
            return np.asarray(im)  # palette index = class index
        return np.asarray(im)


def _index_files(d: Path) -> Dict[str, Path]:
    return {p.stem: p for p in d.iterdir() if p.is_file() and p.suffix.lower() in (".png", ".tif", ".tiff")}


def _valid_masks(set_dir: Optional[str]) -> Optional[Dict[str, Path]]:
    if not set_dir:
        return None
    spec = json.loads((Path(set_dir) / "dataset_spec.json").read_text())
    info = spec["streams"]["valid_mask"]
    base = Path(info["root"])
    d = base if base.name == info["subdir"] else base / info["subdir"]
    return _index_files(d)


def _mask(masks, sid) -> Optional[np.ndarray]:
    if masks is None:
        return None
    if sid not in masks:
        raise SystemExit(f"no valid mask for sample {sid}")
    return _read(masks[sid]) > 127


def run_scd(args) -> Dict[str, object]:
    gt = SecondLike(args.gt, name=args.dataset)
    pred = Path(args.pred)
    f1, f2 = _index_files(pred / "im1"), _index_files(pred / "im2")
    fch = _index_files(pred / "change") if (pred / "change").is_dir() else None
    ids = gt.ids if not args.limit else gt.ids[: args.limit]
    missing = [s for s in ids if s not in f1 or s not in f2 or (fch is not None and s not in fch)]
    if missing:
        raise SystemExit(f"{len(missing)} predictions missing, e.g. {missing[:5]}")
    masks = _valid_masks(args.valid_from)

    meter = SCDMeter(num_classes=args.num_classes)
    for sid in ids:
        g = gt.get(sid)
        g1, g2 = to_index(g["label1"]), to_index(g["label2"])
        p1, p2 = to_index(_read(f1[sid])), to_index(_read(f2[sid]))
        if fch is not None:
            ch = _read(fch[sid])
            ch = (ch.max(-1) if ch.ndim == 3 else ch) > 0
            p1, p2 = compose_prediction(p1, ch), compose_prediction(p2, ch)
        meter.update(p1, p2, g1, g2, valid=_mask(masks, sid), sample_id=sid)
    return {"meter": meter, "keys": ("SeK", "mIoU", "Fscd", "kappa_n0", "IoU_change", "OA")}


def run_bcd(args) -> Dict[str, object]:
    gdir, pred = Path(args.gt_dir), Path(args.pred)
    fg, fp = _index_files(gdir), _index_files(pred)
    ids = sorted(fg) if not args.limit else sorted(fg)[: args.limit]
    missing = [s for s in ids if s not in fp]
    if missing:
        raise SystemExit(f"{len(missing)} predictions missing, e.g. {missing[:5]}")
    masks = _valid_masks(args.valid_from)
    meter = BCDMeter()
    for sid in ids:
        g, p = _read(fg[sid]), _read(fp[sid])
        g = g.max(-1) if g.ndim == 3 else g
        p = p.max(-1) if p.ndim == 3 else p
        meter.update(p, g, valid=_mask(masks, sid), sample_id=sid)
    return {"meter": meter, "keys": ("F1", "IoU_change", "precision", "recall", "OA", "kappa")}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="task", required=True)
    for name in ("scd", "bcd"):
        p = sub.add_parser(name)
        p.add_argument("--pred", required=True)
        p.add_argument("--out", required=True)
        p.add_argument("--valid-from", default=None, help="rendered degraded set whose valid_mask to apply")
        p.add_argument("--bootstrap", type=int, default=1000, help="replicates; 0 disables")
        p.add_argument("--seed", type=int, default=0)
        p.add_argument("--limit", type=int, default=None)
        p.add_argument("--tag", default="", help="free-form run label stored in metrics.json")
    sub.choices["scd"].add_argument("--gt", required=True, help="SECOND-style split root (dir or .zip)")
    sub.choices["scd"].add_argument("--dataset", default="SECOND")
    sub.choices["scd"].add_argument("--num-classes", type=int, default=7)
    sub.choices["bcd"].add_argument("--gt-dir", required=True, help="directory of binary label PNGs")
    args = ap.parse_args()

    t0 = time.time()
    res = run_scd(args) if args.task == "scd" else run_bcd(args)
    meter = res["meter"]
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    save_confusion(out / "confusion.npz", meter.sample_ids, meter.per_sample, args.task)

    report = {
        "task": args.task, "tag": args.tag, "pred": str(Path(args.pred).resolve()),
        "valid_from": args.valid_from, "n_images": len(meter),
        "aggregation": "dataset-level confusion over all images (and both dates for scd)",
        "scores": meter.compute(), "confusion_total": meter.total.tolist(),
    }
    if args.bootstrap and len(meter) > 1:
        report["bootstrap"] = {
            "n_boot": args.bootstrap, "alpha": 0.05, "seed": args.seed, "unit": "image",
            "ci": bootstrap_ci(meter.per_sample, meter.score_fn, keys=res["keys"],
                               n_boot=args.bootstrap, seed=args.seed),
        }
    (out / "metrics.json").write_text(json.dumps(report, indent=2))
    s = report["scores"]
    head = [k for k in res["keys"] if k in s][:4]
    print(f"{len(meter)} images  " + "  ".join(f"{k}={s[k]:.4f}" for k in head) + f"  ({time.time()-t0:.0f}s)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
