"""Write ChangeMamba predictions in the robustcd layout, for scripts/evaluate.py.

    python -m robustcd.adapters.changemamba.predict \\
        --cm-root ~/ext/ChangeMamba --cfg <cfg.yaml> --ckpt best_model.pth \\
        --source $SLURM_TMPDIR/SECOND/test --out preds/clean

    # a degraded set rendered by scripts/render_misregistration.py
    python -m robustcd.adapters.changemamba.predict ... \\
        --rendered-set $SCRATCH/robustcd/degraded/SECOND/test/misreg-shift_int-s2 --out preds/misreg-shift_int-s2

Output: <out>/im1/<id>.png, <out>/im2/<id>.png (uint8, SECOND class index,
0 = no change), <out>/change/<id>.png (0/255), and <out>/predict_info.json.
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cm-root", required=True, help="pinned ChangeMamba checkout")
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--ckpt", required=True, help="full-model checkpoint (upstream release or robustcd run)")
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", help="SECOND-style split root (dir or zip)")
    src.add_argument("--rendered-set", help="rendered degraded set directory")
    ap.add_argument("--ids", default=None, help="optional id list file")
    ap.add_argument("--out", required=True)
    ap.add_argument("--decode", default="restricted", choices=["restricted", "full"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--evaluate", action="store_true", help="also print robustcd scores (needs labels in --source)")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import torch
    from PIL import Image

    from robustcd.adapters import changemamba as cm
    from robustcd.adapters.changemamba.infer import RenderedReader, predict, read_ids
    from robustcd.datasets.second import SecondLike
    from robustcd.metrics import SCDMeter, to_index

    cm.add_to_path(args.cm_root)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, _ = cm.build_model(args.cfg, encoder_pretrained=None)
    info = cm.load_weights(model, args.ckpt)
    model.to(device)

    reader = SecondLike(args.source) if args.source else RenderedReader(args.rendered_set)
    ids = read_ids(args.ids, reader.ids)
    out = Path(args.out)
    for d in ("im1", "im2", "change"):
        (out / d).mkdir(parents=True, exist_ok=True)

    meter = SCDMeter() if args.evaluate else None
    t0 = time.time()
    n = 0
    for chunk, samples, s1, s2, ch in predict(model, reader, ids, device, args.batch_size, args.decode):
        for sid, smp, p1, p2, c in zip(chunk, samples, s1, s2, ch):
            Image.fromarray(p1).save(out / "im1" / f"{sid}.png")
            Image.fromarray(p2).save(out / "im2" / f"{sid}.png")
            Image.fromarray((c * 255).astype(np.uint8)).save(out / "change" / f"{sid}.png")
            if meter is not None:
                meter.update(p1, p2, to_index(smp["label1"]), to_index(smp["label2"]), sample_id=sid)
            n += 1
    dt = time.time() - t0
    meta = {
        "n_images": n, "seconds": round(dt, 1), "images_per_s": round(n / max(dt, 1e-9), 2),
        "ckpt": str(Path(args.ckpt).resolve()), "cfg": args.cfg, "decode": args.decode,
        "source": args.source, "rendered_set": args.rendered_set,
        "cm_commit": cm.check_commit(args.cm_root), "expected_cm_commit": cm.CM_COMMIT,
        "loaded_keys": info.get("loaded_keys"), "unexpected_keys": len(info.get("unexpected_keys") or []),
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
    }
    if meter is not None:
        meta["scores"] = {k: meter.compute()[k] for k in ("SeK", "mIoU", "Fscd", "SeK_fromto", "Fscd_fromto")}
    (out / "predict_info.json").write_text(json.dumps(meta, indent=2))
    msg = f"{n} images -> {out}  ({meta['images_per_s']} img/s, decode={args.decode})"
    if meter is not None:
        msg += "  " + "  ".join(f"{k}={v:.4f}" for k, v in meta["scores"].items())
    print(msg, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
