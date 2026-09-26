"""Write predictions of a Ding-codebase model in the robustcd layout (for scripts/evaluate.py).

    # released SCanNet checkpoint, upstream decoding, scored on the SECOND test split
    python -m robustcd.adapters.ding.predict --model scannet --repo ~/ext/SCanNet \\
        --ckpt ~/weights/scannet/SCanNet_32e_mIoU73.37_Sek23.94_Fscd63.66_OA87.86.pth \\
        --source $SLURM_TMPDIR/SECOND/test --out preds/scannet_released --decode full --evaluate

    # a robustcd run on a rendered degraded set
    python -m robustcd.adapters.ding.predict --model scannet --repo ~/ext/SCanNet \\
        --ckpt runs/scannet/seed0/best_model.pth --rendered-set <set_dir> --out preds/<set>

Output: <out>/im1/<id>.png, <out>/im2/<id>.png (uint8, SECOND class index,
0 = no change), <out>/change/<id>.png (0/255), and <out>/predict_info.json.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["scannet", "ted", "bisrnet", "sscdl", "hrscd4"])
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", required=True)
    src = ap.add_mutually_exclusive_group(required=True)
    src.add_argument("--source", help="SECOND-style split root (dir or zip)")
    src.add_argument("--rendered-set", help="rendered degraded set directory")
    ap.add_argument("--ids", default=None)
    ap.add_argument("--limit", type=int, default=None, help="debug: first N ids")
    ap.add_argument("--out", required=True)
    ap.add_argument("--decode", default="restricted", choices=["restricted", "full"])
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--evaluate", action="store_true", help="also score (needs labels in --source)")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import torch

    from robustcd.adapters import common, ding

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model = ding.build_model(args.model, args.repo, imagenet=False)
    info = ding.load_weights(model, args.ckpt, strict=True)
    model.to(device)
    reader = common.open_reader(args.source, args.rendered_set)
    ids = common.read_ids(args.ids, reader.ids)[: args.limit]
    spec = ding.MODELS[args.model]
    meta = {"model": args.model, "ckpt": str(Path(args.ckpt).resolve()), "source": args.source,
            "rendered_set": args.rendered_set, "repo_commit": ding.check_commit(args.repo),
            "expected_commit": ding.REPOS[spec.repo][1], "loaded_keys": info["loaded_keys"],
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "published": spec.published}
    res = common.write_predictions(model, reader, ids, args.out, device, ding.normalize_pair, ding.decode,
                                   args.batch_size, args.decode, score=args.evaluate, meta=meta)
    msg = f"{res['n_images']} images -> {args.out}  ({res['images_per_s']} img/s, decode={args.decode})"
    if "scores" in res:
        msg += "  " + "  ".join(f"{k}={v:.4f}" for k, v in res["scores"].items())
    print(msg, flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
