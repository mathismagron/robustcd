"""Adapter acceptance check: score a released checkpoint on the SECOND test split.

One forward pass per image (fp32, no TTA), scored twice: with upstream
decoding (argmax over all 7 channels) to compare against the published
numbers, and with the protocol decoding (classes 1..6) as the reference value
for later comparison, with a 1000-replicate image bootstrap CI.

    python -m robustcd.adapters.ding.validate_release --model scannet --repo ext/SCanNet \\
        --ckpt weights/scannet/SCanNet_32e_mIoU73.37_Sek23.94_Fscd63.66_OA87.86.pth \\
        --source /path/to/SECOND_total_test.zip --json results/model_checks/scannet_released.json
"""

from __future__ import annotations

import argparse
import hashlib
import json
import sys
import time
from pathlib import Path


def md5(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 22), b""):
            h.update(chunk)
    return h.hexdigest()


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--source", required=True, help="SECOND test split (dir or zip)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--json", required=True)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import numpy as np
    import torch

    from robustcd.adapters import common, ding
    from robustcd.metrics import SCDMeter, bootstrap_ci, to_index

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    spec = ding.MODELS[args.model]
    model = ding.build_model(args.model, args.repo, imagenet=False)
    info = ding.load_weights(model, args.ckpt, strict=True)
    model.to(device).eval()
    reader = common.open_reader(args.source)
    ids = reader.ids[: args.limit]

    meters = {"full": SCDMeter(), "restricted": SCDMeter()}
    t0 = time.time()
    with torch.no_grad():
        for n_done, chunk in enumerate(common.batches(ids, args.batch_size)):
            samples = [reader.get(s) for s in chunk]
            pairs = [ding.normalize_pair(s["im1"], s["im2"]) for s in samples]
            x1 = torch.from_numpy(np.stack([p[0] for p in pairs])).to(device)
            x2 = torch.from_numpy(np.stack([p[1] for p in pairs])).to(device)
            outputs = tuple(o.float() for o in model(x1, x2))
            for mode, meter in meters.items():
                s1, s2, _ = ding.decode(outputs, mode)
                for sid, smp, p1, p2 in zip(chunk, samples, s1, s2):
                    meter.update(p1, p2, to_index(smp["label1"]), to_index(smp["label2"]), sample_id=sid)
            if n_done % 25 == 0:
                print(f"{min((n_done + 1) * args.batch_size, len(ids))}/{len(ids)} images "
                      f"{time.time() - t0:.0f}s", flush=True)
    dt = time.time() - t0

    keys = ("SeK", "mIoU", "Fscd", "kappa_n0", "SeK_fromto", "Fscd_fromto")
    res = {m: {k: round(float(v), 4) for k, v in meters[m].compute().items() if k in keys} for m in meters}
    r = meters["restricted"]
    ci = bootstrap_ci(r.per_sample, r.score_fn, keys=("SeK", "mIoU", "Fscd"), n_boot=1000, seed=0)
    pub = spec.published or {}
    record = {
        "what": f"{args.model} released checkpoint run through the robustcd adapter on SECOND test "
                f"({len(ids)} images)",
        "checkpoint": f"{Path(args.ckpt).name} (md5 {md5(args.ckpt)})",
        "repo_commit": ding.check_commit(args.repo), "expected_commit": ding.REPOS[spec.repo][1],
        "load": {"loaded_keys": info["loaded_keys"], "missing": len(info["missing_keys"]),
                 "unexpected": len(info["unexpected_keys"])},
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "upstream_decoding": {**res["full"], "published": pub,
                              "abs_diff_vs_published": {k: round(res["full"][k] - pub[k], 4)
                                                        for k in ("SeK", "mIoU", "Fscd") if k in pub}},
        "protocol_decoding": {**res["restricted"],
                              "ci95": {k: [round(v["lo"], 4), round(v["hi"], 4)] for k, v in ci.items()},
                              "bootstrap": "1000 image replicates"},
        "decodings_identical": res["full"] == res["restricted"],
        "seconds": round(dt, 1), "images_per_s": round(len(ids) / max(dt, 1e-9), 2),
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(record, indent=2))
    print(json.dumps({k: record[k] for k in ("upstream_decoding", "protocol_decoding", "decodings_identical")},
                     indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
