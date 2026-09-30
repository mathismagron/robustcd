"""Adapter acceptance check: the released PerASCD checkpoint on the SECOND test split.

Upstream selects on the test split and logs the SECOND definition of SeK
(``SCDD_eval_from_hist``), so upstream decoding should give the TensorBoard
values of epoch 40 (SeK 26.11, Fscd 66.41). The csf-mamba teacher check
reproduced SeK 0.261079 on the same test images with upstream's explicit
attention; this run also exercises the SDPA attention path and the compiled
deformable-attention op.

    python -m robustcd.adapters.perascd.validate_release --repo ~/ext/PerASCD \\
        --ckpt ~/weights/perascd/PerAChain_40e_mIoU74.33_Sek26.11_Fscd66.41_OA88.70.pth \\
        --source $SLURM_TMPDIR/SECOND/test --json results/model_checks/perascd_released.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--source", required=True, help="SECOND test split (dir or zip)")
    ap.add_argument("--msda", default="auto", choices=["auto", "cuda", "pytorch"])
    ap.add_argument("--upstream-attention", action="store_true", help="explicit attention instead of SDPA")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=4)
    ap.add_argument("--json", required=True)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import torch

    from robustcd.adapters import common
    from robustcd.adapters import perascd as P

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, info = P.build_model(args.repo, "ViT-G/16/1024", droppath=0.0, msda=args.msda,
                                sdpa=not args.upstream_attention)
    load = P.load_weights(model, args.ckpt, strict=True)
    model.to(device)
    reader = common.open_reader(args.source)
    ids = reader.ids[: args.limit]
    res = common.score_release(model, reader, ids, device, P.normalize_pair, P.decode_released,
                               args.batch_size, P.RELEASED["published"])
    record = {
        "what": f"PerASCD released checkpoint run through the robustcd adapter on SECOND test ({len(ids)} images)",
        "checkpoint": f"{Path(args.ckpt).name} (md5 {common.md5sum(args.ckpt)})",
        "repo_commit": P.check_commit(args.repo), "expected_commit": P.PERASCD_COMMIT,
        "build": info, "n_params": sum(p.numel() for p in model.parameters()),
        "load": {"loaded_keys": load["loaded_keys"], "renamed": len(load["renamed_keys"]),
                 "missing": len(load["missing_keys"]), "unexpected": len(load["unexpected_keys"]),
                 "checkpoint_meta": load["checkpoint_meta"], "epoch": load["epoch"]},
        "class_order": "released checkpoint predicts in ChangeMamba order, mapped to robustcd",
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        "max_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 1) if device.type == "cuda" else None,
        **res,
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(record, indent=2))
    print(json.dumps({k: record[k] for k in ("upstream_decoding", "protocol_decoding", "decodings_identical",
                                             "images_per_s", "max_mem_gb")}, indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
