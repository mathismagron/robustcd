"""Adapter acceptance check: the released Mamba-FCS checkpoint on the SECOND test split.

Upstream selects its checkpoint on the test set and reports the SECOND
definition of SeK (``SCDD_eval_all``: 7 classes per date, both dates in one
matrix; its ``num_class=37`` argument only adds empty rows and columns). Upstream
decoding should therefore reproduce the published 25.50 up to rounding, as
ChangeMamba did.

    python -m robustcd.adapters.mambafcs.validate_release --fcs-root ~/ext/MambaFCS \\
        --ckpt ~/weights/mambafcs/SECOND_SeK_0.255.pth --source $SLURM_TMPDIR/SECOND/test \\
        --json results/model_checks/mambafcs_released.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fcs-root", required=True)
    ap.add_argument("--cfg", default=None)
    ap.add_argument("--ckpt", required=True)
    ap.add_argument("--source", required=True, help="SECOND test split (dir or zip)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=8)
    ap.add_argument("--json", required=True)
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import torch

    from robustcd.adapters import common
    from robustcd.adapters import mambafcs as fcs

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model, config = fcs.build_model(args.fcs_root, args.cfg)
    info = fcs.load_weights(model, args.ckpt, strict=True)
    model.to(device)
    reader = common.open_reader(args.source)
    ids = reader.ids[: args.limit]
    res = common.score_release(model, reader, ids, device, fcs.normalize_pair, fcs.decode_outputs,
                               args.batch_size, fcs.RELEASED["published"])
    record = {
        "what": f"Mamba-FCS released checkpoint run through the robustcd adapter on SECOND test ({len(ids)} images)",
        "checkpoint": f"{Path(args.ckpt).name} (md5 {common.md5sum(args.ckpt)})",
        "fcs_commit": fcs.check_commit(args.fcs_root), "expected_commit": fcs.FCS_COMMIT,
        "config": {"depths": list(config.MODEL.VSSM.DEPTHS), "dims": config.MODEL.VSSM.EMBED_DIM,
                   "forward_type": config.MODEL.VSSM.SSM_FORWARDTYPE},
        "n_params": sum(p.numel() for p in model.parameters()),
        "load": {"loaded_keys": info["loaded_keys"], "missing": len(info["missing_keys"]),
                 "unexpected": len(info["unexpected_keys"])},
        "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        **res,
    }
    Path(args.json).parent.mkdir(parents=True, exist_ok=True)
    Path(args.json).write_text(json.dumps(record, indent=2))
    print(json.dumps({k: record[k] for k in ("upstream_decoding", "protocol_decoding", "decodings_identical")},
                     indent=1), flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
