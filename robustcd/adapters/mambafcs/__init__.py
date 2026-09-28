"""Adapter that runs Mamba-FCS (Buddhi19/MambaFCS, IEEE JSTARS 2026) under the robustcd protocol.

Mamba-FCS is a fork of the ChangeMamba codebase. It has a VMamba-Base siamese
encoder (ImageNet ``vssm_base_0229``), a change decoder with
spatio-frequency (FFT) fusion, change-guided semantic decoders and a SeK loss.
It shares ChangeMamba's conventions, and the adapter reuses them from
``robustcd.adapters.changemamba`` after checking them against the upstream code:

- data layout ``T1/T2/GT_T1/GT_T2/GT_CD`` with labels in **ChangeMamba class
  order** (``annotations/MambaFCS.ipynb``: 1 low vegetation, 2 non-vegetated
  ground, 3 tree, 4 water, 5 building, 6 playground). ``prepare_split`` writes
  it and ``SECOND_TO_CM`` / ``CM_TO_SECOND`` convert;
- ImageNet input normalisation (``datasets/imutils.normalize_img``);
- a 2-channel change head decoded by argmax, and 7-channel semantic heads
  whose class 0 is ignored (set to 255) in training.

Kept from upstream (commit ``FCS_COMMIT``): the model, its loss (see
``train.py``), AdamW (lr 1e-4, weight decay 5e-4), StepLR with gamma 0.5, and
the augmentation apart from its photometric part. The adapter replaces the
training loop, checkpoint selection and prediction export, as for every model.

Upstream imports its code as the package ``MambaFCS`` (``from
MambaFCS.changedetection ...``), so the checkout directory must be named
``MambaFCS`` and its *parent* goes on ``sys.path``.

The CUDA selective-scan extension (``selective_scan_cuda_oflex``) is built from
the same sources as ChangeMamba's. Upstream only adds a CUDA 13/CUB 3
compatibility shim and different gencode flags, so the extension already built
for ChangeMamba in the robustcd GPU env is reused.
"""

from __future__ import annotations

import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Optional

from robustcd.adapters.changemamba import (  # noqa: F401  (re-exported: identical conventions)
    CM_COLORS,
    CM_TO_SECOND,
    SECOND_TO_CM,
    check_commit,
    decode,
    normalize,
    prepare_split,
)

FCS_URL = "https://github.com/Buddhi19/MambaFCS.git"
FCS_COMMIT = "5c01f1ac86adff89d6dc5eb6e02d83ee8c16f25c"
#: released checkpoint (Hugging Face buddhi19/MambaFCS) and the published SECOND scores (README table)
RELEASED = {"file": "SECOND_SeK_0.255.pth",
            "url": "https://huggingface.co/buddhi19/MambaFCS/resolve/main/SECOND_SeK_0.255.pth",
            "sha256": "b1f3252f2761fd63c0b6076ee0abdc5c95a079716f0ff29733581d6d327a1522",
            "published": {"OA": 0.8862, "Fscd": 0.6578, "mIoU": 0.7407, "SeK": 0.2550}}
#: ImageNet VMamba-Base backbone (ChangeMamba Zenodo record 15479555)
BACKBONE = {"file": "vssm_base_0229_ckpt_epoch_237.pth", "md5": "3e4110259f482f552dc70abcf3381e71"}
CFG_REL = "changedetection/configs/vssm1/vssm_base_224.yaml"


def add_to_path(fcs_root: str | os.PathLike) -> None:
    """Make ``import MambaFCS`` resolve to the pinned checkout."""
    root = Path(fcs_root).resolve()
    if root.name != "MambaFCS":
        raise ValueError(f"the checkout directory must be named 'MambaFCS' (upstream imports), got {root}")
    parent = str(root.parent)
    if parent not in sys.path:
        sys.path.insert(0, parent)


def build_model(fcs_root: str | os.PathLike, cfg_path: Optional[str] = None):
    """STMambaSCD exactly as upstream ``Trainer.__init__``, without the encoder weights.

    Encoder weights are loaded separately with ``load_backbone``: upstream's
    loader catches every exception and only prints it, so a missing or
    mismatched file would silently train from scratch.
    """
    add_to_path(fcs_root)
    from MambaFCS.changedetection.configs.config import get_config
    from MambaFCS.changedetection.models.STMambaSCD import STMambaSCD

    cfg_path = cfg_path or str(Path(fcs_root) / CFG_REL)
    config = get_config(Namespace(cfg=cfg_path, opts=None))
    V = config.MODEL.VSSM
    model = STMambaSCD(
        output_cd=2, output_clf=7, pretrained=None,
        patch_size=V.PATCH_SIZE, in_chans=V.IN_CHANS, num_classes=config.MODEL.NUM_CLASSES,
        depths=V.DEPTHS, dims=V.EMBED_DIM,
        ssm_d_state=V.SSM_D_STATE, ssm_ratio=V.SSM_RATIO, ssm_rank_ratio=V.SSM_RANK_RATIO,
        ssm_dt_rank=("auto" if V.SSM_DT_RANK == "auto" else int(V.SSM_DT_RANK)),
        ssm_act_layer=V.SSM_ACT_LAYER, ssm_conv=V.SSM_CONV, ssm_conv_bias=V.SSM_CONV_BIAS,
        ssm_drop_rate=V.SSM_DROP_RATE, ssm_init=V.SSM_INIT, forward_type=V.SSM_FORWARDTYPE,
        mlp_ratio=V.MLP_RATIO, mlp_act_layer=V.MLP_ACT_LAYER, mlp_drop_rate=V.MLP_DROP_RATE,
        drop_path_rate=config.MODEL.DROP_PATH_RATE, patch_norm=V.PATCH_NORM, norm_layer=V.NORM_LAYER,
        downsample_version=V.DOWNSAMPLE, patchembed_version=V.PATCHEMBED, gmlp=V.GMLP,
        use_checkpoint=config.TRAIN.USE_CHECKPOINT,
    )
    return model, config


def load_backbone(model, path: str | os.PathLike, min_matched: int = 100) -> dict:
    """Load the ImageNet VMamba checkpoint into ``model.encoder`` (upstream: strict=False on key "model")."""
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = ck.get("model", ck)
    enc = model.encoder.state_dict()
    matched = {k: v for k, v in sd.items() if k in enc and enc[k].shape == v.shape}
    res = model.encoder.load_state_dict(matched, strict=False)
    report = {"matched": len(matched),
              "unexpected": sorted(k for k in sd if k not in enc),
              "shape_mismatch": sorted(k for k in sd if k in enc and enc[k].shape != sd[k].shape),
              "missing": sorted(res.missing_keys)}
    if len(matched) < min_matched:
        raise RuntimeError(f"only {len(matched)} backbone tensors matched from {path}: {report}")
    return report


def load_weights(model, path: str | os.PathLike, strict: bool = True) -> dict:
    """Load a released upstream state dict or a robustcd checkpoint ({"model": ...})."""
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    sd = obj["model"] if isinstance(obj, dict) and isinstance(obj.get("model"), dict) else obj
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=strict)
    return {"loaded_keys": len(sd), "missing_keys": list(res.missing_keys),
            "unexpected_keys": list(res.unexpected_keys)}


def normalize_pair(im1, im2):
    return normalize(im1), normalize(im2)


def decode_outputs(outputs, mode: str = "restricted"):
    """(change logits [B,2,H,W], sem_t1, sem_t2) -> uint8 (sem1, sem2, change) in robustcd order."""
    return decode(outputs[0], outputs[1], outputs[2], mode)
