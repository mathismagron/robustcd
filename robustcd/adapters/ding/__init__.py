"""Adapter for the Ding et al. semantic change detection codebases.

Two upstream repositories by the same authors share one data pipeline, one
loss module and one training recipe, so a single adapter covers five models:

==========  =======================  ==================================  =====
name        upstream class           repository                          notes
==========  =======================  ==================================  =====
scannet     ``models.SCanNet``       DingLei14/SCanNet  (``SCANNET``)    released SECOND checkpoint
ted         ``models.TED``           DingLei14/SCanNet                   SCanNet without its transformer
bisrnet     ``models.BiSRNet``       DingLei14/Bi-SRNet (``BISRNET``)
sscdl       ``models.SSCDl``         DingLei14/Bi-SRNet                  not in the campaign
hrscd4      ``models.Daudt.HRSCD4``  DingLei14/Bi-SRNet                  Daudt et al. 2019, strategy 4
==========  =======================  ==================================  =====

Neither repository has a licence file. Upstream code is therefore never copied
into robustcd: models and losses are imported from a pinned checkout
(``activate``). The pseudo-label procedure of SCanNet lives in its training
script, which cannot be imported (it runs at import time), so it is
re-implemented in ``psd.py`` and checked against the upstream functions by
``tests/test_ding_adapter.py``.

Kept from upstream: model definitions, per-date input normalisation, the
``rand_rot90_flip_SCD`` augmentation distribution, the loss
(0.5 * (CE_A + CE_B) with class 0 ignored, weighted BCE on the change logit,
and ChangeSimilarity for the models that use it), SGD with Nesterov momentum,
poly learning-rate decay, and, for SCanNet/TED, the pseudo-label stage.

Class order: upstream ``ST_COLORMAP`` is exactly the robustcd SECOND colour
table (0 unchanged, 1 water, 2 ground, 3 low vegetation, 4 tree, 5 building,
6 playground), so no index conversion is needed. The test suite asserts it.
"""

from __future__ import annotations

import contextlib
import importlib
import os
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Dict, Optional, Tuple

import numpy as np

SCANNET_URL = "https://github.com/DingLei14/SCanNet.git"
SCANNET_COMMIT = "9c80d463bd0ca8cacca68b71c8b9adb7efd9f7a3"
BISRNET_URL = "https://github.com/DingLei14/Bi-SRNet.git"
BISRNET_COMMIT = "012f35aa9742e468b568a71107028fc3aa1e08b8"
REPOS = {"scannet": (SCANNET_URL, SCANNET_COMMIT), "bisrnet": (BISRNET_URL, BISRNET_COMMIT)}

#: upstream datasets/RS_ST.py (identical in both repositories), RGB, 0-255 scale
MEAN_A = np.array([113.40, 114.08, 116.45])
STD_A = np.array([48.30, 46.27, 48.14])
MEAN_B = np.array([111.07, 114.04, 118.18])
STD_B = np.array([49.41, 47.01, 47.94])
#: upstream RS_ST.ST_COLORMAP
ST_COLORMAP = ((255, 255, 255), (0, 0, 255), (128, 128, 128), (0, 128, 0), (0, 255, 0), (128, 0, 0), (255, 0, 0))
NUM_CLASSES = 7

#: torchvision ImageNet ResNet-34 weights that ``resnet34(pretrained=True)`` resolves to
RESNET34_FILE = "resnet34-b627a593.pth"


@dataclass(frozen=True)
class Recipe:
    """Upstream training recipe (train_SCD.py / train_SCD_psd.py)."""

    lr: float = 0.1
    momentum: float = 0.9
    weight_decay: float = 5e-4
    lr_power: float = 1.5
    sc_loss: bool = True          # ChangeSimilarity term
    psd: bool = False             # SCanNet pseudo-label stage
    psd_init_fscd: float = 0.6
    pseudo_thred: float = 0.6
    psd_tta: bool = True
    upstream_batch_size: int = 8
    upstream_epochs: int = 50


@dataclass(frozen=True)
class ModelSpec:
    repo: str
    module: str
    cls: str
    ctor: Tuple = (3,)
    ctor_kwargs: Dict = field(default_factory=dict)
    recipe: Recipe = Recipe()
    published: Optional[Dict[str, float]] = None


MODELS: Dict[str, ModelSpec] = {
    # SCanNet: train_SCD_psd.py (psd_train=True). Published numbers are the
    # released checkpoint's file name (SCanNet_32e_mIoU73.37_Sek23.94_Fscd63.66_OA87.86.pth).
    "scannet": ModelSpec("scannet", "models.SCanNet", "SCanNet", (3,), {"num_classes": 7, "input_size": 512},
                         Recipe(psd=True),
                         published={"mIoU": 0.7337, "SeK": 0.2394, "Fscd": 0.6366, "OA": 0.8786}),
    # TED: same script (commented-out import), same recipe, so that TED vs SCanNet isolates the transformer.
    "ted": ModelSpec("scannet", "models.TED", "TED", (3,), {"num_classes": 7}, Recipe(psd=True)),
    # Bi-SRNet: Bi-SRNet/train_SCD.py (SC loss, no pseudo labels).
    "bisrnet": ModelSpec("bisrnet", "models.BiSRNet", "BiSRNet", (3,), {"num_classes": 7}, Recipe()),
    # SSCD-l: the Bi-SRNet paper's baseline, trained without the SC loss.
    "sscdl": ModelSpec("bisrnet", "models.SSCDl", "SSCDl", (3,), {"num_classes": 7}, Recipe(sc_loss=False)),
    # HRSCD strategy 4 (Daudt et al. 2019): Bi-SRNet's loop without the SC loss, which is Bi-SRNet's contribution.
    "hrscd4": ModelSpec("bisrnet", "models.Daudt.HRSCD4", "HRSCD4", (3, 7), {}, Recipe(sc_loss=False)),
}

_ACTIVE: Optional[str] = None


def check_commit(repo_root: str | os.PathLike) -> Optional[str]:
    head = Path(repo_root) / ".git" / "HEAD"
    try:
        ref = head.read_text().strip()
        if ref.startswith("ref:"):
            return (Path(repo_root) / ".git" / ref.split()[1]).read_text().strip()
        return ref
    except OSError:
        return None


def activate(repo_root: str | os.PathLike) -> None:
    """Put a pinned upstream checkout first on ``sys.path``.

    Both repositories expose top-level ``models`` and ``utils`` packages, so a
    process may activate only one of them.
    """
    global _ACTIVE
    root = str(Path(repo_root).resolve())
    if _ACTIVE is not None and _ACTIVE != root:
        raise RuntimeError(f"upstream checkout {_ACTIVE} already active; cannot also activate {root}")
    for name in [m for m in sys.modules if m.split(".")[0] in ("models", "utils")]:
        mod_file = getattr(sys.modules[name], "__file__", None) or ""
        if not mod_file.startswith(root):
            del sys.modules[name]
    if root not in sys.path:
        sys.path.insert(0, root)
    _ACTIVE = root


def _resnet34_cache() -> Path:
    import torch

    return Path(torch.hub.get_dir()) / "checkpoints" / RESNET34_FILE


@contextlib.contextmanager
def _resnet34_patch(imagenet: bool):
    """Route upstream ``models.resnet34(pretrained)`` calls through the explicit weights API.

    Upstream calls ``resnet34(pretrained)`` positionally or by keyword; newer
    torchvision only accepts ``weights=``. ``imagenet=False`` builds without
    ImageNet weights (unit tests, or loading a full checkpoint afterwards).
    """
    import torchvision.models as tvm

    orig = tvm.resnet34

    def patched(*args, **kwargs):
        pretrained = args[0] if args else kwargs.pop("pretrained", False)
        kwargs.pop("weights", None)
        weights = tvm.ResNet34_Weights.IMAGENET1K_V1 if (pretrained and imagenet) else None
        return orig(weights=weights, **kwargs)

    tvm.resnet34 = patched
    try:
        yield
    finally:
        tvm.resnet34 = orig


def build_model(name: str, repo_root: str | os.PathLike, imagenet: bool = True):
    """Instantiate an upstream model exactly as its training script does.

    With ``imagenet=True`` the ResNet-34 encoder starts from torchvision's
    ImageNet weights, which must already be in the torch hub cache (compute
    nodes have no internet; ``cluster/models/ding/setup.sh`` fetches them).
    """
    spec = MODELS[name]
    activate(repo_root)
    if imagenet and not _resnet34_cache().exists():
        raise FileNotFoundError(f"ImageNet ResNet-34 weights not found at {_resnet34_cache()} "
                                f"(set TORCH_HOME or run cluster/models/ding/setup.sh)")
    cls = getattr(importlib.import_module(spec.module), spec.cls)
    with _resnet34_patch(imagenet):
        return cls(*spec.ctor, **spec.ctor_kwargs)


def load_weights(model, path: str | os.PathLike, strict: bool = True) -> dict:
    """Load a released upstream state dict or a robustcd checkpoint ({"model": ...})."""
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    sd = obj["model"] if isinstance(obj, dict) and "model" in obj and isinstance(obj["model"], dict) else obj
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    res = model.load_state_dict(sd, strict=strict)
    return {"loaded_keys": len(sd), "missing_keys": list(res.missing_keys),
            "unexpected_keys": list(res.unexpected_keys)}


def normalize_pair(im1: np.ndarray, im2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """HxWx3 uint8 pair -> 3xHxW float32 pair with upstream per-date statistics."""
    x1 = ((im1.astype(np.float64) - MEAN_A) / STD_A).astype(np.float32)
    x2 = ((im2.astype(np.float64) - MEAN_B) / STD_B).astype(np.float32)
    return np.ascontiguousarray(x1.transpose(2, 0, 1)), np.ascontiguousarray(x2.transpose(2, 0, 1))


def decode(outputs, mode: str = "restricted"):
    """(change_logit, sem_A, sem_B) -> uint8 (sem1, sem2, change) in robustcd order.

    The change map is ``sigmoid(logit) > 0.5`` as upstream. ``restricted``
    (protocol): semantic argmax over classes 1..6, since class 0 is ignored by
    the loss and never trained. ``full``: argmax over all 7 channels, as the
    upstream evaluator does; used only to reproduce published numbers.
    """
    import torch

    out_change, out_a, out_b = outputs
    change = (out_change[:, 0] > 0).long()
    if mode == "restricted":
        s1 = torch.argmax(out_a[:, 1:], dim=1) + 1
        s2 = torch.argmax(out_b[:, 1:], dim=1) + 1
    elif mode == "full":
        s1 = torch.argmax(out_a, dim=1)
        s2 = torch.argmax(out_b, dim=1)
    else:
        raise ValueError(f"unknown decode mode {mode!r}")
    cpu = lambda t: t.to(torch.uint8).cpu().numpy()  # noqa: E731
    return cpu(s1 * change), cpu(s2 * change), cpu(change)


def upstream_losses():
    """(CrossEntropyLoss2d, weighted_BCE_logits, ChangeSimilarity) from the active checkout.

    The two repositories differ in ChangeSimilarity's cosine margin (0.1 in
    SCanNet, 0.0 in Bi-SRNet); each model uses its own repository's version.
    """
    if _ACTIVE is None:
        raise RuntimeError("call activate() or build_model() first")
    loss = importlib.import_module("utils.loss")
    return loss.CrossEntropyLoss2d, loss.weighted_BCE_logits, loss.ChangeSimilarity
