"""Adapter that runs ChangeMamba (MambaSCD) under the robustcd protocol.

What is kept from upstream (ChenHongruixuan/ChangeMamba, commit ``CM_COMMIT``):
the model, its loss (``SCDTrainer.train_step``), AdamW + StepLR, the
geometric augmentation and the input normalisation.

What the adapter replaces, because the protocol requires it:
the training loop (seeds, bf16 autocast, gradient accumulation, resumable data
order), checkpoint selection (robustcd SeK on the robustcd val split instead
of the 37-class SeK on the test set), and prediction export (index PNGs in
the canonical SECOND class order, scored by ``scripts/evaluate.py``).

Class order
-----------
ChangeMamba's SECOND labels use a different index order from the SECOND
colour table used everywhere in robustcd:

=====  ==================  ===============
index  robustcd / SECOND   ChangeMamba
=====  ==================  ===============
1      water               low vegetation
2      ground              ground (nvg)
3      low vegetation      tree
4      tree                water
5      building            building
6      playground          playground
=====  ==================  ===============

``SECOND_TO_CM`` / ``CM_TO_SECOND`` convert between them; getting this wrong
permutes classes silently and corrupts SeK without any error.
"""

from __future__ import annotations

import os
import sys
from argparse import Namespace
from pathlib import Path
from typing import Iterable, Optional, Tuple

import numpy as np

CM_REPO_URL = "https://github.com/ChenHongruixuan/ChangeMamba.git"
CM_COMMIT = "9ce9cec13f9ea14bc0ad91f071577ec9b3a97983"

#: robustcd (SECOND colour-table) index -> ChangeMamba index
SECOND_TO_CM = np.array([0, 4, 2, 1, 3, 5, 6], dtype=np.uint8)
#: ChangeMamba index -> robustcd index
CM_TO_SECOND = np.argsort(SECOND_TO_CM).astype(np.uint8)

#: ChangeMamba colours in ChangeMamba index order (tasks/metadata.py SECOND_COLOR_MAP)
CM_COLORS = ((0, 0, 0), (0, 128, 0), (128, 128, 128), (0, 255, 0), (0, 0, 255), (128, 0, 0), (255, 0, 0))

#: ImageNet statistics used by ChangeMamba's imutils.normalize_img (RGB, 0-255 scale)
CM_MEAN = np.array([123.675, 116.28, 103.53], dtype=np.float32)
CM_STD = np.array([58.395, 57.12, 57.375], dtype=np.float32)


def add_to_path(cm_root: str | os.PathLike) -> None:
    """Make ``import changedetection`` resolve to the pinned ChangeMamba checkout."""
    root = str(Path(cm_root).resolve())
    if root not in sys.path:
        sys.path.insert(0, root)


def check_commit(cm_root: str | os.PathLike) -> Optional[str]:
    """Return the checkout's HEAD commit, or None if it cannot be read."""
    head = Path(cm_root) / ".git" / "HEAD"
    try:
        ref = head.read_text().strip()
        if ref.startswith("ref:"):
            return (Path(cm_root) / ".git" / ref.split()[1]).read_text().strip()
        return ref
    except OSError:
        return None


def normalize(img: np.ndarray) -> np.ndarray:
    """HxWx3 uint8 -> 3xHxW float32, exactly as ChangeMamba's loader."""
    x = (img.astype(np.float32) - CM_MEAN) / CM_STD
    return np.ascontiguousarray(x.transpose(2, 0, 1))


# --------------------------------------------------------------------------- #
# data preparation: SECOND (RGB labels) -> ChangeMamba layout
# --------------------------------------------------------------------------- #
def prepare_split(second_root: str | os.PathLike, out_dir: str | os.PathLike, ids: Iterable[str]) -> int:
    """Write the ChangeMamba folder layout for ``ids`` under ``out_dir``.

    ``T1``/``T2`` are symlinks to the original images (no copy); ``GT_T1``,
    ``GT_T2`` are single-channel labels in ChangeMamba order, ``GT_CD`` is 0/255.
    """
    from PIL import Image

    from robustcd.metrics.labels import rgb_to_index

    src, out = Path(second_root), Path(out_dir)
    for d in ("T1", "T2", "GT_T1", "GT_T2", "GT_CD"):
        (out / d).mkdir(parents=True, exist_ok=True)
    n = 0
    for sid in ids:
        name = f"{sid}.png"
        for s, d in (("im1", "T1"), ("im2", "T2")):
            link = out / d / name
            if not link.exists():
                os.symlink((src / s / name).resolve(), link)
        idx = []
        for s, d in (("label1", "GT_T1"), ("label2", "GT_T2")):
            i = rgb_to_index(np.asarray(Image.open(src / s / name).convert("RGB")))
            idx.append(i)
            Image.fromarray(SECOND_TO_CM[i]).save(out / d / name)
        Image.fromarray(((idx[0] != 0) * 255).astype(np.uint8)).save(out / "GT_CD" / name)
        n += 1
    return n


# --------------------------------------------------------------------------- #
# model
# --------------------------------------------------------------------------- #
def load_config(cfg_path: str):
    from changedetection.configs.config import get_config

    return get_config(Namespace(cfg=cfg_path, opts=None))


def build_model(cfg_path: str, encoder_pretrained: Optional[str] = None):
    """ChangeMambaSCD exactly as upstream SCDTrainer.build_model."""
    from changedetection.models.ChangeMambaSCD import ChangeMambaSCD
    from changedetection.script.script_utils import get_vssm_kwargs

    config = load_config(cfg_path)
    model = ChangeMambaSCD(output_cd=2, output_clf=7, pretrained=encoder_pretrained, **get_vssm_kwargs(config))
    return model, config


def load_weights(model, path: str, strict: bool = True) -> dict:
    """Load a full-model checkpoint (upstream release or robustcd run)."""
    from changedetection.checkpoints import load_model_weights

    info = load_model_weights(model, path)
    missing = list(info.get("missing_keys") or [])
    mismatched = list(info.get("mismatched_keys") or [])
    if strict and (missing or mismatched):
        raise RuntimeError(
            f"loading {path}: {len(missing)} missing keys (e.g. {missing[:3]}), "
            f"{len(mismatched)} shape-mismatched keys (e.g. {mismatched[:3]})"
        )
    return info


def decode(output_cd, out_t1, out_t2, mode: str = "restricted") -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Logits -> (sem1, sem2, change) as uint8 arrays in **SECOND** class order.

    ``restricted`` (protocol default): semantic argmax over classes 1..6 only.
    ChangeMamba never trains class 0 of its semantic heads (unchanged pixels
    are set to ignore), so that logit carries no meaning.
    ``full``: argmax over all 7 channels, as upstream's evaluator does. It is
    used only to reproduce upstream numbers; a changed pixel assigned class 0
    then counts as unchanged.
    """
    import torch

    change = torch.argmax(output_cd, dim=1)
    if mode == "restricted":
        s1 = torch.argmax(out_t1[:, 1:], dim=1) + 1
        s2 = torch.argmax(out_t2[:, 1:], dim=1) + 1
    elif mode == "full":
        s1 = torch.argmax(out_t1, dim=1)
        s2 = torch.argmax(out_t2, dim=1)
    else:
        raise ValueError(f"unknown decode mode {mode!r}")
    s1 = s1 * change
    s2 = s2 * change
    lut = torch.as_tensor(CM_TO_SECOND, device=s1.device, dtype=torch.long)
    s1, s2 = lut[s1], lut[s2]
    cpu = lambda t: t.to(torch.uint8).cpu().numpy()  # noqa: E731
    return cpu(s1), cpu(s2), cpu(change)
