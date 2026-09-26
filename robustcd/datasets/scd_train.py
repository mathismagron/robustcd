"""Training-time dataset for semantic change detection adapters.

Reads a SECOND-style split with the robustcd reader (RGB colour labels are
converted to robustcd class indices, 0 = no change), applies the protocol's
geometric-only augmentation, and normalises with the model's own function.

Augmentation follows the protocol: rotations by 90 degrees and flips, applied
identically to both dates and both labels, nothing photometric. Two variants
are provided because upstream codebases differ in the exact distribution:

``ding``  (Bi-SRNet / SCanNet ``rand_rot90_flip_SCD``): rot90 with p = 0.5,
          then one of {none, vertical, horizontal, both} with p = 0.25 each.
``none``  no augmentation (debugging, or models that augment on the GPU).

Randomness comes from Python's ``random`` module, which PyTorch reseeds in each
DataLoader worker from the loader generator; the loop seeds that generator from
(seed, start iteration), so runs are reproducible up to cuDNN non-determinism.
"""

from __future__ import annotations

import random
from pathlib import Path
from typing import Callable, Sequence, Tuple

import numpy as np

from robustcd.metrics import to_index

from .second import SecondLike

NormalizePair = Callable[[np.ndarray, np.ndarray], Tuple[np.ndarray, np.ndarray]]


def rot90_flip_ding(im1, im2, l1, l2, rng: random.Random | None = None):
    """Exact distribution of upstream ``rand_rot90_flip_SCD`` (see module docstring)."""
    r = (rng or random).random()
    if r >= 0.5:
        im1, im2, l1, l2 = (np.rot90(a) for a in (im1, im2, l1, l2))
    r = (rng or random).random()
    if r < 0.25:
        pass
    elif r < 0.5:
        im1, im2, l1, l2 = (np.flip(a, axis=0) for a in (im1, im2, l1, l2))
    elif r < 0.75:
        im1, im2, l1, l2 = (np.flip(a, axis=1) for a in (im1, im2, l1, l2))
    else:
        im1, im2, l1, l2 = (a[::-1, ::-1] for a in (im1, im2, l1, l2))
    return tuple(np.ascontiguousarray(a) for a in (im1, im2, l1, l2))


AUGMENTATIONS = {"ding": rot90_flip_ding, "none": None}


class SCDTrainSet:
    """``torch.utils.data.Dataset``-compatible: returns (x1, x2, label1, label2).

    ``x1``/``x2`` are float32 3xHxW tensors produced by ``normalize_pair``;
    labels are int64 HxW tensors in robustcd class order (0 = no change).
    """

    def __init__(self, root: str | Path, ids: Sequence[str], normalize_pair: NormalizePair,
                 augmentation: str = "ding"):
        if augmentation not in AUGMENTATIONS:
            raise ValueError(f"unknown augmentation {augmentation!r}; choose from {sorted(AUGMENTATIONS)}")
        self.reader = SecondLike(root)
        missing = [i for i in ids if i not in self.reader.filenames]
        if missing:
            raise KeyError(f"{len(missing)} ids not found under {root} (e.g. {missing[:3]})")
        if not {"label1", "label2"} <= set(self.reader.available_streams):
            raise FileNotFoundError(f"{root} has no label1/label2 streams")
        self.ids = list(ids)
        self.normalize_pair = normalize_pair
        self.augment = AUGMENTATIONS[augmentation]

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int):
        import torch

        s = self.reader.get(self.ids[i])
        im1, im2 = s["im1"], s["im2"]
        l1, l2 = to_index(s["label1"]), to_index(s["label2"])
        if self.augment is not None:
            im1, im2, l1, l2 = self.augment(im1, im2, l1, l2)
        x1, x2 = self.normalize_pair(im1, im2)
        return (torch.from_numpy(np.ascontiguousarray(x1, dtype=np.float32)),
                torch.from_numpy(np.ascontiguousarray(x2, dtype=np.float32)),
                torch.from_numpy(l1.astype(np.int64)), torch.from_numpy(l2.astype(np.int64)))
