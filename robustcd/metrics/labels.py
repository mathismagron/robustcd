"""Label encodings for semantic change detection datasets.

SECOND ships its semantic maps as RGB PNGs, one per date, where white marks
"no change" and the six land-cover classes are coded by colour.  Decoding is
**strict by default**: a pixel whose colour is not in the table raises instead
of being silently mapped to a class, because a wrong or anti-aliased colour
table corrupts every downstream score without any visible error.
"""

from __future__ import annotations

from typing import Dict, Sequence, Tuple

import numpy as np

#: SECOND colour table, index = class id.  Index 0 is "no change".
SECOND_COLORMAP: Tuple[Tuple[int, int, int], ...] = (
    (255, 255, 255),  # 0 no change
    (0, 0, 255),      # 1 water
    (128, 128, 128),  # 2 non-vegetated ground surface
    (0, 128, 0),      # 3 low vegetation
    (0, 255, 0),      # 4 tree
    (128, 0, 0),      # 5 building
    (255, 0, 0),      # 6 playground / sports field
)
SECOND_CLASSES: Tuple[str, ...] = (
    "no change", "water", "ground", "low vegetation", "tree", "building", "playground",
)

COLORMAPS: Dict[str, Tuple[Tuple[int, int, int], ...]] = {"SECOND": SECOND_COLORMAP}


def rgb_to_index(
    rgb: np.ndarray, colormap: Sequence[Tuple[int, int, int]] = SECOND_COLORMAP, strict: bool = True
) -> np.ndarray:
    """Map an HxWx3 uint8 colour image to an HxW uint8 class-index map."""
    if rgb.ndim != 3 or rgb.shape[2] < 3:
        raise ValueError(f"expected HxWx3 colour image, got shape {rgb.shape}")
    rgb = rgb[..., :3].astype(np.uint32)
    key = (rgb[..., 0] << 16) | (rgb[..., 1] << 8) | rgb[..., 2]
    out = np.full(key.shape, 255, dtype=np.uint8)
    for idx, (r, g, b) in enumerate(colormap):
        out[key == ((r << 16) | (g << 8) | b)] = idx
    if strict and (out == 255).any():
        bad = np.unique(rgb[out == 255].reshape(-1, 3), axis=0)[:5]
        raise ValueError(
            f"{int((out == 255).sum())} pixels have colours outside the colour table, e.g. {bad.tolist()}"
        )
    return out


def index_to_rgb(idx: np.ndarray, colormap: Sequence[Tuple[int, int, int]] = SECOND_COLORMAP) -> np.ndarray:
    """Inverse of :func:`rgb_to_index`."""
    lut = np.asarray(colormap, dtype=np.uint8)
    if idx.max(initial=0) >= len(lut):
        raise ValueError(f"class index {int(idx.max())} outside colour table of size {len(lut)}")
    return lut[idx]


def to_index(arr: np.ndarray, colormap: Sequence[Tuple[int, int, int]] = SECOND_COLORMAP) -> np.ndarray:
    """Accept either an index map (HxW) or a colour map (HxWx3) and return indices."""
    if arr.ndim == 2:
        return arr.astype(np.uint8, copy=False)
    if arr.ndim == 3 and arr.shape[2] == 1:
        return arr[..., 0].astype(np.uint8, copy=False)
    return rgb_to_index(arr, colormap)
