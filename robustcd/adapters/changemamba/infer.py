"""Batched inference and robustcd evaluation for ChangeMamba models.

Inputs are read with robustcd readers (original SECOND RGB images and labels,
or a rendered degraded set), so evaluation never depends on the converted
ChangeMamba label files. Inference runs in fp32 with no test-time augmentation.
"""

from __future__ import annotations

from pathlib import Path
from typing import Dict, Iterator, List, Optional, Sequence

import numpy as np

from robustcd.metrics import SCDMeter, to_index

from . import decode, normalize


class RenderedReader:
    """``get(sid)`` view of a rendered degraded set (see RenderedSet)."""

    def __init__(self, set_dir: str):
        from robustcd.degradations.wrappers import RenderedSet

        self._rs = RenderedSet(set_dir, streams=["im1", "im2"])
        self._pos = {s: i for i, s in enumerate(self._rs.ids)}
        self.ids = list(self._rs.ids)

    def get(self, sid: str) -> Dict[str, np.ndarray]:
        return self._rs[self._pos[sid]]


def batches(ids: Sequence[str], n: int) -> Iterator[List[str]]:
    for i in range(0, len(ids), n):
        yield list(ids[i : i + n])


def predict(model, reader, ids: Sequence[str], device, batch_size: int = 8, decode_mode: str = "restricted"):
    """Yield (chunk_ids, samples, sem1, sem2, change) with sem* in SECOND order."""
    import torch

    model.eval()
    with torch.no_grad():
        for chunk in batches(ids, batch_size):
            samples = [reader.get(s) for s in chunk]
            x1 = torch.from_numpy(np.stack([normalize(s["im1"]) for s in samples])).to(device)
            x2 = torch.from_numpy(np.stack([normalize(s["im2"]) for s in samples])).to(device)
            out_cd, out_t1, out_t2 = model(x1, x2)
            s1, s2, ch = decode(out_cd.float(), out_t1.float(), out_t2.float(), decode_mode)
            yield chunk, samples, s1, s2, ch


def evaluate(model, reader, ids: Sequence[str], device, batch_size: int = 8,
             decode_mode: str = "restricted") -> Dict[str, float]:
    """robustcd SCD scores (SECOND definition + from-to secondary) on ``ids``."""
    meter = SCDMeter()
    for chunk, samples, s1, s2, _ in predict(model, reader, ids, device, batch_size, decode_mode):
        for sid, smp, p1, p2 in zip(chunk, samples, s1, s2):
            meter.update(p1, p2, to_index(smp["label1"]), to_index(smp["label2"]), sample_id=sid)
    out = meter.compute()
    out["n_images"] = len(meter)
    return out


def read_ids(path: Optional[str], default: Sequence[str]) -> List[str]:
    if not path:
        return list(default)
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]
