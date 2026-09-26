"""Inference and evaluation helpers shared by adapters written after ChangeMamba.

An adapter provides ``normalize_pair(im1, im2) -> (x1, x2)`` (3xHxW float32
arrays) and ``decode(outputs, mode) -> (sem1, sem2, change)`` (uint8 HxW
arrays in robustcd class order, semantic maps already masked by the change
map). Everything else is shared: reading clean or rendered sets, batching,
fp32 inference without TTA, scoring with the single robustcd scorer, and
writing predictions in the layout read by ``scripts/evaluate.py``.
"""

from __future__ import annotations

import json
import time
from pathlib import Path
from typing import Callable, Dict, Iterator, List, Optional, Sequence

import numpy as np

from robustcd.metrics import SCDMeter, to_index


class RenderedReader:
    """``get(sid)`` view of a rendered degraded set (see RenderedSet)."""

    def __init__(self, set_dir: str):
        from robustcd.degradations.wrappers import RenderedSet

        self._rs = RenderedSet(set_dir, streams=["im1", "im2"])
        self._pos = {s: i for i, s in enumerate(self._rs.ids)}
        self.ids = list(self._rs.ids)

    def get(self, sid: str) -> Dict[str, np.ndarray]:
        return self._rs[self._pos[sid]]


def open_reader(source: Optional[str] = None, rendered_set: Optional[str] = None):
    from robustcd.datasets.second import SecondLike

    if (source is None) == (rendered_set is None):
        raise ValueError("give exactly one of source / rendered_set")
    return SecondLike(source) if source else RenderedReader(rendered_set)


def read_ids(path: Optional[str], default: Sequence[str]) -> List[str]:
    if not path:
        return list(default)
    return [line.strip() for line in Path(path).read_text().splitlines() if line.strip()]


def batches(ids: Sequence[str], n: int) -> Iterator[List[str]]:
    for i in range(0, len(ids), n):
        yield list(ids[i : i + n])


def predict(model, reader, ids: Sequence[str], device, normalize_pair: Callable, decode: Callable,
            batch_size: int = 8, decode_mode: str = "restricted"):
    """Yield (chunk_ids, samples, sem1, sem2, change); fp32, no TTA."""
    import torch

    model.eval()
    with torch.no_grad():
        for chunk in batches(ids, batch_size):
            samples = [reader.get(s) for s in chunk]
            pairs = [normalize_pair(s["im1"], s["im2"]) for s in samples]
            x1 = torch.from_numpy(np.stack([p[0] for p in pairs]).astype(np.float32)).to(device)
            x2 = torch.from_numpy(np.stack([p[1] for p in pairs]).astype(np.float32)).to(device)
            outputs = model(x1, x2)
            s1, s2, ch = decode(tuple(o.float() for o in outputs), decode_mode)
            yield chunk, samples, s1, s2, ch


def evaluate(model, reader, ids: Sequence[str], device, normalize_pair: Callable, decode: Callable,
             batch_size: int = 8, decode_mode: str = "restricted") -> Dict[str, float]:
    """robustcd SCD scores (SECOND definition + from-to secondary) on ``ids``."""
    meter = SCDMeter()
    for chunk, samples, s1, s2, _ in predict(model, reader, ids, device, normalize_pair, decode,
                                             batch_size, decode_mode):
        for sid, smp, p1, p2 in zip(chunk, samples, s1, s2):
            meter.update(p1, p2, to_index(smp["label1"]), to_index(smp["label2"]), sample_id=sid)
    out = meter.compute()
    out["n_images"] = len(meter)
    return out


def write_predictions(model, reader, ids: Sequence[str], out_dir: str | Path, device, normalize_pair: Callable,
                      decode: Callable, batch_size: int = 8, decode_mode: str = "restricted",
                      score: bool = False, meta: Optional[dict] = None) -> dict:
    """Write <out>/{im1,im2,change}/<id>.png and <out>/predict_info.json; optionally score."""
    from PIL import Image

    out = Path(out_dir)
    for d in ("im1", "im2", "change"):
        (out / d).mkdir(parents=True, exist_ok=True)
    meter = SCDMeter() if score else None
    t0, n = time.time(), 0
    for chunk, samples, s1, s2, ch in predict(model, reader, ids, device, normalize_pair, decode,
                                              batch_size, decode_mode):
        for sid, smp, p1, p2, c in zip(chunk, samples, s1, s2, ch):
            Image.fromarray(p1).save(out / "im1" / f"{sid}.png")
            Image.fromarray(p2).save(out / "im2" / f"{sid}.png")
            Image.fromarray((c * 255).astype(np.uint8)).save(out / "change" / f"{sid}.png")
            if meter is not None:
                meter.update(p1, p2, to_index(smp["label1"]), to_index(smp["label2"]), sample_id=sid)
            n += 1
    dt = time.time() - t0
    info = {"n_images": n, "seconds": round(dt, 1), "images_per_s": round(n / max(dt, 1e-9), 2),
            "decode": decode_mode, **(meta or {})}
    if meter is not None:
        full = meter.compute()
        info["scores"] = {k: full[k] for k in ("SeK", "mIoU", "Fscd", "kappa_n0", "SeK_fromto", "Fscd_fromto")
                          if k in full}
    (out / "predict_info.json").write_text(json.dumps(info, indent=2, default=str))
    return info
