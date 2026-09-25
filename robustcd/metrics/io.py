"""Persist per-image confusion matrices so conditions can be compared later.

A benchmark run evaluates each (model, dataset, degradation, severity) once and
writes ``confusion.npz``.  Every downstream number -- point scores, bootstrap
intervals, paired relative drops between clean and degraded -- is recomputed
from these files without re-reading any image.
"""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import numpy as np


def save_confusion(path: str | Path, sample_ids, per_sample: np.ndarray, task: str) -> None:
    np.savez_compressed(
        path, sample_ids=np.asarray(sample_ids, dtype=str), per_sample=per_sample.astype(np.int64),
        task=np.asarray(task),
    )


def load_confusion(path: str | Path) -> Tuple[np.ndarray, np.ndarray, str]:
    with np.load(path, allow_pickle=False) as z:
        return z["sample_ids"], z["per_sample"], str(z["task"])


def align(ids_a, ps_a, ids_b, ps_b):
    """Restrict two runs to their common images, in the same order (for pairing)."""
    common = sorted(set(ids_a.tolist()) & set(ids_b.tolist()))
    if not common:
        raise ValueError("no common sample ids")
    ia = {s: i for i, s in enumerate(ids_a.tolist())}
    ib = {s: i for i, s in enumerate(ids_b.tolist())}
    return common, ps_a[[ia[s] for s in common]], ps_b[[ib[s] for s in common]]
