"""Image-level bootstrap for dataset-level metrics.

SeK, mIoU and Fscd are not means of per-image quantities, so their uncertainty
cannot be read off a per-image standard deviation.  Instead, *images* are
resampled with replacement, their confusion matrices re-summed and the metric
recomputed.  Two uses in the benchmark:

* :func:`bootstrap_ci` -- percentile interval for one model under one condition.
* :func:`paired_bootstrap` -- interval of a *difference* (clean vs degraded, or
  model A vs model B) with the same resampled images on both sides.  Relative
  degradation curves should carry this one: the pairing cancels the
  between-image variance both conditions share.
"""

from __future__ import annotations

from typing import Callable, Dict, Optional, Sequence

import numpy as np


def _resampled_totals(per_sample: np.ndarray, idx: np.ndarray) -> np.ndarray:
    """Sum of per-image matrices for each bootstrap replicate; idx is (B, N)."""
    b, n = idx.shape
    weights = np.zeros((b, per_sample.shape[0]), dtype=np.int64)
    np.add.at(weights, (np.repeat(np.arange(b), n), idx.ravel()), 1)
    return np.einsum("bn,nij->bij", weights, per_sample, optimize=True)


def bootstrap_ci(
    per_sample: np.ndarray,
    score_fn: Callable[[np.ndarray], Dict[str, float]],
    keys: Sequence[str] = ("SeK", "mIoU", "Fscd"),
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, Dict[str, float]]:
    """Percentile bootstrap CI for each metric in ``keys``."""
    n = per_sample.shape[0]
    if n < 2:
        raise ValueError("need at least 2 samples to bootstrap")
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    scores = [score_fn(t) for t in _resampled_totals(per_sample, idx)]
    point = score_fn(per_sample.sum(0))
    out = {}
    for k in keys:
        vals = np.array([s[k] for s in scores])
        lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
        out[k] = {"value": point[k], "lo": float(lo), "hi": float(hi), "se": float(np.nanstd(vals, ddof=1))}
    return out


def paired_bootstrap(
    per_sample_a: np.ndarray,
    per_sample_b: np.ndarray,
    score_fn: Callable[[np.ndarray], Dict[str, float]],
    key: str = "SeK",
    stat: str = "relative_drop",
    n_boot: int = 1000,
    alpha: float = 0.05,
    seed: int = 0,
) -> Dict[str, object]:
    """CI of a paired statistic between condition A (e.g. clean) and B (degraded).

    ``stat`` is ``"difference"`` (B - A) or ``"relative_drop"`` ((A - B) / A).
    Inputs must be aligned image-for-image (same order, same ids).
    """
    if per_sample_a.shape != per_sample_b.shape:
        raise ValueError(f"unaligned inputs: {per_sample_a.shape} vs {per_sample_b.shape}")
    fns = {
        "difference": lambda a, b: b - a,
        "relative_drop": lambda a, b: (a - b) / a if a != 0 else float("nan"),
    }
    f = fns[stat]
    n = per_sample_a.shape[0]
    idx = np.random.default_rng(seed).integers(0, n, size=(n_boot, n))
    ta, tb = _resampled_totals(per_sample_a, idx), _resampled_totals(per_sample_b, idx)
    vals = np.array([f(score_fn(x)[key], score_fn(y)[key]) for x, y in zip(ta, tb)])
    point = f(score_fn(per_sample_a.sum(0))[key], score_fn(per_sample_b.sum(0))[key])
    lo, hi = np.nanpercentile(vals, [100 * alpha / 2, 100 * (1 - alpha / 2)])
    return {"value": float(point), "lo": float(lo), "hi": float(hi),
            "se": float(np.nanstd(vals, ddof=1)), "stat": stat, "key": key}
