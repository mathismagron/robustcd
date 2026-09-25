"""Semantic and binary change detection metrics, from one confusion matrix.

Why a separate module
---------------------
Published SeK for the same architecture differs across papers.  Several
implementation choices that change the number are rarely stated:

1. **Aggregation.**  SeK, mIoU and Fscd are defined on a confusion matrix
   accumulated over the *whole test set*.  Averaging per-image scores gives a
   different number -- per-image kappa is ill-conditioned on tiles with little
   or no change.  The widely copied Bi-SRNet utilities ship both variants side
   by side (``SCDD_eval_all`` and ``SCDD_eval``).  This module only exposes
   dataset-level accumulation; per-image matrices are kept solely to resample
   images for confidence intervals.
2. **Consistency of the prediction.**  A semantic prediction must be 0 ("no
   change") wherever the predicted change mask is 0.  Scoring raw semantic
   heads without applying the change mask scores a different quantity.
   :func:`compose_prediction` does it explicitly.
3. **Both dates.**  SECOND-style SCD scores the date-1 and date-2 maps in *one*
   confusion matrix (2N maps), not two averaged scores.
4. **Degenerate denominators.**  Handled explicitly and documented per metric
   (see :func:`scd_scores`); the reference code returns NaN from
   ``hmean([nan, 0])`` where this module returns 0.

Definitions (SECOND; Yang et al., 2021)
----------------------------------------
With H the K x K confusion matrix over both dates (class 0 = no change):

* binary change matrix: TN = H[0,0], FP = sum_j H[0,j>0], FN = sum_i H[i>0,0],
  TP = sum_{i,j>0} H[i,j]  (a changed pixel counts as detected even if its class
  is wrong -- class errors are penalised by the kappa term)
* ``mIoU``  = (IoU_nochange + IoU_change) / 2
* ``kappa_n0`` = Cohen's kappa of H with H[0,0] set to 0
* ``SeK``   = kappa_n0 * exp(IoU_change - 1)
* ``Fscd``  = harmonic mean of Pscd = sum_{i>0} H[i,i] / (#pred changed) and
  Rscd = sum_{i>0} H[i,i] / (#gt changed)

All of these are invariant to transposing H, so the row/column convention is
immaterial; here rows are ground truth and columns are prediction.
"""

from __future__ import annotations

import math
from typing import Dict, List, Optional

import numpy as np

# --------------------------------------------------------------------------- #
# confusion accumulation
# --------------------------------------------------------------------------- #


def confusion(
    pred: np.ndarray,
    gt: np.ndarray,
    num_classes: int,
    valid: Optional[np.ndarray] = None,
    ignore_index: Optional[int] = 255,
) -> np.ndarray:
    """K x K int64 confusion matrix, rows = ground truth, cols = prediction.

    Pixels are excluded when ``valid`` is False or ``gt == ignore_index``.
    Out-of-range predictions raise: silently dropping them hides a decoding bug.
    """
    if pred.shape != gt.shape:
        raise ValueError(f"shape mismatch: pred {pred.shape} vs gt {gt.shape}")
    g = gt.astype(np.int64, copy=False).ravel()
    p = pred.astype(np.int64, copy=False).ravel()
    keep = np.ones_like(g, dtype=bool)
    if ignore_index is not None:
        keep &= g != ignore_index
    if valid is not None:
        if valid.shape != gt.shape:
            raise ValueError(f"valid mask shape {valid.shape} != gt {gt.shape}")
        keep &= valid.ravel().astype(bool)
    g, p = g[keep], p[keep]
    if g.size and (g.min() < 0 or g.max() >= num_classes):
        raise ValueError(f"ground truth values outside [0, {num_classes})")
    if p.size and (p.min() < 0 or p.max() >= num_classes):
        bad = np.unique(p[(p < 0) | (p >= num_classes)])[:5]
        raise ValueError(f"prediction values outside [0, {num_classes}): {bad}")
    return np.bincount(num_classes * g + p, minlength=num_classes**2).reshape(num_classes, num_classes)


def compose_prediction(sem: np.ndarray, change: np.ndarray, no_change_index: int = 0) -> np.ndarray:
    """Force a semantic map to agree with a binary change map.

    Returns ``sem`` where ``change`` is non-zero and ``no_change_index``
    elsewhere.  Pixels predicted as changed but assigned the no-change class by
    the semantic head are left as-is and therefore count as unchanged.
    """
    if sem.shape != change.shape:
        raise ValueError(f"shape mismatch: sem {sem.shape} vs change {change.shape}")
    out = sem.copy()
    out[~change.astype(bool)] = no_change_index
    return out


def fromto_map(sem1: np.ndarray, sem2: np.ndarray, n_semantic: int = 6) -> np.ndarray:
    """Encode a date pair as a single "from -> to" transition map.

    Changed pixels get ``(c1 - 1) * n_semantic + c2`` (1 .. n_semantic**2); a
    pixel is 0 if either date is 0.  This is the 37-class space in which some
    SCD codebases (e.g. ChangeMamba) compute SeK on SECOND.  It is reported here
    only as a *secondary* score, for comparison with such papers: it is not the
    SECOND definition, and it penalises a semantic error on either date as a
    full transition error.
    """
    a = sem1.astype(np.int64)
    b = sem2.astype(np.int64)
    out = (a - 1) * n_semantic + b
    out[(a == 0) | (b == 0)] = 0
    return out


# --------------------------------------------------------------------------- #
# scoring
# --------------------------------------------------------------------------- #


def _div(a: float, b: float) -> float:
    return float(a) / float(b) if b != 0 else float("nan")


def cohen_kappa(h: np.ndarray) -> float:
    """Cohen's kappa of a confusion matrix; 0 when undefined (as in the reference)."""
    h = h.astype(np.float64)
    n = h.sum()
    if n == 0:
        return 0.0
    po = np.trace(h) / n
    pe = float(h.sum(1) @ h.sum(0)) / n**2
    return 0.0 if pe == 1 else float((po - pe) / (1 - pe))


def binary_from_semantic(h: np.ndarray) -> np.ndarray:
    """Collapse a K x K SCD matrix to the 2 x 2 matrix [[TN, FP], [FN, TP]]."""
    b = np.zeros((2, 2), dtype=np.int64)
    b[0, 0] = h[0, 0]
    b[0, 1] = h[0, 1:].sum()
    b[1, 0] = h[1:, 0].sum()
    b[1, 1] = h[1:, 1:].sum()
    return b


def binary_scores(b: np.ndarray) -> Dict[str, float]:
    """Scores of a 2 x 2 matrix [[TN, FP], [FN, TP]] (rows = gt).

    Precision / recall / IoU are NaN when their denominator is 0.  F1 is 0 when
    TP = 0 and at least one of (predicted change, true change) is non-empty, and
    NaN only when both are empty.
    """
    tn, fp, fn, tp = (float(x) for x in np.asarray(b).ravel())
    f1 = float("nan") if tp + fp + fn == 0 else 2 * tp / (2 * tp + fp + fn)
    return {
        "precision": _div(tp, tp + fp),
        "recall": _div(tp, tp + fn),
        "F1": f1,
        "IoU_change": _div(tp, tp + fp + fn),
        "IoU_nochange": _div(tn, tn + fp + fn),
        "OA": _div(tp + tn, tn + fp + fn + tp),
        "kappa": cohen_kappa(np.asarray(b)),
    }


def scd_scores(h: np.ndarray) -> Dict[str, float]:
    """All SCD scores from one K x K matrix (rows = gt, cols = pred, class 0 = no change)."""
    h = np.asarray(h, dtype=np.int64)
    bs = binary_scores(binary_from_semantic(h))
    iou_c, iou_n = bs["IoU_change"], bs["IoU_nochange"]

    h_n0 = h.copy()
    h_n0[0, 0] = 0
    kappa_n0 = cohen_kappa(h_n0)
    sek = kappa_n0 * math.exp(iou_c - 1.0) if not math.isnan(iou_c) else float("nan")

    sc_tp = float(np.trace(h[1:, 1:]))
    pred_changed = float(h[:, 1:].sum())
    gt_changed = float(h[1:, :].sum())
    if pred_changed == 0 and gt_changed == 0:
        f_scd = float("nan")
    else:
        # harmonic mean of Pscd and Rscd written in counts; 0 when TP = 0
        f_scd = 2 * sc_tp / (pred_changed + gt_changed)

    return {
        "SeK": sek,
        "kappa_n0": kappa_n0,
        "mIoU": (iou_n + iou_c) / 2,
        "IoU_change": iou_c,
        "IoU_nochange": iou_n,
        "Fscd": f_scd,
        "Pscd": _div(sc_tp, pred_changed),
        "Rscd": _div(sc_tp, gt_changed),
        "OA": _div(np.trace(h), h.sum()),
        "change_F1": bs["F1"],
        "change_ratio_gt": _div(gt_changed, h.sum()),
        "n_pixels": float(h.sum()),
    }


# --------------------------------------------------------------------------- #
# streaming meters
# --------------------------------------------------------------------------- #


class _Meter:
    num_classes: int

    def __init__(self) -> None:
        k = self.num_classes
        self.total = np.zeros((k, k), dtype=np.int64)
        self._per_sample: List[np.ndarray] = []
        self.sample_ids: List[str] = []

    def _add(self, h: np.ndarray, sample_id: Optional[str]) -> None:
        self.total += h
        self._per_sample.append(h)
        self.sample_ids.append(str(sample_id) if sample_id is not None else str(len(self.sample_ids)))

    @property
    def per_sample(self) -> np.ndarray:
        """(N, K, K) per-image matrices in update order (for the bootstrap)."""
        k = self.num_classes
        return np.stack(self._per_sample) if self._per_sample else np.zeros((0, k, k), np.int64)

    def __len__(self) -> int:
        return len(self._per_sample)


class SCDMeter(_Meter):
    """Accumulate SCD confusion over a test set.

    ``update`` takes *consistent* date-1 / date-2 maps (see
    :func:`compose_prediction`) and adds both to one matrix.  ``valid`` (HxW
    bool, e.g. the misregistration valid mask) removes pixels from both dates.
    """

    score_fn = staticmethod(scd_scores)

    def __init__(self, num_classes: int = 7, ignore_index: Optional[int] = 255, track_fromto: bool = True):
        self.num_classes = num_classes
        self.ignore_index = ignore_index
        self.track_fromto = track_fromto
        n_ft = (num_classes - 1) ** 2 + 1
        self.total_fromto = np.zeros((n_ft, n_ft), dtype=np.int64)
        super().__init__()

    def update(self, pred1, pred2, gt1, gt2, valid=None, sample_id=None) -> np.ndarray:
        kw = dict(num_classes=self.num_classes, valid=valid, ignore_index=self.ignore_index)
        h = confusion(pred1, gt1, **kw) + confusion(pred2, gt2, **kw)
        self._add(h, sample_id)
        if self.track_fromto:
            n_sem = self.num_classes - 1
            g = fromto_map(gt1, gt2, n_sem)
            if self.ignore_index is not None:
                g[(gt1 == self.ignore_index) | (gt2 == self.ignore_index)] = self.ignore_index
            self.total_fromto += confusion(
                fromto_map(pred1, pred2, n_sem), g, n_sem**2 + 1, valid=valid, ignore_index=self.ignore_index
            )
        return h

    def compute(self) -> Dict[str, float]:
        """Primary scores (SECOND definition) plus secondary ``*_fromto`` scores."""
        out = scd_scores(self.total)
        if self.track_fromto:
            ft = scd_scores(self.total_fromto)
            out.update({f"{k}_fromto": ft[k] for k in ("SeK", "mIoU", "Fscd", "kappa_n0")})
        return out


class BCDMeter(_Meter):
    """Accumulate binary change confusion (LEVIR-CD style). Any value > 0 is change."""

    num_classes = 2
    score_fn = staticmethod(binary_scores)

    def __init__(self, ignore_index: Optional[int] = None):
        self.ignore_index = ignore_index
        super().__init__()

    def update(self, pred, gt, valid=None, sample_id=None) -> np.ndarray:
        p = (np.asarray(pred) > 0).astype(np.uint8)
        g = np.asarray(gt)
        if self.ignore_index is not None:
            g = np.where(g == self.ignore_index, self.ignore_index, (g > 0).astype(np.uint8))
        else:
            g = (g > 0).astype(np.uint8)
        h = confusion(p, g, 2, valid=valid, ignore_index=self.ignore_index)
        self._add(h, sample_id)
        return h

    def compute(self) -> Dict[str, float]:
        return binary_scores(self.total)
