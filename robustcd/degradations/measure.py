"""Dependency-free displacement measurement, to verify a render did what it says.

``estimate_shift`` returns the global translation ``(dy, dx)`` such that
``mov(x) ~= ref(x - d)`` -- i.e. the displacement applied to ``ref`` to obtain
``mov`` -- by normalised phase correlation with a parabolic sub-pixel fit.

A Hann window is applied by default: without it the border strip vacated by the
shift contributes a strong zero-frequency artefact and biases the peak.
"""

from __future__ import annotations

from typing import Tuple

import numpy as np


def _gray(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float64)
    if a.ndim == 3:
        a = a @ np.array([0.299, 0.587, 0.114])[: a.shape[2]] if a.shape[2] == 3 else a.mean(2)
    return a


def _hann2d(shape: Tuple[int, int]) -> np.ndarray:
    h, w = shape
    return np.outer(np.hanning(h), np.hanning(w))


def _parabolic(c: np.ndarray, peak: Tuple[int, int]) -> Tuple[float, float]:
    """Sub-pixel offset of the correlation peak by 1-D parabolic fit per axis."""
    h, w = c.shape
    py, px = peak
    out = []
    for axis, p, n in ((0, py, h), (1, px, w)):
        m = c[(p - 1) % n, px] if axis == 0 else c[py, (p - 1) % n]
        z = c[p, px] if axis == 0 else c[py, p]
        pl = c[(p + 1) % n, px] if axis == 0 else c[py, (p + 1) % n]
        denom = m - 2 * z + pl
        out.append(0.0 if denom == 0 else 0.5 * (m - pl) / denom)
    return out[0], out[1]


def estimate_shift_field(
    ref: np.ndarray, mov: np.ndarray, block: int = 96, stride: int = 64
) -> tuple[np.ndarray, np.ndarray]:
    """Block-wise displacement estimate.

    Returns ``(centres, disp)`` with ``centres`` of shape ``(n, 2)`` in ``(y, x)``
    and ``disp`` of shape ``(n, 2)``.  This is what validates the non-uniform
    modes (``affine``, ``local_warp``), where a single global shift is not a
    meaningful summary of the applied error.
    """
    h, w = ref.shape[:2]
    centres, disp = [], []
    for y in range(0, h - block + 1, stride):
        for x in range(0, w - block + 1, stride):
            a = ref[y : y + block, x : x + block]
            b = mov[y : y + block, x : x + block]
            centres.append((y + block / 2.0, x + block / 2.0))
            disp.append(estimate_shift(a, b))
    return np.asarray(centres), np.asarray(disp)


def estimate_shift(ref: np.ndarray, mov: np.ndarray, window: bool = True) -> Tuple[float, float]:
    a, b = _gray(ref), _gray(mov)
    if a.shape != b.shape:
        raise ValueError(f"shape mismatch {a.shape} vs {b.shape}")
    a = a - a.mean()
    b = b - b.mean()
    if window:
        wnd = _hann2d(a.shape)
        a, b = a * wnd, b * wnd
    fa, fb = np.fft.fft2(a), np.fft.fft2(b)
    cps = fa * np.conj(fb)
    mag = np.abs(cps)
    cps = np.divide(cps, mag, out=np.zeros_like(cps), where=mag > 1e-12)
    corr = np.real(np.fft.ifft2(cps))
    peak = np.unravel_index(int(np.argmax(corr)), corr.shape)
    sub_y, sub_x = _parabolic(corr, peak)  # type: ignore[arg-type]
    h, w = corr.shape
    py = peak[0] + sub_y
    px = peak[1] + sub_x
    if py > h / 2:
        py -= h
    if px > w / 2:
        px -= w
    return -py, -px
