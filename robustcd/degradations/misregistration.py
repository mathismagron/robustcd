"""Misregistration degradation family for bi-temporal change detection.

Physical model
--------------
A delivered bi-temporal product carries a *relative geometric error* between the
two dates: orthorectification residuals, DEM error, platform attitude error and
independent co-registration of each date leave date-2 displaced with respect to
date-1.  Four sub-modes cover the error shapes seen in real VHR products:

``shift_int``     integer global translation.  Applied by pure array indexing, so
                  it introduces geometric error with *zero* interpolation blur.
                  This is the control that separates "the model is sensitive to
                  displacement" from "the model is sensitive to resampling".
``shift_subpix``  fractional global translation, resampled (cubic by default).
                  The regime of a nominally well co-registered product
                  (sub-pixel CE90 specs), where blur and displacement co-occur.
``affine``        global rotation + scale residual about the image centre, i.e.
                  a displacement that grows linearly from the scene centre.
                  Parameterised by the *maximum corner displacement* so its
                  severity ladder is directly comparable to the shift modes.
``local_warp``    smooth non-rigid displacement field from a coarse random
                  control grid (terrain/parallax-driven local residuals).
                  Normalised so the peak displacement equals the nominal level.

Every mode's severity is expressed in the same unit -- pixels of characteristic
displacement -- so degradation curves from different modes share an x-axis.

Ground-truth convention (important, and configurable)
-----------------------------------------------------
Default ``apply_to="t2"``: date-1 is the geometric reference and is passed
through untouched; only date-2 is warped; **labels are not warped**.  The
degradation represents an *error in the product*, so the correct answer is
unchanged and the metric drop measures robustness.  Warping the labels along
with date-2 would instead measure a model's behaviour under
consistent-but-displaced evidence, which is a different question -- available as
``warp_labels=True`` for an ablation.

``apply_to="split"`` splits the error symmetrically (t1 by -d/2, t2 by +d/2),
matching products where neither date is the reference; note it resamples both
dates, so it is not blur-free even in ``shift_int`` mode.

Determinism
-----------
Transform parameters are drawn from a generator seeded by
``blake2b(dataset | sample_id | family | mode | severity | global_seed)``.
The same sample gets byte-identical degradation on any machine with the same
numpy Generator stream (PCG64 is version-stable), which is what makes an
offline render on a laptop and an evaluation on a cluster comparable.

No OpenCV / scikit-image dependency: numpy + scipy.ndimage only.
"""

from __future__ import annotations

import hashlib
from dataclasses import dataclass, field
from typing import Dict, Optional, Sequence, Tuple

import numpy as np
from scipy import ndimage

FAMILY = "misreg"

MODES: Tuple[str, ...] = ("shift_int", "shift_subpix", "affine", "local_warp")

#: Characteristic displacement in pixels for severity levels 1, 2, 3.
#: Edit here to re-ladder the benchmark; the value used is recorded per sample
#: in the manifest, so rendered sets are self-describing.
SEVERITY_PX: Dict[str, Tuple[float, float, float]] = {
    "shift_int": (4.0, 8.0, 16.0),
    "shift_subpix": (0.5, 1.0, 2.0),
    "affine": (4.0, 8.0, 16.0),
    "local_warp": (4.0, 8.0, 16.0),
}

#: Control-grid size (knots per axis) for the local warp field.
LOCAL_WARP_KNOTS = 6

_BORDER_TO_SCIPY = {"reflect": "reflect", "constant": "constant", "nearest": "nearest"}


# --------------------------------------------------------------------------- #
# spec
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class MisregSpec:
    """One point of the misregistration grid."""

    mode: str
    severity: int  # 1, 2 or 3
    apply_to: str = "t2"  # "t2" | "split"
    border: str = "reflect"  # "reflect" | "constant" | "nearest"
    interp_order: int = 3  # scipy spline order; 1=bilinear, 3=cubic
    warp_labels: bool = False
    global_seed: int = 20260101

    def __post_init__(self) -> None:
        if self.mode not in MODES:
            raise ValueError(f"mode must be one of {MODES}, got {self.mode!r}")
        if self.severity not in (1, 2, 3):
            raise ValueError(f"severity must be 1, 2 or 3, got {self.severity!r}")
        if self.apply_to not in ("t2", "split"):
            raise ValueError(f"apply_to must be 't2' or 'split', got {self.apply_to!r}")
        if self.border not in _BORDER_TO_SCIPY:
            raise ValueError(f"border must be one of {tuple(_BORDER_TO_SCIPY)}")
        if not 0 <= self.interp_order <= 5:
            raise ValueError("interp_order must be in [0, 5]")

    @property
    def displacement_px(self) -> float:
        return SEVERITY_PX[self.mode][self.severity - 1]

    @property
    def tag(self) -> str:
        """Filesystem-safe identifier, e.g. ``misreg-shift_int-s2``."""
        return f"{FAMILY}-{self.mode}-s{self.severity}"

    def to_dict(self) -> Dict[str, object]:
        return {
            "family": FAMILY,
            "mode": self.mode,
            "severity": self.severity,
            "displacement_px": self.displacement_px,
            "apply_to": self.apply_to,
            "border": self.border,
            "interp_order": self.interp_order,
            "warp_labels": self.warp_labels,
            "global_seed": self.global_seed,
        }


def grid(
    modes: Sequence[str] = MODES,
    severities: Sequence[int] = (1, 2, 3),
    **spec_kwargs,
) -> list[MisregSpec]:
    """The full (mode x severity) grid, as a flat list of specs."""
    return [MisregSpec(mode=m, severity=s, **spec_kwargs) for m in modes for s in severities]


# --------------------------------------------------------------------------- #
# seeding
# --------------------------------------------------------------------------- #
def make_rng(spec: MisregSpec, sample_id: str, dataset: str = "") -> np.random.Generator:
    """Deterministic per-(dataset, sample, mode, severity) generator."""
    key = "|".join(
        [dataset, str(sample_id), FAMILY, spec.mode, str(spec.severity), str(spec.global_seed)]
    )
    digest = hashlib.blake2b(key.encode("utf-8"), digest_size=8).digest()
    return np.random.default_rng(int.from_bytes(digest, "little"))


# --------------------------------------------------------------------------- #
# transform parameter sampling
# --------------------------------------------------------------------------- #
@dataclass
class Warp:
    """A sampled geometric error, in *output-to-input* (pull) convention.

    ``kind`` is ``"translate_int"``, ``"matrix"`` or ``"field"``.  ``params``
    records exactly what was applied, for the manifest and later error analysis.
    """

    kind: str
    params: Dict[str, object] = field(default_factory=dict)
    dyx: Optional[Tuple[int, int]] = None  # translate_int: (dy, dx)
    matrix: Optional[np.ndarray] = None  # matrix: 3x3 input->output, about centre
    field_yx: Optional[np.ndarray] = None  # field: (2, H, W) output displacement

    def scaled(self, factor: float, shape: Tuple[int, int]) -> "Warp":
        """Same error scaled by ``factor`` (used for the symmetric split)."""
        if self.kind == "translate_int":
            dy, dx = self.dyx  # type: ignore[misc]
            # a scaled integer shift is generally fractional -> becomes a matrix
            return Warp(
                kind="matrix",
                params={**self.params, "scaled_by": factor},
                matrix=_translation_matrix(dy * factor, dx * factor),
            )
        if self.kind == "matrix":
            m = _matrix_power(self.matrix, factor)  # type: ignore[arg-type]
            return Warp(kind="matrix", params={**self.params, "scaled_by": factor}, matrix=m)
        return Warp(
            kind="field",
            params={**self.params, "scaled_by": factor},
            field_yx=self.field_yx * factor,  # type: ignore[operator]
        )


def _matrix_power(m: np.ndarray, factor: float) -> np.ndarray:
    """``m`` composed with itself ``factor`` times (fractional / negative ok).

    Used for the symmetric split, where each date must carry exactly half the
    error so that composing the two halves reproduces the full transform.
    Falls back to the first-order approximation if the power is not real-valued.
    """
    from scipy.linalg import fractional_matrix_power

    try:
        out = np.asarray(fractional_matrix_power(m, factor))
        if np.iscomplexobj(out):
            if np.abs(out.imag).max() > 1e-8:
                raise ValueError("complex matrix power")
            out = out.real
        if not np.isfinite(out).all():
            raise ValueError("non-finite matrix power")
        return out
    except Exception:
        return np.eye(3) + factor * (m - np.eye(3))


def _translation_matrix(dy: float, dx: float) -> np.ndarray:
    m = np.eye(3)
    m[0, 2] = dy
    m[1, 2] = dx
    return m


def sample_warp(spec: MisregSpec, shape: Tuple[int, int], rng: np.random.Generator) -> Warp:
    """Draw the geometric error for one sample."""
    h, w = shape
    d = spec.displacement_px

    if spec.mode in ("shift_int", "shift_subpix"):
        theta = rng.uniform(0.0, 2.0 * np.pi)
        dy, dx = d * np.sin(theta), d * np.cos(theta)
        if spec.mode == "shift_int":
            dy_i, dx_i = int(round(dy)), int(round(dx))
            if dy_i == 0 and dx_i == 0:  # d < 0.5 px guard
                dx_i = int(np.sign(dx) or 1)
            return Warp(
                kind="translate_int",
                params={
                    "direction_rad": float(theta),
                    "nominal_px": d,
                    "realized_px": float(np.hypot(dy_i, dx_i)),
                },
                dyx=(dy_i, dx_i),
            )
        return Warp(
            kind="matrix",
            params={
                "direction_rad": float(theta),
                "nominal_px": d,
                "realized_px": float(np.hypot(dy, dx)),
                "dy": float(dy),
                "dx": float(dx),
            },
            matrix=_translation_matrix(dy, dx),
        )

    if spec.mode == "affine":
        # Split the displacement budget between a rotation and a scale residual.
        # For a transform about the centre, displacement grows with radius and
        # peaks at the corners, at radius R = half-diagonal.
        # The rotation displacement is tangential and the scale displacement is
        # radial, so they are orthogonal: |disp| = R * hypot(theta, s - 1).
        # Splitting the budget as (f, sqrt(1 - f^2)) therefore lands the corner
        # displacement exactly on the nominal level.
        r_max = 0.5 * float(np.hypot(h, w))
        frac_rot = float(rng.uniform(0.3, 0.7))
        sgn_rot = 1.0 if rng.random() < 0.5 else -1.0
        sgn_scale = 1.0 if rng.random() < 0.5 else -1.0
        theta = sgn_rot * frac_rot * d / r_max  # small-angle: arc = theta * R
        s = 1.0 + sgn_scale * float(np.sqrt(1.0 - frac_rot**2)) * d / r_max
        cos, sin = np.cos(theta), np.sin(theta)
        a = np.array([[s * cos, -s * sin], [s * sin, s * cos]])  # (y, x) rotation+scale
        cy, cx = (h - 1) / 2.0, (w - 1) / 2.0
        c = np.array([cy, cx])
        m = np.eye(3)
        m[:2, :2] = a
        m[:2, 2] = c - a @ c
        corners = np.array([[0, 0], [0, w - 1], [h - 1, 0], [h - 1, w - 1]], float)
        disp = (corners @ a.T + m[:2, 2]) - corners
        return Warp(
            kind="matrix",
            params={
                "nominal_px": d,
                "rotation_deg": float(np.degrees(theta)),
                "scale": float(s),
                "frac_rotation": frac_rot,
                "realized_px": float(np.linalg.norm(disp, axis=1).max()),
            },
            matrix=m,
        )

    # local_warp: coarse random control grid, smoothly upsampled.
    k = LOCAL_WARP_KNOTS
    coarse = rng.normal(size=(2, k, k))
    fine = np.stack(
        [ndimage.zoom(coarse[i], (h / k, w / k), order=3, mode="reflect") for i in range(2)]
    )
    mag = np.hypot(fine[0], fine[1])
    peak = float(mag.max())
    fine = fine * (d / peak) if peak > 0 else fine
    mag = mag * (d / peak) if peak > 0 else mag
    return Warp(
        kind="field",
        params={
            "nominal_px": d,
            "knots": k,
            "realized_peak_px": float(mag.max()),
            "realized_rms_px": float(np.sqrt((mag**2).mean())),
        },
        field_yx=fine.astype(np.float32),
    )


# --------------------------------------------------------------------------- #
# transform application
# --------------------------------------------------------------------------- #
def _pull_coords(warp: Warp, shape: Tuple[int, int]) -> np.ndarray:
    """Input coordinates to sample for each output pixel; shape (2, H, W)."""
    h, w = shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")
    if warp.kind == "field":
        return np.stack([yy - warp.field_yx[0], xx - warp.field_yx[1]])  # type: ignore[index]
    if warp.kind == "translate_int":
        dy, dx = warp.dyx  # type: ignore[misc]
        return np.stack([yy - dy, xx - dx])
    m = np.linalg.inv(warp.matrix)  # type: ignore[arg-type]
    src_y = m[0, 0] * yy + m[0, 1] * xx + m[0, 2]
    src_x = m[1, 0] * yy + m[1, 1] * xx + m[1, 2]
    return np.stack([src_y, src_x])


def displacement_field(warp: Warp, shape: Tuple[int, int]) -> np.ndarray:
    """Per-pixel applied displacement ``(2, H, W)`` in ``(dy, dx)``.

    Ground truth for validating a render: block-wise phase correlation on the
    rendered pair should reproduce this field.
    """
    h, w = shape
    yy, xx = np.meshgrid(np.arange(h, dtype=np.float64), np.arange(w, dtype=np.float64), indexing="ij")
    return np.stack([yy, xx]) - _pull_coords(warp, shape)


def valid_mask(warp: Warp, shape: Tuple[int, int]) -> np.ndarray:
    """True where the output pixel pulled from inside the source image.

    Misregistration necessarily vacates a strip at the border: those pixels have
    no true observation.  Keeping the mask lets evaluation either include the
    border (harsher, product-realistic) or exclude it (isolates the interior
    effect) without re-rendering.
    """
    h, w = shape
    c = _pull_coords(warp, shape)
    return (c[0] >= 0) & (c[0] <= h - 1) & (c[1] >= 0) & (c[1] <= w - 1)


def apply_warp(
    img: np.ndarray,
    warp: Warp,
    border: str = "reflect",
    interp_order: int = 3,
    nearest: bool = False,
) -> np.ndarray:
    """Apply ``warp`` to an HxW or HxWxC array, preserving dtype and range."""
    if img.ndim == 2:
        return apply_warp(img[..., None], warp, border, interp_order, nearest)[..., 0]
    h, w = img.shape[:2]
    order = 0 if nearest else interp_order
    mode = _BORDER_TO_SCIPY[border]

    if warp.kind == "translate_int":
        dy, dx = warp.dyx  # type: ignore[misc]
        pad_y, pad_x = abs(dy), abs(dx)
        np_mode = {"reflect": "symmetric", "constant": "constant", "nearest": "edge"}[border]
        padded = np.pad(img, ((pad_y, pad_y), (pad_x, pad_x), (0, 0)), mode=np_mode)
        y0, x0 = pad_y - dy, pad_x - dx
        return padded[y0 : y0 + h, x0 : x0 + w]

    coords = _pull_coords(warp, (h, w))
    out = np.empty_like(img)
    info = np.iinfo(img.dtype) if np.issubdtype(img.dtype, np.integer) else None
    for ch in range(img.shape[2]):
        res = ndimage.map_coordinates(
            img[..., ch].astype(np.float64), coords, order=order, mode=mode, cval=0.0, prefilter=order > 1
        )
        if info is not None:
            res = np.clip(np.rint(res), info.min, info.max)
        out[..., ch] = res.astype(img.dtype)
    return out


# --------------------------------------------------------------------------- #
# sample-level entry point
# --------------------------------------------------------------------------- #
def degrade_pair(
    im1: np.ndarray,
    im2: np.ndarray,
    spec: MisregSpec,
    sample_id: str,
    dataset: str = "",
    label1: Optional[np.ndarray] = None,
    label2: Optional[np.ndarray] = None,
) -> Dict[str, object]:
    """Apply the misregistration family to one bi-temporal sample.

    Returns a dict with ``im1``, ``im2``, ``valid_mask`` (bool HxW), optionally
    ``label1`` / ``label2``, and ``record`` -- the JSON-serialisable description
    of exactly what was applied.
    """
    if im1.shape[:2] != im2.shape[:2]:
        raise ValueError(f"date shapes differ: {im1.shape[:2]} vs {im2.shape[:2]}")
    shape = im1.shape[:2]
    rng = make_rng(spec, sample_id, dataset)
    warp = sample_warp(spec, shape, rng)

    kw = dict(border=spec.border, interp_order=spec.interp_order)
    if spec.apply_to == "t2":
        w1, w2 = None, warp
    else:
        w1, w2 = warp.scaled(-0.5, shape), warp.scaled(+0.5, shape)

    out1 = im1 if w1 is None else apply_warp(im1, w1, **kw)
    out2 = apply_warp(im2, w2, **kw)

    vm = valid_mask(w2, shape)
    if w1 is not None:
        vm &= valid_mask(w1, shape)

    # warp_t1 / warp_t2 are returned for validation and error analysis; they are
    # Warp objects, not arrays, and are ignored by the renderer's PNG writer.
    result: Dict[str, object] = {
        "im1": out1, "im2": out2, "valid_mask": vm, "warp_t1": w1, "warp_t2": w2,
    }
    for name, lab, wl in (("label1", label1, w1), ("label2", label2, w2)):
        if lab is None:
            continue
        result[name] = apply_warp(lab, wl, nearest=True, **kw) if (spec.warp_labels and wl is not None) else lab

    result["record"] = {
        "sample_id": sample_id,
        "dataset": dataset,
        **spec.to_dict(),
        "warp_kind": warp.kind,
        "warp_params": {k: v for k, v in warp.params.items()},
        "matrix": warp.matrix.tolist() if warp.matrix is not None else None,
        "dyx": list(warp.dyx) if warp.dyx is not None else None,
        "valid_fraction": float(vm.mean()),
    }
    return result
