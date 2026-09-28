"""Tests for robustcd.adapters.mambafcs that need neither a GPU nor the CUDA kernel.

Upstream-comparison tests need the pinned checkout at ``$ROBUSTCD_EXT/MambaFCS``
(default ``~/ext``) and are skipped otherwise.

Run with ``pytest tests/`` or ``python tests/test_mambafcs_adapter.py``.
"""

from __future__ import annotations

import ast
import json
import os
import re
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robustcd.adapters import changemamba as cm  # noqa: E402
from robustcd.metrics.labels import SECOND_COLORMAP  # noqa: E402

FCS = Path(os.environ.get("ROBUSTCD_EXT", Path.home() / "ext")) / "MambaFCS"


class Skip(Exception):
    pass


def need(cond, why):
    if not cond:
        try:
            import pytest
            pytest.skip(why)
        except ImportError:
            raise Skip(why)


def _notebook_dicts():
    nb = json.loads((FCS / "annotations" / "MambaFCS.ipynb").read_text())
    src = "\n".join("".join(c["source"]) for c in nb["cells"] if c["cell_type"] == "code")
    out = {}
    for name in ("ori_label_value_dict", "target_label_value_dict"):
        m = re.search(rf"{name}\s*=\s*(\{{[^}}]*\}})", src)
        out[name] = ast.literal_eval(m.group(1))
    return out


def test_upstream_class_order_is_changemamba_order():
    need((FCS / "annotations" / "MambaFCS.ipynb").exists(), "MambaFCS checkout not found")
    d = _notebook_dicts()
    colour_of = d["ori_label_value_dict"]
    for name, idx in d["target_label_value_dict"].items():
        if idx == 0:
            continue
        # the upstream index must carry the same colour as ChangeMamba's palette at that index
        assert tuple(colour_of[name]) == tuple(cm.CM_COLORS[idx]), (name, idx)
        # and SECOND_TO_CM must send the robustcd class of that colour to this index
        k = [tuple(c) for c in SECOND_COLORMAP].index(tuple(colour_of[name]))
        assert cm.SECOND_TO_CM[k] == idx, (name, k, idx)


def test_normalisation_matches_upstream_imutils():
    need((FCS / "changedetection" / "datasets" / "imutils.py").exists(), "MambaFCS checkout not found")
    src = (FCS / "changedetection" / "datasets" / "imutils.py").read_text()
    m = re.search(r"def normalize_img\(img, mean=(\[[^\]]*\]), std=(\[[^\]]*\])\)", src)
    assert np.allclose(ast.literal_eval(m.group(1)), cm.CM_MEAN)
    assert np.allclose(ast.literal_eval(m.group(2)), cm.CM_STD)


def test_scd_eval_all_with_37_classes_is_the_second_definition():
    """Upstream scores SCDD_eval_all(preds, labels, 37) on 7-class maps; it must equal the 7-class value."""
    need((FCS / "changedetection" / "utils_func" / "mcd_utils.py").exists(), "MambaFCS checkout not found")
    src = (FCS / "changedetection" / "utils_func" / "mcd_utils.py").read_text()
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, ast.FunctionDef)
            and n.name in ("fast_hist", "get_hist", "cal_kappa", "SCDD_eval_all")]
    import math
    from scipy import stats
    ns = {"np": np, "math": math, "stats": stats}
    exec(compile("\n\n".join(ast.get_source_segment(src, n) for n in keep), "mcd_utils.py", "exec"), ns)
    rng = np.random.default_rng(0)
    labels = [rng.integers(0, 7, (32, 32)) * (rng.random((32, 32)) < 0.3) for _ in range(6)]
    preds = [np.where(rng.random(l.shape) < 0.7, l, rng.integers(0, 7, l.shape)) for l in labels]
    a = ns["SCDD_eval_all"](preds, labels, 7)
    b = ns["SCDD_eval_all"](preds, labels, 37)
    assert np.allclose(a, b), (a, b)


def test_ce_guard_equals_mean_ce_and_is_zero_when_all_ignored():
    try:
        import torch
        import torch.nn.functional as F
    except ImportError:
        need(False, "torch not installed")
    g = torch.Generator().manual_seed(0)
    logits = torch.randn(2, 7, 8, 8, generator=g)
    target = torch.randint(0, 7, (2, 8, 8), generator=g)
    target[0, :4] = 255
    n = (target != 255).sum().clamp(min=1)
    guarded = F.cross_entropy(logits, target, ignore_index=255, reduction="sum") / n
    assert torch.allclose(guarded, F.cross_entropy(logits, target, ignore_index=255))
    allign = torch.full_like(target, 255)
    n0 = (allign != 255).sum().clamp(min=1)
    z = F.cross_entropy(logits, allign, ignore_index=255, reduction="sum") / n0
    assert float(z) == 0.0 and torch.isnan(F.cross_entropy(logits, allign, ignore_index=255))


if __name__ == "__main__":
    fails = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Skip as e:
                print(f"SKIP {name}: {e}")
            except Exception as e:  # noqa: BLE001
                fails += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    raise SystemExit(1 if fails else 0)
