"""Tests for robustcd.metrics.

Run with ``pytest tests/`` or ``python tests/test_metrics.py``.

The equivalence test against the widely used Bi-SRNet ``SCDD_eval_all`` runs
only when ``ROBUSTCD_REFERENCE_UTILS`` points to a local copy of that repo's
``utils/utils.py`` (it is not vendored here).
"""

from __future__ import annotations

import importlib.util
import math
import os
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robustcd.metrics import (  # noqa: E402
    BCDMeter, SCDMeter, bootstrap_ci, compose_prediction, confusion, index_to_rgb,
    paired_bootstrap, rgb_to_index, scd_scores,
)

K = 7


def _random_pair(rng, shape=(64, 64), p_change=0.3):
    """Consistent (sem1, sem2) GT: 0 outside change, classes 1..6 inside."""
    change = rng.random(shape) < p_change
    s1 = np.where(change, rng.integers(1, K, shape), 0).astype(np.uint8)
    s2 = np.where(change, rng.integers(1, K, shape), 0).astype(np.uint8)
    return s1, s2, change


def _noisy(rng, gt, change, flip=0.1, cls_err=0.2):
    """Plausible prediction: flip some change pixels, corrupt some classes."""
    ch = change ^ (rng.random(change.shape) < flip)
    sem = np.where(rng.random(gt.shape) < cls_err, rng.integers(1, K, gt.shape), gt)
    sem = np.where(sem == 0, rng.integers(1, K, gt.shape), sem).astype(np.uint8)
    return compose_prediction(sem, ch)


def _manual_sek(h):
    """Independent loop-based SeK, written from the paper's definition."""
    h = h.astype(float)
    n0 = h.copy()
    n0[0, 0] = 0
    tot = n0.sum()
    po = sum(n0[i, i] for i in range(len(h))) / tot
    pe = sum(n0[i, :].sum() * n0[:, i].sum() for i in range(len(h))) / tot**2
    kappa = (po - pe) / (1 - pe)
    tp = h[1:, 1:].sum()
    iou_c = tp / (h.sum() - h[0, 0])
    return kappa * math.exp(iou_c) / math.e


# --------------------------------------------------------------------------- #


def test_perfect_prediction_scores_one():
    rng = np.random.default_rng(0)
    m = SCDMeter()
    for _ in range(4):
        s1, s2, _ = _random_pair(rng)
        m.update(s1, s2, s1, s2)
    r = m.compute()
    for k in ("SeK", "mIoU", "Fscd", "OA", "change_F1"):
        assert abs(r[k] - 1.0) < 1e-12, (k, r[k])


def test_all_unchanged_prediction_is_degenerate_not_nan():
    rng = np.random.default_rng(1)
    s1, s2, _ = _random_pair(rng)
    z = np.zeros_like(s1)
    r = SCDMeter().update(z, z, s1, s2)
    r = scd_scores(r)
    assert r["Fscd"] == 0.0 and r["IoU_change"] == 0.0
    assert math.isnan(r["Pscd"])  # nothing predicted as changed
    assert r["SeK"] <= 0.0


def test_matches_independent_definition():
    rng = np.random.default_rng(2)
    m = SCDMeter()
    for _ in range(6):
        s1, s2, ch = _random_pair(rng)
        m.update(_noisy(rng, s1, ch), _noisy(rng, s2, ch), s1, s2)
    assert abs(m.compute()["SeK"] - _manual_sek(m.total)) < 1e-12


def test_transpose_invariance():
    rng = np.random.default_rng(3)
    h = rng.integers(0, 1000, (K, K))
    a, b = scd_scores(h), scd_scores(h.T)
    for k in ("SeK", "mIoU", "Fscd", "OA", "kappa_n0"):
        assert abs(a[k] - b[k]) < 1e-12, k


def test_streaming_equals_single_pass():
    rng = np.random.default_rng(4)
    pairs = [_random_pair(rng) for _ in range(5)]
    preds = [(_noisy(rng, s1, c), _noisy(rng, s2, c)) for s1, s2, c in pairs]
    m = SCDMeter()
    for (s1, s2, _), (p1, p2) in zip(pairs, preds):
        m.update(p1, p2, s1, s2)
    cat = lambda xs: np.concatenate(xs, 0)  # noqa: E731
    one = SCDMeter()
    one.update(cat([p[0] for p in preds]), cat([p[1] for p in preds]),
               cat([s[0] for s in pairs]), cat([s[1] for s in pairs]))
    assert np.array_equal(m.total, one.total)


def test_valid_mask_equals_cropping():
    rng = np.random.default_rng(5)
    s1, s2, ch = _random_pair(rng, (64, 64))
    p1, p2 = _noisy(rng, s1, ch), _noisy(rng, s2, ch)
    valid = np.zeros((64, 64), bool)
    valid[8:56, 4:60] = True
    a = SCDMeter()
    a.update(p1, p2, s1, s2, valid=valid)
    b = SCDMeter()
    sl = np.s_[8:56, 4:60]
    b.update(p1[sl], p2[sl], s1[sl], s2[sl])
    assert np.array_equal(a.total, b.total)


def test_ignore_index_and_range_checks():
    gt = np.array([[0, 1], [255, 2]], np.uint8)
    pr = np.array([[0, 1], [3, 2]], np.uint8)
    h = confusion(pr, gt, 7)
    assert h.sum() == 3
    try:
        confusion(np.array([[9]]), np.array([[1]]), 7)
    except ValueError:
        pass
    else:
        raise AssertionError("out-of-range prediction must raise")


def test_fromto_secondary_score():
    from robustcd.metrics.scd import fromto_map
    a = np.array([[0, 1, 6]], np.uint8)
    b = np.array([[0, 2, 6]], np.uint8)
    assert fromto_map(a, b).tolist() == [[0, 2, 36]]
    # class error on one date only: SECOND-definition Fscd = 0.5, from-to Fscd = 0
    m = SCDMeter()
    g1 = np.array([[1, 1]], np.uint8); g2 = np.array([[2, 2]], np.uint8)
    p1 = np.array([[1, 1]], np.uint8); p2 = np.array([[3, 3]], np.uint8)
    m.update(p1, p2, g1, g2)
    r = m.compute()
    assert abs(r["Fscd"] - 0.5) < 1e-12 and r["Fscd_fromto"] == 0.0
    assert m.total_fromto.shape == (37, 37)


def test_compose_prediction():
    sem = np.array([[3, 4], [0, 5]], np.uint8)
    ch = np.array([[1, 0], [1, 1]], np.uint8)
    assert compose_prediction(sem, ch).tolist() == [[3, 0], [0, 5]]


def test_rgb_roundtrip_and_strict_decoding():
    idx = np.arange(7, dtype=np.uint8).reshape(1, 7).repeat(3, 0)
    assert np.array_equal(rgb_to_index(index_to_rgb(idx)), idx)
    bad = index_to_rgb(idx).copy()
    bad[0, 0] = (1, 2, 3)
    try:
        rgb_to_index(bad)
    except ValueError:
        pass
    else:
        raise AssertionError("unknown colour must raise in strict mode")


def test_binary_meter():
    gt = np.array([[0, 1, 1, 0]], np.uint8) * 255
    pr = np.array([[0, 1, 0, 1]], np.uint8)
    m = BCDMeter()
    m.update(pr, gt)
    r = m.compute()
    assert r["precision"] == 0.5 and r["recall"] == 0.5 and r["F1"] == 0.5
    assert abs(r["IoU_change"] - 1 / 3) < 1e-12


def test_bootstrap_is_deterministic_and_brackets_point():
    rng = np.random.default_rng(6)
    m = SCDMeter()
    for _ in range(30):
        s1, s2, ch = _random_pair(rng, (32, 32))
        m.update(_noisy(rng, s1, ch), _noisy(rng, s2, ch), s1, s2)
    a = bootstrap_ci(m.per_sample, scd_scores, n_boot=200, seed=1)
    b = bootstrap_ci(m.per_sample, scd_scores, n_boot=200, seed=1)
    assert a == b
    for k, v in a.items():
        assert v["lo"] <= v["value"] <= v["hi"], (k, v)


def test_paired_bootstrap_is_narrower_than_marginals():
    rng = np.random.default_rng(7)
    clean, deg = SCDMeter(), SCDMeter()
    for _ in range(40):
        s1, s2, ch = _random_pair(rng, (32, 32), p_change=rng.uniform(0.05, 0.6))
        p1, p2 = _noisy(rng, s1, ch, flip=0.05), _noisy(rng, s2, ch, flip=0.05)
        clean.update(p1, p2, s1, s2)
        q1, q2 = _noisy(rng, s1, ch, flip=0.15), _noisy(rng, s2, ch, flip=0.15)
        deg.update(q1, q2, s1, s2)
    d = paired_bootstrap(clean.per_sample, deg.per_sample, scd_scores, stat="difference", n_boot=300)
    ca = bootstrap_ci(clean.per_sample, scd_scores, keys=("SeK",), n_boot=300)["SeK"]
    cb = bootstrap_ci(deg.per_sample, scd_scores, keys=("SeK",), n_boot=300)["SeK"]
    assert d["value"] < 0
    assert d["se"] < math.hypot(ca["se"], cb["se"])


def test_equivalence_with_bisrnet_reference():
    path = os.environ.get("ROBUSTCD_REFERENCE_UTILS")
    if not path:
        try:
            import pytest
            pytest.skip("ROBUSTCD_REFERENCE_UTILS not set")
        except ImportError:
            print("  skipped: ROBUSTCD_REFERENCE_UTILS not set")
            return
    # the reference file imports a sibling module unrelated to SeK; stub it
    import types
    stub = types.ModuleType("utils")
    stub.eval_segm = types.ModuleType("utils.eval_segm")
    saved = {k: sys.modules.get(k) for k in ("utils", "utils.eval_segm")}
    sys.modules["utils"], sys.modules["utils.eval_segm"] = stub, stub.eval_segm
    try:
        spec = importlib.util.spec_from_file_location("ref_utils", path)
        ref = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(ref)
    finally:
        for k, v in saved.items():
            if v is None:
                sys.modules.pop(k, None)
            else:
                sys.modules[k] = v
    rng = np.random.default_rng(8)
    preds, labels, m = [], [], SCDMeter()
    for _ in range(20):
        s1, s2, ch = _random_pair(rng, (48, 48), p_change=rng.uniform(0.02, 0.7))
        p1, p2 = _noisy(rng, s1, ch), _noisy(rng, s2, ch)
        preds += [p1, p2]
        labels += [s1, s2]
        m.update(p1, p2, s1, s2)
    fscd, miou, sek = ref.SCDD_eval_all(preds, labels, K)
    r = m.compute()
    assert abs(r["SeK"] - sek) < 1e-12 and abs(r["mIoU"] - miou) < 1e-12 and abs(r["Fscd"] - fscd) < 1e-12


if __name__ == "__main__":
    failed = 0
    for name, fn in sorted(globals().items()):
        if name.startswith("test_") and callable(fn):
            try:
                fn()
                print(f"PASS {name}")
            except Exception as e:  # noqa: BLE001
                failed += 1
                print(f"FAIL {name}: {type(e).__name__}: {e}")
    sys.exit(1 if failed else 0)
