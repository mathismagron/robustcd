"""Tests for robustcd.adapters.changemamba that need neither a GPU nor the upstream repo.

Run with ``pytest tests/`` or ``python tests/test_changemamba_adapter.py``.
"""

from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robustcd.adapters import changemamba as cm  # noqa: E402
from robustcd.metrics.labels import SECOND_COLORMAP  # noqa: E402


def test_class_luts_follow_the_colour_tables():
    # colour is the ground truth: class k in robustcd order must map to the
    # ChangeMamba index whose colour is the same (index 0 colours differ by
    # convention: white in SECOND files, black in ChangeMamba's palette)
    for k in range(1, 7):
        j = int(cm.SECOND_TO_CM[k])
        assert tuple(SECOND_COLORMAP[k]) == tuple(cm.CM_COLORS[j]), (k, j)
    assert cm.SECOND_TO_CM[0] == 0 and cm.CM_TO_SECOND[0] == 0


def test_luts_are_inverse_permutations():
    idx = np.arange(7, dtype=np.uint8)
    assert np.array_equal(cm.CM_TO_SECOND[cm.SECOND_TO_CM[idx]], idx)
    assert np.array_equal(cm.SECOND_TO_CM[cm.CM_TO_SECOND[idx]], idx)
    assert sorted(cm.SECOND_TO_CM.tolist()) == list(range(7))


def test_normalize_matches_upstream_formula():
    img = np.random.default_rng(0).integers(0, 256, (8, 8, 3), dtype=np.uint8)
    x = cm.normalize(img)
    assert x.shape == (3, 8, 8) and x.dtype == np.float32
    ref = (img[..., 0].astype(np.float32) - 123.675) / 58.395
    assert np.allclose(x[0], ref)


def _torch():
    try:
        import torch  # noqa: F401
        return True
    except ImportError:
        print("  skipped: torch not installed")
        return False


def test_sampler_resume_replays_the_same_order():
    if not _torch():
        return
    from robustcd.adapters.changemamba.train import EpochOrderSampler

    full = list(EpochOrderSampler(n=7, total=30, seed=3))
    assert len(full) == 30
    for start in (0, 5, 7, 13, 29):
        assert list(EpochOrderSampler(n=7, total=30, seed=3, start=start)) == full[start:]
    # every epoch is a permutation: each sample seen exactly once per epoch
    for e in range(4):
        assert sorted(full[7 * e : 7 * e + 7]) == list(range(7))
    assert list(EpochOrderSampler(n=7, total=30, seed=4)) != full


def test_decode_maps_to_second_order_and_masks_unchanged():
    if not _torch():
        return
    import torch

    # 1 x 1 x 2 image: pixel 0 changed, pixel 1 unchanged
    cd = torch.tensor([[[[0.0, 5.0]], [[5.0, 0.0]]]])          # argmax -> [1, 0]
    t1 = torch.zeros(1, 7, 1, 2); t1[0, 4, 0, :] = 9.0            # CM class 4 (water)
    t2 = torch.zeros(1, 7, 1, 2); t2[0, 0, 0, :] = 9.0; t2[0, 1, 0, :] = 5.0  # CM 0 > CM 1 (low veg)
    s1, s2, ch = cm.decode(cd, t1, t2, "restricted")
    assert ch.tolist() == [[[1, 0]]]
    assert s1.tolist() == [[[1, 0]]]            # CM water (4) -> SECOND water (1); unchanged -> 0
    assert s2.tolist() == [[[3, 0]]]            # restricted ignores CM class 0 -> low veg -> SECOND 3
    _, f2, _ = cm.decode(cd, t1, t2, "full")
    assert f2.tolist() == [[[0, 0]]]            # full argmax picks the untrained class 0


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
