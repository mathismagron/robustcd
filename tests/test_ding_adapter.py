"""Tests for robustcd.adapters.ding, robustcd.datasets.scd_train and robustcd.training.

Upstream-comparison tests need the pinned checkouts under ``$ROBUSTCD_EXT``
(default ``~/ext``: ``SCanNet/`` and ``Bi-SRNet/``) and are skipped otherwise.
Torch-dependent tests are skipped when torch is not installed.

Run with ``pytest tests/`` or ``python tests/test_ding_adapter.py``.
"""

from __future__ import annotations

import ast
import os
import random
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robustcd.adapters import ding  # noqa: E402
from robustcd.datasets.scd_train import rot90_flip_ding  # noqa: E402
from robustcd.metrics.labels import SECOND_COLORMAP  # noqa: E402

EXT = Path(os.environ.get("ROBUSTCD_EXT", Path.home() / "ext"))
SCANNET = EXT / "SCanNet"
BISRNET = EXT / "Bi-SRNet"

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:  # pragma: no cover
    HAVE_TORCH = False


class Skip(Exception):
    pass


def need(cond, why):
    if not cond:
        try:
            import pytest
            pytest.skip(why)
        except ImportError:
            raise Skip(why)


def upstream_rs_st(repo: Path) -> dict:
    src = (repo / "datasets" / "RS_ST.py").read_text().replace("\r", "")
    out = {}
    for name in ("MEAN_A", "STD_A", "MEAN_B", "STD_B"):
        m = re.search(rf"^{name}\s*=\s*np\.array\((\[[^\]]*\])\)", src, re.M)
        out[name] = np.array(ast.literal_eval(m.group(1)))
    out["ST_COLORMAP"] = ast.literal_eval(re.search(r"^ST_COLORMAP\s*=\s*(\[.*\])", src, re.M).group(1))
    return out


# --------------------------------------------------------------------------- #
# numpy-only
# --------------------------------------------------------------------------- #
def test_upstream_class_order_is_robustcd_order():
    assert [tuple(c) for c in ding.ST_COLORMAP] == [tuple(c) for c in SECOND_COLORMAP]


def test_constants_match_upstream_files():
    for repo in (SCANNET, BISRNET):
        need((repo / "datasets" / "RS_ST.py").exists(), f"{repo} not found")
        up = upstream_rs_st(repo)
        for name in ("MEAN_A", "STD_A", "MEAN_B", "STD_B"):
            assert np.allclose(getattr(ding, name), up[name]), (repo, name)
        assert [tuple(c) for c in up["ST_COLORMAP"]] == [tuple(c) for c in ding.ST_COLORMAP]


def test_normalize_pair_uses_per_date_statistics():
    rng = np.random.default_rng(0)
    im1, im2 = (rng.integers(0, 256, (8, 8, 3), dtype=np.uint8) for _ in range(2))
    x1, x2 = ding.normalize_pair(im1, im2)
    assert x1.shape == (3, 8, 8) and x1.dtype == np.float32
    assert np.allclose(x1[1], (im1[..., 1] - 114.08) / 46.27, atol=1e-5)
    assert np.allclose(x2[2], (im2[..., 2] - 118.18) / 47.94, atol=1e-5)


def test_augmentation_is_identical_across_streams_and_geometric_only():
    rng = np.random.default_rng(1)
    im = rng.integers(0, 256, (6, 6, 3), dtype=np.uint8)
    lab = rng.integers(0, 7, (6, 6), dtype=np.uint8)
    seen = set()
    r = random.Random(0)
    for _ in range(400):
        a, b, la, lb = rot90_flip_ding(im, im.copy(), lab, lab.copy(), rng=r)
        assert np.array_equal(a, b) and np.array_equal(la, lb)
        # images and labels undergo the same transform: channel 0 tracks the label layout
        key = a[..., 0].tobytes()
        seen.add(key)
        assert sorted(a.reshape(-1, 3).tolist()) == sorted(im.reshape(-1, 3).tolist())  # pixels permuted only
    assert len(seen) == 8  # the dihedral group: 2 rotations x 4 flips reach all 8 elements


def test_augmentation_distribution_matches_upstream():
    need((SCANNET / "utils" / "transform.py").exists(), "SCanNet checkout not found")
    # upstream transform.py imports cv2 at top level: extract only the three functions
    src = (SCANNET / "utils" / "transform.py").read_text().replace("\r", "")
    tree = ast.parse(src)
    names = ("rand_rot90_SCD", "rand_flip_SCD", "rand_rot90_flip_SCD")
    code = "\n\n".join(ast.get_source_segment(src, n) for n in tree.body
                       if isinstance(n, ast.FunctionDef) and n.name in names)
    ns = {"np": np, "random": random}
    exec(compile(code, "transform.py", "exec"), ns)
    im = np.arange(4 * 4 * 3, dtype=np.uint8).reshape(4, 4, 3)
    lab = np.arange(16, dtype=np.uint8).reshape(4, 4)
    for seed in range(50):
        random.seed(seed)
        ref = ns["rand_rot90_flip_SCD"](im, im.copy(), lab, lab.copy())
        ours = rot90_flip_ding(im, im.copy(), lab, lab.copy(), rng=random.Random(seed))
        for x, y in zip(ref, ours):
            assert np.array_equal(x, y), seed


def test_registry_is_complete():
    for name, spec in ding.MODELS.items():
        assert spec.repo in ding.REPOS, name
    assert ding.MODELS["scannet"].recipe.psd and not ding.MODELS["bisrnet"].recipe.psd
    assert not ding.MODELS["hrscd4"].recipe.sc_loss and ding.MODELS["bisrnet"].recipe.sc_loss


# --------------------------------------------------------------------------- #
# torch
# --------------------------------------------------------------------------- #
def test_decode_masks_semantics_and_restricts_classes():
    need(HAVE_TORCH, "torch not installed")
    import torch

    oc = torch.tensor([[[[1.0, -1.0]]]])                     # changed, unchanged
    oa = torch.zeros(1, 7, 1, 2)
    oa[0, 0] = 5.0                                           # class 0 dominates everywhere
    oa[0, 3] = 1.0
    s1, s2, ch = ding.decode((oc, oa, oa.clone()), "restricted")
    assert ch.tolist() == [[[1, 0]]] and s1.tolist() == [[[3, 0]]]
    s1f, _, _ = ding.decode((oc, oa, oa.clone()), "full")
    assert s1f.tolist() == [[[0, 0]]]


def _upstream_psd_functions():
    """Extract AverageThred / calc_conf from train_SCD_psd.py without running the script."""
    src = (SCANNET / "train_SCD_psd.py").read_text().replace("\r", "")
    tree = ast.parse(src)
    keep = [n for n in tree.body if isinstance(n, (ast.ClassDef, ast.FunctionDef))
            and n.name in ("AverageThred", "calc_conf")]
    code = "\n\n".join(ast.get_source_segment(src, n) for n in keep).replace(".cuda()", "")
    import torch
    import torch.nn.functional as F

    class RS:
        num_classes = 7

    ns = {"np": np, "torch": torch, "F": F, "RS": RS, "args": {"pseudo_thred": 0.6}}
    exec(compile(code, "train_SCD_psd.py", "exec"), ns)
    return ns["AverageThred"], ns["calc_conf"]


def test_pseudo_label_rule_matches_upstream():
    need(HAVE_TORCH, "torch not installed")
    need((SCANNET / "train_SCD_psd.py").exists(), "SCanNet checkout not found")
    import torch

    from robustcd.adapters.ding.psd import AverageThreshold, confident

    UpThred, up_calc = _upstream_psd_functions()
    g = torch.Generator().manual_seed(0)
    th_up, th_ours = UpThred(7), AverageThreshold(7, 0.6)
    for step in range(3):                       # thresholds are running averages: check several updates
        prob = torch.softmax(3 * torch.randn(2, 7, 16, 16, generator=g), dim=1)
        c_up, i_up = up_calc(prob, th_up)
        c_ours, i_ours = confident(prob, th_ours, 0.6)
        assert torch.equal(c_up, c_ours) and torch.equal(i_up, i_ours), step
        assert np.allclose(th_up.value(), th_ours.value())


def test_pseudo_labeler_only_fills_unchanged_pixels_and_roundtrips():
    need(HAVE_TORCH, "torch not installed")
    import torch

    from robustcd.adapters.ding.psd import PseudoLabeler

    class Tiny(torch.nn.Module):
        def __init__(self):
            super().__init__()
            self.c = torch.nn.Conv2d(3, 15, 1)

        def forward(self, a, b):
            y = self.c(a + b)
            return y[:, :1], 4 * y[:, 1:8], 4 * y[:, 8:]

    torch.manual_seed(0)
    m = Tiny()
    p = PseudoLabeler(m, tta=True)
    x = torch.randn(2, 3, 8, 8)
    la = torch.randint(0, 7, (2, 8, 8))
    la[:, :4] = 0
    lb = la.clone()
    bn = (la > 0).unsqueeze(1).float()
    assert not p.active and p.apply(x, x, la, lb, bn)[0] is la
    p.on_eval({"Fscd": 0.5}, 2000)
    assert p.teacher is not None and not p.active          # below psd_init_fscd
    p.on_eval({"Fscd": 0.7}, 4000)
    assert p.active and p.teacher_iteration == 4000
    na, nb = p.apply(x, x, la, lb, bn)
    changed = la > 0
    assert torch.equal(na[changed], la[changed]) and torch.equal(nb[changed], lb[changed])
    assert torch.equal(na, nb)  # same pseudo label on both dates (inputs identical here)
    sd = p.state_dict()
    q = PseudoLabeler(m)
    q.load_state_dict(sd)
    assert q.active and q.best_fscd == 0.7
    for a, b in zip(q.teacher.state_dict().values(), p.teacher.state_dict().values()):
        assert torch.equal(a, b)


def test_poly_lr_matches_upstream_formula():
    from robustcd.training import poly_lr

    f = poly_lr(0.1, 1000, 1.5)
    for it in (1, 10, 500, 999, 1000):
        assert abs(f(it) - 0.1 * (1 - it / 1000) ** 1.5) < 1e-12


def test_models_build_and_forward():
    need(HAVE_TORCH, "torch not installed")
    import torch

    need(BISRNET.exists(), f"{BISRNET} not found")
    # One checkout may be active per process, so only the cheapest model is built here;
    # all five were built and run at 512x512 during adapter development (cluster/models/ding/README.md).
    ding._ACTIVE = None
    m = ding.build_model("hrscd4", BISRNET, imagenet=False).eval()
    with torch.no_grad():
        oc, oa, ob = m(torch.randn(1, 3, 64, 64), torch.randn(1, 3, 64, 64))
    assert oc.shape == (1, 1, 64, 64) and oa.shape == (1, 7, 64, 64)
    CE, wBCE, CS = ding.upstream_losses()
    la = torch.randint(0, 7, (1, 64, 64))
    loss = 0.5 * CE(ignore_index=0)(oa, la) + wBCE(oc, (la > 0).unsqueeze(1).float())
    assert torch.isfinite(loss)
    ding._ACTIVE = None


def test_training_loop_resumes_where_it_stopped():
    need(HAVE_TORCH, "torch not installed")
    import json

    import torch

    from robustcd.training import EXIT_REQUEUE, LoopConfig, run

    class DS:
        def __len__(self):
            return 10

        def __getitem__(self, i):
            g = torch.Generator().manual_seed(i)
            return torch.randn(3, 8, 8, generator=g), torch.randint(0, 2, (8, 8), generator=g)

    def make():
        torch.manual_seed(0)
        m = torch.nn.Conv2d(3, 2, 1)
        return m, torch.optim.SGD(m.parameters(), lr=0.1)

    def step_for(m):
        def step(batch, it):
            x, y = batch
            return torch.nn.functional.cross_entropy(m(x), y), {}
        return step

    counter = {"n": 0}

    def ev(m):
        counter["n"] += 1
        return {"SeK": 0.1 * counter["n"], "mIoU": 0.5, "Fscd": 0.5}

    with tempfile.TemporaryDirectory() as d:
        # uninterrupted reference
        m_ref, o_ref = make()
        cfg = LoopConfig(out=f"{d}/ref", seed=0, max_iters=6, batch_size=4, accum=2, eval_interval=2,
                         log_interval=1, workers=0, bf16=False)
        assert run(cfg, model=m_ref, optimizer=o_ref, dataset=DS(), train_step=step_for(m_ref), evaluate=ev) == 0
        # interrupted at iteration 2 (time guard of 0 minutes), then resumed
        m1, o1 = make()
        cfg1 = LoopConfig(out=f"{d}/cut", seed=0, max_iters=6, batch_size=4, accum=2, eval_interval=2,
                          log_interval=1, workers=0, bf16=False, stop_after_min=0.0)
        assert run(cfg1, model=m1, optimizer=o1, dataset=DS(), train_step=step_for(m1), evaluate=ev) == EXIT_REQUEUE
        m2, o2 = make()
        cfg2 = LoopConfig(out=f"{d}/cut", seed=0, max_iters=6, batch_size=4, accum=2, eval_interval=2,
                          log_interval=1, workers=0, bf16=False)
        assert run(cfg2, model=m2, optimizer=o2, dataset=DS(), train_step=step_for(m2), evaluate=ev) == 0
        for a, b in zip(m_ref.state_dict().values(), m2.state_dict().values()):
            assert torch.allclose(a, b, atol=1e-6)
        summ = json.loads(Path(f"{d}/cut/summary.json").read_text())
        assert summ["last_iteration"] == 6 and (Path(d) / "cut" / "best_model.pth").exists()


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
