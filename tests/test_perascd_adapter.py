"""Tests for robustcd.adapters.perascd that need neither a GPU nor the CUDA op nor the 9 GB weights.

Model tests build the small ``ViT-B/16`` variant with random weights and the
PyTorch reference of the deformable attention. They need the pinned checkout
at ``$ROBUSTCD_EXT/PerASCD`` (default ``~/ext``) and are skipped otherwise.

Run with ``pytest tests/`` or ``python tests/test_perascd_adapter.py``.
"""

from __future__ import annotations

import ast
import os
import re
import sys
import tempfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from robustcd.adapters import changemamba as cm  # noqa: E402
from robustcd.adapters import perascd as P  # noqa: E402
from robustcd.metrics.labels import SECOND_COLORMAP  # noqa: E402

REPO = Path(os.environ.get("ROBUSTCD_EXT", Path.home() / "ext")) / "PerASCD"


class Skip(Exception):
    pass


def need(cond, why):
    if not cond:
        try:
            import pytest
            pytest.skip(why)
        except ImportError:
            raise Skip(why)


def _src(rel):
    return (REPO / rel).read_text().replace("\r", "")


def _activate():
    """Activate the PerASCD checkout, dropping another upstream's ``models``/``utils`` (pytest runs
    every adapter's tests in one process; the adapters themselves refuse to mix checkouts)."""
    need(REPO.is_dir(), f"PerASCD checkout not found at {REPO}")
    root = str(REPO.resolve())
    if P._ACTIVE != root:
        from robustcd.adapters import ding
        for name in [m for m in sys.modules if m.split(".")[0] in ("models", "utils", "datasets")]:
            del sys.modules[name]
        sys.path[:] = [p for p in sys.path if p != ding._ACTIVE]
        ding._ACTIVE = None
        P._ACTIVE = None
    return P.activate(REPO, "pytorch")


def _small_model(droppath=0.0, checkpointing="none", sdpa=True, seed=0):
    import torch

    _activate()
    torch.manual_seed(seed)
    model, info = P.build_model(REPO, "ViT-B/16", droppath=droppath, checkpointing=checkpointing,
                                msda="pytorch", sdpa=sdpa)
    return model, info


def test_upstream_class_order_is_changemamba_order():
    need(REPO.is_dir(), "PerASCD checkout not found")
    m = re.search(r"^ST_COLORMAP\s*=\s*(\[.*\])", _src("datasets/RS_ST.py"), re.M)
    colors = [tuple(c) for c in ast.literal_eval(m.group(1))]
    assert colors[1:] == list(cm.CM_COLORS[1:])
    # and decode(order="cm") sends each released index to the robustcd class of the same colour
    for i in range(1, 7):
        assert SECOND_COLORMAP[int(P._cm_to_robustcd()[i])] == colors[i]


def test_normalization_matches_upstream_constants():
    src = _src("datasets/RS_ST.py") if REPO.is_dir() else None
    if src is not None:
        blk = src[src.index("class DataPerAAUG"):]
        mean = ast.literal_eval(re.search(r"self\.mean\s*=\s*(\[[^\]]*\])", blk).group(1))
        std = ast.literal_eval(re.search(r"self\.std\s*=\s*(\[[^\]]*\])", blk).group(1))
        assert np.allclose(mean, P.PERA_MEAN) and np.allclose(std, P.PERA_STD)
    rng = np.random.default_rng(0)
    a = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    b = rng.integers(0, 256, (8, 8, 3), dtype=np.uint8)
    x1, x2 = P.normalize_pair(a, b)
    ref = (a.transpose(2, 0, 1) / 255.0 - P.PERA_MEAN[:, None, None]) / P.PERA_STD[:, None, None]
    assert x1.dtype == np.float32 and x1.shape == (3, 8, 8) and np.allclose(x1, ref, atol=1e-5)
    assert not np.allclose(x1, x2)


def test_decode_orders_and_modes():
    import torch

    ch = torch.tensor([[[[1.0, -1.0]]]])                      # changed, unchanged
    a = torch.zeros(1, 7, 1, 2); a[0, 0] = 5.0; a[0, 4] = 1.0  # class 0 wins "full"; 4 wins "restricted"
    b = torch.zeros(1, 7, 1, 2); b[0, 1] = 1.0
    s1, s2, c = P.decode((ch, a, b), "restricted")
    assert c.tolist() == [[[1, 0]]] and s1.tolist() == [[[4, 0]]] and s2.tolist() == [[[1, 0]]]
    s1f, _, _ = P.decode((ch, a, b), "full")
    assert s1f.tolist() == [[[0, 0]]]
    r1, r2, _ = P.decode_released((ch, a, b), "restricted")
    # ChangeMamba index 4 is water (robustcd 1); index 1 is low vegetation (robustcd 3)
    assert r1.tolist() == [[[1, 0]]] and r2.tolist() == [[[3, 0]]]


def test_warmup_poly_matches_upstream_adjust_lr():
    need(REPO.is_dir(), "PerASCD checkout not found")
    from robustcd.training import warmup_poly_lr

    tree = ast.parse(_src("train.py"))
    fn = next(n for n in tree.body if isinstance(n, ast.FunctionDef) and n.name == "adjust_lr")
    ns = {"args": {"lr": 0.1, "lr_decay_power": 1.5}}
    exec(compile(ast.Module([fn], []), "adjust_lr", "exec"), ns)

    class Opt:
        param_groups = [{"lr": None}]
    total = 1000
    f = warmup_poly_lr(0.1, total, 1.5, 0.1, 0.0)
    for it in (1, 50, 99, 100, 101, 500, 999, 1000):
        ns["adjust_lr"](Opt, it / total, init_lr=0.1, warmup_ratio=0.1, min_lr=0.0)
        assert abs(Opt.param_groups[0]["lr"] - f(it)) < 1e-12, it


def test_sdpa_attention_equals_upstream_attention():
    import torch

    model, info = _small_model(sdpa=True)
    assert info["attention"] == "sdpa"
    from models.pera_layers import attention as att

    blk_attn = model.backbone.blocks[0].attn
    x = torch.randn(2, 17, 768)
    blk_attn.eval()
    with torch.no_grad():
        y_sdpa = blk_attn(x)
        y_up = att.MemEffAttention.upstream_forward(blk_attn, x)
    assert torch.allclose(y_sdpa, y_up, atol=1e-5), (y_sdpa - y_up).abs().max()


def test_forward_shapes_and_train_step():
    import torch

    from robustcd.adapters.perascd.train import make_train_step

    model, _ = _small_model(droppath=0.3)
    x1, x2 = torch.randn(2, 3, 128, 128), torch.randn(2, 3, 128, 128)
    model.eval()
    with torch.no_grad():
        ch, a, b = model(x1, x2)
    assert ch.shape == (2, 1, 512, 512) and a.shape == (2, 7, 512, 512) and b.shape == a.shape
    model.output_size = 128
    model.train()
    CE, wbce, sc = P.upstream_losses(0.01)
    step = make_train_step(model, torch.device("cpu"), CE(ignore_index=0), wbce, sc)
    la = torch.randint(0, 7, (2, 128, 128)); lb = torch.where(la > 0, torch.randint(1, 7, la.shape), 0)
    loss, logs = step((x1, x2, la, lb), 1)
    assert torch.isfinite(loss) and set(logs) == {"loss_seg", "loss_bn", "loss_sc"}
    loss.backward()
    assert all(p.grad is not None for p in model.classifierA.parameters())
    # micro-batch without any changed pixel: upstream NLLLoss would be NaN
    model.zero_grad()
    z = torch.zeros(2, 128, 128, dtype=torch.long)
    loss0, logs0 = step((x1, x2, z, z), 2)
    assert torch.isfinite(loss0) and logs0["loss_seg"] == 0.0


def test_block_checkpointing_is_exact():
    import torch

    torch.manual_seed(1)
    x1, x2 = torch.randn(2, 3, 64, 64), torch.randn(2, 3, 64, 64)
    grads, losses = [], []
    for level in ("none", "full"):
        model, info = _small_model(droppath=0.3, checkpointing=level, seed=0)
        if level == "full":
            assert info["checkpointing"]["vit_blocks"] == 12
        model.output_size = 64
        model.train()
        torch.manual_seed(123)                   # same drop-path draws in both runs
        ch, a, b = model(x1, x2)
        loss = ch.mean() + a.square().mean() + b.abs().mean()
        loss.backward()
        losses.append(float(loss))
        grads.append(model.backbone.blocks[3].attn.qkv.weight.grad.clone())
    assert abs(losses[0] - losses[1]) < 1e-6
    assert torch.allclose(grads[0], grads[1], atol=1e-6)


def test_load_pretrained_checks_every_tensor():
    import torch

    src, _ = _small_model(seed=1)
    vit = {k: v for k, v in src.backbone.state_dict().items()
           if not k.startswith(("spm.", "interactions.", "level_embed", "up.", "norm1.", "norm2.",
                                "norm3.", "norm4."))}
    ck = {"model": {**{"teacher.backbone." + k: v for k, v in vit.items()},
                    **{"student.backbone." + k: torch.zeros_like(v) for k, v in vit.items()},
                    "pred_head.w": torch.zeros(3)}}
    with tempfile.TemporaryDirectory() as d:
        good = Path(d) / "pera.params"
        torch.save(ck, good)
        dst, _ = _small_model(seed=2)
        rep = P.load_pretrained(dst, good)
        assert rep["loaded"] == len(vit)
        assert torch.equal(dst.backbone.blocks[5].mlp.fc1.weight, src.backbone.blocks[5].mlp.fc1.weight)
        bad = Path(d) / "bad.params"
        k0 = "teacher.backbone.blocks.0.attn.qkv.weight"
        torch.save({"model": {**ck["model"], k0: torch.zeros(3, 3)}}, bad)
        try:
            P.load_pretrained(dst, bad)
            raise AssertionError("shape mismatch not detected")
        except RuntimeError:
            pass
        partial = Path(d) / "partial.params"
        torch.save({"model": {k: v for k, v in ck["model"].items() if "blocks.7." not in k}}, partial)
        try:
            P.load_pretrained(dst, partial)
            raise AssertionError("missing ViT tensors not detected")
        except RuntimeError:
            pass


def test_released_cagm_rename():
    import torch

    target = {"decoder.blocks.0.cagm.conv_local.weight": torch.zeros(2, 32, 1, 1)}
    state = {"decoder.blocks.0.cagm.conv2.weight": torch.ones(2, 32, 1, 1), "other.conv2.weight": torch.ones(1)}
    out, renamed = P.rename_cagm_keys(state, target)
    assert "decoder.blocks.0.cagm.conv_local.weight" in out and "other.conv2.weight" in out and len(renamed) == 1


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
