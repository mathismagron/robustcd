#!/usr/bin/env python3
"""GPU environment witness: does the training stack actually work on this node?

Import success proves little -- a wheel can import and still ship kernels built
for the wrong GPU architecture, or be ABI-incompatible with the installed torch.
Each check below runs a kernel and, where a pure-PyTorch reference exists,
compares against it numerically:

1. device      torch sees the GPU, capability, bf16 support, a bf16 matmul
2. causal_conv1d   CUDA kernel vs ``causal_conv1d_ref``
3. selective_scan  mamba_ssm CUDA kernel vs ``selective_scan_ref`` (forward + grads)
4. mamba_block     ``mamba_ssm.Mamba`` forward/backward in bf16 autocast, timing
5. data        read a batch of SECOND from the staged copy, timing

Writes a JSON report; exit code 1 if any check fails.

    python scripts/gpu_smoke_test.py --data $SLURM_TMPDIR/SECOND --out gpu_smoke.json
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
import time
import traceback
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

RESULTS: dict = {}


def check(name):
    def deco(fn):
        def run(*a, **k):
            t0 = time.time()
            try:
                info = fn(*a, **k) or {}
                ok = info.pop("_ok", True)
                RESULTS[name] = {"ok": bool(ok), **info}
            except Exception as e:  # noqa: BLE001
                RESULTS[name] = {"ok": False, "error": f"{type(e).__name__}: {e}",
                                 "trace": traceback.format_exc(limit=4)}
            RESULTS[name]["seconds"] = round(time.time() - t0, 2)
            status = "PASS" if RESULTS[name]["ok"] else "FAIL"
            detail = {k: v for k, v in RESULTS[name].items() if k not in ("ok", "trace")}
            print(f"{status} {name}: {detail}", flush=True)
        return run
    return deco


def _rel(a, b):
    import torch
    return float((a.float() - b.float()).abs().max() / (b.float().abs().max() + 1e-6))


@check("device")
def t_device():
    import torch
    assert torch.cuda.is_available(), "torch.cuda.is_available() is False"
    d = torch.device("cuda")
    x = torch.randn(4096, 4096, device=d, dtype=torch.bfloat16)
    for _ in range(3):  # warm-up: cuBLAS handle creation and kernel selection happen on first calls
        y = x @ x
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(50):
        y = x @ x
    torch.cuda.synchronize()
    tflops = 50 * 2 * 4096**3 / (time.time() - t0) / 1e12
    return {
        "torch": torch.__version__, "cuda_build": torch.version.cuda,
        "cudnn": torch.backends.cudnn.version(), "device": torch.cuda.get_device_name(0),
        "capability": list(torch.cuda.get_device_capability(0)),
        "n_visible_gpus": torch.cuda.device_count(),
        "mem_total_gb": round(torch.cuda.get_device_properties(0).total_memory / 2**30, 1),
        "bf16_supported": torch.cuda.is_bf16_supported(),
        "bf16_matmul_tflops": round(tflops, 1), "finite": bool(torch.isfinite(y).all()),
    }


@check("causal_conv1d")
def t_causal_conv1d():
    import torch
    import causal_conv1d
    from causal_conv1d import causal_conv1d_fn
    from causal_conv1d.causal_conv1d_interface import causal_conv1d_ref
    torch.manual_seed(0)
    x = torch.randn(2, 64, 512, device="cuda")
    w = torch.randn(64, 4, device="cuda")
    b = torch.randn(64, device="cuda")
    out = causal_conv1d_fn(x, w, b, activation="silu")
    ref = causal_conv1d_ref(x, w, b, activation="silu")
    rel = _rel(out, ref)
    return {"version": causal_conv1d.__version__, "max_rel_err": rel, "_ok": rel < 1e-3}


@check("selective_scan")
def t_selective_scan():
    import torch
    from mamba_ssm.ops.selective_scan_interface import selective_scan_fn, selective_scan_ref
    torch.manual_seed(0)
    b, d, n, L = 2, 64, 16, 256
    kw = dict(device="cuda", dtype=torch.float32)
    u = torch.randn(b, d, L, **kw).requires_grad_()
    delta = (0.5 * torch.rand(b, d, L, **kw)).requires_grad_()
    A = (-torch.rand(d, n, **kw)).requires_grad_()
    B = torch.randn(b, n, L, **kw).requires_grad_()
    C = torch.randn(b, n, L, **kw).requires_grad_()
    D = torch.randn(d, **kw).requires_grad_()
    z = torch.randn(b, d, L, **kw).requires_grad_()
    args = (u, delta, A, B, C, D, z)
    out = selective_scan_fn(*args, delta_softplus=True)
    g = torch.randn_like(out)
    grads = torch.autograd.grad(out, args, g)
    ref = selective_scan_ref(*args, delta_softplus=True)
    grads_ref = torch.autograd.grad(ref, args, g)
    fwd = _rel(out, ref)
    bwd = max(_rel(a, r) for a, r in zip(grads, grads_ref))
    return {"fwd_max_rel_err": fwd, "bwd_max_rel_err": bwd, "_ok": fwd < 1e-3 and bwd < 1e-2}


@check("mamba_block")
def t_mamba_block():
    import torch
    import mamba_ssm
    from mamba_ssm import Mamba
    torch.manual_seed(0)
    m = Mamba(d_model=256, d_state=16, d_conv=4, expand=2).cuda()
    x = torch.randn(8, 4096, 256, device="cuda", requires_grad=True)   # 64x64 tokens, e.g. a 1/8 feature map of 512 px
    with torch.autocast("cuda", dtype=torch.bfloat16):
        y = m(x)
    y.float().mean().backward()
    torch.cuda.synchronize()
    t0 = time.time()
    for _ in range(10):
        with torch.autocast("cuda", dtype=torch.bfloat16):
            y = m(x)
        y.float().mean().backward()
    torch.cuda.synchronize()
    ms = (time.time() - t0) / 10 * 1000
    return {"version": mamba_ssm.__version__, "out_shape": list(y.shape),
            "fwd_bwd_ms": round(ms, 2), "peak_mem_gb": round(torch.cuda.max_memory_allocated() / 2**30, 2),
            "finite": bool(torch.isfinite(y).all() and torch.isfinite(x.grad).all()),
            "_ok": bool(torch.isfinite(y).all() and torch.isfinite(x.grad).all())}


@check("data")
def t_data(root: str, n: int = 64):
    from robustcd.datasets.second import SecondLike
    ds = SecondLike(Path(root) / "train")
    t0 = time.time()
    for sid in ds.ids[:n]:
        s = ds.get(sid)
    dt = time.time() - t0
    return {"root": str(root), "n_train": len(ds), "samples_read": n,
            "ms_per_sample": round(1000 * dt / n, 1), "shape": list(s["im1"].shape),
            "_ok": len(ds) == 2968}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default=None, help="staged SECOND root containing train/ and test/")
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    RESULTS["host"] = {"node": platform.node(), "python": platform.python_version()}
    t_device()
    t_causal_conv1d()
    t_selective_scan()
    t_mamba_block()
    if args.data:
        t_data(args.data)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(RESULTS, indent=2))
    failed = [k for k, v in RESULTS.items() if isinstance(v, dict) and v.get("ok") is False]
    print("ALL PASS" if not failed else f"FAILED: {failed}")
    return 1 if failed else 0


if __name__ == "__main__":
    raise SystemExit(main())
