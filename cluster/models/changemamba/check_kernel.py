#!/usr/bin/env python3
"""Numeric check of VMamba's selective_scan_cuda_oflex against a pure-PyTorch scan.

Covers the VMamba call pattern: grouped B/C of shape (b, k, n, l) with
d = k * c channels, delta_bias, delta_softplus, D, forward and gradients, in
fp32, and a bf16-input forward (as under autocast).
"""

import sys

import torch
import torch.nn.functional as F


def scan_ref(u, delta, A, B, C, D=None, delta_bias=None, delta_softplus=False):
    b, d, l = u.shape
    g = B.shape[1]
    u, delta, A, B, C = u.float(), delta.float(), A.float(), B.float(), C.float()
    if delta_bias is not None:
        delta = delta + delta_bias[..., None].float()
    if delta_softplus:
        delta = F.softplus(delta)
    B = B.repeat_interleave(d // g, dim=1)
    C = C.repeat_interleave(d // g, dim=1)
    dA = torch.exp(delta.unsqueeze(2) * A[None, :, :, None])
    dBu = delta.unsqueeze(2) * B * u.unsqueeze(2)
    x = torch.zeros(b, d, A.shape[1], device=u.device)
    ys = []
    for i in range(l):
        x = dA[..., i] * x + dBu[..., i]
        ys.append((x * C[..., i]).sum(-1))
    y = torch.stack(ys, -1)
    if D is not None:
        y = y + u * D[:, None].float()
    return y


def main() -> int:
    import selective_scan_cuda_oflex  # noqa: F401
    sys.path.insert(0, __import__("os").path.expanduser("~/ext/ChangeMamba"))
    from changedetection.models.vmamba import SelectiveScanOflex

    torch.manual_seed(0)
    b, k, c, n, l = 2, 4, 32, 1, 256  # MambaSCD-Tiny uses d_state = 1
    d = k * c
    kw = dict(device="cuda", dtype=torch.float32)
    u = torch.randn(b, d, l, **kw).requires_grad_()
    delta = (0.5 * torch.randn(b, d, l, **kw)).requires_grad_()
    A = (-torch.rand(d, n, **kw)).requires_grad_()
    B = torch.randn(b, k, n, l, **kw).requires_grad_()
    C = torch.randn(b, k, n, l, **kw).requires_grad_()
    D = torch.randn(d, **kw).requires_grad_()
    bias = torch.randn(d, **kw).requires_grad_()
    args = (u, delta, A, B, C, D, bias)

    out = SelectiveScanOflex.apply(*args, True, 1, 1, True)
    ref = scan_ref(*args, delta_softplus=True)
    g = torch.randn_like(ref)
    grads = torch.autograd.grad(out, args, g)
    grads_ref = torch.autograd.grad(ref, args, g)
    rel = lambda a, r: float((a.float() - r.float()).abs().max() / (r.float().abs().max() + 1e-6))  # noqa: E731
    fwd = rel(out, ref)
    bwd = max(rel(a, r) for a, r in zip(grads, grads_ref))

    with torch.no_grad():
        out16 = SelectiveScanOflex.apply(u.bfloat16(), delta.bfloat16(), A, B.bfloat16(), C.bfloat16(),
                                         D, bias, True, 1, 1, True)
    fwd16 = rel(out16, ref.detach())
    ok = fwd < 1e-4 and bwd < 1e-3 and fwd16 < 5e-2 and torch.isfinite(out16).all()
    print(f"{'PASS' if ok else 'FAIL'} selective_scan_cuda_oflex: fwd {fwd:.2e}  bwd {bwd:.2e}  "
          f"bf16-input fwd {fwd16:.2e}  (device {torch.cuda.get_device_name(0)}, "
          f"capability {torch.cuda.get_device_capability(0)})")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
