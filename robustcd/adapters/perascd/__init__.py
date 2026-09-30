"""Adapter that runs PerASCD (SathShen/PerASCD, ``legacy`` branch) under the robustcd protocol.

PerASCD is a siamese ViT-G/16 (1024-d, 40 blocks) initialised from the PerA
remote-sensing foundation model, wrapped in a ViT-Adapter (spatial prior
module + multi-scale deformable-attention interactions), with a cascade gated
decoder, two 7-way semantic heads and a 1-channel change head (548 M
parameters). The code lives in the paper's ``legacy`` branch, pinned at
``PERASCD_COMMIT``; its top-level packages are ``models`` / ``utils`` /
``datasets``, as in the Ding repositories, so one process activates one
upstream checkout only.

Conventions checked against the pinned code:

- input normalisation: PerA statistics on [0, 1] RGB, the same for both dates
  (``datasets/RS_ST.py`` ``DataPerAAUG``);
- the ViT is built at ``input_size=448`` (pretrained ``pos_embed`` is 1 + 28 x 28)
  and interpolates its position embedding to 512 tiles; outputs are
  bilinearly upsampled to ``output_size=512``;
- change map ``sigmoid(logit) > 0.5``; semantic class 0 is ignored by the loss;
- the **released** SECOND checkpoint predicts in ChangeMamba class order
  (``RS_ST.ST_COLORMAP`` equals ChangeMamba's palette; the csf-mamba teacher
  check found the identity to be the optimal class assignment). Models trained
  here use robustcd order, so ``decode`` takes the order as an argument.

Two implementation substitutions, both computing the same function:

- **attention**: upstream's ``MemEffAttention`` calls xformers
  ``memory_efficient_attention`` (upstream ``requirements.txt`` pins
  xformers 0.0.32) and, without xformers, falls back to an explicit N x N
  attention that also stores every attention map in eval mode. The robustcd GPU
  env has no xformers, so ``use_sdpa_attention`` routes it to
  ``torch.nn.functional.scaled_dot_product_attention``. The stored maps are
  only read by visualisation code;
- **deformable attention**: the ``MultiScaleDeformableAttention`` CUDA op is
  compiled from the pinned sources for the cluster GPU; on CPU (tests) the
  upstream PyTorch reference ``ms_deform_attn_core_pytorch`` is used.
"""

from __future__ import annotations

import os
import re
import sys
import types
from pathlib import Path
from typing import Optional, Tuple

import numpy as np

PERASCD_URL = "https://github.com/SathShen/PerASCD.git"
PERASCD_BRANCH = "legacy"
PERASCD_COMMIT = "a4d808a6cfb5df7efeee186730ac26b4504c9ed6"
ARCHS = ("ViT-G/16/1024", "ViT-B/16")
NUM_CLASSES = 7

#: PerA statistics (RGB on [0, 1]), legacy/datasets/RS_ST.py DataPerAAUG
PERA_MEAN = np.array([0.3585, 0.3741, 0.3155], dtype=np.float64)
PERA_STD = np.array([0.1483, 0.1283, 0.1198], dtype=np.float64)

#: PerA ViT-G/16-1024 foundation-model weights (github.com/SathShen/PerA README, Google Drive).
#: The archive inside is ``pera_ViTGall22601_ep42_auto``: the file named by
#: ``pretrained_pera_path`` in legacy ``train.py``, i.e. the initialisation of the
#: released PerASCD. 997 tensors; the 486 ``teacher.backbone.*`` ones are loaded
#: (upstream ``load_dict_to_backbone``, non-distilled branch).
PRETRAINED = {"file": "pera_ViTG161024.params", "drive_id": "1YhmDNLyyqbIfRFCfwZTGyyqYMQncamPE",
              "bytes": 9097269032, "archive": "pera_ViTGall22601_ep42_auto", "prefix": "teacher.backbone.",
              "n_tensors": 486}

#: released SECOND checkpoint (Hugging Face SathShen/PerASCD-Checkpoint) and its TensorBoard
#: values at epoch 40 (upstream selects on the SECOND test split by Fscd)
RELEASED = {"zip": "PerASCD_260128115444_vitg01min0Clip15LsscTau001.zip",
            "url": "https://huggingface.co/SathShen/PerASCD-Checkpoint/resolve/main/"
                   "PerASCD_260128115444_vitg01min0Clip15LsscTau001.zip",
            "file": "PerAChain_40e_mIoU74.33_Sek26.11_Fscd66.41_OA88.70.pth",
            "published": {"SeK": 0.2611, "Fscd": 0.6641, "mIoU": 0.7433}}

_ACTIVE: Optional[str] = None
CAGM_KEY = re.compile(r"^(.*\.cagm\.)conv2\.(weight|bias)$")


def check_commit(repo_root: str | os.PathLike) -> Optional[str]:
    head = Path(repo_root) / ".git" / "HEAD"
    try:
        ref = head.read_text().strip()
        if ref.startswith("ref:"):
            return (Path(repo_root) / ".git" / ref.split()[1]).read_text().strip()
        return ref
    except OSError:
        return None


class _MSDA:
    """Stands in for upstream ``MSDeformAttnFunction``: compiled CUDA op, or the PyTorch reference."""

    cuda_fn = None
    core_pytorch = None
    use_pytorch = True

    @classmethod
    def apply(cls, value, shapes, level_start, sampling_locations, attention_weights, im2col_step):
        if cls.use_pytorch or cls.cuda_fn is None or not value.is_cuda:
            shapes_list = [(int(h), int(w)) for h, w in shapes.tolist()]
            return cls.core_pytorch(value, shapes_list, sampling_locations, attention_weights)
        return cls.cuda_fn.apply(value, shapes, level_start, sampling_locations, attention_weights, im2col_step)


def activate(repo_root: str | os.PathLike, msda: str = "auto") -> dict:
    """Put the pinned checkout first on ``sys.path`` and select the deformable-attention backend.

    ``msda``: ``cuda`` (compiled op required), ``pytorch`` (reference, CPU tests) or ``auto``.
    Returns ``{"msda": backend, "msda_file": path or None}``.
    """
    global _ACTIVE
    root = str(Path(repo_root).resolve())
    if _ACTIVE is not None:
        if _ACTIVE != root:
            raise RuntimeError(f"upstream checkout {_ACTIVE} already active; cannot also activate {root}")
    else:
        for name in [m for m in sys.modules if m.split(".")[0] in ("models", "utils")]:
            mod_file = getattr(sys.modules[name], "__file__", None) or ""
            if mod_file and not mod_file.startswith(root):
                raise RuntimeError(f"module {name} already imported from {mod_file}; use a fresh process")
        sys.path.insert(0, root)
        _ACTIVE = root
    try:
        import MultiScaleDeformableAttention as _ext  # noqa: F401
        compiled = not getattr(_ext, "_robustcd_stub", False)
        ext_file = getattr(_ext, "__file__", None) if compiled else None
    except ImportError:
        compiled, ext_file = False, None
    if not compiled:
        if msda == "cuda":
            raise ImportError("MultiScaleDeformableAttention is not compiled (msda='cuda')")
        # upstream ms_deform_attn_func imports the extension at module load
        stub = types.ModuleType("MultiScaleDeformableAttention")
        stub._robustcd_stub = True
        sys.modules.setdefault("MultiScaleDeformableAttention", stub)
    from models.pera_layers.vit_adapter_layers.ops.functions import ms_deform_attn_func as fmod
    from models.pera_layers.vit_adapter_layers.ops.modules import ms_deform_attn as mmod

    _MSDA.cuda_fn = fmod.MSDeformAttnFunction if compiled else None
    _MSDA.core_pytorch = fmod.ms_deform_attn_core_pytorch
    _MSDA.use_pytorch = msda == "pytorch" or not compiled
    mmod.MSDeformAttnFunction = _MSDA
    return {"msda": "pytorch" if _MSDA.use_pytorch else "cuda", "msda_file": ext_file}


def use_sdpa_attention() -> str:
    """Route upstream ``MemEffAttention`` to PyTorch SDPA when xformers is absent (see module doc)."""
    import torch.nn.functional as F
    from models.pera_layers import attention as att

    if att.XFORMERS_AVAILABLE:
        return "xformers"

    def forward(self, x, attn_bias=None):
        assert attn_bias is None, "nested tensors need xformers"
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        p = self.attn_drop.p if self.training else 0.0
        x = F.scaled_dot_product_attention(q, k, v, dropout_p=p)   # default scale = head_dim ** -0.5
        x = x.transpose(1, 2).reshape(B, N, C)
        return self.proj_drop(self.proj(x))

    if not hasattr(att.MemEffAttention, "upstream_forward"):
        att.MemEffAttention.upstream_forward = att.MemEffAttention.forward   # kept for tests
    att.MemEffAttention.forward = forward
    return "sdpa"


def build_model(repo_root: str | os.PathLike, arch: str = "ViT-G/16/1024", droppath: float = 0.0,
                checkpointing: str = "none", msda: str = "auto", sdpa: bool = True):
    """Upstream ``PerASCD`` as in legacy ``train.py``, without pretrained weights.

    Upstream loads the foundation weights with ``strict=False`` and prints a
    success line whatever matched, so they are loaded separately with
    ``load_pretrained``, which checks every tensor. ``checkpointing``: see
    ``set_checkpointing``.
    """
    if arch not in ARCHS:
        raise ValueError(f"unknown arch {arch!r}")
    info = activate(repo_root, msda)
    info["attention"] = use_sdpa_attention() if sdpa else "upstream"
    from models.PerAChain import PerASCD

    model = PerASCD(in_channels=3, num_classes=NUM_CLASSES, input_size=448, output_size=512, arch=arch,
                    droppath=droppath, pretrained_pera_path=None, is_distilled_pera=False,
                    is_freeze_backbone=False)
    info["checkpointing"] = set_checkpointing(model, checkpointing)
    return model, info


CHECKPOINTING = ("none", "adapter", "full")


def set_checkpointing(model, level: str) -> dict:
    """Activation checkpointing: ``none``; ``adapter`` = upstream ``with_cp`` (injectors and
    extractors of the ViT-Adapter interactions only); ``full`` = ``adapter`` plus every ViT
    block (robustcd addition: the 40 blocks hold most activations). Recomputation replays
    the same values, drop-path draws included (``torch.utils.checkpoint`` restores the RNG
    state), and the state-dict keys are unchanged.
    """
    from torch.utils.checkpoint import checkpoint

    if level not in CHECKPOINTING:
        raise ValueError(f"checkpointing must be one of {CHECKPOINTING}")
    n_adapter = 0
    for m in model.backbone.interactions.modules():
        if hasattr(m, "with_cp"):
            m.with_cp = level != "none"
            n_adapter += 1
    n_blocks = 0
    for blk in model.backbone.blocks:
        if "forward" in blk.__dict__:          # undo a previous wrap
            del blk.forward
        if level == "full":
            plain = blk.forward

            def fwd(x, _plain=plain, _blk=blk):
                if _blk.training and x.requires_grad:
                    return checkpoint(_plain, x, use_reentrant=False)
                return _plain(x)
            blk.forward = fwd
            n_blocks += 1
    return {"level": level, "adapter_modules": n_adapter, "vit_blocks": n_blocks}


def load_pretrained(model, path: str | os.PathLike, prefix: str = PRETRAINED["prefix"]) -> dict:
    """Load the PerA foundation weights into ``model.backbone`` and verify them.

    Same selection as upstream (``teacher.backbone.*`` of ``["model"]``), but every
    selected tensor must exist in the backbone with the same shape, and the only
    backbone tensors left at their initialisation must belong to the adapter
    (spatial prior module, interactions, level embedding, output norms/up-sampler).
    """
    import torch

    ck = torch.load(path, map_location="cpu", weights_only=False)
    sd = {k[len(prefix):]: v for k, v in ck["model"].items() if k.startswith(prefix)}
    del ck
    own = model.backbone.state_dict()
    unexpected = sorted(k for k in sd if k not in own)
    mismatch = sorted(k for k in sd if k in own and own[k].shape != sd[k].shape)
    if unexpected or mismatch or not sd:
        raise RuntimeError(f"pretrained {path}: {len(sd)} tensors, unexpected {unexpected[:5]}, "
                           f"shape mismatch {mismatch[:5]}")
    res = model.backbone.load_state_dict(sd, strict=False)
    adapter = ("spm.", "interactions.", "level_embed", "up.", "norm1.", "norm2.", "norm3.", "norm4.")
    foreign = [k for k in res.missing_keys if not k.startswith(adapter)]
    if foreign:
        raise RuntimeError(f"ViT tensors missing from {path}: {foreign[:5]}")
    return {"loaded": len(sd), "adapter_initialised": len(res.missing_keys)}


def rename_cagm_keys(state: dict, target: dict) -> Tuple[dict, list]:
    """``cagm.conv2`` (released checkpoint) -> ``cagm.conv_local`` (pinned code).

    The legacy branch has two copies of ``ChangeAwareGatingModule``: ``Encoders.py``
    names the local 1x1 convolution ``conv2``, ``PerAChain.py`` ``conv_local``; the
    released checkpoint uses the first name, the model the second. Same
    computation on same-shape tensors (checked in csf-mamba, eval_perascd.py).
    """
    out, renamed = {}, []
    for k, v in state.items():
        m = CAGM_KEY.match(k)
        if m:
            new = f"{m.group(1)}conv_local.{m.group(2)}"
            if new in target and new not in state:
                assert tuple(target[new].shape) == tuple(v.shape), (k, v.shape, target[new].shape)
                out[new] = v
                renamed.append(f"{k} -> {new}")
                continue
        out[k] = v
    return out, renamed


def load_weights(model, path: str | os.PathLike, strict: bool = True) -> dict:
    """Load the released upstream checkpoint ({"model": ...}, CAGM rename) or a robustcd one."""
    import torch

    obj = torch.load(path, map_location="cpu", weights_only=False)
    sd = obj["model"] if isinstance(obj, dict) and isinstance(obj.get("model"), dict) else obj
    sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
    sd, renamed = rename_cagm_keys(sd, model.state_dict())
    res = model.load_state_dict(sd, strict=strict)
    meta = {k: float(obj[k]) for k in ("Fscd", "Sek", "mIoU") if isinstance(obj, dict) and k in obj}
    return {"loaded_keys": len(sd), "renamed_keys": renamed, "missing_keys": list(res.missing_keys),
            "unexpected_keys": list(res.unexpected_keys), "checkpoint_meta": meta,
            "epoch": obj.get("epoch") if isinstance(obj, dict) else None}


def normalize_pair(im1: np.ndarray, im2: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    """HxWx3 uint8 pair -> 3xHxW float32 pair: ``(x / 255 - PERA_MEAN) / PERA_STD`` (torchvision to_tensor + Normalize)."""
    def f(im):
        x = (im.astype(np.float32) / 255.0 - PERA_MEAN.astype(np.float32)) / PERA_STD.astype(np.float32)
        return np.ascontiguousarray(x.transpose(2, 0, 1))
    return f(im1), f(im2)


def _cm_to_robustcd():
    from robustcd.adapters.changemamba import CM_TO_SECOND
    return CM_TO_SECOND


def decode(outputs, mode: str = "restricted", order: str = "robustcd"):
    """(change_logit [B,1,H,W], sem_A, sem_B) -> uint8 (sem1, sem2, change) in robustcd order.

    ``restricted`` (protocol): semantic argmax over classes 1..6; ``full``: over all 7
    channels, as upstream's evaluator (used only to reproduce published numbers).
    ``order="cm"`` maps the released checkpoint's ChangeMamba class indices to robustcd.
    """
    import torch

    out_change, out_a, out_b = outputs
    change = (out_change[:, 0] > 0).long()
    if mode == "restricted":
        s1 = torch.argmax(out_a[:, 1:], dim=1) + 1
        s2 = torch.argmax(out_b[:, 1:], dim=1) + 1
    elif mode == "full":
        s1 = torch.argmax(out_a, dim=1)
        s2 = torch.argmax(out_b, dim=1)
    else:
        raise ValueError(f"unknown decode mode {mode!r}")
    s1 = (s1 * change).to(torch.uint8).cpu().numpy()
    s2 = (s2 * change).to(torch.uint8).cpu().numpy()
    if order == "cm":
        lut = _cm_to_robustcd()
        s1, s2 = lut[s1], lut[s2]
    elif order != "robustcd":
        raise ValueError(f"unknown class order {order!r}")
    return s1, s2, change.to(torch.uint8).cpu().numpy()


def decode_released(outputs, mode: str = "restricted"):
    return decode(outputs, mode, order="cm")


def upstream_losses(tau: float = 0.01):
    """(CrossEntropyLoss2d, weighted_BCE_logits, SoftSemanticConsistency(tau)) from the active checkout."""
    if _ACTIVE is None:
        raise RuntimeError("call activate() or build_model() first")
    import importlib

    loss = importlib.import_module("utils.loss")
    return loss.CrossEntropyLoss2d, loss.weighted_BCE_logits, loss.SoftSemanticConsistency(reduction="mean", tau=tau)
