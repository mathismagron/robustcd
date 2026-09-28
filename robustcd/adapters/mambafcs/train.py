"""Train Mamba-FCS under the robustcd protocol (docs/protocol.md).

    python -m robustcd.adapters.changemamba.prepare --second-root $SLURM_TMPDIR/SECOND \\
        --splits ~/robustcd/splits/SECOND --out $SLURM_TMPDIR/cm_SECOND
    python -m robustcd.adapters.mambafcs.train --fcs-root ~/ext/MambaFCS \\
        --backbone ~/weights/mambafcs/vssm_base_0229_ckpt_epoch_237.pth \\
        --second-root $SLURM_TMPDIR/SECOND --cm-data $SLURM_TMPDIR/cm_SECOND/train \\
        --splits ~/robustcd/splits/SECOND --out $SCRATCH/robustcd/runs/mambafcs/seed0 --seed 0 --accum 4

Upstream recipe kept (changedetection/script/train_MambaSCD.py, SECOND):

- loss per micro-batch:
  ``1.0 * CE(change) + 0.5 * (CE(t1) + CE(t2)) + 0.5 * (Lovasz(t1) + Lovasz(t2) + Lovasz(change))
  + 0.05 * MSE(softmax t1, softmax t2 on unchanged pixels) + 0.5 * SeK_Loss``.
  Semantic class 0 is set to ignore (255). The SeK term is off only at the
  first iteration (upstream ``itera > 0``);
- AdamW (lr 1e-4, weight decay 5e-4), StepLR gamma 0.5;
- augmentation, from upstream ``SemanticChangeDetectionDatset``, applied
  identically to both dates and all labels: random left-right flip, random
  up-down flip, rotation by 90/180/270 degrees (never 0, as upstream), and
  swap of the two dates with their labels (p = 0.5). The photometric
  jitter is removed (protocol).

Schedule rescaling: upstream trains 800k samples (``max_iters: 800000`` samples
at batch 2, i.e. 400k iterations) with StepLR every 10k iterations, i.e. every
20k samples. The protocol budget is also 800k samples (16 x 50k), so the step
is kept at 20k samples: ``--lr-step 1250`` iterations at batch 16. The learning
rate is therefore below 1e-7 after about 12.5k iterations, as in upstream's own
schedule.

Losses are computed in fp32 outside autocast. The SeK term builds soft
confusion matrices by summing over every changed pixel of the batch, and bf16
accumulation would degrade it.
"""

from __future__ import annotations

import argparse
import os
import random
import sys
from pathlib import Path


class FCSTrainSet:
    """Upstream ``SemanticChangeDetectionDatset`` (train branch) without photometric jitter.

    Reads the ChangeMamba layout written by ``changemamba.prepare`` and applies
    upstream's own augmentation functions (``datasets/imutils.py``) in
    upstream's order. The date swap uses ``random`` rather than
    ``np.random``: the draw is identical in distribution, and ``random`` is
    reseeded per DataLoader worker by PyTorch, whereas NumPy's global state
    is copied unchanged into every forked worker.
    """

    def __init__(self, cm_data: str, ids, imutils, debug_crop: int = 0):
        self.root = Path(cm_data)
        self.ids = list(ids)
        self.imutils = imutils
        self.debug_crop = debug_crop      # CPU smoke tests only; the protocol trains on full 512 tiles

    def __len__(self):
        return len(self.ids)

    def _load(self, sub, sid):
        import numpy as np
        from PIL import Image

        return np.asarray(Image.open(self.root / sub / f"{sid}.png"), dtype=np.float32)

    def __getitem__(self, i):
        import numpy as np
        import torch

        sid = self.ids[i]
        pre, post = self._load("T1", sid), self._load("T2", sid)
        t1, t2 = self._load("GT_T1", sid), self._load("GT_T2", sid)
        cd = self._load("GT_CD", sid) / 255
        if self.debug_crop:
            c = self.debug_crop
            pre, post, t1, t2, cd = (a[:c, :c] for a in (pre, post, t1, t2, cd))
        u = self.imutils
        pre, post, cd, t1, t2 = u.random_fliplr_mcd(pre, post, cd, t1, t2)
        pre, post, cd, t1, t2 = u.random_flipud_mcd(pre, post, cd, t1, t2)
        pre, post, cd, t1, t2 = u.random_rot_mcd(pre, post, cd, t1, t2)
        if random.random() < 0.5:
            pre, post = post, pre
            t1, t2 = t2, t1
        # (upstream: random_photometric_imgs on both dates -- removed by the protocol)
        pre = np.transpose(u.normalize_img(pre), (2, 0, 1))
        post = np.transpose(u.normalize_img(post), (2, 0, 1))
        c = lambda a, dt: torch.from_numpy(np.ascontiguousarray(a).astype(dt))  # noqa: E731
        return c(pre, np.float32), c(post, np.float32), c(cd, np.int64), c(t1, np.int64), c(t2, np.int64)


def make_train_step(model, device, sek_criterion, lovasz_softmax):
    import torch
    import torch.nn.functional as F

    def step(batch, it):
        pre, post, label_cd, t1, t2 = (t.to(device, non_blocking=True) for t in batch)
        change_mask = (label_cd != 0).float()
        t1 = t1.clone()
        t2 = t2.clone()
        t1[t1 == 0] = 255
        t2[t2 == 0] = 255
        out_cd, out_t1, out_t2 = model(pre, post)
        with torch.autocast(device_type=device.type, enabled=False):
            out_cd, out_t1, out_t2 = out_cd.float(), out_t1.float(), out_t2.float()
            # mean CE over non-ignored pixels, as upstream, but 0 instead of NaN when a
            # micro-batch has no changed pixel at all (every semantic target ignored)
            def ce(logits, target):
                n = (target != 255).sum().clamp(min=1)
                return F.cross_entropy(logits, target, ignore_index=255, reduction="sum") / n
            ce_cd = ce(out_cd, label_cd)
            ce_t1 = ce(out_t1, t1)
            ce_t2 = ce(out_t2, t2)
            p_cd, p_t1, p_t2 = (F.softmax(o, dim=1) for o in (out_cd, out_t1, out_t2))
            # upstream lovasz_softmax returns an empty tensor (not 0) when every pixel is ignored,
            # e.g. semantic terms of a micro-batch without any changed pixel; .sum() makes it 0
            one = lambda t: t if t.numel() == 1 else t.sum()  # noqa: E731
            lz_cd = one(lovasz_softmax(p_cd, label_cd, ignore=255))
            lz_t1 = one(lovasz_softmax(p_t1, t1, ignore=255))
            lz_t2 = one(lovasz_softmax(p_t2, t2, ignore=255))
            sim_mask = (t1 == 255).float().unsqueeze(1)
            sim = F.mse_loss(p_t1 * sim_mask, p_t2 * sim_mask, reduction="mean")
            sek = sek_criterion(out_t1, out_t2, t1, t2, change_mask)
            # Upstream SeK_Loss takes log(kappa * exp(1.5 * mIoU) + eps). The soft kappa is
            # negative for some batches while the heads are near chance (checked: 1 in 4 random
            # inits on real SECOND tiles), which gives NaN, and a NaN loss would poison AdamW.
            # Such micro-batches drop the SeK term; every other term and every other batch is unchanged.
            sek_ok = bool(torch.isfinite(sek))
            w_sek = 0.5 if (it > 1 and sek_ok) else 0.0   # upstream: itera + start_iter > 0 (0-based)
            loss = 1.0 * ce_cd + 0.5 * (ce_t1 + ce_t2) + 0.5 * (lz_t1 + lz_t2 + lz_cd) + 0.05 * sim
            if w_sek:
                loss = loss + w_sek * sek
        logs = {"ce_cd": float(ce_cd.detach()), "ce_sem": float((ce_t1 + ce_t2).detach()),
                "lovasz": float((lz_t1 + lz_t2 + lz_cd).detach()), "sim": float(sim.detach()),
                "sek_loss": float(sek.detach()) if sek_ok else 0.0,
                "sek_skipped": 0.0 if sek_ok else 1.0}
        return loss, logs

    return step


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--fcs-root", required=True, help="pinned checkout, directory named MambaFCS")
    ap.add_argument("--cfg", default=None, help="default: <fcs-root>/changedetection/configs/vssm1/vssm_base_224.yaml")
    ap.add_argument("--backbone", default=None, help="ImageNet vssm_base_0229_ckpt_epoch_237.pth (protocol: required)")
    ap.add_argument("--second-root", required=True, help="robustcd SECOND root with train/ (RGB labels, for val)")
    ap.add_argument("--cm-data", required=True, help="ChangeMamba-layout train dir (changemamba.prepare output)")
    ap.add_argument("--splits", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-iters", type=int, default=50_000)
    ap.add_argument("--budget-iters", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-4)
    ap.add_argument("--lr-step", type=int, default=1250, help="StepLR step in iterations (20k samples at batch 16)")
    ap.add_argument("--eval-interval", type=int, default=2000)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--stop-after-min", type=float, default=None)
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--val-limit", type=int, default=None, help="debug only")
    ap.add_argument("--train-limit", type=int, default=None, help="debug only")
    ap.add_argument("--debug-crop", type=int, default=0, help="debug only: top-left crop of training tiles")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import torch

    from robustcd.adapters import common
    from robustcd.adapters import mambafcs as fcs
    from robustcd.datasets.second import SecondLike
    from robustcd.training import LoopConfig, run, seed_everything

    head = fcs.check_commit(args.fcs_root)
    if head != fcs.FCS_COMMIT:
        print(f"WARNING: {args.fcs_root} is at {head}, expected {fcs.FCS_COMMIT}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    model, _ = fcs.build_model(args.fcs_root, args.cfg)
    bb = fcs.load_backbone(model, args.backbone) if args.backbone else None
    if bb is not None:
        print(f"backbone: {bb['matched']} tensors loaded, {len(bb['unexpected'])} unexpected "
              f"(e.g. {bb['unexpected'][:2]}), {len(bb['missing'])} missing (e.g. {bb['missing'][:2]})", flush=True)
    model.to(device)

    import MambaFCS.changedetection.datasets.imutils as imutils
    import MambaFCS.changedetection.utils_func.lovasz_loss as L
    from MambaFCS.changedetection.utils_func.loss import SeK_Loss

    sek_criterion = SeK_Loss(num_classes=7, non_change_class=0, beta=1.5).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.lr_step, gamma=0.5)

    splits = Path(args.splits)
    train_ids = common.read_ids(str(splits / "train.txt"), [])[: args.train_limit]
    val_ids = common.read_ids(str(splits / "val.txt"), [])[: args.val_limit]
    dataset = FCSTrainSet(args.cm_data, train_ids, imutils, args.debug_crop)
    val_reader = SecondLike(Path(args.second_root) / "train")

    def evaluate(m):
        return common.evaluate(m, val_reader, val_ids, device, fcs.normalize_pair, fcs.decode_outputs,
                               args.eval_batch_size)

    cfg = LoopConfig(out=args.out, seed=args.seed, max_iters=args.max_iters, budget_iters=args.budget_iters,
                     batch_size=args.batch_size, accum=args.accum, eval_interval=args.eval_interval,
                     log_interval=args.log_interval, workers=args.workers, bf16=not args.no_bf16,
                     stop_after_min=args.stop_after_min,
                     extra={"adapter": "mambafcs", "fcs_root": args.fcs_root, "fcs_commit": head,
                            "expected_commit": fcs.FCS_COMMIT, "backbone": args.backbone,
                            "backbone_load": None if bb is None else {k: (v if isinstance(v, int) else len(v))
                                                                      for k, v in bb.items()},
                            "lr": args.lr, "weight_decay": args.weight_decay, "lr_step_iters": args.lr_step,
                            "photometric_aug": False, "debug_crop": args.debug_crop, "n_train": len(train_ids), "n_val": len(val_ids)})
    return run(cfg, model=model, optimizer=optimizer, dataset=dataset,
               train_step=make_train_step(model, device, sek_criterion, L.lovasz_softmax),
               evaluate=evaluate, scheduler=scheduler, device=device)


if __name__ == "__main__":
    raise SystemExit(main())
