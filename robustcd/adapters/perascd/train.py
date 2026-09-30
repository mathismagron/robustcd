"""Train PerASCD under the robustcd protocol (docs/protocol.md).

    python -m robustcd.adapters.perascd.train --repo ~/ext/PerASCD \\
        --pretrained ~/weights/perascd/pera_ViTG161024.params \\
        --second-root $SLURM_TMPDIR/SECOND --splits ~/robustcd/splits/SECOND \\
        --out $SCRATCH/robustcd/runs/perascd/seed0 --seed 0 --accum 4

Upstream recipe kept (legacy ``train.py`` @ PERASCD_COMMIT, SECOND run ``vitg01m0C15LsscTau001``):

- PerA ViT-G/16-1024 initialisation (``teacher.backbone.*``), backbone fine-tuned
  (``is_freeze_backbone=False``), uniform drop path 0.3;
- loss per micro-batch ``0.5 * (CE_A + CE_B) + wBCE(change) + SoftSemanticConsistency(tau=0.01)``,
  semantic class 0 ignored, the consistency term on classes 1..6;
- SGD, lr 0.1, momentum 0.9, Nesterov, weight decay 1e-5 on all parameters;
- per-iteration schedule: linear warm-up over the first 10 % then poly decay with
  power 1.5 to 0 (``adjust_lr``), over the whole schedule length;
- gradient-norm clipping at 1.5 on all parameters (after unscaling in upstream);
- augmentation: random rot90 (p = 0.5) then one of {none, vertical, horizontal,
  both} flips, identical for both dates and labels (``rand_rot90_flip_SCD``, same
  distribution as the Ding repositories). No date swap.

Protocol changes: colour jitter removed (upstream ``CDMColorJitter`` 0.2/0.2/0.1/0.1);
effective batch 16 (upstream 4 x 2 accumulation = 8); 50k iterations
(upstream 50 epochs); bf16 autocast instead of fp16 + GradScaler; robustcd val
split and SeK-based selection (upstream selects on the test split by Fscd).
Losses are computed in fp32. Attention runs through PyTorch SDPA (see the
package docstring).

The warm-up/poly schedule depends on its total length, so the pilot is a
separate run (``--max-iters 75000 --budget-iters 50000``), as for the Ding models.
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def make_train_step(model, device, criterion, wbce, criterion_sc):
    import torch

    def step(batch, it):
        x1, x2, la, lb = (t.to(device, non_blocking=True) for t in batch)
        labels_bn = (la > 0).unsqueeze(1).float()
        out_change, out_a, out_b = model(x1, x2)
        with torch.autocast(device_type=device.type, enabled=False):
            out_change, out_a, out_b = out_change.float(), out_a.float(), out_b.float()
            # upstream NLLLoss(mean, ignore 0) is NaN when a micro-batch has no changed pixel
            if bool((la > 0).any()):
                loss_seg = 0.5 * (criterion(out_a, la) + criterion(out_b, lb))
            else:
                loss_seg = out_a.sum() * 0.0
            loss_bn = wbce(out_change, labels_bn)
            loss_sc = criterion_sc(out_a[:, 1:], out_b[:, 1:], labels_bn)
            loss = loss_seg + loss_bn + loss_sc
        logs = {"loss_seg": float(loss_seg.detach()), "loss_bn": float(loss_bn.detach()),
                "loss_sc": float(loss_sc.detach())}
        return loss, logs

    return step


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--repo", required=True, help="pinned PerASCD legacy checkout")
    ap.add_argument("--pretrained", default=None, help="pera_ViTG161024.params (protocol: required)")
    ap.add_argument("--arch", default="ViT-G/16/1024", choices=["ViT-G/16/1024", "ViT-B/16"])
    ap.add_argument("--second-root", required=True, help="robustcd SECOND root with train/ (RGB labels)")
    ap.add_argument("--splits", required=True, help="dir with train.txt and val.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-iters", type=int, default=50_000)
    ap.add_argument("--budget-iters", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16, help="effective batch size")
    ap.add_argument("--accum", type=int, default=4)
    ap.add_argument("--lr", type=float, default=0.1)
    ap.add_argument("--momentum", type=float, default=0.9)
    ap.add_argument("--weight-decay", type=float, default=1e-5)
    ap.add_argument("--lr-power", type=float, default=1.5)
    ap.add_argument("--warmup-ratio", type=float, default=0.1)
    ap.add_argument("--min-lr", type=float, default=0.0)
    ap.add_argument("--clip-grad", type=float, default=1.5)
    ap.add_argument("--droppath", type=float, default=0.3)
    ap.add_argument("--tau", type=float, default=0.01, help="SoftSemanticConsistency temperature")
    ap.add_argument("--checkpointing", default="none", choices=["none", "adapter", "full"])
    ap.add_argument("--msda", default="auto", choices=["auto", "cuda", "pytorch"])
    ap.add_argument("--eval-interval", type=int, default=2000)
    ap.add_argument("--eval-batch-size", type=int, default=4)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--stop-after-min", type=float, default=None)
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--val-limit", type=int, default=None, help="debug only")
    ap.add_argument("--train-limit", type=int, default=None, help="debug only")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import torch

    from robustcd.adapters import common
    from robustcd.adapters import perascd as P
    from robustcd.datasets.scd_train import SCDTrainSet
    from robustcd.datasets.second import SecondLike
    from robustcd.training import LoopConfig, run, seed_everything, warmup_poly_lr

    head = P.check_commit(args.repo)
    if head != P.PERASCD_COMMIT:
        print(f"WARNING: {args.repo} is at {head}, expected {P.PERASCD_COMMIT}", flush=True)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    model, info = P.build_model(args.repo, args.arch, droppath=args.droppath, checkpointing=args.checkpointing,
                                msda=args.msda)
    print(f"built PerASCD {args.arch}: {info}", flush=True)
    pre = P.load_pretrained(model, args.pretrained) if args.pretrained else None
    if pre is not None:
        print(f"pretrained: {pre['loaded']} tensors loaded, {pre['adapter_initialised']} adapter tensors at init",
              flush=True)
    elif args.arch == "ViT-G/16/1024":
        print("WARNING: no --pretrained, random ViT init (debug only)", flush=True)
    model.to(device)

    CE, wbce, criterion_sc = P.upstream_losses(args.tau)
    criterion = CE(ignore_index=0).to(device)
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=args.lr,
                                momentum=args.momentum, weight_decay=args.weight_decay, nesterov=True)

    splits = Path(args.splits)
    train_ids = common.read_ids(str(splits / "train.txt"), [])[: args.train_limit]
    val_ids = common.read_ids(str(splits / "val.txt"), [])[: args.val_limit]
    train_root = Path(args.second_root) / "train"
    dataset = SCDTrainSet(train_root, train_ids, P.normalize_pair, augmentation="ding")
    val_reader = SecondLike(train_root)

    def evaluate(m):
        return common.evaluate(m, val_reader, val_ids, device, P.normalize_pair, P.decode, args.eval_batch_size)

    cfg = LoopConfig(out=args.out, seed=args.seed, max_iters=args.max_iters, budget_iters=args.budget_iters,
                     batch_size=args.batch_size, accum=args.accum, eval_interval=args.eval_interval,
                     log_interval=args.log_interval, workers=args.workers, bf16=not args.no_bf16,
                     stop_after_min=args.stop_after_min, clip_grad_norm=args.clip_grad,
                     extra={"adapter": "perascd", "repo": args.repo, "repo_commit": head,
                            "expected_commit": P.PERASCD_COMMIT, "arch": args.arch, "build": info,
                            "pretrained": args.pretrained, "pretrained_load": pre,
                            "recipe": {"lr": args.lr, "momentum": args.momentum, "nesterov": True,
                                       "weight_decay": args.weight_decay, "lr_power": args.lr_power,
                                       "warmup_ratio": args.warmup_ratio, "min_lr": args.min_lr,
                                       "clip_grad": args.clip_grad, "droppath": args.droppath, "tau": args.tau},
                            "colour_jitter": False, "n_train": len(train_ids), "n_val": len(val_ids),
                            "schedule": f"warmup_poly(lr={args.lr}, power={args.lr_power}, "
                                        f"warmup={args.warmup_ratio}, total={args.max_iters})"})
    return run(cfg, model=model, optimizer=optimizer, dataset=dataset,
               train_step=make_train_step(model, device, criterion, wbce, criterion_sc),
               evaluate=evaluate,
               set_lr=warmup_poly_lr(args.lr, args.max_iters, args.lr_power, args.warmup_ratio, args.min_lr),
               device=device)


if __name__ == "__main__":
    raise SystemExit(main())
