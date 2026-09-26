"""Train a Ding-codebase model (SCanNet, TED, Bi-SRNet, SSCD-l, HRSCD-str4) under the protocol.

    python -m robustcd.adapters.ding.train --model scannet --repo ~/ext/SCanNet \\
        --second-root $SLURM_TMPDIR/SECOND --splits ~/robustcd/splits/SECOND \\
        --out $SCRATCH/robustcd/runs/scannet/seed0 --seed 0

Upstream recipe (kept): SGD (lr 0.1, momentum 0.9, Nesterov, weight decay
5e-4 on all parameters), per-iteration poly decay with power 1.5 over the
whole schedule, loss 0.5 * (CE_A + CE_B) [class 0 ignored] + weighted BCE on
the change logit [+ ChangeSimilarity] and, for SCanNet/TED, the pseudo-label
stage (``psd.py``). Protocol changes: effective batch 16 (upstream 8),
50k iterations (upstream 50 epochs of 2375 images, about 14.8k iterations),
bf16 autocast, robustcd val split and SeK-based checkpoint selection.

The poly schedule depends on its total length, so a longer run is not a
continuation of a shorter one. The pilot for these models is therefore a
separate run (``--max-iters 75000 --budget-iters 50000``).
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["scannet", "ted", "bisrnet", "sscdl", "hrscd4"])
    ap.add_argument("--repo", required=True, help="pinned upstream checkout (SCanNet or Bi-SRNet, see MODELS)")
    ap.add_argument("--second-root", required=True, help="robustcd SECOND root with train/ (RGB labels)")
    ap.add_argument("--splits", required=True, help="dir with train.txt and val.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-iters", type=int, default=50_000)
    ap.add_argument("--budget-iters", type=int, default=None)
    ap.add_argument("--batch-size", type=int, default=16, help="effective batch size")
    ap.add_argument("--accum", type=int, default=1)
    ap.add_argument("--eval-interval", type=int, default=2000)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--stop-after-min", type=float, default=None)
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--no-psd", action="store_true", help="ablation: disable the pseudo-label stage")
    ap.add_argument("--psd-force", action="store_true",
                    help="timing only: activate the pseudo-label teacher from iteration 1 (worst-case cost)")
    ap.add_argument("--no-imagenet", action="store_true", help="debug only: random encoder init")
    ap.add_argument("--val-limit", type=int, default=None, help="debug only")
    ap.add_argument("--train-limit", type=int, default=None, help="debug only")
    args = ap.parse_args(argv)

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import torch

    from robustcd.adapters import common
    from robustcd.adapters import ding
    from robustcd.adapters.ding.psd import PseudoLabeler
    from robustcd.datasets.scd_train import SCDTrainSet
    from robustcd.datasets.second import SecondLike
    from robustcd.training import LoopConfig, poly_lr, run, seed_everything

    spec = ding.MODELS[args.model]
    rec = spec.recipe
    url, commit = ding.REPOS[spec.repo]
    head = ding.check_commit(args.repo)
    if head != commit:
        print(f"WARNING: {args.repo} is at {head}, expected {commit} ({url})", flush=True)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    seed_everything(args.seed)
    model = ding.build_model(args.model, args.repo, imagenet=not args.no_imagenet).to(device)
    CE, wBCE, ChangeSim = ding.upstream_losses()
    criterion = CE(ignore_index=0).to(device)
    criterion_sc = ChangeSim().to(device) if rec.sc_loss else None
    optimizer = torch.optim.SGD([p for p in model.parameters() if p.requires_grad], lr=rec.lr,
                                momentum=rec.momentum, weight_decay=rec.weight_decay, nesterov=True)
    use_psd = rec.psd and not args.no_psd
    psd = PseudoLabeler(model, ding.NUM_CLASSES, rec.psd_init_fscd, rec.pseudo_thred, rec.psd_tta) if use_psd else None
    if args.psd_force:
        if psd is None:
            raise SystemExit("--psd-force needs a model whose recipe uses pseudo labels")
        psd.on_eval({"Fscd": 1.0}, 0)

    splits = Path(args.splits)
    train_ids = common.read_ids(str(splits / "train.txt"), [])[: args.train_limit]
    val_ids = common.read_ids(str(splits / "val.txt"), [])[: args.val_limit]
    train_root = Path(args.second_root) / "train"
    dataset = SCDTrainSet(train_root, train_ids, ding.normalize_pair, augmentation="ding")
    val_reader = SecondLike(train_root)

    def train_step(batch, it):
        x1, x2, la, lb = (t.to(device, non_blocking=True) for t in batch)
        labels_bn = (la > 0).unsqueeze(1).float()
        if psd is not None:
            la, lb = psd.apply(x1, x2, la, lb, labels_bn)
        out_change, out_a, out_b = model(x1, x2)
        out_change, out_a, out_b = out_change.float(), out_a.float(), out_b.float()
        loss_seg = 0.5 * criterion(out_a, la) + 0.5 * criterion(out_b, lb)
        loss_bn = wBCE(out_change, labels_bn)
        loss = loss_seg + loss_bn
        logs = {"loss_seg": float(loss_seg.detach()), "loss_bn": float(loss_bn.detach())}
        if criterion_sc is not None:
            loss_sc = criterion_sc(out_a[:, 1:], out_b[:, 1:], labels_bn)
            loss = loss + loss_sc
            logs["loss_sc"] = float(loss_sc.detach())
        if psd is not None:
            logs["psd_active"] = float(psd.active)
            logs["psd_pixels"] = float(psd.n_pseudo)
            psd.n_pseudo = 0
        return loss, logs

    def evaluate(m):
        return common.evaluate(m, val_reader, val_ids, device, ding.normalize_pair, ding.decode,
                               args.eval_batch_size)

    def on_eval(scores, it):
        if psd is not None:
            before = psd.teacher_iteration
            psd.on_eval(scores, it)
            if psd.teacher_iteration != before:
                print(f"PSD teacher <- iteration {it} (val Fscd {scores['Fscd']:.4f}, "
                      f"active={psd.active})", flush=True)

    cfg = LoopConfig(out=args.out, seed=args.seed, max_iters=args.max_iters, budget_iters=args.budget_iters,
                     batch_size=args.batch_size, accum=args.accum, eval_interval=args.eval_interval,
                     log_interval=args.log_interval, workers=args.workers, bf16=not args.no_bf16,
                     stop_after_min=args.stop_after_min,
                     extra={"adapter": "ding", "model": args.model, "repo": args.repo, "repo_commit": head,
                            "expected_commit": commit, "recipe": rec.__dict__, "psd": use_psd,
                            "imagenet": not args.no_imagenet, "n_train": len(train_ids), "n_val": len(val_ids),
                            "schedule": f"poly(lr={rec.lr}, power={rec.lr_power}, total={args.max_iters})"})
    return run(cfg, model=model, optimizer=optimizer, dataset=dataset, train_step=train_step, evaluate=evaluate,
               set_lr=poly_lr(rec.lr, args.max_iters, rec.lr_power), extra_state=psd, on_eval=on_eval,
               device=device)


if __name__ == "__main__":
    raise SystemExit(main())
