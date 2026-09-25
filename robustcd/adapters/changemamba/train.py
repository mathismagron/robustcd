"""Train ChangeMamba (MambaSCD) under the robustcd protocol (docs/protocol.md).

Upstream pieces reused unchanged: ChangeMambaSCD, SCDTrainer.train_step (loss:
CE + Lovasz-softmax on the change and both semantic heads, plus the
semantic-consistency MSE), AdamW, StepLR(10k, 0.5), the SECOND dataset class
with its geometric augmentation and normalisation.

Protocol pieces added: seeded and resumable data order (epoch-wise
permutations), bf16 autocast, gradient accumulation to a fixed effective batch
size, val-only checkpoint selection on robustcd SeK every ``--eval-interval``
iterations, best + latest checkpoints, JSONL logs, and a wall-clock guard that
saves and exits with code 3 so that the SLURM script can resubmit.

    python -m robustcd.adapters.changemamba.train --cm-root ~/ext/ChangeMamba \\
        --cfg ~/ext/ChangeMamba/changedetection/configs/vssm1/vssm_tiny_224_0229flex.yaml \\
        --encoder-pretrained ~/weights/changemamba/vssm_tiny_0230_ckpt_epoch_262.pth \\
        --second-root $SLURM_TMPDIR/SECOND --cm-data $SLURM_TMPDIR/cm_SECOND \\
        --splits ~/robustcd/splits/SECOND --out $SCRATCH/robustcd/runs/changemamba_tiny/seed0 --seed 0
"""

from __future__ import annotations

import argparse
import json
import math
import os
import platform
import random
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np

EXIT_REQUEUE = 3


class EpochOrderSampler:
    """Deterministic epoch-wise permutations, resumable at any sample position.

    Epoch e uses ``torch.randperm(n, generator=seed * 1_000_003 + e)``, so sample
    position p is fully determined by (seed, p): resuming at iteration k
    replays exactly the order an uninterrupted run would have used.
    """

    def __init__(self, n: int, total: int, seed: int, start: int = 0):
        self.n, self.total, self.seed, self.start = n, total, seed, start

    def __iter__(self):
        import torch

        pos = self.start
        epoch, offset = divmod(pos, self.n)
        while pos < self.total:
            g = torch.Generator().manual_seed(self.seed * 1_000_003 + epoch)
            perm = torch.randperm(self.n, generator=g).tolist()
            for i in perm[offset:]:
                if pos >= self.total:
                    return
                yield i
                pos += 1
            epoch, offset = epoch + 1, 0

    def __len__(self):
        return max(0, self.total - self.start)


def seed_everything(seed: int) -> None:
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def _jsonl(path: Path, rec: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cm-root", required=True)
    ap.add_argument("--cfg", required=True)
    ap.add_argument("--encoder-pretrained", default=None, help="ImageNet VMamba backbone (required by the protocol)")
    ap.add_argument("--second-root", required=True, help="robustcd SECOND root with train/ (RGB labels, for val)")
    ap.add_argument("--cm-data", required=True, help="ChangeMamba-layout train dir (prepare.py output)")
    ap.add_argument("--splits", required=True, help="dir with train.txt and val.txt")
    ap.add_argument("--out", required=True)
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--max-iters", type=int, default=50_000)
    ap.add_argument("--budget-iters", type=int, default=None,
                    help="protocol budget (default: --max-iters). best_model.pth is selected among evaluations "
                         "at iterations <= budget; best_model_extended.pth over the whole run (pilot rule)")
    ap.add_argument("--batch-size", type=int, default=16, help="effective batch size")
    ap.add_argument("--accum", type=int, default=1, help="gradient accumulation steps")
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--weight-decay", type=float, default=5e-3)
    ap.add_argument("--eval-interval", type=int, default=2000)
    ap.add_argument("--eval-batch-size", type=int, default=8)
    ap.add_argument("--workers", type=int, default=int(os.environ.get("SLURM_CPUS_PER_TASK", 4)))
    ap.add_argument("--no-bf16", action="store_true")
    ap.add_argument("--stop-after-min", type=float, default=None,
                    help="save and exit with code 3 after this many minutes (for SLURM resubmission)")
    ap.add_argument("--log-interval", type=int, default=50)
    ap.add_argument("--val-limit", type=int, default=None, help="debug only: evaluate on the first N val ids")
    ap.add_argument("--crop-size", type=int, default=512, help="debug only; the protocol trains on full 512 tiles")
    args = ap.parse_args(argv)
    if args.batch_size % args.accum:
        raise SystemExit("--batch-size must be divisible by --accum")

    sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
    import torch
    from torch.utils.data import DataLoader

    from robustcd.adapters import changemamba as cm
    from robustcd.adapters.changemamba.infer import evaluate, read_ids
    from robustcd.datasets.second import SecondLike

    cm.add_to_path(args.cm_root)
    from changedetection.datasets.semantic_change_detection import SemanticChangeDetectionDataset
    from changedetection.tasks.scd import SCDTrainer

    t_start = time.time()
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    latest_p, best_p, best_ext_p = out / "latest.pth", out / "best_model.pth", out / "best_model_extended.pth"
    budget = args.budget_iters or args.max_iters
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # non-determinism accepted and documented (protocol)

    seed_everything(args.seed)
    model, _ = cm.build_model(args.cfg, encoder_pretrained=args.encoder_pretrained)
    model.to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay)
    shim = SimpleNamespace(model=model, device=device, optimizer=optimizer)
    scheduler = SCDTrainer.build_scheduler(shim)  # upstream StepLR(step_size=10000, gamma=0.5)

    start_iter, best = 0, {"SeK": -math.inf, "iteration": None}
    best_ext = {"SeK": -math.inf, "iteration": None}
    if latest_p.exists():
        ck = torch.load(latest_p, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        scheduler.load_state_dict(ck["scheduler"])
        start_iter, best = ck["iteration"], ck["best"]
        best_ext = ck.get("best_ext", best_ext)
        torch.set_rng_state(ck["torch_rng"])
        if torch.cuda.is_available() and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        print(f"resumed from {latest_p} at iteration {start_iter}, best val SeK {best['SeK']:.4f}", flush=True)
    else:
        (out / "run_config.json").write_text(json.dumps({
            **vars(args), "cm_commit": cm.check_commit(args.cm_root), "expected_cm_commit": cm.CM_COMMIT,
            "torch": torch.__version__, "cuda": torch.version.cuda, "node": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
        }, indent=2))
    if cm.check_commit(args.cm_root) != cm.CM_COMMIT:
        print(f"WARNING: ChangeMamba checkout is {cm.check_commit(args.cm_root)}, expected {cm.CM_COMMIT}", flush=True)

    train_ids = read_ids(str(Path(args.splits) / "train.txt"), [])
    val_ids = read_ids(str(Path(args.splits) / "val.txt"), [])[: args.val_limit]
    micro = args.batch_size // args.accum
    ds = SemanticChangeDetectionDataset(dataset_path=args.cm_data, data_list=train_ids,
                                        crop_size=args.crop_size, max_iters=None, batch_size=1, split="train")
    sampler = EpochOrderSampler(len(ds), args.max_iters * args.batch_size, args.seed,
                                start=start_iter * args.batch_size)
    loader = DataLoader(ds, batch_size=micro, sampler=sampler, num_workers=args.workers,
                        pin_memory=device.type == "cuda", drop_last=False,
                        persistent_workers=args.workers > 0,
                        generator=torch.Generator().manual_seed(args.seed + start_iter))
    val_reader = SecondLike(Path(args.second_root) / "train")
    use_bf16 = (not args.no_bf16) and device.type == "cuda"

    def save_latest(it):
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict(), "iteration": it, "best": best, "best_ext": best_ext,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None},
                   str(latest_p) + ".tmp")
        os.replace(str(latest_p) + ".tmp", latest_p)

    model.train()
    batches = iter(loader)
    t_log, it = time.time(), start_iter
    for it in range(start_iter + 1, args.max_iters + 1):
        loss_sum, logs = 0.0, {}
        for _ in range(args.accum):
            batch = next(batches)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                step = SCDTrainer.train_step(shim, batch)
            (step["loss"] / args.accum).backward()
            loss_sum += float(step["loss"].detach()) / args.accum
            for k, v in step.get("log_items", {}).items():
                logs[k] = logs.get(k, 0.0) + float(v) / args.accum
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        scheduler.step()
        if not math.isfinite(loss_sum):
            _jsonl(out / "train_log.jsonl", {"iteration": it, "event": "non-finite loss", "loss": loss_sum})
            print(f"non-finite loss at iteration {it}", flush=True)
            return 2

        if it % args.log_interval == 0:
            dt = time.time() - t_log
            t_log = time.time()
            rec = {"iteration": it, "loss": loss_sum, **logs, "lr": scheduler.get_last_lr()[0],
                   "s_per_iter": dt / args.log_interval,
                   "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0}
            _jsonl(out / "train_log.jsonl", rec)
            print(f"it {it}/{args.max_iters} loss {loss_sum:.4f} lr {rec['lr']:.2e} "
                  f"{rec['s_per_iter']:.3f}s/it mem {rec['max_mem_gb']:.1f}G", flush=True)

        if it % args.eval_interval == 0 or it == args.max_iters or it == budget:
            t_ev = time.time()
            scores = evaluate(model, val_reader, val_ids, device, args.eval_batch_size)
            model.train()
            rec_best = {"SeK": scores["SeK"], "iteration": it,
                        **{k: scores[k] for k in ("mIoU", "Fscd", "SeK_fromto")}}
            is_best = it <= budget and scores["SeK"] > best["SeK"]
            if is_best:
                best = rec_best
                torch.save({"model": model.state_dict(), "iteration": it, "val": scores},
                           str(best_p) + ".tmp")
                os.replace(str(best_p) + ".tmp", best_p)
            if it == budget and budget < args.max_iters:
                # the protocol's "last checkpoint" of a budget-length run
                torch.save({"model": model.state_dict(), "iteration": it, "val": scores}, out / "last_model_budget.pth")
            if budget < args.max_iters and scores["SeK"] > best_ext["SeK"]:
                best_ext = rec_best
                torch.save({"model": model.state_dict(), "iteration": it, "val": scores},
                           str(best_ext_p) + ".tmp")
                os.replace(str(best_ext_p) + ".tmp", best_ext_p)
            _jsonl(out / "val_log.jsonl", {"iteration": it, "is_best": is_best,
                                          "eval_seconds": round(time.time() - t_ev, 1),
                                          **{k: scores[k] for k in ("SeK", "mIoU", "Fscd", "kappa_n0",
                                                                    "SeK_fromto", "Fscd_fromto", "n_images")}})
            print(f"VAL it {it}: SeK {scores['SeK']:.4f} mIoU {scores['mIoU']:.4f} Fscd {scores['Fscd']:.4f} "
                  f"(best {best['SeK']:.4f} @ {best['iteration']})", flush=True)
            save_latest(it)

        if args.stop_after_min and (time.time() - t_start) / 60 > args.stop_after_min and it < args.max_iters:
            save_latest(it)
            print(f"time guard: saved at iteration {it}, exiting for resubmission", flush=True)
            return EXIT_REQUEUE

    torch.save({"model": model.state_dict(), "iteration": it}, out / "last_model.pth")
    summary = {"best": best, "best_extended": best_ext if budget < args.max_iters else None,
               "budget_iters": budget, "last_iteration": it, "wall_minutes_this_segment": round((time.time() - t_start) / 60, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"DONE best val SeK {best['SeK']:.4f} at iteration {best['iteration']}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
