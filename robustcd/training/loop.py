"""Protocol training loop shared by model adapters (docs/protocol.md, "Training").

An adapter supplies the model, its optimiser, its learning-rate rule, a
``train_step`` that returns the upstream loss for one micro-batch, and an
``evaluate`` callable returning robustcd val scores. The loop provides what the
protocol fixes for every model:

- a seeded, resumable data order (``EpochOrderSampler``);
- bf16 autocast and gradient accumulation to a fixed effective batch size;
- val evaluation every ``eval_interval`` iterations, and best-val checkpoint
  selection on robustcd SeK among evaluations at iterations <= budget;
- ``best_model_extended.pth`` over the whole run when ``max_iters > budget``
  (pilot rule), and ``last_model_budget.pth`` at the budget iteration;
- ``latest.pth`` for resumption, JSONL logs, and a wall-clock guard that saves
  and returns ``EXIT_REQUEUE`` so the SLURM script can resubmit itself.

ChangeMamba predates this module and keeps its own copy of the same logic
(``robustcd/adapters/changemamba/train.py``), so that its running campaign is
not affected by later changes here.
"""

from __future__ import annotations

import json
import math
import os
import platform
import random
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any, Callable, Dict, Optional, Tuple

import numpy as np

EXIT_REQUEUE = 3


class EpochOrderSampler:
    """Deterministic epoch-wise permutations, resumable at any sample position.

    Epoch e uses ``torch.randperm(n, generator=seed * 1_000_003 + e)``, so sample
    position p is fully determined by (seed, p): resuming at iteration k replays
    exactly the order an uninterrupted run would have used.
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


def jsonl(path: Path, rec: dict) -> None:
    with path.open("a") as fh:
        fh.write(json.dumps(rec) + "\n")


@dataclass
class LoopConfig:
    out: str
    seed: int
    max_iters: int = 50_000
    budget_iters: Optional[int] = None
    batch_size: int = 16          # effective batch size
    accum: int = 1                # gradient accumulation steps
    eval_interval: int = 2000
    log_interval: int = 50
    workers: int = 4
    bf16: bool = True
    stop_after_min: Optional[float] = None
    extra: Dict[str, Any] = field(default_factory=dict)   # recorded in run_config.json

    @property
    def budget(self) -> int:
        return self.budget_iters or self.max_iters

    @property
    def micro_batch(self) -> int:
        if self.batch_size % self.accum:
            raise ValueError("batch_size must be divisible by accum")
        return self.batch_size // self.accum


def run(
    cfg: LoopConfig,
    *,
    model,
    optimizer,
    dataset,
    train_step: Callable[[Any, int], Tuple[Any, Dict[str, float]]],
    evaluate: Callable[[Any], Dict[str, float]],
    set_lr: Optional[Callable[[int], float]] = None,
    scheduler=None,
    extra_state=None,
    on_eval: Optional[Callable[[Dict[str, float], int], None]] = None,
    device=None,
    collate_fn=None,
) -> int:
    """Train ``model`` under the protocol. Returns 0, 2 (non-finite loss) or EXIT_REQUEUE.

    ``train_step(batch, iteration)`` returns ``(loss, logs)`` for one micro-batch;
    the loop scales by ``1/accum`` and calls backward. ``set_lr(iteration)``
    (called before each optimiser step, iteration starting at 1) is for
    iteration-indexed rules such as poly decay and must be a pure function of
    the iteration; ``scheduler`` is a torch LR scheduler stepped after each
    optimiser step. ``extra_state`` is any object with ``state_dict()`` /
    ``load_state_dict()`` saved in ``latest.pth`` (e.g. a pseudo-label teacher).
    ``on_eval(scores, iteration)`` runs after each val evaluation.
    """
    import torch
    from torch.utils.data import DataLoader

    t_start = time.time()
    out = Path(cfg.out)
    out.mkdir(parents=True, exist_ok=True)
    latest_p, best_p, best_ext_p = out / "latest.pth", out / "best_model.pth", out / "best_model_extended.pth"
    budget = cfg.budget
    device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True  # non-determinism accepted and documented (protocol)

    start_iter = 0
    best = {"SeK": -math.inf, "iteration": None}
    best_ext = {"SeK": -math.inf, "iteration": None}
    if latest_p.exists():
        ck = torch.load(latest_p, map_location="cpu", weights_only=False)
        model.load_state_dict(ck["model"])
        optimizer.load_state_dict(ck["optimizer"])
        if scheduler is not None:
            scheduler.load_state_dict(ck["scheduler"])
        if extra_state is not None and ck.get("extra_state") is not None:
            extra_state.load_state_dict(ck["extra_state"])
        start_iter, best = ck["iteration"], ck["best"]
        best_ext = ck.get("best_ext", best_ext)
        torch.set_rng_state(ck["torch_rng"])
        if torch.cuda.is_available() and ck.get("cuda_rng") is not None:
            torch.cuda.set_rng_state_all(ck["cuda_rng"])
        random.setstate(ck["py_rng"])
        np.random.set_state(ck["np_rng"])
        print(f"resumed from {latest_p} at iteration {start_iter}, best val SeK {best['SeK']:.4f}", flush=True)
    else:
        (out / "run_config.json").write_text(json.dumps({
            **asdict(cfg), "torch": torch.__version__, "cuda": torch.version.cuda, "node": platform.node(),
            "slurm_job_id": os.environ.get("SLURM_JOB_ID"),
            "device": torch.cuda.get_device_name(0) if device.type == "cuda" else "cpu",
            "n_params": sum(p.numel() for p in model.parameters()),
        }, indent=2, default=str))
    if start_iter >= cfg.max_iters:
        print(f"nothing to do: latest.pth is at iteration {start_iter} >= max_iters {cfg.max_iters}", flush=True)
        return 0

    sampler = EpochOrderSampler(len(dataset), cfg.max_iters * cfg.batch_size, cfg.seed,
                                start=start_iter * cfg.batch_size)
    loader = DataLoader(dataset, batch_size=cfg.micro_batch, sampler=sampler, num_workers=cfg.workers,
                        pin_memory=device.type == "cuda", drop_last=False,
                        persistent_workers=cfg.workers > 0, collate_fn=collate_fn,
                        generator=torch.Generator().manual_seed(cfg.seed + start_iter))
    use_bf16 = cfg.bf16 and device.type == "cuda"

    def save_latest(it):
        torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                    "scheduler": scheduler.state_dict() if scheduler is not None else None,
                    "extra_state": extra_state.state_dict() if extra_state is not None else None,
                    "iteration": it, "best": best, "best_ext": best_ext,
                    "torch_rng": torch.get_rng_state(),
                    "cuda_rng": torch.cuda.get_rng_state_all() if torch.cuda.is_available() else None,
                    "py_rng": random.getstate(), "np_rng": np.random.get_state()},
                   str(latest_p) + ".tmp")
        os.replace(str(latest_p) + ".tmp", latest_p)

    def save_model(path, it, scores=None):
        torch.save({"model": model.state_dict(), "iteration": it, "val": scores}, str(path) + ".tmp")
        os.replace(str(path) + ".tmp", path)

    def current_lr():
        return optimizer.param_groups[0]["lr"]

    model.train()
    batches = iter(loader)
    t_log, it = time.time(), start_iter
    for it in range(start_iter + 1, cfg.max_iters + 1):
        if set_lr is not None:
            lr = set_lr(it)
            for g in optimizer.param_groups:
                g["lr"] = lr
        loss_sum, logs = 0.0, {}
        for _ in range(cfg.accum):
            batch = next(batches)
            with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=use_bf16):
                loss, step_logs = train_step(batch, it)
            (loss / cfg.accum).backward()
            loss_sum += float(loss.detach()) / cfg.accum
            for k, v in step_logs.items():
                logs[k] = logs.get(k, 0.0) + float(v) / cfg.accum
        optimizer.step()
        optimizer.zero_grad(set_to_none=True)
        if scheduler is not None:
            scheduler.step()
        if not math.isfinite(loss_sum):
            jsonl(out / "train_log.jsonl", {"iteration": it, "event": "non-finite loss", "loss": loss_sum})
            print(f"non-finite loss at iteration {it}", flush=True)
            return 2

        if it % cfg.log_interval == 0:
            dt = time.time() - t_log
            t_log = time.time()
            rec = {"iteration": it, "loss": loss_sum, **logs, "lr": current_lr(),
                   "s_per_iter": dt / cfg.log_interval,
                   "max_mem_gb": torch.cuda.max_memory_allocated() / 2**30 if device.type == "cuda" else 0.0}
            jsonl(out / "train_log.jsonl", rec)
            print(f"it {it}/{cfg.max_iters} loss {loss_sum:.4f} lr {rec['lr']:.2e} "
                  f"{rec['s_per_iter']:.3f}s/it mem {rec['max_mem_gb']:.1f}G", flush=True)

        if it % cfg.eval_interval == 0 or it == cfg.max_iters or it == budget:
            t_ev = time.time()
            scores = evaluate(model)
            model.train()
            rec_best = {"SeK": scores["SeK"], "iteration": it,
                        **{k: scores[k] for k in ("mIoU", "Fscd", "SeK_fromto") if k in scores}}
            is_best = it <= budget and scores["SeK"] > best["SeK"]
            if is_best:
                best = rec_best
                save_model(best_p, it, scores)
            if it == budget and budget < cfg.max_iters:
                save_model(out / "last_model_budget.pth", it, scores)
            if budget < cfg.max_iters and scores["SeK"] > best_ext["SeK"]:
                best_ext = rec_best
                save_model(best_ext_p, it, scores)
            if on_eval is not None:
                on_eval(scores, it)
            jsonl(out / "val_log.jsonl", {"iteration": it, "is_best": is_best,
                                          "eval_seconds": round(time.time() - t_ev, 1),
                                          **{k: v for k, v in scores.items()
                                             if isinstance(v, (int, float))}})
            print(f"VAL it {it}: SeK {scores['SeK']:.4f} mIoU {scores['mIoU']:.4f} Fscd {scores['Fscd']:.4f} "
                  f"(best {best['SeK']:.4f} @ {best['iteration']})", flush=True)
            save_latest(it)

        if cfg.stop_after_min is not None and (time.time() - t_start) / 60 > cfg.stop_after_min and it < cfg.max_iters:
            save_latest(it)
            print(f"time guard: saved at iteration {it}, exiting for resubmission", flush=True)
            return EXIT_REQUEUE

    save_model(out / "last_model.pth", it)
    summary = {"best": best, "best_extended": best_ext if budget < cfg.max_iters else None,
               "budget_iters": budget, "last_iteration": it,
               "wall_minutes_this_segment": round((time.time() - t_start) / 60, 1)}
    (out / "summary.json").write_text(json.dumps(summary, indent=2))
    print(f"DONE best val SeK {best['SeK']:.4f} at iteration {best['iteration']}", flush=True)
    return 0


def poly_lr(base_lr: float, total_iters: int, power: float) -> Callable[[int], float]:
    """Poly decay ``base_lr * (1 - it / total) ** power`` (Ding et al. codebases).

    ``total_iters`` must be the length of the *schedule*, which is fixed at
    launch: extending a poly run is not equivalent to a longer run.
    """
    def f(it: int) -> float:
        return base_lr * max(0.0, 1.0 - float(it) / total_iters) ** power
    return f
