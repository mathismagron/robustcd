"""SCanNet's pseudo-label stage (train_SCD_psd.py), re-implemented for the robustcd loop.

Upstream behaviour, reproduced here:

1. The teacher is a frozen copy of the network taken whenever the val Fscd
   improves. The per-class confidence thresholds are reset at each copy.
2. The stage is active once the best val Fscd exceeds ``psd_init_fscd`` (0.6).
3. For each training batch, the teacher predicts both dates, averaged over
   four flips (identity, vertical, horizontal, both).
4. A pixel is confident if its max probability reaches the running threshold
   of its predicted class. The running threshold is the count-weighted mean of
   the per-batch ``pseudo_thred`` percentile of that class's max-probabilities,
   clipped to [0.5, 0.9].
5. On pixels labelled unchanged, where both dates are confident and predict the
   same class, that class is written into both semantic labels. Those pixels
   were ignored by the semantic loss before.

Adaptation to the protocol: upstream validates once per epoch (about 297
iterations at batch 8). Here the teacher can only change at the protocol's val
evaluations, every ``eval_interval`` iterations, on the robustcd val split.
The teacher's flip averaging is a training-time procedure; inference remains
single-pass (no TTA), as for every model.
"""

from __future__ import annotations

import copy
from typing import Optional

import numpy as np


class AverageThreshold:
    """Upstream ``AverageThred``: count-weighted running mean of per-class thresholds."""

    def __init__(self, num_classes: int, init: float):
        self.threds = np.ones(num_classes, dtype=float) * init
        self.count = np.ones(num_classes, dtype=int)
        self.sum = self.threds * self.count

    def update(self, threds: np.ndarray, count: np.ndarray) -> None:
        self.count += np.array(count, dtype=int)
        self.sum += threds * count
        self.threds = self.sum / self.count

    def value(self) -> np.ndarray:
        return np.clip(self.threds, 0.5, 0.9)

    def state_dict(self) -> dict:
        return {"threds": self.threds.copy(), "count": self.count.copy(), "sum": self.sum.copy()}

    def load_state_dict(self, sd: dict) -> None:
        self.threds, self.count, self.sum = (np.array(sd[k]) for k in ("threds", "count", "sum"))


def confident(prob, thresholds: AverageThreshold, pseudo_thred: float):
    """Upstream ``calc_conf``: returns (confident mask, argmax index); updates ``thresholds``."""
    import torch
    import torch.nn.functional as F

    c = prob.shape[1]
    conf, index = torch.max(prob, dim=1)
    onehot = F.one_hot(index.long(), num_classes=c).permute(0, 3, 1, 2)
    masked = onehot * prob
    threds, counts = np.zeros(c), np.zeros(c)
    for k in range(c):
        v = torch.flatten(masked[:, k])
        v = v[v.nonzero()]
        if v.numel() > 0:
            threds[k] = np.percentile(v.float().cpu().numpy().flatten(), 100 * pseudo_thred)
            counts[k] = v.shape[0]
        else:
            threds[k] = pseudo_thred
            counts[k] = 0
    thresholds.update(threds, counts)
    t = torch.from_numpy(thresholds.value()).to(prob.device).unsqueeze(1).unsqueeze(2)
    thredmap, _ = torch.max(onehot * t, dim=1)
    return torch.ge(conf, thredmap), index


class PseudoLabeler:
    """Teacher state plus the labelling rule; pass as ``extra_state`` to the loop."""

    def __init__(self, model, num_classes: int = 7, psd_init_fscd: float = 0.6,
                 pseudo_thred: float = 0.6, tta: bool = True):
        self.model = model
        self.num_classes = num_classes
        self.psd_init_fscd = psd_init_fscd
        self.pseudo_thred = pseudo_thred
        self.tta = tta
        self.best_fscd = 0.0
        self.teacher = None
        self.teacher_iteration: Optional[int] = None
        self.thresholds = AverageThreshold(num_classes, pseudo_thred)
        self.n_pseudo = 0          # pseudo-labelled pixels since last log (per date)

    # ---------------------------------------------------------------- teacher
    def _snapshot(self):
        t = copy.deepcopy(self.model)
        t.eval()
        for p in t.parameters():
            p.requires_grad_(False)
        return t

    def on_eval(self, scores: dict, iteration: int) -> None:
        if scores["Fscd"] > self.best_fscd:
            self.best_fscd = scores["Fscd"]
            self.teacher = self._snapshot()
            self.teacher_iteration = iteration
            self.thresholds = AverageThreshold(self.num_classes, self.pseudo_thred)

    @property
    def active(self) -> bool:
        return self.teacher is not None and self.best_fscd > self.psd_init_fscd

    # ------------------------------------------------------------- labelling
    def _teacher_probs(self, x1, x2):
        import torch
        import torch.nn.functional as F

        flips = [None, [2], [3], [2, 3]] if self.tta else [None]
        pa = pb = pc = 0
        with torch.no_grad():
            for dims in flips:
                a, b = (x1, x2) if dims is None else (torch.flip(x1, dims), torch.flip(x2, dims))
                oc, oa, ob = self.teacher(a, b)
                oc, oa, ob = oc.float(), oa.float(), ob.float()
                if dims is not None:
                    oc, oa, ob = torch.flip(oc, dims), torch.flip(oa, dims), torch.flip(ob, dims)
                pa = pa + F.softmax(oa, dim=1)
                pb = pb + F.softmax(ob, dim=1)
                pc = pc + torch.sigmoid(oc)
        n = len(flips)
        return pa / n, pb / n, pc / n

    def apply(self, x1, x2, labels_a, labels_b, labels_bn):
        """Return labels with pseudo labels added on confident unchanged pixels (upstream rule)."""
        import torch

        if not self.active:
            return labels_a, labels_b
        prob_a, prob_b, _ = self._teacher_probs(x1, x2)
        conf_a, idx_a = confident(prob_a, self.thresholds, self.pseudo_thred)
        conf_b, idx_b = confident(prob_b, self.thresholds, self.pseudo_thred)
        mask = conf_a & conf_b & torch.eq(idx_a, idx_b) & torch.logical_not(labels_bn.bool()).squeeze(1)
        pseudo = idx_a * mask
        self.n_pseudo += int((pseudo > 0).sum())
        return labels_a + pseudo, labels_b + pseudo

    # ------------------------------------------------------------ state I/O
    def state_dict(self) -> dict:
        return {"best_fscd": self.best_fscd, "teacher_iteration": self.teacher_iteration,
                "teacher": None if self.teacher is None else
                {k: v.detach().cpu() for k, v in self.teacher.state_dict().items()},
                "thresholds": self.thresholds.state_dict()}

    def load_state_dict(self, sd: dict) -> None:
        self.best_fscd = sd["best_fscd"]
        self.teacher_iteration = sd.get("teacher_iteration")
        self.thresholds.load_state_dict(sd["thresholds"])
        if sd.get("teacher") is not None:
            self.teacher = self._snapshot()
            self.teacher.load_state_dict(sd["teacher"])
        else:
            self.teacher = None
