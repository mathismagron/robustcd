"""Two thin wrappers over the same core transform.

``MisregTransform``  on-the-fly: seeded by sample id, so it needs no dataloader
                     RNG discipline -- worker count, shuffling and epoch order
                     cannot change what a given sample receives.  Use for quick
                     experiments.
``RenderedSet``      read a set produced by ``scripts/render_misregistration.py``
                     via its ``dataset_spec.json``.  Every model sees identical
                     bytes; use for the numbers that go in the paper.

Both return numpy arrays in the dataset's native dtype -- normalisation and
tensor conversion stay in each model repo's own pipeline, so this module drops
into ten different codebases without fighting their transforms.  ``torch`` is
imported lazily and only by ``TorchRenderedSet``.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Callable, Dict, List, Optional

import numpy as np
from PIL import Image

from . import misregistration as mis


class MisregTransform:
    """Callable applying one ``MisregSpec`` to a sample dict, keyed by sample id."""

    def __init__(self, spec: mis.MisregSpec, dataset: str = ""):
        self.spec = spec
        self.dataset = dataset

    def __call__(self, sample: Dict[str, np.ndarray], sample_id: str) -> Dict[str, np.ndarray]:
        res = mis.degrade_pair(
            sample["im1"], sample["im2"], self.spec, sample_id=sample_id, dataset=self.dataset,
            label1=sample.get("label1"), label2=sample.get("label2"),
        )
        drop = {"record", "warp_t1", "warp_t2"}  # not collatable; kept off the batch
        out = {k: v for k, v in res.items() if k not in drop}
        for k, v in sample.items():  # pass through anything we do not touch
            out.setdefault(k, v)
        return out

    def __repr__(self) -> str:
        return f"MisregTransform({self.spec.tag}, apply_to={self.spec.apply_to})"


class RenderedSet:
    """Index a rendered degraded set; ``__getitem__`` returns a dict of arrays."""

    def __init__(self, set_dir: str | Path, streams: Optional[List[str]] = None):
        self.set_dir = Path(set_dir)
        self.spec_json = json.loads((self.set_dir / "dataset_spec.json").read_text())
        self.streams = streams or list(self.spec_json["streams"])
        roots = self.spec_json["streams"]
        self._paths: Dict[str, Dict[str, Path]] = {}
        for s in self.streams:
            info = roots[s]
            base = Path(info["root"])
            d = base if base.name == info["subdir"] else base / info["subdir"]
            if not d.is_dir():
                raise FileNotFoundError(
                    f"stream {s!r} expected at {d}; the source dataset must be an extracted "
                    "directory (not a zip) for RenderedSet to resolve pass-through streams"
                )
            self._paths[s] = {p.stem: p for p in sorted(d.iterdir()) if p.is_file()}
        ids = set.intersection(*(set(v) for v in self._paths.values()))
        self.ids: List[str] = sorted(ids)
        self.manifest = {}
        mf = self.set_dir / "manifest.jsonl"
        if mf.exists():
            self.manifest = {r["sample_id"]: r for r in (json.loads(l) for l in mf.read_text().splitlines())}

    def __len__(self) -> int:
        return len(self.ids)

    def __getitem__(self, i: int) -> Dict[str, object]:
        sid = self.ids[i]
        out: Dict[str, object] = {"sample_id": sid}
        for s in self.streams:
            with Image.open(self._paths[s][sid]) as im:
                arr = np.asarray(im.convert("L") if s == "valid_mask" else im)
            out[s] = arr > 127 if s == "valid_mask" else arr
        return out


class TorchRenderedSet:
    """``torch.utils.data.Dataset`` view of a ``RenderedSet``.

    ``collate_fn`` is intentionally left to the caller; ``to_tensor`` receives the
    raw sample dict so each model repo can apply its own normalisation.
    """

    def __new__(cls, set_dir, to_tensor: Optional[Callable] = None, **kw):
        from torch.utils.data import Dataset  # lazy: keep the core torch-free

        base = RenderedSet(set_dir, **kw)

        class _DS(Dataset):
            def __len__(self):
                return len(base)

            def __getitem__(self, i):
                s = base[i]
                return to_tensor(s) if to_tensor else s

            ids = base.ids
            manifest = base.manifest
            spec_json = base.spec_json

        return _DS()
