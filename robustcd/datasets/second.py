"""Minimal reader for SECOND-style bi-temporal semantic change detection sets.

Layout (SECOND, Hi-UCD and LEVIR-CD after renaming all follow this shape)::

    <root>/im1/<id>.png      date 1
    <root>/im2/<id>.png      date 2
    <root>/label1/<id>.png   semantic map of date 1   (optional)
    <root>/label2/<id>.png   semantic map of date 2   (optional)

``root`` may be a directory or a ``.zip`` archive (the SECOND test split ships as
one), in which case members are read without extracting.  Reading from a zip is
convenient locally; for cluster runs point at an extracted directory.
"""

from __future__ import annotations

import io
import zipfile
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
from PIL import Image

STREAMS = ("im1", "im2", "label1", "label2")


def _to_array(raw: bytes) -> np.ndarray:
    with Image.open(io.BytesIO(raw)) as im:
        return np.asarray(im.convert("RGB") if im.mode in ("P", "L", "RGBA") else im)


class SecondLike:
    """Index a SECOND-style split and read samples as numpy arrays."""

    def __init__(self, root: str | Path, inner_prefix: str = "", name: Optional[str] = None):
        self.root = Path(root)
        self.name = name or self.root.stem
        self._zip: Optional[zipfile.ZipFile] = None
        self._members: Dict[str, Dict[str, str]] = {}

        if self.root.suffix.lower() == ".zip":
            self._zip = zipfile.ZipFile(self.root)
            names = [n for n in self._zip.namelist() if not n.endswith("/")]
            prefix = inner_prefix
            if not prefix:
                # infer: the path component directly above "im1"
                for n in names:
                    if "/im1/" in n:
                        prefix = n.split("/im1/")[0]
                        break
            self.inner_prefix = prefix
            for n in names:
                for s in STREAMS:
                    key = f"{prefix}/{s}/" if prefix else f"{s}/"
                    if n.startswith(key):
                        self._members.setdefault(s, {})[Path(n).name] = n
        else:
            self.inner_prefix = ""
            for s in STREAMS:
                d = self.root / s
                if d.is_dir():
                    self._members[s] = {p.name: str(p) for p in sorted(d.iterdir()) if p.is_file()}

        if "im1" not in self._members or "im2" not in self._members:
            raise FileNotFoundError(f"no im1/im2 streams found under {self.root}")
        common = set(self._members["im1"]) & set(self._members["im2"])
        self.ids: List[str] = sorted(Path(f).stem for f in common)
        self.filenames: Dict[str, str] = {Path(f).stem: f for f in common}

    def __len__(self) -> int:
        return len(self.ids)

    @property
    def available_streams(self) -> List[str]:
        return [s for s in STREAMS if s in self._members]

    def _read(self, stream: str, sample_id: str) -> Optional[np.ndarray]:
        fname = self.filenames[sample_id]
        entry = self._members.get(stream, {}).get(fname)
        if entry is None:
            return None
        raw = self._zip.read(entry) if self._zip is not None else Path(entry).read_bytes()
        return _to_array(raw)

    def get(self, sample_id: str) -> Dict[str, np.ndarray]:
        out = {}
        for s in self.available_streams:
            arr = self._read(s, sample_id)
            if arr is not None:
                out[s] = arr
        return out

    def raw_bytes(self, stream: str, sample_id: str) -> Optional[bytes]:
        """Original encoded bytes -- used to copy pass-through streams verbatim."""
        fname = self.filenames[sample_id]
        entry = self._members.get(stream, {}).get(fname)
        if entry is None:
            return None
        return self._zip.read(entry) if self._zip is not None else Path(entry).read_bytes()

    def close(self) -> None:
        if self._zip is not None:
            self._zip.close()
