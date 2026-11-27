"""공용 배경 캐시 읽기 (`data/build_canvas_cache.py` 가 만든 것).

배열은 memmap 으로 연다. DataLoader worker 가 fork 된 뒤에 각자 열도록 처음 쓸 때 연다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import config_genaug as cfg


class CanvasPool:
    def __init__(self, builds=cfg.TRAIN_BUILDS, root: Path | None = None, log=None):
        self.root = Path(root) if root is not None else cfg.CANVAS_ROOT
        self.builds = tuple(builds)
        self.n_layers: dict[str, int] = {}
        self.orig_layers: dict[str, np.ndarray] = {}
        self.foot_frac: dict[str, np.ndarray] = {}
        for b in self.builds:
            d = self.root / b
            if not all((d / f).is_file() for f in ("visible_0.npy", "visible_1.npy", "part_mask.npy", "meta.npz")):
                raise FileNotFoundError(
                    f"FATAL: 배경 캐시 {d} 없음 — `python -m DefSeg_AM.genaug.data.build_canvas_cache` (compose 의 canvas phase) 로 만든다"
                )
            m = np.load(d / "meta.npz")
            self.orig_layers[b] = m["orig_layer_idxs"]
            self.foot_frac[b] = m["foot_frac"]
            self.n_layers[b] = int(len(m["orig_layer_idxs"]))
        self._arr: dict[str, tuple] = {}
        self.img_size = int(np.load(self.root / self.builds[0] / "visible_0.npy", mmap_mode="r").shape[1])
        if log is not None:
            log.info("주석 없는 배경: " + ", ".join(f"{b} {self.n_layers[b]}" for b in self.builds)
                     + f" 레이어 (원본 밝기, {self.root.name})")

    def __getstate__(self):                       # worker 로 넘어갈 때 memmap 핸들을 들고 가지 않는다
        st = dict(self.__dict__)
        st["_arr"] = {}
        return st

    def arrays(self, build: str):
        """(visible_0, visible_1, part_mask) memmap."""
        if build not in self._arr:
            d = self.root / build
            self._arr[build] = tuple(np.load(d / f"{a}.npy", mmap_mode="r") for a in ("visible_0", "visible_1", "part_mask"))
        return self._arr[build]

    def total_layers(self) -> int:
        return sum(self.n_layers.values())

    def pick_layer(self, rng: np.random.Generator) -> tuple[str, int]:
        """레이어 수에 비례해 빌드를 고르고 그 안에서 한 층을 고른다."""
        r = int(rng.integers(self.total_layers()))
        for b in self.builds:
            if r < self.n_layers[b]:
                return b, r
            r -= self.n_layers[b]
        raise AssertionError

    def crop(self, build: str, row: int, y0: int, x0: int, h: int, w: int):
        """창을 잘라 복사본으로 돌려준다 → (i0, i1, part) — uint8, uint8, bool."""
        v0, v1, pm = self.arrays(build)
        sl = (row, slice(y0, y0 + h), slice(x0, x0 + w))
        return np.array(v0[sl]), np.array(v1[sl]), np.array(pm[sl]) > 0
