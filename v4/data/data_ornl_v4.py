"""v4 ORNL Stage 1 dataset.

v1 cache 의 image 채널 (visible_0.npy, visible_1.npy) 그대로 재사용,
**label 만** v4 cache 디렉터리 (label_v4.npy) 에서 읽음 (11-class 재매핑).

v4 신규: pix_per_layer (n_layers, 11) int64 도 split_index 에서 로드 — class-aware
sampler 의 layer weight 계산에 사용. + Copy-Paste 용 source fetch.

PLAN_v4 §4.1, §6.2 참조.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ...common.data.image_utils import (
    normalize_image,
    resize_image_uint8,
    resize_label,
    ornl_image_to_uint8,
)
from .. import config_v4 as cfg
from .augmentation import augment_v4


def ornl_segmentation_argmax_v4(
    seg_grp: h5py.Group, layer_idx: int,
) -> np.ndarray:
    """ORNL HDF5 의 12 boolean mask → v4 11-class argmax label.

    겹친 픽셀은 small ID 부터 적용 → 결함 (큰 ID) 이 Powder/Printed (작은 ID) 를 덮어씀.
    IGNORE 매핑된 class (9 Misprint 만) 의 픽셀은 -1 로 남음.

    Returns:
        (H, W) int8, 값 ∈ {-1} ∪ {0..10}
    """
    sample = seg_grp["0"][layer_idx]
    H, W = sample.shape
    out = np.full((H, W), cfg.IGNORE_INDEX, dtype=np.int8)
    for c_old in range(12):
        new_id = cfg.ORNL_12_TO_NEW_11.get(c_old, cfg.IGNORE_INDEX)
        if new_id == cfg.IGNORE_INDEX:
            continue
        m = seg_grp[str(c_old)][layer_idx]
        out[m] = new_id
    return out


class DefSegORNLCachedDatasetV4(Dataset):
    """v4 Stage 1 dataset — image 는 v1 cache 재사용, label 만 v4 cache 에서.

    cache 구조:
      v1: cache/resized_sz<IMG>/<build_id>/visible_0.npy, visible_1.npy
      v4: cache/resized_sz<IMG>_v4/<build_id>/label_v4.npy
                                            /pix_per_layer.npy
                                /{train,val}_index.npz  (build_ids, cache_rows, defect_ratios, pix_per_layer)
    """

    def __init__(
        self,
        split: str,                    # 'train' or 'val'
        img_size: int = cfg.IMG_SIZE,
        training: bool = True,
        cp_sampler=None,
    ):
        self.split = split
        self.img_size = img_size
        self.training = training
        self.cp_sampler = cp_sampler
        self.v1_root = cfg.v1_cache_dir(img_size)
        self.v4_root = cfg.v4_cache_dir(img_size)

        idx_path = self.v4_root / f"{split}_index.npz"
        if not idx_path.exists():
            raise FileNotFoundError(
                f"v4 cache index not found: {idx_path}\n"
                "Run `python -m DefSeg_AM.v4.data.build_cache_v4` first."
            )
        d = np.load(idx_path, allow_pickle=False)
        self.build_ids = d["build_ids"]
        self.cache_rows = d["cache_rows"]
        self.defect_ratios = d["defect_ratios"].astype(np.float32)
        # v4 신규
        self.pix_per_layer = (
            d["pix_per_layer"] if "pix_per_layer" in d.files
            else np.zeros((len(self.build_ids), cfg.N_CLASSES_V4), dtype=np.int64)
        )

        # per-process lazy memmap cache
        self._mm_v1: dict[str, dict] = {}
        self._mm_v4: dict[str, np.memmap] = {}

    def __len__(self) -> int:
        return len(self.build_ids)

    def _open_v1(self, build_id: str) -> dict:
        if build_id not in self._mm_v1:
            bdir = self.v1_root / build_id
            self._mm_v1[build_id] = {
                "v0": np.load(bdir / "visible_0.npy", mmap_mode="r"),
                "v1": np.load(bdir / "visible_1.npy", mmap_mode="r"),
            }
        return self._mm_v1[build_id]

    def _open_v4_label(self, build_id: str) -> np.memmap:
        if build_id not in self._mm_v4:
            self._mm_v4[build_id] = np.load(
                self.v4_root / build_id / "label_v4.npy", mmap_mode="r",
            )
        return self._mm_v4[build_id]

    def fetch_by_build_row(self, build_id: str, row: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Copy-Paste supplier 용: (build_id, row) → (i0, i1, ann). augmentation 없이."""
        v1m = self._open_v1(build_id)
        lab_mm = self._open_v4_label(build_id)
        i0 = np.array(v1m["v0"][row])
        i1 = np.array(v1m["v1"][row])
        ann = np.array(lab_mm[row])
        return i0, i1, ann

    def __getitem__(self, idx: int) -> dict:
        build = str(self.build_ids[idx])
        row = int(self.cache_rows[idx])
        v1m = self._open_v1(build)
        lab_mm = self._open_v4_label(build)
        i0 = np.array(v1m["v0"][row])
        i1 = np.array(v1m["v1"][row])
        ann = np.array(lab_mm[row])

        # v4 augmentation (Copy-Paste + flip LR/UD + 180° + cyclic + DSCNN noise/intensity + brightness)
        i0, i1, ann = augment_v4(i0, i1, ann, training=self.training, cp_sampler=self.cp_sampler)

        return {
            "img0": torch.from_numpy(normalize_image(i0)),
            "img1": torch.from_numpy(normalize_image(i1)),
            "label": torch.from_numpy(ann.astype(np.int64)),
            "build_id": build,
            "cache_row": row,
        }


def estimate_class_counts_v4(
    dataset: DefSegORNLCachedDatasetV4,
    n_sample: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """v4 label cache 에서 일부 sample 의 class count 추정 (loss class weight 계산용)."""
    rng = np.random.default_rng(seed)
    n = len(dataset)
    idxs = rng.choice(n, size=min(n_sample, n), replace=False)
    counts = np.zeros(cfg.N_CLASSES_V4, dtype=np.int64)
    for i in idxs:
        build = str(dataset.build_ids[i])
        row = int(dataset.cache_rows[i])
        lab_mm = dataset._open_v4_label(build)
        lab = np.array(lab_mm[row])
        valid = lab[lab >= 0]
        u, c = np.unique(valid, return_counts=True)
        for k, v in zip(u, c):
            counts[k] += v
    return counts
