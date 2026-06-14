"""v2 ORNL Stage 1 dataset.

v1 cache 의 image 채널 (visible_0.npy, visible_1.npy) 은 그대로 재사용,
**label 만** v2 cache 디렉터리 (label_v2.npy) 에서 읽음 (8-class 재매핑).

PLAN_v2 §5.1 참조.
"""
from __future__ import annotations

from pathlib import Path

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

# 공통 함수 재사용 (common)
from ...common.data.image_utils import (
    normalize_image,
    resize_image_uint8,
    resize_label,
    ornl_image_to_uint8,
)
from .. import config_v2 as cfg
from .augmentation import augment_v2


def ornl_segmentation_argmax_v2(
    seg_grp: h5py.Group, layer_idx: int,
) -> np.ndarray:
    """ORNL HDF5 의 12 boolean mask → v2 8-class argmax label.

    겹친 픽셀은 small ID 부터 적용 → 결함 (큰 ID) 이 Powder/Printed (작은 ID) 를 덮어씀.
    IGNORE 매핑된 class (4 Incomplete Spreading, 9 Misprint, 11 Under Melting) 의
    픽셀은 -1 로 남음.

    Args:
        seg_grp: f["slices/segmentation_results"] (12 boolean masks per layer)
        layer_idx: layer index

    Returns:
        (H, W) int8, 값 ∈ {-1} ∪ {0..7}
    """
    sample = seg_grp["0"][layer_idx]
    H, W = sample.shape
    out = np.full((H, W), cfg.IGNORE_INDEX, dtype=np.int8)
    for c_old in range(12):
        new_id = cfg.ORNL_12_TO_NEW_8.get(c_old, cfg.IGNORE_INDEX)
        if new_id == cfg.IGNORE_INDEX:
            continue
        m = seg_grp[str(c_old)][layer_idx]
        out[m] = new_id
    return out


class DefSegORNLCachedDatasetV2(Dataset):
    """v2 Stage 1 dataset — image 는 v1 cache 재사용, label 만 v2 cache 에서.

    cache 구조:
      v1: cache/resized_sz<IMG>/<build_id>/visible_0.npy, visible_1.npy
      v2: cache/resized_sz<IMG>_v2/<build_id>/label_v2.npy

    {split}_index.npz (build_ids, cache_rows, defect_ratios) 는 v2 cache 디렉터리에 저장.
    defect_ratio 는 v2 class 인덱스 (2..7) 기준 재계산되어 있어야 함 (build_cache_v2 가 책임).
    """

    def __init__(
        self,
        split: str,                    # 'train' or 'val'
        img_size: int = cfg.IMG_SIZE,
        training: bool = True,
    ):
        self.split = split
        self.img_size = img_size
        self.training = training
        self.v1_root = cfg.v1_cache_dir(img_size)   # image
        self.v2_root = cfg.v2_cache_dir(img_size)   # label

        idx_path = self.v2_root / f"{split}_index.npz"
        if not idx_path.exists():
            raise FileNotFoundError(
                f"v2 cache index not found: {idx_path}\n"
                "Run `python -m DefSeg_AM.v2.data.build_cache_v2` first."
            )
        d = np.load(idx_path, allow_pickle=False)
        self.build_ids = d["build_ids"]
        self.cache_rows = d["cache_rows"]
        self.defect_ratios = d["defect_ratios"].astype(np.float32)
        # per-process lazy memmap cache (h5py 처럼 fork-unsafe 아니지만 worker 마다 분리하는 게 안전)
        self._mm_v1: dict[str, dict] = {}
        self._mm_v2: dict[str, np.memmap] = {}

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

    def _open_v2_label(self, build_id: str) -> np.memmap:
        if build_id not in self._mm_v2:
            self._mm_v2[build_id] = np.load(
                self.v2_root / build_id / "label_v2.npy", mmap_mode="r",
            )
        return self._mm_v2[build_id]

    def __getitem__(self, idx: int) -> dict:
        build = str(self.build_ids[idx])
        row = int(self.cache_rows[idx])
        v1m = self._open_v1(build)
        lab_mm = self._open_v2_label(build)
        i0 = np.array(v1m["v0"][row])       # (img_size, img_size) uint8
        i1 = np.array(v1m["v1"][row])
        ann = np.array(lab_mm[row])         # int8 (v2 8-class)

        # v2 augmentation (D4 + cyclic + DSCNN noise/intensity + brightness)
        i0, i1, ann = augment_v2(i0, i1, ann, training=self.training)

        return {
            "img0": torch.from_numpy(normalize_image(i0)),
            "img1": torch.from_numpy(normalize_image(i1)),
            "label": torch.from_numpy(ann.astype(np.int64)),
            "build_id": build,
            "cache_row": row,
        }


def estimate_class_counts_v2(
    dataset: DefSegORNLCachedDatasetV2,
    n_sample: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """v2 label cache 에서 일부 sample 의 class count 추정 (class weight 계산용)."""
    rng = np.random.default_rng(seed)
    n = len(dataset)
    idxs = rng.choice(n, size=min(n_sample, n), replace=False)
    counts = np.zeros(cfg.N_CLASSES_V2, dtype=np.int64)
    for i in idxs:
        build = str(dataset.build_ids[i])
        row = int(dataset.cache_rows[i])
        lab_mm = dataset._open_v2_label(build)
        lab = np.array(lab_mm[row])
        valid = lab[lab >= 0]
        u, c = np.unique(valid, return_counts=True)
        for k, v in zip(u, c):
            counts[k] += v
    return counts
