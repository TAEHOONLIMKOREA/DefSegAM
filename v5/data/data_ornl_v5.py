"""v5 ORNL 참고 평가용 데이터 — **학습에는 쓰지 않는다** (PLAN_v5 §6.4).

ORNL Co-Registered HDF5 의 `slices/segmentation_results` 는 DSCNN 의 예측 mask 다.
v5 는 이것으로 학습하지도, 길이 · ckpt 를 고르지도 않는다. 다만 v1–v4 가 모두 Build 1 에서
"DSCNN 예측과의 일치도" 를 보고했으므로, 같은 표를 참고용으로 내기 위해 읽는다.

클래스 공간이 v4 와 같아서 새 캐시를 만들지 않고 기존 캐시를 그대로 읽는다:
  이미지: v1 cache  cache/resized_sz<IMG>/<build>/visible_{0,1}.npy
  라벨:   v4 cache  cache/resized_sz<IMG>_v4/<build>/label_v4.npy  (+ {train,val}_index.npz)
"""
from __future__ import annotations

import h5py
import numpy as np
import torch
from torch.utils.data import Dataset

from ...common.data.image_utils import normalize_image
from .. import config_v5 as cfg


def ornl_segmentation_argmax_v5(
    seg_grp: h5py.Group, layer_idx: int,
) -> np.ndarray:
    """ORNL HDF5 의 12 boolean mask (DSCNN 예측) → v5 11-class argmax label.

    겹친 픽셀은 small ID 부터 적용 → 결함 (큰 ID) 이 Powder/Printed (작은 ID) 를 덮어씀.
    IGNORE 매핑된 class (9 Misprint) 의 픽셀은 -1 로 남음. v4 의 같은 함수와 동작이 같다.

    Returns:
        (H, W) int8, 값 ∈ {-1} ∪ {0..10}
    """
    sample = seg_grp["0"][layer_idx]
    H, W = sample.shape
    out = np.full((H, W), cfg.IGNORE_INDEX, dtype=np.int8)
    for c_old in range(12):
        new_id = cfg.ORNL_12_TO_V5.get(c_old, cfg.IGNORE_INDEX)
        if new_id == cfg.IGNORE_INDEX:
            continue
        m = seg_grp[str(c_old)][layer_idx]
        out[m] = new_id
    return out


class DefSegORNLCachedDatasetV5(Dataset):
    """ORNL 캐시 (v1 이미지 + v4 라벨) 읽기 전용 Dataset — 참고 평가 전용, augmentation 없음.

    split: 'val' = Build 1, 'train' = Builds 2-5 (v4 의 index 를 그대로 쓴다).
    """

    def __init__(self, split: str = "val", img_size: int = cfg.IMG_SIZE):
        self.split = split
        self.img_size = img_size
        self.v1_root = cfg.v1_cache_dir(img_size)
        self.v4_root = cfg.v4_cache_dir(img_size)

        idx_path = self.v4_root / f"{split}_index.npz"
        if not idx_path.exists():
            raise FileNotFoundError(
                f"v4 label cache index not found: {idx_path}\n"
                "참고 평가는 v4 의 라벨 캐시를 읽는다 — `DEFSEG_V4_PHASE=build_cache` 를 먼저 실행할 것."
            )
        d = np.load(idx_path, allow_pickle=False)
        self.build_ids = d["build_ids"]
        self.cache_rows = d["cache_rows"]

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

    def __getitem__(self, idx: int) -> dict:
        build = str(self.build_ids[idx])
        row = int(self.cache_rows[idx])
        v1m = self._open_v1(build)
        lab_mm = self._open_v4_label(build)
        i0 = np.array(v1m["v0"][row])
        i1 = np.array(v1m["v1"][row])
        ann = np.array(lab_mm[row])
        return {
            "img0": torch.from_numpy(normalize_image(i0)),
            "img1": torch.from_numpy(normalize_image(i1)),
            "label": torch.from_numpy(ann.astype(np.int64)),
            "build_id": build,
            "cache_row": row,
        }
