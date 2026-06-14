"""v2 DSCNN_Dataset Stage 2 dataset — 8-class + BJ source 추가 + cross-validation 지원.

PLAN_v2 §5.2 참조.

  - 8 source (LPBF 6 + BJ 2, EBPBF 제외) 통합 enumerate
  - native class → ORNL 12-class → v2 8-class (2단계 매핑)
  - fold k 기준 train/val 분리 (leave-one-source-out)
  - v2 augmentation 적용
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

# 공통 함수 재사용 (common)
from ...common.data.image_utils import (
    normalize_image,
    resize_image_uint8,
    resize_label,
)
from .. import config_v2 as cfg
from .augmentation import augment_v2


@dataclass
class SampleSpecV2:
    source_name: str
    img0_path: Path
    img1_path: Path
    ann_path: Path
    mapping_key: str


def enumerate_samples_v2(
    sources: list[dict] | None = None,
) -> list[SampleSpecV2]:
    """8 source 의 (img0, img1, ann) 튜플 모두 나열."""
    if sources is None:
        sources = cfg.DSCNN_TRAIN_SOURCES_V2
    samples: list[SampleSpecV2] = []
    for src in sources:
        root = src["root"]
        ann_dir = root / "annotations"
        v0_dir = root / "data" / "visible" / "0"
        v1_dir = root / "data" / "visible" / "1"
        if not ann_dir.is_dir():
            print(f"[v2/skip] {src['name']}: no annotations dir at {ann_dir}")
            continue
        for ann_file in sorted(ann_dir.glob("*.npy")):
            v0 = v0_dir / f"{ann_file.stem}.tif"
            v1 = v1_dir / f"{ann_file.stem}.tif"
            if not (v0.is_file() and v1.is_file()):
                continue
            samples.append(SampleSpecV2(
                source_name=src["name"],
                img0_path=v0, img1_path=v1, ann_path=ann_file,
                mapping_key=src["mapping_key"],
            ))
    return samples


def remap_label_v2(ann_native: np.ndarray, mapping_key: str) -> np.ndarray:
    """native class ID → v2 8-class. 2 단계 매핑:
       native → ORNL 12-class (MATERIAL_TO_ORNL_V2) → v2 8-class (ORNL_12_TO_NEW_8)
    """
    mapping = cfg.MATERIAL_TO_ORNL_V2[mapping_key]
    out = np.full_like(ann_native, fill_value=cfg.IGNORE_INDEX, dtype=np.int8)
    # 원본 unlabeled (-1) 보존
    out[ann_native == -1] = cfg.IGNORE_INDEX
    for native_id, ornl_12_id in mapping.items():
        if ornl_12_id == cfg.IGNORE_INDEX:
            continue
        new_id = cfg.ORNL_12_TO_NEW_8.get(ornl_12_id, cfg.IGNORE_INDEX)
        if new_id == cfg.IGNORE_INDEX:
            continue
        out[ann_native == native_id] = new_id
    return out


def split_train_val_by_fold(
    fold_id: int,
    sources: list[dict] | None = None,
) -> tuple[list[SampleSpecV2], list[SampleSpecV2], str]:
    """8-fold leave-one-source-out.

    Args:
        fold_id: 0..7. fold_id 번째 source 가 val, 나머지 7 이 train.
        sources: default = cfg.DSCNN_TRAIN_SOURCES_V2

    Returns:
        (train_specs, val_specs, val_source_name)
    """
    if sources is None:
        sources = cfg.DSCNN_TRAIN_SOURCES_V2
    if not (0 <= fold_id < len(sources)):
        raise ValueError(f"fold_id {fold_id} out of range [0, {len(sources)})")
    val_source = sources[fold_id]
    train_sources = [s for i, s in enumerate(sources) if i != fold_id]
    train_specs = enumerate_samples_v2(train_sources)
    val_specs = enumerate_samples_v2([val_source])
    return train_specs, val_specs, val_source["name"]


class DefSegDSCNNDatasetV2(Dataset):
    """v2 Stage 2 dataset (DSCNN_Dataset).

    - 8 source 의 사람 GT 라벨 → 8-class 재매핑
    - 1036 resize + augmentation
    - 8-fold cross-validation 지원 (외부 split_train_val_by_fold 가 fold 별 specs 반환)
    """

    def __init__(
        self,
        samples: list[SampleSpecV2],
        img_size: int = cfg.IMG_SIZE,
        training: bool = True,
    ):
        self.samples = samples
        self.img_size = img_size
        self.training = training

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> dict:
        spec = self.samples[idx]
        img0 = np.array(Image.open(spec.img0_path))
        img1 = np.array(Image.open(spec.img1_path))
        ann = remap_label_v2(np.load(spec.ann_path), spec.mapping_key)

        sz = self.img_size
        img0 = resize_image_uint8(img0, sz)
        img1 = resize_image_uint8(img1, sz)
        ann = resize_label(ann, sz)

        # v2 augmentation
        img0, img1, ann = augment_v2(img0, img1, ann, training=self.training)

        return {
            "img0": torch.from_numpy(normalize_image(img0)),
            "img1": torch.from_numpy(normalize_image(img1)),
            "label": torch.from_numpy(ann.astype(np.int64)),
            "source": spec.source_name,
        }


def compute_class_counts_v2(
    specs: list[SampleSpecV2], n_classes: int = cfg.N_CLASSES_V2,
) -> np.ndarray:
    """주어진 specs 전체에서 v2 8-class 의 픽셀 count 집계 (class weight 계산용)."""
    counts = np.zeros(n_classes, dtype=np.int64)
    for s in specs:
        ann = remap_label_v2(np.load(s.ann_path), s.mapping_key)
        valid = ann[ann >= 0]
        u, c = np.unique(valid, return_counts=True)
        for k, v in zip(u, c):
            counts[k] += v
    return counts
