"""v3 DSCNN_Dataset Stage 2 dataset — 10-class + BJ source + 전수 학습 (CV 없음).

PLAN_v3 §4.2 참조.

  - 8 source (LPBF 6 + BJ 2, EBPBF 제외) 통합 enumerate
  - native class → ORNL 12-class → v3 10-class (2단계 매핑)
  - 8 source 전부 train (val 분리 없음 — ORNL Build 1 로 별도 평가)
  - Replicate factor K 로 __len__ = K × N (epoch 당 effective sample 수 ↑)
  - v3 augmentation 적용
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ...common.data.image_utils import (
    normalize_image,
    resize_image_uint8,
    resize_label,
)
from .. import config_v3 as cfg
from .augmentation import augment_v3


@dataclass
class SampleSpecV3:
    source_name: str
    img0_path: Path
    img1_path: Path
    ann_path: Path
    mapping_key: str


def enumerate_samples_v3(
    sources: list[dict] | None = None,
) -> list[SampleSpecV3]:
    """8 source 의 (img0, img1, ann) 튜플 모두 나열."""
    if sources is None:
        sources = cfg.DSCNN_TRAIN_SOURCES_V3
    samples: list[SampleSpecV3] = []
    for src in sources:
        root = src["root"]
        ann_dir = root / "annotations"
        v0_dir = root / "data" / "visible" / "0"
        v1_dir = root / "data" / "visible" / "1"
        if not ann_dir.is_dir():
            print(f"[v3/skip] {src['name']}: no annotations dir at {ann_dir}")
            continue
        for ann_file in sorted(ann_dir.glob("*.npy")):
            v0 = v0_dir / f"{ann_file.stem}.tif"
            v1 = v1_dir / f"{ann_file.stem}.tif"
            if not (v0.is_file() and v1.is_file()):
                continue
            samples.append(SampleSpecV3(
                source_name=src["name"],
                img0_path=v0, img1_path=v1, ann_path=ann_file,
                mapping_key=src["mapping_key"],
            ))
    return samples


def remap_label_v3(ann_native: np.ndarray, mapping_key: str) -> np.ndarray:
    """native class ID → v3 10-class. 2 단계 매핑:
       native → ORNL 12-class (MATERIAL_TO_ORNL_V3) → v3 10-class (ORNL_12_TO_NEW_10)
    """
    mapping = cfg.MATERIAL_TO_ORNL_V3[mapping_key]
    out = np.full_like(ann_native, fill_value=cfg.IGNORE_INDEX, dtype=np.int8)
    out[ann_native == -1] = cfg.IGNORE_INDEX
    for native_id, ornl_12_id in mapping.items():
        if ornl_12_id == cfg.IGNORE_INDEX:
            continue
        new_id = cfg.ORNL_12_TO_NEW_10.get(ornl_12_id, cfg.IGNORE_INDEX)
        if new_id == cfg.IGNORE_INDEX:
            continue
        out[ann_native == native_id] = new_id
    return out


class DefSegDSCNNDatasetV3(Dataset):
    """v3 Stage 2 dataset — 8 source 전부 train + replicate factor K.

    - DSCNN_Dataset GT 한 픽셀도 학습에서 빠지지 않음 (CV 없음)
    - __len__ = K × N → 한 epoch 에 같은 GT sample 을 K 번 다른 augmentation 으로
    - 평가는 ORNL Build 1 (Stage 2 의 별도 evaluation set, train_stage2 에서 처리)
    """

    def __init__(
        self,
        samples: list[SampleSpecV3],
        img_size: int = cfg.IMG_SIZE,
        training: bool = True,
        replicate_factor: int = 1,
    ):
        self.samples = samples
        self.img_size = img_size
        self.training = training
        self.replicate_factor = max(1, int(replicate_factor))

    def __len__(self) -> int:
        return len(self.samples) * self.replicate_factor

    def __getitem__(self, idx: int) -> dict:
        spec = self.samples[idx % len(self.samples)]
        img0 = np.array(Image.open(spec.img0_path))
        img1 = np.array(Image.open(spec.img1_path))
        ann = remap_label_v3(np.load(spec.ann_path), spec.mapping_key)

        sz = self.img_size
        img0 = resize_image_uint8(img0, sz)
        img1 = resize_image_uint8(img1, sz)
        ann = resize_label(ann, sz)

        # augmentation — idx 마다 다른 random sequence (replicate 의 K 번이 다 다르게)
        img0, img1, ann = augment_v3(img0, img1, ann, training=self.training)

        return {
            "img0": torch.from_numpy(normalize_image(img0)),
            "img1": torch.from_numpy(normalize_image(img1)),
            "label": torch.from_numpy(ann.astype(np.int64)),
            "source": spec.source_name,
        }


def compute_class_counts_v3(
    specs: list[SampleSpecV3], n_classes: int = cfg.N_CLASSES_V3,
) -> np.ndarray:
    """주어진 specs 전체에서 v3 10-class 의 픽셀 count 집계 (class weight 계산용)."""
    counts = np.zeros(n_classes, dtype=np.int64)
    for s in specs:
        ann = remap_label_v3(np.load(s.ann_path), s.mapping_key)
        valid = ann[ann >= 0]
        u, c = np.unique(valid, return_counts=True)
        for k, v in zip(u, c):
            counts[k] += v
    return counts


def source_to_index_map(samples: list[SampleSpecV3]) -> dict[str, list[int]]:
    """source name → sample index 목록. WeightedSampler 의 source 별 균등 가중치 계산용."""
    out: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        out.setdefault(s.source_name, []).append(i)
    return out


def compute_source_balanced_weights(samples: list[SampleSpecV3]) -> np.ndarray:
    """source 별 균등 추출용 sample 가중치. 한 sample 의 weight = 1 / (n_source × n_in_source)."""
    src_idx = source_to_index_map(samples)
    n_src = len(src_idx)
    w = np.zeros(len(samples), dtype=np.float64)
    for src, idxs in src_idx.items():
        per = 1.0 / (n_src * len(idxs))
        w[idxs] = per
    return w
