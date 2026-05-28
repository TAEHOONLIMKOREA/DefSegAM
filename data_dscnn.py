"""Stage 2 dataset: DSCNN_Dataset annotations (human GT) → ORNL 12-class 재매핑.

PLAN.md §3.2 / §5.2 참조.

- Source: 6 개 LPBF dataset (seung_dscnn 와 동일; EBPBF/BJ 제외)
- annotations/*.npy (재료별 native class) → ORNL 12-class (MATERIAL_TO_ORNL)
- val source: v2022_Maraging
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import config
from .data_ornl import (
    normalize_image,
    resize_image_uint8,
    resize_label,
)


@dataclass
class SampleSpec:
    source_name: str
    img0_path: Path
    img1_path: Path
    ann_path: Path
    mapping_key: str


def enumerate_samples(sources: list[dict] | None = None) -> list[SampleSpec]:
    """학습 가능한 (img0, img1, ann) 튜플 모두 나열."""
    if sources is None:
        sources = config.DSCNN_TRAIN_SOURCES
    samples: list[SampleSpec] = []
    for src in sources:
        root = src["root"]
        ann_dir = root / "annotations"
        v0_dir = root / "data" / "visible" / "0"
        v1_dir = root / "data" / "visible" / "1"
        if not ann_dir.is_dir():
            print(f"[skip] {src['name']}: no annotations dir at {ann_dir}")
            continue
        for ann_file in sorted(ann_dir.glob("*.npy")):
            v0 = v0_dir / f"{ann_file.stem}.tif"
            v1 = v1_dir / f"{ann_file.stem}.tif"
            if not (v0.is_file() and v1.is_file()):
                continue
            samples.append(SampleSpec(
                source_name=src["name"],
                img0_path=v0, img1_path=v1, ann_path=ann_file,
                mapping_key=src["mapping_key"],
            ))
    return samples


def remap_label(ann_native: np.ndarray, mapping_key: str) -> np.ndarray:
    """재료별 native class ID 맵 → ORNL 12-class. 매핑 안 되는 픽셀은 IGNORE."""
    mapping = config.MATERIAL_TO_ORNL[mapping_key]
    out = np.full_like(ann_native, fill_value=config.IGNORE_INDEX, dtype=np.int8)
    out[ann_native == -1] = config.IGNORE_INDEX  # 원본 unlabeled 보존
    for native_id, ornl_id in mapping.items():
        if ornl_id == config.IGNORE_INDEX:
            continue
        out[ann_native == native_id] = ornl_id
    return out


def split_train_val(
    sources: list[dict] | None = None,
    val_source_names: list[str] | None = None,
) -> tuple[list[SampleSpec], list[SampleSpec]]:
    if sources is None:
        sources = config.DSCNN_TRAIN_SOURCES
    if val_source_names is None:
        val_source_names = config.DSCNN_VAL_SOURCE_NAMES
    train_src = [s for s in sources if s["name"] not in val_source_names]
    val_src = [s for s in sources if s["name"] in val_source_names]
    return enumerate_samples(train_src), enumerate_samples(val_src)


class DefSegDSCNNDataset(Dataset):
    """layer-단위 sample. 모든 이미지/라벨을 img_size×img_size 로 resize."""

    def __init__(
        self,
        samples: list[SampleSpec],
        img_size: int = config.IMG_SIZE,
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
        ann = remap_label(np.load(spec.ann_path), spec.mapping_key)

        sz = self.img_size
        img0 = resize_image_uint8(img0, sz)
        img1 = resize_image_uint8(img1, sz)
        ann = resize_label(ann, sz)

        # brightness jitter only (recoater 방향성 보존)
        if self.training and np.random.random() < 0.5:
            scale = np.random.uniform(0.85, 1.15)
            img0 = np.clip(img0.astype(np.float32) * scale, 0, 255).astype(np.uint8)
            img1 = np.clip(img1.astype(np.float32) * scale, 0, 255).astype(np.uint8)

        return {
            "img0": torch.from_numpy(normalize_image(img0)),
            "img1": torch.from_numpy(normalize_image(img1)),
            "label": torch.from_numpy(ann.astype(np.int64)),
            "source": spec.source_name,
        }


def compute_class_counts(
    specs: list[SampleSpec], n_classes: int = config.N_CLASSES,
) -> np.ndarray:
    """Stage 2 class weight 계산용 — annotation 전체를 한 번 읽어 class count 집계."""
    counts = np.zeros(n_classes, dtype=np.int64)
    for s in specs:
        ann = remap_label(np.load(s.ann_path), s.mapping_key)
        valid = ann[ann >= 0]
        u, c = np.unique(valid, return_counts=True)
        for k, v in zip(u, c):
            counts[k] += v
    return counts
