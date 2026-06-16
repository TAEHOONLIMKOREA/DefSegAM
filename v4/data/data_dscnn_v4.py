"""v3 DSCNN_Dataset Stage 2 dataset — 10-class + BJ source + 전수 학습 (CV 없음).

PLAN_v4 §4.2 참조.

  - 8 source (LPBF 6 + BJ 2, EBPBF 제외) 통합 enumerate
  - native class → ORNL 12-class → v4 11-class (2단계 매핑)
  - 8 source 전부 train (val 분리 없음 — ORNL Build 1 로 별도 평가)
  - Replicate factor K 로 __len__ = K × N (epoch 당 effective sample 수 ↑)
  - v4 augmentation 적용
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
from .. import config_v4 as cfg
from .augmentation import augment_v4


@dataclass
class SampleSpecV4:
    source_name: str
    img0_path: Path
    img1_path: Path
    ann_path: Path
    mapping_key: str


def enumerate_samples_v4(
    sources: list[dict] | None = None,
) -> list[SampleSpecV4]:
    """8 source 의 (img0, img1, ann) 튜플 모두 나열."""
    if sources is None:
        sources = cfg.DSCNN_TRAIN_SOURCES_V4
    samples: list[SampleSpecV4] = []
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
            samples.append(SampleSpecV4(
                source_name=src["name"],
                img0_path=v0, img1_path=v1, ann_path=ann_file,
                mapping_key=src["mapping_key"],
            ))
    return samples


def remap_label_v4(ann_native: np.ndarray, mapping_key: str) -> np.ndarray:
    """native class ID → v4 11-class. 2 단계 매핑:
       native → ORNL 12-class (MATERIAL_TO_ORNL_V4) → v4 11-class (ORNL_12_TO_NEW_11)
    """
    mapping = cfg.MATERIAL_TO_ORNL_V4[mapping_key]
    out = np.full_like(ann_native, fill_value=cfg.IGNORE_INDEX, dtype=np.int8)
    out[ann_native == -1] = cfg.IGNORE_INDEX
    for native_id, ornl_12_id in mapping.items():
        if ornl_12_id == cfg.IGNORE_INDEX:
            continue
        new_id = cfg.ORNL_12_TO_NEW_11.get(ornl_12_id, cfg.IGNORE_INDEX)
        if new_id == cfg.IGNORE_INDEX:
            continue
        out[ann_native == native_id] = new_id
    return out


class DefSegDSCNNDatasetV4(Dataset):
    """v4 Stage 2 dataset — 8 source 전부 train + replicate factor K + Copy-Paste 지원.

    - DSCNN_Dataset GT 한 픽셀도 학습에서 빠지지 않음 (CV 없음)
    - __len__ = K × N → 한 epoch 에 같은 GT sample 을 K 번 다른 augmentation 으로
    - 평가는 ORNL Build 1 (Stage 2 의 별도 evaluation set, train_stage2 에서 처리)
    """

    def __init__(
        self,
        samples: list[SampleSpecV4],
        img_size: int = cfg.IMG_SIZE,
        training: bool = True,
        replicate_factor: int = 1,
        cp_sampler=None,
    ):
        self.samples = samples
        self.img_size = img_size
        self.training = training
        self.replicate_factor = max(1, int(replicate_factor))
        self.cp_sampler = cp_sampler

    def __len__(self) -> int:
        return len(self.samples) * self.replicate_factor

    def fetch_by_sample_idx(self, idx: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Copy-Paste supplier 용: sample index → (i0, i1, ann). augmentation 없이."""
        spec = self.samples[int(idx) % len(self.samples)]
        img0 = np.array(Image.open(spec.img0_path))
        img1 = np.array(Image.open(spec.img1_path))
        ann = remap_label_v4(np.load(spec.ann_path), spec.mapping_key)
        sz = self.img_size
        img0 = resize_image_uint8(img0, sz)
        img1 = resize_image_uint8(img1, sz)
        ann = resize_label(ann, sz)
        return img0, img1, ann

    def __getitem__(self, idx: int) -> dict:
        spec = self.samples[idx % len(self.samples)]
        img0 = np.array(Image.open(spec.img0_path))
        img1 = np.array(Image.open(spec.img1_path))
        ann = remap_label_v4(np.load(spec.ann_path), spec.mapping_key)

        sz = self.img_size
        img0 = resize_image_uint8(img0, sz)
        img1 = resize_image_uint8(img1, sz)
        ann = resize_label(ann, sz)

        # v4 augmentation (Copy-Paste + flip LR/UD + 180° + cyclic + DSCNN noise/intensity + brightness)
        img0, img1, ann = augment_v4(img0, img1, ann, training=self.training, cp_sampler=self.cp_sampler)

        return {
            "img0": torch.from_numpy(normalize_image(img0)),
            "img1": torch.from_numpy(normalize_image(img1)),
            "label": torch.from_numpy(ann.astype(np.int64)),
            "source": spec.source_name,
        }


def compute_class_counts_v4(
    specs: list[SampleSpecV4], n_classes: int = cfg.N_CLASSES_V4,
) -> np.ndarray:
    """주어진 specs 전체에서 v4 11-class 의 픽셀 count 집계 (class weight 계산용)."""
    counts = np.zeros(n_classes, dtype=np.int64)
    for s in specs:
        ann = remap_label_v4(np.load(s.ann_path), s.mapping_key)
        valid = ann[ann >= 0]
        u, c = np.unique(valid, return_counts=True)
        for k, v in zip(u, c):
            counts[k] += v
    return counts


def source_to_index_map(samples: list[SampleSpecV4]) -> dict[str, list[int]]:
    """source name → sample index 목록. WeightedSampler 의 source 별 균등 가중치 계산용."""
    out: dict[str, list[int]] = {}
    for i, s in enumerate(samples):
        out.setdefault(s.source_name, []).append(i)
    return out


def compute_source_balanced_weights(samples: list[SampleSpecV4]) -> np.ndarray:
    """source 별 균등 추출용 sample 가중치. 한 sample 의 weight = 1 / (n_source × n_in_source)."""
    src_idx = source_to_index_map(samples)
    n_src = len(src_idx)
    w = np.zeros(len(samples), dtype=np.float64)
    for src, idxs in src_idx.items():
        per = 1.0 / (n_src * len(idxs))
        w[idxs] = per
    return w


def compute_sample_pix_per_class(
    samples: list[SampleSpecV4], n_classes: int = cfg.N_CLASSES_V4,
) -> np.ndarray:
    """각 sample 의 GT label 의 per-class 픽셀 카운트.

    Returns: (n_samples, n_classes) int64. class-aware sampler 와 Copy-Paste
    supplier index 작성에 사용.
    """
    out = np.zeros((len(samples), n_classes), dtype=np.int64)
    for i, s in enumerate(samples):
        ann = remap_label_v4(np.load(s.ann_path), s.mapping_key)
        valid = ann[ann >= 0]
        u, c = np.unique(valid, return_counts=True)
        for k, v in zip(u, c):
            out[i, k] = v
    return out


def build_stage2_cp_supplier_items(
    sample_pix: np.ndarray,
    rare_classes: tuple[int, ...] = cfg.CP_RARE_CLASSES_S2,
    min_component_px: int = cfg.CP_MIN_COMPONENT_PX,
) -> dict:
    """Stage 2 의 Copy-Paste supplier index.

    rare class 마다 `sample_pix[i, c] ≥ threshold` 인 sample index 모음.

    Returns: {class_id: [CPSupplierItem, ...]}  — copy_paste 의 sampler 형식
    """
    from .copy_paste import CPSupplierItem
    out: dict[int, list[CPSupplierItem]] = {}
    for c in rare_classes:
        idxs = np.where(sample_pix[:, c] >= min_component_px)[0]
        out[int(c)] = [
            CPSupplierItem(key=int(i), class_id=int(c), n_pixels=int(sample_pix[i, c]))
            for i in idxs
        ]
    return out
