"""공용 이미지/라벨 유틸 — v1·v2 공유 (seung_dscnn carry-over, PLAN §6.1).

ORNL float32 이미지 → uint8 정규화, resize, ImageNet 정규화 등.
v1 (data_ornl.py), v2 (data_ornl_v2.py / data_dscnn_v2.py) 양쪽이 임포트한다.
"""
from __future__ import annotations

import numpy as np
from PIL import Image

from ..config import IMAGENET_MEAN, IMAGENET_STD


def normalize_image(img_uint8: np.ndarray) -> np.ndarray:
    """uint8 (H, W) grayscale → (3, H, W) float32, ImageNet 정규화."""
    arr = img_uint8.astype(np.float32) / 255.0
    arr = np.stack([arr, arr, arr], axis=0)
    mean = np.array(IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.array(IMAGENET_STD, dtype=np.float32)[:, None, None]
    return (arr - mean) / std


def resize_image_uint8(arr: np.ndarray, size: int) -> np.ndarray:
    return np.array(Image.fromarray(arr).resize((size, size), Image.BILINEAR))


def resize_label(ann: np.ndarray, size: int) -> np.ndarray:
    # int8 은 PIL 이 직접 지원 못 함 → int16 우회
    return np.array(
        Image.fromarray(ann.astype(np.int16)).resize((size, size), Image.NEAREST)
    ).astype(np.int8)


def ornl_image_to_uint8(img: np.ndarray) -> np.ndarray:
    """ORNL float32 → uint8. percentile (1, 99) 기반 per-image normalize."""
    if img.dtype == np.uint8:
        return img
    nz = img[img > 0] if (img > 0).any() else img
    lo, hi = np.percentile(nz, [1, 99])
    if hi <= lo:
        hi = lo + 1
    norm = np.clip((img - lo) / (hi - lo), 0, 1)
    return (norm * 255).astype(np.uint8)
