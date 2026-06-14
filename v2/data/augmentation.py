"""v2 Augmentation — PLAN_v2 §4 + DSCNN_Summary §6 종합.

Recoater Hopping + Streaking → Recoater Disturbance 통합 덕에 **D4 group
rotation/flip 가능**. + Cyclic shift (사용자 신규 제안) + DSCNN 원본의
Gaussian noise + intensity shift.

모든 aug 는 image (visible/0, visible/1) 와 label (annotation) 을 **동시에**
변환하여 픽셀 단위 정합 유지.

사용:
    from DefSeg_AM.v2.data.augmentation import augment_v2
    i0, i1, ann = augment_v2(i0, i1, ann, training=True)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .. import config_v2 as cfg


def _d4_rot_flip(
    i0: np.ndarray, i1: np.ndarray, ann: np.ndarray,
    k: int, do_flip: bool,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """D4 group: k ∈ {0,1,2,3} (90° rotation count) + horizontal flip."""
    if k > 0:
        i0 = np.rot90(i0, k=k).copy()
        i1 = np.rot90(i1, k=k).copy()
        ann = np.rot90(ann, k=k).copy()
    if do_flip:
        i0 = i0[:, ::-1].copy()
        i1 = i1[:, ::-1].copy()
        ann = ann[:, ::-1].copy()
    return i0, i1, ann


def _cyclic_shift(
    i0: np.ndarray, i1: np.ndarray, ann: np.ndarray,
    dx: int, dy: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """np.roll 기반 x/y shift. 사라진 영역은 반대쪽에서 채움 (torus topology)."""
    if dx == 0 and dy == 0:
        return i0, i1, ann
    i0 = np.roll(i0, shift=(dy, dx), axis=(0, 1))
    i1 = np.roll(i1, shift=(dy, dx), axis=(0, 1))
    ann = np.roll(ann, shift=(dy, dx), axis=(0, 1))
    return i0, i1, ann


def _gaussian_noise(
    i0: np.ndarray, i1: np.ndarray, sigma_pct: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """DSCNN Eq. § — image stack A, C 에 σ²=pct%·DR² 추가 (B 는 보존).

    여기서는 dual visible (after-melt + after-spread) 둘 다에 적용.
    (DSCNN 의 stack A=global, B=local, C=regional 분리와 다름. 우리는 dual visible 만.)
    """
    if sigma_pct <= 0:
        return i0, i1
    DR = 255.0  # 8-bit dynamic range
    noise_std = DR * sigma_pct
    i0 = np.clip(i0.astype(np.float32) + np.random.randn(*i0.shape) * noise_std, 0, 255).astype(np.uint8)
    i1 = np.clip(i1.astype(np.float32) + np.random.randn(*i1.shape) * noise_std, 0, 255).astype(np.uint8)
    return i0, i1


def _intensity_shift(
    i0: np.ndarray, i1: np.ndarray, shift_pct: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """DSCNN Eq. § — image stack 의 평균 강도 ±10% 의 DR 만큼 이동.

    additive shift (multiplicative 가 아님).
    """
    if shift_pct == 0.0:
        return i0, i1
    DR = 255.0
    shift = DR * shift_pct
    i0 = np.clip(i0.astype(np.float32) + shift, 0, 255).astype(np.uint8)
    i1 = np.clip(i1.astype(np.float32) + shift, 0, 255).astype(np.uint8)
    return i0, i1


def _brightness_jitter(
    i0: np.ndarray, i1: np.ndarray, scale: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """v1 의 기존 multiplicative jitter (±15%). 명암비도 약간 바꿈."""
    if scale == 1.0:
        return i0, i1
    i0 = np.clip(i0.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    i1 = np.clip(i1.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return i0, i1


def augment_v2(
    i0: np.ndarray,
    i1: np.ndarray,
    ann: np.ndarray,
    training: bool,
    rng: np.random.Generator | None = None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """v2 augmentation 종합 적용.

    Args:
        i0, i1: uint8 (H, W) — after-melt / after-spread images
        ann:    int8  (H, W) — label (8-class + IGNORE=-1)
        training: True 면 random aug 적용. False 면 통과.
        rng: 외부 RNG (재현성 필요 시). None 이면 numpy default.

    Returns:
        (i0_aug, i1_aug, ann_aug). aug 후에도 동일 shape (H, W) 보장.
    """
    if not training:
        return i0, i1, ann

    if rng is None:
        rng = np.random.default_rng()

    # 1. D4 group rotation + flip — 라벨도 동일하게 변환
    if cfg.ENABLE_D4_AUGMENTATION:
        k = int(rng.integers(0, 4))             # 0, 90, 180, 270
        do_flip = bool(rng.integers(0, 2) == 1)  # 50% horizontal flip
        i0, i1, ann = _d4_rot_flip(i0, i1, ann, k, do_flip)

    # 2. Cyclic shift — 라벨도 동일하게 roll
    if cfg.ENABLE_CYCLIC_SHIFT:
        max_shift = int(i0.shape[0] * cfg.CYCLIC_SHIFT_MAX_FRAC)
        dx = int(rng.integers(-max_shift, max_shift + 1))
        dy = int(rng.integers(-max_shift, max_shift + 1))
        i0, i1, ann = _cyclic_shift(i0, i1, ann, dx, dy)

    # 3. Gaussian noise (image only) — DSCNN 원본
    sigma_pct = float(rng.choice(cfg.DSCNN_NOISE_SIGMA_PCT_CHOICES)) / 100.0
    i0, i1 = _gaussian_noise(i0, i1, sigma_pct)

    # 4. Mean intensity shift (image only) — DSCNN 원본
    shift_pct = float(rng.choice(cfg.DSCNN_INTENSITY_SHIFT_PCT_CHOICES))
    i0, i1 = _intensity_shift(i0, i1, shift_pct)

    # 5. Brightness multiplicative jitter (v1 유지)
    if cfg.ENABLE_BRIGHTNESS_JITTER and bool(rng.integers(0, 2) == 1):
        lo, hi = cfg.BRIGHTNESS_JITTER_RANGE
        scale = float(rng.uniform(lo, hi))
        i0, i1 = _brightness_jitter(i0, i1, scale)

    return i0, i1, ann
