"""v5 Augmentation — v4 와 같은 파이프라인 (PLAN_v5 §5.5).

  - Copy-Paste (rare class object 합성; supplier 는 학습 분할의 사람 GT)
  - cyclic shift (1-px uniform ±IMG/4, prob 0.5)
  - flip LR (50%) + flip UD (50%) 독립, 180° rot (50%)
  - Gaussian noise / intensity shift (DSCNN 논문의 증강 방식) / brightness jitter (이미지만)

모든 기하 변환은 image (visible/0, visible/1) 와 label 을 동시에 변환하여 픽셀 단위 정합 유지.
입력 배열은 고치지 않는다 (RAM 에 올려 둔 원본을 여러 번 다시 쓰기 때문).

사용:
    from DefSeg_AM.v5.data.augmentation import augment_v5
    i0, i1, ann = augment_v5(i0, i1, ann, training=True, rng=rng)
"""
from __future__ import annotations

from typing import Tuple

import numpy as np

from .. import config_v5 as cfg


def _flip_lr(i0, i1, ann):
    return i0[:, ::-1].copy(), i1[:, ::-1].copy(), ann[:, ::-1].copy()


def _flip_ud(i0, i1, ann):
    return i0[::-1, :].copy(), i1[::-1, :].copy(), ann[::-1, :].copy()


def _rot180(i0, i1, ann):
    return np.rot90(i0, k=2).copy(), np.rot90(i1, k=2).copy(), np.rot90(ann, k=2).copy()


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
    rng: np.random.Generator,
) -> Tuple[np.ndarray, np.ndarray]:
    """σ²=pct%·DR² 의 additive Gaussian (image only)."""
    if sigma_pct <= 0:
        return i0, i1
    DR = 255.0
    noise_std = DR * sigma_pct
    i0 = np.clip(i0.astype(np.float32) + rng.standard_normal(i0.shape) * noise_std, 0, 255).astype(np.uint8)
    i1 = np.clip(i1.astype(np.float32) + rng.standard_normal(i1.shape) * noise_std, 0, 255).astype(np.uint8)
    return i0, i1


def _intensity_shift(
    i0: np.ndarray, i1: np.ndarray, shift_pct: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """평균 강도 ±10% DR additive shift (image only)."""
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
    """multiplicative jitter (×U(0.85, 1.15))."""
    if scale == 1.0:
        return i0, i1
    i0 = np.clip(i0.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    i1 = np.clip(i1.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    return i0, i1


def augment_v5(
    i0: np.ndarray,
    i1: np.ndarray,
    ann: np.ndarray,
    training: bool,
    rng: np.random.Generator | None = None,
    cp_sampler=None,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """v5 augmentation — v4 의 augment_v4 와 같은 순서 · 같은 확률.

    Args:
        i0, i1: uint8 (H, W) — after-melt / after-spread images
        ann:    int8  (H, W) — label (11-class + IGNORE=-1)
        training: True 면 random aug 적용. False 면 통과.
        rng: 외부 RNG (worker 별로 고정해 넘긴다).
        cp_sampler: data.copy_paste.CopyPasteSampler. None 이면 Copy-Paste 적용 안 함.

    적용 순서:
        Copy-Paste → cyclic shift → flip LR/UD → rot180 → noise/intensity → brightness
    """
    if not training:
        return i0, i1, ann

    if rng is None:
        rng = np.random.default_rng()

    # 0. Copy-Paste (rare class object 합성) — 가장 먼저 적용해 후속 기하 augmentation 으로 다양화
    if cp_sampler is not None and cfg.CP_ENABLE:
        i0, i1, ann = cp_sampler.paste(i0, i1, ann, rng)

    # 1. Cyclic shift (1-px uniform) — 라벨도 동일하게 roll
    if cfg.ENABLE_CYCLIC_SHIFT and rng.random() < cfg.CYCLIC_SHIFT_PROB:
        max_shift = int(i0.shape[0] * cfg.CYCLIC_SHIFT_MAX_FRAC)
        dx = int(rng.integers(-max_shift, max_shift + 1))
        dy = int(rng.integers(-max_shift, max_shift + 1))
        i0, i1, ann = _cyclic_shift(i0, i1, ann, dx, dy)

    # 2. Flip LR / UD — 독립 50%
    if cfg.ENABLE_FLIP_LR and rng.random() < 0.5:
        i0, i1, ann = _flip_lr(i0, i1, ann)
    if cfg.ENABLE_FLIP_UD and rng.random() < 0.5:
        i0, i1, ann = _flip_ud(i0, i1, ann)

    # 3. 180° rotation (50%)
    if cfg.ENABLE_ROT180 and rng.random() < 0.5:
        i0, i1, ann = _rot180(i0, i1, ann)

    # 4. Gaussian noise (image only)
    sigma_pct = float(rng.choice(cfg.DSCNN_NOISE_SIGMA_PCT_CHOICES)) / 100.0
    i0, i1 = _gaussian_noise(i0, i1, sigma_pct, rng)

    # 5. Mean intensity shift (image only)
    shift_pct = float(rng.choice(cfg.DSCNN_INTENSITY_SHIFT_PCT_CHOICES))
    i0, i1 = _intensity_shift(i0, i1, shift_pct)

    # 6. Brightness multiplicative jitter — 50%
    if cfg.ENABLE_BRIGHTNESS_JITTER and rng.random() < 0.5:
        lo, hi = cfg.BRIGHTNESS_JITTER_RANGE
        scale = float(rng.uniform(lo, hi))
        i0, i1 = _brightness_jitter(i0, i1, scale)

    return i0, i1, ann
