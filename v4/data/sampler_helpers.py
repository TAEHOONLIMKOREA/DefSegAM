"""v4 Class-Aware Layer Sampler helpers — PLAN_v4 §4.

per-layer per-class 픽셀 카운트 (pix_per_layer) → layer 별 weight 계산.

  class_w[c]  = 1 / max(total_pix[c], eps) ** α
  layer_w[ℓ] = Σ_c class_w[c] × pix[ℓ, c]   + eps

rare class 픽셀 많은 layer 가 큰 weight → DistributedWeightedSampler 가 자주 추출.
"""
from __future__ import annotations

import numpy as np

from .. import config_v4 as cfg


def compute_class_weights(
    pix_per_layer: np.ndarray,
    alpha: float = cfg.CLASS_AWARE_ALPHA,
    eps: float = cfg.CLASS_AWARE_EPS,
) -> np.ndarray:
    """class 별 inverse-frequency weight.

    Args:
        pix_per_layer: (n_layers, n_classes) int64 — per-layer per-class 픽셀 카운트
        alpha: 강조 강도. 1.0 = inverse-frequency, 0.5 = sqrt-inv
        eps: 분모 0 회피

    Returns:
        class_w: (n_classes,) float32. 합 = 1 로 정규화.
    """
    total_pix = pix_per_layer.sum(axis=0).astype(np.float64)  # (C,)
    safe = np.maximum(total_pix, eps)
    w = 1.0 / np.power(safe, alpha)
    s = w.sum()
    if s > 0:
        w = w / s
    return w.astype(np.float32)


def compute_layer_weights(
    pix_per_layer: np.ndarray,
    class_w: np.ndarray | None = None,
    alpha: float = cfg.CLASS_AWARE_ALPHA,
    eps: float = cfg.CLASS_AWARE_EPS,
) -> np.ndarray:
    """layer 별 sampler weight (class-aware).

    layer_w[ℓ] = Σ_c class_w[c] × pix[ℓ, c] + eps

    Args:
        pix_per_layer: (n_layers, n_classes) int64
        class_w: pre-computed class weights. None 면 compute_class_weights 호출
        alpha, eps: class_w None 일 때 사용

    Returns:
        layer_w: (n_layers,) float32. 합이 0 이 아니도록 eps 더함.
    """
    if class_w is None:
        class_w = compute_class_weights(pix_per_layer, alpha=alpha, eps=eps)
    w = pix_per_layer.astype(np.float64) @ class_w.astype(np.float64) + eps
    return w.astype(np.float32)


def compute_sample_weights_stage2(
    sample_pix: np.ndarray,
    source_idx: np.ndarray,
    n_sources: int,
    n_samples_per_source: np.ndarray,
    alpha: float = cfg.CLASS_AWARE_ALPHA,
    eps: float = cfg.CLASS_AWARE_EPS,
) -> np.ndarray:
    """Stage 2 의 combined weight = source 균등 × class-aware (곱).

    Args:
        sample_pix: (n_samples, n_classes) int64
        source_idx: (n_samples,) int — sample 의 source 인덱스 (0..n_sources-1)
        n_sources: source 개수
        n_samples_per_source: (n_sources,) int — 각 source 의 sample 수

    Returns:
        combined_w: (n_samples,) float32
    """
    # source 균등 weight: 한 sample 의 weight = 1 / (n_sources × n_in_source)
    src_w = np.zeros(len(source_idx), dtype=np.float64)
    for i in range(n_sources):
        n_in_src = max(int(n_samples_per_source[i]), 1)
        src_w[source_idx == i] = 1.0 / (n_sources * n_in_src)

    # class-aware weight
    class_w = compute_class_weights(sample_pix, alpha=alpha, eps=eps)
    cls_w = sample_pix.astype(np.float64) @ class_w.astype(np.float64)

    combined = src_w * cls_w + eps
    return combined.astype(np.float32)
