"""Focal Loss + sqrt-inverse-frequency class weight (PLAN §5.1).

Focal Loss 는 Stage 1 (KD pretrain) 에서만 사용. Stage 2 는 표준 CE.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def focal_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    gamma: float = 2.0,
    alpha_weight: torch.Tensor | None = None,
    ignore_index: int = -1,
) -> torch.Tensor:
    """Focal Loss with per-class α weight.

    Args:
        logits: (B, C, H, W) raw logits
        target: (B, H, W) int64, IGNORE 는 ignore_index
        gamma: focusing parameter (PLAN default 2.0)
        alpha_weight: (C,) per-class weight (sqrt-inv freq). None 이면 1.
        ignore_index: 무시할 라벨 값

    Returns:
        scalar loss (valid pixel 평균)
    """
    ce = F.cross_entropy(
        logits, target,
        weight=alpha_weight,
        ignore_index=ignore_index,
        reduction="none",
    )  # (B, H, W); ignore pixel 은 0
    pt = torch.exp(-ce)  # 정답 class 의 확률 (alpha weight 효과는 ce 자체에 이미 반영)
    fl = (1.0 - pt) ** gamma * ce

    valid = (target != ignore_index)
    n_valid = valid.sum()
    if n_valid == 0:
        return logits.sum() * 0.0  # graph 유지용 0 (안전장치)
    return fl[valid].mean()


def sqrt_inv_class_weight(
    counts: np.ndarray,
    clip: float = 50.0,
) -> torch.Tensor:
    """sqrt((1 - f) / f) — seung_dscnn/data.py:158-173 동일 공식.

    Args:
        counts: (C,) per-class valid pixel count
        clip: 최대 weight (rare class 의 폭주 방지)

    Returns:
        (C,) float32 tensor; counts == 0 인 class 는 weight=0 (학습에 영향 없음)
    """
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return torch.zeros(len(counts), dtype=torch.float32)
    freq = counts / total
    w = np.where(freq > 0, np.sqrt((1.0 - freq) / np.maximum(freq, 1e-8)), 0.0)
    w = np.clip(w, 0.0, clip)
    return torch.from_numpy(w.astype(np.float32))
