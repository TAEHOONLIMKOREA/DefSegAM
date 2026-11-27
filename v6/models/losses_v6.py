"""v6 손실 보조 — class weight (v5 사본).

학습 손실은 표준 cross-entropy + class weight 다 (v4 Stage 2 · v5 와 같음, PLAN_v6 §5.6).
sqrt-inv weight 는 [DefSeg_AM.common.models.losses](../../common/models/losses.py) 의 것을 그대로 쓴다.
"""
from __future__ import annotations

import numpy as np
import torch


def median_inv_class_weight(
    counts: np.ndarray,
    clip: float = 10.0,
) -> torch.Tensor:
    """Median frequency 기준 inverse-frequency: w_k = Median({f}) / f_k (v4 · v5 사본).

    sqrt-inv 보다 rare class 에 더 큰 weight 부여.

    Args:
        counts: (C,) per-class pixel count
        clip: 최대 weight (안전장치)

    Returns:
        (C,) float32; counts == 0 인 class 는 weight=0
    """
    counts = np.asarray(counts, dtype=np.float64)
    total = counts.sum()
    if total <= 0:
        return torch.zeros(len(counts), dtype=torch.float32)
    freq = counts / total
    nonzero = freq[freq > 0]
    if nonzero.size == 0:
        return torch.zeros(len(counts), dtype=torch.float32)
    median_f = float(np.median(nonzero))
    w = np.where(freq > 0, median_f / np.maximum(freq, 1e-8), 0.0)
    w = np.clip(w, 0.0, clip)
    return torch.from_numpy(w.astype(np.float32))
