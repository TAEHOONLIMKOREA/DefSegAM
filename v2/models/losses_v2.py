"""v2 신규 손실 함수 — DSCNN_Summary.md §5.2, §5.3 (Eq. 4, Eq. 6).

기존 focal_loss / sqrt_inv_class_weight 는 [DefSeg_AM.common.models.losses](../../common/models/losses.py)
에서 그대로 import 해서 사용.
"""
from __future__ import annotations

import numpy as np
import torch
import torch.nn.functional as F


def hard_bootstrap_loss(
    logits: torch.Tensor,
    target: torch.Tensor,
    lambda_trust: float = 0.8,
    alpha_weight: torch.Tensor | None = None,
    ignore_index: int = -1,
    eps: float = 1e-8,
) -> torch.Tensor:
    """Hard-bootstrapping loss (Reed et al., DSCNN Eq. 4).

    E = - Σ_k (λ·t_k + (1-λ)·z_k) · log(q_k + ε)
    where z_k = onehot(argmax(q)).

    Noisy label 대응 — 모델이 confident 한 mislabel pixel 의 gradient 를 줄여서
    teacher pred (DSCNN model 의 출력) 의 noise 흡수.

    Args:
        logits: (B, C, H, W) raw logits
        target: (B, H, W) int64, IGNORE 는 ignore_index
        lambda_trust: GT 신뢰도 ∈ [0,1]. 1=standard CE, <1=model self-trust
        alpha_weight: (C,) per-class weight. None 가능.
        ignore_index: 무시할 라벨 값
        eps: log 안정성용

    Returns:
        scalar loss (valid pixel 평균)
    """
    B, C, H, W = logits.shape
    valid = (target != ignore_index)
    if valid.sum() == 0:
        return logits.sum() * 0.0

    log_q = F.log_softmax(logits, dim=1)                # (B, C, H, W)
    q = log_q.exp()
    # z = onehot(argmax q) — model 의 self prediction
    z = F.one_hot(q.argmax(dim=1), num_classes=C).permute(0, 3, 1, 2).float()
    # t = onehot(target) — GT one-hot (IGNORE 는 0 으로 처리 후 mask 로 제외)
    target_safe = target.clamp(min=0)
    t = F.one_hot(target_safe, num_classes=C).permute(0, 3, 1, 2).float()

    mix = lambda_trust * t + (1.0 - lambda_trust) * z   # (B, C, H, W)
    # 안정성을 위해 log(q + eps) 사용 (log_softmax 는 eps 미사용)
    nll_per_pix = -(mix * torch.log(q + eps)).sum(dim=1)  # (B, H, W)

    if alpha_weight is not None:
        # per-class weight 를 target 의 위치별로 적용
        w_per_pix = alpha_weight[target_safe]            # (B, H, W)
        nll_per_pix = nll_per_pix * w_per_pix

    return nll_per_pix[valid].mean()


def median_inv_class_weight(
    counts: np.ndarray,
    clip: float = 10.0,
) -> torch.Tensor:
    """DSCNN 원본 (Eq. 6): w_k = Median({f}) / f_k.

    Median frequency 기준 inverse-frequency. sqrt-inv 보다 rare class 에 더 큰 weight 부여.

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
