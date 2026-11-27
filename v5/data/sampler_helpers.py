"""v5 sampler weight — 소스 균등과 클래스 균등의 혼합 + 표본당 상한 (PLAN_v5 §5.4).

  u_i = 1 / (소스 수 × 그 소스의 표본 수)                      # 소스 균등, Σ u = 1
  c_i = (1 / 클래스 수) × Σ_c pix[i, c] / total_pix[c]          # 클래스 균등, Σ c = 1
  p_i = mix × u_i + (1 − mix) × c_i
  p_i ← min(p_i, max_repeat / N) 후 남은 몫을 나머지 표본에 비례 배분

c_i 는 v4 의 class-aware weight (α = 1.0) 와 같은 양이다: 클래스마다 같은 몫 (1 / 클래스 수) 을 갖고,
그 몫을 그 클래스 화소를 가진 레이어들이 화소 비율대로 나눠 갖는다.

v4 는 u_i × c_i (곱) 을 썼다. 사람 GT 125 레이어에 v4 의 식을 그대로 걸면 (2026-11-16 측정) 5 레이어가
추출의 39% 를 가져가고, 결함이 적은 소스 (17-4PH 14 레이어) 는 0.3% 로 떨어지며, 유효 표본 수 (1 / Σp²) 가
25 다. Stage 1 없이 GT 만으로 처음부터 배우는 v5 에서는 레이어 하나하나가 아쉬우므로 합 (혼합) 으로 바꾸고
상한을 둔다. 같은 데이터에서 v5 의 식은 상위 5 레이어 17%, 소스별 몫 7–18%, 유효 표본 수 75 다.

또 v4 의 식은 화소가 0 인 클래스가 있으면 그 클래스의 weight (1 / eps) 가 정규화를 지배해 전체가
균등 추출로 무너진다. v4 에서는 생기지 않았지만 v5 의 검증 fold 에서는 생기므로 (소스 하나를 빼면
그 소스에만 있는 클래스가 0 이 된다) 화소가 0 인 클래스는 계산에서 뺀다.
"""
from __future__ import annotations

import numpy as np

from .. import config_v5 as cfg


def source_balanced_weights(source_idx: np.ndarray, n_sources: int) -> np.ndarray:
    """소스 균등 weight. 한 표본의 weight = 1 / (표본이 있는 소스 수 × 그 소스의 표본 수). 합 = 1."""
    source_idx = np.asarray(source_idx)
    counts = np.array([(source_idx == i).sum() for i in range(n_sources)], dtype=np.float64)
    n_present = int((counts > 0).sum())
    w = np.zeros(len(source_idx), dtype=np.float64)
    for i in range(n_sources):
        if counts[i] > 0:
            w[source_idx == i] = 1.0 / (n_present * counts[i])
    return w


def class_balanced_weights(pix: np.ndarray) -> np.ndarray:
    """클래스 균등 weight. 화소가 있는 클래스마다 같은 몫을 주고, 레이어는 화소 비율대로 나눠 갖는다. 합 = 1.

    Args:
        pix: (n_samples, n_classes) — 클래스별 화소 수
    """
    pix = np.asarray(pix, dtype=np.float64)
    total = pix.sum(axis=0)
    present = total > 0
    if not present.any():
        return np.full(pix.shape[0], 1.0 / max(pix.shape[0], 1), dtype=np.float64)
    share = pix[:, present] / total[present]           # 열마다 합 = 1
    return share.sum(axis=1) / int(present.sum())


def cap_weights(p: np.ndarray, max_repeat: float) -> np.ndarray:
    """한 표본의 몫을 max_repeat / N 으로 제한한다 (균등 추출의 max_repeat 배까지). 합 = 1 유지."""
    q = np.asarray(p, dtype=np.float64)
    q = q / q.sum()
    lim = float(max_repeat) / len(q)
    if lim >= 1.0:
        return q
    capped = np.zeros(len(q), dtype=bool)
    for _ in range(len(q)):
        over = (q > lim + 1e-15) & ~capped
        if not over.any():
            break
        capped |= over
        q[capped] = lim
        free = ~capped
        rest = 1.0 - lim * int(capped.sum())
        if not free.any() or rest <= 0:
            break
        q[free] = q[free] / q[free].sum() * rest
    return q / q.sum()


def compute_sample_weights(
    pix: np.ndarray,
    source_idx: np.ndarray,
    n_sources: int,
    mix: float = cfg.SAMPLER_SOURCE_MIX,
    max_repeat: float = cfg.SAMPLER_MAX_REPEAT,
) -> np.ndarray:
    """v5 학습 sampler weight.

    Args:
        pix: (n_samples, n_classes) — 학습 표본의 클래스별 화소 수 (학습 해상도 기준)
        source_idx: (n_samples,) int — 표본의 소스 인덱스 (0..n_sources-1)
        n_sources: 소스 개수
        mix: 소스 균등의 비중 (0 = 클래스 균등만, 1 = 소스 균등만)
        max_repeat: 한 표본이 균등 추출 대비 몇 배까지 뽑힐 수 있는지

    Returns:
        (n_samples,) float64, 합 = 1
    """
    u = source_balanced_weights(source_idx, n_sources)
    c = class_balanced_weights(pix)
    return cap_weights(mix * u + (1.0 - mix) * c, max_repeat)


def effective_sample_size(p: np.ndarray) -> float:
    """1 / Σ p² — 추출이 사실상 몇 개의 표본에 퍼져 있는지."""
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum()
    return float(1.0 / np.square(p).sum())
