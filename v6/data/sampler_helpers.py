"""v6 sampler weight — v5 의 식을 도메인별로 적용하고 도메인 사이의 비중을 따로 정한다 (PLAN_v6 §5.4).

도메인 d (Peregrine 주석 / DIVA 내보내기) 안에서 — v5 와 같은 식:

  u_i = 1 / (그룹 수 × 그 그룹의 표본 수)                      # 그룹 균등, Σ u = 1.  그룹 = Peregrine 소스 / DIVA 공정
  c_i = (1 / 클래스 수) × Σ_c pix[i, c] / total_pix_d[c]        # 클래스 균등, Σ c = 1 (화소가 0 인 클래스는 뺀다)
  q_i = mix × u_i + (1 − mix) × c_i
  q_i ← min(q_i, max_repeat / N_d) 후 남은 몫을 나머지 표본에 비례 배분

도메인 사이:

  p_i = share[d] × q_i,   share[diva] = min(DOMAIN_SHARE_DIVA_MAX, N_diva / N),  share[legacy] = 1 − share[diva]

도메인이 하나뿐이면 (P0 — Peregrine 주석만) v5 의 식 그대로다.

곱 (u_i × c_i) 이 아니라 합 (혼합) 인 까닭은 v5 의 같은 파일에 적혀 있다: 곱은 Peregrine 주석 125 레이어 가운데 5개에
추출의 39% 를 몰아주고 (유효 표본 수 25), 화소가 0 인 클래스가 있으면 전체가 균등 추출로 무너진다.
"""
from __future__ import annotations

import numpy as np

from .. import config_v6 as cfg


def group_balanced_weights(group_idx: np.ndarray, n_groups: int) -> np.ndarray:
    """그룹 균등 weight. 한 표본의 weight = 1 / (표본이 있는 그룹 수 × 그 그룹의 표본 수). 합 = 1."""
    group_idx = np.asarray(group_idx)
    counts = np.array([(group_idx == i).sum() for i in range(n_groups)], dtype=np.float64)
    n_present = int((counts > 0).sum())
    w = np.zeros(len(group_idx), dtype=np.float64)
    for i in range(n_groups):
        if counts[i] > 0:
            w[group_idx == i] = 1.0 / (n_present * counts[i])
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
    group_idx: np.ndarray,
    n_groups: int,
    mix: float = cfg.SAMPLER_SOURCE_MIX,
    max_repeat: float = cfg.SAMPLER_MAX_REPEAT,
) -> np.ndarray:
    """한 도메인 안의 sampler weight (v5 의 식).

    Args:
        pix: (n_samples, n_classes) — 학습 표본의 클래스별 화소 수 (학습 해상도 기준)
        group_idx: (n_samples,) int — 표본의 그룹 인덱스 (0..n_groups-1)
        n_groups: 그룹 개수
        mix: 그룹 균등의 비중 (0 = 클래스 균등만, 1 = 그룹 균등만)
        max_repeat: 한 표본이 균등 추출 대비 몇 배까지 뽑힐 수 있는지

    Returns:
        (n_samples,) float64, 합 = 1
    """
    u = group_balanced_weights(group_idx, n_groups)
    c = class_balanced_weights(pix)
    return cap_weights(mix * u + (1.0 - mix) * c, max_repeat)


def domain_shares(domains: list[str], diva_max: float = cfg.DOMAIN_SHARE_DIVA_MAX) -> dict[str, float]:
    """도메인별 추출 비중. DIVA 쪽은 표본 수 비율을 따르되 diva_max 를 넘지 않는다. 합 = 1."""
    n = len(domains)
    present = sorted(set(domains))
    unknown = set(present) - {cfg.DOMAIN_LEGACY, cfg.DOMAIN_DIVA}
    if unknown:
        raise ValueError(f"unknown domain(s): {sorted(unknown)}")
    if present == [cfg.DOMAIN_LEGACY] or present == [cfg.DOMAIN_DIVA]:
        return {present[0]: 1.0}
    n_diva = sum(1 for d in domains if d == cfg.DOMAIN_DIVA)
    share_diva = min(float(diva_max), n_diva / n)
    return {cfg.DOMAIN_DIVA: share_diva, cfg.DOMAIN_LEGACY: 1.0 - share_diva}


def compute_domain_sample_weights(
    pix: np.ndarray,
    groups: list[str],
    domains: list[str],
    mix: float = cfg.SAMPLER_SOURCE_MIX,
    max_repeat: float = cfg.SAMPLER_MAX_REPEAT,
    diva_max: float = cfg.DOMAIN_SHARE_DIVA_MAX,
) -> tuple[np.ndarray, dict[str, float]]:
    """v6 학습 sampler weight — 도메인마다 v5 의 식을 따로 적용하고 도메인 비중을 곱한다.

    Args:
        pix: (n_samples, n_classes) — 학습 표본의 클래스별 화소 수 (학습 해상도 기준)
        groups: 표본의 그룹 이름 (Peregrine 소스 / DIVA 공정)
        domains: 표본의 도메인 이름

    Returns:
        ((n_samples,) float64 합 = 1, {도메인: 비중})
    """
    pix = np.asarray(pix)
    if not (len(groups) == len(domains) == pix.shape[0]):
        raise ValueError("pix · groups · domains 의 길이가 다르다")
    shares = domain_shares(domains, diva_max)
    p = np.zeros(pix.shape[0], dtype=np.float64)
    dom = np.array(domains)
    for d, share in shares.items():
        m = np.where(dom == d)[0]
        names = sorted({groups[i] for i in m})
        lut = {n: k for k, n in enumerate(names)}
        g_idx = np.array([lut[groups[i]] for i in m], dtype=np.int64)
        p[m] = share * compute_sample_weights(pix[m], g_idx, len(names), mix=mix, max_repeat=max_repeat)
    return p / p.sum(), shares


def effective_sample_size(p: np.ndarray) -> float:
    """1 / Σ p² — 추출이 사실상 몇 개의 표본에 퍼져 있는지."""
    p = np.asarray(p, dtype=np.float64)
    p = p / p.sum()
    return float(1.0 / np.square(p).sum())
