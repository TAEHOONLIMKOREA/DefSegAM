"""시제품 — 두 사진의 관계로 정의되는 결함을 규칙으로 만들어 넣는다 (학습 코드와 무관. docs/SYNTHETIC_DATA_NOTES.md §4).

Super-Elevation / Swelling 의 실제 모습 (2026-11-17, GP · L21 · I2 예시를 확대해 본 것):
    도포 직후 사진 (visible/1) 에서, 부품 자리 (visible/0 에서 Printed) 의 금속이 분말 위로 드러나 밝게 보인다.
    SE 는 넓게 (촘촘한 얼룩 – 완전 노출), Swelling 은 부품 가장자리의 작은 점 · 짧은 호.

만드는 법:
    부품 자리 P = (정답 == Printed) 안에서 노출 확률 장 cov(x) 를 정하고, 분말 알갱이 크기의 난수로 얼룩 α 를 만든 뒤
    visible/1 ← visible/1 · (1 − α) + (visible/0 의 그 자리) · α.    visible/0 은 건드리지 않는다.
    정답: 얼룩이 촘촘한 곳 = 결함 클래스, 듬성듬성한 띠 = IGNORE (−1, 어느 쪽으로 칠해도 틀렸다고 하지 않는다).

입력 · 출력은 v5 GTPool 과 같은 형식: (S, S) uint8 두 장 + (S, S) int8 정답.

실행 (CWD = 3DP_VPPM/, DefSeg_AM/venv):
    (라이브러리 — eval_on_gt · preflight_ornl 이 불러 쓴다)
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, gaussian_filter, label as cc_label

POWDER, PRINTED, SWELL, SE = 0, 1, 5, 7
IGNORE = -1


def _smooth_noise(shape, sigma: float, rng: np.random.Generator) -> np.ndarray:
    x = gaussian_filter(rng.standard_normal(shape).astype(np.float32), sigma)
    return (x - x.mean()) / (x.std() + 1e-6)


def _speckle(cov: np.ndarray, grain: float, rng: np.random.Generator) -> np.ndarray:
    """노출 확률 cov ∈ [0, 1] → 알갱이 얼룩 α ∈ [0, 1]. grain = 알갱이 크기 (px, 0 이면 화소 단위)."""
    u = rng.random(cov.shape).astype(np.float32)
    if grain > 0:
        u = gaussian_filter(u, grain)
        order = u.ravel().argsort()
        rank = np.empty(order.size, dtype=np.float32)
        rank[order] = (np.arange(order.size, dtype=np.float32) + 0.5) / order.size
        u = rank.reshape(cov.shape)                      # 다시 균등 분포로
    a = (u < cov).astype(np.float32)
    return np.clip(gaussian_filter(a, 0.5) * 1.15, 0.0, 1.0)


def _drop_small(mask: np.ndarray, min_px: int) -> np.ndarray:
    lab, n = cc_label(mask, structure=np.ones((3, 3), dtype=np.int8))
    if n == 0:
        return mask
    sizes = np.bincount(lab.ravel())
    keep = sizes >= min_px
    keep[0] = False
    return keep[lab]


def _blend(i0: np.ndarray, i1: np.ndarray, alpha: np.ndarray, gain: float) -> np.ndarray:
    src = np.clip(i0.astype(np.float32) * gain, 0, 255)
    return np.clip(i1.astype(np.float32) * (1.0 - alpha) + src * alpha, 0, 255).astype(np.uint8)


def sim_super_elevation(i0, i1, ann, rng: np.random.Generator, zone: str | None = None):
    """부품이 도포 뒤에도 드러난 구역을 만든다. 반환: (i1_new, ann_new, new_mask) 또는 None (부품이 없다)."""
    P = ann == PRINTED
    if int(P.sum()) < 1500:
        return None
    H, W = ann.shape
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    zone = zone or str(rng.choice(["blob", "blob", "band"]))
    if zone == "blob":                                   # 부품 위의 한 점을 중심으로 한 불규칙한 타원
        ys, xs = np.nonzero(P)
        k = int(rng.integers(len(ys)))
        cy, cx = float(ys[k]), float(xs[k])
        r = float(rng.uniform(0.05, 0.22)) * H
        ang, ax = float(rng.uniform(0, np.pi)), float(rng.uniform(0.6, 1.7))
        dy, dx = yy - cy, xx - cx
        u = dx * np.cos(ang) + dy * np.sin(ang)
        v = -dx * np.sin(ang) + dy * np.cos(ang)
        d = np.sqrt((u / ax) ** 2 + (v * ax) ** 2) / r + 0.25 * _smooth_noise((H, W), 0.03 * H, rng)
        cov = float(rng.uniform(0.45, 1.0)) * np.clip(1.25 - d, 0, 1) ** float(rng.uniform(0.6, 2.2))
        grain = float(rng.uniform(0.0, 0.9))
    else:                                                # 판 가장자리의 띠 — 분말이 닿지 않은 구역의 부품 (L21 의 모습)
        f = float(rng.uniform(0.05, 0.3))
        x = np.arange(W, dtype=np.float32)
        wave = sum(float(rng.uniform(0.004, 0.02)) * H * np.sin(2 * np.pi * x / (W / float(rng.uniform(1, 6))) + float(rng.uniform(0, 6.28)))
                   for _ in range(3))
        if rng.random() < 0.5:
            d = (yy - (H * (1 - f) + wave[None, :])) / (0.01 * H)      # 아래쪽 띠
        else:
            d = ((H * f + wave[None, :]) - yy) / (0.01 * H)            # 위쪽 띠
        cov = np.clip(d, 0, 1) * float(rng.uniform(0.85, 1.0))
        grain = float(rng.uniform(0.0, 0.4))
    cov = np.clip(cov, 0, 1) * P
    alpha = _speckle(cov, grain, rng) * P
    if float(alpha.sum()) < 150:
        return None
    i1_new = _blend(i0, i1, alpha, gain=float(rng.uniform(0.9, 1.08)))
    dens = gaussian_filter(alpha, 1.5)
    new = _drop_small((dens > 0.45) & P, 12)
    ring = (dens > 0.08) & P & ~new
    ann_new = ann.copy()
    ann_new[ring] = IGNORE
    ann_new[new] = SE
    return i1_new, ann_new, new


def sim_swelling(i0, i1, ann, rng: np.random.Generator):
    """부품 가장자리가 도포 뒤에 점 · 짧은 호로 드러난 것을 만든다."""
    P = ann == PRINTED
    if int(P.sum()) < 1500:
        return None
    H, W = ann.shape
    rim = P & ~binary_erosion(P, iterations=int(rng.integers(1, 4)))
    ys, xs = np.nonzero(rim)
    if len(ys) < 50:
        return None
    yy, xx = np.mgrid[0:H, 0:W].astype(np.float32)
    cov = np.zeros((H, W), dtype=np.float32)
    for _ in range(int(rng.integers(2, 9))):
        k = int(rng.integers(len(ys)))
        r = float(rng.uniform(0.008, 0.05)) * H
        d = np.sqrt((yy - ys[k]) ** 2 + (xx - xs[k]) ** 2) / r
        cov = np.maximum(cov, float(rng.uniform(0.5, 1.0)) * np.clip(1.1 - d, 0, 1))
    alpha = _speckle(cov * rim, float(rng.uniform(0.0, 0.5)), rng) * binary_dilation(rim, iterations=1)
    if float(alpha.sum()) < 20:
        return None
    i1_new = _blend(i0, i1, alpha, gain=float(rng.uniform(0.95, 1.1)))
    dens = gaussian_filter(alpha, 0.8)
    new = _drop_small(dens > 0.4, 3)
    ring = (dens > 0.1) & ~new
    ann_new = ann.copy()
    ann_new[ring & (ann == PRINTED)] = IGNORE
    ann_new[new & ((ann == PRINTED) | (ann == POWDER))] = SWELL
    new = ann_new == SWELL
    new &= ~(ann == SWELL)
    return i1_new, ann_new, new
