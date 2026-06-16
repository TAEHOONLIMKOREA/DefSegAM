"""v2 inference 휴리스틱 후처리.

학습된 모델의 argmax pred 를 입력으로 받아 도메인 규칙으로 정제한다.
Loss / 학습과는 무관 — inference 단계 (시각화, metric 계산) 직전에만 적용.

규칙은 두 가지:
  1. remove_static_outside_powder
        |visible/1 − visible/0| ≤ T_static  AND  powder ROI 바깥
        조건을 만족하는 픽셀이 결함 (class 2..7) 으로 분류돼 있으면 IGNORE 처리.
        → 빌드플레이트 / fixture / 챔버 frame 등 카메라 setup 의 정적 객체가
          결함으로 오분류되는 경우 제거.

  2. relabel_far_se_swelling_to_debris
        Super-Elevation(5) / Swelling(3) 의 connected component 가 Printed(1)
        mask 에서 충분히 멀리 (≥ T_far) 있고 면적도 일정 (≥ T_min) 이면
        Debris(7) 로 재분류.
        → 부품에서 떨어진 곳의 raised feature 는 사실상 떨어진 입자.

모든 임계치는 v2/config_v2.py 의 PP_* 상수에서 가져오며, 실 결함이 사라지지
않도록 보수적으로 설정돼 있다.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    binary_closing,
    binary_dilation,
    distance_transform_edt,
    label as cc_label,
)

from .. import config_v2 as cfg


# ---------------------------------------------------------------------------
# 규칙 1 — 정적 + powder ROI 바깥 → IGNORE
# ---------------------------------------------------------------------------

def _powder_roi(pred: np.ndarray) -> np.ndarray:
    """Powder ROI = pred ∈ {Powder, (옵션) Printed} 의 largest CC + closing + dilation.

    Returns:
        (H, W) bool — ROI 안 = True.
    """
    if cfg.PP_POWDER_ROI_USE_PRINTED:
        base = (pred == 0) | (pred == 1)
    else:
        base = (pred == 0)
    if not base.any():
        return np.zeros_like(pred, dtype=bool)

    # 작은 구멍 메우기
    if cfg.PP_POWDER_ROI_CLOSING_PX > 0:
        st_close = np.ones(
            (cfg.PP_POWDER_ROI_CLOSING_PX, cfg.PP_POWDER_ROI_CLOSING_PX),
            dtype=bool,
        )
        base = binary_closing(base, structure=st_close)

    # largest connected component 만 채택
    lab, n = cc_label(base, structure=np.ones((3, 3), dtype=np.int8))
    if n == 0:
        return np.zeros_like(pred, dtype=bool)
    sizes = np.bincount(lab.ravel())
    sizes[0] = 0  # background 제외
    largest = int(sizes.argmax())
    roi = (lab == largest)

    # 보수적 dilation — 부품 가장자리의 진짜 결함이 ROI 밖으로 새지 않게
    if cfg.PP_POWDER_ROI_DILATE_PX > 0:
        st_dil = np.ones(
            (cfg.PP_POWDER_ROI_DILATE_PX, cfg.PP_POWDER_ROI_DILATE_PX),
            dtype=bool,
        )
        roi = binary_dilation(roi, structure=st_dil)
    return roi


def remove_static_outside_powder(
    pred: np.ndarray,
    i0_u8: np.ndarray,
    i1_u8: np.ndarray,
    *,
    static_diff_threshold: int | None = None,
) -> tuple[np.ndarray, int]:
    """규칙 1.

    Args:
        pred: (H, W) int8 — argmax label (0..N_CLASSES_V2-1)
        i0_u8, i1_u8: (H, W) uint8 — visible/0, visible/1
        static_diff_threshold: 기본값 = cfg.PP_STATIC_DIFF_THRESHOLD

    Returns:
        (cleaned_pred, n_ignored_pixels)
    """
    T = cfg.PP_STATIC_DIFF_THRESHOLD if static_diff_threshold is None else static_diff_threshold

    static = np.abs(i0_u8.astype(np.int16) - i1_u8.astype(np.int16)) <= T
    outside = ~_powder_roi(pred)
    defect = pred >= 2  # 결함 클래스: 2..7

    target = static & outside & defect
    if not target.any():
        return pred, 0
    out = pred.copy()
    out[target] = cfg.IGNORE_INDEX
    return out, int(target.sum())


# ---------------------------------------------------------------------------
# 규칙 2 — SE/Swelling 이 part 에서 멀면 → Debris
# ---------------------------------------------------------------------------

def relabel_far_se_swelling_to_debris(
    pred: np.ndarray,
    *,
    far_distance_px: int | None = None,
    min_component_px: int | None = None,
) -> tuple[np.ndarray, int]:
    """규칙 2.

    Printed(1) mask 의 nearest pixel 까지 거리(px) 가 far_distance_px 이상인
    Super-Elevation/Swelling component 를 Debris 로 재분류.

    Args:
        pred: (H, W) int8
        far_distance_px: 기본값 = cfg.PP_SE_SWELLING_FAR_DISTANCE_PX
        min_component_px: 기본값 = cfg.PP_SE_SWELLING_MIN_COMPONENT_PX

    Returns:
        (relabeled_pred, n_relabeled_pixels)
    """
    far_T = cfg.PP_SE_SWELLING_FAR_DISTANCE_PX if far_distance_px is None else far_distance_px
    min_sz = cfg.PP_SE_SWELLING_MIN_COMPONENT_PX if min_component_px is None else min_component_px
    src_classes = tuple(cfg.PP_SE_SWELLING_SOURCE_CLASSES)
    tgt_class = int(cfg.PP_SE_SWELLING_TARGET_CLASS)

    printed = (pred == 1)
    if printed.all() or not printed.any():
        return pred, 0  # part 정의 불가 — 안전하게 변경 안 함

    # Printed 까지의 distance map (px). EDT 의 anisotropy 는 1.0 픽셀 가정.
    dist_to_part = distance_transform_edt(~printed)

    out = pred.copy()
    n_relabeled = 0
    for src in src_classes:
        src_mask = (pred == src)
        if not src_mask.any():
            continue
        lab, n = cc_label(src_mask, structure=np.ones((3, 3), dtype=np.int8))
        for k in range(1, n + 1):
            comp = (lab == k)
            size = int(comp.sum())
            if size < min_sz:
                continue
            # component 내부의 최소 거리: 부품에 가장 가까운 점 기준 — 보수적
            min_d = float(dist_to_part[comp].min())
            if min_d >= far_T:
                out[comp] = tgt_class
                n_relabeled += size
    return out, n_relabeled


# ---------------------------------------------------------------------------
# Combined entry — 인자 한 번에 적용
# ---------------------------------------------------------------------------

def apply_postprocess(
    pred: np.ndarray,
    i0_u8: np.ndarray,
    i1_u8: np.ndarray,
    *,
    do_static_outside: bool = True,
    do_far_se_swelling: bool = True,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """두 규칙 순차 적용.

    순서:
      1) remove_static_outside_powder  (powder ROI 는 원본 pred 로 계산)
      2) relabel_far_se_swelling_to_debris

    Returns:
        (postprocessed_pred, stats dict {static_ignored, se_swelling_relabeled})
    """
    stats = {"static_ignored": 0, "se_swelling_relabeled": 0}
    out = pred
    if do_static_outside:
        out, n1 = remove_static_outside_powder(out, i0_u8, i1_u8)
        stats["static_ignored"] = n1
    if do_far_se_swelling:
        out, n2 = relabel_far_se_swelling_to_debris(out)
        stats["se_swelling_relabeled"] = n2
    if verbose and (stats["static_ignored"] or stats["se_swelling_relabeled"]):
        print(
            f"  [pp] static_ignored={stats['static_ignored']:>7d}  "
            f"se_swelling_relabeled={stats['se_swelling_relabeled']:>7d}"
        )
    return out, stats
