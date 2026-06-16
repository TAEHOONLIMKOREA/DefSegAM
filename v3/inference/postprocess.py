"""v3 inference 휴리스틱 후처리.

학습된 모델의 argmax pred 를 도메인 규칙으로 정제. Loss/학습과 무관 —
inference 단계 (시각화, metric 계산) 직전에만 적용.

규칙 (PLAN_v3 §5.1):

  relabel_far_se_swelling_to_debris
        Super-Elevation(7) / Swelling(5) 의 connected component 가 Printed(1)
        mask 에서 충분히 멀리 있고 면적도 일정 이상이면 Debris(6) 로 재분류.
        → 부품에서 떨어진 곳의 raised feature 는 사실상 떨어진 입자.

모든 임계치는 v3/config_v3.py 의 PP_* 상수에서 가져오며, 실 결함이 사라지지
않도록 보수적.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    distance_transform_edt,
    label as cc_label,
)

from .. import config_v3 as cfg


# ---------------------------------------------------------------------------
# 규칙 — SE/Swelling 이 part 에서 멀면 → Debris
# ---------------------------------------------------------------------------

def relabel_far_se_swelling_to_debris(
    pred: np.ndarray,
    *,
    far_distance_px: int | None = None,
    min_component_px: int | None = None,
) -> tuple[np.ndarray, int]:
    """Printed(1) mask 의 nearest pixel 까지 거리(px) 가 far_distance_px 이상인
    SE/Swelling component (면적 ≥ min_component_px) 를 Debris 로 재분류.

    component 내 **최소** 거리 사용 → 부품에 조금이라도 닿으면 변경 안 함 (보수적).

    Returns:
        (relabeled_pred, n_relabeled_pixels)
    """
    far_T = cfg.PP_SE_SWELLING_FAR_DISTANCE_PX if far_distance_px is None else far_distance_px
    min_sz = cfg.PP_SE_SWELLING_MIN_COMPONENT_PX if min_component_px is None else min_component_px
    src_classes = tuple(cfg.PP_SE_SWELLING_SOURCE_CLASSES)
    tgt_class = int(cfg.PP_SE_SWELLING_TARGET_CLASS)

    printed = (pred == 1)
    if printed.all() or not printed.any():
        return pred, 0

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
            min_d = float(dist_to_part[comp].min())
            if min_d >= far_T:
                out[comp] = tgt_class
                n_relabeled += size
    return out, n_relabeled


# ---------------------------------------------------------------------------
# Entry — apply_postprocess (단일 규칙이지만 v2 와 동일한 호출 인터페이스 유지)
# ---------------------------------------------------------------------------

def apply_postprocess(
    pred: np.ndarray,
    i0_u8: np.ndarray,
    i1_u8: np.ndarray,
    *,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """후처리 적용.

    i0_u8 / i1_u8 인자는 v2 호환을 위해 받지만 현재 규칙에선 사용되지 않음
    (향후 multi-rule 확장 여지).

    Returns:
        (postprocessed_pred, {se_swelling_relabeled})
    """
    out, n = relabel_far_se_swelling_to_debris(pred)
    stats = {"se_swelling_relabeled": n}
    if verbose and n > 0:
        print(f"  [pp] se_swelling_relabeled={n:>7d}")
    return out, stats
