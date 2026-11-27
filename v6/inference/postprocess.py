"""v6 inference 휴리스틱 후처리 — v4 · v5 의 두 규칙을 새 id 로 옮긴 것 (PLAN_v6 §8).

규칙 1: SE (6) 성분이 Printed (1) 에서 멀리 있으면 Debris (9) 로 — 제자리의 파트 / 떨어져 나온 조각
규칙 2: Spatter (7) 성분의 50% 이상이 Printed 위에 있으면 Over Melting (8) 으로 — 분말 위 / 파트 위

주 지표는 후처리 없이 낸다. 후처리는 켰을 때의 변화를 보는 보조 지표이고, 검증에서 결함 mIoU 가 오르는 규칙만 켠다.

임계값 (config_v6 의 PP_*) 의 단위는 **이 함수에 넘긴 예측의 화소**다. 계획은 1036² 기준 값으로 적었으므로
그 기준으로 보려면 1036² 예측에 건다 (`evaluate --resolution resized --postprocess`). 원본 해상도 예측에 걸면
소스마다 실제 거리가 달라진다 (v4 · v5 도 그렇게 동작했다).
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    label as cc_label,
)

from .. import config_v6 as cfg

PRINTED = 1


# ---------------------------------------------------------------------------
# 규칙 1 — 부품에서 먼 SE → Debris
# ---------------------------------------------------------------------------

def relabel_far_se_to_debris(
    pred: np.ndarray,
    *,
    far_distance_px: int | None = None,
    min_component_px: int | None = None,
) -> tuple[np.ndarray, int]:
    """SE (6) 의 connected component 가 Printed (1) 에서 멀리 있으면 Debris (9) 로.

    component 내 **최소** 거리 사용 → 부품에 조금이라도 닿으면 변경 안 함 (보수적).
    """
    far_T = cfg.PP_SE_FAR_DISTANCE_PX if far_distance_px is None else far_distance_px
    min_sz = cfg.PP_SE_MIN_COMPONENT_PX if min_component_px is None else min_component_px
    src_classes = tuple(cfg.PP_SE_SOURCE_CLASSES)
    tgt_class = int(cfg.PP_SE_TARGET_CLASS)

    printed = (pred == PRINTED)
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
# 규칙 2 — Printed 영역 위의 Spatter → Over Melting
# ---------------------------------------------------------------------------

def relabel_printed_spatter_to_overmelt(
    pred: np.ndarray,
    *,
    overlap_frac: float | None = None,
    min_component_px: int | None = None,
    printed_dilate_px: int | None = None,
) -> tuple[np.ndarray, int]:
    """Spatter (7) component 의 일정 비율 이상이 Printed (1) (+ dilation) 위에 있으면 Over Melting (8) 으로.

    pred 는 mutual exclusive 라 Spatter 픽셀 ∩ Printed 픽셀 = 0. 따라서
    Printed mask 를 dilation 시켜 "Printed 영역 또는 그 즉시 주변" 을
    정의 후 component 와의 overlap 비율 측정.

    component 단위 판정 → 산발 픽셀 노이즈가 아닌 spatial cluster 만.
    """
    overlap_T = cfg.PP_SP_OVERMELT_OVERLAP_FRAC if overlap_frac is None else overlap_frac
    min_sz = cfg.PP_SP_MIN_COMPONENT_PX if min_component_px is None else min_component_px
    dilate_px = cfg.PP_SP_PRINTED_DILATE_PX if printed_dilate_px is None else printed_dilate_px
    src_class = int(cfg.PP_SP_SOURCE_CLASS)
    tgt_class = int(cfg.PP_SP_TARGET_CLASS)

    spatter = (pred == src_class)
    printed = (pred == PRINTED)
    if not spatter.any() or not printed.any():
        return pred, 0

    # Printed mask dilation — "Printed 또는 그 즉시 주변" 정의
    if dilate_px > 0:
        st = np.ones((dilate_px, dilate_px), dtype=bool)
        printed_neighborhood = binary_dilation(printed, structure=st)
    else:
        printed_neighborhood = printed

    out = pred.copy()
    lab, n = cc_label(spatter, structure=np.ones((3, 3), dtype=np.int8))
    n_relabeled = 0
    for k in range(1, n + 1):
        comp = (lab == k)
        size = int(comp.sum())
        if size < min_sz:
            continue
        overlap_ct = int((comp & printed_neighborhood).sum())
        if overlap_ct / size >= overlap_T:
            out[comp] = tgt_class
            n_relabeled += size
    return out, n_relabeled


# ---------------------------------------------------------------------------
# Entry — apply_postprocess
# ---------------------------------------------------------------------------

def apply_postprocess(
    pred: np.ndarray,
    *,
    do_rule1: bool = True,
    do_rule2: bool = True,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """두 규칙 순차 적용.

    Returns:
        (postprocessed_pred, stats {se_relabeled, sp_overmelt_relabeled})
    """
    out = pred
    stats = {"se_relabeled": 0, "sp_overmelt_relabeled": 0}
    if do_rule1:
        out, n1 = relabel_far_se_to_debris(out)
        stats["se_relabeled"] = n1
    if do_rule2:
        out, n2 = relabel_printed_spatter_to_overmelt(out)
        stats["sp_overmelt_relabeled"] = n2
    if verbose and any(stats.values()):
        print(
            f"  [pp] se_relabeled={stats['se_relabeled']:>7d}  "
            f"sp_overmelt_relabeled={stats['sp_overmelt_relabeled']:>7d}"
        )
    return out, stats
