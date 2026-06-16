"""v4 inference 휴리스틱 후처리 — PLAN_v4 §8.

규칙 1: SE/Swelling 의 component 가 Printed 에서 멀리 있으면 Debris 로 재분류
규칙 2: Spatter component 의 50% 이상이 Printed 위에 있으면 Over Melting 으로 재분류

모든 임계치는 v4/config_v4.py 의 PP_* 상수에서 가져오며, 실 결함이 사라지지
않도록 보수적.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import (
    binary_dilation,
    distance_transform_edt,
    label as cc_label,
)

from .. import config_v4 as cfg


# ---------------------------------------------------------------------------
# 규칙 1 — 부품에서 먼 SE/Swelling → Debris
# ---------------------------------------------------------------------------

def relabel_far_se_to_debris(
    pred: np.ndarray,
    *,
    far_distance_px: int | None = None,
    min_component_px: int | None = None,
) -> tuple[np.ndarray, int]:
    """SE(7)/Swelling(5) 의 connected component 가 Printed(1) 에서 멀리 있으면 Debris(6) 로.

    component 내 **최소** 거리 사용 → 부품에 조금이라도 닿으면 변경 안 함 (보수적).
    """
    far_T = cfg.PP_SE_FAR_DISTANCE_PX if far_distance_px is None else far_distance_px
    min_sz = cfg.PP_SE_MIN_COMPONENT_PX if min_component_px is None else min_component_px
    src_classes = tuple(cfg.PP_SE_SOURCE_CLASSES)
    tgt_class = int(cfg.PP_SE_TARGET_CLASS)

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
# 규칙 2 — Printed 영역 위의 Spatter → Over Melting
# ---------------------------------------------------------------------------

def relabel_printed_spatter_to_overmelt(
    pred: np.ndarray,
    *,
    overlap_frac: float | None = None,
    min_component_px: int | None = None,
    printed_dilate_px: int | None = None,
) -> tuple[np.ndarray, int]:
    """Spatter(8) component 의 일정 비율 이상이 Printed(1) (+ dilation) 위에 있으면 Over Melting(9) 으로.

    pred 는 mutual exclusive 라 Spatter 픽셀 ∩ Printed 픽셀 = 0. 따라서
    Printed mask 를 dilation 시켜 "Printed 영역 또는 그 즉시 주변" 을
    정의 후 component 와의 overlap 비율 측정.

    component 단위 판정 → 산발 픽셀 노이즈가 아닌 spatial cluster 만.
    """
    overlap_T = cfg.PP_PS_OVERMELT_OVERLAP_FRAC if overlap_frac is None else overlap_frac
    min_sz = cfg.PP_PS_MIN_COMPONENT_PX if min_component_px is None else min_component_px
    dilate_px = cfg.PP_PS_PRINTED_DILATE_PX if printed_dilate_px is None else printed_dilate_px
    src_class = int(cfg.PP_PS_SOURCE_CLASS)
    tgt_class = int(cfg.PP_PS_TARGET_CLASS)

    spatter = (pred == src_class)
    printed = (pred == 1)
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
    i0_u8: np.ndarray,
    i1_u8: np.ndarray,
    *,
    do_rule1: bool = True,
    do_rule2: bool = True,
    verbose: bool = False,
) -> tuple[np.ndarray, dict]:
    """v4 두 규칙 순차 적용.

    i0_u8 / i1_u8 인자는 호출 인터페이스 유지를 위한 것 (현재 규칙들에서는
    사용 안 함).

    Returns:
        (postprocessed_pred, stats {se_relabeled, ps_overmelt_relabeled})
    """
    out = pred
    stats = {"se_relabeled": 0, "ps_overmelt_relabeled": 0}
    if do_rule1:
        out, n1 = relabel_far_se_to_debris(out)
        stats["se_relabeled"] = n1
    if do_rule2:
        out, n2 = relabel_printed_spatter_to_overmelt(out)
        stats["ps_overmelt_relabeled"] = n2
    if verbose and any(stats.values()):
        print(
            f"  [pp] se_relabeled={stats['se_relabeled']:>7d}  "
            f"ps_overmelt_relabeled={stats['ps_overmelt_relabeled']:>7d}"
        )
    return out, stats
