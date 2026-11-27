"""Peregrine 주석 (native id) → `pbf-2026.11` 라벨 (PLAN_v6 §2.3).

v4 · v5 의 2단계 매핑 (native → ORNL 12 → 버전 class) 과 달리 **native 이름 → 새 id** 의 1단계다.
표는 `config_v6.py` 에 있고, 여기에는 표를 라벨 맵에 적용하는 함수만 있다.

  - 이름이 새 체계에 있는 것은 그 id 로 (`LEGACY_NAME_TO_V6`)
  - Swelling / Edge Swelling 은 D3 에 따라 Super Elevation 또는 미레이블
  - 기존 IS (Incomplete Spreading / Powder Short Feed / Short Feed) 는 D4 에 따라 레이어 단위로 IS / PS / 미레이블
  - Soot 는 Spatter. 다만 CAD 파트 마스크가 있는 레이어에서 마스크 안쪽은 Over Melting (D10)
  - 나머지 이름과 주석의 -1 은 미레이블
  - 원본이 기존 IS 를 경계선으로만 칠한 레이어는 (판정표의 fill) 선이 가두는 쪽의 Powder 를 같은 클래스로 채운다
"""
from __future__ import annotations

import numpy as np
from scipy import ndimage as ndi

from .. import config_v6 as cfg


def target_of(name: str, swelling: str = cfg.LEGACY_SWELLING_DEFAULT, is_decision: str = "ignore") -> int:
    """native 이름 하나가 가는 새 id (-1 = 미레이블). Soot 의 파트 마스크 분할은 여기서 하지 않는다."""
    if name in cfg.LEGACY_NAME_TO_V6:
        return cfg.LEGACY_NAME_TO_V6[name]
    if name in cfg.LEGACY_SWELLING_NAMES:
        return cfg.LEGACY_SWELLING_CHOICES[swelling]
    if name in cfg.LEGACY_IS_NAMES:
        return cfg.LEGACY_IS_CHOICES[is_decision]
    if name in cfg.LEGACY_NAME_IGNORED:
        return cfg.IGNORE_INDEX
    raise KeyError(f"매핑 표에 없는 native 이름: {name!r}")


def enclosed_side(line: np.ndarray) -> np.ndarray:
    """경계선 mask (H, W) → 선이 이미지 가장자리와 함께 가두는 쪽의 mask (선 자신은 넣지 않는다).

    원본 주석이 덮이지 않은 (무너진) 영역 대신 그 경계선만 칠한 레이어에 쓴다.
      1. 선의 끝이 가장자리에 조금 못 미치는 틈을 막는다 — 가장자리에서 d px 안쪽 띠 가운데 선에서 d px 안에 있는 화소
      2. 선과 막은 틈을 벽으로 삼아 나머지 화소를 4-이웃 덩어리로 나눈다
      3. 가장 큰 덩어리 (분말이 덮인 쪽) 를 뺀 나머지가 가둔 쪽이다
    선이 아무것도 가두지 못하면 빈 mask 를 돌려준다.
    """
    line = np.asarray(line, dtype=bool)
    h, w = line.shape
    d = max(3, int(round(cfg.LEGACY_FILL_SEAL_FRAC * min(h, w))))
    dist = ndi.distance_transform_edt(~line)
    strip = np.zeros_like(line)
    strip[:d] = True
    strip[-d:] = True
    strip[:, :d] = True
    strip[:, -d:] = True
    seal = strip & (dist <= d) & ~line
    wall = line | seal
    cc, n = ndi.label(~wall)
    if n < 2:
        return np.zeros_like(line)
    areas = ndi.sum(~wall, cc, index=np.arange(1, n + 1))
    region = (~wall) & (cc != 1 + int(np.argmax(areas)))
    region |= seal & ndi.binary_dilation(region, iterations=1)      # 막은 틈 가운데 영역에 붙은 화소
    return region


def remap_legacy(
    ann_native: np.ndarray,
    source_name: str,
    part_mask: np.ndarray | None = None,
    swelling: str = cfg.LEGACY_SWELLING_DEFAULT,
    is_decision: str = "ignore",
    is_fill: bool = False,
) -> tuple[np.ndarray, dict]:
    """native 주석 (H, W) → 새 체계 라벨 (H, W) int16, 값은 0–9 와 -1.

    part_mask: 그 레이어의 CAD 파트 마스크 (H, W). 없거나 전부 0 이거나 크기가 다르면 쓰지 않는다
               (v2022 판은 형상 정보가 없는 레이어의 마스크를 전부 0 으로 준다).
    swelling:  "se" | "ignore" (D3)
    is_decision: 이 레이어의 기존 IS 를 어디로 보낼지 — "ignore" | "is" | "ps" (D4)
    is_fill:   기존 IS 가 경계선으로만 칠해진 레이어 — 선이 가두는 쪽의 Powder 화소를 is_decision 의 클래스로 채운다.
               그 안의 다른 클래스 (Printed · SE · SP …) 와 미레이블 화소는 그대로 둔다

    Returns:
        label, stats — stats 는 {"native_px": {이름: 화소 수}, "native_unlabeled_px": int, "soot_to_om_px": int,
                                 "part_mask": "none" | "empty" | "shape_mismatch" | "used",
                                 "is_fill_px": 채운 Powder 화소 수, "is_fill_region_px": 선이 가둔 영역의 화소 수}
    """
    names = cfg.LEGACY_NATIVE_NAMES[source_name]
    max_id = max(names)
    ann = np.asarray(ann_native)
    if ann.ndim != 2:
        raise ValueError(f"expected (H, W) annotation, got shape={ann.shape}")
    lo, hi = int(ann.min()), int(ann.max())
    if lo < cfg.NATIVE_UNLABELED or hi > max_id:
        raise ValueError(f"{source_name}: 표에 없는 native 값이 있다 (min={lo}, max={hi}, 표는 -1..{max_id})")

    # 표 조회: index = native id + 1 (index 0 은 주석의 미레이블 -1)
    lut = np.full(max_id + 2, cfg.IGNORE_INDEX, dtype=np.int16)
    for nid, name in names.items():
        lut[nid + 1] = target_of(name, swelling, is_decision)
    shifted = ann.astype(np.int64) + 1
    label = lut[shifted]

    counts = np.bincount(shifted.ravel(), minlength=max_id + 2)
    stats = {
        "native_px": {names[nid]: int(counts[nid + 1]) for nid in sorted(names) if counts[nid + 1]},
        "native_unlabeled_px": int(counts[0]),
        "soot_to_om_px": 0,
        "part_mask": "none",
        "is_fill_px": 0,
        "is_fill_region_px": 0,
    }

    # D10 — 파트 위의 Soot 는 Over Melting. CAD 마스크가 있는 레이어에서만 가를 수 있다.
    if part_mask is not None:
        part = np.asarray(part_mask)
        if part.ndim == 3:
            part = part.max(axis=2)
        if part.shape != ann.shape:
            stats["part_mask"] = "shape_mismatch"
        elif not (part > 0).any():
            stats["part_mask"] = "empty"
        else:
            stats["part_mask"] = "used"
            soot_ids = [nid for nid, name in names.items() if name == cfg.LEGACY_SOOT_NAME]
            if soot_ids:
                on_part = np.isin(ann, soot_ids) & (part > 0)
                label[on_part] = cfg.LEGACY_SOOT_IN_PART
                stats["soot_to_om_px"] = int(on_part.sum())

    # D4 — 경계선만 칠해진 기존 IS: 선이 가두는 쪽의 Powder 를 같은 클래스로 채운다
    if is_fill:
        target = cfg.LEGACY_IS_CHOICES[is_decision]
        if target == cfg.IGNORE_INDEX:
            raise ValueError(f"{source_name}: fill 은 IS / PS 판정과 함께만 쓸 수 있다")
        line = np.isin(ann, [nid for nid, name in names.items() if name in cfg.LEGACY_IS_NAMES])
        region = enclosed_side(line)
        frac = float(region.mean())
        if not region.any() or frac > cfg.LEGACY_FILL_MAX_FRAC:
            raise ValueError(f"{source_name}: 경계선이 가두는 영역을 정할 수 없다 (이미지의 {frac * 100:.1f}%) — 판정표의 fill 을 확인한다")
        paint = region & (label == 0)
        label[paint] = target
        stats["is_fill_px"] = int(paint.sum())
        stats["is_fill_region_px"] = int(region.sum())
    return label, stats


def count_classes(label: np.ndarray) -> tuple[np.ndarray, int]:
    """(H, W) 새 체계 라벨 → ((10,) 클래스별 화소 수, 미레이블 화소 수)."""
    lab = np.asarray(label).astype(np.int64)
    if lab.min() < cfg.IGNORE_INDEX or lab.max() >= cfg.N_CLASSES_V6:
        raise ValueError(f"라벨 값이 0–{cfg.N_CLASSES_V6 - 1} · -1 밖이다 (min={lab.min()}, max={lab.max()})")
    counts = np.bincount((lab + 1).ravel(), minlength=cfg.N_CLASSES_V6 + 1)
    return counts[1:].astype(np.int64), int(counts[0])
