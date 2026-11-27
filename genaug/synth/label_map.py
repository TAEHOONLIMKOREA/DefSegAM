"""라벨 맵 — ORNL 빌드의 CAD 파트 배치 위에 결함을 규칙대로 얹는다 (v7/PLAN_v7.md §2.2).

    bank = ShapeBank(gt, rules)                                   # 생성 모델이 봐도 되는 사람 GT 에서 모양을 가져온다
    lab, placed = build_label_map(part, rng, bank, rules, edge)   # (S, S) uint8 — 0–9, "모름" = 10

바탕은 CAD 다: 파트 안 = Printed, 밖 = Powder, 파트 경계 ±3 px = "모름". 그 위에 네 가지를 얹는다.

  RS  굵기 2 px 의 가로 줄. 분말 위의 화소만 RS 로 적는다 (파트를 지나는 구간은 Printed 그대로)
  RH  세로 줄의 묶음 — 사람 GT 의 180² 조각에 든 RH 화소 전체. 분말 위에 놓는다
  SE  사람 GT 의 모양. 파트 가장자리에 걸쳐 놓는다
  DB  사람 GT 의 모양. 분말 위에 놓는다

자리 규칙의 근거 — 생성 모델이 본 사람 GT 87 레이어에서 잰 값 (2026-11-19, 객체 둘레 2–6 px 의 라벨):
  SE 의 98% 가 Printed 가 아닌 화소에서 3 px 안에 있다 (둘레의 49% 가 Powder, 35% 가 Printed) → 파트 안쪽이 아니라 가장자리다
  RS 둘레의 Printed 는 2.9%, RH 는 0.9% 다 → 사람은 파트 위에 줄을 칠하지 않는다
  DB 둘레의 93% 가 Powder 다

한 장에 넣는 개수는 "레이어당 평균 화소 수가 사람 GT 를 넘지 않게" 맞춘 값이다 (PLAN_v7 §1.5).
모양은 직선인 RS 를 빼고 모두 진짜에서 가져온다. 90° 회전은 쓰지 않는다 (리코팅 축이 가로다 — RS 는 가로 줄, RH 는 세로 줄).
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
from scipy import ndimage

from .. import config_genaug as cfg
from ..data.gt_pool import GTPool, O_CLS, O_SAMPLE, O_SX, O_SY, O_X0, O_X1, O_Y0, O_Y1
from ..sampling.preview import cad_label

RS, RH, SE, DB = 4, 5, 6, 9
_SQ = np.ones((3, 3), dtype=bool)


@dataclass(frozen=True)
class Rules:
    classes: tuple = cfg.SET_CLASSES       # 넣는 결함
    p_empty: float = 1.0 / 3.0             # 결함을 하나도 넣지 않는 레이어의 비율
    p_class: float = 0.75                  # 결함을 넣는 레이어에서 클래스마다 들어갈 확률 → 전체의 절반
    margin: int = 12                       # 그리는 안쪽의 가장자리에서 결함을 띄우는 거리
    tries: int = 200
    # RS — 가로 줄
    rs_lines: tuple = (1, 5)               # 줄 수 (고르게)
    rs_thickness: int = 2                  # L21 주석의 굵기 (1036² 에서 1.8 px)
    rs_full_prob: float = 0.5              # 안쪽 끝에서 끝까지 가는 줄의 비율
    rs_partial_len: tuple = (200, 420)
    rs_min_gap: int = 8                    # 줄 사이의 최소 간격 (행)
    rs_min_px: int = 150                   # 분말 위에 남는 화소가 이보다 적은 줄은 다시 고른다
    # RH — 세로 줄의 묶음
    rh_groups: tuple = (1, 4)
    rh_patch: int = 180
    rh_min_px: int = 60
    rh_on_powder: float = 0.95
    # SE · DB — 사람 GT 의 모양
    se_count: tuple = (10.0, 0.8, 36)      # 개수 = lognormal (중앙값, σ), 상한
    se_in_part: float = 0.5                # 모양의 이만큼 이상이 CAD 의 파트 안
    db_count: tuple = (10.0, 0.9, 40)
    db_on_powder: float = 0.9
    shape_max_side: int = 120              # 이보다 긴 객체는 쓰지 않는다
    scale: tuple = (0.8, 1.25)             # SE · DB 의 크기 배율

    def to_dict(self) -> dict:
        d = asdict(self)
        d["classes"] = [cfg.CLASS_ABBRS[c] for c in self.classes]
        return d


class ShapeBank:
    """결함의 모양 — 생성 모델이 봐도 되는 사람 GT (holdout 을 뺀 레이어) 에서만 가져온다."""

    def __init__(self, gt: GTPool, rules: Rules):
        self.gt = gt
        self.rows = {c: gt.objects_fitting(c, rules.shape_max_side) for c in (RH, SE, DB)}

    def available(self, cls: int) -> bool:
        return cls == RS or len(self.rows.get(cls, ())) > 0

    def blob(self, cls: int, rng: np.random.Generator) -> tuple[np.ndarray, str]:
        """객체 하나 → (bbox 로 자른 마스크, 가져온 레이어의 key)."""
        gt = self.gt
        o = gt.objects[int(rng.choice(self.rows[cls]))]
        j = int(o[O_SAMPLE])
        sl = (slice(int(o[O_Y0]), int(o[O_Y1])), slice(int(o[O_X0]), int(o[O_X1])))
        lab, _ = ndimage.label(gt.ann[j][sl] == cls, structure=_SQ)
        cid = int(lab[int(o[O_SY]) - sl[0].start, int(o[O_SX]) - sl[1].start])
        return lab == cid, gt.keys[j]

    def rh_group(self, rng: np.random.Generator, patch: int, min_px: int) -> tuple[np.ndarray, str] | None:
        """RH 객체 하나를 가운데 둔 patch² 조각의 RH 화소 전체 → (bbox 로 자른 마스크, key)."""
        gt = self.gt
        for _ in range(20):
            o = gt.objects[int(rng.choice(self.rows[RH]))]
            j, cy, cx = int(o[O_SAMPLE]), int(o[O_SY]), int(o[O_SX])
            y0 = int(np.clip(cy - patch // 2, 0, gt.img_size - patch))
            x0 = int(np.clip(cx - patch // 2, 0, gt.img_size - patch))
            m = gt.ann[j][y0:y0 + patch, x0:x0 + patch] == RH
            if int(m.sum()) >= min_px:
                return _trim(m), gt.keys[j]
        return None


def _trim(mask: np.ndarray) -> np.ndarray:
    ys, xs = np.nonzero(mask)
    return mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]


def _flip(mask: np.ndarray, rng: np.random.Generator) -> np.ndarray:
    if rng.random() < 0.5:
        mask = mask[:, ::-1]
    if rng.random() < 0.5:
        mask = mask[::-1]
    return np.ascontiguousarray(mask)


def _rescale(mask: np.ndarray, rng: np.random.Generator, lo: float, hi: float) -> np.ndarray:
    s = float(rng.uniform(lo, hi))
    out = ndimage.zoom(mask.astype(np.uint8), s, order=0) > 0
    return _trim(out) if out.any() else mask


def _count(rng: np.random.Generator, spec: tuple) -> int:
    median, sigma, top = spec
    return int(np.clip(round(float(rng.lognormal(np.log(median), sigma))), 1, top))


def _is_defect(a: np.ndarray) -> np.ndarray:
    return (a >= 2) & (a != cfg.UNKNOWN)


def _record(cls: int, y: int, x: int, mask: np.ndarray, px: int, src: str | None = None) -> dict:
    cy, cx = ndimage.center_of_mass(mask)
    d = {"cls": cfg.CLASS_ABBRS[cls], "y": int(round(y + cy)), "x": int(round(x + cx)), "px": int(px),
         "h": int(mask.shape[0]), "w": int(mask.shape[1])}
    if src is not None:
        d["shape_from"] = src
    return d


def _place_on_powder(lab, mask, cls, min_frac, rng, lo, hi, tries, overwrite_printed: bool) -> tuple[int, int, int] | None:
    """마스크를 분말 위 (min_frac 이상) 에 놓는다 → (y, x, 적은 화소 수)."""
    h, w = mask.shape
    if h >= hi - lo or w >= hi - lo:
        return None
    for _ in range(tries):
        y, x = int(rng.integers(lo, hi - h + 1)), int(rng.integers(lo, hi - w + 1))
        sub = lab[y:y + h, x:x + w]
        if float((sub[mask] == cfg.POWDER).mean()) < min_frac:
            continue
        sel = mask & ((sub < 2) if overwrite_printed else (sub == cfg.POWDER))
        sub[sel] = cls
        return y, x, int(sel.sum())
    return None


def _place_on_part_edge(lab, part, edge_yx, mask, rng, lo, hi, tries, in_part: float) -> tuple[int, int, int] | None:
    """SE — 마스크를 파트 가장자리에 걸쳐 놓는다 → (y, x, 적은 화소 수).

    조건: 모양의 in_part 이상이 CAD 의 파트 안이고, 모양이 (2 px 불린 것이) 파트 밖에 닿는다. 다른 결함과는 겹치지 않는다.
    파트 경계의 "모름" 띠도 SE 로 적는다 — 가장자리에 놓이는 결함이라 띠를 비우면 결함의 대부분이 라벨 없이 남는다.
    """
    ey, ex = edge_yx
    h, w = mask.shape
    if len(ey) == 0 or h >= hi - lo or w >= hi - lo:
        return None
    cy, cx = (int(round(v)) for v in ndimage.center_of_mass(mask))
    grown = ndimage.binary_dilation(np.pad(mask, 2), structure=_SQ, iterations=2)
    for _ in range(tries):
        k = int(rng.integers(len(ey)))
        y = int(ey[k]) - cy + int(rng.integers(-2, 3))
        x = int(ex[k]) - cx + int(rng.integers(-2, 3))
        if y < lo or x < lo or y + h > hi or x + w > hi:
            continue
        sub = lab[y:y + h, x:x + w]
        psub = part[y:y + h, x:x + w]
        if float(psub[mask].mean()) < in_part or _is_defect(sub[mask]).any():
            continue
        if not (~part[y - 2:y + h + 2, x - 2:x + w + 2])[grown].any():
            continue
        sel = mask & ~_is_defect(sub)
        sub[sel] = SE
        return y, x, int(sel.sum())
    return None


def build_label_map(part: np.ndarray, rng: np.random.Generator, bank: ShapeBank, rules: Rules,
                    edge: int) -> tuple[np.ndarray, list[dict]]:
    """CAD 의 파트 자리 (S, S) bool → (라벨 맵 uint8, 넣은 결함의 목록).

    edge = 사진 가장자리의 진짜 화소 폭 (그리지 않는 곳). 결함은 그 안쪽에서 margin 만큼 더 들어간 곳에만 놓는다.
    """
    S = part.shape[0]
    lab = cad_label(part)
    placed: list[dict] = []
    if rng.random() < rules.p_empty:
        return lab, placed
    classes = [c for c in rules.classes if bank.available(c) and rng.random() < rules.p_class]
    if not classes:
        ok = [c for c in rules.classes if bank.available(c)]
        if not ok:
            return lab, placed
        classes = [int(rng.choice(ok))]
    lo, hi = edge + rules.margin, S - edge - rules.margin

    if RS in classes:
        t = rules.rs_thickness
        rows: list[int] = []
        for _ in range(int(rng.integers(rules.rs_lines[0], rules.rs_lines[1] + 1))):
            for _ in range(rules.tries):
                y = int(rng.integers(lo + 20, hi - 20 - t))
                if any(abs(y - r) < rules.rs_min_gap + t for r in rows):
                    continue
                full = rng.random() < rules.rs_full_prob
                if full:
                    xa, xb = lo, hi
                else:
                    xa = int(rng.integers(lo, hi - rules.rs_partial_len[1]))
                    xb = xa + int(rng.integers(rules.rs_partial_len[0], rules.rs_partial_len[1] + 1))
                seg = lab[y:y + t, xa:xb]
                sel = seg == cfg.POWDER
                if int(sel.sum()) < rules.rs_min_px:
                    continue
                seg[sel] = RS
                rows.append(y)
                placed.append({"cls": "RS", "y": y, "x": (xa + xb) // 2, "px": int(sel.sum()),
                               "x0": xa, "x1": xb, "full": bool(full)})
                break

    if RH in classes:
        for _ in range(int(rng.integers(rules.rh_groups[0], rules.rh_groups[1] + 1))):
            got = bank.rh_group(rng, rules.rh_patch, rules.rh_min_px)
            if got is None:
                continue
            mask, src = _flip(got[0], rng), got[1]
            p = _place_on_powder(lab, mask, RH, rules.rh_on_powder, rng, lo, hi, rules.tries, overwrite_printed=False)
            if p is not None:
                placed.append(_record(RH, p[0], p[1], mask, p[2], src))

    if SE in classes:
        band = part & ~ndimage.binary_erosion(part, structure=_SQ, iterations=2)
        band[:lo], band[hi:], band[:, :lo], band[:, hi:] = False, False, False, False
        edge_yx = np.nonzero(band)
        for _ in range(_count(rng, rules.se_count)):
            mask, src = bank.blob(SE, rng)
            mask = _rescale(_flip(mask, rng), rng, *rules.scale)
            p = _place_on_part_edge(lab, part, edge_yx, mask, rng, lo, hi, rules.tries, rules.se_in_part)
            if p is not None:
                placed.append(_record(SE, p[0], p[1], mask, p[2], src))

    if DB in classes:
        for _ in range(_count(rng, rules.db_count)):
            mask, src = bank.blob(DB, rng)
            mask = _rescale(_flip(mask, rng), rng, *rules.scale)
            p = _place_on_powder(lab, mask, DB, rules.db_on_powder, rng, lo, hi, rules.tries, overwrite_printed=True)
            if p is not None:
                placed.append(_record(DB, p[0], p[1], mask, p[2], src))
    return lab, placed


def to_annotation(lab: np.ndarray, edge: int) -> np.ndarray:
    """라벨 맵 → 학습용 라벨 (int8). "모름" 과 사진 가장자리 (진짜 화소) 는 미레이블 -1."""
    ann = lab.astype(np.int8)
    ann[lab == cfg.UNKNOWN] = cfg.IGNORE_INDEX
    ann[:edge], ann[-edge:], ann[:, :edge], ann[:, -edge:] = cfg.IGNORE_INDEX, cfg.IGNORE_INDEX, cfg.IGNORE_INDEX, cfg.IGNORE_INDEX
    return ann
