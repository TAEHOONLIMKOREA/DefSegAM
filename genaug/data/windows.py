"""생성 모델의 학습 창 (PLAN_GENAUG §3.2).

창 하나 = 256×256 의 (visible/0, visible/1) + 라벨 맵 + 그릴 자리.

  학습 자료 ①  사람 GT 창 — 70% 는 결함 객체를 가운데 두고 (클래스 균등), 30% 는 아무 데서나
  학습 자료 ②  주석 없는 ORNL 창 — 라벨 맵은 Q0 을 통과한 빌드에서만 적는다 (덮임 판정을 통과한 파트 = Printed,
               파트 밖 = Powder, 나머지 = "모름"). 그 밖의 빌드는 전부 "모름"

그릴 자리는 작은 것 (결함 하나) 부터 창 전체까지 섞는다 — 같은 모델이 "결함 자리만 그리기" 와 "창 전체 그리기" 를 다 하도록.
라벨 맵은 **그릴 자리 안에만** 적는다. 그 밖의 화소는 모델이 사진으로 직접 보므로 "모름" 으로 둔다
(그릴 때 주변의 라벨을 모르는 배경에서도 학습 때와 같은 조건이 되도록).
"""
from __future__ import annotations

import numpy as np
import torch
from scipy import ndimage
from torch.utils.data import Dataset, get_worker_info

from .. import config_genaug as cfg
from .canvas_pool import CanvasPool
from .covered import covered_mask
from .gt_pool import GTPool, O_CLS, O_SAMPLE, O_SX, O_SY, O_X0, O_X1, O_Y0, O_Y1

_SQ = np.ones((3, 3), dtype=bool)
KIND_GT_DEFECT, KIND_GT_RANDOM, KIND_ORNL = 0, 1, 2
OBJECT_MARGIN = cfg.RING_PX + 2          # 객체와 창 가장자리 사이에 두는 여유 (띠가 들어갈 자리)


# ---------------------------------------------------------------------------
# 그릴 자리
# ---------------------------------------------------------------------------

def region_box(rng: np.random.Generator, win: int) -> np.ndarray:
    lo, hi = max(8, win // 8), max(9, int(win * 0.78) + 1)
    h, w = int(rng.integers(lo, hi)), int(rng.integers(lo, hi))
    y, x = int(rng.integers(0, win - h + 1)), int(rng.integers(0, win - w + 1))
    m = np.zeros((win, win), dtype=bool)
    m[y:y + h, x:x + w] = True
    return m


def region_blob(rng: np.random.Generator, win: int) -> np.ndarray:
    sigma = float(rng.uniform(win / 32, win / 10))
    f = ndimage.gaussian_filter(rng.standard_normal((win, win)).astype(np.float32), sigma)
    return f > np.quantile(f, 1.0 - float(rng.uniform(0.1, 0.6)))


def region_full(win: int, border: int) -> np.ndarray:
    m = np.zeros((win, win), dtype=bool)
    m[border:win - border, border:win - border] = True
    return m


def region_object(mask: np.ndarray, rng: np.random.Generator, as_box_prob: float = 0.3) -> np.ndarray:
    """결함 객체의 마스크 → 그릴 자리 (객체 + 둘레의 띠). 가끔은 객체를 감싼 사각형으로."""
    r = int(rng.integers(4, 11))
    if rng.random() < as_box_prob:
        ys, xs = np.nonzero(mask)
        m = np.zeros_like(mask)
        m[max(0, ys.min() - r):ys.max() + r + 1, max(0, xs.min() - r):xs.max() + r + 1] = True
        return m
    return ndimage.binary_dilation(mask, structure=_SQ, iterations=r)


def make_region(kind: str, rng: np.random.Generator, win: int, border: int = cfg.BORDER) -> np.ndarray:
    if kind == "box":
        return region_box(rng, win)
    if kind == "blob":
        return region_blob(rng, win)
    if kind == "full":
        return region_full(win, min(border, win // 8))
    if kind == "full0":
        return np.ones((win, win), dtype=bool)
    raise ValueError(kind)


def pick(rng: np.random.Generator, mix: dict[str, float]) -> str:
    names = list(mix)
    p = np.array([mix[k] for k in names], dtype=np.float64)
    return names[int(rng.choice(len(names), p=p / p.sum()))]


# ---------------------------------------------------------------------------
# 주석 없는 ORNL 창의 라벨 맵
# ---------------------------------------------------------------------------

def ornl_label_map(i1: np.ndarray, part: np.ndarray, normal: bool) -> np.ndarray:
    """ORNL 창 (둘레에 여유를 두고 자른 것) 의 라벨 맵 — uint8, 값은 Powder · Printed · "모름".

    normal = 이 빌드에 정상 라벨을 적어도 되는가 (Q0 통과). 아니면 전부 "모름".
    """
    lab = np.full(part.shape, cfg.UNKNOWN, dtype=np.uint8)
    if not normal:
        return lab
    edge = (ndimage.binary_dilation(part, structure=_SQ, iterations=cfg.PART_EDGE_PX)
            & ~ndimage.binary_erosion(part, structure=_SQ, iterations=cfg.PART_EDGE_PX))
    lab[~part & ~edge] = cfg.POWDER
    if part.any():
        lab[covered_mask(i1, part) & ~edge] = cfg.PRINTED
    return lab


def crop_ornl(canvas: CanvasPool, build: str, row: int, y0: int, x0: int, win: int, normal: bool):
    """ORNL 창 하나 → (i0, i1, 라벨 맵, 파트 자리). 덮임 판정은 창 둘레에 여유를 두고 계산한 뒤 가운데만 쓴다."""
    S = canvas.img_size
    pad = cfg.COVERED_PAD if normal else 0
    ya, xa = max(0, y0 - pad), max(0, x0 - pad)
    yb, xb = min(S, y0 + win + pad), min(S, x0 + win + pad)
    i0, i1, part = canvas.crop(build, row, ya, xa, yb - ya, xb - xa)
    lab = ornl_label_map(i1, part, normal)
    sl = (slice(y0 - ya, y0 - ya + win), slice(x0 - xa, x0 - xa + win))
    return i0[sl], i1[sl], lab[sl], part[sl]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

def to_model_arrays(i0: np.ndarray, i1: np.ndarray, label: np.ndarray, region: np.ndarray):
    """uint8 창 → 모델에 넣는 텐서. 라벨 맵은 그릴 자리 밖을 "모름" 으로 바꾼다."""
    x = np.stack([i0, i1]).astype(np.float32) / 127.5 - 1.0
    lab = np.where(region, label, cfg.UNKNOWN).astype(np.uint8)
    return (torch.from_numpy(np.ascontiguousarray(x)),
            torch.from_numpy(np.ascontiguousarray(lab)),
            torch.from_numpy(np.ascontiguousarray(region[None].astype(np.float32))))


class FillWindows(Dataset):
    """끝없이 무작위 창을 내는 Dataset. 색인은 쓰지 않는다 (길이는 학습 step × batch 로 준다)."""

    def __init__(self, gt: GTPool, canvas: CanvasPool | None, n_virtual: int, seed: int, rank: int = 0,
                 normal_builds=cfg.NORMAL_LABEL_BUILDS, win: int = cfg.WIN, p_labeled: float = cfg.P_LABELED,
                 gen_classes=cfg.GEN_CLASSES, stream: int = 0):
        self.gt, self.canvas = gt, canvas
        self.n_virtual = int(n_virtual)
        self.seed, self.rank, self.stream = int(seed), int(rank), int(stream)
        self.normal_builds = tuple(normal_builds)
        self.win = int(win)
        self.p_labeled = float(p_labeled) if canvas is not None else 1.0
        self.S = gt.img_size
        if canvas is not None and canvas.img_size != self.S:
            raise RuntimeError(f"FATAL: 배경 캐시의 크기 {canvas.img_size} 가 사람 GT 의 {self.S} 와 다르다")
        if self.win > self.S:
            raise ValueError("창이 사진보다 크다")
        fit = self.win - 2 * OBJECT_MARGIN
        self.obj_rows = {c: gt.objects_fitting(c, fit) for c in gen_classes}
        self.def_classes = [c for c in gen_classes if len(self.obj_rows[c]) > 0]
        self.edge = int(round(cfg.EDGE_FRAC * self.S))
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return self.n_virtual

    def rng(self) -> np.random.Generator:
        if self._rng is None:
            wi = get_worker_info()
            self._rng = np.random.default_rng([self.seed, self.rank, wi.id if wi is not None else 0, self.stream])
        return self._rng

    # -- 사람 GT 창 ---------------------------------------------------------
    def _gt_defect(self, rng):
        win, S, m = self.win, self.S, OBJECT_MARGIN
        cls = int(self.def_classes[int(rng.integers(len(self.def_classes)))])
        rows = self.obj_rows[cls]
        o = self.gt.objects[int(rows[int(rng.integers(len(rows)))])]
        j = int(o[O_SAMPLE])

        def span(a0, a1):
            lo, hi = max(0, a1 + m - win), min(S - win, a0 - m)
            if lo > hi:                                            # 사진 가장자리에 붙은 객체 — 여유를 포기한다
                lo, hi = max(0, a1 - win), min(S - win, a0)
            return int(rng.integers(lo, hi + 1))

        y0, x0 = span(int(o[O_Y0]), int(o[O_Y1])), span(int(o[O_X0]), int(o[O_X1]))
        sl = (slice(y0, y0 + win), slice(x0, x0 + win))
        i0, i1, ann = self.gt.i0[j][sl], self.gt.i1[j][sl], self.gt.ann[j][sl]
        kind = pick(rng, cfg.REGION_MIX_DEFECT)
        if kind == "object":
            lab, _ = ndimage.label(ann == cls, structure=_SQ)
            cid = int(lab[int(o[O_SY]) - y0, int(o[O_SX]) - x0])
            region = region_object(lab == cid, rng) if cid > 0 else make_region("box", rng, win)
        else:
            region = make_region(kind, rng, win)
        return i0, i1, ann, region, KIND_GT_DEFECT, cls

    def _gt_random(self, rng):
        win, S = self.win, self.S
        j = int(rng.integers(len(self.gt)))
        y0, x0 = int(rng.integers(0, S - win + 1)), int(rng.integers(0, S - win + 1))
        sl = (slice(y0, y0 + win), slice(x0, x0 + win))
        region = make_region(pick(rng, cfg.REGION_MIX_OTHER), rng, win)
        return self.gt.i0[j][sl], self.gt.i1[j][sl], self.gt.ann[j][sl], region, KIND_GT_RANDOM, -1

    # -- 주석 없는 ORNL 창 --------------------------------------------------
    def _ornl(self, rng):
        win, S, e = self.win, self.S, self.edge
        build, row = self.canvas.pick_layer(rng)
        y0, x0 = int(rng.integers(e, S - e - win + 1)), int(rng.integers(e, S - e - win + 1))
        i0, i1, lab, _ = crop_ornl(self.canvas, build, row, y0, x0, win, build in self.normal_builds)
        region = make_region(pick(rng, cfg.REGION_MIX_OTHER), rng, win)
        return i0, i1, lab, region, KIND_ORNL, -1

    def sample(self, rng: np.random.Generator) -> dict:
        if rng.random() < self.p_labeled:
            if self.def_classes and rng.random() < cfg.P_DEFECT_CENTERED:
                i0, i1, lab, region, kind, cls = self._gt_defect(rng)
            else:
                i0, i1, lab, region, kind, cls = self._gt_random(rng)
            lab = np.where(lab < 0, cfg.UNKNOWN, lab).astype(np.uint8)        # 미레이블 → "모름"
        else:
            i0, i1, lab, region, kind, cls = self._ornl(rng)

        # 증강 — 좌우 · 상하 뒤집기 (둘 다 걸리면 180° 회전). 90° 회전은 쓰지 않는다 (RS 는 가로 줄, RH 는 세로 줄)
        if rng.random() < 0.5:
            i0, i1, lab, region = i0[:, ::-1], i1[:, ::-1], lab[:, ::-1], region[:, ::-1]
        if rng.random() < 0.5:
            i0, i1, lab, region = i0[::-1], i1[::-1], lab[::-1], region[::-1]
        g = float(rng.uniform(*cfg.GAIN_RANGE))                                # 밝기 — 두 사진에 같은 배율
        i0 = np.clip(i0.astype(np.float32) * g, 0, 255)
        i1 = np.clip(i1.astype(np.float32) * g, 0, 255)

        x, lab_t, region_t = to_model_arrays(i0, i1, lab, region)
        return {"x": x, "label": lab_t, "region": region_t, "kind": kind, "cls": cls}

    def __getitem__(self, idx: int) -> dict:
        return self.sample(self.rng())
