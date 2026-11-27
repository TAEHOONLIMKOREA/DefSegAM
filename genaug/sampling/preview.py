"""표본 시트 — 학습 중과 학습 뒤에 생성 모델이 무엇을 그리는지 눈으로 본다 (PLAN_GENAUG §4 의 Q1 로 가는 첫 단계).

시트의 한 줄 = 창 하나:  진짜 visible/0 | 진짜 visible/1 | 라벨 맵 · 그릴 자리 | 그린 visible/0 | 그린 visible/1

줄의 종류 (모두 seed 로 고정된 같은 창이라 학습이 진행되는 동안 나란히 견줄 수 있다)
  A. GT  redraw   사람 GT 의 결함 객체를 제자리에서 다시 그린다 (그릴 자리 = 객체 + 띠)
  B. GT  window   사람 GT 창을 라벨 맵대로 창 전체 다시 그린다 (테두리 16 px 만 진짜)
  C. ORNL insert  주석 없는 ORNL 창에 결함 하나를 넣는다 — 실제로 쓸 방식 (PLAN §3.1)
  D. ORNL window  ORNL 창을 라벨 맵 (CAD 의 파트 배치 + 결함 하나) 대로 창 전체 그린다 (PLAN §3.9)

보기 좋게 하려고 줄마다 밝기를 1–99 백분위로 늘려 그린다 (진짜와 그린 것에 같은 변환). 표시만 그렇고 값은 원본 밝기다.
그림 안의 글자는 영문이다 (기본 글꼴에 한글이 없다).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage

from .. import config_genaug as cfg
from ..data.canvas_pool import CanvasPool
from ..data.gt_pool import GTPool, O_CLS, O_SAMPLE, O_SX, O_SY, O_X0, O_X1, O_Y0, O_Y1
from ..data.windows import OBJECT_MARGIN, crop_ornl, region_full
from ..models.diffusion import Diffusion
from .fill import fill_windows

_SQ = np.ones((3, 3), dtype=bool)
# 자리 규칙 (PLAN §3.3): 이 클래스는 이 바탕 위에 놓는다. 없는 클래스는 제한 없음
REQUIRED_BASE = {7: cfg.POWDER, 9: cfg.POWDER, 2: cfg.POWDER, 3: cfg.POWDER, 8: cfg.PRINTED, 6: cfg.PRINTED}


@dataclass
class PreviewItem:
    title: str
    i0: np.ndarray            # (W, W) uint8
    i1: np.ndarray
    label: np.ndarray         # (W, W) uint8 — 0–9, "모름" = 10
    region: np.ndarray        # (W, W) bool
    extra: dict = field(default_factory=dict)


def cad_label(part: np.ndarray) -> np.ndarray:
    """파트 자리 → 그리라고 줄 라벨 맵: 파트 안 = Printed, 밖 = Powder, 경계 ±3 px = "모름"."""
    edge = (ndimage.binary_dilation(part, structure=_SQ, iterations=cfg.PART_EDGE_PX)
            & ~ndimage.binary_erosion(part, structure=_SQ, iterations=cfg.PART_EDGE_PX))
    lab = np.where(part, cfg.PRINTED, cfg.POWDER).astype(np.uint8)
    lab[edge] = cfg.UNKNOWN
    return lab


def object_mask(gt: GTPool, row: int) -> tuple[np.ndarray, int, str]:
    """객체 표의 한 행 → (bbox 로 자른 마스크, 클래스, 소스)."""
    o = gt.objects[row]
    j, cls = int(o[O_SAMPLE]), int(o[O_CLS])
    sl = (slice(int(o[O_Y0]), int(o[O_Y1])), slice(int(o[O_X0]), int(o[O_X1])))
    lab, _ = ndimage.label(gt.ann[j][sl] == cls, structure=_SQ)
    cid = int(lab[int(o[O_SY]) - sl[0].start, int(o[O_SX]) - sl[1].start])
    return lab == cid, cls, gt.sources[j]


def place_mask(mask: np.ndarray, base_lab: np.ndarray, cls: int, rng: np.random.Generator, win: int, tries: int = 30):
    """객체 마스크를 창 안의 자리 규칙에 맞는 곳에 놓는다. 못 찾으면 None."""
    h, w = mask.shape
    m = OBJECT_MARGIN
    if h > win - 2 * m or w > win - 2 * m:
        return None
    need = REQUIRED_BASE.get(cls)
    for _ in range(tries):
        y, x = int(rng.integers(m, win - m - h + 1)), int(rng.integers(m, win - m - w + 1))
        full = np.zeros((win, win), dtype=bool)
        full[y:y + h, x:x + w] = mask
        if need is None or (base_lab[full] == need).mean() >= 0.9:
            return full
    return None


def build_preview(gt: GTPool, canvas: CanvasPool | None, normal_builds, win: int = cfg.WIN, seed: int = cfg.SEED,
                  per_class: int = 2, gen_classes=cfg.GEN_CLASSES) -> list[PreviewItem]:
    rng = np.random.default_rng([seed, 777])
    S = gt.img_size
    fit = win - 2 * OBJECT_MARGIN
    items: list[PreviewItem] = []
    classes = [c for c in gen_classes if len(gt.objects_fitting(c, fit)) > 0]

    def gt_window(row: int):
        o = gt.objects[row]
        j = int(o[O_SAMPLE])
        cy, cx = (int(o[O_Y0]) + int(o[O_Y1])) // 2, (int(o[O_X0]) + int(o[O_X1])) // 2
        y0, x0 = int(np.clip(cy - win // 2, 0, S - win)), int(np.clip(cx - win // 2, 0, S - win))
        sl = (slice(y0, y0 + win), slice(x0, x0 + win))
        ann = gt.ann[j][sl]
        lab, _ = ndimage.label(ann == int(o[O_CLS]), structure=_SQ)
        cid = int(lab[int(o[O_SY]) - y0, int(o[O_SX]) - x0])
        full_label = np.where(ann < 0, cfg.UNKNOWN, ann).astype(np.uint8)
        return j, gt.i0[j][sl].copy(), gt.i1[j][sl].copy(), full_label, lab == cid

    # A · B — 사람 GT
    picked: dict[int, list[int]] = {}
    for c in classes:
        rows = gt.objects_fitting(c, fit)
        picked[c] = [int(r) for r in rng.choice(rows, size=min(per_class, len(rows)), replace=False)]
        for r in picked[c]:
            j, i0, i1, lab, mask = gt_window(r)
            region = ndimage.binary_dilation(mask, structure=_SQ, iterations=cfg.RING_PX)
            items.append(PreviewItem(f"A  GT redraw   {cfg.CLASS_ABBRS[c]}  {gt.keys[j]}", i0, i1, lab, region))
    for c in classes[:3]:
        j, i0, i1, lab, _ = gt_window(picked[c][0])
        items.append(PreviewItem(f"B  GT window   {cfg.CLASS_ABBRS[c]}  {gt.keys[j]}", i0, i1, lab,
                                 region_full(win, min(cfg.BORDER, win // 8))))

    # C · D — 주석 없는 ORNL 창
    if canvas is not None:
        e = int(round(cfg.EDGE_FRAC * S))

        def ornl_window(need_part: bool):
            for _ in range(300):
                b, row = canvas.pick_layer(rng)
                y0, x0 = int(rng.integers(e, S - e - win + 1)), int(rng.integers(e, S - e - win + 1))
                i0, i1, _, part = crop_ornl(canvas, b, row, y0, x0, win, False)
                frac = float(part.mean())
                if (need_part and frac > 0.05) or (not need_part and frac < 0.5):
                    return b, row, y0, x0, i0, i1, part
            return None

        def insert(c: int, whole: bool):
            mask, _, src = object_mask(gt, picked[c][-1])
            need_part = REQUIRED_BASE.get(c) == cfg.PRINTED
            for _ in range(20):
                w_ = ornl_window(need_part)
                if w_ is None:
                    return
                b, row, y0, x0, i0, i1, part = w_
                base = cad_label(part)
                placed = place_mask(mask, base, c, rng, win)
                if placed is None:
                    continue
                lab = base.copy()
                lab[placed] = c
                where = f"{b}/L{int(canvas.orig_layers[b][row])}"
                if whole:
                    region = region_full(win, min(cfg.BORDER, win // 8))
                    items.append(PreviewItem(f"D  ORNL window {cfg.CLASS_ABBRS[c]}  {where}  shape from {src}", i0, i1, lab, region))
                else:
                    region = ndimage.binary_dilation(placed, structure=_SQ, iterations=cfg.RING_PX)
                    items.append(PreviewItem(f"C  ORNL insert {cfg.CLASS_ABBRS[c]}  {where}  shape from {src}", i0, i1, lab, region))
                return

        for c in classes:
            insert(c, whole=False)
        for c in classes[:3]:
            insert(c, whole=True)
        w_ = ornl_window(need_part=True)                              # 결함 없는 창 — "정상 창"
        if w_ is not None:
            b, row, y0, x0, i0, i1, part = w_
            items.append(PreviewItem(f"D  ORNL window normal  {b}/L{int(canvas.orig_layers[b][row])}", i0, i1, cad_label(part),
                                     region_full(win, min(cfg.BORDER, win // 8))))
    return items


def generate(model, diff: Diffusion, items: list[PreviewItem], steps: int, guidance: float, seed: int = 0, amp: bool = False):
    i0 = np.stack([it.i0 for it in items])
    i1 = np.stack([it.i1 for it in items])
    region = np.stack([it.region for it in items])
    label = np.stack([it.label for it in items])
    return fill_windows(model, diff, i0, i1, region, label, steps=steps, guidance=guidance, seed=seed, batch=16, amp=amp)


def _stretch(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    return np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1.0) * 255.0, 0, 255).astype(np.uint8)


def _outline(mask: np.ndarray) -> np.ndarray:
    return mask & ~ndimage.binary_erosion(mask, structure=_SQ)


def render_sheet(items: list[PreviewItem], g0: np.ndarray, g1: np.ndarray, path: Path, header: str = "") -> None:
    win = items[0].i0.shape[0]
    scale = 1 if win >= 256 else max(1, 256 // win)
    P = win * scale
    gap, th = 6, 14
    W = 5 * P + 6 * gap
    H = 34 + len(items) * (P + th + gap)
    sheet = Image.new("RGB", (W, H), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    d.text((gap, 3), header[:200], fill=(255, 255, 255))
    d.text((gap, 17), "real visible/0 | real visible/1 | label map (class outlines) + region (yellow) | generated visible/0 | generated visible/1"
                      "     [display: per-row 1-99% stretch]", fill=(190, 190, 190))

    def up(a: np.ndarray) -> np.ndarray:
        return a.repeat(scale, axis=0).repeat(scale, axis=1) if scale > 1 else a

    y = 34
    for k, it in enumerate(items):
        d.text((gap, y), it.title, fill=(255, 230, 120))
        y += th
        panels = []
        for ch, (real, gen) in enumerate(((it.i0, g0[k]), (it.i1, g1[k]))):
            lo, hi = np.percentile(real, [1, 99])
            panels.append((_stretch(real, lo, hi), _stretch(gen, lo, hi)))
        base = np.stack([panels[1][0]] * 3, axis=-1)
        for c in range(cfg.N_CLASSES):
            m = (it.label == c) & it.region
            if c >= 2 and m.any():
                base[_outline(m)] = cfg.CLASS_COLORS[c]
        base[_outline(it.region)] = (255, 255, 0)
        tiles = [np.stack([panels[0][0]] * 3, -1), np.stack([panels[1][0]] * 3, -1), base,
                 np.stack([panels[0][1]] * 3, -1), np.stack([panels[1][1]] * 3, -1)]
        for i, t in enumerate(tiles):
            sheet.paste(Image.fromarray(up(t)), (gap + i * (P + gap), y))
        y += P + gap
    path.parent.mkdir(parents=True, exist_ok=True)
    sheet.save(path)


@torch.no_grad()
def write_preview(model, diff: Diffusion, items: list[PreviewItem], path: Path, header: str,
                  steps: int = cfg.SAMPLE_STEPS, guidance: float = cfg.SAMPLE_CFG, seed: int = 0, amp: bool = False) -> None:
    was = model.training
    model.eval()
    g0, g1 = generate(model, diff, items, steps, guidance, seed=seed, amp=amp)
    render_sheet(items, g0, g1, path, header)
    if was:
        model.train()
