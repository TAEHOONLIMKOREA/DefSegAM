"""K1 — 합성 세트를 눈으로 보는 시트 (v7/PLAN_v7.md §2.6). **사람이 본다.**

만드는 것 (figures/genaug/sets/<세트>/):
  overview_<빌드>.png   레이어를 통째로 — 한 줄이 한 장:  진짜 visible/0 | 진짜 visible/1 | 라벨 맵 | 그린 visible/0 | 그린 visible/1
  zoom_<클래스>.png     결함 하나씩 확대 (128 px → 2배) — 한 칸이 [라벨 맵 | 그린 visible/0 | 그린 visible/1]
  zoom_normal.png       결함을 넣지 않은 곳의 확대 — 한 줄이 [진짜 v0 | 그린 v0 | 진짜 v1 | 그린 v1] (같은 레이어의 같은 자리)

볼 것: 그 클래스로 보이는가 (zoom_<클래스>), 분말과 파트가 그 장비의 것으로 보이는가 (overview · zoom_normal).
보기 좋게 하려고 밝기를 늘려 그린다. 진짜와 그린 것을 나란히 놓는 시트 (overview · zoom_normal) 는 **둘에 같은 변환**을 쓴다
(두 사진의 1–99 백분위를 함께 덮는 범위 — 밝기의 차이가 그대로 보인다). 결함 확대 (zoom_<클래스>) 는 그린 사진 자신의 1–99 백분위로 늘린다.
표시만 그렇고 저장된 값은 원본 밝기다. 그림 안의 글자는 영문이다.

실행 (CWD = 3DP_VPPM/, DefSeg_AM/venv — compose 의 check phase. GPU 를 쓰지 않는다):
    python -m DefSeg_AM.genaug.checks.synth_sheet --set w600_v1
"""
from __future__ import annotations

import argparse

import numpy as np
from PIL import Image, ImageDraw

from ...common.utils.log import setup_logger
from .. import config_genaug as cfg
from ..data.canvas_pool import CanvasPool
from ..synth.layer import frame_px
from ..synth.set_io import read_layer, read_manifest, resolve_set

BG = (24, 24, 24)


def color(ann: np.ndarray) -> np.ndarray:
    out = np.zeros(ann.shape + (3,), dtype=np.uint8)
    out[ann == cfg.POWDER] = (70, 70, 70)
    out[ann == cfg.PRINTED] = (205, 205, 205)
    for c in range(2, cfg.N_CLASSES):
        out[ann == c] = cfg.CLASS_COLORS[c]
    return out                                    # 미레이블 (-1) 은 검정


def stretch(a: np.ndarray, lo: float, hi: float) -> np.ndarray:
    g = np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1.0) * 255.0, 0, 255).astype(np.uint8)
    return np.stack([g, g, g], axis=-1)


class Layer:
    """한 장을 읽어 표시용으로 준비한다.

    gen · real   진짜와 그린 것에 같은 변환 (두 사진의 1–99 백분위를 함께 덮는 범위)
    gen_own      그린 사진 자신의 1–99 백분위 (결함 확대용)
    """

    def __init__(self, root, canvas: CanvasPool, m: dict, S: int):
        self.m = m
        g0, g1, self.ann = read_layer(root, m["id"])
        r0, r1, _ = canvas.crop(m["build"], m["row"], 0, 0, S, S)
        E = frame_px(S)
        inner = (slice(E, S - E), slice(E, S - E))
        both, own = [], []
        for g, r in ((g0, r0), (g1, r1)):
            pg, pr = np.percentile(g[inner], [1, 99]), np.percentile(r[inner], [1, 99])
            both.append((min(pg[0], pr[0]), max(pg[1], pr[1])))
            own.append(pg)
        self.gen = (stretch(g0, *both[0]), stretch(g1, *both[1]))
        self.real = (stretch(r0, *both[0]), stretch(r1, *both[1]))
        self.gen_own = (stretch(g0, *own[0]), stretch(g1, *own[1]))
        self.lab = color(self.ann)
        self.name = f"{m['id']} {m['build']}/L{m['orig_layer']}"


def crop(a: np.ndarray, cy: int, cx: int, half: int, out: int) -> Image.Image:
    S = a.shape[0]
    y0, x0 = int(np.clip(cy - half, 0, S - 2 * half)), int(np.clip(cx - half, 0, S - 2 * half))
    return Image.fromarray(a[y0:y0 + 2 * half, x0:x0 + 2 * half]).resize((out, out), Image.NEAREST)


def overview(root, canvas, layers: list[dict], S: int, path, scale: float = 0.4) -> None:
    P = int(round(S * scale))
    gap, th = 6, 14
    sheet = Image.new("RGB", (5 * P + 6 * gap, 22 + len(layers) * (P + th + gap)), BG)
    d = ImageDraw.Draw(sheet)
    d.text((gap, 4), "real visible/0 | real visible/1 | label map | generated visible/0 | generated visible/1"
                     "     [only the outer 5% of the generated photos is real; display: one stretch for real and generated]", fill=(220, 220, 220))
    y = 22
    for m in layers:
        L = Layer(root, canvas, m, S)
        d.text((gap, y), f"{L.name}   defects: " + (", ".join(sorted({x['cls'] for x in m['defects']})) or "none"), fill=(255, 230, 120))
        y += th
        for i, t in enumerate((L.real[0], L.real[1], L.lab, L.gen[0], L.gen[1])):
            im = Image.fromarray(t).resize((P, P), Image.NEAREST if i == 2 else Image.LANCZOS)
            sheet.paste(im, (gap + i * (P + gap), y))
        y += P + gap
    sheet.save(path)


def zoom_class(root, canvas, man: dict, cls: str, n: int, S: int, path, rng: np.random.Generator) -> int:
    half = 96 if cls == "RH" else 64
    T, gap, th, per_row = 256, 4, 14, 2
    cand = [(m, d) for m in man["layers"] for d in m["defects"] if d["cls"] == cls]
    if not cand:
        return 0
    seen: dict[str, int] = {}
    picks = []
    for k in rng.permutation(len(cand)):
        m, d = cand[int(k)]
        if seen.get(m["id"], 0) >= 2:                            # 한 장에서 둘까지만
            continue
        seen[m["id"]] = seen.get(m["id"], 0) + 1
        picks.append((m, d))
        if len(picks) == n:
            break
    rows = (len(picks) + per_row - 1) // per_row
    cellw = 3 * T + 2 * gap
    sheet = Image.new("RGB", (per_row * cellw + (per_row + 1) * 3 * gap, 22 + rows * (T + th + gap)), BG)
    dr = ImageDraw.Draw(sheet)
    dr.text((gap, 4), f"{cls}: label map | generated visible/0 | generated visible/1   ({2 * half} px crops, {len(picks)} of {len(cand)} objects; "
                      "display: 1-99% stretch of the generated photo)", fill=(220, 220, 220))
    cache: dict[str, Layer] = {}
    for i, (m, d) in enumerate(picks):
        if m["id"] not in cache:
            if len(cache) > 8:
                cache.clear()
            cache[m["id"]] = Layer(root, canvas, m, S)
        L = cache[m["id"]]
        cx = d["x"] if cls != "RS" else int(rng.integers(d["x0"] + half // 2, max(d["x0"] + half // 2 + 1, d["x1"] - half // 2)))
        x0 = 3 * gap + (i % per_row) * (cellw + 3 * gap)
        y0 = 22 + (i // per_row) * (T + th + gap)
        dr.text((x0, y0), f"{L.name}  ({d['px']} px)", fill=(255, 230, 120))
        for j, a in enumerate((L.lab, L.gen_own[0], L.gen_own[1])):
            sheet.paste(crop(a, d["y"], cx, half, T), (x0 + j * (T + gap), y0 + th))
    sheet.save(path)
    return len(picks)


def zoom_normal(root, canvas, man: dict, n: int, S: int, path, rng: np.random.Generator) -> int:
    E = frame_px(S)
    half, T, gap, th = 64, 256, 4, 14
    layers = [m for m in man["layers"] if not m["defects"]] or man["layers"]
    picks = [layers[int(k)] for k in rng.permutation(len(layers))[:n]]
    sheet = Image.new("RGB", (4 * T + 5 * gap, 22 + len(picks) * (T + th + gap)), BG)
    dr = ImageDraw.Draw(sheet)
    dr.text((gap, 4), "no defect placed: real visible/0 | generated visible/0 | real visible/1 | generated visible/1   (128 px crops, same layer, same place)",
            fill=(220, 220, 220))
    for i, m in enumerate(picks):
        L = Layer(root, canvas, m, S)
        cy, cx = int(rng.integers(E + 100, S - E - 100)), int(rng.integers(E + 100, S - E - 100))
        y0 = 22 + i * (T + th + gap)
        dr.text((gap, y0), f"{L.name}  at ({cy}, {cx})", fill=(255, 230, 120))
        for j, a in enumerate((L.real[0], L.gen[0], L.real[1], L.gen[1])):
            sheet.paste(crop(a, cy, cx, half, T), (gap + j * (T + gap), y0 + th))
    sheet.save(path)
    return len(picks)


def main():
    ap = argparse.ArgumentParser(description="GenAug K1 — 합성 세트의 시트")
    ap.add_argument("--set", default=cfg.SET_NAME)
    ap.add_argument("--per-class", type=int, default=24)
    ap.add_argument("--overview", type=int, default=4, help="빌드마다 통째로 보여 줄 장 수")
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    log = setup_logger(0, "genaug.sheet")
    root = resolve_set(args.set)
    man = read_manifest(root)
    S = man["img_size"]
    canvas = CanvasPool(tuple(man["builds"]))
    out = cfg.set_fig_dir(man["name"])
    out.mkdir(parents=True, exist_ok=True)
    rng = np.random.default_rng(args.seed)
    for b in man["builds"]:
        ls = [m for m in man["layers"] if m["build"] == b]
        with_def = [m for m in ls if m["defects"]]
        pick = (with_def or ls)[:max(args.overview - 1, 1)] + [m for m in ls if not m["defects"]][:1]
        overview(root, canvas, pick[:args.overview], S, out / f"overview_{b}.png")
    made = {c: zoom_class(root, canvas, man, c, args.per_class, S, out / f"zoom_{c}.png", rng) for c in man["classes"]}
    made["normal"] = zoom_normal(root, canvas, man, 8, S, out / "zoom_normal.png", rng)
    log.info(f"K1 시트 — 세트 {man['name']}: overview_{{{','.join(man['builds'])}}}.png, 확대 {made} → {out}")
    log.info("  사람이 본다: 클래스마다 그 결함으로 보이는가, 분말 · 파트가 그 장비의 것으로 보이는가")


if __name__ == "__main__":
    main()
