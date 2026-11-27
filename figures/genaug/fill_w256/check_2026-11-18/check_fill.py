"""fill_w256 (100,000 step) 결과 점검 — 표본 시트의 원본 값 통계와 확대 그림. 저장소 밖 (scratchpad) 에만 쓴다."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage

from DefSeg_AM.common.utils.log import setup_logger
from DefSeg_AM.genaug import config_genaug as cfg
from DefSeg_AM.genaug.data.canvas_pool import CanvasPool
from DefSeg_AM.genaug.data.gt_pool import GTPool
from DefSeg_AM.genaug.data.windows import OBJECT_MARGIN, crop_ornl, region_full
from DefSeg_AM.genaug.sampling.fill import fill_windows, load_generator
from DefSeg_AM.genaug.sampling.preview import REQUIRED_BASE, build_preview, cad_label, generate, object_mask, place_mask

OUT = Path(sys.argv[1])
CKPT = sys.argv[2]
OUT.mkdir(parents=True, exist_ok=True)
log = setup_logger(0, "check")
dev = torch.device("cuda:0")
W = cfg.WIN
SQ = np.ones((3, 3), bool)
FULL = region_full(W, cfg.BORDER)
res = {}


def grain(a):
    a = a.astype(np.float32)
    return a - ndimage.gaussian_filter(a, 1.5)


def win_stats(img, region):
    a = img.astype(np.float32)
    g = grain(img)
    b = cfg.BORDER
    r = a[b:W - b, b:W - b].mean(axis=1)
    band_in = region & ~ndimage.binary_erosion(region, structure=SQ, iterations=8)
    band_out = ~region
    return dict(mean=float(a[region].mean()), std=float(a[region].std()), grain=float(g[region].std()),
                streak=float((r - ndimage.gaussian_filter1d(r, 8)).std()),
                seam_mean=float(a[band_in].mean() - a[band_out].mean()),
                seam_grain=float(g[band_in].std() / max(g[band_out].std(), 1e-6)))


def med(xs, k):
    return float(np.median([x[k] for x in xs]))


def stretch(a, lo, hi):
    return np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1.0) * 255.0, 0, 255).astype(np.uint8)


def zoom(a, cy, cx, half=48, scale=3):
    y0, x0 = int(np.clip(cy - half, 0, W - 2 * half)), int(np.clip(cx - half, 0, W - 2 * half))
    return a[y0:y0 + 2 * half, x0:x0 + 2 * half].repeat(scale, 0).repeat(scale, 1)


t0 = time.time()
model, diff, ck = load_generator(CKPT, dev)
log.info(f"ckpt step {ck['step']}  normal_builds {ck['config']['normal_builds']}  holdout {ck['config']['holdout']}")
gt = GTPool("both", log=log)
canvas = CanvasPool(cfg.TRAIN_BUILDS, log=log)
S = gt.img_size
e = int(round(cfg.EDGE_FRAC * S))

# ── 1. 학습 중 시트와 같은 22 줄 — 작은 그릴 자리 (A · C) 를 확대해서 본다 ─────────────
items = build_preview(gt, canvas, tuple(ck["config"]["normal_builds"]), win=W, seed=cfg.SEED)
g0, g1 = generate(model, diff, items, steps=cfg.SAMPLE_STEPS, guidance=cfg.SAMPLE_CFG, seed=cfg.SEED, amp=True)
small = [k for k, it in enumerate(items) if it.title[0] in "AC"]
T = 96 * 3
sheet = Image.new("RGB", (4 * T + 5 * 6, len(small) * (T + 20) + 20), (24, 24, 24))
d = ImageDraw.Draw(sheet)
d.text((6, 3), "zoom x3 around the region:  real v0 | generated v0 | real v1 | generated v1   (yellow = region, on the real tiles)", fill=(255, 255, 255))
y = 20
for k in small:
    it = items[k]
    cy, cx = (int(v) for v in ndimage.center_of_mass(it.region))
    d.text((6, y), it.title, fill=(255, 230, 120))
    out = it.region & ~ndimage.binary_erosion(it.region, structure=SQ)
    for ch, (real, gen) in enumerate(((it.i0, g0[k]), (it.i1, g1[k]))):
        lo, hi = np.percentile(real, [1, 99])
        r = np.stack([stretch(real, lo, hi)] * 3, -1)
        r[out] = (255, 255, 0)
        gg = np.stack([stretch(gen, lo, hi)] * 3, -1)
        sheet.paste(Image.fromarray(zoom(r, cy, cx)), (6 + (2 * ch) * (T + 6), y + 14))
        sheet.paste(Image.fromarray(zoom(gg, cy, cx)), (6 + (2 * ch + 1) * (T + 6), y + 14))
    y += T + 20
sheet.save(OUT / "zoom_sheet_rows.png")
# 그릴 자리 밖이 그대로인가
same = all(np.array_equal(g0[k][~it.region], it.i0[~it.region]) and np.array_equal(g1[k][~it.region], it.i1[~it.region])
           for k, it in enumerate(items))
res["outside_unchanged_22rows"] = bool(same)
log.info(f"1. 22 줄 다시 그림 — 그릴 자리 밖이 그대로: {same}  ({time.time() - t0:.0f}초)")

# ── 2. 창 전체 그리기 — 원본 값으로 견준 질감 (진짜 대 그린 것) ─────────────────────
N = 32
rng = np.random.default_rng(2026)
res["whole_window"] = {}


def run_whole(name, i0, i1, lab, guidance):
    region = np.stack([FULL] * len(i0))
    q0, q1 = fill_windows(model, diff, i0, i1, region, lab, steps=cfg.SAMPLE_STEPS, guidance=guidance, seed=7, batch=32, amp=True)
    out = {}
    for ch, (real, gen) in (("v0", (i0, q0)), ("v1", (i1, q1))):
        rs = [win_stats(real[n], FULL) for n in range(len(real))]
        gs = [win_stats(gen[n], FULL) for n in range(len(gen))]
        out[ch] = {k: dict(real=round(med(rs, k), 3), gen=round(med(gs, k), 3)) for k in rs[0]}
        out[ch]["mean_abs_diff"] = round(float(np.median([abs(g["mean"] - r["mean"]) for g, r in zip(gs, rs)])), 3)
        out[ch]["seam_mean_abs"] = dict(real=round(float(np.median([abs(r["seam_mean"]) for r in rs])), 3),
                                        gen=round(float(np.median([abs(g["seam_mean"]) for g in gs])), 3))
    res["whole_window"][name] = out
    log.info(f"2. {name}: " + "  ".join(
        f"{ch} 평균 {out[ch]['mean']['real']:.1f}/{out[ch]['mean']['gen']:.1f} 결 {out[ch]['grain']['real']:.2f}/{out[ch]['grain']['gen']:.2f} "
        f"줄 {out[ch]['streak']['real']:.2f}/{out[ch]['streak']['gen']:.2f} 이음매결 {out[ch]['seam_grain']['real']:.2f}/{out[ch]['seam_grain']['gen']:.2f}"
        for ch in ("v0", "v1")) + "   (진짜/그린 것)")
    return q0, q1


keep = {}
for b in cfg.TRAIN_BUILDS:
    I0, I1, PART = [], [], []
    for _ in range(N):
        row = int(rng.integers(canvas.n_layers[b]))
        y0, x0 = int(rng.integers(e, S - e - W + 1)), int(rng.integers(e, S - e - W + 1))
        a0, a1, _, part = crop_ornl(canvas, b, row, y0, x0, W, False)
        I0.append(a0.copy()); I1.append(a1.copy()); PART.append(part.copy())
    I0, I1, PART = np.stack(I0), np.stack(I1), np.stack(PART)
    lab_cad = np.stack([cad_label(p) for p in PART])
    lab_unk = np.full_like(lab_cad, cfg.UNKNOWN)
    qc = run_whole(f"{b}_cad_label", I0, I1, lab_cad, cfg.SAMPLE_CFG)
    qu = run_whole(f"{b}_no_label", I0, I1, lab_unk, 1.0)
    keep[b] = (I0, I1, qc, qu)
# 사람 GT (학습에 쓴 87 레이어)
I0, I1, LAB = [], [], []
for _ in range(N):
    j = int(rng.integers(len(gt)))
    y0, x0 = int(rng.integers(0, S - W + 1)), int(rng.integers(0, S - W + 1))
    sl = (slice(y0, y0 + W), slice(x0, x0 + W))
    ann = gt.ann[j][sl]
    I0.append(gt.i0[j][sl].copy()); I1.append(gt.i1[j][sl].copy()); LAB.append(np.where(ann < 0, cfg.UNKNOWN, ann).astype(np.uint8))
I0, I1, LAB = np.stack(I0), np.stack(I1), np.stack(LAB)
qg = run_whole("GT_train_layers_gt_label", I0, I1, LAB, cfg.SAMPLE_CFG)
keep["GT"] = (I0, I1, qg, None)

# 창 전체 그리기의 그림 — 빌드마다 4 창: 진짜 v1 | 라벨 맵대로 그린 v1 | 라벨 없이 그린 v1 (같은 변환)
P = W
sheet = Image.new("RGB", (6 * P + 7 * 6, 4 * 4 * (P + 6) + 40), (24, 24, 24))
d = ImageDraw.Draw(sheet)
d.text((6, 3), "whole-window generation, raw values with per-window 1-99% stretch of the real photo", fill=(255, 255, 255))
d.text((6, 17), "real v0 | gen v0 (label) | gen v0 (no label) | real v1 | gen v1 (label) | gen v1 (no label)", fill=(190, 190, 190))
y = 36
for b in list(cfg.TRAIN_BUILDS) + ["GT"]:
    I0, I1, qc, qu = keep[b]
    for n in range(4):
        tiles = []
        for real, gc, gu in ((I0[n], qc[0][n], None if qu is None else qu[0][n]), (I1[n], qc[1][n], None if qu is None else qu[1][n])):
            lo, hi = np.percentile(real, [1, 99])
            tiles += [stretch(real, lo, hi), stretch(gc, lo, hi), np.zeros_like(real) if gu is None else stretch(gu, lo, hi)]
        for i, t in enumerate(tiles):
            sheet.paste(Image.fromarray(t), (6 + i * (P + 6), y))
        d.text((8, y + 2), b, fill=(255, 230, 120))
        y += P + 6
sheet.save(OUT / "whole_window_sheet.png")

# ── 3. 결함 넣기 (실제로 쓸 방식) — 클래스마다 ORNL 창에 24 개씩 ────────────────────
K = 24
fit = W - 2 * OBJECT_MARGIN
res["insert"] = {}
ex = {}


def contrast(img, obj, ref):
    a = img.astype(np.float32)
    return float(a[obj].mean() - a[ref].mean())


for c in cfg.GEN_CLASSES:
    name = cfg.CLASS_ABBRS[c]
    rows = gt.objects_fitting(c, fit)
    pick = rng.choice(rows, size=min(K, len(rows)), replace=False)
    # 진짜 객체 (학습 레이어) 의 대비
    real_c = {"v0": [], "v1": []}
    real_tiles = []
    for r in pick:
        o = gt.objects[int(r)]
        j = int(o[0])
        mask, _, _ = object_mask(gt, int(r))
        yb, xb = int(o[2]), int(o[4])
        full = np.zeros((S, S), bool)
        full[yb:yb + mask.shape[0], xb:xb + mask.shape[1]] = mask
        cy, cx = (int(v) for v in ndimage.center_of_mass(full))
        y0, x0 = int(np.clip(cy - W // 2, 0, S - W)), int(np.clip(cx - W // 2, 0, S - W))
        sl = (slice(y0, y0 + W), slice(x0, x0 + W))
        m = full[sl]
        reg = ndimage.binary_dilation(m, structure=SQ, iterations=cfg.RING_PX)
        ref = ndimage.binary_dilation(reg, structure=SQ, iterations=6) & ~reg
        if not ref.any():
            continue
        real_c["v0"].append(contrast(gt.i0[j][sl], m, ref)); real_c["v1"].append(contrast(gt.i1[j][sl], m, ref))
        if len(real_tiles) < 8:
            real_tiles.append((gt.i0[j][sl].copy(), gt.i1[j][sl].copy(), cy - y0, cx - x0))
    # ORNL 창에 넣는다
    I0, I1, LAB, REG, OBJ = [], [], [], [], []
    need_part = REQUIRED_BASE.get(c) == cfg.PRINTED
    tries = 0
    while len(I0) < len(pick) and tries < 4000:
        tries += 1
        mask, _, _ = object_mask(gt, int(pick[len(I0)]))
        b, row = canvas.pick_layer(rng)
        y0, x0 = int(rng.integers(e, S - e - W + 1)), int(rng.integers(e, S - e - W + 1))
        a0, a1, _, part = crop_ornl(canvas, b, row, y0, x0, W, False)
        if need_part and part.mean() < 0.05:
            continue
        base = cad_label(part)
        placed = place_mask(mask, base, c, rng, W)
        if placed is None:
            continue
        lab = base.copy(); lab[placed] = c
        I0.append(a0.copy()); I1.append(a1.copy()); LAB.append(lab); OBJ.append(placed)
        REG.append(ndimage.binary_dilation(placed, structure=SQ, iterations=cfg.RING_PX))
    I0, I1, LAB, REG, OBJ = (np.stack(v) for v in (I0, I1, LAB, REG, OBJ))
    q0, q1 = fill_windows(model, diff, I0, I1, REG, LAB, steps=cfg.SAMPLE_STEPS, guidance=cfg.SAMPLE_CFG, seed=11, batch=32, amp=True)
    gen_c, bg_c = {"v0": [], "v1": []}, {"v0": [], "v1": []}
    for n in range(len(I0)):
        ref = ndimage.binary_dilation(REG[n], structure=SQ, iterations=6) & ~REG[n]
        for ch, bgimg, gimg in (("v0", I0[n], q0[n]), ("v1", I1[n], q1[n])):
            gen_c[ch].append(contrast(gimg, OBJ[n], ref)); bg_c[ch].append(contrast(bgimg, OBJ[n], ref))
    res["insert"][name] = {ch: dict(real_objects=round(float(np.median(real_c[ch])), 2), generated=round(float(np.median(gen_c[ch])), 2),
                                    background_before=round(float(np.median(bg_c[ch])), 2),
                                    real_q25_q75=[round(float(v), 2) for v in np.percentile(real_c[ch], [25, 75])],
                                    gen_q25_q75=[round(float(v), 2) for v in np.percentile(gen_c[ch], [25, 75])],
                                    same_sign_frac=round(float(np.mean(np.sign(gen_c[ch]) == np.sign(np.median(real_c[ch])))), 2))
                           for ch in ("v0", "v1")}
    res["insert"][name]["n"] = int(len(I0))
    log.info(f"3. {name} (n={len(I0)}): " + "  ".join(
        f"{ch} 대비 진짜 {res['insert'][name][ch]['real_objects']:+.1f} / 그린 것 {res['insert'][name][ch]['generated']:+.1f} / 넣기 전 {res['insert'][name][ch]['background_before']:+.1f}"
        for ch in ("v0", "v1")))
    ex[c] = (real_tiles, [(I0[n], I1[n], q0[n], q1[n], *(int(v) for v in ndimage.center_of_mass(OBJ[n]))) for n in range(min(8, len(I0)))])

# 결함 넣기 그림 — 클래스마다 4 줄: 진짜 v0 · 진짜 v1 (학습 레이어의 객체), 그린 v0 · 그린 v1 (ORNL 창에 넣은 것). 한 줄에 8 개
Z = 96 * 2
sheet = Image.new("RGB", (8 * Z + 9 * 4 + 70, len(ex) * (4 * (Z + 4) + 10) + 20), (24, 24, 24))
d = ImageDraw.Draw(sheet)
d.text((6, 3), "x2 zoom, object at the centre.  rows per class: real v0, real v1 (objects of the training layers) | gen v0, gen v1 (inserted into ORNL windows)", fill=(255, 255, 255))
y = 20
for c, (rt, gtiles) in ex.items():
    for r, lab_ in enumerate(("real v0", "real v1", "gen v0", "gen v1")):
        d.text((4, y + 4), f"{cfg.CLASS_ABBRS[c]} {lab_}", fill=(255, 230, 120))
        for i in range(8):
            if r < 2:
                if i >= len(rt):
                    continue
                a0, a1, cy, cx = rt[i]
                img = a0 if r == 0 else a1
                lo, hi = np.percentile(img, [1, 99])
                tile = zoom(stretch(img, lo, hi), cy, cx, 48, 2)
            else:
                if i >= len(gtiles):
                    continue
                b0, b1, q0_, q1_, cy, cx = gtiles[i]
                bgimg, img = (b0, q0_) if r == 2 else (b1, q1_)
                lo, hi = np.percentile(bgimg, [1, 99])
                tile = zoom(stretch(img, lo, hi), cy, cx, 48, 2)
            sheet.paste(Image.fromarray(tile), (70 + i * (Z + 4), y))
        y += Z + 4
    y += 10
sheet.save(OUT / "insert_sheet.png")

(OUT / "check.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
log.info(f"끝 — {OUT}  ({time.time() - t0:.0f}초)")
