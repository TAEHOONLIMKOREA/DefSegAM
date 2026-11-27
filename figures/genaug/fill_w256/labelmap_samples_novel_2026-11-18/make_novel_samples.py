"""데이터셋에 없는 모양으로 만든 라벨 맵 3장 → 사진 두 장씩. 파트도 결함도 전부 이 스크립트가 그린 도형이다 (사람 GT · ORNL 의 마스크를 하나도 읽지 않는다).

  novel_1_polygons : 삼각형 · 육각형 · 별 · 십자 · 톱니바퀴 · 화살표
  novel_2_letters  : 큰 글자
  novel_3_curves   : 나선 · 물결 띠 · 꽃 모양 · 불규칙한 덩어리 · 가지
결함: DB (불규칙한 덩어리 + 삼각형 · 십자 · 작은 고리), SP (작은 점의 무리), SE (파트 가장자리를 따라가는 띠 + 파트 안의 지그재그), RH (나란한 세로 물결 줄)
"""
import json, math, os, sys, time
from pathlib import Path
import matplotlib
import numpy as np
import torch
from PIL import Image, ImageDraw, ImageFont
from scipy import ndimage
from DefSeg_AM.common.utils.log import setup_logger
from DefSeg_AM.genaug import config_genaug as cfg
from DefSeg_AM.genaug.models.diffusion import model_input
from DefSeg_AM.genaug.sampling.fill import load_generator
from DefSeg_AM.genaug.sampling.preview import cad_label

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
log = setup_logger(0, "novel")
dev = torch.device("cuda:0")
W, T, S = cfg.WIN, cfg.T_STEPS, cfg.IMG_SIZE
SQ = np.ones((3, 3), bool)
model, diff, ck = load_generator(sys.argv[2], dev)
rng = np.random.default_rng(424242)
TS = ((torch.linspace(1, 0, 50) ** 2) * (T - 1)).round().long().unique_consecutive().tolist()
FONT = os.path.join(matplotlib.get_data_path(), "fonts/ttf/DejaVuSans-Bold.ttf")
DB, SE, SP, RH = 9, 6, 7, 5


@torch.no_grad()
def sample(x_clean, region, label, guidance=cfg.SAMPLE_CFG, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    rnd = lambda: torch.randn(x_clean.shape, device=dev, generator=g)
    label = torch.where(region[:, 0] > 0.5, label, torch.full_like(label, cfg.UNKNOWN))
    null = torch.full_like(label, cfg.UNKNOWN)
    tb = torch.full((1,), TS[0], device=dev, dtype=torch.long)
    x = region * rnd() + (1 - region) * diff.q_sample(x_clean, tb, rnd())
    for i, t in enumerate(TS):
        tb = torch.full((1,), t, device=dev, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            inp = torch.cat([model_input(x, x_clean, region, label), model_input(x, x_clean, region, null)])
            vc, vu = model(inp, torch.cat([tb, tb])).float().chunk(2)
        v = vu + guidance * (vc - vu)
        x0, _ = diff.x0_eps_from_v(x, tb, v)
        x0 = x0.clamp(-1, 1)
        if i + 1 < len(TS):
            eps = (x - diff._b(diff.sqrt_ac[tb]) * x0) / diff._b(diff.sqrt_1mac[tb])
            tp = torch.full((1,), TS[i + 1], device=dev, dtype=torch.long)
            x = diff._b(diff.sqrt_ac[tp]) * x0 + diff._b(diff.sqrt_1mac[tp]) * eps
            x = region * x + (1 - region) * diff.q_sample(x_clean, tp, rnd())
        else:
            x = x0
    return region * x + (1 - region) * x_clean


def draw_layer(lab_full, seed):
    out = torch.zeros((2, S, S), device=dev)
    done = torch.zeros((S, S), device=dev, dtype=torch.bool)
    lab_t = torch.from_numpy(lab_full.astype(np.int64)).to(dev)
    pos = sorted(set(list(range(0, S - W, 192)) + [S - W]))
    n = 0
    for y in pos:
        for x in pos:
            sl = (slice(y, y + W), slice(x, x + W))
            region = (~done[sl]).float()[None, None]
            if region.sum() == 0:
                continue
            out[:, sl[0], sl[1]] = sample(out[:, sl[0], sl[1]][None], region, lab_t[sl][None], seed=seed * 1000 + n)[0]
            done[sl] = True
            n += 1
    return ((out.clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()


# ── 도형 ─────────────────────────────────────────────────────────────────────
def pts_radial(cx, cy, rfun, n=240, rot=0.0):
    th = np.linspace(0, 2 * math.pi, n, endpoint=False)
    r = rfun(th)
    return [(cx + float(r[i]) * math.cos(th[i] + rot), cy + float(r[i]) * math.sin(th[i] + rot)) for i in range(n)]


def regular(cx, cy, r, n, rot):
    return [(cx + r * math.cos(rot + 2 * math.pi * k / n), cy + r * math.sin(rot + 2 * math.pi * k / n)) for k in range(n)]


def star(cx, cy, R, r, n, rot):
    return [(cx + (R if k % 2 == 0 else r) * math.cos(rot + math.pi * k / n), cy + (R if k % 2 == 0 else r) * math.sin(rot + math.pi * k / n)) for k in range(2 * n)]


def rotate(points, cx, cy, rot):
    c, s = math.cos(rot), math.sin(rot)
    return [(cx + c * x - s * y, cy + s * x + c * y) for x, y in points]


def plus(cx, cy, L, w, rot):
    p = [(-w, -L), (w, -L), (w, -w), (L, -w), (L, w), (w, w), (w, L), (-w, L), (-w, w), (-L, w), (-L, -w), (-w, -w)]
    return rotate(p, cx, cy, rot)


def arrow(cx, cy, s, rot):
    p = [(-s, -0.3 * s), (0.1 * s, -0.3 * s), (0.1 * s, -0.7 * s), (s, 0), (0.1 * s, 0.7 * s), (0.1 * s, 0.3 * s), (-s, 0.3 * s)]
    return rotate(p, cx, cy, rot)


def blob_pts(cx, cy, r0, rough, rot=0.0):
    a = [rng.uniform(0, rough / k) for k in range(2, 7)]
    ph = [rng.uniform(0, 2 * math.pi) for _ in range(2, 7)]
    return pts_radial(cx, cy, lambda th: r0 * (1 + sum(a[i] * np.cos((i + 2) * th + ph[i]) for i in range(5))), rot=rot)


def layout_polygons():
    im = Image.new("L", (S, S), 0); d = ImageDraw.Draw(im)
    kinds = ["tri", "hex", "star", "plus", "gear", "arrow", "hexhole", "star7", "tri"]
    k = 0
    for gy in range(5):
        for gx in range(5):
            cx, cy = 118 + 200 * gx + int(rng.integers(-22, 23)), 118 + 200 * gy + int(rng.integers(-22, 23))
            r, rot = int(rng.integers(38, 72)), float(rng.uniform(0, 2 * math.pi))
            kind = kinds[k % len(kinds)]; k += 1
            if kind == "tri":
                d.polygon(regular(cx, cy, r, 3, rot), fill=255)
            elif kind == "hex":
                d.polygon(regular(cx, cy, r, 6, rot), fill=255)
            elif kind == "hexhole":
                d.polygon(regular(cx, cy, r, 6, rot), fill=255); d.polygon(regular(cx, cy, r * 0.55, 6, rot), fill=0)
            elif kind == "star":
                d.polygon(star(cx, cy, r, r * 0.42, 5, rot), fill=255)
            elif kind == "star7":
                d.polygon(star(cx, cy, r, r * 0.6, 7, rot), fill=255)
            elif kind == "plus":
                d.polygon(plus(cx, cy, r, r * 0.3, rot), fill=255)
            elif kind == "gear":
                d.polygon(pts_radial(cx, cy, lambda th: r * (0.86 + 0.14 * np.sign(np.sin(10 * th))), n=400, rot=rot), fill=255)
                d.ellipse([cx - r * 0.3, cy - r * 0.3, cx + r * 0.3, cy + r * 0.3], fill=0)
            elif kind == "arrow":
                d.polygon(arrow(cx, cy, r, rot), fill=255)
    return np.array(im) > 127


def layout_letters():
    im = Image.new("L", (S, S), 0); d = ImageDraw.Draw(im)
    for text, y in (("DEFSEG", 60), ("AM 2026", 390), ("LPBF", 720)):
        size = 200
        f = ImageFont.truetype(FONT, size)
        l, t, r, b = d.textbbox((0, 0), text, font=f)
        size = int(size * 880 / (r - l)); size = min(size, 290)
        f = ImageFont.truetype(FONT, size)
        l, t, r, b = d.textbbox((0, 0), text, font=f)
        d.text(((S - (r - l)) // 2 - l, y - t), text, font=f, fill=255)
    return np.array(im) > 127


def layout_curves():
    im = Image.new("L", (S, S), 0); d = ImageDraw.Draw(im)
    th = np.linspace(0.6, 5.2 * math.pi, 700)                                   # 나선 띠
    d.line([(250 + (14 + 8.2 * t) * math.cos(t), 250 + (14 + 8.2 * t) * math.sin(t)) for t in th], fill=255, width=12, joint="curve")
    for y0, amp, lam in ((120, 26, 150), (215, 34, 210), (320, 22, 110)):        # 물결 띠
        xs = np.arange(480, 990, 3)
        d.line([(float(x), y0 + amp * math.sin(2 * math.pi * x / lam)) for x in xs], fill=255, width=14, joint="curve")
    for cx, cy, r0 in ((770, 560, 78), (240, 640, 66), (560, 880, 58)):          # 불규칙한 덩어리
        d.polygon(blob_pts(cx, cy, r0, 0.55), fill=255)
    d.polygon(pts_radial(860, 850, lambda t: 78 * (1 + 0.36 * np.cos(7 * t))), fill=255)   # 꽃 모양 (가운데 구멍)
    d.ellipse([838, 828, 882, 872], fill=0)

    def branch(x, y, ang, length, w, depth):                                     # 가지
        x2, y2 = x + length * math.cos(ang), y + length * math.sin(ang)
        d.line([(x, y), (x2, y2)], fill=255, width=w)
        if depth:
            for da in (-0.55, 0.5):
                branch(x2, y2, ang + da + float(rng.uniform(-0.12, 0.12)), length * 0.68, max(w - 2, 5), depth - 1)
    branch(470, 760, -math.pi / 2, 120, 13, 3)
    return np.array(im) > 127


# ── 결함 (전부 여기서 그린 모양) ─────────────────────────────────────────────
def small_mask(draw_fn, size=64):
    im = Image.new("L", (size, size), 0)
    draw_fn(ImageDraw.Draw(im), size / 2)
    return np.array(im) > 127


def db_shapes():
    out = []
    for _ in range(4):
        r0 = float(rng.uniform(5, 11)); out.append(("DB blob", small_mask(lambda d, c: d.polygon(blob_pts(c, c, r0, 0.7), fill=255))))
    rot = float(rng.uniform(0, 6.28))
    out.append(("DB triangle", small_mask(lambda d, c: d.polygon(regular(c, c, 11, 3, rot), fill=255))))
    out.append(("DB plus", small_mask(lambda d, c: d.polygon(plus(c, c, 11, 2.5, rot), fill=255))))
    out.append(("DB ring", small_mask(lambda d, c: (d.ellipse([c - 10, c - 10, c + 10, c + 10], fill=255), d.ellipse([c - 6, c - 6, c + 6, c + 6], fill=0)))))
    return out


def sp_shapes():
    out = []
    for _ in range(5):
        dots = [(float(rng.normal(32, 9)), float(rng.normal(32, 9)), float(rng.uniform(1.5, 3.2))) for _ in range(int(rng.integers(5, 11)))]
        out.append(("SP dots", small_mask(lambda d, c: [d.ellipse([x - r, y - r, x + r, y + r], fill=255) for x, y, r in dots])))
    return out


def put(lab, mask, cls, placed, name, tries=600):
    ys, xs = np.nonzero(mask)
    m = mask[ys.min():ys.max() + 1, xs.min():xs.max() + 1]
    h, w = m.shape
    grown = ndimage.binary_dilation(np.pad(m, 3), structure=SQ, iterations=3)
    for _ in range(tries):
        y, x = int(rng.integers(30, S - 30 - h)), int(rng.integers(30, S - 30 - w))
        if (lab[y - 3:y + h + 3, x - 3:x + w + 3][grown] == cfg.POWDER).all():
            lab[y:y + h, x:x + w][m] = cls
            placed.append((name, y + h // 2, x + w // 2))
            return


def put_se(lab, part, placed, n_edge=4, n_zig=1):
    band = ndimage.binary_erosion(part, structure=SQ, iterations=4) & ~ndimage.binary_erosion(part, structure=SQ, iterations=9)
    ys, xs = np.nonzero(band)
    yy, xx = np.mgrid[:S, :S]
    for _ in range(n_edge):                                  # 가장자리를 따라가는 띠
        for _ in range(50):
            i = int(rng.integers(len(ys))); py, px = int(ys[i]), int(xs[i])
            R = int(rng.integers(22, 40))
            m = band & ((yy - py) ** 2 + (xx - px) ** 2 <= R * R)
            cc, _ = ndimage.label(m, structure=SQ)
            m = cc == cc[py, px]
            if m.sum() >= 60 and (lab[ndimage.binary_dilation(m, structure=SQ, iterations=6)] <= cfg.UNKNOWN).all() and not (lab[ndimage.binary_dilation(m, structure=SQ, iterations=6)] >= 2).any() if False else (m.sum() >= 60 and not ((lab[ndimage.binary_dilation(m, structure=SQ, iterations=6)] >= 2) & (lab[ndimage.binary_dilation(m, structure=SQ, iterations=6)] != cfg.UNKNOWN)).any()):
                lab[m] = SE
                placed.append(("SE edge", py, px))
                break
    deep = ndimage.distance_transform_edt(part) >= 22
    ys, xs = np.nonzero(deep)
    for _ in range(n_zig if len(ys) else 0):                 # 파트 안쪽의 지그재그 — 진짜 SE 에는 없는 모양
        i = int(rng.integers(len(ys))); py, px = int(ys[i]), int(xs[i])
        im = Image.new("L", (S, S), 0)
        ImageDraw.Draw(im).line([(px - 18, py - 12), (px - 9, py + 12), (px, py - 12), (px + 9, py + 12), (px + 18, py - 12)], fill=255, width=4)
        m = (np.array(im) > 127) & ndimage.binary_erosion(part, structure=SQ, iterations=4) & (lab == cfg.PRINTED)
        if m.sum() >= 40:
            lab[m] = SE
            placed.append(("SE zigzag", py, px))


def put_rh(lab, placed, n=2):
    for _ in range(n):
        for _ in range(200):
            zh, zw = int(rng.integers(150, 230)), int(rng.integers(120, 190))
            y0, x0 = int(rng.integers(30, S - 30 - zh)), int(rng.integers(30, S - 30 - zw))
            z = lab[y0:y0 + zh, x0:x0 + zw]
            if (z == cfg.POWDER).mean() >= 0.75 and not ((z >= 2) & (z != cfg.UNKNOWN)).any():
                im = Image.new("L", (zw, zh), 0); d = ImageDraw.Draw(im)
                x = 4.0
                while x < zw - 4:
                    a, b = sorted(rng.uniform(0, zh, 2))
                    if b - a > 40:
                        lam, ph, amp = rng.uniform(40, 90), rng.uniform(0, 6.28), rng.uniform(0.8, 2.2)
                        d.line([(x + amp * math.sin(2 * math.pi * yy_ / lam + ph), float(yy_)) for yy_ in np.arange(a, b, 2.0)], fill=255, width=2)
                    x += float(rng.uniform(6, 12))
                m = (np.array(im) > 127) & (z < 2)
                z[m] = RH
                placed.append(("RH wavy lines", y0 + zh // 2, x0 + zw // 2))
                break


def color(lab):
    out = np.zeros(lab.shape + (3,), np.uint8)
    out[lab == cfg.POWDER] = (70, 70, 70); out[lab == cfg.PRINTED] = (205, 205, 205); out[lab == cfg.UNKNOWN] = (0, 0, 0)
    for c in range(2, cfg.N_CLASSES):
        m = lab == c
        if m.any():
            out[m] = cfg.CLASS_COLORS[c]
    return out


def stretch(a, lo, hi):
    return np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1.0) * 255.0, 0, 255).astype(np.uint8)


def save(name, lab, G, placed, note):
    d = OUT / name; d.mkdir(exist_ok=True)
    ann = lab.astype(np.int8); ann[lab == cfg.UNKNOWN] = -1
    np.save(d / "label.npy", ann)
    Image.fromarray(G[0]).save(d / "visible0.png"); Image.fromarray(G[1]).save(d / "visible1.png")
    col = color(lab)
    st = [stretch(G[c], *np.percentile(G[c], [1, 99])) for c in (0, 1)]
    gap = np.full((S, 10, 3), 24, np.uint8)
    Image.fromarray(np.concatenate([col, gap, np.stack([st[0]] * 3, -1), gap, np.stack([st[1]] * 3, -1)], 1)).save(OUT / f"{name}_sheet.png")
    Z, half = 2, 64
    zs = Image.new("RGB", (len(placed) * (2 * half * Z + 4), 3 * (2 * half * Z + 4) + 16), (24, 24, 24))
    dr = ImageDraw.Draw(zs)
    for i, (cls, cy, cx) in enumerate(placed):
        y0, x0 = int(np.clip(cy - half, 0, S - 2 * half)), int(np.clip(cx - half, 0, S - 2 * half))
        dr.text((i * (2 * half * Z + 4) + 3, 2), cls, fill=(255, 230, 120))
        for r, im in enumerate((col, np.stack([st[0]] * 3, -1), np.stack([st[1]] * 3, -1))):
            zs.paste(Image.fromarray(im[y0:y0 + 2 * half, x0:x0 + 2 * half].repeat(Z, 0).repeat(Z, 1)), (i * (2 * half * Z + 4), 16 + r * (2 * half * Z + 4)))
    zs.save(OUT / f"{name}_zoom.png")
    info = dict(name=name, note=note, defects=placed, mean=[round(float(G[0].mean()), 1), round(float(G[1].mean()), 1)],
                class_px={cfg.CLASS_ABBRS[c]: int((lab == c).sum()) for c in range(cfg.N_CLASSES)}, unlabeled_px=int((lab == cfg.UNKNOWN).sum()),
                sampler="50 steps spaced toward low noise, guidance 2.0, 36 windows stitched, no real pixels", ckpt=str(sys.argv[2]))
    (d / "info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    log.info(f"{name}: 결함 {len(placed)} 개 {sorted(set(p[0] for p in placed))}, Printed {info['class_px']['Prt']} px, 평균 밝기 v0 {info['mean'][0]} v1 {info['mean'][1]}")


t0 = time.time()
for k, (name, fn, note) in enumerate((("novel_1_polygons", layout_polygons, "파트 = 삼각형 · 육각형 · 별 · 십자 · 톱니바퀴 · 화살표"),
                                      ("novel_2_letters", layout_letters, "파트 = 큰 글자 (DEFSEG / AM 2026 / LPBF)"),
                                      ("novel_3_curves", layout_curves, "파트 = 나선 · 물결 띠 · 불규칙한 덩어리 · 꽃 모양 · 가지")), 1):
    part = fn()
    lab = cad_label(part)
    placed = []
    put_rh(lab, placed)
    put_se(lab, part, placed)
    for nm, m in db_shapes():
        put(lab, m, DB, placed, nm)
    for nm, m in sp_shapes():
        put(lab, m, SP, placed, nm)
    G = draw_layer(lab, seed=40 + k)
    save(name, lab, G, placed, note + " — 파트와 결함의 모양을 전부 스크립트가 그렸다 (데이터셋의 마스크를 읽지 않음)")
    log.info(f"  ({time.time() - t0:.0f}초)")
log.info(f"끝 — {OUT}")
