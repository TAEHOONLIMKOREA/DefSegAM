"""생성 세트 w600_v1 에서 결함 클래스마다 한 레이어를 골라 [라벨 맵 | 생성한 v0 | 생성한 v1] 로 그린다 (저장소 코드가 아니다 — 그림 옆에 둔 스크립트).

  DefSeg_AM/venv/bin/python make_samples.py            (CPU, 몇 초)

레이어는 사람이 골랐다 (PICKS): 그 클래스의 화소가 많은 레이어 가운데 다른 결함이 적고 사진에서 잘 보이는 것.
윗줄 = 레이어 전체 (1036², 원본 화소 그대로), 아랫줄 = 그 클래스가 가장 몰린 259² 창을 4배로 키운 것 (윗줄의 하늘색 상자).
사진의 밝기는 보기 좋게 늘렸다 (사진마다 1–99 백분위). 세트에 저장된 값은 원본 밝기다.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw, ImageFont

HERE = Path(__file__).resolve().parent
SET = HERE.parents[4] / "cache" / "genaug" / "sets" / "w600_v1"
FONT = "/usr/share/fonts/opentype/noto/NotoSansCJK-{}.ttc"

PICKS = [("RS", "0000126"), ("RH", "0000324"), ("SE", "0000155"), ("DB", "0000282")]
# 확대 창을 손으로 정한 것 (y0, x0). RS 는 화소가 가장 몰린 창이 RH 조각과 겹쳐서, 분말 위의 두 줄 (y 157 · 201) 로 옮겼다
ZOOM = {"RS": (60, 150)}
CLS_ID = {"RS": 4, "RH": 5, "SE": 6, "DB": 9}
NAMES = {"RS": "Recoater Streaking", "RH": "Recoater Hopping", "SE": "Super Elevation", "DB": "Debris"}
# genaug 의 표시 색 (config_genaug.CLASS_COLORS) — 같은 폴더의 다른 점검 그림과 같다
COL = {-1: (0, 0, 0), 0: (70, 70, 70), 1: (205, 205, 205), 4: (255, 220, 0), 5: (255, 140, 0), 6: (255, 0, 255), 9: (170, 110, 255)}
LEGEND = [(0, "Powder (정상 분말)"), (1, "Printed (파트)"), (4, "Recoater Streaking"), (5, "Recoater Hopping"),
          (6, "Super Elevation"), (9, "Debris"), (-1, "미레이블 (바깥 5% 의 진짜 사진 · 파트 경계)")]
COLS = ["라벨 맵 — 생성 모델에 준 입력이자 학습의 정답", "생성한 사진 — 용융 직후 (visible/0)", "생성한 사진 — 도포 직후 (visible/1)"]
S, Z, GAP, M = 1036, 259, 14, 24
BG, FG, BOX = (24, 24, 24), (235, 235, 235), (0, 230, 255)


def font(size, bold=False):
    return ImageFont.truetype(FONT.format("Bold" if bold else "Regular"), size, index=1)


def color(lab):
    out = np.zeros(lab.shape + (3,), np.uint8)
    for c, rgb in COL.items():
        out[lab == c] = rgb
    return out


def stretch(a):
    lo, hi = np.percentile(a, [1, 99])
    g = np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1.0) * 255.0, 0, 255).astype(np.uint8)
    return np.stack([g] * 3, -1)


def densest_window(mask, z):
    """그 클래스의 화소가 가장 많이 드는 z×z 창의 왼쪽 위 모서리."""
    ii = np.pad(mask.astype(np.int64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    s = ii[z:, z:] - ii[:-z, z:] - ii[z:, :-z] + ii[:-z, :-z]
    y, x = np.unravel_index(int(s.argmax()), s.shape)
    return int(y), int(x), int(s.max())


def load(i):
    lab = np.load(SET / "annotations" / f"{i}.npy")
    v0 = np.asarray(Image.open(SET / "visible" / "0" / f"{i}.png"))
    v1 = np.asarray(Image.open(SET / "visible" / "1" / f"{i}.png"))
    meta = json.loads((SET / "meta" / f"{i}.json").read_text())
    return lab, v0, v1, meta


def legend(d, x, y, f):
    for c, name in LEGEND:
        d.rectangle([x, y + 4, x + 26, y + 30], fill=COL[c], outline=(150, 150, 150))
        d.text((x + 34, y), name, fill=FG, font=f)
        x += 34 + int(d.textlength(name, font=f)) + 34
    return x


def one(cls, i):
    lab, v0, v1, meta = load(i)
    c = CLS_ID[cls]
    whole = [color(lab), stretch(v0), stretch(v1)]
    zy, zx, zpx = densest_window(lab == c, Z)
    if cls in ZOOM:
        zy, zx = ZOOM[cls]
        zpx = int((lab[zy:zy + Z, zx:zx + Z] == c).sum())
    zoom = [np.asarray(Image.fromarray(t[zy:zy + Z, zx:zx + Z]).resize((S, S), Image.NEAREST)) for t in whole]
    n_obj = sum(1 for o in meta["defects"] if o["cls"] == cls)
    others = ", ".join(f"{k} {meta['class_px'][v]:,}" for k, v in CLS_ID.items() if k != cls and meta["class_px"][v]) or "없음"

    W = 2 * M + 3 * S + 2 * GAP
    head, cap = 150, 44
    H = head + cap + S + cap + S + M
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((M, 14), f"{NAMES[cls]} ({cls})  —  생성 세트 w600_v1 의 레이어 {i}", fill=(255, 255, 255), font=font(40, True))
    d.text((M, 68), f"파트 배치: ORNL {meta['build']} 의 layer {meta['orig_layer']}    ·    {cls} {meta['class_px'][c]:,} 화소 ({n_obj} 개)"
                    f"    ·    이 레이어의 다른 결함: {others}", fill=FG, font=font(26))
    legend(d, M, 108, font(24))
    y1 = head + cap
    y2 = y1 + S + cap
    for k in range(3):
        x = M + k * (S + GAP)
        d.text((x, head + 4), COLS[k], fill=FG, font=font(26))
        im.paste(Image.fromarray(whole[k]), (x, y1))
        d.rectangle([x + zx - 2, y1 + zy - 2, x + zx + Z + 1, y1 + zy + Z + 1], outline=BOX, width=2)
        d.text((x, y1 + S + 4), f"위의 하늘색 상자를 4배로 키운 것 ({Z}×{Z} 화소)", fill=BOX, font=font(26))
        im.paste(Image.fromarray(zoom[k]), (x, y2))
    out = HERE / f"sample_{cls}_{i}.png"
    im.save(out)
    print(f"{out.name}: {cls} {meta['class_px'][c]:,} px, {n_obj} 개, 확대 창 (y {zy}, x {zx}) 에 {zpx:,} px, 다른 결함: {others}")
    return whole, meta


def overview(rows):
    P, lab_w, head, cap = 760, 250, 110, 44
    W = 2 * M + lab_w + 3 * P + 2 * GAP
    H = head + cap + len(rows) * (P + GAP) + M
    im = Image.new("RGB", (W, H), BG)
    d = ImageDraw.Draw(im)
    d.text((M, 12), "생성 세트 w600_v1 — 결함 클래스마다 한 레이어 (레이어 전체)", fill=(255, 255, 255), font=font(36, True))
    legend(d, M, 66, font(22))
    for k in range(3):
        d.text((M + lab_w + k * (P + GAP), head + 4), COLS[k], fill=FG, font=font(24))
    for r, (cls, i, whole, meta) in enumerate(rows):
        y = head + cap + r * (P + GAP)
        d.rectangle([M, y + 6, M + 26, y + 32], fill=COL[CLS_ID[cls]])
        d.text((M + 36, y), cls, fill=(255, 255, 255), font=font(30, True))
        d.text((M, y + 44), NAMES[cls].replace(" ", "\n"), fill=FG, font=font(24), spacing=4)
        d.text((M, y + 130), f"레이어 {i}\n{meta['build']} layer {meta['orig_layer']}\n{meta['class_px'][CLS_ID[cls]]:,} 화소", fill=(170, 170, 170), font=font(22), spacing=6)
        for k in range(3):
            t = Image.fromarray(whole[k]).resize((P, P), Image.NEAREST if k == 0 else Image.LANCZOS)
            im.paste(t, (M + lab_w + k * (P + GAP), y))
    im.save(HERE / "overview.png")
    print(f"overview.png: {W}×{H}")


rows = []
for cls, i in PICKS:
    whole, meta = one(cls, i)
    rows.append((cls, i, whole, meta))
overview(rows)
