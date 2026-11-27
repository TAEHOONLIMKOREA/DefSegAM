"""라벨 맵만 넣어 사진 두 장을 만든 표본 — 진짜 화소는 하나도 쓰지 않는다 (1036² 레이어, 창 36 개를 이어 그림).

저장: <out>/sample_<k>_<이름>/{label.npy (int8, 0–9 · 미레이블 -1), visible0.png, visible1.png (그린 값 그대로)}
      <out>/sample_<k>_<이름>_sheet.png  (라벨 맵 | visible/0 | visible/1, 보기 좋게 밝기를 늘린 것)
      <out>/sample_<k>_<이름>_zoom.png   (결함마다 확대: 라벨 맵 · visible/0 · visible/1)
"""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage
from DefSeg_AM.common.utils.log import setup_logger
from DefSeg_AM.genaug import config_genaug as cfg
from DefSeg_AM.genaug.data.canvas_pool import CanvasPool
from DefSeg_AM.genaug.data.gt_pool import GTPool, O_SAMPLE, O_SX, O_SY
from DefSeg_AM.genaug.models.diffusion import model_input
from DefSeg_AM.genaug.sampling.fill import load_generator
from DefSeg_AM.genaug.sampling.preview import REQUIRED_BASE, cad_label, object_mask

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
log = setup_logger(0, "samples")
dev = torch.device("cuda:0")
W, T = cfg.WIN, cfg.T_STEPS
SQ = np.ones((3, 3), bool)
model, diff, ck = load_generator(sys.argv[2], dev)
gt = GTPool("both", log=None)
canvas = CanvasPool(("B3", "B4", "B5"), log=None)
S = canvas.img_size
rng = np.random.default_rng(20261118)
TS = ((torch.linspace(1, 0, 50) ** 2) * (T - 1)).round().long().unique_consecutive().tolist()   # 낮은 잡음 쪽에 몬 50 단계


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
    """라벨 맵 (S, S) → (visible/0, visible/1) uint8. 진짜 화소 없음 — 첫 창은 라벨 맵만 보고, 그 뒤는 이미 그린 이웃을 보고 잇는다."""
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


def place(lab, mask, c, need, tries=400):
    h, w = mask.shape
    for _ in range(tries):
        y, x = int(rng.integers(30, S - 30 - h)), int(rng.integers(30, S - 30 - w))
        cur = lab[y:y + h, x:x + w][mask]
        if (need is None and (cur < 2).mean() >= 0.99) or (need is not None and (cur == need).mean() >= 0.9):
            lab[y:y + h, x:x + w][mask & (lab[y:y + h, x:x + w] < 2)] = c
            cy, cx = ndimage.center_of_mass(mask)
            return (cfg.CLASS_ABBRS[c], int(y + cy), int(x + cx))
    return None


def put_defects(base):
    """파트 배치 위에 결함을 얹는다: DB · SP (분말 위) 각 4, SE (파트 위) 4, RH 는 여러 줄을 한 묶음으로 2 곳."""
    lab = base.copy()
    placed = []
    for c in (9, 9, 9, 9, 7, 7, 7, 7, 6, 6, 6, 6):
        mask, _, _ = object_mask(gt, int(rng.choice(gt.objects_fitting(c, 120))))
        p = place(lab, mask, c, REQUIRED_BASE.get(c))
        if p:
            placed.append(p)
    for _ in range(2):                                   # RH — 진짜는 나란한 여러 줄이라, 사람 GT 의 180² 조각에 든 줄을 통째로 옮긴다
        o = gt.objects[int(rng.choice(gt.objects_fitting(5, 120)))]
        j, cy, cx = int(o[O_SAMPLE]), int(o[O_SY]), int(o[O_SX])
        y0, x0 = int(np.clip(cy - 90, 0, gt.img_size - 180)), int(np.clip(cx - 90, 0, gt.img_size - 180))
        mask = gt.ann[j][y0:y0 + 180, x0:x0 + 180] == 5
        if mask.sum() < 60:
            continue
        p = place(lab, mask, 5, None)
        if p:
            placed.append(("RH", p[1], p[2]))
    return lab, placed


def handmade_layout():
    """본 적 없는 파트 배치 — 원 · 사각형 · 고리 · 긴 막대를 아무 데나."""
    part = np.zeros((S, S), bool)
    yy, xx = np.mgrid[:S, :S]
    for _ in range(22):
        cy, cx = int(rng.integers(80, S - 80)), int(rng.integers(80, S - 80))
        kind = rng.choice(["disc", "rect", "ring", "bar"])
        if kind == "disc":
            r = int(rng.integers(14, 46)); m = (yy - cy) ** 2 + (xx - cx) ** 2 <= r * r
        elif kind == "rect":
            a, b = int(rng.integers(14, 60)), int(rng.integers(14, 60)); m = (abs(yy - cy) <= a) & (abs(xx - cx) <= b)
        elif kind == "ring":
            r, t = int(rng.integers(24, 50)), int(rng.integers(3, 6)); d = np.sqrt((yy - cy) ** 2 + (xx - cx) ** 2); m = (d <= r) & (d >= r - t)
        else:
            a, b = int(rng.integers(60, 130)), int(rng.integers(6, 14))
            m = (abs(yy - cy) <= a) & (abs(xx - cx) <= b) if rng.random() < 0.5 else (abs(yy - cy) <= b) & (abs(xx - cx) <= a)
        if not (ndimage.binary_dilation(m, structure=SQ, iterations=10) & part).any():
            part |= m
    return part


def color(lab):
    out = np.zeros(lab.shape + (3,), np.uint8)
    out[lab == cfg.POWDER] = (70, 70, 70); out[lab == cfg.PRINTED] = (205, 205, 205); out[lab == cfg.UNKNOWN] = (0, 0, 0)
    for c in range(2, cfg.N_CLASSES):
        m = lab == c
        if m.any():
            out[ndimage.binary_dilation(m, structure=SQ, iterations=1)] = cfg.CLASS_COLORS[c]
    return out


def stretch(a, lo, hi):
    return np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1.0) * 255.0, 0, 255).astype(np.uint8)


def save(k, name, lab, G, placed, note):
    d = OUT / f"sample_{k}_{name}"; d.mkdir(exist_ok=True)
    ann = lab.astype(np.int8); ann[lab == cfg.UNKNOWN] = -1
    np.save(d / "label.npy", ann)
    Image.fromarray(G[0]).save(d / "visible0.png"); Image.fromarray(G[1]).save(d / "visible1.png")
    col = color(lab)
    st = [stretch(G[c], *np.percentile(G[c], [1, 99])) for c in (0, 1)]
    gap = np.full((S, 10, 3), 24, np.uint8)
    Image.fromarray(np.concatenate([col, gap, np.stack([st[0]] * 3, -1), gap, np.stack([st[1]] * 3, -1)], 1)).save(OUT / f"sample_{k}_{name}_sheet.png")
    Z, half = 2, 64
    zs = Image.new("RGB", (len(placed) * (2 * half * Z + 4), 3 * (2 * half * Z + 4) + 16), (24, 24, 24))
    dr = ImageDraw.Draw(zs)
    for i, (cls, cy, cx) in enumerate(placed):
        y0, x0 = int(np.clip(cy - half, 0, S - 2 * half)), int(np.clip(cx - half, 0, S - 2 * half))
        dr.text((i * (2 * half * Z + 4) + 3, 2), cls, fill=(255, 230, 120))
        for r, im in enumerate((col, np.stack([st[0]] * 3, -1), np.stack([st[1]] * 3, -1))):
            t = im[y0:y0 + 2 * half, x0:x0 + 2 * half].repeat(Z, 0).repeat(Z, 1)
            zs.paste(Image.fromarray(t), (i * (2 * half * Z + 4), 16 + r * (2 * half * Z + 4)))
    zs.save(OUT / f"sample_{k}_{name}_zoom.png")
    info = dict(name=name, note=note, defects=placed, mean=[round(float(G[0].mean()), 1), round(float(G[1].mean()), 1)],
                class_px={cfg.CLASS_ABBRS[c]: int((lab == c).sum()) for c in range(cfg.N_CLASSES)}, unlabeled_px=int((lab == cfg.UNKNOWN).sum()))
    (d / "info.json").write_text(json.dumps(info, indent=1, ensure_ascii=False), encoding="utf-8")
    log.info(f"sample_{k}_{name}: 결함 {len(placed)} 개, 평균 밝기 v0 {info['mean'][0]} v1 {info['mean'][1]}")


t0 = time.time()
jobs = []
for b in ("B3", "B4", "B5"):
    row = int(rng.integers(canvas.n_layers[b]))
    _, _, part = canvas.crop(b, row, 0, 0, S, S)
    jobs.append((f"{b}_layout", np.asarray(part).astype(bool), f"파트 배치 = {b} 의 L{int(canvas.orig_layers[b][row])} CAD 마스크"))
jobs.append(("handmade_layout", handmade_layout(), "파트 배치 = 손으로 만든 도형 (학습 때 본 적 없는 배치)"))
first = None
for k, (name, part, note) in enumerate(jobs, 1):
    lab, placed = put_defects(cad_label(part))
    G = draw_layer(lab, seed=k)
    save(k, name, lab, G, placed, note)
    if first is None:
        first = (name, lab, placed, note)
    log.info(f"  ({time.time() - t0:.0f}초)")
name, lab, placed, note = first                      # 같은 라벨 맵을 다른 seed 로 한 번 더
save(5, name + "_seed2", lab, draw_layer(lab, seed=77), placed, note + " — sample_1 과 같은 라벨 맵, 다른 seed")
log.info(f"끝 ({time.time() - t0:.0f}초) — {OUT}")
