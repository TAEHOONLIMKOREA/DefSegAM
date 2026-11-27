"""v7 계획을 쓰기 전의 확인 (scratchpad 전용, 저장소 코드가 아니다).
  1) 생성 모델이 RS (가로 줄) · IS (미도포 영역) 를 라벨 맵대로 그리는가 — 지금까지 시험하지 않은 클래스
  2) 생성한 레이어를 v6 의 P0 모델 (cv/inlayer/last.pt) 이 어떻게 읽는가 — 넣은 클래스로 알아보는 비율, 정상으로 그리라고 한 화소를 결함으로 내는 비율
레이어: B3 · B4 · B5 의 CAD 배치 2장씩. 가장자리 5% 만 진짜, 안쪽 932² 는 라벨 맵만으로 그린다.
"""
import json, math, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy import ndimage
from DefSeg_AM.common.utils.log import setup_logger
from DefSeg_AM.genaug import config_genaug as cfg
from DefSeg_AM.genaug.data.canvas_pool import CanvasPool
from DefSeg_AM.genaug.data.gt_pool import GTPool, O_CLS, O_SAMPLE, O_SX, O_SY, O_X0, O_X1, O_Y0, O_Y1
from DefSeg_AM.genaug.models.diffusion import model_input
from DefSeg_AM.genaug.sampling.fill import load_generator
from DefSeg_AM.genaug.sampling.preview import REQUIRED_BASE, cad_label, object_mask
from DefSeg_AM.v6.inference.evaluate import load_model, predict_pair

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
log = setup_logger(0, "probe")
dev = torch.device("cuda:0")
W, T = cfg.WIN, cfg.T_STEPS
SQ = np.ones((3, 3), bool)
gen, diff, ck = load_generator(sys.argv[2], dev)
seg, _ = load_model(Path(sys.argv[3]), dev)
gt = GTPool("both", log=None)
canvas = CanvasPool(("B3", "B4", "B5"), log=None)
S = canvas.img_size
E = int(round(cfg.EDGE_FRAC * S))
rng = np.random.default_rng(8008)
TS = ((torch.linspace(1, 0, 50) ** 2) * (T - 1)).round().long().unique_consecutive().tolist()
IS_, PS, RS, RH, SE, SP, OM, DB = 2, 3, 4, 5, 6, 7, 8, 9
AB = cfg.CLASS_ABBRS


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
            vc, vu = gen(inp, torch.cat([tb, tb])).float().chunk(2)
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


def draw_framed(lab_full, r0, r1, seed):
    out = torch.zeros((2, S, S), device=dev)
    done = torch.zeros((S, S), device=dev, dtype=torch.bool)
    real_t = torch.from_numpy(np.stack([r0, r1]).astype(np.float32) / 127.5 - 1).to(dev)
    fr = torch.ones((S, S), device=dev, dtype=torch.bool); fr[E:S - E, E:S - E] = False
    out[:, fr] = real_t[:, fr]; done |= fr
    lab_t = torch.from_numpy(lab_full.astype(np.int64)).to(dev)
    pos = sorted(set(list(range(E - 16, S - E + 16 - W, 192)) + [S - E + 16 - W]))
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


LO, HI = E + 12, S - E - 12            # 결함을 놓는 범위 (그리는 안쪽)


def place(lab, mask, c, need, tries=400):
    h, w = mask.shape
    if h >= HI - LO or w >= HI - LO:
        return None
    for _ in range(tries):
        y, x = int(rng.integers(LO, HI - h)), int(rng.integers(LO, HI - w))
        cur = lab[y:y + h, x:x + w][mask]
        if (need is None and (cur < 2).mean() >= 0.99) or (need is not None and (cur == need).mean() >= 0.9):
            lab[y:y + h, x:x + w][mask & (lab[y:y + h, x:x + w] < 2)] = c
            cy, cx = ndimage.center_of_mass(mask)
            return (AB[c], int(y + cy), int(x + cx))
    return None


def big_piece(c, size, min_px, min_side):
    """큰 객체 (bbox 한 변이 min_side 이상) 하나를 골라, 그 씨앗 둘레 size² 조각 안의 그 클래스 마스크를 돌려준다 (진짜 모양)."""
    o = gt.objects
    side = np.maximum(o[:, O_Y1] - o[:, O_Y0], o[:, O_X1] - o[:, O_X0])
    rows = np.nonzero((o[:, O_CLS] == c) & (side >= min_side))[0]
    for _ in range(50):
        r = o[int(rng.choice(rows))]
        j, cy, cx = int(r[O_SAMPLE]), int(r[O_SY]), int(r[O_SX])
        y0, x0 = int(np.clip(cy - size // 2, 0, gt.img_size - size)), int(np.clip(cx - size // 2, 0, gt.img_size - size))
        m = gt.ann[j][y0:y0 + size, x0:x0 + size] == c
        if m.sum() >= min_px:
            return m
    return None


def place_area(lab, mask, c, tries=300):
    h, w = mask.shape
    for _ in range(tries):
        y, x = int(rng.integers(LO, HI - h)), int(rng.integers(LO, HI - w))
        cur = lab[y:y + h, x:x + w]
        if (cur[mask] == cfg.POWDER).mean() >= 0.85 and not ((cur[mask] >= 2) & (cur[mask] != cfg.UNKNOWN)).any():
            cur[mask & (cur == cfg.POWDER)] = c
            cy, cx = ndimage.center_of_mass(mask)
            return (AB[c], int(y + cy), int(x + cx))
    return None


def build_label(part):
    lab = cad_label(part)
    placed = []
    m = big_piece(PS, 320, 8000, 200)
    if m is not None:
        p = place_area(lab, m, PS)
        if p: placed.append(p)
    for _ in range(2):
        m = big_piece(SP, 220, 3000, 150)
        if m is not None:
            p = place_area(lab, m, SP)
            if p: placed.append(p)
    for _ in range(2):
        y = int(rng.integers(LO + 20, HI - 20))
        seg_ = lab[y:y + 2, LO:HI]
        seg_[seg_ < 2] = RS
        placed.append(("RS", y, (LO + HI) // 2))
    for _ in range(6):
        mask, _, _ = object_mask(gt, int(rng.choice(gt.objects_fitting(OM, 120))))
        p = place(lab, mask, OM, cfg.PRINTED)
        if p: placed.append(p)
    return lab, placed


def color(lab):
    out = np.zeros(lab.shape + (3,), np.uint8)
    out[lab == cfg.POWDER] = (70, 70, 70); out[lab == cfg.PRINTED] = (205, 205, 205)
    for c in range(2, cfg.N_CLASSES):
        out[lab == c] = cfg.CLASS_COLORS[c]
    return out


def stretch(a, lo, hi):
    return np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1.0) * 255.0, 0, 255).astype(np.uint8)


t0 = time.time()
inner = np.zeros((S, S), bool); inner[E:S - E, E:S - E] = True
conf_gen = np.zeros((11, 10), np.int64)        # 의도한 라벨 (0–9, 10 = 모름) × v6 예측 — 생성한 레이어의 안쪽
conf_real = np.zeros((2, 10), np.int64)        # CAD 로 본 (분말, 파트) × v6 예측 — 같은 ORNL 레이어의 진짜 사진 (안쪽, 경계 ±3 px 제외)
per_build = {}
zoom_rows = []
for b in ("B3", "B4", "B5"):
    for k in range(2):
        row = int(rng.integers(canvas.n_layers[b]))
        r0, r1, part = (np.asarray(a) for a in canvas.crop(b, row, 0, 0, S, S))
        part = part.astype(bool)
        lab, placed = build_label(part)
        G = draw_framed(lab, r0, r1, seed=len(zoom_rows) + 1)
        pg = predict_pair(seg, G[0], G[1], S, dev, resolutions=("resized",))["resized"].cpu().numpy()
        pr = predict_pair(seg, r0, r1, S, dev, resolutions=("resized",))["resized"].cpu().numpy()
        cg = np.zeros((11, 10), np.int64)
        np.add.at(cg, (lab[inner].astype(np.int64), pg[inner].astype(np.int64)), 1)
        conf_gen += cg
        base = cad_label(part)
        cr = np.zeros((2, 10), np.int64)
        for i, c in enumerate((cfg.POWDER, cfg.PRINTED)):
            m = inner & (base == c)
            cr[i] = np.bincount(pr[m].astype(np.int64), minlength=10)
        conf_real += cr
        d = per_build.setdefault(b, dict(gen=np.zeros((11, 10), np.int64), real=np.zeros((2, 10), np.int64)))
        d["gen"] += cg; d["real"] += cr
        name = f"{b}_L{int(canvas.orig_layers[b][row])}"
        col, colp, colr = color(lab), color(pg.astype(np.uint8)), color(pr.astype(np.uint8))
        st = [np.stack([stretch(G[c], *np.percentile((r0, r1)[c], [1, 99]))] * 3, -1) for c in (0, 1)]
        sr = [np.stack([stretch((r0, r1)[c], *np.percentile((r0, r1)[c], [1, 99]))] * 3, -1) for c in (0, 1)]
        gap = np.full((S, 8, 3), 24, np.uint8)
        Image.fromarray(np.concatenate([col, gap, st[0], gap, st[1], gap, colp], 1)).save(OUT / f"{name}_label_gen0_gen1_v6pred.png")
        Image.fromarray(np.concatenate([sr[0], gap, sr[1], gap, colr], 1)).save(OUT / f"{name}_real0_real1_v6pred.png")
        for cls, cy, cx in placed:
            if sum(1 for z in zoom_rows if z[0] == cls) < 6:
                h = 128 if cls in ("PS", "SP") else 64
                y0, x0 = int(np.clip(cy - h, 0, S - 2 * h)), int(np.clip(cx - h, 0, S - 2 * h))
                tiles = [im[y0:y0 + 2 * h, x0:x0 + 2 * h] for im in (col, st[0], st[1], colp)]
                zoom_rows.append((cls, name, [np.asarray(Image.fromarray(t).resize((256, 256), Image.NEAREST)) for t in tiles]))
        log.info(f"{name}: 결함 {len(placed)} 개  ({time.time() - t0:.0f}초)")

# 확대 시트 — 클래스별로 [라벨 맵 | 그린 v0 | 그린 v1 | v6 P0 모델의 예측]
order = ["PS", "SP", "OM", "RS"]
for cls in order:
    rows = [z for z in zoom_rows if z[0] == cls][:8]
    if not rows:
        continue
    sheet = Image.new("RGB", (4 * 260, len(rows) * 260 + 16), (24, 24, 24))
    ImageDraw.Draw(sheet).text((4, 2), f"{cls}: label map | generated v0 | generated v1 | prediction of the v6 P0 model", fill=(255, 255, 255))
    for i, (_, name, tiles) in enumerate(rows):
        for j, t in enumerate(tiles):
            sheet.paste(Image.fromarray(t), (j * 260, 16 + i * 260))
    sheet.save(OUT / f"zoom_{cls}.png")


def table(conf, names):
    out = {}
    for i, n in enumerate(names):
        tot = int(conf[i].sum())
        if tot:
            out[n] = dict(px=tot, **{AB[c]: round(100.0 * conf[i, c] / tot, 2) for c in range(10) if conf[i, c] / tot >= 0.0005})
    return out


names11 = AB + ["unknown"]
res = dict(generated=table(conf_gen, names11), real=table(conf_real, ["CAD powder", "CAD part"]),
           per_build={b: dict(generated=table(d["gen"], names11), real=table(d["real"], ["CAD powder", "CAD part"])) for b, d in per_build.items()})
(OUT / "probe.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
log.info("=== 생성한 레이어 (6장, 안쪽): 의도한 라벨 → v6 P0 모델의 예측 (%)")
for n, r in res["generated"].items():
    log.info(f"  {n:8s} {r}")
log.info("=== 같은 ORNL 레이어의 진짜 사진: CAD 구역 → v6 P0 모델의 예측 (%)")
for n, r in res["real"].items():
    log.info(f"  {n:10s} {r}")
for b, d in res["per_build"].items():
    log.info(f"--- {b}: 그린 Powder → {d['generated'].get('Pow')}   진짜 CAD 분말 → {d['real'].get('CAD powder')}")
log.info(f"끝 ({time.time() - t0:.0f}초)")
