"""라벨 맵만 넣어 사진 두 장을 만든다 — 진짜 화소를 하나도 쓰지 않고 (가) 256² 창, (나) 1036² 레이어 (창을 이어 그림). scratchpad 전용."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from PIL import Image
from scipy import ndimage
from DefSeg_AM.common.utils.log import setup_logger
from DefSeg_AM.genaug import config_genaug as cfg
from DefSeg_AM.genaug.data.canvas_pool import CanvasPool
from DefSeg_AM.genaug.data.gt_pool import GTPool
from DefSeg_AM.genaug.models.diffusion import model_input
from DefSeg_AM.genaug.sampling.fill import load_generator
from DefSeg_AM.genaug.sampling.preview import REQUIRED_BASE, cad_label, object_mask

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
log = setup_logger(0, "layer")
dev = torch.device("cuda:0")
W, T = cfg.WIN, cfg.T_STEPS
SQ = np.ones((3, 3), bool)
model, diff, ck = load_generator(sys.argv[2], dev)
gt = GTPool("both", log=None)
canvas = CanvasPool(("B3",), log=None)
S = canvas.img_size
rng = np.random.default_rng(11)
TS = ((torch.linspace(1, 0, 50) ** 2) * (T - 1)).round().long().unique_consecutive().tolist()   # 낮은 잡음 쪽에 몬 50 단계


def grain_std(a, m=None):
    a = a.astype(np.float32)
    g = a - ndimage.gaussian_filter(a, 1.5)
    return float(g.std() if m is None else g[m].std())


@torch.no_grad()
def sample(x_clean, region, label, guidance=cfg.SAMPLE_CFG, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    rnd = lambda: torch.randn(x_clean.shape, device=dev, generator=g)
    label = torch.where(region[:, 0] > 0.5, label, torch.full_like(label, cfg.UNKNOWN))
    null = torch.full_like(label, cfg.UNKNOWN)
    B = x_clean.shape[0]
    tb = torch.full((B,), TS[0], device=dev, dtype=torch.long)
    x = region * rnd() + (1 - region) * diff.q_sample(x_clean, tb, rnd())
    for i, t in enumerate(TS):
        tb = torch.full((B,), t, device=dev, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            inp = torch.cat([model_input(x, x_clean, region, label), model_input(x, x_clean, region, null)])
            vc, vu = model(inp, torch.cat([tb, tb])).float().chunk(2)
        v = vu + guidance * (vc - vu)
        x0, _ = diff.x0_eps_from_v(x, tb, v)
        x0 = x0.clamp(-1, 1)
        if i + 1 < len(TS):
            eps = (x - diff._b(diff.sqrt_ac[tb]) * x0) / diff._b(diff.sqrt_1mac[tb])
            tp = torch.full((B,), TS[i + 1], device=dev, dtype=torch.long)
            x = diff._b(diff.sqrt_ac[tp]) * x0 + diff._b(diff.sqrt_1mac[tp]) * eps
            x = region * x + (1 - region) * diff.q_sample(x_clean, tp, rnd())
        else:
            x = x0
    return region * x + (1 - region) * x_clean


def put_defects(label, classes, tries=400):
    """라벨 맵에 진짜 결함 모양을 자리 규칙대로 얹는다."""
    lab = label.copy()
    H = lab.shape[0]
    placed = []
    for c in classes:
        rows = gt.objects_fitting(c, 120)
        for _ in range(tries):
            mask, _, _ = object_mask(gt, int(rng.choice(rows)))
            h, w = mask.shape
            if h >= H - 40 or w >= H - 40:
                continue
            y, x = int(rng.integers(20, H - 20 - h)), int(rng.integers(20, H - 20 - w))
            need = REQUIRED_BASE.get(c)
            cur = lab[y:y + h, x:x + w][mask]
            if (need is None and (cur < 2).mean() >= 0.99) or (need is not None and (cur == need).mean() >= 0.9):
                lab[y:y + h, x:x + w][mask] = c
                placed.append((cfg.CLASS_ABBRS[c], y + h // 2, x + w // 2))
                break
    return lab, placed


def color(lab):
    out = np.zeros(lab.shape + (3,), np.uint8)
    out[lab == cfg.POWDER] = (70, 70, 70); out[lab == cfg.PRINTED] = (200, 200, 200); out[lab == cfg.UNKNOWN] = (0, 0, 0)
    for c in range(2, cfg.N_CLASSES):
        m = lab == c
        if m.any():
            out[ndimage.binary_dilation(m, structure=SQ, iterations=2)] = cfg.CLASS_COLORS[c]
    return out


def stretch(a, lo, hi):
    return np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1.0) * 255.0, 0, 255).astype(np.uint8)


res = {}
t0 = time.time()
# ── (나) 레이어: B3 의 CAD 배치 + 결함 12 개 → 창을 이어 1036² 를 통째로. 진짜 화소 없음 ──
row = int(rng.integers(canvas.n_layers["B3"]))
r0, r1, part = canvas.crop("B3", row, 0, 0, S, S)
r0, r1, part = np.asarray(r0), np.asarray(r1), np.asarray(part).astype(bool)
lab_full, placed = put_defects(cad_label(part), [9, 9, 9, 7, 7, 7, 6, 6, 6, 5, 5, 5])
out = torch.zeros((2, S, S), device=dev)
done = torch.zeros((S, S), device=dev, dtype=torch.bool)
E = int(round(cfg.EDGE_FRAC * S))                                     # 사진 가장자리 5% (52 px) 는 진짜로 둔다 — 학습 때도 창을 놓지 않은 곳
real_t = torch.from_numpy(np.stack([r0, r1]).astype(np.float32) / 127.5 - 1).to(dev)
fr = torch.ones((S, S), device=dev, dtype=torch.bool); fr[E:S - E, E:S - E] = False
out[:, fr] = real_t[:, fr]; done |= fr
pos = sorted(set(list(range(E - 16, S - E + 16 - W, 192)) + [S - E + 16 - W]))
lab_t = torch.from_numpy(lab_full.astype(np.int64)).to(dev)
n = 0
for y in pos:
    for x in pos:
        sl = (slice(y, y + W), slice(x, x + W))
        region = (~done[sl]).float()[None, None]
        if region.sum() == 0:
            continue
        gx = sample(out[:, sl[0], sl[1]][None], region, lab_t[sl][None], seed=100 + n)
        out[:, sl[0], sl[1]] = gx[0]
        done[sl] = True
        n += 1
G = ((out.clamp(-1, 1) + 1) * 127.5).round().byte().cpu().numpy()
res["layer"] = dict(build="B3", orig_layer=int(canvas.orig_layers["B3"][row]), tiles=n, defects=placed,
                    mean=dict(real=[round(float(r0.mean()), 1), round(float(r1.mean()), 1)], gen=[round(float(G[0].mean()), 1), round(float(G[1].mean()), 1)]),
                    grain_powder=dict(real=[round(grain_std(r0, ~part & ~fr.cpu().numpy()), 2), round(grain_std(r1, ~part & ~fr.cpu().numpy()), 2)],
                                      gen=[round(grain_std(G[0], ~part & ~fr.cpu().numpy()), 2), round(grain_std(G[1], ~part & ~fr.cpu().numpy()), 2)]),
                    mean_inside=dict(real=[round(float(r0[E:S-E, E:S-E].mean()), 1), round(float(r1[E:S-E, E:S-E].mean()), 1)],
                                     gen=[round(float(G[0][E:S-E, E:S-E].mean()), 1), round(float(G[1][E:S-E, E:S-E].mean()), 1)]))
log.info(f"안쪽 평균 밝기 진짜 {res['layer']['mean_inside']['real']} / 그린 것 {res['layer']['mean_inside']['gen']}")
log.info(f"(나) 레이어 B3/L{res['layer']['orig_layer']} — 창 {n} 개, 결함 {len(placed)} 개.  평균 밝기 진짜 {res['layer']['mean']['real']} / 그린 것 {res['layer']['mean']['gen']}   "
         f"분말의 결 진짜 {res['layer']['grain_powder']['real']} / 그린 것 {res['layer']['grain_powder']['gen']}  ({time.time() - t0:.0f}초)")
for c, (real, gen) in enumerate(((r0, G[0]), (r1, G[1]))):
    lo, hi = np.percentile(real, [1, 99])
    Image.fromarray(np.concatenate([stretch(real, lo, hi), np.full((S, 8), 30, np.uint8), stretch(gen, lo, hi)], 1)).save(OUT / f"framed_v{c}_real_vs_generated.png")
Image.fromarray(color(lab_full)).save(OUT / "framed_labelmap.png")
(OUT / "layer_framed.json").write_text(json.dumps(res, indent=1, ensure_ascii=False), encoding="utf-8")
log.info("끝")
