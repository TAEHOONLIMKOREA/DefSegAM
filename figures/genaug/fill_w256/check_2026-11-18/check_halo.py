"""결함 넣기에서 다시 그린 띠의 결 (매끈한 자국) 과, 그리는 단계의 배치만 바꾼 변형. scratchpad 전용."""
import json, sys, time
from pathlib import Path
import numpy as np
import torch
from scipy import ndimage
from DefSeg_AM.common.utils.log import setup_logger
from DefSeg_AM.genaug import config_genaug as cfg
from DefSeg_AM.genaug.data.canvas_pool import CanvasPool
from DefSeg_AM.genaug.data.gt_pool import GTPool
from DefSeg_AM.genaug.data.windows import OBJECT_MARGIN, crop_ornl, region_full
from DefSeg_AM.genaug.models.diffusion import model_input
from DefSeg_AM.genaug.sampling.fill import load_generator, to_uint8
from DefSeg_AM.genaug.sampling.preview import cad_label, object_mask, place_mask

OUT = Path(sys.argv[1])
log = setup_logger(0, "halo")
dev = torch.device("cuda:0")
W, T = cfg.WIN, cfg.T_STEPS
SQ = np.ones((3, 3), bool)
FULL = region_full(W, cfg.BORDER)
model, diff, ck = load_generator(sys.argv[2], dev)
gt = GTPool("both", log=None)
canvas = CanvasPool(cfg.TRAIN_BUILDS, log=None)
S = canvas.img_size
e = int(round(cfg.EDGE_FRAC * S))


def grain(a):
    a = a.astype(np.float32)
    return a - ndimage.gaussian_filter(a, 1.5)


def steps_of(n, spacing):
    u = torch.linspace(1, 0, n)
    if spacing == "quad":
        u = u ** 2
    return (u * (T - 1)).round().long().unique_consecutive().tolist()


@torch.no_grad()
def sample(x_clean, region, label, ts, guidance, seed=0):
    g = torch.Generator(device=dev).manual_seed(seed)
    rnd = lambda: torch.randn(x_clean.shape, device=dev, generator=g)
    label = torch.where(region[:, 0] > 0.5, label, torch.full_like(label, cfg.UNKNOWN))
    null = torch.full_like(label, cfg.UNKNOWN)
    B = x_clean.shape[0]
    tb = torch.full((B,), ts[0], device=dev, dtype=torch.long)
    x = region * rnd() + (1 - region) * diff.q_sample(x_clean, tb, rnd())
    for i, t in enumerate(ts):
        tb = torch.full((B,), t, device=dev, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            if guidance != 1.0:
                inp = torch.cat([model_input(x, x_clean, region, label), model_input(x, x_clean, region, null)])
                vc, vu = model(inp, torch.cat([tb, tb])).float().chunk(2)
                v = vu + guidance * (vc - vu)
            else:
                v = model(model_input(x, x_clean, region, label), tb).float()
        x0, _ = diff.x0_eps_from_v(x, tb, v)
        x0 = x0.clamp(-1, 1)
        if i + 1 < len(ts):
            eps = (x - diff._b(diff.sqrt_ac[tb]) * x0) / diff._b(diff.sqrt_1mac[tb])
            tp = torch.full((B,), ts[i + 1], device=dev, dtype=torch.long)
            x = diff._b(diff.sqrt_ac[tp]) * x0 + diff._b(diff.sqrt_1mac[tp]) * eps
            x = region * x + (1 - region) * diff.q_sample(x_clean, tp, rnd())
        else:
            x = x0
    return to_uint8(region * x + (1 - region) * x_clean)


res = {}
rng = np.random.default_rng(2026)
t0 = time.time()
# (가) 창 전체 — 단계 배치만 바꾼다 (라벨 없음)
for b in cfg.TRAIN_BUILDS:
    I0, I1 = [], []
    for _ in range(32):
        row = int(rng.integers(canvas.n_layers[b]))
        y0, x0 = int(rng.integers(e, S - e - W + 1)), int(rng.integers(e, S - e - W + 1))
        a0, a1, _, _ = crop_ornl(canvas, b, row, y0, x0, W, False)
        I0.append(a0.copy()); I1.append(a1.copy())
    I0, I1 = np.stack(I0), np.stack(I1)
    x = torch.from_numpy(np.stack([I0, I1], 1).astype(np.float32) / 127.5 - 1).to(dev)
    reg = torch.from_numpy(FULL[None, None].astype(np.float32)).to(dev).expand(len(x), -1, -1, -1)
    lab = torch.full((len(x), W, W), cfg.UNKNOWN, device=dev, dtype=torch.long)
    real = [np.median([grain(im[n])[FULL].std() for n in range(len(im))]) for im in (I0, I1)]
    for name, n, sp in (("quad50", 50, "quad"), ("quad100", 100, "quad")):
        u = sample(x, reg, lab, steps_of(n, sp), 1.0, seed=3)
        r = [float(np.median([grain(u[k, c])[FULL].std() for k in range(len(u))]) / real[c]) for c in (0, 1)]
        res[f"whole_{b}_{name}"] = r
        log.info(f"창 전체 {b} {name:8s} 결 비율 v0 {r[0]:.2f} v1 {r[1]:.2f}  ({time.time() - t0:.0f}초)")

# (나) 결함 넣기 — 다시 그린 띠 (객체 둘레 6 px) 의 결을 넣기 전 같은 화소와 견준다
I0, I1, LAB, REG, OBJ = [], [], [], [], []
fit = W - 2 * OBJECT_MARGIN
for c in (9, 7, 5):                                   # DB · SP · RH
    rows = rng.choice(gt.objects_fitting(c, fit), size=16, replace=False)
    k = 0
    while k < 16:
        mask, _, _ = object_mask(gt, int(rows[k]))
        b, row = canvas.pick_layer(rng)
        y0, x0 = int(rng.integers(e, S - e - W + 1)), int(rng.integers(e, S - e - W + 1))
        a0, a1, _, part = crop_ornl(canvas, b, row, y0, x0, W, False)
        base = cad_label(part)
        placed = place_mask(mask, base, c, rng, W)
        if placed is None:
            continue
        lab = base.copy(); lab[placed] = c
        I0.append(a0.copy()); I1.append(a1.copy()); LAB.append(lab); OBJ.append(placed)
        REG.append(ndimage.binary_dilation(placed, structure=SQ, iterations=cfg.RING_PX))
        k += 1
I0, I1, LAB, REG, OBJ = (np.stack(v) for v in (I0, I1, LAB, REG, OBJ))
x = torch.from_numpy(np.stack([I0, I1], 1).astype(np.float32) / 127.5 - 1).to(dev)
reg = torch.from_numpy(REG[:, None].astype(np.float32)).to(dev)
lab = torch.from_numpy(LAB.astype(np.int64)).to(dev)
RING = REG & ~ndimage.binary_dilation(OBJ, structure=SQ[None], iterations=2)      # 객체에서 2 px 떨어진 띠 — 결함의 가장자리를 뺀다
log.info(f"결함 넣기 창 {len(I0)} 개, 띠의 화소 수 중앙값 {int(np.median(RING.sum((1, 2))))}")
for name, n, sp in (("uniform50", 50, "uniform"), ("quad50", 50, "quad"), ("quad100", 100, "quad"), ("uniform250", 250, "uniform")):
    outs = [sample(x[a:a + 24], reg[a:a + 24], lab[a:a + 24], steps_of(n, sp), cfg.SAMPLE_CFG, seed=3) for a in range(0, len(x), 24)]
    u = np.concatenate(outs)
    r = []
    for c, bg in enumerate((I0, I1)):
        ratios = [grain(u[k, c])[RING[k]].std() / max(grain(bg[k])[RING[k]].std(), 1e-6) for k in range(len(u)) if RING[k].sum() >= 60]
        r.append(float(np.median(ratios)))
    res[f"insert_ring_{name}"] = r
    log.info(f"결함 넣기 {name:10s} 띠의 결 비율 (그린 것/넣기 전) v0 {r[0]:.2f} v1 {r[1]:.2f}  ({time.time() - t0:.0f}초)")
(OUT / "halo.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
log.info("끝")
