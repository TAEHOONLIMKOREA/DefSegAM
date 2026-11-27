"""도포 직후 사진 (visible/1) 의 결이 왜 매끈하게 그려지는가 — 잡음 바닥 가설과 그리는 방법만 바꾼 변형들. scratchpad 전용."""
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from scipy import ndimage

from DefSeg_AM.common.utils.log import setup_logger
from DefSeg_AM.genaug import config_genaug as cfg
from DefSeg_AM.genaug.data.canvas_pool import CanvasPool
from DefSeg_AM.genaug.data.windows import crop_ornl, region_full
from DefSeg_AM.genaug.models.diffusion import model_input
from DefSeg_AM.genaug.sampling.fill import load_generator, to_uint8

OUT = Path(sys.argv[1])
log = setup_logger(0, "grain")
dev = torch.device("cuda:0")
W = cfg.WIN
SQ = np.ones((3, 3), bool)
FULL = region_full(W, cfg.BORDER)
model, diff, ck = load_generator(sys.argv[2], dev)
canvas = CanvasPool(cfg.TRAIN_BUILDS, log=log)
S = canvas.img_size
e = int(round(cfg.EDGE_FRAC * S))
log.info(f"잡음 바닥: t=0 의 잡음 표준편차 {float(diff.sqrt_1mac[0]):.5f} ([-1,1]) = {float(diff.sqrt_1mac[0]) * 127.5:.2f} 단계 (0–255),  "
         f"t=10 {float(diff.sqrt_1mac[10]) * 127.5:.2f},  t=50 {float(diff.sqrt_1mac[50]) * 127.5:.2f}")


def grain(a):
    a = a.astype(np.float32)
    return a - ndimage.gaussian_filter(a, 1.5)


def gstat(img, m):
    g = grain(img)
    gi = g.copy()
    h = (m[:, :-1] & m[:, 1:])
    v = (m[:-1] & m[1:])
    ac_h = float(np.corrcoef(gi[:, :-1][h], gi[:, 1:][h])[0, 1])
    ac_v = float(np.corrcoef(gi[:-1][v], gi[1:][v])[0, 1])
    return float(g[m].std()), ac_h, ac_v


@torch.no_grad()
def sample(x_clean, region, label, steps, eta, last="mean", seed=0):
    """sample_fill 과 같은 절차 + eta (0 = DDIM, 1 = DDPM 식) + 마지막 단계를 고르는 옵션. guidance 없음 (라벨이 전부 '모름')."""
    g = torch.Generator(device=dev).manual_seed(seed)

    def rnd():
        return torch.randn(x_clean.shape, device=dev, generator=g)

    B = x_clean.shape[0]
    ts = torch.linspace(diff.T - 1, 0, steps, device=dev).round().long().unique_consecutive().tolist()
    tb = torch.full((B,), ts[0], device=dev, dtype=torch.long)
    x = region * rnd() + (1 - region) * diff.q_sample(x_clean, tb, rnd())
    for i, t in enumerate(ts):
        tb = torch.full((B,), t, device=dev, dtype=torch.long)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(model_input(x, x_clean, region, label), tb).float()
        x0, _ = diff.x0_eps_from_v(x, tb, v)
        x0 = x0.clamp(-1, 1)
        a_t, s_t = diff._b(diff.sqrt_ac[tb]), diff._b(diff.sqrt_1mac[tb])
        eps = (x - a_t * x0) / s_t
        if i + 1 < len(ts):
            tp = torch.full((B,), ts[i + 1], device=dev, dtype=torch.long)
            a_p, s_p = diff._b(diff.sqrt_ac[tp]), diff._b(diff.sqrt_1mac[tp])
            sig = eta * (s_p / s_t) * torch.sqrt((1 - (a_t / a_p) ** 2).clamp(min=0))
            x = a_p * x0 + torch.sqrt((s_p ** 2 - sig ** 2).clamp(min=0)) * eps + sig * rnd()
            x = region * x + (1 - region) * diff.q_sample(x_clean, tp, rnd())
        else:
            x = x if last == "keep" else x0          # keep = 마지막 잡음 (t=0 의 잡음) 을 걷어내지 않고 둔다
    return region * x + (1 - region) * x_clean


N = 32
rng = np.random.default_rng(2026)
res = {"noise_floor_levels": float(diff.sqrt_1mac[0]) * 127.5}
data = {}
for b in cfg.TRAIN_BUILDS:
    I0, I1 = [], []
    for _ in range(N):
        row = int(rng.integers(canvas.n_layers[b]))
        y0, x0 = int(rng.integers(e, S - e - W + 1)), int(rng.integers(e, S - e - W + 1))
        a0, a1, _, _ = crop_ornl(canvas, b, row, y0, x0, W, False)
        I0.append(a0.copy()); I1.append(a1.copy())
    data[b] = (np.stack(I0), np.stack(I1))

reg_t = torch.from_numpy(FULL[None, None].astype(np.float32)).to(dev)
VARIANTS = [("ddim50", 50, 0.0, "mean"), ("ddim250", 250, 0.0, "mean"), ("ddpm50", 50, 1.0, "mean"), ("ddpm250", 250, 1.0, "mean"),
            ("ddpm1000", 1000, 1.0, "mean"), ("ddim50_keep", 50, 0.0, "keep"), ("ddpm250_keep", 250, 1.0, "keep")]
t0 = time.time()
for b, (I0, I1) in data.items():
    x = torch.from_numpy(np.stack([I0, I1], 1).astype(np.float32) / 127.5 - 1).to(dev)
    reg = reg_t.expand(len(x), -1, -1, -1)
    lab = torch.full((len(x), W, W), cfg.UNKNOWN, device=dev, dtype=torch.long)
    real = {ch: [gstat(img[n], FULL) for n in range(N)] for ch, img in (("v0", I0), ("v1", I1))}
    res[b] = {"real": {ch: [round(float(np.median([s[k] for s in real[ch]])), 3) for k in range(3)] for ch in real}}
    log.info(f"{b} 진짜: " + "  ".join(f"{ch} 결 {res[b]['real'][ch][0]:.2f} (가로 상관 {res[b]['real'][ch][1]:+.2f}, 세로 {res[b]['real'][ch][2]:+.2f})" for ch in real))
    # (가) 한 번만 걷어내기 — t 단계의 잡음을 섞은 진짜 창을 모델에 한 번 넣어 원본을 맞히게 한다
    one = {}
    for t in (0, 2, 5, 10, 20, 50, 100):
        tb = torch.full((len(x),), t, device=dev, dtype=torch.long)
        gg = torch.Generator(device=dev).manual_seed(5)
        xt = diff.q_sample(x, tb, torch.randn(x.shape, device=dev, generator=gg))
        xt = reg * xt + (1 - reg) * x
        with torch.no_grad(), torch.autocast("cuda", dtype=torch.bfloat16):
            v = model(model_input(xt, x, reg, lab), tb).float()
        x0p, _ = diff.x0_eps_from_v(xt, tb, v)
        u = to_uint8(reg * x0p + (1 - reg) * x)
        one[t] = {ch: round(float(np.median([gstat(u[n, c], FULL)[0] for n in range(N)])) / res[b]["real"][ch][0], 3)
                  for c, ch in enumerate(("v0", "v1"))}
    res[b]["one_step_grain_ratio"] = one
    log.info(f"{b} 한 번 걷어내기의 결 비율 (그린 것/진짜) — " + "  ".join(f"t={t}: v0 {one[t]['v0']:.2f} v1 {one[t]['v1']:.2f}" for t in one))
    # (나) 그리는 방법만 바꾼다
    res[b]["variants"] = {}
    for name, steps, eta, last in VARIANTS:
        outs = []
        for a in range(0, len(x), 32):
            outs.append(to_uint8(sample(x[a:a + 32], reg[a:a + 32], lab[a:a + 32], steps, eta, last, seed=3)))
        u = np.concatenate(outs)
        r = {}
        for c, ch in enumerate(("v0", "v1")):
            st = [gstat(u[n, c], FULL) for n in range(N)]
            r[ch] = dict(grain_ratio=round(float(np.median([s[0] for s in st])) / res[b]["real"][ch][0], 3),
                         ac_h=round(float(np.median([s[1] for s in st])), 3), ac_v=round(float(np.median([s[2] for s in st])), 3))
        res[b]["variants"][name] = r
        log.info(f"{b} {name:13s} 결 비율 v0 {r['v0']['grain_ratio']:.2f} v1 {r['v1']['grain_ratio']:.2f}   "
                 f"v1 상관 가로 {r['v1']['ac_h']:+.2f} 세로 {r['v1']['ac_v']:+.2f}   ({time.time() - t0:.0f}초)")
(OUT / "grain.json").write_text(json.dumps(res, indent=1), encoding="utf-8")
log.info("끝")
