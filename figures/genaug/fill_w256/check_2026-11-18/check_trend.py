"""학습이 진행되면서 결의 세기가 진짜에 가까워졌는가 — 남겨 둔 EMA 사본 (25k · 50k · 75k · 100k) 으로 같은 창을 그린다. scratchpad 전용."""
import sys
import numpy as np
import torch
from scipy import ndimage
from DefSeg_AM.common.utils.log import setup_logger
from DefSeg_AM.genaug import config_genaug as cfg
from DefSeg_AM.genaug.data.canvas_pool import CanvasPool
from DefSeg_AM.genaug.data.windows import crop_ornl, region_full
from DefSeg_AM.genaug.sampling.fill import fill_windows, load_generator

log = setup_logger(0, "trend")
dev = torch.device("cuda:0")
W = cfg.WIN
FULL = region_full(W, cfg.BORDER)
canvas = CanvasPool(cfg.TRAIN_BUILDS, log=None)
S = canvas.img_size
e = int(round(cfg.EDGE_FRAC * S))


def g_std(img):
    a = img.astype(np.float32)
    return float((a - ndimage.gaussian_filter(a, 1.5))[FULL].std())


rng = np.random.default_rng(2026)
data = {}
for b in cfg.TRAIN_BUILDS:
    I0, I1 = [], []
    for _ in range(32):
        row = int(rng.integers(canvas.n_layers[b]))
        y0, x0 = int(rng.integers(e, S - e - W + 1)), int(rng.integers(e, S - e - W + 1))
        a0, a1, _, _ = crop_ornl(canvas, b, row, y0, x0, W, False)
        I0.append(a0.copy()); I1.append(a1.copy())
    data[b] = (np.stack(I0), np.stack(I1))
for path in sys.argv[1:]:
    model, diff, ck = load_generator(path, dev)
    parts = []
    for b, (I0, I1) in data.items():
        reg = np.stack([FULL] * len(I0))
        lab = np.full(I0.shape, cfg.UNKNOWN, np.uint8)
        q0, q1 = fill_windows(model, diff, I0, I1, reg, lab, steps=50, guidance=1.0, seed=7, batch=32, amp=True)
        r0 = np.median([g_std(q0[n]) for n in range(len(I0))]) / np.median([g_std(I0[n]) for n in range(len(I0))])
        r1 = np.median([g_std(q1[n]) for n in range(len(I1))]) / np.median([g_std(I1[n]) for n in range(len(I1))])
        parts.append(f"{b} v0 {r0:.2f} v1 {r1:.2f}")
    log.info(f"step {ck['step']:6d}  결 비율 (그린 것/진짜):  " + "   ".join(parts))
    del model
    torch.cuda.empty_cache()
