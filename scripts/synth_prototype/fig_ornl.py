"""ORNL 사전 점검 그림: 전체 프레임 (왼쪽 3칸) + 합성 구역 확대 (오른쪽 4칸).
실행 (CWD = 3DP_VPPM/, DefSeg_AM/venv):
    python -m DefSeg_AM.scripts.synth_prototype.fig_ornl DefSeg_AM/cache/synth_prototype/ornl_raw \
        DefSeg_AM/figures/synth_prototype/fig_ornl_raw_blob.png blob      # | band
"""
import sys, json
from pathlib import Path
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

D, OUT = Path(sys.argv[1]), Path(sys.argv[2])
ZONE = sys.argv[3] if len(sys.argv) > 3 else "blob"
PICK = sys.argv[4].split(",") if len(sys.argv) > 4 else None
Z = np.load(D / "arrays.npz")
keys = sorted({k.split("|")[0] for k in Z.files})
if PICK:
    keys = [k for k in keys if k in PICK]
else:
    keys = [k for i, k in enumerate(keys) if i % 5 in (0, 2)][:8]

def outline(rgb, m, col, it=1):
    e = binary_dilation(m, iterations=it) & ~m
    rgb[e] = col
    return rgb

def box(m, S, lo=170, hi=420, pad=1.5):
    ys, xs = np.nonzero(m); cy, cx = int(np.median(ys)), int(np.median(xs))
    ext = max(np.percentile(ys, 97) - np.percentile(ys, 3), np.percentile(xs, 97) - np.percentile(xs, 3))
    w = int(np.clip(pad * ext, lo, hi)); y0 = int(np.clip(cy - w // 2, 0, S - w)); x0 = int(np.clip(cx - w // 2, 0, S - w))
    return slice(y0, y0 + w), slice(x0, x0 + w)

fig, ax = plt.subplots(len(keys), 7, figsize=(21, 3.25 * len(keys)), dpi=90)
ax = np.atleast_2d(ax)
for r, k in enumerate(keys):
    i0, i1, foot = Z[f"{k}|i0"], Z[f"{k}|i1"], Z[f"{k}|foot"]
    p = Z[f"{k}|pred_inlayer"]
    tag = f"{k}|SE_{ZONE}"
    S = i0.shape[0]
    g3 = lambda a: np.stack([a] * 3, -1).astype(np.float32) / 255
    ax[r, 0].imshow(i0, cmap="gray", vmin=0, vmax=255); ax[r, 0].set_title(f"{k} · visible/0 (after melt), full frame", fontsize=8, loc="left")
    ax[r, 1].imshow(outline(g3(i0), foot, (0.1, 1.0, 0.2), 2)); ax[r, 1].set_title("visible/0 + CAD part outline (green)", fontsize=8, loc="left")
    pv = g3(i1)
    pv[p == 1] = 0.5 * pv[p == 1] + 0.5 * np.array((0.2, 0.5, 1.0))
    for c, col in ((7, (1, .1, .9)), (5, (1, .55, 0))):
        pv[p == c] = 0.4 * pv[p == c] + 0.6 * np.array(col)
    other = (p >= 2) & (p != 7) & (p != 5)
    pv[other] = 0.5 * pv[other] + 0.5 * np.array((0.1, 0.9, 0.9))
    ax[r, 2].imshow(pv); ax[r, 2].set_title("visible/1 + prediction (inlayer): blue = Printed,\nmagenta = SE, orange = Sw, cyan = other", fontsize=7.5, loc="left")
    if f"{tag}|new" in Z.files:
        new, i1n, pn = Z[f"{tag}|new"], Z[f"{tag}|i1"], Z[f"{tag}|pred_inlayer"]
        sy, sx = box(new, S)
        ax[r, 3].imshow(outline(g3(i0)[sy, sx].copy(), foot[sy, sx], (0.1, 1.0, 0.2))); ax[r, 3].set_title("zoom: visible/0 + CAD outline", fontsize=8, loc="left")
        ax[r, 4].imshow(i1[sy, sx], cmap="gray", vmin=0, vmax=255); ax[r, 4].set_title("zoom: visible/1 original", fontsize=8, loc="left")
        ax[r, 5].imshow(outline(g3(i1n)[sy, sx].copy(), new[sy, sx], (1, .15, .15))); ax[r, 5].set_title(f"zoom: visible/1 synthetic ({ZONE}), red = SE label", fontsize=8, loc="left")
        qv = g3(i1n)[sy, sx].copy(); q = pn[sy, sx]
        qv[q == 1] = 0.5 * qv[q == 1] + 0.5 * np.array((0.2, 0.5, 1.0))
        for c, col in ((7, (1, .1, .9)), (5, (1, .55, 0))):
            qv[q == c] = 0.4 * qv[q == c] + 0.6 * np.array(col)
        oth = (q >= 2) & (q != 7) & (q != 5); qv[oth] = 0.5 * qv[oth] + 0.5 * np.array((0.1, 0.9, 0.9))
        rec = float((pn[new] == 7).mean())
        ax[r, 6].imshow(qv); ax[r, 6].set_title(f"zoom: prediction on synthetic (inlayer)\nred pixels → SE {100 * rec:.0f}%", fontsize=8, loc="left")
    for j in range(7):
        ax[r, j].axis("off")
fig.tight_layout(h_pad=1.6)
fig.savefig(OUT); print("saved", OUT, keys)
