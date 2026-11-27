"""실제 결함이 두 사진 (visible/0 = 용융 직후, visible/1 = 도포 직후) 에서 어떻게 보이는지 — 클래스별 예시 판.

클래스마다 서로 다른 소스에서 최대 3개. 열: visible/0 | visible/1 | 정답 윤곽. CPU 전용, 읽기만 한다.

실행 (CWD = 3DP_VPPM/, DefSeg_AM/venv):
    python -m DefSeg_AM.scripts.synth_prototype.look_real <out.png> 7,5,4 320 [fixed]
    (인자: 출력 그림, 클래스 번호 목록, 창 크기 native px, 'fixed' = 창 크기 고정)
"""
import sys
from pathlib import Path

import numpy as np
from PIL import Image
from scipy.ndimage import label as cc_label, binary_dilation
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from DefSeg_AM.v5 import config_v5 as cfg
from DefSeg_AM.v5.data.data_gt_v5 import enumerate_samples_v5, load_image_uint8, remap_label_v5

OUT = Path(sys.argv[1])
CLASSES = [int(c) for c in sys.argv[2].split(",")] if len(sys.argv) > 2 else [7, 5, 4, 3, 2, 6, 8, 9, 10]
WIN = int(sys.argv[3]) if len(sys.argv) > 3 else 320          # native px 창
FIXED = len(sys.argv) > 4 and sys.argv[4] == "fixed"
A = cfg.CLASS_ABBRS_V5
samples = enumerate_samples_v5()
anns = [remap_label_v5(np.load(s.ann_path), s.mapping_key) for s in samples]
pix = np.stack([np.bincount(a[a >= 0].astype(np.int64), minlength=11)[:11] for a in anns])

rows = []
for c in CLASSES:
    by_src = {}
    for i, s in enumerate(samples):
        if pix[i, c] > 0:
            by_src.setdefault(s.source_name, []).append(i)
    # 소스마다 그 클래스 화소가 중앙값인 레이어 하나
    picks = []
    for src, idxs in by_src.items():
        idxs = sorted(idxs, key=lambda i: pix[i, c])
        picks.append(idxs[len(idxs) // 2])
    picks = sorted(picks, key=lambda i: -pix[i, c])[:3]
    for i in picks:
        ann = anns[i]
        lab, n = cc_label(ann == c, structure=np.ones((3, 3)))
        sizes = np.bincount(lab.ravel())[1:]
        k = int(np.argsort(sizes)[len(sizes) * 3 // 4]) + 1 if n > 3 else int(np.argmax(sizes)) + 1
        ys, xs = np.where(lab == k)
        cy, cx = int(ys.mean()), int(xs.mean())
        H, W = ann.shape
        win = WIN if FIXED else max(WIN, int(1.4 * max(ys.max() - ys.min(), xs.max() - xs.min())))
        win = min(win, H, W)
        y0 = int(np.clip(cy - win // 2, 0, H - win)); x0 = int(np.clip(cx - win // 2, 0, W - win))
        i0 = load_image_uint8(samples[i].img0_path)[y0:y0 + win, x0:x0 + win]
        i1 = load_image_uint8(samples[i].img1_path)[y0:y0 + win, x0:x0 + win]
        a = ann[y0:y0 + win, x0:x0 + win]
        rows.append((c, samples[i].key, win, i0, i1, a, int(sizes[k - 1]), n))

fig, axes = plt.subplots(len(rows), 3, figsize=(9.6, 3.2 * len(rows)), dpi=110)
axes = np.atleast_2d(axes)
for r, (c, key, win, i0, i1, a, sz, n) in enumerate(rows):
    m = a == c
    edge = binary_dilation(m, iterations=max(1, win // 250)) & ~m
    for j, img in enumerate((i0, i1)):
        axes[r, j].imshow(img, cmap="gray", vmin=0, vmax=255)
    ov = np.stack([i1, i1, i1], -1).astype(np.float32) / 255.0
    ov[edge] = (1.0, 0.2, 0.2)
    other = (a >= 2) & ~m
    ov[binary_dilation(other, iterations=1) & ~other & ~m] = (0.2, 0.7, 1.0)
    axes[r, 2].imshow(ov)
    axes[r, 0].set_title(f"{A[c]} · {cfg.SOURCE_SHORT[key.split('/')[0]]}/{key.split('/')[1]} · visible/0 (after melt)", fontsize=8, loc="left")
    axes[r, 1].set_title(f"visible/1 (after spreading) · window {win}px native", fontsize=8, loc="left")
    axes[r, 2].set_title(f"visible/1 + GT outline (red = {A[c]}, blue = other defects) · comp {sz}px of {n}", fontsize=8, loc="left")
    for j in range(3):
        axes[r, j].axis("off")
fig.tight_layout()
fig.savefig(OUT)
print("saved", OUT, "rows", len(rows))
