"""시제품 결과 그림 — 실제 결함과 합성 결함을 같은 배율로 나란히.

열: visible/0 (용융 직후) | visible/1 원본 | visible/1 합성 | 정답 (빨강 = 결함, 노랑 = IGNORE 띠) | 실제 사진만으로 학습한 모델의 예측

실행 (CWD = 3DP_VPPM/, DefSeg_AM/venv):
    python -m DefSeg_AM.scripts.synth_prototype.fig_on_gt DefSeg_AM/cache/synth_prototype/eval_gt \
        DefSeg_AM/figures/synth_prototype/fig_se_sources.png se_sources      # | new_sources | swelling | loso
"""
import sys
from pathlib import Path

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from scipy.ndimage import binary_dilation

D, OUT, WHICH = Path(sys.argv[1]), Path(sys.argv[2]), sys.argv[3]
Z = np.load(D / "arrays.npz")
files = set(Z.files)
SHORT = {"v2021_LPBF": "L21", "v2022_17-4PH": "17-4", "v2022_GammaPrint": "GP", "v2022_Inc718_1": "I1",
         "v2022_Inc718_2": "I2", "v2022_Maraging": "MS", "v2021_BJ": "BJ21", "v2022_BJ_H13": "H13"}
keys = sorted({k.split("|")[0] for k in files})
by_src = {}
for k in keys:
    by_src.setdefault(SHORT[k.split("/")[0]], []).append(k)


def box(mask, S, lo=150, hi=380, pad=1.6):
    ys, xs = np.nonzero(mask)
    cy, cx = int(np.median(ys)), int(np.median(xs))
    ext = max(np.percentile(ys, 97) - np.percentile(ys, 3), np.percentile(xs, 97) - np.percentile(xs, 3))
    w = int(np.clip(pad * ext, lo, hi))
    y0 = int(np.clip(cy - w // 2, 0, S - w)); x0 = int(np.clip(cx - w // 2, 0, S - w))
    return slice(y0, y0 + w), slice(x0, x0 + w)


def first(src, tag, min_px=40):
    """그 소스의 검증 레이어 가운데 tag 합성이 있고 새 화소가 가장 많은 것."""
    best = None
    for k in by_src.get(src, []):
        f = f"{k}|{tag}|new"
        if f in files:
            n = int(Z[f].sum())
            if n >= min_px and (best is None or n > best[1]):
                best = (k, n)
    return best[0] if best else None


def real(src, cls, min_px=150):
    best = None
    for k in by_src.get(src, []):
        n = int((Z[f"{k}|orig|ann"] == cls).sum())
        if n >= min_px and (best is None or n > best[1]):
            best = (k, n)
    return best[0] if best else None


rows = []        # (title, i0, i1_orig, i1_new or None, defect mask, ignore mask, pred, model name)
def add_real(src, cls, name):
    k = real(src, cls)
    if k is None:
        return
    ann = Z[f"{k}|orig|ann"]
    rows.append((f"REAL {name} — {src}/{k.split('/')[1]}", Z[f"{k}|orig|i0"], Z[f"{k}|orig|i1"], None,
                 ann == cls, ann == -1, Z[f"{k}|orig|pred_inlayer"], "inlayer"))


def add_sim(src, tag, name, model="inlayer"):
    k = first(src, tag)
    if k is None:
        return
    annn = Z[f"{k}|{tag}|ann"]; ann = Z[f"{k}|orig|ann"]
    pk = f"{k}|{tag}|pred_{model}"
    if pk not in files:
        model = "inlayer"; pk = f"{k}|{tag}|pred_{model}"
    rows.append((f"SYNTHETIC {name} — {src}/{k.split('/')[1]}", Z[f"{k}|orig|i0"], Z[f"{k}|orig|i1"], Z[f"{k}|{tag}|i1"],
                 Z[f"{k}|{tag}|new"], (annn == -1) & (ann != -1), Z[pk], model))


if WHICH == "se_sources":            # 실제 SE 가 있는 소스: 실제와 합성을 나란히
    add_real("GP", 7, "Super-Elevation"); add_sim("GP", "SE0", "Super-Elevation (blob)")
    add_real("L21", 7, "Super-Elevation"); add_sim("L21", "SE0", "Super-Elevation (blob)"); add_sim("L21", "SE2", "Super-Elevation (band)")
    add_sim("I2", "SE0", "Super-Elevation (blob)")
elif WHICH == "new_sources":         # 실제 SE 주석이 하나도 없는 소스
    for s in ("MS", "17-4", "I1", "BJ21", "H13"):
        add_sim(s, "SE0", "Super-Elevation (blob)")
elif WHICH == "swelling":
    add_real("GP", 5, "Swelling"); add_real("L21", 5, "Swelling")
    add_sim("L21", "Sw4", "Swelling"); add_sim("GP", "Sw4", "Swelling"); add_sim("17-4", "Sw4", "Swelling")
elif WHICH == "loso":                # L21 을 본 적 없는 모델의 예측
    add_sim("L21", "SE0", "Super-Elevation (blob)", model="loso_L21"); add_sim("L21", "SE2", "Super-Elevation (band)", model="loso_L21")

cols = ["visible/0 (after melt)", "visible/1 (after spreading) — original", "visible/1 — with synthetic defect",
        "label: red = defect, yellow = ignore", "prediction (model trained on real photos only)"]
fig, axes = plt.subplots(len(rows), 5, figsize=(15.5, 3.3 * len(rows)), dpi=100)
axes = np.atleast_2d(axes)
for r, (title, i0, i1, i1n, m, ign, pred, model) in enumerate(rows):
    S = i0.shape[0]
    sy, sx = box(m, S)
    shown = i1 if i1n is None else i1n
    g = lambda a: a[sy, sx]
    axes[r, 0].imshow(g(i0), cmap="gray", vmin=0, vmax=255)
    axes[r, 1].imshow(g(i1), cmap="gray", vmin=0, vmax=255)
    if i1n is None:
        axes[r, 2].imshow(np.full_like(g(i1), 245), cmap="gray", vmin=0, vmax=255)
        axes[r, 2].text(0.5, 0.5, "real defect\n(nothing synthesized)", ha="center", va="center", transform=axes[r, 2].transAxes, fontsize=9, color="#555")
    else:
        axes[r, 2].imshow(g(i1n), cmap="gray", vmin=0, vmax=255)
    ov = np.stack([g(shown)] * 3, -1).astype(np.float32) / 255.0
    mm, ii = g(m), g(ign)
    ov[binary_dilation(mm, iterations=1) & ~mm] = (1.0, 0.15, 0.15)
    ov[binary_dilation(ii, iterations=1) & ~ii & ~mm] = (1.0, 0.85, 0.1)
    axes[r, 3].imshow(ov)
    pv = np.stack([g(shown)] * 3, -1).astype(np.float32) / 255.0
    p = g(pred)
    for cls, colr in ((7, (1.0, 0.1, 0.9)), (5, (1.0, 0.55, 0.0))):
        sel = p == cls
        pv[sel] = 0.45 * pv[sel] + 0.55 * np.array(colr, dtype=np.float32)
    other = (p >= 2) & (p != 7) & (p != 5)
    pv[other] = 0.6 * pv[other] + 0.4 * np.array((0.1, 0.8, 1.0), dtype=np.float32)
    axes[r, 4].imshow(pv)
    rec = float((p[mm] == 7).mean()) if mm.any() else float("nan")
    rec5 = float((p[mm] == 5).mean()) if mm.any() else float("nan")
    head, where = title.split(" — ")
    axes[r, 0].set_title(f"{head}\n{where} · visible/0 (after melt)", fontsize=8.5, loc="left", fontweight="bold")
    axes[r, 4].set_title(f"prediction of the model trained on real photos ({model})\n"
                         f"magenta = SE · orange = Swelling · cyan = other defect\n"
                         f"red pixels → SE {100 * rec:.0f}% · Swelling {100 * rec5:.0f}%", fontsize=7.5, loc="left")
    for j in range(1, 4):
        axes[r, j].set_title(cols[j], fontsize=8, loc="left")
    for j in range(5):
        axes[r, j].axis("off")
fig.tight_layout(h_pad=2.2)
fig.savefig(OUT)
print("saved", OUT, "rows", len(rows))
