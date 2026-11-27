"""v5 시각화 공용 — 11-class 팔레트와 4-panel 비교 그림 (v4 의 infer.py 에서 옮김, 색은 v4 와 같다)."""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import config_v5 as cfg


# v4 와 같은 11-class 팔레트 (figure 를 v4 와 나란히 볼 수 있게 그대로 둔다)
PALETTE_V5 = np.array([
    [0.85, 0.85, 0.85],  # 0 Powder
    [0.30, 0.50, 0.85],  # 1 Printed
    [0.95, 0.55, 0.10],  # 2 Recoater Hopping
    [0.90, 0.85, 0.15],  # 3 Recoater Streaking
    [0.20, 0.65, 0.30],  # 4 Incomplete Spreading
    [0.55, 0.30, 0.75],  # 5 Swelling
    [0.50, 0.35, 0.20],  # 6 Debris
    [0.95, 0.40, 0.65],  # 7 Super-Elevation
    [0.40, 0.40, 0.40],  # 8 Spatter
    [0.10, 0.70, 0.70],  # 9 Over Melting
    [0.75, 0.20, 0.10],  # 10 Under Melting
])

IGNORE_COLOR = np.array([0.05, 0.05, 0.05])   # 미레이블 / IGNORE 화소


def colorize(label: np.ndarray) -> np.ndarray:
    """v5 11-class label → RGB. IGNORE_INDEX 픽셀은 별도 어두운 색."""
    rgb = np.zeros((*label.shape, 3), dtype=np.float32)
    for c in range(cfg.N_CLASSES_V5):
        rgb[label == c] = PALETTE_V5[c]
    rgb[label == cfg.IGNORE_INDEX] = IGNORE_COLOR
    return rgb


def save_panel(
    img0_u8: np.ndarray,
    img1_u8: np.ndarray,
    ref_label: np.ndarray,
    pred_label: np.ndarray,
    out_path: Path,
    title: str,
    ref_title: str,
    pred_title: str = "our prediction (DefSeg-AM v5)",
) -> None:
    """[visible/0, visible/1, 기준 라벨, 예측] 4-panel PNG (v4 의 visualize 와 같은 배치)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patches

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    axes[0].imshow(img0_u8, cmap="gray"); axes[0].set_title("visible/0 (after melt)")
    axes[1].imshow(img1_u8, cmap="gray"); axes[1].set_title("visible/1 (after spread)")
    axes[2].imshow(colorize(ref_label));  axes[2].set_title(ref_title)
    axes[3].imshow(colorize(pred_label)); axes[3].set_title(pred_title)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    handles = [
        patches.Patch(color=tuple(PALETTE_V5[c]), label=f"{c} {cfg.CLASS_NAMES_V5[c]}")
        for c in range(cfg.N_CLASSES_V5)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
