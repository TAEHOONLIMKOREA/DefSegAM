"""v6 시각화 공용 — 10-class 팔레트와 4-panel 비교 그림.

색은 DIVA 화면의 표시 색 (`schema_pbf_2026_11.json` 의 color) 을 그대로 쓴다 — 레이블링 화면과 나란히 볼 수 있게.
v4 · v5 의 그림과는 색이 다르다 (클래스 번호도 다르다).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .. import config_v6 as cfg


PALETTE_V6 = np.array(cfg.DIVA_COLORS, dtype=np.float32) / 255.0          # (10, 3)
UNLABELED_RGB = np.array(cfg.UNLABELED_COLOR, dtype=np.float32) / 255.0    # 미레이블 (-1) · 대응 없는 예측


def colorize(label: np.ndarray) -> np.ndarray:
    """10-class label → RGB. 0–9 밖의 값 (미레이블 -1, 기준선의 "대응 없음") 은 미레이블 색."""
    rgb = np.empty((*label.shape, 3), dtype=np.float32)
    rgb[...] = UNLABELED_RGB
    for c in range(cfg.N_CLASSES_V6):
        rgb[label == c] = PALETTE_V6[c]
    return rgb


def save_panel(
    img0_u8: np.ndarray,
    img1_u8: np.ndarray,
    ref_label: np.ndarray,
    pred_label: np.ndarray,
    out_path: Path,
    title: str,
    ref_title: str,
    pred_title: str = "our prediction (DefSeg-AM v6)",
) -> None:
    """[visible/0, visible/1, 기준 라벨, 예측] 4-panel PNG (v4 · v5 의 그림과 같은 배치)."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import patches

    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    axes[0].imshow(img0_u8, cmap="gray", vmin=0, vmax=255); axes[0].set_title("visible/0 (after melt)")
    axes[1].imshow(img1_u8, cmap="gray", vmin=0, vmax=255); axes[1].set_title("visible/1 (after spread)")
    axes[2].imshow(colorize(ref_label), interpolation="nearest");  axes[2].set_title(ref_title)
    axes[3].imshow(colorize(pred_label), interpolation="nearest"); axes[3].set_title(pred_title)
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    handles = [
        patches.Patch(color=tuple(PALETTE_V6[c]), label=f"{c} {cfg.CLASS_NAMES_V6[c]}")
        for c in range(cfg.N_CLASSES_V6)
    ]
    handles.append(patches.Patch(color=tuple(UNLABELED_RGB), label="unlabeled / no class"))
    fig.legend(handles=handles, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")
