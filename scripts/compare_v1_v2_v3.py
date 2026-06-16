"""v1 / v2 / v3 stage 1·2 ckpt 를 같은 ORNL build 의 같은 layer 들에 대해 추론
→ 공통 8-class 어휘로 매핑해 정성 figure + 정량 mIoU 비교.

각 버전의 native class 가 다르므로 모든 prediction 을 다음 공통 어휘로 reduce:

  0 Powder              1 Printed
  2 Recoater Disturbance         (v1/v3 의 Hopping+Streaking)
  3 Swelling            4 Debris
  5 Super-Elevation     6 Spatter
  7 Over Melting

매핑되지 않는 native class (Incomplete Spreading / Misprint / Under Melting) 는
IGNORE 처리. GT 는 ORNL HDF5 의 native 12 mask 의 argmax 를 공통으로 매핑.

사용:
    python -m DefSeg_AM.scripts.compare_v1_v2_v3 \\
        [--build "2021-07-13 TCR Phase 1 Build 1"] [--n-layers 8] \\
        [--n-confusion-layers 200] [--img-size 1036]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import patches
from PIL import Image

from DefSeg_AM.common.models.model import DefSegModel, round_to_patch
from DefSeg_AM.common.data.image_utils import normalize_image, ornl_image_to_uint8


# ============================================================================
# Common 8-class taxonomy + per-version mapping
# ============================================================================
COMMON_CLASS_NAMES = [
    "Powder",
    "Printed",
    "Recoater Disturbance",
    "Swelling",
    "Debris",
    "Super-Elevation",
    "Spatter",
    "Over Melting",
]
N_COMMON = len(COMMON_CLASS_NAMES)
IGNORE_INDEX = -1

# v1 = ORNL 12-class 원본 (Hopping+Streaking 분리)
V1_TO_COMMON = {
    0: 0,    # Powder
    1: 1,    # Printed
    2: 2,    # Recoater Hopping    → Recoater Disturbance
    3: 2,    # Recoater Streaking  → Recoater Disturbance
    4: -1,   # Incomplete Spreading → IGNORE
    5: 3,    # Swelling
    6: 4,    # Debris
    7: 5,    # Super-Elevation
    8: 6,    # Spatter
    9: -1,   # Misprint            → IGNORE
    10: 7,   # Over Melting
    11: -1,  # Under Melting       → IGNORE
}

# v2 = 8-class (Recoater 통합)
V2_TO_COMMON = {
    0: 0,    # Powder
    1: 1,    # Printed
    2: 2,    # Recoater Disturbance
    3: 3,    # Swelling
    4: 6,    # Spatter
    5: 5,    # Super-Elevation
    6: 7,    # Over Melting
    7: 4,    # Debris
}

# v3 = 10-class (Hopping/Streaking 분리)
V3_TO_COMMON = {
    0: 0,    # Powder
    1: 1,    # Printed
    2: 2,    # Recoater Hopping    → Recoater Disturbance
    3: 2,    # Recoater Streaking  → Recoater Disturbance
    4: -1,   # Incomplete Spreading → IGNORE
    5: 3,    # Swelling
    6: 4,    # Debris
    7: 5,    # Super-Elevation
    8: 6,    # Spatter
    9: 7,    # Over Melting
}

# ORNL HDF5 native 12 → common (= V1_TO_COMMON 와 동일)
ORNL_12_TO_COMMON = V1_TO_COMMON


# ============================================================================
# Color palette (공통 8)
# ============================================================================
PALETTE = np.array([
    [0.85, 0.85, 0.85],  # 0 Powder
    [0.30, 0.50, 0.85],  # 1 Printed
    [0.95, 0.55, 0.10],  # 2 Recoater Disturbance
    [0.55, 0.30, 0.75],  # 3 Swelling
    [0.50, 0.35, 0.20],  # 4 Debris
    [0.95, 0.40, 0.65],  # 5 Super-Elevation
    [0.40, 0.40, 0.40],  # 6 Spatter
    [0.10, 0.70, 0.70],  # 7 Over Melting
])
IGNORE_COLOR = np.array([0.05, 0.05, 0.05])


def remap_to_common(label_native: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    """native class label → common 8-class label (-1 = IGNORE)."""
    out = np.full_like(label_native, fill_value=IGNORE_INDEX, dtype=np.int8)
    for src, dst in mapping.items():
        if dst == IGNORE_INDEX:
            continue
        out[label_native == src] = dst
    return out


def colorize_common(label: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*label.shape, 3), dtype=np.float32)
    for c in range(N_COMMON):
        rgb[label == c] = PALETTE[c]
    rgb[label == IGNORE_INDEX] = IGNORE_COLOR
    return rgb


# ============================================================================
# Model load + inference
# ============================================================================
def load_model(path: Path, device: torch.device) -> tuple[DefSegModel, int, int]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    c = ckpt["config"]
    n_cls = c.get("n_classes")
    model = DefSegModel(backbone_name=c["backbone"], n_classes=n_cls).to(device)
    model.load_trainable_state_dict(ckpt["model_state"])
    model.eval()
    img_size = c.get("img_size", 1036)
    return model, n_cls, img_size


@torch.no_grad()
def infer_pred_native(model: DefSegModel, i0_u8: np.ndarray, i1_u8: np.ndarray,
                      img_size: int, device: torch.device) -> np.ndarray:
    H, W = i0_u8.shape
    i0_r = np.array(Image.fromarray(i0_u8).resize((img_size, img_size), Image.BILINEAR))
    i1_r = np.array(Image.fromarray(i1_u8).resize((img_size, img_size), Image.BILINEAR))
    t0 = torch.from_numpy(normalize_image(i0_r)).unsqueeze(0).to(device)
    t1 = torch.from_numpy(normalize_image(i1_r)).unsqueeze(0).to(device)
    logits = model(t0, t1)
    logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    return logits.argmax(dim=1)[0].cpu().numpy().astype(np.int8)


def gt_12_from_hdf5(seg_grp: h5py.Group, layer_idx: int) -> np.ndarray:
    """ORNL HDF5 의 12 boolean masks → 12-class argmax (small ID 가 먼저 그려져 큰 ID 가 덮음)."""
    sample = seg_grp["0"][layer_idx]
    H, W = sample.shape
    out = np.full((H, W), IGNORE_INDEX, dtype=np.int8)
    for c in range(12):
        m = seg_grp[str(c)][layer_idx]
        out[m] = c
    return out


# ============================================================================
# Figure
# ============================================================================
def visualize_comparison(
    i0_u8: np.ndarray, i1_u8: np.ndarray,
    gt_common: np.ndarray,
    preds_common: dict[str, np.ndarray],
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(2, 5, figsize=(28, 12))
    axes = axes.flatten()
    axes[0].imshow(i0_u8, cmap="gray"); axes[0].set_title("visible/0 (after melt)")
    axes[1].imshow(i1_u8, cmap="gray"); axes[1].set_title("visible/1 (after spread)")
    axes[2].imshow(colorize_common(gt_common)); axes[2].set_title("GT (common 8-class)")
    axes[3].imshow(colorize_common(preds_common["v1_s1"])); axes[3].set_title("v1 Stage 1")
    axes[4].imshow(colorize_common(preds_common["v1_s2"])); axes[4].set_title("v1 Stage 2")
    axes[5].imshow(colorize_common(preds_common["v2_s1"])); axes[5].set_title("v2 Stage 1")
    axes[6].imshow(colorize_common(preds_common["v2_s2"])); axes[6].set_title("v2 Stage 2 (fold5)")
    axes[7].imshow(colorize_common(preds_common["v3_s1"])); axes[7].set_title("v3 Stage 1")
    axes[8].imshow(colorize_common(preds_common["v3_s2"])); axes[8].set_title("v3 Stage 2")
    axes[9].axis("off")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    handles = [patches.Patch(color=tuple(PALETTE[c]), label=f"{c} {COMMON_CLASS_NAMES[c]}")
               for c in range(N_COMMON)]
    fig.legend(handles=handles, loc="lower center", ncol=4,
               bbox_to_anchor=(0.5, 0.0), frameon=False)
    fig.suptitle(title, fontsize=14)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Confusion + per-class metric (공통 8-class)
# ============================================================================
def accumulate(cm: np.ndarray, g_common: np.ndarray, p_common: np.ndarray) -> None:
    valid = (g_common != IGNORE_INDEX) & (p_common != IGNORE_INDEX)
    g = g_common[valid].astype(np.int64)
    p = p_common[valid].astype(np.int64)
    idx = g * N_COMMON + p
    cnt = np.bincount(idx, minlength=N_COMMON * N_COMMON)
    cm += cnt.reshape(N_COMMON, N_COMMON)


def per_class_iou(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """returns (iou (N,), support (N,), mIoU)."""
    cm_f = cm.astype(np.float64)
    tp = np.diag(cm_f)
    support = cm_f.sum(axis=1)
    pred_sum = cm_f.sum(axis=0)
    denom = tp + (pred_sum - tp) + (support - tp)
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = np.where(denom > 0, tp / denom, np.nan)
    valid = ~np.isnan(iou) & (support > 0)
    miou = float(np.mean(iou[valid])) if valid.any() else float("nan")
    return iou, support, miou


# ============================================================================
# Main
# ============================================================================
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--build", default="2021-07-13 TCR Phase 1 Build 1")
    parser.add_argument("--n-layers", type=int, default=8,
                        help="정성 figure 용 layer 수")
    parser.add_argument("--n-confusion-layers", type=int, default=200,
                        help="confusion 누적 layer 수 (전체 ~3.2k 중 균등 샘플링)")
    parser.add_argument("--img-size", type=int, default=1036)
    parser.add_argument("--out-root", default=None)
    args = parser.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    # ckpt 6 종 로딩
    REPO_ROOT = Path(__file__).resolve().parents[2]   # = repo root (/.../3DP_VPPM)
    ckpt_dir = REPO_ROOT / "DefSeg_AM" / "checkpoints"
    versions = [
        ("v1_s1", ckpt_dir / "vits14_dpt_dual_sz1036_1gpu_nanfix" / "stage1_best.pt", V1_TO_COMMON),
        ("v1_s2", ckpt_dir / "vits14_dpt_dual_sz1036_1gpu_nanfix" / "stage2_best.pt", V1_TO_COMMON),
        ("v2_s1", ckpt_dir / "vits14_dpt_dual_sz1036_8cls_v2" / "stage1_best.pt", V2_TO_COMMON),
        ("v2_s2", ckpt_dir / "vits14_dpt_dual_sz1036_8cls_v2" / "stage2_best_fold5_v2022_Maraging.pt", V2_TO_COMMON),
        ("v3_s1", ckpt_dir / "vits14_dpt_dual_sz1036_10cls_v3" / "stage1_best.pt", V3_TO_COMMON),
        ("v3_s2", ckpt_dir / "vits14_dpt_dual_sz1036_10cls_v3" / "stage2_best.pt", V3_TO_COMMON),
    ]

    models = {}
    for name, p, m in versions:
        mdl, n_cls, img_size = load_model(p, device)
        img_size = round_to_patch(args.img_size, mdl.patch_size)
        models[name] = {"model": mdl, "mapping": m, "n_cls": n_cls, "img_size": img_size}
        print(f"  loaded {name:7s}  n_cls={n_cls}  img_size={img_size}")

    # HDF5 path
    from DefSeg_AM.v3 import config_v3 as cfg
    hdf5_path = cfg.ORNL_HDF5_DIR / f"{args.build}.hdf5"
    print(f"build: {args.build}  ({hdf5_path})")

    out_root = Path(args.out_root) if args.out_root else \
        REPO_ROOT / "DefSeg_AM" / "figures" / "comparison_v1_v2_v3" / args.build.replace(" ", "_")
    out_root.mkdir(parents=True, exist_ok=True)

    with h5py.File(hdf5_path, "r") as f:
        vis0 = f["slices/camera_data/visible/0"]
        vis1 = f["slices/camera_data/visible/1"]
        seg = f["slices/segmentation_results"]
        n_layers_total = vis0.shape[0]
        print(f"total layers: {n_layers_total}")

        # ----- 1) 정성 figure (n_layers 균등) -----
        lo = int(n_layers_total * 0.1)
        hi = int(n_layers_total * 0.9)
        layer_idxs_fig = np.linspace(lo, hi, args.n_layers, dtype=int).tolist()
        print(f"\n=== 정성 figure layers: {layer_idxs_fig} ===")
        fig_dir = out_root / "comparison"
        for li in layer_idxs_fig:
            i0 = ornl_image_to_uint8(vis0[li])
            i1 = ornl_image_to_uint8(vis1[li])
            gt_12 = gt_12_from_hdf5(seg, li)
            gt_common = remap_to_common(gt_12, ORNL_12_TO_COMMON)

            preds_common = {}
            for name, m in models.items():
                p_native = infer_pred_native(m["model"], i0, i1, m["img_size"], device)
                preds_common[name] = remap_to_common(p_native, m["mapping"])

            out_path = fig_dir / f"layer{li:04d}.png"
            visualize_comparison(i0, i1, gt_common, preds_common, out_path,
                                 title=f"Build 1 layer {li} — v1/v2/v3 비교 (공통 8-class)")
            print(f"  saved {out_path.relative_to(REPO_ROOT)}")

        # ----- 2) Confusion 누적 (n_confusion_layers 균등) -----
        layer_idxs_conf = np.linspace(0, n_layers_total - 1, args.n_confusion_layers, dtype=int).tolist()
        print(f"\n=== Confusion 누적: {len(layer_idxs_conf)} layers ===")
        confs = {name: np.zeros((N_COMMON, N_COMMON), dtype=np.int64) for name in models}

        for i, li in enumerate(layer_idxs_conf):
            if i % 20 == 0:
                print(f"  [{i:3d}/{len(layer_idxs_conf)}]")
            i0 = ornl_image_to_uint8(vis0[li])
            i1 = ornl_image_to_uint8(vis1[li])
            gt_12 = gt_12_from_hdf5(seg, li)
            gt_common = remap_to_common(gt_12, ORNL_12_TO_COMMON)

            for name, m in models.items():
                p_native = infer_pred_native(m["model"], i0, i1, m["img_size"], device)
                p_common = remap_to_common(p_native, m["mapping"])
                accumulate(confs[name], gt_common, p_common)
        print("=== done ===\n")

    # ----- 3) 결과 정리 -----
    results = {}
    print(f"{'model':10s}  {'mIoU':>7s}  | " + "  ".join(f"{c[:5]:>5s}" for c in COMMON_CLASS_NAMES))
    for name, cm in confs.items():
        iou, support, miou = per_class_iou(cm)
        results[name] = {
            "mIoU_common": miou,
            "per_class_iou": [None if np.isnan(x) else float(x) for x in iou],
            "support_px": [int(s) for s in support],
            "confusion_8x8": cm.tolist(),
        }
        print(f"{name:10s}  {miou:>7.4f}  | " +
              "  ".join(f"{(0 if np.isnan(iou[c]) else iou[c]):>5.3f}" for c in range(N_COMMON)))

    with open(out_root / "comparison_metrics.json", "w") as f:
        json.dump({
            "build": args.build,
            "n_confusion_layers": len(layer_idxs_conf),
            "common_class_names": COMMON_CLASS_NAMES,
            "version_mappings": {
                "V1_TO_COMMON": V1_TO_COMMON,
                "V2_TO_COMMON": V2_TO_COMMON,
                "V3_TO_COMMON": V3_TO_COMMON,
            },
            "results": results,
        }, f, indent=2)
    print(f"\nsaved metrics → {out_root / 'comparison_metrics.json'}")


if __name__ == "__main__":
    main()
