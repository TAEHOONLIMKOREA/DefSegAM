"""v1 / v3 / v4 Stage 2 ckpt 를 ORNL 의 여러 build 에서 추론 → 11-class 통합 비교.

v1 (12 native) / v3 (10) / v4 (11) 의 클래스 ID 가 거의 동일:
- v1: 12 native (Misprint=9 만 v4 에서 IGNORE)
- v3: 10-class — 0..9 동일 (Under Melting 없음)
- v4: 11-class — 0..10 (Under Melting 신규)

GT 도 ORNL 12-class native 를 동일 매핑.

사용:
    python -m DefSeg_AM.scripts.compare_v3_v4_multi_build \\
        [--builds B1 B5] [--n-fig-layers 6] [--n-conf-layers 150]
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


# v4 11-class — 통합 어휘
CLASS_NAMES = [
    "Powder",                # 0
    "Printed",               # 1
    "Recoater Hopping",      # 2
    "Recoater Streaking",    # 3
    "Incomplete Spreading",  # 4
    "Swelling",              # 5
    "Debris",                # 6
    "Super-Elevation",       # 7
    "Spatter",               # 8
    "Over Melting",          # 9
    "Under Melting",         # 10
]
N_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = -1

# v1 ORNL 12-class → 11-class (Misprint=9 → IGNORE)
ORNL_12_TO_11 = {
    0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8,
    9: -1,        # Misprint
    10: 9,        # Over Melting
    11: 10,       # Under Melting
}

# v1 의 12-class → 11-class (Misprint=9 → IGNORE, 그 외 ORNL_12_TO_11 와 동일)
V1_TO_11 = ORNL_12_TO_11
# v3 의 10-class → 11-class (그대로, v3 는 Under Melting 안 예측)
V3_TO_11 = {i: i for i in range(10)}
# v4 의 11-class → 11-class
V4_TO_11 = {i: i for i in range(11)}

PALETTE = np.array([
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
IGNORE_COLOR = np.array([0.05, 0.05, 0.05])

# Build → HDF5 파일명
BUILD_TO_HDF5 = {
    "B1": "2021-07-13 TCR Phase 1 Build 1",
    "B2": "2021-04-16 TCR Phase 1 Build 2",
    "B3": "2021-04-28 TCR Phase 1 Build 3",
    "B4": "2021-08-03 TCR Phase 1 Build 4",
    "B5": "2021-08-23 TCR Phase 1 Build 5",
}


def remap_to_11(label_native: np.ndarray, mapping: dict[int, int]) -> np.ndarray:
    out = np.full_like(label_native, fill_value=IGNORE_INDEX, dtype=np.int8)
    for src, dst in mapping.items():
        if dst == IGNORE_INDEX:
            continue
        out[label_native == src] = dst
    return out


def colorize(label: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*label.shape, 3), dtype=np.float32)
    for c in range(N_CLASSES):
        rgb[label == c] = PALETTE[c]
    rgb[label == IGNORE_INDEX] = IGNORE_COLOR
    return rgb


def load_model(path: Path, device: torch.device) -> tuple[DefSegModel, int, int]:
    ckpt = torch.load(path, map_location=device, weights_only=False)
    c = ckpt["config"]
    model = DefSegModel(backbone_name=c["backbone"], n_classes=c["n_classes"]).to(device)
    model.load_trainable_state_dict(ckpt["model_state"])
    model.eval()
    return model, int(c["n_classes"]), int(c.get("img_size", 1036))


@torch.no_grad()
def infer_pred(model: DefSegModel, i0_u8: np.ndarray, i1_u8: np.ndarray,
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
    sample = seg_grp["0"][layer_idx]
    H, W = sample.shape
    out = np.full((H, W), IGNORE_INDEX, dtype=np.int8)
    for c in range(12):
        m = seg_grp[str(c)][layer_idx]
        out[m] = c
    return out


def accumulate(cm: np.ndarray, gt: np.ndarray, pred: np.ndarray) -> None:
    valid = (gt != IGNORE_INDEX) & (pred != IGNORE_INDEX)
    g = gt[valid].astype(np.int64)
    p = pred[valid].astype(np.int64)
    idx = g * N_CLASSES + p
    cnt = np.bincount(idx, minlength=N_CLASSES * N_CLASSES)
    cm += cnt.reshape(N_CLASSES, N_CLASSES)


def per_class_iou(cm: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
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


def visualize(i0, i1, gt, preds: dict, out_path, title):
    """preds: {"v1": np.ndarray, "v3": ..., "v4": ...} — 11-class 매핑된 prediction"""
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    axes = axes.flatten()
    axes[0].imshow(i0, cmap="gray"); axes[0].set_title("visible/0 (after melt)")
    axes[1].imshow(i1, cmap="gray"); axes[1].set_title("visible/1 (after spread)")
    axes[2].imshow(colorize(gt));    axes[2].set_title("GT (11-class)")
    axes[3].imshow(colorize(preds["v1"])); axes[3].set_title("v1 Stage 2 (vits14, 12-cls)")
    axes[4].imshow(colorize(preds["v3"])); axes[4].set_title("v3 Stage 2 (vits14, 10-cls)")
    axes[5].imshow(colorize(preds["v4"])); axes[5].set_title("v4 Stage 2 (vitb14_reg, 11-cls)")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    handles = [patches.Patch(color=tuple(PALETTE[c]), label=f"{c} {CLASS_NAMES[c]}")
               for c in range(N_CLASSES)]
    fig.legend(handles=handles, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.01), frameon=False)
    fig.suptitle(title, fontsize=13)
    fig.tight_layout(rect=[0, 0.04, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--builds", nargs="+", default=["B1", "B5"], choices=list(BUILD_TO_HDF5))
    ap.add_argument("--n-fig-layers", type=int, default=8)
    ap.add_argument("--n-conf-layers", type=int, default=200)
    ap.add_argument("--img-size", type=int, default=1036)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    REPO = Path(__file__).resolve().parents[2]
    ckpts = {
        "v1": REPO / "DefSeg_AM/checkpoints/vits14_dpt_dual_sz1036_1gpu_nanfix/stage2_best.pt",
        "v3": REPO / "DefSeg_AM/checkpoints/vits14_dpt_dual_sz1036_10cls_v3/stage2_best.pt",
        "v4": REPO / "DefSeg_AM/checkpoints/vitb14_reg_dpt_dual_sz1036_11cls_v4/stage2_best.pt",
    }
    mappings = {"v1": V1_TO_11, "v3": V3_TO_11, "v4": V4_TO_11}

    models = {}
    for name, p in ckpts.items():
        mdl, n_cls, img_size = load_model(p, device)
        img_size = round_to_patch(args.img_size, mdl.patch_size)
        models[name] = {"model": mdl, "n_cls": n_cls, "img_size": img_size, "mapping": mappings[name]}
        print(f"  loaded {name}: n_cls={n_cls} img_size={img_size}")

    out_root = REPO / "DefSeg_AM" / "figures" / "compare_v3_v4_multi_build"
    out_root.mkdir(parents=True, exist_ok=True)

    HDF5_DIR = REPO / "ORNL_Data/Co-Registered In-Situ and Ex-Situ Dataset/[baseline] (Peregrine v2023-11)"

    results: dict[str, dict] = {}

    for build_key in args.builds:
        build_name = BUILD_TO_HDF5[build_key]
        hdf5_path = HDF5_DIR / f"{build_name}.hdf5"
        print(f"\n===== {build_key} = {build_name} =====")
        confs = {name: np.zeros((N_CLASSES, N_CLASSES), dtype=np.int64) for name in models}

        build_fig_dir = out_root / build_key / "figures"

        with h5py.File(hdf5_path, "r") as f:
            vis0 = f["slices/camera_data/visible/0"]
            vis1 = f["slices/camera_data/visible/1"]
            seg = f["slices/segmentation_results"]
            n_total = vis0.shape[0]
            print(f"  total layers: {n_total}")

            # 정성 figure 용 layer (10%-90% 균등 sampling)
            lo, hi = int(n_total * 0.1), int(n_total * 0.9)
            fig_idxs = np.linspace(lo, hi, args.n_fig_layers, dtype=int).tolist()
            print(f"  figure layers: {fig_idxs}")
            for li in fig_idxs:
                i0 = ornl_image_to_uint8(vis0[li])
                i1 = ornl_image_to_uint8(vis1[li])
                gt12 = gt_12_from_hdf5(seg, li)
                gt11 = remap_to_11(gt12, ORNL_12_TO_11)
                preds = {}
                for name, m in models.items():
                    p_native = infer_pred(m["model"], i0, i1, m["img_size"], device)
                    preds[name] = remap_to_11(p_native, m["mapping"])
                visualize(i0, i1, gt11, preds,
                          build_fig_dir / f"layer{li:04d}.png",
                          title=f"{build_name} layer {li} — v1 / v3 / v4")
                print(f"  saved {build_fig_dir.relative_to(REPO)}/layer{li:04d}.png")

            # confusion 누적
            conf_idxs = np.linspace(0, n_total - 1, args.n_conf_layers, dtype=int).tolist()
            print(f"  confusion 누적: {len(conf_idxs)} layers")
            for i, li in enumerate(conf_idxs):
                if i % 20 == 0:
                    print(f"    [{i:3d}/{len(conf_idxs)}]")
                i0 = ornl_image_to_uint8(vis0[li])
                i1 = ornl_image_to_uint8(vis1[li])
                gt12 = gt_12_from_hdf5(seg, li)
                gt11 = remap_to_11(gt12, ORNL_12_TO_11)
                for name, m in models.items():
                    p_native = infer_pred(m["model"], i0, i1, m["img_size"], device)
                    p11 = remap_to_11(p_native, m["mapping"])
                    accumulate(confs[name], gt11, p11)

        # 정량 결과
        build_results = {}
        print(f"\n  {'model':6s}  {'mIoU':>7s}  | " +
              "  ".join(f"{c[:5]:>5s}" for c in CLASS_NAMES))
        for name, cm in confs.items():
            iou, support, miou = per_class_iou(cm)
            build_results[name] = {
                "mIoU": miou,
                "per_class_iou": [None if np.isnan(x) else float(x) for x in iou],
                "support_px": [int(s) for s in support],
                "confusion_11x11": cm.tolist(),
            }
            print(f"  {name:6s}  {miou:>7.4f}  | " +
                  "  ".join(f"{(0 if np.isnan(iou[c]) else iou[c]):>5.3f}" for c in range(N_CLASSES)))
        results[build_key] = build_results

    out_path = out_root / "metrics.json"
    with open(out_path, "w") as f:
        json.dump({
            "builds": args.builds,
            "n_conf_layers": args.n_conf_layers,
            "class_names": CLASS_NAMES,
            "results": results,
        }, f, indent=2)
    print(f"\nsaved metrics → {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
