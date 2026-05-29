"""학습된 DefSegModel 로 ORNL HDF5 layer 추론 → 4-panel 비교 PNG.

PLAN.md §5.3 / seung_dscnn/infer_ornl.py 포팅.

panel 구성: [visible/0, visible/1, DSCNN GT argmax, our prediction]

사용:
    python -m DefSeg_AM.inference.infer --run-name <run> --build "2021-07-13 TCR Phase 1 Build 1" --stage 2
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import patches
from PIL import Image

from .. import config
from ..data.data_ornl import (
    normalize_image,
    ornl_image_to_uint8,
    ornl_segmentation_argmax,
)
from ..models.model import DefSegModel, round_to_patch


# 12-class 팔레트 (seung_dscnn/infer_ornl.py 와 동일)
_PALETTE = np.array([
    [0.85, 0.85, 0.85],  # 0  Powder
    [0.30, 0.50, 0.85],  # 1  Printed
    [0.95, 0.55, 0.10],  # 2  Recoater Hopping
    [0.20, 0.70, 0.30],  # 3  Recoater Streaking
    [0.85, 0.15, 0.15],  # 4  Incomplete Spreading
    [0.55, 0.30, 0.75],  # 5  Swelling
    [0.50, 0.35, 0.20],  # 6  Debris
    [0.95, 0.40, 0.65],  # 7  Super-Elevation
    [0.40, 0.40, 0.40],  # 8  Spatter
    [0.75, 0.75, 0.10],  # 9  Misprint
    [0.10, 0.70, 0.70],  # 10 Over Melting
    [0.30, 0.30, 0.85],  # 11 Under Melting
])


def colorize(label: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*label.shape, 3), dtype=np.float32)
    for c in range(config.N_CLASSES):
        rgb[label == c] = _PALETTE[c]
    return rgb


def select_default_layers(n_layers: int, n_select: int) -> list[int]:
    lo = int(n_layers * config.ORNL_LAYER_LO_FRAC)
    hi = int(n_layers * config.ORNL_LAYER_HI_FRAC)
    return np.linspace(lo, hi, n_select, dtype=int).tolist()


def resize_inference(
    model: DefSegModel,
    img0_u8: np.ndarray,
    img1_u8: np.ndarray,
    img_size: int = config.IMG_SIZE,
    device: torch.device = torch.device("cuda"),
) -> np.ndarray:
    """원본 해상도 → img_size resize → forward → 원본 해상도로 logits upsample → softmax."""
    H, W = img0_u8.shape
    i0_r = np.array(Image.fromarray(img0_u8).resize((img_size, img_size), Image.BILINEAR))
    i1_r = np.array(Image.fromarray(img1_u8).resize((img_size, img_size), Image.BILINEAR))
    t0 = torch.from_numpy(normalize_image(i0_r)).unsqueeze(0).to(device)
    t1 = torch.from_numpy(normalize_image(i1_r)).unsqueeze(0).to(device)
    model.eval()
    with torch.no_grad():
        logits = model(t0, t1)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logits, dim=1)[0].cpu().numpy()
    return prob.transpose(1, 2, 0)


def visualize(
    img0_u8: np.ndarray,
    img1_u8: np.ndarray,
    gt_label: np.ndarray,
    pred_label: np.ndarray,
    out_path: Path,
    title: str,
) -> None:
    fig, axes = plt.subplots(1, 4, figsize=(22, 6))
    axes[0].imshow(img0_u8, cmap="gray"); axes[0].set_title("visible/0 (after melt)")
    axes[1].imshow(img1_u8, cmap="gray"); axes[1].set_title("visible/1 (after spread)")
    axes[2].imshow(colorize(gt_label));   axes[2].set_title("DSCNN GT (argmax of 12 masks)")
    axes[3].imshow(colorize(pred_label)); axes[3].set_title("our prediction (DefSeg-AM)")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])

    handles = [
        patches.Patch(color=tuple(_PALETTE[c]), label=f"{c} {config.ORNL_CLASS_NAMES[c]}")
        for c in range(config.N_CLASSES)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=6,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


def infer_layers_to_dir(
    model: DefSegModel,
    hdf5_path: Path,
    layers: list[int],
    out_dir: Path,
    img_size: int = config.IMG_SIZE,
    device: torch.device | None = None,
    title_prefix: str = "",
) -> None:
    if device is None:
        device = next(model.parameters()).device
    out_dir.mkdir(parents=True, exist_ok=True)
    with h5py.File(hdf5_path, "r") as f:
        vis0 = f["slices/camera_data/visible/0"]
        vis1 = f["slices/camera_data/visible/1"]
        seg = f["slices/segmentation_results"]
        n_layers = vis0.shape[0]
        for li in layers:
            if li < 0 or li >= n_layers:
                continue
            i0 = ornl_image_to_uint8(vis0[li])
            i1 = ornl_image_to_uint8(vis1[li])
            gt = ornl_segmentation_argmax(seg, li)
            prob = resize_inference(model, i0, i1, img_size=img_size, device=device)
            pred = prob.argmax(axis=-1).astype(np.int8)
            visualize(
                i0, i1, gt, pred,
                out_path=out_dir / f"layer{li:04d}.png",
                title=f"{title_prefix}layer {li}  (infer@{img_size}, vis@{i0.shape[0]})",
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True,
                    help="checkpoints/<run-name>/{stage1,stage2}_best.pt 자동 탐색")
    ap.add_argument("--stage", type=int, default=2, choices=[1, 2],
                    help="어느 단계의 best ckpt 를 쓸지 (default 2)")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="직접 ckpt 경로 (run-name 우회)")
    ap.add_argument("--build", type=str, default="2021-07-13 TCR Phase 1 Build 1",
                    help="HDF5 file basename (without .hdf5)")
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--n-layers", type=int, default=config.N_INFER_LAYERS)
    ap.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    ap.add_argument("--out-dir", type=str, default=None,
                    help="저장 디렉토리. 미지정 시 figures/<run-name>/stage<stage>/inference/<build>/")
    args = ap.parse_args()

    if args.checkpoint is None:
        args.checkpoint = str(
            config.CHECKPOINT_DIR / args.run_name / f"stage{args.stage}_best.pt"
        )

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = DefSegModel(backbone_name=cfg["backbone"]).to(device)
    model.load_trainable_state_dict(ckpt["model_state"])
    print(f"loaded {args.checkpoint} (val_acc={ckpt.get('val_acc', '?')})")

    target_size = cfg.get("img_size", args.img_size)
    img_size = round_to_patch(
        args.img_size if args.img_size != config.IMG_SIZE else target_size,
        model.patch_size,
    )

    hdf5_path = config.ORNL_HDF5_DIR / f"{args.build}.hdf5"
    with h5py.File(hdf5_path, "r") as f:
        n_total = f["slices/camera_data/visible/0"].shape[0]
    layers = args.layers if args.layers is not None else select_default_layers(n_total, args.n_layers)

    if args.out_dir is None:
        out_dir = config.FIGURE_DIR / args.run_name / f"stage{args.stage}" / "inference" / args.build.replace(" ", "_")
    else:
        out_dir = Path(args.out_dir)

    print(f"build {args.build}: n_layers={n_total}, layers={layers}, out={out_dir}")
    infer_layers_to_dir(
        model, hdf5_path, layers, out_dir,
        img_size=img_size, device=device,
        title_prefix=f"{args.build} / stage{args.stage} / ",
    )


if __name__ == "__main__":
    main()
