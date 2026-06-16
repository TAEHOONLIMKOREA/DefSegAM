"""v4 학습된 모델로 ORNL HDF5 layer 추론 → 4-panel 비교 PNG.

panel 구성: [visible/0, visible/1, DSCNN GT argmax (11-class 재매핑), our v4 prediction]

(옵션) Test-Time Augmentation:
  --tta                      : flip(LR) × flip(UD) × 180° rot = 8-way 평균
(옵션) 휴리스틱 후처리:
  --postprocess              : SE/Swelling 이 Printed 에서 멀면 Debris (§5.1)

사용 예:
    # Stage 1 (KD pretrain ckpt) 단독
    python -m DefSeg_AM.v4.inference.infer --run-name <run> --stage 1

    # Stage 2 의 단일 ckpt
    python -m DefSeg_AM.v4.inference.infer --run-name <run> --stage 2 --layers 1500

    # Stage 2 + TTA + 휴리스틱 후처리
    python -m DefSeg_AM.v4.inference.infer --run-name <run> --stage 2 --tta --postprocess
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import h5py
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import patches
from PIL import Image

from ...common.data.image_utils import normalize_image, ornl_image_to_uint8
from ...common.models.model import DefSegModel, round_to_patch

from .. import config_v4 as cfg
from ..data.data_ornl_v4 import ornl_segmentation_argmax_v4
from .postprocess import apply_postprocess


# v4 11-class 팔레트
_PALETTE_V4 = np.array([
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
    [0.75, 0.20, 0.10],  # 10 Under Melting (신규)
])

_IGNORE_COLOR = np.array([0.05, 0.05, 0.05])   # 후처리로 IGNORE 된 픽셀


def colorize(label: np.ndarray) -> np.ndarray:
    """v4 11-class label → RGB. IGNORE_INDEX 픽셀은 별도 어두운 색."""
    rgb = np.zeros((*label.shape, 3), dtype=np.float32)
    for c in range(cfg.N_CLASSES_V4):
        rgb[label == c] = _PALETTE_V4[c]
    rgb[label == cfg.IGNORE_INDEX] = _IGNORE_COLOR
    return rgb


def select_default_layers(n_layers: int, n_select: int) -> list[int]:
    lo = int(n_layers * cfg.ORNL_LAYER_LO_FRAC)
    hi = int(n_layers * cfg.ORNL_LAYER_HI_FRAC)
    return np.linspace(lo, hi, n_select, dtype=int).tolist()


# ---------------------------------------------------------------------------
# 추론 helper — TTA / ensemble 모두 softmax(=prob) level 에서 평균
# ---------------------------------------------------------------------------

def _forward_prob(
    model: DefSegModel,
    i0_u8: np.ndarray, i1_u8: np.ndarray,
    img_size: int, device: torch.device,
) -> np.ndarray:
    """단일 forward → (H, W, C) softmax prob."""
    H, W = i0_u8.shape
    i0_r = np.array(Image.fromarray(i0_u8).resize((img_size, img_size), Image.BILINEAR))
    i1_r = np.array(Image.fromarray(i1_u8).resize((img_size, img_size), Image.BILINEAR))
    t0 = torch.from_numpy(normalize_image(i0_r)).unsqueeze(0).to(device)
    t1 = torch.from_numpy(normalize_image(i1_r)).unsqueeze(0).to(device)
    with torch.no_grad():
        logits = model(t0, t1)
        logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
        prob = F.softmax(logits, dim=1)[0].cpu().numpy()
    return prob.transpose(1, 2, 0)


def _tta_v4_prob(
    model: DefSegModel,
    i0_u8: np.ndarray, i1_u8: np.ndarray,
    img_size: int, device: torch.device,
) -> np.ndarray:
    """v4 학습 augmentation 일치 TTA: flip(LR) × flip(UD) × rot180 = 8-way 평균."""
    probs = []
    for flip_lr, flip_ud, rot180 in product([False, True], repeat=3):
        i0 = i0_u8
        i1 = i1_u8
        if flip_lr:
            i0 = i0[:, ::-1].copy(); i1 = i1[:, ::-1].copy()
        if flip_ud:
            i0 = i0[::-1, :].copy(); i1 = i1[::-1, :].copy()
        if rot180:
            i0 = np.rot90(i0, k=2).copy(); i1 = np.rot90(i1, k=2).copy()

        p = _forward_prob(model, i0, i1, img_size, device)

        # 역변환 (적용 역순)
        if rot180:
            p = np.rot90(p, k=-2, axes=(0, 1)).copy()
        if flip_ud:
            p = p[::-1, :, :].copy()
        if flip_lr:
            p = p[:, ::-1, :].copy()
        probs.append(p)
    return np.mean(probs, axis=0)


def infer_layers_to_dir(
    model: DefSegModel,
    hdf5_path: Path,
    layers: list[int],
    out_dir: Path,
    img_size: int,
    device: torch.device,
    title_prefix: str = "",
    use_tta: bool = False,
    postprocess: bool = False,
) -> None:
    """단일 모델로 layer 별 추론.

    Args:
        use_tta: True 면 flip LR×UD×rot180 = 8-way TTA
        postprocess: True 면 §5 의 두 규칙 적용
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    model.eval()
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
            gt = ornl_segmentation_argmax_v4(seg, li)

            if use_tta:
                prob = _tta_v4_prob(model, i0, i1, img_size, device)
            else:
                prob = _forward_prob(model, i0, i1, img_size, device)
            pred = prob.argmax(axis=-1).astype(np.int8)

            if postprocess:
                pred, s = apply_postprocess(pred, i0, i1, verbose=False)
                print(
                    f"  layer {li}: pp se_relabeled={s['se_relabeled']:>7d}  ps_overmelt_relabeled={s['ps_overmelt_relabeled']:>7d}"
                )

            visualize(
                i0, i1, gt, pred,
                out_path=out_dir / f"layer{li:04d}.png",
                title=f"{title_prefix}layer {li}",
            )


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
    axes[2].imshow(colorize(gt_label));   axes[2].set_title("DSCNN GT (v4 11-class argmax)")
    axes[3].imshow(colorize(pred_label)); axes[3].set_title("our prediction (DefSeg-AM v4)")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    handles = [
        patches.Patch(color=tuple(_PALETTE_V4[c]),
                      label=f"{c} {cfg.ORNL_CLASS_NAMES_V4[c]}")
        for c in range(cfg.N_CLASSES_V4)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=5,
               bbox_to_anchor=(0.5, -0.02), frameon=False)
    fig.suptitle(title, fontsize=12)
    fig.tight_layout(rect=[0, 0.05, 1, 0.97])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)
    print(f"saved {out_path}")


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def _load_model_from_ckpt(ckpt_path: Path, device: torch.device) -> DefSegModel:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ckpt["config"]
    model = DefSegModel(
        backbone_name=c["backbone"],
        n_classes=c.get("n_classes", cfg.N_CLASSES_V4),
    ).to(device)
    model.load_trainable_state_dict(ckpt["model_state"])
    print(f"loaded {ckpt_path}  (val_acc={ckpt.get('val_acc', '?')})")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True,
                    help="checkpoints/<run-name>/{stage1_best, stage2_best}.pt")
    ap.add_argument("--stage", type=int, default=2, choices=[1, 2])
    ap.add_argument("--tta", action="store_true",
                    help="TTA: flip(LR) × flip(UD) × rot180 = 8-way 평균")
    ap.add_argument("--postprocess", action="store_true",
                    help="휴리스틱 후처리: 부품에서 멀리 떨어진 SE/Swelling → Debris (§5.1)")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="직접 ckpt 경로 지정")
    ap.add_argument("--build", type=str, default="2021-07-13 TCR Phase 1 Build 1")
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--n-layers", type=int, default=cfg.N_INFER_LAYERS)
    ap.add_argument("--img-size", type=int, default=cfg.IMG_SIZE)
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = cfg.CHECKPOINT_DIR / args.run_name

    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
    elif args.stage == 1:
        ckpt_path = run_dir / "stage1_best.pt"
    else:
        ckpt_path = run_dir / "stage2_best.pt"
    if not ckpt_path.exists():
        ap.error(f"ckpt not found: {ckpt_path}")

    model = _load_model_from_ckpt(ckpt_path, device)
    img_size = round_to_patch(args.img_size, model.patch_size)

    hdf5_path = cfg.ORNL_HDF5_DIR / f"{args.build}.hdf5"
    with h5py.File(hdf5_path, "r") as f:
        n_total = f["slices/camera_data/visible/0"].shape[0]
    layers = args.layers if args.layers is not None else select_default_layers(n_total, args.n_layers)

    if args.out_dir is None:
        tag = (
            f"stage{args.stage}"
            + ("_tta" if args.tta else "")
            + ("_pp" if args.postprocess else "")
        )
        out_dir = cfg.FIGURE_DIR / args.run_name / "v4" / tag / "inference" / args.build.replace(" ", "_")
    else:
        out_dir = Path(args.out_dir)

    print(f"build {args.build}: n_layers={n_total}, layers={layers}, out={out_dir}")
    title_prefix = (
        f"{args.build} / v4-stage{args.stage}"
        f"{' / TTA' if args.tta else ''}"
        f"{' / PP' if args.postprocess else ''} / "
    )
    infer_layers_to_dir(
        model, hdf5_path, layers, out_dir,
        img_size=img_size, device=device,
        title_prefix=title_prefix, use_tta=args.tta,
        postprocess=args.postprocess,
    )


if __name__ == "__main__":
    main()
