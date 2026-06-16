"""v2 학습된 모델로 ORNL HDF5 layer 추론 → 4-panel 비교 PNG.

panel 구성: [visible/0, visible/1, DSCNN GT argmax (8-class 재매핑), our v2 prediction]

8-fold cross-validation 대응:
  --stage 2 --fold k        : 특정 fold ckpt 단독 추론
  --stage 2 --ensemble       : 8 fold softmax 평균 + argmax (cross-val ensemble)

(옵션) Test-Time Augmentation:
  --tta                      : D4 group 8 변환 softmax 평균

사용 예:
    # Stage 1 (KD pretrain ckpt) 단독
    python -m DefSeg_AM.v2.inference.infer --run-name <run> --stage 1

    # Stage 2 의 fold 0 ckpt 로 한 layer
    python -m DefSeg_AM.v2.inference.infer --run-name <run> --stage 2 --fold 0 --layers 1500

    # Stage 2 의 8 fold ensemble + TTA + 휴리스틱 후처리
    python -m DefSeg_AM.v2.inference.infer --run-name <run> --stage 2 --ensemble --tta --postprocess
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

# 공통 함수 재사용 (common)
from ...common.data.image_utils import normalize_image, ornl_image_to_uint8
from ...common.models.model import DefSegModel, round_to_patch

# v2
from .. import config_v2 as cfg
from ..data.data_ornl_v2 import ornl_segmentation_argmax_v2
from .postprocess import apply_postprocess


# v2 8-class 팔레트 (v1 12-class 의 일부 색상 보존)
_PALETTE_V2 = np.array([
    [0.85, 0.85, 0.85],  # 0 Powder
    [0.30, 0.50, 0.85],  # 1 Printed
    [0.95, 0.55, 0.10],  # 2 Recoater Disturbance (old 2 Hopping 색)
    [0.55, 0.30, 0.75],  # 3 Swelling             (old 5)
    [0.40, 0.40, 0.40],  # 4 Spatter              (old 8)
    [0.95, 0.40, 0.65],  # 5 Super-Elevation      (old 7)
    [0.10, 0.70, 0.70],  # 6 Over Melting         (old 10)
    [0.50, 0.35, 0.20],  # 7 Debris               (old 6)
])


_IGNORE_COLOR = np.array([0.05, 0.05, 0.05])   # 후처리로 IGNORE 된 픽셀


def colorize(label: np.ndarray) -> np.ndarray:
    """v2 8-class label → RGB. IGNORE_INDEX 픽셀은 별도 어두운 색."""
    rgb = np.zeros((*label.shape, 3), dtype=np.float32)
    for c in range(cfg.N_CLASSES_V2):
        rgb[label == c] = _PALETTE_V2[c]
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


def _tta_d4_prob(
    model: DefSegModel,
    i0_u8: np.ndarray, i1_u8: np.ndarray,
    img_size: int, device: torch.device,
) -> np.ndarray:
    """D4 group (4 rotations × 2 flips) 변환 후 평균. 각 변환의 역변환 적용 후 평균."""
    probs = []
    for k in range(4):
        for flip in (False, True):
            i0 = np.rot90(i0_u8, k=k).copy()
            i1 = np.rot90(i1_u8, k=k).copy()
            if flip:
                i0 = i0[:, ::-1].copy()
                i1 = i1[:, ::-1].copy()
            p = _forward_prob(model, i0, i1, img_size, device)  # (H,W,C) on aug
            # 역변환: 먼저 flip 다시 (if applied)
            if flip:
                p = p[:, ::-1, :].copy()
            # 그 다음 rotation 역방향 (np.rot90(k=-k) on H,W axes)
            if k > 0:
                p = np.rot90(p, k=-k, axes=(0, 1)).copy()
            probs.append(p)
    return np.mean(probs, axis=0)


def infer_layers_to_dir(
    models: list[DefSegModel],
    hdf5_path: Path,
    layers: list[int],
    out_dir: Path,
    img_size: int,
    device: torch.device,
    title_prefix: str = "",
    use_tta: bool = False,
    postprocess: bool = False,
) -> None:
    """주어진 models (1 개 또는 ensemble) 로 layer 별 추론.

    Args:
        models: ckpt 마다 1 개 model. len(models) > 1 이면 softmax 평균 → ensemble.
        use_tta: True 면 각 model 에 대해 D4 8 변환 softmax 평균
        postprocess: True 면 휴리스틱 후처리 (정적+ROI외 IGNORE / 부품에서 먼
            SE·Swelling → Debris) 적용. v2/inference/postprocess.py 참조.
    """
    out_dir.mkdir(parents=True, exist_ok=True)
    for m in models:
        m.eval()
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
            gt = ornl_segmentation_argmax_v2(seg, li)

            # ensemble over models + (optional) TTA per model
            probs = []
            for m in models:
                if use_tta:
                    probs.append(_tta_d4_prob(m, i0, i1, img_size, device))
                else:
                    probs.append(_forward_prob(m, i0, i1, img_size, device))
            prob = np.mean(probs, axis=0)
            pred = prob.argmax(axis=-1).astype(np.int8)

            if postprocess:
                pred, pp_stats = apply_postprocess(pred, i0, i1, verbose=False)
                print(
                    f"  layer {li}: pp static_ignored={pp_stats['static_ignored']:>7d}  "
                    f"se_swelling_relabeled={pp_stats['se_swelling_relabeled']:>7d}"
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
    axes[2].imshow(colorize(gt_label));   axes[2].set_title("DSCNN GT (v2 8-class argmax)")
    axes[3].imshow(colorize(pred_label)); axes[3].set_title("our prediction (DefSeg-AM v2)")
    for ax in axes:
        ax.set_xticks([]); ax.set_yticks([])
    handles = [
        patches.Patch(color=tuple(_PALETTE_V2[c]),
                      label=f"{c} {cfg.ORNL_CLASS_NAMES_V2[c]}")
        for c in range(cfg.N_CLASSES_V2)
    ]
    fig.legend(handles=handles, loc="lower center", ncol=4,
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
        n_classes=c.get("n_classes", cfg.N_CLASSES_V2),
    ).to(device)
    model.load_trainable_state_dict(ckpt["model_state"])
    print(f"loaded {ckpt_path}  (val_acc={ckpt.get('val_acc', '?')})")
    return model


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, required=True,
                    help="checkpoints/<run-name>/{stage1_best, stage2_best_fold*}.pt 자동 탐색")
    ap.add_argument("--stage", type=int, default=2, choices=[1, 2])
    ap.add_argument("--fold", type=int, default=None,
                    help="stage=2 단일 fold 추론 시 fold id (0..7). --ensemble 와 상호 배타")
    ap.add_argument("--ensemble", action="store_true",
                    help="stage=2 시 8 fold ckpt 의 softmax 평균. --fold 와 상호 배타.")
    ap.add_argument("--tta", action="store_true",
                    help="Test-Time Augmentation: D4 group 8 변환 softmax 평균.")
    ap.add_argument("--postprocess", action="store_true",
                    help="휴리스틱 후처리 적용: "
                         "(1) 정적 + powder ROI 밖 → IGNORE, "
                         "(2) 부품에서 멀리 떨어진 SE/Swelling → Debris.")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="직접 ckpt 경로 지정 (run-name/stage/fold 무시).")
    ap.add_argument("--build", type=str, default="2021-07-13 TCR Phase 1 Build 1")
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--n-layers", type=int, default=cfg.N_INFER_LAYERS)
    ap.add_argument("--img-size", type=int, default=cfg.IMG_SIZE)
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    if args.fold is not None and args.ensemble:
        ap.error("--fold 와 --ensemble 동시 사용 불가")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = cfg.CHECKPOINT_DIR / args.run_name

    # ----- ckpt path 결정 -----
    ckpt_paths: list[Path] = []
    if args.checkpoint is not None:
        ckpt_paths = [Path(args.checkpoint)]
    elif args.stage == 1:
        ckpt_paths = [run_dir / "stage1_best.pt"]
    else:  # stage 2
        if args.ensemble:
            # stage2_best_fold{k}_*.pt 8 개 모두 수집
            ckpt_paths = sorted(run_dir.glob("stage2_best_fold*.pt"))
            if not ckpt_paths:
                ap.error(f"no stage2 ckpt found in {run_dir}")
            print(f"ensemble of {len(ckpt_paths)} fold ckpts")
        else:
            fold = args.fold if args.fold is not None else 0
            # stage2_best_fold{fold}_*.pt 매칭
            matches = sorted(run_dir.glob(f"stage2_best_fold{fold}_*.pt"))
            if not matches:
                ap.error(f"no stage2 fold {fold} ckpt found in {run_dir}")
            ckpt_paths = [matches[0]]

    # ----- 모델 로드 -----
    models = [_load_model_from_ckpt(p, device) for p in ckpt_paths]
    img_size = round_to_patch(args.img_size, models[0].patch_size)

    # ----- 추론 layer 선택 -----
    hdf5_path = cfg.ORNL_HDF5_DIR / f"{args.build}.hdf5"
    with h5py.File(hdf5_path, "r") as f:
        n_total = f["slices/camera_data/visible/0"].shape[0]
    layers = args.layers if args.layers is not None else select_default_layers(n_total, args.n_layers)

    # ----- 출력 디렉터리 -----
    if args.out_dir is None:
        sub = "ensemble" if args.ensemble else (f"fold{args.fold}" if args.fold is not None else "single")
        tag = (
            f"stage{args.stage}_{sub}"
            + ("_tta" if args.tta else "")
            + ("_pp" if args.postprocess else "")
        )
        out_dir = cfg.FIGURE_DIR / args.run_name / "v2" / tag / "inference" / args.build.replace(" ", "_")
    else:
        out_dir = Path(args.out_dir)

    print(f"build {args.build}: n_layers={n_total}, layers={layers}, out={out_dir}")
    title_prefix = (
        f"{args.build} / v2-stage{args.stage} / "
        f"{'ensemble' if args.ensemble else ('fold'+str(args.fold) if args.fold is not None else 'single')}"
        f"{' / TTA' if args.tta else ''}"
        f"{' / PP' if args.postprocess else ''} / "
    )
    infer_layers_to_dir(
        models, hdf5_path, layers, out_dir,
        img_size=img_size, device=device,
        title_prefix=title_prefix, use_tta=args.tta,
        postprocess=args.postprocess,
    )


if __name__ == "__main__":
    main()
