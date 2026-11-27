"""v5 학습된 모델로 ORNL HDF5 layer 추론 → 4-panel 비교 PNG.

panel 구성: [visible/0, visible/1, DSCNN 예측 argmax (11-class 재매핑, 참고용), our v5 prediction]
v4 의 infer 와 같은 배치 · 같은 팔레트라 나란히 볼 수 있다. 세 번째 panel 은 정답이 아니라
DSCNN 의 예측이다 (ORNL 빌드에는 사람 GT 가 없다).

(옵션) Test-Time Augmentation:
  --tta                      : flip(LR) × flip(UD) × 180° rot = 8-way 평균
(옵션) 휴리스틱 후처리:
  --postprocess              : v4 의 두 규칙

사용 예 (CWD = 3DP_VPPM/):
    python -m DefSeg_AM.v5.inference.infer                              # 전수 학습 last.pt, Build 1, 12 레이어
    python -m DefSeg_AM.v5.inference.infer --tta --postprocess
    python -m DefSeg_AM.v5.inference.infer --layers 1500 --build "2021-04-16 TCR Phase 1 Build 2"
"""
from __future__ import annotations

import argparse
from itertools import product
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

from ...common.data.image_utils import normalize_image, ornl_image_to_uint8
from ...common.models.model import DefSegModel, round_to_patch

from .. import config_v5 as cfg
from ..data.data_ornl_v5 import ornl_segmentation_argmax_v5
from .evaluate_gt import load_model
from .postprocess import apply_postprocess
from .viz import save_panel


def select_default_layers(n_layers: int, n_select: int) -> list[int]:
    lo = int(n_layers * cfg.ORNL_LAYER_LO_FRAC)
    hi = int(n_layers * cfg.ORNL_LAYER_HI_FRAC)
    return np.linspace(lo, hi, n_select, dtype=int).tolist()


# ---------------------------------------------------------------------------
# 추론 helper — TTA 는 softmax(=prob) level 에서 평균 (v4 와 같음)
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


def _tta_prob(
    model: DefSegModel,
    i0_u8: np.ndarray, i1_u8: np.ndarray,
    img_size: int, device: torch.device,
) -> np.ndarray:
    """학습 augmentation 일치 TTA: flip(LR) × flip(UD) × rot180 = 8-way 평균."""
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
            ref = ornl_segmentation_argmax_v5(seg, li)

            if use_tta:
                prob = _tta_prob(model, i0, i1, img_size, device)
            else:
                prob = _forward_prob(model, i0, i1, img_size, device)
            pred = prob.argmax(axis=-1).astype(np.int8)

            if postprocess:
                pred, s = apply_postprocess(pred, i0, i1, verbose=False)
                print(
                    f"  layer {li}: pp se_relabeled={s['se_relabeled']:>7d}  ps_overmelt_relabeled={s['ps_overmelt_relabeled']:>7d}"
                )

            save_panel(
                i0, i1, ref, pred,
                out_path=out_dir / f"layer{li:04d}.png",
                title=f"{title_prefix}layer {li}",
                ref_title="DSCNN prediction (11-class argmax, reference)",
                pred_title="our prediction (DefSeg-AM v5, GT-only)",
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--backbone", type=str, default=cfg.DINO_BACKBONE)
    ap.add_argument("--fold", type=str, default=None, choices=sorted(cfg.CV_FOLDS),
                    help="검증 fold 의 ckpt 로 추론 (기본: 전수 학습 모델)")
    ap.add_argument("--which", type=str, default="last", choices=["last", "best", "last_raw"])
    ap.add_argument("--tta", action="store_true",
                    help="TTA: flip(LR) × flip(UD) × rot180 = 8-way 평균")
    ap.add_argument("--postprocess", action="store_true", help="휴리스틱 후처리 2 규칙")
    ap.add_argument("--checkpoint", type=str, default=None, help="직접 ckpt 경로 지정")
    ap.add_argument("--build", type=str, default=cfg.REF_EVAL_BUILD)
    ap.add_argument("--layers", type=int, nargs="+", default=None)
    ap.add_argument("--n-layers", type=int, default=cfg.N_INFER_LAYERS)
    ap.add_argument("--img-size", type=int, default=cfg.IMG_SIZE)
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or cfg.default_run_name(args.backbone, args.img_size)

    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = cfg.run_ckpt_dir(run_name, args.fold) / f"{args.which}.pt"
    if not ckpt_path.exists():
        ap.error(f"ckpt not found: {ckpt_path}")

    model, _ = load_model(ckpt_path, device)
    img_size = round_to_patch(args.img_size, model.patch_size)

    hdf5_path = cfg.ORNL_HDF5_DIR / f"{args.build}.hdf5"
    with h5py.File(hdf5_path, "r") as f:
        n_total = f["slices/camera_data/visible/0"].shape[0]
    layers = args.layers if args.layers is not None else select_default_layers(n_total, args.n_layers)

    if args.out_dir is None:
        tag = (
            f"inference_{args.which}"
            + ("_tta" if args.tta else "")
            + ("_pp" if args.postprocess else "")
        )
        out_dir = cfg.run_fig_dir(run_name, args.fold) / tag / args.build.replace(" ", "_")
    else:
        out_dir = Path(args.out_dir)

    print(f"build {args.build}: n_layers={n_total}, layers={layers}, out={out_dir}")
    title_prefix = (
        f"{args.build} / v5 {cfg.run_tag(args.fold)}"
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
