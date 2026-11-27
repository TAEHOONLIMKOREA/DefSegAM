"""v6 모델로 ORNL HDF5 layer 추론 → 4-panel 비교 PNG (참고용 — PLAN_v6 §6.5).

panel 구성: [visible/0, visible/1, DSCNN 예측 (새 id 로 옮긴 것, 참고용), v6 예측]
세 번째 panel 은 정답이 아니라 DSCNN 의 예측이다 (ORNL 빌드에는 사람 GT 가 없다). 학습 · 길이 · ckpt 선정에 쓰지 않는다.
DSCNN 의 12-class 를 새 id 로 옮길 때 Swelling 은 SE 로, 옛 IS 는 IS 로 보낸다 (옛 IS 는 새 IS 와 PS 가 섞인 것이다). Misprint · Under Melting 은 대응이 없다.

**밝기는 원본 그대로 넣는다.** ORNL HDF5 의 사진 값은 사람 GT 의 사진과 같은 0–255 눈금이다.
v1 캐시와 v4 · v5 의 infer 는 레이어마다 1–99 백분위를 0–255 로 늘렸는데 (`ornl_image_to_uint8`),
사람 GT 만으로 배운 모델에게 그렇게 늘린 사진은 학습 때 본 적 없는 밝기다 — v5 로 잰 값: Build 1 의 Printed 일치도가 늘린 입력 0.56, 원본 0.945.
그래서 v4 · v5 의 같은 그림과 나란히 볼 때는 입력 밝기가 다르다는 것을 함께 적는다.

방향: 사진을 돌리지 않는다 (v1–v5 와 같다). ORNL 빌드는 Peregrine L21 과 같은 기종 (Concept Laser M2) 이지만,
이 HDF5 의 리코팅 축이 이미지 가로인지는 메타데이터로 확인하지 않았다 — 그림에서 RS 가 가로 줄로 보이는지로 확인한다.

(옵션) --tta: flip(LR) × flip(UD) × 180° rot = 8-way 평균.  --postprocess: 후처리 두 규칙 (PLAN_v6 §8).

사용 예 (CWD = 3DP_VPPM/):
    python -m DefSeg_AM.v6.inference.infer --fold inlayer                 # P0 의 last.pt, Build 1, 12 레이어
    python -m DefSeg_AM.v6.inference.infer                                # 전수 학습 모델
    python -m DefSeg_AM.v6.inference.infer --fold inlayer --layers 1500 --build "2021-04-16 TCR Phase 1 Build 2"
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import numpy as np
import torch

from ...common.models.model import DefSegModel, round_to_patch

from .. import config_v6 as cfg
from .evaluate import load_model, predict_pair
from .postprocess import apply_postprocess
from .viz import save_panel

# ORNL HDF5 의 DSCNN 12-class → 옛 11-class (Misprint 9 는 v4 부터 뺐다) → 새 id. 참고 panel 에만 쓴다.
_ORNL12_TO_V4 = {0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7, 8: 8, 9: None, 10: 9, 11: 10}


def _ornl12_to_v6(c12: int) -> int:
    v4 = _ORNL12_TO_V4[c12]
    if v4 is None:
        return cfg.IGNORE_INDEX
    if v4 == cfg.V4_SWELLING_ID:
        return cfg.V4_SWELLING_AS
    new = cfg.V4_TO_NEW[v4]
    return cfg.IGNORE_INDEX if new is None else new


def ornl_raw_uint8(img: np.ndarray) -> np.ndarray:
    """ORNL HDF5 의 사진 (float32, 0–255 눈금) → uint8. 밝기를 늘리지 않는다."""
    if img.dtype == np.uint8:
        return img
    return np.clip(np.rint(img), 0, 255).astype(np.uint8)


def ornl_segmentation_argmax_v6(seg_grp: h5py.Group, layer_idx: int) -> np.ndarray:
    """ORNL HDF5 의 12 boolean mask (DSCNN 예측) → 새 id label (참고용).

    겹친 픽셀은 작은 id 부터 적용 → 결함 (큰 id) 이 Powder/Printed (작은 id) 를 덮어쓴다 (v4 · v5 와 같은 규칙).
    대응이 없는 class (Misprint, Under Melting) 의 픽셀은 덮어쓰지 않는다.

    Returns: (H, W) int8, 값 ∈ {-1} ∪ {0..9}
    """
    sample = seg_grp["0"][layer_idx]
    out = np.full(sample.shape, cfg.IGNORE_INDEX, dtype=np.int8)
    for c_old in range(12):
        new_id = _ornl12_to_v6(c_old)
        if new_id == cfg.IGNORE_INDEX:
            continue
        m = seg_grp[str(c_old)][layer_idx]
        out[m] = new_id
    return out


def select_default_layers(n_layers: int, n_select: int) -> list[int]:
    lo = int(n_layers * cfg.ORNL_LAYER_LO_FRAC)
    hi = int(n_layers * cfg.ORNL_LAYER_HI_FRAC)
    return np.linspace(lo, hi, n_select, dtype=int).tolist()


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
            i0 = ornl_raw_uint8(vis0[li])
            i1 = ornl_raw_uint8(vis1[li])
            ref = ornl_segmentation_argmax_v6(seg, li)

            pred = predict_pair(model, i0, i1, img_size, device, resolutions=("native",), tta=use_tta)["native"]
            pred = pred.cpu().numpy().astype(np.int8)
            if postprocess:
                pred, s = apply_postprocess(pred, verbose=False)
                print(
                    f"  layer {li}: pp se_relabeled={s['se_relabeled']:>7d}  sp_overmelt_relabeled={s['sp_overmelt_relabeled']:>7d}"
                )

            save_panel(
                i0, i1, ref, pred,
                out_path=out_dir / f"layer{li:04d}.png",
                title=f"{title_prefix}layer {li}",
                ref_title="DSCNN prediction → pbf-2026.11 ids (reference, not ground truth)",
                pred_title="our prediction (DefSeg-AM v6, GT-only, raw brightness)",
            )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--backbone", type=str, default=cfg.DINO_BACKBONE)
    ap.add_argument("--fold", type=str, default=None, choices=list(cfg.CV_FOLDS),
                    help="검증 fold 의 ckpt 로 추론 (기본: 전수 학습 모델)")
    ap.add_argument("--which", type=str, default="last", choices=["last", "best", "last_raw"])
    ap.add_argument("--tta", action="store_true",
                    help="TTA: flip(LR) × flip(UD) × rot180 = 8-way 평균")
    ap.add_argument("--postprocess", action="store_true", help="휴리스틱 후처리 2 규칙")
    ap.add_argument("--checkpoint", type=str, default=None, help="직접 ckpt 경로 지정 (v6 10-class)")
    ap.add_argument("--build", type=str, default=cfg.REF_INFER_BUILD)
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
    if not hdf5_path.exists():
        ap.error(f"HDF5 not found: {hdf5_path}")
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

    print(f"build {args.build}: n_layers={n_total}, layers={layers}, out={out_dir}  (입력 밝기: 원본 그대로)")
    title_prefix = (
        f"{args.build} / v6 {cfg.run_tag(args.fold)}"
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
