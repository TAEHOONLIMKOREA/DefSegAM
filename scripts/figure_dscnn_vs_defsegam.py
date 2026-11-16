"""DSCNN (ORNL baseline) vs DefSeg-AM v4 정성 비교 figure.

여러 layer 를 한 figure 에 5-row × 4-col 로 배치:
  col 0: visible/0 (after melt)
  col 1: visible/1 (after spread)
  col 2: DSCNN (ORNL baseline pred, HDF5 의 segmentation_results argmax)
  col 3: DefSeg-AM v4 (11-class prediction)

Palette 는 배경 (Powder/Printed) 은 낮은 톤, 결함 9종은 검증된 categorical
hue + cyan 으로 서로 확실히 대비되도록 선정. 배경 이미지와 겹칠 때 결함이
도드라지는 것이 목표.

사용:
    python -m DefSeg_AM.scripts.figure_dscnn_vs_defsegam \
        [--build B2] [--layers 330 525 619 2054 3095]
"""
from __future__ import annotations

import argparse
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
# Palette — 배경 톤 낮춤 + 결함 색 대비 강조
# ============================================================================
CLASS_NAMES = [
    "Powder", "Printed", "Recoater Hopping", "Recoater Streaking",
    "Incomplete Spreading", "Swelling", "Debris", "Super-Elevation",
    "Spatter", "Over Melting", "Under Melting",
]
N_CLASSES = len(CLASS_NAMES)
IGNORE_INDEX = -1

# 배경 (톤 낮춤 — 결함이 도드라지게)
PALETTE = np.array([
    [0.910, 0.902, 0.875],   # 0 Powder      — 연회색 #e8e6df
    [0.415, 0.408, 0.384],   # 1 Printed     — 진회색 #6a6862

    # 결함 9종 (dataviz skill 의 검증 8-slot categorical + 도메인 cyan)
    [0.165, 0.471, 0.839],   # 2 Hopping     — blue    #2a78d6
    [0.106, 0.686, 0.478],   # 3 Streaking   — aqua    #1baf7a
    [0.929, 0.631, 0.000],   # 4 Inc. Spread — yellow  #eda100
    [0.000, 0.514, 0.000],   # 5 Swelling    — green   #008300
    [0.290, 0.227, 0.655],   # 6 Debris      — violet  #4a3aa7
    [0.890, 0.286, 0.282],   # 7 Super-Elev  — red     #e34948
    [0.910, 0.482, 0.643],   # 8 Spatter     — magenta #e87ba4
    [0.922, 0.408, 0.204],   # 9 Over Melt   — orange  #eb6834
    [0.000, 0.749, 0.843],   # 10 Under Melt — cyan    #00bfd7 (도메인상 "cold")
])
IGNORE_COLOR = np.array([0.05, 0.05, 0.05])

# ORNL 12-class → 11-class (Misprint=9 만 IGNORE)
ORNL_12_TO_11 = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5, 6:6, 7:7, 8:8, 9:-1, 10:9, 11:10}
V4_TO_11 = {i: i for i in range(11)}

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


def load_model(path: Path, device: torch.device):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    c = ckpt["config"]
    m = DefSegModel(backbone_name=c["backbone"], n_classes=c["n_classes"]).to(device)
    m.load_trainable_state_dict(ckpt["model_state"])
    m.eval()
    return m, int(c.get("img_size", 1036))


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


def dscnn_pred_11(seg_grp: h5py.Group, layer_idx: int) -> np.ndarray:
    """ORNL baseline (DSCNN) 12 boolean mask → 11-class argmax."""
    sample = seg_grp["0"][layer_idx]
    H, W = sample.shape
    out = np.full((H, W), IGNORE_INDEX, dtype=np.int8)
    for c in range(12):
        m = seg_grp[str(c)][layer_idx]
        out[m] = c
    return remap_to_11(out, ORNL_12_TO_11)


def parse_pair(s: str) -> tuple[str, int]:
    """`B2:331` 형식 → ("B2", 331)."""
    build, layer = s.split(":")
    build = build.strip()
    if build not in BUILD_TO_HDF5:
        raise argparse.ArgumentTypeError(f"unknown build '{build}' (choose from {list(BUILD_TO_HDF5)})")
    return build, int(layer)


def main():
    ap = argparse.ArgumentParser()
    # 단일 build 모드
    ap.add_argument("--build", default=None, choices=list(BUILD_TO_HDF5))
    ap.add_argument("--layers", type=int, nargs="+", default=None,
                    help="단일 build 모드: 시각화할 layer index 목록")
    # 다중 build 모드
    ap.add_argument("--pairs", nargs="+", type=parse_pair, default=None,
                    help="다중 build 모드: 'B2:331 B3:1206 B4:761' 형식")
    ap.add_argument("--img-size", type=int, default=1036)
    ap.add_argument("--out-name", default=None)
    ap.add_argument("--font-scale", type=float, default=1.0,
                    help="인쇄용 폰트 배율 (예: 논문 삽입 시 1.9)")
    ap.add_argument("--no-suptitle", action="store_true",
                    help="suptitle 생략 (LaTeX 캡션이 제목 담당)")
    ap.add_argument("--short-titles", action="store_true",
                    help="열 제목 축약 (큰 폰트에서 겹침 방지)")
    ap.add_argument("--no-class-index", action="store_true",
                    help="legend 클래스명 앞 인덱스 번호 생략")
    args = ap.parse_args()

    # 인자 정규화
    if args.pairs:
        pairs = args.pairs
    elif args.build and args.layers:
        pairs = [(args.build, li) for li in args.layers]
    else:
        # default: 5 build 대표 layer
        pairs = [("B1", 300), ("B2", 331), ("B3", 1206), ("B4", 761), ("B5", 860)]

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    REPO = Path(__file__).resolve().parents[2]
    ckpt = REPO / "DefSeg_AM/checkpoints/vitb14_reg_dpt_dual_sz1036_11cls_v4/stage2_best.pt"
    model, img_size = load_model(ckpt, device)
    img_size = round_to_patch(args.img_size, model.patch_size)
    print(f"loaded v4: img_size={img_size}")

    # HDF5 파일 캐싱
    hdf5_by_build: dict[str, Path] = {}
    for b, _ in pairs:
        hdf5_by_build[b] = REPO / "ORNL_Data/Co-Registered In-Situ and Ex-Situ Dataset" / \
                          "[baseline] (Peregrine v2023-11)" / f"{BUILD_TO_HDF5[b]}.hdf5"

    n_rows = len(pairs)
    fig, axes = plt.subplots(n_rows, 4, figsize=(22, 5.5 * n_rows))
    if n_rows == 1:
        axes = axes[None, :]

    col_titles = ["Melt Image", "Powder Image",
                  "DSCNN (ORNL baseline)", "Defect Seg-AM"]
    if args.short_titles:
        col_titles = ["Melt Image", "Powder Image", "DSCNN", "Defect Seg-AM"]

    for r, (build, li) in enumerate(pairs):
        print(f"  {build} layer {li}: reading + inferring …")
        with h5py.File(hdf5_by_build[build], "r") as f:
            vis0 = f["slices/camera_data/visible/0"]
            vis1 = f["slices/camera_data/visible/1"]
            seg = f["slices/segmentation_results"]
            i0 = ornl_image_to_uint8(vis0[li])
            i1 = ornl_image_to_uint8(vis1[li])
            dscnn = dscnn_pred_11(seg, li)
        p_native = infer_pred(model, i0, i1, img_size, device)
        defsegam = remap_to_11(p_native, V4_TO_11)

        axes[r, 0].imshow(i0, cmap="gray", interpolation="nearest")
        axes[r, 1].imshow(i1, cmap="gray", interpolation="nearest")
        axes[r, 2].imshow(colorize(dscnn), interpolation="nearest")
        axes[r, 3].imshow(colorize(defsegam), interpolation="nearest")

        axes[r, 0].set_ylabel(f"{build}  layer {li}", fontsize=15 * args.font_scale)

        if r == 0:
            for c in range(4):
                axes[r, c].set_title(col_titles[c], fontsize=16 * args.font_scale)

        for c in range(4):
            axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
            for spine in axes[r, c].spines.values():
                spine.set_edgecolor("#c3c2b7"); spine.set_linewidth(0.5)

            for c in range(4):
                axes[r, c].set_xticks([]); axes[r, c].set_yticks([])
                for spine in axes[r, c].spines.values():
                    spine.set_edgecolor("#c3c2b7"); spine.set_linewidth(0.5)

    # 하단 legend (한 번만)
    lab = (lambda c: CLASS_NAMES[c]) if args.no_class_index else (lambda c: f"{c} {CLASS_NAMES[c]}")
    handles = [patches.Patch(color=tuple(PALETTE[c]), label=lab(c))
               for c in range(N_CLASSES)]
    leg_ncol = 6
    leg_h = 0.045 if args.font_scale < 1.4 else 0.055
    leg_fs = 18 * args.font_scale if args.font_scale < 1.4 else 10.5 * args.font_scale
    fig.legend(handles=handles, ncol=leg_ncol, fontsize=leg_fs, frameon=False,
               loc="lower center", bbox_to_anchor=(0.02, 0.02, 0.96, leg_h),
               mode="expand", handlelength=1.2, handleheight=1.2,
               handletextpad=0.4, columnspacing=0.8)

    builds_used = sorted({b for b, _ in pairs})
    title_builds = "+".join(builds_used) if len(builds_used) > 1 else BUILD_TO_HDF5[builds_used[0]]
    if not args.no_suptitle:
        fig.suptitle(
            f"DSCNN vs Defect Seg-AM — {title_builds}",
            fontsize=26 * args.font_scale, y=0.995,
        )
    bottom = 0.05 if args.font_scale < 1.4 else 0.062
    fig.tight_layout(rect=[0, bottom, 1, 0.98 if not args.no_suptitle else 1.0])

    out_dir = REPO / "DefSeg_AM/figures/dscnn_vs_defsegam"
    out_dir.mkdir(parents=True, exist_ok=True)
    default_name = f"dscnn_vs_defsegam_v4_{'_'.join(builds_used)}.png"
    out_name = args.out_name or default_name
    out_path = out_dir / out_name
    fig.savefig(out_path, dpi=120, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"\nsaved → {out_path.relative_to(REPO)}")


if __name__ == "__main__":
    main()
