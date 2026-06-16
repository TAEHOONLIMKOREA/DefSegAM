"""v1 / v2 / v3 serving 체크포인트(stage 2)를 같은 ORNL layer 에 추론해
**각 버전의 native class 공간(그대로)** 으로 나란히 시각화한 figure 를 생성한다.

compare_v1_v2_v3.py 가 모든 예측을 공통 8-class 로 reduce 하는 것과 달리, 본
스크립트는 DefSegAM_API 가 실제로 반환하는 모습 그대로 —
  v1 = 12-class, v2 = 8-class, v3 = 10-class —
각 버전 고유 팔레트·범례로 보여준다.

figure 1장 레이아웃 (per layer):
    상단:  [ visible/0 (after melt) | visible/1 (after spread) ]
    하단:  [ v1 12-class | v2 8-class | v3 10-class ]  (각 패널 아래 해당 버전 범례)

사용 (repo root = /home/taehoon/3DP_VPPM 에서):
    DefSeg_AM/venv/bin/python -m DefSeg_AM.scripts.figure_v1_v2_v3_native
    # 옵션
    ... figure_v1_v2_v3_native --n-figures 10 --img-size 1036 --device cuda:0
"""
from __future__ import annotations

import argparse
from pathlib import Path

import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.font_manager as _fm

# 한국어 제목 렌더링용 CJK 폰트 (없으면 기본 폰트 — 한글은 □ 로 표시됨)
for _f in ("Noto Sans CJK KR", "Noto Sans CJK JP", "NanumGothic", "Malgun Gothic"):
    if any(_f == ff.name for ff in _fm.fontManager.ttflist):
        plt.rcParams["font.family"] = _f
        break
plt.rcParams["axes.unicode_minus"] = False
import numpy as np
import torch
import torch.nn.functional as F
from matplotlib import gridspec, patches
from PIL import Image

from DefSeg_AM.common.models.model import DefSegModel, round_to_patch
from DefSeg_AM.common.data.image_utils import normalize_image, ornl_image_to_uint8


# ============================================================================
# 버전별 native class 이름 + 팔레트 (DefSegAM_API/app/config.py 레지스트리와 동일)
# ============================================================================
V1_NAMES = [
    "Powder", "Printed", "Recoater Hopping", "Recoater Streaking",
    "Incomplete Spreading", "Swelling", "Debris", "Super-Elevation",
    "Spatter", "Misprint", "Over Melting", "Under Melting",
]
V1_PALETTE = np.array([
    [0.85, 0.85, 0.85], [0.30, 0.50, 0.85], [0.95, 0.55, 0.10], [0.20, 0.70, 0.30],
    [0.85, 0.15, 0.15], [0.55, 0.30, 0.75], [0.50, 0.35, 0.20], [0.95, 0.40, 0.65],
    [0.40, 0.40, 0.40], [0.75, 0.75, 0.10], [0.10, 0.70, 0.70], [0.30, 0.30, 0.85],
])

V2_NAMES = [
    "Powder", "Printed", "Recoater Disturbance", "Swelling",
    "Spatter", "Super-Elevation", "Over Melting", "Debris",
]
V2_PALETTE = np.array([
    [0.85, 0.85, 0.85], [0.30, 0.50, 0.85], [0.95, 0.55, 0.10], [0.55, 0.30, 0.75],
    [0.40, 0.40, 0.40], [0.95, 0.40, 0.65], [0.10, 0.70, 0.70], [0.50, 0.35, 0.20],
])

V3_NAMES = [
    "Powder", "Printed", "Recoater Hopping", "Recoater Streaking",
    "Incomplete Spreading", "Swelling", "Debris", "Super-Elevation",
    "Spatter", "Over Melting",
]
V3_PALETTE = np.array([
    [0.85, 0.85, 0.85], [0.30, 0.50, 0.85], [0.95, 0.55, 0.10], [0.90, 0.85, 0.15],
    [0.20, 0.65, 0.30], [0.55, 0.30, 0.75], [0.50, 0.35, 0.20], [0.95, 0.40, 0.65],
    [0.40, 0.40, 0.40], [0.10, 0.70, 0.70],
])

VERSIONS = [
    ("v1", "12-class", V1_NAMES, V1_PALETTE,
     "vits14_dpt_dual_sz1036_1gpu_nanfix/stage2_best.pt"),
    ("v2", "8-class",  V2_NAMES, V2_PALETTE,
     "vits14_dpt_dual_sz1036_8cls_v2/stage2_best_fold5_v2022_Maraging.pt"),
    ("v3", "10-class", V3_NAMES, V3_PALETTE,
     "vits14_dpt_dual_sz1036_10cls_v3/stage2_best.pt"),
]


def colorize(label: np.ndarray, palette: np.ndarray) -> np.ndarray:
    rgb = np.zeros((*label.shape, 3), dtype=np.float32)
    for c in range(len(palette)):
        rgb[label == c] = palette[c]
    return rgb


# ============================================================================
# Model load + inference
# ============================================================================
def load_model(path: Path, device: torch.device, img_size_req: int):
    ckpt = torch.load(path, map_location=device, weights_only=False)
    c = ckpt["config"]
    n_cls = c.get("n_classes")
    model = DefSegModel(backbone_name=c["backbone"], n_classes=n_cls).to(device)
    model.load_trainable_state_dict(ckpt["model_state"])
    model.eval()
    img_size = round_to_patch(img_size_req, model.patch_size)
    return model, n_cls, img_size, ckpt.get("val_acc")


@torch.no_grad()
def infer_native(model: DefSegModel, i0_u8, i1_u8, img_size, device) -> np.ndarray:
    H, W = i0_u8.shape
    i0_r = np.array(Image.fromarray(i0_u8).resize((img_size, img_size), Image.BILINEAR))
    i1_r = np.array(Image.fromarray(i1_u8).resize((img_size, img_size), Image.BILINEAR))
    t0 = torch.from_numpy(normalize_image(i0_r)).unsqueeze(0).to(device)
    t1 = torch.from_numpy(normalize_image(i1_r)).unsqueeze(0).to(device)
    logits = model(t0, t1)
    logits = F.interpolate(logits, size=(H, W), mode="bilinear", align_corners=False)
    return logits.argmax(dim=1)[0].cpu().numpy().astype(np.int16)


def present_class_handles(pred, names, palette):
    """예측에 실제 등장한 클래스만 범례로 (px 내림차순)."""
    counts = np.bincount(pred.reshape(-1), minlength=len(names))
    order = [c for c in np.argsort(counts)[::-1] if counts[c] > 0]
    return [patches.Patch(color=tuple(palette[c]),
                          label=f"{c} {names[c]} ({counts[c]/pred.size*100:.1f}%)")
            for c in order]


# ============================================================================
# Figure (per layer)
# ============================================================================
def make_figure(i0, i1, preds, out_path: Path, title: str) -> None:
    fig = plt.figure(figsize=(21, 13))
    gs = gridspec.GridSpec(2, 6, figure=fig, height_ratios=[1.0, 1.25],
                           hspace=0.12, wspace=0.06)

    # 상단: 입력 2장 (가운데 정렬되게 col 1-2, 3-4 사용)
    ax0 = fig.add_subplot(gs[0, 1:3]); ax0.imshow(i0, cmap="gray")
    ax0.set_title("visible/0 (after melt)", fontsize=13)
    ax1 = fig.add_subplot(gs[0, 3:5]); ax1.imshow(i1, cmap="gray")
    ax1.set_title("visible/1 (after spread)", fontsize=13)

    # 하단: v1/v2/v3 native (각 col 2칸씩)
    for k, (vid, vlabel, names, palette, _) in enumerate(VERSIONS):
        ax = fig.add_subplot(gs[1, 2 * k:2 * k + 2])
        pred = preds[vid]
        ax.imshow(colorize(pred, palette))
        ax.set_title(f"{vid} prediction ({vlabel})", fontsize=13)
        handles = present_class_handles(pred, names, palette)
        ax.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, -0.02),
                  ncol=2, fontsize=8, frameon=False, handlelength=1.2, columnspacing=1.0)

    for ax in fig.axes:
        ax.set_xticks([]); ax.set_yticks([])

    fig.suptitle(title, fontsize=15)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110, bbox_inches="tight")
    plt.close(fig)


# ============================================================================
# Main
# ============================================================================
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--n-figures", type=int, default=10,
                    help="생성할 figure(=layer) 총 개수")
    ap.add_argument("--builds", nargs="*", default=None,
                    help="사용할 build 파일명(.hdf5 제외). 기본=5개 build 전체에 분산")
    ap.add_argument("--img-size", type=int, default=1036)
    ap.add_argument("--device", default="cuda:0")
    ap.add_argument("--out-dir", default=None)
    args = ap.parse_args()

    device = torch.device(args.device if torch.cuda.is_available() else "cpu")
    print(f"device: {device}")

    REPO_ROOT = Path(__file__).resolve().parents[2]
    ckpt_dir = REPO_ROOT / "DefSeg_AM" / "checkpoints"

    # 모델 3종 로딩
    models = {}
    for vid, vlabel, names, palette, rel in VERSIONS:
        mdl, n_cls, img_size, val_acc = load_model(ckpt_dir / rel, device, args.img_size)
        models[vid] = {"model": mdl, "img_size": img_size}
        print(f"  loaded {vid} ({vlabel})  n_cls={n_cls}  img_size={img_size}  val_acc={val_acc}")

    # build 목록 (경로의 '[' 때문에 glob 대신 iterdir)
    from DefSeg_AM.v3 import config_v3 as cfg
    hdf5_root = Path(cfg.ORNL_HDF5_DIR)
    all_builds = sorted(p.stem for p in hdf5_root.iterdir() if p.suffix == ".hdf5")
    builds = args.builds if args.builds else all_builds
    print(f"builds: {builds}")

    # n_figures 를 build 들에 균등 분배
    per_build = [args.n_figures // len(builds)] * len(builds)
    for i in range(args.n_figures % len(builds)):
        per_build[i] += 1

    out_dir = Path(args.out_dir) if args.out_dir else \
        REPO_ROOT / "DefSeg_AM" / "figures" / "native_v1_v2_v3"
    out_dir.mkdir(parents=True, exist_ok=True)

    fig_count = 0
    for build, n_sel in zip(builds, per_build):
        if n_sel == 0:
            continue
        hdf5_path = hdf5_root / f"{build}.hdf5"
        with h5py.File(hdf5_path, "r") as f:
            vis0 = f["slices/camera_data/visible/0"]
            vis1 = f["slices/camera_data/visible/1"]
            n_total = vis0.shape[0]
            lo = int(n_total * 0.10)
            hi = int(n_total * 0.90)
            layer_idxs = np.linspace(lo, hi, n_sel, dtype=int).tolist()
            print(f"\n[{build}] {n_total} layers → figure layers {layer_idxs}")

            for li in layer_idxs:
                i0 = ornl_image_to_uint8(vis0[li])
                i1 = ornl_image_to_uint8(vis1[li])
                preds = {vid: infer_native(m["model"], i0, i1, m["img_size"], device)
                         for vid, m in models.items()}
                short = build.replace(" ", "_").replace("TCR_Phase_1_", "")
                out_path = out_dir / f"{short}_layer{li:04d}.png"
                make_figure(i0, i1, preds, out_path,
                            title=f"{build} — layer {li} — v1/v2/v3 native 추론 비교")
                fig_count += 1
                print(f"  [{fig_count:2d}] saved {out_path.relative_to(REPO_ROOT)}")

    print(f"\n=== done: {fig_count} figures → {out_dir.relative_to(REPO_ROOT)} ===")


if __name__ == "__main__":
    main()
