"""v5 모델의 ORNL 참고 평가 — DSCNN 예측과의 일치도 (학습 없음, 1-pass). PLAN_v5 §6.4.

**이 수치는 정답 대비 성능이 아니다.** ORNL HDF5 의 `segmentation_results` 는 DSCNN 의 예측이고,
v5 는 이것을 학습에도 길이 · ckpt 선정에도 쓰지 않는다. v1–v4 가 모두 Build 1 에서 같은 표를 냈으므로
나란히 볼 수 있게 같은 형식으로 낸다 (v4: figures/<v4 run>/v4/stage2/confusion/).

클래스 공간이 v4 와 같아 v4 의 라벨 캐시 (cache/resized_sz1036_v4) 를 그대로 읽는다.

출력 (기본 figures/<run>/v5/full/build1_vs_dscnn/):
    confusion_counts.npy, confusion_matrix.csv, per_class_metrics.csv, confusion_row_normalized.png

사용 (CWD = 3DP_VPPM/):
    python -m DefSeg_AM.v5.inference.confusion                       # 전수 학습 모델의 last.pt, Build 1
    python -m DefSeg_AM.v5.inference.confusion --max-batches 2       # 스모크
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from ...common.models.model import round_to_patch
from .. import config_v5 as cfg
from ..data.data_ornl_v5 import DefSegORNLCachedDatasetV5
from .evaluate_gt import load_model
from .metrics import accumulate_confusion, dump_all, report


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--backbone", type=str, default=cfg.DINO_BACKBONE)
    ap.add_argument("--fold", type=str, default=None, choices=sorted(cfg.CV_FOLDS),
                    help="검증 fold 의 ckpt 를 채점 (기본: 전수 학습 모델)")
    ap.add_argument("--which", type=str, default="last", choices=["last", "best", "last_raw"])
    ap.add_argument("--checkpoint", type=str, default=None, help="직접 ckpt 경로 지정")
    ap.add_argument("--split", type=str, default="val", choices=["val", "train"],
                    help="val=Build 1, train=Builds 2-5")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=cfg.IMG_SIZE,
                    help="run-name 과 ckpt 를 찾는 데 쓴다. 캐시는 1036² 뿐이라 채점은 항상 캐시 해상도")
    ap.add_argument("--cache-img-size", type=int, default=cfg.IMG_SIZE)
    ap.add_argument("--max-batches", type=int, default=None, help="스모크용: loader 앞 N 배치만")
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or cfg.default_run_name(args.backbone, args.img_size)
    names = cfg.CLASS_NAMES_V5
    n_classes = cfg.N_CLASSES_V5

    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = cfg.run_ckpt_dir(run_name, args.fold) / f"{args.which}.pt"
    if not ckpt_path.exists():
        ap.error(f"ckpt not found: {ckpt_path}")

    model, _ = load_model(ckpt_path, device)
    cache_size = round_to_patch(args.cache_img_size, model.patch_size)
    ds = DefSegORNLCachedDatasetV5(args.split, cache_size)
    builds = "Build 1" if args.split == "val" else "Builds 2-5"
    loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                        num_workers=args.num_workers, pin_memory=True, drop_last=False)
    print(f"{ckpt_path.name} | ORNL cached {args.split} ({builds}) — {len(ds)} layers | img_size={cache_size} "
          f"| 정답 = DSCNN 예측 (참고용)")

    cm = accumulate_confusion(model, loader, device, n_classes, log_every=20, max_batches=args.max_batches)
    sub = "build1_vs_dscnn" if args.split == "val" else "builds2-5_vs_dscnn"
    if args.which != "last" or args.checkpoint is not None:
        sub += f"_{Path(ckpt_path).stem}"
    out_dir = Path(args.out_dir) if args.out_dir else cfg.run_fig_dir(run_name, args.fold) / sub
    dump_all(cm, out_dir, names, title=f"v5 {cfg.run_tag(args.fold)} vs DSCNN prediction ({builds}, reference only)")
    report(cm, names, n_classes)
    print(f"\nsaved → {out_dir}")


if __name__ == "__main__":
    main()
