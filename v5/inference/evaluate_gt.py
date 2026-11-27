"""사람 GT 로 채점 — native 해상도 (학습 없음, 1-pass). PLAN_v5 §6.2.

서빙과 같은 경로로 추론한다: 원본 → img_size 로 줄임 → dual forward → logits 를 원본 크기로
bilinear upsample → argmax. 정답은 native 해상도의 사람 주석 (v5 11-class 로 재매핑).

채점할 표본:
  --fold <name>       그 fold 가 떼어 둔 표본. 학습이 남긴 figures/<run>/v5/cv/<fold>/run.json 의 목록을 읽는다
                      (소스 통째 fold 는 run.json 이 없어도 계산할 수 있다)
  --sources a b ...   소스 이름으로 직접 지정
  (둘 다 없으면 8 소스 전부 — 전수 학습 모델에게는 학습 데이터 자체이므로 일반화 지표가 아니다)

채점할 ckpt:
  기본                checkpoints/<run>/cv/<fold>/<which>.pt (fold 지정 시), 아니면 checkpoints/<run>/full/<which>.pt
  --which             last (기본) | best | last_raw
  --checkpoint PATH   임의 ckpt. v4 의 stage2_best.pt 도 클래스 · 구조가 같아 그대로 읽힌다
                      (이때는 --tag-prefix 로 출력 폴더를 구분한다. 예: --tag-prefix ref_v4)

출력 (figures/<run>/v5/<full | cv/<fold>>/<tag>/, tag = <tag-prefix>[_tta][_pp], 기본 prefix 는 eval_native_<which>):
    confusion_counts.npy, confusion_matrix.csv, per_class_metrics.csv,
    confusion_row_normalized.png, metrics.json, (옵션) panels/*.png

사용 (CWD = 3DP_VPPM/):
    python -m DefSeg_AM.v5.inference.evaluate_gt --fold loso_L21
    python -m DefSeg_AM.v5.inference.evaluate_gt --fold loso_L21 --tag-prefix ref_v4 \
        --checkpoint DefSeg_AM/checkpoints/vitb14_reg_dpt_dual_sz1036_11cls_v4/stage2_best.pt
"""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ...common.data.image_utils import normalize_image, resize_image_uint8
from ...common.models.model import DefSegModel, round_to_patch

from .. import config_v5 as cfg
from ..data.data_gt_v5 import (
    SampleSpecV5,
    enumerate_samples_v5,
    load_image_uint8,
    remap_label_v5,
)
from .metrics import dump_all, report, summarize
from .postprocess import apply_postprocess
from .viz import save_panel


def load_model(ckpt_path: Path, device: torch.device) -> tuple[DefSegModel, dict]:
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ckpt.get("config", {}) or {}
    n_classes = c.get("n_classes", cfg.N_CLASSES_V5)
    if n_classes != cfg.N_CLASSES_V5:
        raise ValueError(f"{ckpt_path}: n_classes={n_classes} — v5 11-class 로 채점할 수 없다")
    model = DefSegModel(
        backbone_name=c.get("backbone", cfg.DINO_BACKBONE), n_classes=n_classes,
    ).to(device)
    model.load_trainable_state_dict(ckpt["model_state"])
    model.eval()
    print(
        f"loaded {ckpt_path}  (version={c.get('version', '?')}, weights={ckpt.get('weights', '?')}, "
        f"step={ckpt.get('step', '?')}, epoch={ckpt.get('epoch', '?')}, img_size={c.get('img_size', '?')})"
    )
    return model, ckpt


@torch.no_grad()
def _logits_native(
    model: DefSegModel, i0_u8: np.ndarray, i1_u8: np.ndarray,
    out_hw: tuple[int, int], img_size: int, device: torch.device,
) -> torch.Tensor:
    """원본 한 쌍 → (1, C, H, W) logits (원본 크기로 올린 것)."""
    t0 = torch.from_numpy(normalize_image(resize_image_uint8(i0_u8, img_size))).unsqueeze(0).to(device)
    t1 = torch.from_numpy(normalize_image(resize_image_uint8(i1_u8, img_size))).unsqueeze(0).to(device)
    logits = model(t0, t1)
    return F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)


@torch.no_grad()
def predict_native(
    model: DefSegModel, i0_u8: np.ndarray, i1_u8: np.ndarray,
    out_hw: tuple[int, int], img_size: int, device: torch.device,
    tta: bool = False,
) -> torch.Tensor:
    """(H, W) long 예측 (GPU tensor). tta=True 면 flip(LR) × flip(UD) × rot180 = 8-way softmax 평균 (v4 와 같은 방식)."""
    if not tta:
        return _logits_native(model, i0_u8, i1_u8, out_hw, img_size, device)[0].argmax(dim=0)

    prob_sum = None
    for flip_lr, flip_ud, rot180 in product([False, True], repeat=3):
        i0, i1 = i0_u8, i1_u8
        if flip_lr:
            i0 = i0[:, ::-1].copy(); i1 = i1[:, ::-1].copy()
        if flip_ud:
            i0 = i0[::-1, :].copy(); i1 = i1[::-1, :].copy()
        if rot180:
            i0 = np.rot90(i0, k=2).copy(); i1 = np.rot90(i1, k=2).copy()
        p = F.softmax(_logits_native(model, i0, i1, out_hw, img_size, device), dim=1)[0]   # (C, H, W)
        # 역변환 (적용 역순)
        if rot180:
            p = torch.rot90(p, k=-2, dims=(1, 2))
        if flip_ud:
            p = torch.flip(p, dims=(1,))
        if flip_lr:
            p = torch.flip(p, dims=(2,))
        prob_sum = p if prob_sum is None else prob_sum + p
    return prob_sum.argmax(dim=0)


def select_samples(
    samples: list[SampleSpecV5], fold: str | None, sources: list[str] | None,
    fold_run_json: Path | None,
) -> tuple[list[SampleSpecV5], list[int] | None]:
    """채점 표본과, (알 수 있으면) 결함 mIoU 에 넣을 클래스 목록."""
    if fold is not None:
        if fold_run_json is not None and fold_run_json.exists():
            meta = json.loads(fold_run_json.read_text())
            keys = set(meta["val_samples"])
            sel = [s for s in samples if s.key in keys]
            missing = keys - {s.key for s in sel}
            if missing:
                raise FileNotFoundError(f"run.json 의 검증 표본을 찾지 못했다: {sorted(missing)[:5]} …")
            return sel, [int(c) for c in meta.get("defect_eval_classes", [])] or None
        spec = cfg.CV_FOLDS[fold]
        if spec[0] != "source":
            raise FileNotFoundError(
                f"{fold_run_json} 이 없다 — 소스 안 분할 fold ('{fold}') 는 학습이 남긴 표본 목록이 있어야 채점할 수 있다. "
                f"먼저 `--fold {fold}` 로 학습할 것."
            )
        return [s for s in samples if s.source_name == spec[1]], None
    if sources:
        unknown = set(sources) - {s.source_name for s in samples}
        if unknown:
            raise ValueError(f"unknown source(s): {sorted(unknown)}")
        return [s for s in samples if s.source_name in set(sources)], None
    return samples, None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--backbone", type=str, default=cfg.DINO_BACKBONE)
    ap.add_argument("--img-size", type=int, default=cfg.IMG_SIZE)
    ap.add_argument("--fold", type=str, default=None, choices=sorted(cfg.CV_FOLDS))
    ap.add_argument("--sources", type=str, nargs="+", default=None)
    ap.add_argument("--which", type=str, default="last", choices=["last", "best", "last_raw"])
    ap.add_argument("--checkpoint", type=str, default=None, help="직접 ckpt 경로 지정 (예: v4 ckpt)")
    ap.add_argument("--tag-prefix", type=str, default=None,
                    help="출력 폴더 이름의 앞부분 (기본 eval_native_<which>). 뒤에 _tta · _pp 가 붙는다")
    ap.add_argument("--tta", action="store_true", help="flip(LR) × flip(UD) × rot180 8-way 평균")
    ap.add_argument("--postprocess", action="store_true", help="휴리스틱 후처리 2 규칙")
    ap.add_argument("--save-figs", type=int, default=0, help="비교 panel 을 이만큼 저장 (균등 추출)")
    ap.add_argument("--max-layers", type=int, default=None, help="스모크용: 앞 N 레이어만")
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or cfg.default_run_name(args.backbone, args.img_size)
    fig_dir = cfg.run_fig_dir(run_name, args.fold)

    if args.checkpoint is not None:
        ckpt_path = Path(args.checkpoint)
    else:
        ckpt_path = cfg.run_ckpt_dir(run_name, args.fold) / f"{args.which}.pt"
    if not ckpt_path.exists():
        ap.error(f"ckpt not found: {ckpt_path}")

    model, ckpt = load_model(ckpt_path, device)
    img_size = round_to_patch(args.img_size, model.patch_size)

    all_samples = enumerate_samples_v5()
    samples, defect_eval = select_samples(
        all_samples, args.fold, args.sources,
        fold_run_json=(fig_dir / "run.json") if args.fold else None,
    )
    if args.max_layers is not None:
        samples = samples[: args.max_layers]
    if not samples:
        ap.error("채점할 표본이 없다")

    tag = (
        (args.tag_prefix or f"eval_native_{args.which}")
        + ("_tta" if args.tta else "") + ("_pp" if args.postprocess else "")
    )
    out_dir = Path(args.out_dir) if args.out_dir else fig_dir / tag
    print(
        f"{len(samples)} 레이어 채점 | fold={args.fold} | img_size={img_size} | tta={args.tta} pp={args.postprocess} "
        f"| out={out_dir}"
    )

    n_cls = cfg.N_CLASSES_V5
    cm_total = np.zeros((n_cls, n_cls), dtype=np.int64)
    cm_by_source: dict[str, np.ndarray] = {}
    fig_idx = set()
    if args.save_figs > 0:
        fig_idx = set(np.linspace(0, len(samples) - 1, min(args.save_figs, len(samples)), dtype=int).tolist())

    for n, s in enumerate(samples):
        gt = remap_label_v5(np.load(s.ann_path), s.mapping_key)        # (H, W) int8, native
        i0 = load_image_uint8(s.img0_path)
        i1 = load_image_uint8(s.img1_path)
        pred_t = predict_native(model, i0, i1, gt.shape, img_size, device, tta=args.tta)
        if args.postprocess:
            pred_np, _ = apply_postprocess(pred_t.cpu().numpy().astype(np.int8), i0, i1)
            pred_t = torch.from_numpy(pred_np.astype(np.int64)).to(device)
        gt_t = torch.from_numpy(gt.astype(np.int64)).to(device)
        valid = gt_t != cfg.IGNORE_INDEX
        idx = gt_t[valid] * n_cls + pred_t[valid]
        cm = torch.bincount(idx, minlength=n_cls * n_cls).reshape(n_cls, n_cls).cpu().numpy()
        cm_total += cm
        cm_by_source[s.source_name] = cm_by_source.get(s.source_name, 0) + cm
        if n in fig_idx:
            save_panel(
                i0, i1, gt, pred_t.cpu().numpy().astype(np.int8),
                out_path=out_dir / "panels" / f"{s.source_name}_{s.ann_path.stem}.png",
                title=f"{s.key}{' / TTA' if args.tta else ''}{' / PP' if args.postprocess else ''}",
                ref_title="human GT (11-class)",
                pred_title=f"prediction ({ckpt_path.parent.name}/{ckpt_path.name})",
            )
        if (n + 1) % 10 == 0 or n + 1 == len(samples):
            print(f"  [{n + 1:3d}/{len(samples)}] 채점 …", flush=True)
        del pred_t, gt_t

    names = cfg.CLASS_NAMES_V5
    dump_all(cm_total, out_dir, names, title=f"{run_name} {cfg.run_tag(args.fold)} / {tag} (human GT, native)")
    report(cm_total, names, n_cls)

    s_all = summarize(cm_total, defect_eval)

    def _j(s: dict) -> dict:
        return {
            "acc": None if s["acc"] != s["acc"] else s["acc"],
            "miou": None if s["miou"] != s["miou"] else s["miou"],
            "defect_miou": None if s["defect_miou"] != s["defect_miou"] else s["defect_miou"],
            "iou": [None if v != v else float(v) for v in s["iou"]],
            "defect_classes": s["defect_classes"],
        }

    with (out_dir / "metrics.json").open("w") as fh:
        json.dump({
            "checkpoint": str(ckpt_path),
            "ckpt_version": (ckpt.get("config", {}) or {}).get("version"),
            "ckpt_weights": ckpt.get("weights"),
            "ckpt_step": ckpt.get("step"),
            "run_name": run_name, "fold": args.fold, "tag": tag,
            "img_size": img_size, "tta": args.tta, "postprocess": args.postprocess,
            "resolution": "native",
            "n_layers": len(samples),
            "samples": [s.key for s in samples],
            "class_names": names,
            "overall": _j(s_all),
            "by_source": {src: _j(summarize(cm, defect_eval)) for src, cm in cm_by_source.items()},
        }, fh, ensure_ascii=False, indent=1)
    dm = s_all["defect_miou"]
    print(
        f"\n결함 mIoU ({[cfg.CLASS_ABBRS_V5[c] for c in s_all['defect_classes']]}) = "
        f"{dm:.4f}   mIoU = {s_all['miou']:.4f}   acc = {s_all['acc']:.4f}"
    )
    print(f"saved → {out_dir}")


if __name__ == "__main__":
    main()
