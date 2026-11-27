"""사람 GT 로 채점 (학습 없음, 1-pass). PLAN_v6 §6.2.

채점 해상도는 둘이다 (--resolution, 기본 both — forward 한 번으로 둘 다 낸다):
  native    서빙과 같은 경로. 원본 → img_size 로 줄임 → dual forward → logits 를 원본 크기로 bilinear upsample → argmax.
            정답은 원본 해상도의 라벨. **주 지표는 이쪽이다.**
  resized   학습 중 검증과 같은 방식. img_size² 에서 argmax, 정답은 img_size² 로 줄인 라벨 (NEAREST).
            학습 곡선 (curve.csv) 과 복사 기준선 (R1) 이 이 해상도이므로 그것들과 견줄 때 쓴다.
원본 크기가 다른 소스를 합친 IoU 는 해상도에 따라 달라진다 (화소가 많은 소스의 값이 된다 — PLAN_v6 §1.1-15).
그래서 그룹 (소스) 별 표를 함께 낸다.

채점할 표본:
  --fold <name>       그 fold 가 떼어 둔 표본. 학습이 남긴 figures/<run>/v6/cv/<fold>/run.json 의 목록을 읽고, 없으면 fold 정의로 다시 계산한다
  --groups a b ...    그룹 (Peregrine 소스) 이름으로 직접 지정
  (둘 다 없으면 125 레이어 전부 — 전수 학습 모델에게는 학습 데이터 자체이므로 일반화 지표가 아니다)

채점할 ckpt:
  기본                checkpoints/<run>/cv/<fold>/<which>.pt (fold 지정 시), 아니면 checkpoints/<run>/full/<which>.pt
  --which             last (기본 — 배포 후보) | best | last_raw
  --checkpoint PATH   임의의 v6 (10-class) ckpt. 옛 11-class 모델은 inference/baselines.py 로 채점한다

출력 (figures/<run>/v6/<full | cv/<fold>>/<tag>/, tag = <prefix>_<native|resized>_<which>[_tta][_pp], 기본 prefix 는 eval):
    confusion_counts.npy, confusion_matrix.csv, per_class_metrics.csv, confusion_row_normalized.png,
    metrics.json (클래스별 IoU · 정밀도 · 재현율, 놓침 · 잘못 짚음, 혼동 쌍, IS ∪ PS, 그룹별 표), (옵션) panels/*.png

사용 (CWD = 3DP_VPPM/):
    python -m DefSeg_AM.v6.inference.evaluate --fold inlayer
    python -m DefSeg_AM.v6.inference.evaluate --fold inlayer --which best --save-figs 22
    python -m DefSeg_AM.v6.inference.evaluate --fold inlayer --tta --postprocess       # 보조 지표
"""
from __future__ import annotations

import argparse
import json
from itertools import product
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ...common.data.image_utils import normalize_image, resize_image_uint8, resize_label
from ...common.models.model import DefSegModel, round_to_patch

from .. import config_v6 as cfg
from ..data.data_gt_v6 import (
    SampleSpecV6,
    count_pixels,
    enumerate_samples_v6,
    load_label,
    load_sample_native,
    split_by_fold,
)
from .metrics import (
    all_pair_confusions,
    confusion_from_pred,
    dump_all,
    merged_iou,
    miss_false_alarm,
    per_class_metrics,
    report,
    summarize,
)
from .postprocess import apply_postprocess
from .viz import save_panel

RESOLUTIONS = ("native", "resized")


def load_model(ckpt_path: Path, device: torch.device, n_classes: int = cfg.N_CLASSES_V6) -> tuple[DefSegModel, dict]:
    """ckpt → (모델, ckpt dict). ckpt 의 클래스 수가 n_classes 와 다르면 멈춘다.

    v6 ckpt (10-class) 는 기본값으로, 옛 11-class ckpt (v4 · v5) 는 n_classes=11 로 읽는다 — 구조가 같다.
    """
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ckpt.get("config", {}) or {}
    n_ckpt = c.get("n_classes")
    if n_ckpt != n_classes:
        raise ValueError(f"{ckpt_path}: n_classes={n_ckpt} — {n_classes}-class 모델로 읽을 수 없다")
    if n_classes == cfg.N_CLASSES_V6 and c.get("schema") != cfg.SCHEMA_VERSION:
        raise ValueError(f"{ckpt_path}: schema={c.get('schema')} — {cfg.SCHEMA_VERSION} 의 ckpt 가 아니다")
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


# ---------------------------------------------------------------------------
# 예측 — 한 번의 forward 로 두 해상도
# ---------------------------------------------------------------------------

@torch.no_grad()
def predict_pair(
    model: DefSegModel, i0_u8: np.ndarray, i1_u8: np.ndarray,
    img_size: int, device: torch.device,
    resolutions: tuple[str, ...] = RESOLUTIONS, tta: bool = False,
) -> dict[str, torch.Tensor]:
    """원본 한 쌍 → {"native": (H, W) long, "resized": (S, S) long} (요청한 것만, GPU tensor).

    native 는 logits 를 원본 크기로 올린 뒤 argmax (서빙과 같은 경로), resized 는 img_size² 에서 argmax.
    tta=True 면 flip(LR) × flip(UD) × rot180 = 8-way softmax 평균 (v4 · v5 와 같은 방식. 리코팅 축을 바꾸는 변환은 없다).
    """
    out_hw = i0_u8.shape
    combos = list(product([False, True], repeat=3)) if tta else [(False, False, False)]
    acc: dict[str, torch.Tensor | None] = {r: None for r in resolutions}
    for flip_lr, flip_ud, rot180 in combos:
        i0, i1 = i0_u8, i1_u8
        if flip_lr:
            i0 = i0[:, ::-1].copy(); i1 = i1[:, ::-1].copy()
        if flip_ud:
            i0 = i0[::-1, :].copy(); i1 = i1[::-1, :].copy()
        if rot180:
            i0 = np.rot90(i0, k=2).copy(); i1 = np.rot90(i1, k=2).copy()
        t0 = torch.from_numpy(normalize_image(resize_image_uint8(i0, img_size))).unsqueeze(0).to(device)
        t1 = torch.from_numpy(normalize_image(resize_image_uint8(i1, img_size))).unsqueeze(0).to(device)
        logits = model(t0, t1)                                           # (1, C, S, S)
        for r in resolutions:
            z = logits if r == "resized" else F.interpolate(logits, size=out_hw, mode="bilinear", align_corners=False)
            p = F.softmax(z, dim=1)[0] if tta else z[0]                  # (C, h, w)
            # 역변환 (적용 역순)
            if rot180:
                p = torch.rot90(p, k=-2, dims=(1, 2))
            if flip_ud:
                p = torch.flip(p, dims=(1,))
            if flip_lr:
                p = torch.flip(p, dims=(2,))
            acc[r] = p if acc[r] is None else acc[r] + p
    return {r: v.argmax(dim=0) for r, v in acc.items()}


# ---------------------------------------------------------------------------
# 채점할 표본
# ---------------------------------------------------------------------------

def label_pix(samples: list[SampleSpecV6], img_size: int) -> np.ndarray:
    """표본마다 학습 해상도 기준 클래스별 화소 수 (라벨만 읽는다) — fold 를 다시 계산할 때 쓴다."""
    return np.stack([count_pixels(resize_label(load_label(s.ann_path), img_size)) for s in samples])


def select_samples(
    samples: list[SampleSpecV6], fold: str | None, groups: list[str] | None,
    fold_run_json: Path | None, img_size: int,
) -> tuple[list[SampleSpecV6], list[int] | None, list[SampleSpecV6]]:
    """(채점 표본, 결함 mIoU 에 넣을 클래스 목록 (모르면 None), 그 fold 의 학습 표본 (모르면 빈 목록))."""
    if fold is not None:
        if fold_run_json is not None and fold_run_json.exists():
            meta = json.loads(fold_run_json.read_text())
            by_key = {s.key: s for s in samples}
            missing = [k for k in meta["val_samples"] + meta["train_samples"] if k not in by_key]
            if missing:
                raise FileNotFoundError(f"run.json 의 표본을 데이터셋에서 찾지 못했다: {missing[:5]} …")
            return (
                [by_key[k] for k in meta["val_samples"]],
                [int(c) for c in meta.get("defect_eval_classes", [])] or None,
                [by_key[k] for k in meta["train_samples"]],
            )
        pix = label_pix(samples, img_size)
        train_idx, val_idx = split_by_fold(samples, pix, cfg.CV_FOLDS[fold])
        tp, vp = pix[train_idx].sum(axis=0), pix[val_idx].sum(axis=0)
        defect = [c for c in cfg.DEFECT_IDS_V6 if tp[c] > 0 and vp[c] > 0]
        return [samples[i] for i in val_idx], defect or None, [samples[i] for i in train_idx]
    if groups:
        unknown = set(groups) - {s.group for s in samples}
        if unknown:
            raise ValueError(f"unknown group(s): {sorted(unknown)}")
        return [s for s in samples if s.group in set(groups)], None, []
    return samples, None, []


# ---------------------------------------------------------------------------
# 채점 루프와 저장 — baselines.py 도 함께 쓴다
# ---------------------------------------------------------------------------

def score_model(
    model: DefSegModel,
    samples: list[SampleSpecV6],
    img_size: int,
    device: torch.device,
    resolutions: tuple[str, ...],
    tta: bool = False,
    postprocess: bool = False,
    luts: dict[str, torch.Tensor | None] | None = None,
    n_cm: int = cfg.N_CLASSES_V6,
    panel_dir: Path | None = None,
    n_panels: int = 0,
    panel_titles: tuple[str, str] = ("human GT (pbf-2026.11)", "prediction"),
) -> dict[str, dict[str, dict]]:
    """표본을 차례로 추론해 confusion matrix 를 쌓는다.

    luts: {변형 이름: 모델의 class id → 채점 열 id 로 옮기는 (C_model,) long tensor, 또는 None (그대로)}.
          옛 11-class 모델을 새 체계로 옮겨 채점할 때 쓴다. None 이면 {"": None}.
    n_cm: confusion matrix 의 한 변 (10, 또는 "대응 없음" 열을 둔 11).

    Returns: {변형: {해상도: {"total": (n_cm, n_cm) int64, "by_group": {그룹: (n_cm, n_cm)}}}}
    """
    luts = {"": None} if luts is None else luts
    out = {
        v: {r: {"total": np.zeros((n_cm, n_cm), dtype=np.int64), "by_group": {}} for r in resolutions}
        for v in luts
    }
    fig_idx: set[int] = set()
    if n_panels > 0 and panel_dir is not None:
        fig_idx = set(np.linspace(0, len(samples) - 1, min(n_panels, len(samples)), dtype=int).tolist())
    first = next(iter(luts))

    for n, s in enumerate(samples):
        i0, i1, gt = load_sample_native(s)
        gts = {}
        if "native" in resolutions:
            gts["native"] = gt
        if "resized" in resolutions:
            gts["resized"] = resize_label(gt, img_size)
        preds = predict_pair(model, i0, i1, img_size, device, resolutions=resolutions, tta=tta)
        for r in resolutions:
            gt_t = torch.from_numpy(gts[r].astype(np.int64)).to(device)
            for v, lut in luts.items():
                p = preds[r] if lut is None else lut[preds[r]]
                if postprocess:
                    p_np, _ = apply_postprocess(p.cpu().numpy().astype(np.int8))
                    p = torch.from_numpy(p_np.astype(np.int64)).to(device)
                cm = confusion_from_pred(p, gt_t, n_cm).reshape(n_cm, n_cm).cpu().numpy()
                slot = out[v][r]
                slot["total"] += cm
                slot["by_group"][s.group] = slot["by_group"].get(s.group, 0) + cm
                if n in fig_idx and r == "native" and v == first:
                    save_panel(
                        i0, i1, gt, p.cpu().numpy().astype(np.int16),
                        out_path=panel_dir / f"{s.group}_{s.layer:07d}.png",
                        title=f"{s.key}{' / TTA' if tta else ''}{' / PP' if postprocess else ''}",
                        ref_title=panel_titles[0], pred_title=panel_titles[1],
                    )
            del gt_t
        if (n + 1) % 10 == 0 or n + 1 == len(samples):
            print(f"  [{n + 1:3d}/{len(samples)}] 채점 …", flush=True)
        del preds
    return out


def _none(x) -> float | None:
    x = float(x)
    return None if x != x else x


def summary_json(cm: np.ndarray, defect_eval: list[int] | None, none_id: int | None = None) -> dict:
    """confusion matrix 하나 → metrics.json 에 넣는 요약 (nan 은 null)."""
    s = summarize(cm, defect_eval)
    m = per_class_metrics(cm)
    mfa = miss_false_alarm(cm, none_id=none_id)
    return {
        "acc": _none(s["acc"]), "miou": _none(s["miou"]), "defect_miou": _none(s["defect_miou"]),
        "defect_classes": s["defect_classes"],
        "iou": [_none(v) for v in s["iou"]],
        "recall": [_none(v) if m["support"][c] > 0 else None for c, v in enumerate(m["recall"])],
        "precision": [_none(v) for v in m["precision"]],
        "support_px": [int(v) for v in m["support"]],
        "miss_false_alarm": {k: (_none(v) if isinstance(v, float) else v) for k, v in mfa.items()},
        "pairs": {
            name: {k: (_none(v) if isinstance(v, float) else v) for k, v in d.items()}
            for name, d in all_pair_confusions(cm).items()
        },
        "is_ps_union_iou": _none(merged_iou(cm, [2, 3])),
    }


def dump_scores(
    cm_total: np.ndarray,
    cm_by_group: dict[str, np.ndarray],
    out_dir: Path,
    names: list[str],
    defect_eval: list[int] | None,
    meta: dict,
    title: str,
    none_id: int | None = None,
) -> dict:
    """confusion matrix → 파일 (행렬 · 클래스별 표 · heatmap · metrics.json) + 화면 요약. 요약 dict 를 돌려준다."""
    n = cm_total.shape[0]
    dump_all(cm_total, out_dir, names, title=title)
    report(cm_total, names, n)
    overall = summary_json(cm_total, defect_eval, none_id=none_id)
    doc = dict(meta)
    doc["class_names"] = names
    doc["class_abbrs"] = list(cfg.CLASS_ABBRS_V6) + (["none"] if none_id is not None else [])
    doc["overall"] = overall
    doc["by_group"] = {g: summary_json(cm, defect_eval, none_id=none_id) for g, cm in cm_by_group.items()}
    with (out_dir / "metrics.json").open("w") as fh:
        json.dump(doc, fh, ensure_ascii=False, indent=1)

    mfa = overall["miss_false_alarm"]
    dm = overall["defect_miou"]
    print(
        f"\n결함 mIoU ({[cfg.CLASS_ABBRS_V6[c] for c in overall['defect_classes']]}) = "
        f"{dm if dm is None else round(dm, 4)}   mIoU = {overall['miou']:.4f}   acc = {overall['acc']:.4f}"
    )
    if mfa["miss"] is not None:
        print(
            f"놓침 (결함 정답 → 정상 예측) = {100 * mfa['miss']:.1f}%"
            + (f"   대응 없음으로 예측 = {100 * mfa['to_none']:.1f}%" if none_id is not None else "")
            + f"   다른 결함으로 = {100 * mfa['wrong_defect']:.1f}%   잘못 짚음 (정상 정답 → 결함 예측) = {100 * mfa['false_alarm']:.2f}%"
        )
    for name, d in overall["pairs"].items():
        if d["within_pair"] is not None:
            print(f"  혼동 쌍 {name}: 쌍 안 혼동률 {100 * d['within_pair']:.1f}%  "
                  f"(앞→뒤 {100 * (d['a_as_b'] or 0):.1f}%, 뒤→앞 {100 * (d['b_as_a'] or 0):.1f}%)")
    print(f"saved → {out_dir}")
    return overall


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--backbone", type=str, default=cfg.DINO_BACKBONE)
    ap.add_argument("--img-size", type=int, default=cfg.IMG_SIZE)
    ap.add_argument("--fold", type=str, default=None, choices=list(cfg.CV_FOLDS))
    ap.add_argument("--groups", type=str, nargs="+", default=None, help="그룹 (Peregrine 소스) 이름으로 표본 지정")
    ap.add_argument("--which", type=str, default="last", choices=["last", "best", "last_raw"])
    ap.add_argument("--checkpoint", type=str, default=None, help="직접 ckpt 경로 지정 (v6 10-class)")
    ap.add_argument("--resolution", type=str, default="both", choices=["native", "resized", "both"])
    ap.add_argument("--tag-prefix", type=str, default="eval",
                    help="출력 폴더 이름의 앞부분. 뒤에 _<native|resized>_<which> 와 _tta · _pp 가 붙는다")
    ap.add_argument("--tta", action="store_true", help="flip(LR) × flip(UD) × rot180 8-way 평균")
    ap.add_argument("--postprocess", action="store_true", help="휴리스틱 후처리 2 규칙 (PLAN_v6 §8)")
    ap.add_argument("--save-figs", type=int, default=0, help="비교 panel 을 이만큼 저장 (균등 추출, native)")
    ap.add_argument("--max-layers", type=int, default=None, help="스모크용: 앞 N 레이어만")
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

    samples, defect_eval, _ = select_samples(
        enumerate_samples_v6(), args.fold, args.groups,
        fold_run_json=(fig_dir / "run.json") if args.fold else None, img_size=img_size,
    )
    if args.max_layers is not None:
        samples = samples[: args.max_layers]
    if not samples:
        ap.error("채점할 표본이 없다")

    resolutions = RESOLUTIONS if args.resolution == "both" else (args.resolution,)
    suffix = ("_tta" if args.tta else "") + ("_pp" if args.postprocess else "")
    which = Path(ckpt_path).stem if args.checkpoint is not None else args.which
    tags = {r: f"{args.tag_prefix}_{r}_{which}{suffix}" for r in resolutions}
    print(
        f"{len(samples)} 레이어 채점 | fold={args.fold} | img_size={img_size} | 해상도={list(resolutions)} "
        f"| tta={args.tta} pp={args.postprocess} | out={fig_dir}/{{{', '.join(tags.values())}}}"
    )

    panel_dir = fig_dir / tags["native"] / "panels" if "native" in resolutions else None
    res = score_model(
        model, samples, img_size, device, resolutions, tta=args.tta, postprocess=args.postprocess,
        panel_dir=panel_dir, n_panels=args.save_figs,
        panel_titles=("human GT (pbf-2026.11)", f"prediction ({ckpt_path.parent.name}/{ckpt_path.name})"),
    )[""]

    for r in resolutions:
        meta = {
            "kind": "v6",
            "checkpoint": str(ckpt_path),
            "ckpt_version": (ckpt.get("config", {}) or {}).get("version"),
            "ckpt_weights": ckpt.get("weights"),
            "ckpt_step": ckpt.get("step"),
            "run_name": run_name, "fold": args.fold, "tag": tags[r],
            "img_size": img_size, "tta": args.tta, "postprocess": args.postprocess,
            "resolution": r,
            "n_layers": len(samples),
            "samples": [s.key for s in samples],
        }
        print(f"\n################ {tags[r]} ({'원본 해상도' if r == 'native' else f'{img_size}²'}) ################")
        dump_scores(
            res[r]["total"], res[r]["by_group"], fig_dir / tags[r], list(cfg.CLASS_NAMES_V6), defect_eval, meta,
            title=f"{run_name} {cfg.run_tag(args.fold)} / {tags[r]} (human GT, {r})",
        )


if __name__ == "__main__":
    main()
