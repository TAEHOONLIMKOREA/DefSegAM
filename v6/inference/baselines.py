"""기준선 채점 — 새 학습 없이, v6 모델과 같은 검증 레이어 · 같은 정답으로 (PLAN_v6 §6.3, §7.1).

copy — 복사 기준선 (R1)
    검증 레이어마다 **같은 그룹 (소스) 의 가장 비슷한 학습 레이어의 정답을 그대로 예측으로** 낸다. 모델이 없다.
    inlayer 의 값은 낙관적이다 — 같은 빌드의 이웃 레이어가 학습에 있어 "베끼기만 해도" 어느 정도 맞는다. 이 기준선이 그 크기를 잰다.
    v6 모델의 IoU 가 이 값보다 낮은 클래스는 가장 쉬운 조건에서도 배우지 못한 것이다.
      - 비슷함: 두 사진을 img_size² 로 줄이고 7×7 평균으로 더 줄여 사진마다 표준화한 뒤, 두 사진의 상관계수의 평균
      - 복사한 정답의 미레이블 화소는 Powder 로 센다
      - img_size² 에서 채점한다 (같은 소스에도 원본 크기가 다른 레이어가 있어 원본 해상도로는 복사할 수 없다)
      - 같은 그룹의 학습 레이어가 없는 fold (loso) 에서는 낼 수 없다 — 그때는 아무것도 쓰지 않고 끝난다

old — 옛 11-class 모델 (R2 = v5, R3 · B0 = v4, B1 = v5 전수 학습)
    v4 · v5 ckpt 의 11-class 예측을 `V4_TO_NEW` 로 새 id 로 옮겨 새 체계의 정답으로 채점한다 (v5 는 v4 와 클래스 번호가 같다).
      - Swelling 예측: `--swelling se` (기본) 는 SE 로 옮긴다 — 정답 쪽도 Swelling 을 SE 로 옮겼기 때문이다 (D3).
        `none` 은 DIVA 가 지금 하는 대로 "대응 없음" 으로 둔다 (그 화소는 놓친 것으로 센다). `both` 는 둘 다 낸다 (폴더 이름에 _swnone)
      - Under Melting 예측은 언제나 "대응 없음" 이다
      - 옛 모델은 PS 를 낼 수 없다. 옛 IS 예측은 새 IS (2) 로 옮겨지므로 IS · PS 의 IoU 는 견줄 수 없고,
        IS 와 PS 를 합쳐 본 값 (metrics.json 의 is_ps_union_iou) 만 참고로 본다
      - 옛 모델은 Soot 를 미레이블로 두고 배웠다. 새 정답의 SP (대부분 Soot) 에서는 낮게 나오는 것이 정상이다
    행렬은 "대응 없음" 열을 하나 더 둔 11 × 11 이다 (정답 쪽 11번째 행은 비어 있다).

출력 (figures/<run>/v6/<full | cv/<fold>>/):
    ref_copy_resized/                                copy
    <tag-prefix>[_swnone]_<native|resized>/          old
  파일 구성은 evaluate.py 와 같다 (metrics.json, confusion_*.{npy,csv,png}, per_class_metrics.csv).

사용 (CWD = 3DP_VPPM/):
    python -m DefSeg_AM.v6.inference.baselines copy --fold inlayer
    python -m DefSeg_AM.v6.inference.baselines old --fold inlayer --tag-prefix ref_v5 \\
        --checkpoint DefSeg_AM/checkpoints/vitb14_reg_dpt_dual_sz1036_11cls_v5/cv/inlayer/last.pt
    python -m DefSeg_AM.v6.inference.baselines old --fold inlayer --tag-prefix ref_v4 \\
        --checkpoint DefSeg_AM/checkpoints/vitb14_reg_dpt_dual_sz1036_11cls_v4/stage2_best.pt
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ...common.models.model import round_to_patch

from .. import config_v6 as cfg
from ..data.data_gt_v6 import GTPool, enumerate_samples_v6
from .evaluate import RESOLUTIONS, dump_scores, load_model, score_model, select_samples

NONE_ID = cfg.N_CLASSES_V6            # "대응 없음" 예측을 담는 열 (10)
COPY_POOL = 7                         # 복사 기준선: 비슷함을 잴 때 사진을 줄이는 평균 창 (1036 → 148)


# ---------------------------------------------------------------------------
# copy — 복사 기준선
# ---------------------------------------------------------------------------

def _photo_feature(i0: np.ndarray, i1: np.ndarray) -> np.ndarray:
    """(S, S) 두 사진 → 줄이고 사진마다 표준화해 이어 붙인 벡터. 두 벡터의 내적 / 길이 = 두 사진 상관계수의 평균."""
    feats = []
    for a in (i0, i1):
        s = a.shape[0] // COPY_POOL
        a = a[: s * COPY_POOL, : s * COPY_POOL].astype(np.float32)
        a = a.reshape(s, COPY_POOL, s, COPY_POOL).mean(axis=(1, 3))
        feats.append(((a - a.mean()) / (a.std() + 1e-6)).ravel())
    return np.concatenate(feats)


def copy_baseline(pool: GTPool, train_idx: list[int], val_idx: list[int]) -> tuple[np.ndarray, dict, list[dict]]:
    """→ (confusion (C, C), {그룹: confusion}, [{val, copied_from, similarity}]). 같은 그룹의 학습 레이어가 없는 검증 레이어는 건너뛴다."""
    n_cls = cfg.N_CLASSES_V6
    feat = {i: _photo_feature(pool.i0[i], pool.i1[i]) for i in list(train_idx) + list(val_idx)}
    cm_total = np.zeros((n_cls, n_cls), dtype=np.int64)
    cm_by_group: dict[str, np.ndarray] = {}
    copied: list[dict] = []
    for i in val_idx:
        s = pool.samples[i]
        cand = [j for j in train_idx if pool.samples[j].group == s.group]
        if not cand:
            continue
        sims = np.array([float(np.dot(feat[i], feat[j]) / feat[i].size) for j in cand])
        j = cand[int(np.argmax(sims))]
        g = pool.ann[i].ravel().astype(np.int64)
        p = pool.ann[j].ravel().astype(np.int64)
        p = np.where(p >= 0, p, 0)                       # 복사한 정답의 미레이블은 Powder 로
        ok = g >= 0
        cm = np.bincount(g[ok] * n_cls + p[ok], minlength=n_cls * n_cls).reshape(n_cls, n_cls)
        cm_total += cm
        cm_by_group[s.group] = cm_by_group.get(s.group, 0) + cm
        copied.append({"val": s.key, "copied_from": pool.samples[j].key, "similarity": float(sims.max())})
    return cm_total, cm_by_group, copied


def run_copy(args, ap) -> None:
    if args.fold is None:
        ap.error("copy 는 --fold 가 필요하다 (학습 · 검증 레이어가 정해져야 한다)")
    run_name = args.run_name or cfg.default_run_name(args.backbone, args.img_size)
    fig_dir = cfg.run_fig_dir(run_name, args.fold)
    img_size = round_to_patch(args.img_size, 14)
    if img_size % COPY_POOL:
        ap.error(f"img_size {img_size} 가 {COPY_POOL} 의 배수가 아니다")
    val_s, defect_eval, train_s = select_samples(
        enumerate_samples_v6(), args.fold, None, fold_run_json=fig_dir / "run.json", img_size=img_size,
    )
    if args.max_layers is not None:
        val_s = val_s[: args.max_layers]
    groups_train = {s.group for s in train_s}
    if not any(s.group in groups_train for s in val_s):
        print(f"[copy] fold {args.fold}: 검증 레이어와 같은 그룹의 학습 레이어가 없다 (소스를 통째로 뺀 fold) — 복사 기준선을 낼 수 없어 건너뛴다")
        return

    print(f"[copy] fold={args.fold}  학습 {len(train_s)} · 검증 {len(val_s)} 레이어를 {img_size}² 로 줄인다 …")
    pool = GTPool(train_s + val_s, img_size)
    train_idx = list(range(len(train_s)))
    val_idx = list(range(len(train_s), len(train_s) + len(val_s)))
    cm_total, cm_by_group, copied = copy_baseline(pool, train_idx, val_idx)

    tag = "ref_copy_resized"
    sims = [c["similarity"] for c in copied]
    meta = {
        "kind": "copy",
        "run_name": run_name, "fold": args.fold, "tag": tag,
        "img_size": img_size, "resolution": "resized", "tta": False, "postprocess": False,
        "n_layers": len(copied),
        "samples": [c["val"] for c in copied],
        "skipped": [s.key for s in val_s if s.key not in {c["val"] for c in copied}],
        "copied": copied,
        "similarity_median": float(np.median(sims)),
        "note": "prediction = ground truth of the most similar training layer of the same group; its unlabeled pixels count as Powder",
    }
    print(f"\n################ {tag} — 복사 기준선 ({img_size}²) ################")
    print(f"비슷함 (두 사진 상관계수의 평균): 중앙값 {np.median(sims):.3f}, 최소 {np.min(sims):.3f}, 최대 {np.max(sims):.3f}")
    dump_scores(
        cm_total, cm_by_group, fig_dir / tag, list(cfg.CLASS_NAMES_V6), defect_eval, meta,
        title=f"{run_name} {cfg.run_tag(args.fold)} / copy baseline (nearest training layer of the same source, {img_size}²)",
    )


# ---------------------------------------------------------------------------
# old — 옛 11-class 모델을 새 체계로 옮겨 채점
# ---------------------------------------------------------------------------

def old_to_new_lut(swelling: str, device: torch.device) -> torch.Tensor:
    """옛 11-class id → 새 id (대응 없음은 NONE_ID). swelling: "se" | "none"."""
    lut = torch.full((cfg.N_CLASSES_V4,), NONE_ID, dtype=torch.long, device=device)
    for old, new in cfg.V4_TO_NEW.items():
        if new is not None:
            lut[old] = new
    if swelling == "se":
        lut[cfg.V4_SWELLING_ID] = cfg.V4_SWELLING_AS
    elif swelling != "none":
        raise ValueError(f"unknown swelling option: {swelling}")
    return lut


def run_old(args, ap) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or cfg.default_run_name(args.backbone, args.img_size)
    fig_dir = cfg.run_fig_dir(run_name, args.fold)
    ckpt_path = Path(args.checkpoint)
    if not ckpt_path.exists():
        ap.error(f"ckpt not found: {ckpt_path}")

    model, ckpt = load_model(ckpt_path, device, n_classes=cfg.N_CLASSES_V4)
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
    variants = ("se", "none") if args.swelling == "both" else (args.swelling,)
    luts = {v: old_to_new_lut(v, device) for v in variants}
    suffix = ("_tta" if args.tta else "") + ("_pp" if args.postprocess else "")

    def tag_of(v: str, r: str) -> str:
        return f"{args.tag_prefix}{'' if v == 'se' else '_swnone'}_{r}{suffix}"

    print(
        f"{len(samples)} 레이어 채점 | 옛 11-class 모델 {ckpt_path} → {cfg.SCHEMA_VERSION} | fold={args.fold} | img_size={img_size} "
        f"| 해상도={list(resolutions)} | Swelling 예측={list(variants)} | tta={args.tta} pp={args.postprocess}"
    )
    panel_dir = fig_dir / tag_of(variants[0], "native") / "panels" if "native" in resolutions else None
    res = score_model(
        model, samples, img_size, device, resolutions, tta=args.tta, postprocess=args.postprocess,
        luts=luts, n_cm=cfg.N_CLASSES_V6 + 1, panel_dir=panel_dir, n_panels=args.save_figs,
        panel_titles=("human GT (pbf-2026.11)", f"old 11-class model → new ids ({ckpt_path.parent.name}/{ckpt_path.name})"),
    )

    names = list(cfg.CLASS_NAMES_V6) + ["(no class)"]
    c = ckpt.get("config", {}) or {}
    for v in variants:
        for r in resolutions:
            tag = tag_of(v, r)
            meta = {
                "kind": "old",
                "checkpoint": str(ckpt_path),
                "ckpt_version": c.get("version"),
                "ckpt_weights": ckpt.get("weights"),
                "ckpt_step": ckpt.get("step"),
                "old_class_names": list(cfg.V4_CLASS_NAMES),
                "old_to_new": {str(k): (NONE_ID if n is None else n) for k, n in cfg.V4_TO_NEW.items()},
                "swelling_as": "SE (6)" if v == "se" else "no class",
                "none_id": NONE_ID,
                "run_name": run_name, "fold": args.fold, "tag": tag,
                "img_size": img_size, "tta": args.tta, "postprocess": args.postprocess,
                "resolution": r,
                "n_layers": len(samples),
                "samples": [s.key for s in samples],
                "note": "old models cannot output PS; old IS predictions map to new IS — compare IS/PS only through is_ps_union_iou",
            }
            print(f"\n################ {tag} — Swelling 예측 → {meta['swelling_as']} "
                  f"({'원본 해상도' if r == 'native' else f'{img_size}²'}) ################")
            dump_scores(
                res[v][r]["total"], res[v][r]["by_group"], fig_dir / tag, names, defect_eval, meta,
                title=f"{ckpt_path.parent.name}/{ckpt_path.name} → {cfg.SCHEMA_VERSION} / {tag} (human GT, {r})",
                none_id=NONE_ID,
            )


def main():
    ap = argparse.ArgumentParser(description="v6 기준선 채점 (학습 없음)")
    sub = ap.add_subparsers(dest="kind", required=True)

    def common(p):
        p.add_argument("--run-name", type=str, default=None, help="결과를 둘 v6 run (기본: 기본 run 이름)")
        p.add_argument("--backbone", type=str, default=cfg.DINO_BACKBONE)
        p.add_argument("--img-size", type=int, default=cfg.IMG_SIZE)
        p.add_argument("--fold", type=str, default=None, choices=list(cfg.CV_FOLDS))
        p.add_argument("--max-layers", type=int, default=None, help="스모크용: 앞 N 레이어만")

    p_copy = sub.add_parser("copy", help="복사 기준선 (R1)")
    common(p_copy)

    p_old = sub.add_parser("old", help="옛 11-class 모델을 새 체계로 옮겨 채점 (R2 · R3 · B0 · B1)")
    common(p_old)
    p_old.add_argument("--checkpoint", type=str, required=True, help="v4 · v5 의 11-class ckpt")
    p_old.add_argument("--tag-prefix", type=str, required=True, help="출력 폴더 이름의 앞부분 (예: ref_v5, ref_v4)")
    p_old.add_argument("--groups", type=str, nargs="+", default=None)
    p_old.add_argument("--resolution", type=str, default="both", choices=["native", "resized", "both"])
    p_old.add_argument("--swelling", type=str, default="se", choices=["se", "none", "both"],
                       help="옛 모델의 Swelling 예측을 어디로 옮길지 (기본 se — 정답 쪽 D3 과 같게)")
    p_old.add_argument("--tta", action="store_true")
    p_old.add_argument("--postprocess", action="store_true")
    p_old.add_argument("--save-figs", type=int, default=0)

    args = ap.parse_args()
    if args.kind == "copy":
        run_copy(args, ap)
    else:
        run_old(args, ap)


if __name__ == "__main__":
    main()
