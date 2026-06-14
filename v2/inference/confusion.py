"""v2 DefSegModel 의 confusion matrix 평가 (학습 없음, 1-pass) — 8-class.

v1 [DefSeg_AM/v1/inference/confusion.py](../../v1/inference/confusion.py) 의 v2 대응판.
아키텍처 규칙(v1↔v2 상호 import 금지)에 따라 generic 한 metric/save helper 는
v2/inference/infer.py 처럼 이 파일 안에 self-contained 로 둔다 (공유는 common 만).

평가 모드:
  - --stage 1            : stage1_best.pt → ORNL v2 cached val(Build1) / --split train(B2-5)
  - --stage 2 --fold k   : stage2_best_fold{k}_*.pt → fold k 의 held-out val source
  - --stage 2 --cv       : 8 fold 각각을 자기 held-out source 에 평가 → fold별 + 통합 행렬
                           (= 진짜 leave-one-source-out cross-val confusion)

출력 (기본 figures/<run>/v2/<tag>/confusion/):
    confusion_counts.npy          : (8, 8) int64 — rows=GT, cols=Pred
    confusion_matrix.csv          : 위 행렬 + class 이름 헤더
    per_class_metrics.csv         : support / IoU / precision / recall / top-confusion
    confusion_row_normalized.png  : row-정규화 히트맵 (GT 기준 예측 분포)

사용:
    python -m DefSeg_AM.v2.inference.confusion --run-name <run> --stage 1
    python -m DefSeg_AM.v2.inference.confusion --run-name <run> --stage 2 --cv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from ...common.models.model import DefSegModel, round_to_patch
from .. import config_v2 as cfg


# ---------------------------------------------------------------------------
# Confusion accumulation
# ---------------------------------------------------------------------------

@torch.no_grad()
def accumulate_confusion(
    model: DefSegModel,
    loader: DataLoader,
    device: torch.device,
    n_classes: int,
    cm: torch.Tensor | None = None,
    log_every: int = 20,
    max_batches: int | None = None,
) -> torch.Tensor:
    """loader 1-pass → (N*N,) confusion counts 누적 (rows=GT, cols=Pred).

    cm 를 넘기면 그 위에 누적 (cross-val 통합용). 반환은 flat tensor.
    """
    if cm is None:
        cm = torch.zeros(n_classes * n_classes, dtype=torch.long, device=device)
    model.eval()
    n_batches = len(loader)
    for step, batch in enumerate(loader):
        if max_batches is not None and step >= max_batches:
            print(f"  [max_batches={max_batches}] 조기 종료", flush=True)
            break
        img0 = batch["img0"].to(device, non_blocking=True)
        img1 = batch["img1"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        logits = model(img0, img1)
        pred = logits.argmax(dim=1)
        valid = label != cfg.IGNORE_INDEX
        g = label[valid].reshape(-1)
        p = pred[valid].reshape(-1)
        idx = g * n_classes + p                       # flatten (gt, pred)
        cm += torch.bincount(idx, minlength=n_classes * n_classes)
        if step % log_every == 0:
            print(f"  [{step:4d}/{n_batches}] confusion 누적 …", flush=True)
    return cm


# ---------------------------------------------------------------------------
# Metrics derived from confusion matrix
# ---------------------------------------------------------------------------

def per_class_metrics(cm: np.ndarray) -> dict:
    cm = cm.astype(np.float64)
    tp = np.diag(cm)
    support = cm.sum(axis=1)            # GT pixel count per class (row sum)
    pred_sum = cm.sum(axis=0)           # predicted count per class (col sum)
    fp = pred_sum - tp
    fn = support - tp
    with np.errstate(divide="ignore", invalid="ignore"):
        iou = tp / (tp + fp + fn)
        recall = tp / support
        precision = tp / pred_sum
    return {
        "support": support, "tp": tp, "fp": fp, "fn": fn,
        "iou": iou, "recall": recall, "precision": precision,
    }


def top_confusions_per_class(cm: np.ndarray) -> list[tuple[int, int, float]]:
    """각 GT class 가 가장 많이 새는 (잘못 예측되는) 다른 class. (gt, pred, frac)."""
    out = []
    for i in range(cm.shape[0]):
        row = cm[i].astype(np.float64).copy()
        support = row.sum()
        row[i] = -1                                    # 자기 자신 제외
        j = int(row.argmax())
        frac = (cm[i, j] / support) if support > 0 else float("nan")
        out.append((i, j, frac))
    return out


def merge_candidate_pairs(cm: np.ndarray, top_k: int = 8) -> list[tuple[int, int, float, int]]:
    """대칭 혼동 점수 = (cm[i,j]+cm[j,i]) / (support_i + support_j). 높을수록 병합 후보."""
    n = cm.shape[0]
    support = cm.sum(axis=1).astype(np.float64)
    pairs = []
    for i in range(n):
        for j in range(i + 1, n):
            denom = support[i] + support[j]
            if denom <= 0:
                continue
            cross = cm[i, j] + cm[j, i]
            score = cross / denom
            pairs.append((i, j, float(score), int(cross)))
    pairs.sort(key=lambda x: x[2], reverse=True)
    return pairs[:top_k]


# ---------------------------------------------------------------------------
# Save artifacts
# ---------------------------------------------------------------------------

def _fmt(x: float) -> str:
    return "nan" if (x != x) else f"{x:.4f}"


def save_csv_matrix(cm: np.ndarray, path: Path, names: list[str]) -> None:
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["GT\\Pred"] + [f"{c}:{names[c]}" for c in range(len(names))])
        for i in range(cm.shape[0]):
            w.writerow([f"{i}:{names[i]}"] + cm[i].tolist())


def save_metrics_csv(cm: np.ndarray, m: dict, path: Path, names: list[str]) -> None:
    tc = top_confusions_per_class(cm)
    with path.open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["class", "name", "support_px", "IoU", "precision", "recall",
                    "top_confused_with", "confused_frac"])
        for c in range(len(names)):
            gt, pj, frac = tc[c]
            w.writerow([
                c, names[c], int(m["support"][c]),
                _fmt(m["iou"][c]), _fmt(m["precision"][c]), _fmt(m["recall"][c]),
                f"{pj}:{names[pj]}", _fmt(frac),
            ])


def save_heatmap(cm: np.ndarray, path: Path, names: list[str], title: str | None = None) -> None:
    """row-정규화 히트맵: 각 GT 행이 1 이 되도록 (= GT 기준 예측 분포)."""
    n = cm.shape[0]
    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=np.float64),
                     where=row_sum > 0)
    fig, ax = plt.subplots(figsize=(9, 7.5))
    im = ax.imshow(norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([f"{c}" for c in range(n)])
    ax.set_yticklabels([f"{c} {names[c]}" for c in range(n)])
    ax.set_xlabel("Predicted class"); ax.set_ylabel("Ground-truth class")
    ax.set_title(title or "Row-normalized confusion (GT 기준 예측 분포)")
    for i in range(n):
        for j in range(n):
            v = norm[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.6 else "black", fontsize=8)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of GT-class pixels")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def dump_all(cm: np.ndarray, out_dir: Path, names: list[str], title: str | None = None) -> dict:
    """confusion 행렬 1개를 out_dir 에 4종 산출물로 저장하고 metric dict 반환."""
    out_dir.mkdir(parents=True, exist_ok=True)
    m = per_class_metrics(cm)
    np.save(out_dir / "confusion_counts.npy", cm)
    save_csv_matrix(cm, out_dir / "confusion_matrix.csv", names)
    save_metrics_csv(cm, m, out_dir / "per_class_metrics.csv", names)
    save_heatmap(cm, out_dir / "confusion_row_normalized.png", names, title=title)
    return m


def report(cm: np.ndarray, names: list[str], n_classes: int) -> None:
    """console per-class metric + merge 후보 요약."""
    m = per_class_metrics(cm)
    tc = top_confusions_per_class(cm)
    valid_iou = [m["iou"][c] for c in range(n_classes)
                 if m["support"][c] > 0 and m["iou"][c] == m["iou"][c]]
    miou = float(np.mean(valid_iou)) if valid_iou else float("nan")
    total = m["support"].sum()
    print("\n=== Per-class metrics (sorted by GT pixel support) ===")
    print(f"{'#':>2} {'class':<22} {'support':>13} {'frac%':>7} {'IoU':>7} {'prec':>7} {'recall':>7}  top-confusion")
    order = np.argsort(-m["support"])
    for c in order:
        gt, pj, frac = tc[c]
        sup = int(m["support"][c])
        fracpct = 100 * sup / total if total > 0 else 0
        conf = f"→ {pj}:{names[pj]} ({_fmt(frac)})" if sup > 0 else "(no GT pixels)"
        print(f"{c:>2} {names[c]:<22} {sup:>13,} {fracpct:>6.3f}% "
              f"{_fmt(m['iou'][c]):>7} {_fmt(m['precision'][c]):>7} {_fmt(m['recall'][c]):>7}  {conf}")
    print(f"\nmIoU (support>0 classes) = {miou:.4f}   |   total valid GT pixels = {total:,.0f}")
    print("\n=== Merge-candidate pairs (대칭 혼동 점수 상위) ===")
    print(f"{'pair':<46} {'score':>8} {'cross_px':>12}")
    for i, j, score, cross in merge_candidate_pairs(cm):
        label = f"{i}:{names[i]}  <->  {j}:{names[j]}"
        print(f"{label:<46} {score:>8.4f} {cross:>12,}")


# ---------------------------------------------------------------------------
# Model / loader
# ---------------------------------------------------------------------------

def load_model(ckpt_path: Path, device: torch.device) -> tuple[DefSegModel, int]:
    """ckpt → (model, img_size). v2 infer.py 와 동일한 로드 규약."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ckpt["config"]
    model = DefSegModel(
        backbone_name=c["backbone"],
        n_classes=c.get("n_classes", cfg.N_CLASSES_V2),
    ).to(device)
    model.load_trainable_state_dict(ckpt["model_state"])
    img_size = c.get("img_size", cfg.IMG_SIZE)
    print(f"loaded {ckpt_path.name}  "
          f"(epoch={ckpt.get('epoch','?')}, val_acc={ckpt.get('val_acc','?')}, "
          f"miou={ckpt.get('miou','?')}, img_size={img_size})")
    return model, img_size


def stage1_loader(split: str, img_size: int, batch_size: int, num_workers: int):
    from ..data.data_ornl_v2 import DefSegORNLCachedDatasetV2
    ds = DefSegORNLCachedDatasetV2(split, img_size, training=False)
    builds = "Build 1" if split == "val" else "Builds 2-5"
    desc = f"ORNL v2 cached {split} ({builds}) — {len(ds)} layers"
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, drop_last=False)
    return loader, desc


def stage2_fold_val_loader(fold: int, img_size: int, batch_size: int, num_workers: int):
    """fold 의 held-out val source loader."""
    from ..data.data_dscnn_v2 import split_train_val_by_fold, DefSegDSCNNDatasetV2
    _, val_specs, val_source = split_train_val_by_fold(fold)
    ds = DefSegDSCNNDatasetV2(val_specs, img_size=img_size, training=False)
    desc = f"DSCNN val fold{fold} (held-out: {val_source}) — {len(ds)} layers"
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, drop_last=False)
    return loader, desc, val_source


def find_fold_ckpt(run_dir: Path, fold: int) -> Path:
    matches = sorted(run_dir.glob(f"stage2_best_fold{fold}_*.pt"))
    if not matches:
        raise FileNotFoundError(f"no stage2 fold {fold} ckpt in {run_dir}")
    return matches[0]


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, default="vits14_dpt_dual_sz1036_8cls_v2")
    ap.add_argument("--stage", type=int, default=2, choices=[1, 2])
    ap.add_argument("--split", type=str, default="val", choices=["val", "train"],
                    help="stage1 전용: train 은 결함 풍부한 B2-5 로 8x8 채움")
    ap.add_argument("--fold", type=int, default=None,
                    help="stage2 단일 fold confusion (0..7). --cv 와 상호배타")
    ap.add_argument("--cv", action="store_true",
                    help="stage2: 8 fold 전부 held-out 평가 → fold별 + 통합 행렬")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=None,
                    help="미지정 시 ckpt config 의 img_size 사용")
    ap.add_argument("--max-batches", type=int, default=None,
                    help="스모크용: loader 앞 N 배치만 (정식 평가는 미지정)")
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    if args.fold is not None and args.cv:
        ap.error("--fold 와 --cv 동시 사용 불가")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = cfg.CHECKPOINT_DIR / args.run_name
    names = cfg.ORNL_CLASS_NAMES_V2
    n_classes = cfg.N_CLASSES_V2
    fig_root = cfg.FIGURE_DIR / args.run_name / "v2"

    # ============================ Stage 1 ============================
    if args.stage == 1:
        model, ck_size = load_model(run_dir / "stage1_best.pt", device)
        img_size = round_to_patch(args.img_size or ck_size, model.patch_size)
        loader, desc = stage1_loader(args.split, img_size, args.batch_size, args.num_workers)
        print(f"stage1 | {desc} | img_size={img_size}")
        cm = accumulate_confusion(model, loader, device, n_classes,
                                  max_batches=args.max_batches)
        cm = cm.reshape(n_classes, n_classes).cpu().numpy()
        sub = "confusion" if args.split == "val" else f"confusion_{args.split}"
        out_dir = Path(args.out_dir) if args.out_dir else fig_root / "stage1" / sub
        dump_all(cm, out_dir, names, title=f"v2 stage1 ({args.split})")
        report(cm, names, n_classes)
        print(f"\nsaved → {out_dir}")
        return

    # ============================ Stage 2 ============================
    if args.fold is not None:
        # 단일 fold
        ckpt = find_fold_ckpt(run_dir, args.fold)
        model, ck_size = load_model(ckpt, device)
        img_size = round_to_patch(args.img_size or ck_size, model.patch_size)
        loader, desc, val_source = stage2_fold_val_loader(
            args.fold, img_size, args.batch_size, args.num_workers)
        print(f"stage2 fold{args.fold} | {desc} | img_size={img_size}")
        cm = accumulate_confusion(model, loader, device, n_classes,
                                  max_batches=args.max_batches)
        cm = cm.reshape(n_classes, n_classes).cpu().numpy()
        out_dir = (Path(args.out_dir) if args.out_dir
                   else fig_root / f"stage2_fold{args.fold}_{val_source}" / "confusion")
        dump_all(cm, out_dir, names, title=f"v2 stage2 fold{args.fold} ({val_source})")
        report(cm, names, n_classes)
        print(f"\nsaved → {out_dir}")
        return

    # --cv (기본): 8 fold leave-one-source-out 통합
    if not args.cv:
        print("[v2/confusion] stage2 는 --fold k 또는 --cv 가 필요 — 기본 --cv 로 진행")
    cv_root = fig_root / "stage2_cv"
    agg = torch.zeros(n_classes * n_classes, dtype=torch.long, device=device)
    for fold in range(cfg.N_FOLDS):
        ckpt = find_fold_ckpt(run_dir, fold)
        model, ck_size = load_model(ckpt, device)
        img_size = round_to_patch(args.img_size or ck_size, model.patch_size)
        loader, desc, val_source = stage2_fold_val_loader(
            fold, img_size, args.batch_size, args.num_workers)
        print(f"\n--- fold {fold}/{cfg.N_FOLDS} | {desc} | img_size={img_size} ---")
        fcm = accumulate_confusion(model, loader, device, n_classes,
                                   max_batches=args.max_batches)
        agg += fcm
        fcm_np = fcm.reshape(n_classes, n_classes).cpu().numpy()
        fold_dir = cv_root / f"fold{fold}_{val_source}"
        dump_all(fcm_np, fold_dir, names, title=f"v2 stage2 fold{fold} ({val_source})")
        print(f"  saved fold → {fold_dir}")

    agg_np = agg.reshape(n_classes, n_classes).cpu().numpy()
    out_dir = Path(args.out_dir) if args.out_dir else cv_root / "confusion"
    dump_all(agg_np, out_dir, names, title="v2 stage2 8-fold CV (leave-one-source-out)")
    print("\n========== 8-fold CV 통합 confusion ==========")
    report(agg_np, names, n_classes)
    print(f"\nsaved 통합 → {out_dir}")
    print(f"saved fold별 → {cv_root}/fold*_<source>/")


if __name__ == "__main__":
    main()
