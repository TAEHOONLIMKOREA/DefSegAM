"""학습된 DefSegModel 의 validation confusion matrix 평가 (학습 없음, 1-pass).

어떤 결함 class 가 어떤 class 로 혼동되는지 → 병합(merge) 후보 분석용.

- Stage 1: ORNL cached val (Build 1, DSCNN pred GT) — 대규모, 통계적으로 안정
- Stage 2: DSCNN_Dataset val (Maraging, human GT) — 소규모

출력 (out_dir, 기본 figures/<run>/stage<ckpt>_model_on_stage<data>_data/confusion/):
    confusion_counts.npy          : (N, N) int64 — rows=GT, cols=Pred
    confusion_matrix.csv          : 위 행렬 + class 이름 헤더
    per_class_metrics.csv         : support / IoU / precision / recall / top-confusion
    confusion_row_normalized.png  : row-정규화 히트맵 (GT 기준 예측 분포)

사용:
    python -m DefSeg_AM.v1.inference.confusion --run-name <run> --stage 1
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

from .. import config
from ...common.models.model import DefSegModel, round_to_patch


# ---------------------------------------------------------------------------
# Confusion accumulation
# ---------------------------------------------------------------------------

@torch.no_grad()
def accumulate_confusion(
    model: DefSegModel,
    loader: DataLoader,
    device: torch.device,
    n_classes: int,
    log_every: int = 20,
) -> np.ndarray:
    """val loader 1-pass → (N, N) confusion counts (rows=GT, cols=Pred)."""
    cm = torch.zeros(n_classes * n_classes, dtype=torch.long, device=device)
    model.eval()
    n_batches = len(loader)
    for step, batch in enumerate(loader):
        img0 = batch["img0"].to(device, non_blocking=True)
        img1 = batch["img1"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)
        logits = model(img0, img1)
        pred = logits.argmax(dim=1)
        valid = label != config.IGNORE_INDEX
        g = label[valid].reshape(-1)
        p = pred[valid].reshape(-1)
        idx = g * n_classes + p                       # flatten (gt, pred)
        cm += torch.bincount(idx, minlength=n_classes * n_classes)
        if step % log_every == 0:
            print(f"  [{step:4d}/{n_batches}] confusion 누적 …", flush=True)
    return cm.reshape(n_classes, n_classes).cpu().numpy()


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


def _fmt(x: float) -> str:
    return "nan" if (x != x) else f"{x:.4f}"


def save_heatmap(cm: np.ndarray, path: Path, names: list[str]) -> None:
    """row-정규화 히트맵: 각 GT 행이 1 이 되도록 (= GT 기준 예측 분포)."""
    n = cm.shape[0]
    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=np.float64),
                     where=row_sum > 0)
    fig, ax = plt.subplots(figsize=(11, 9))
    im = ax.imshow(norm, cmap="viridis", vmin=0, vmax=1)
    ax.set_xticks(range(n)); ax.set_yticks(range(n))
    ax.set_xticklabels([f"{c}" for c in range(n)])
    ax.set_yticklabels([f"{c} {names[c]}" for c in range(n)])
    ax.set_xlabel("Predicted class"); ax.set_ylabel("Ground-truth class")
    ax.set_title("Row-normalized confusion (GT 기준 예측 분포)")
    for i in range(n):
        for j in range(n):
            v = norm[i, j]
            if v >= 0.01:
                ax.text(j, i, f"{v:.2f}", ha="center", va="center",
                        color="white" if v < 0.6 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of GT-class pixels")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


# ---------------------------------------------------------------------------
# Dataset / loader per stage
# ---------------------------------------------------------------------------

def build_val_loader(stage: int, img_size: int, batch_size: int, num_workers: int,
                     split: str = "val"):
    if stage == 1:
        from ..data.build_cache_stage1 import cache_dir_for
        from ..data.data_ornl import DefSegORNLCachedDataset
        cache_root = cache_dir_for(img_size)
        ds = DefSegORNLCachedDataset(cache_root, split, img_size, training=False)
        builds = "Build 1" if split == "val" else "Builds 2-5"
        desc = f"ORNL cached {split} ({builds}) — {len(ds)} layers"
    elif stage == 2:
        from ..data.data_dscnn import DefSegDSCNNDataset, split_train_val
        train_specs, val_specs = split_train_val()
        specs = val_specs if split == "val" else train_specs
        ds = DefSegDSCNNDataset(specs, img_size=img_size, training=False)
        srcs = config.DSCNN_VAL_SOURCE_NAMES if split == "val" else ["train sources"]
        desc = f"DSCNN_Dataset {split} ({','.join(srcs)}) — {len(ds)} layers"
    else:
        raise ValueError(stage)
    loader = DataLoader(
        ds, batch_size=batch_size, shuffle=False,
        num_workers=num_workers, pin_memory=True, drop_last=False,
    )
    return loader, desc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, default=None,
                    help="checkpoints/<run-name>/stage<stage>_best.pt 자동 탐색")
    ap.add_argument("--stage", type=int, default=1, choices=[1, 2])
    ap.add_argument("--split", type=str, default="val", choices=["val", "train"],
                    help="train: 결함 풍부한 빌드까지 평가해 12x12 완성 (stage1=B2-5)")
    ap.add_argument("--checkpoint", type=str, default=None,
                    help="직접 ckpt 경로 (run-name 우회)")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=None,
                    help="미지정 시 ckpt config 의 img_size 사용")
    ap.add_argument("--out-dir", type=str, default=None)
    ap.add_argument("--ckpt-stage", type=int, default=None, choices=[1, 2],
                    help="평가할 체크포인트 stage (미지정 시 --stage). data-stage 와 분리 가능")
    ap.add_argument("--data-stage", type=int, default=None, choices=[1, 2],
                    help="평가에 쓸 데이터셋 stage (미지정 시 --stage). "
                         "예: --ckpt-stage 1 --data-stage 2 → Stage1 모델을 사람 GT(stage2)로 평가")
    args = ap.parse_args()

    # ckpt/data stage 분리 (기본은 --stage 로 둘 다)
    ckpt_stage = args.ckpt_stage or args.stage
    data_stage = args.data_stage or args.stage
    cross = ckpt_stage != data_stage

    if args.checkpoint is None:
        if args.run_name is None:
            ap.error("--run-name 또는 --checkpoint 중 하나는 필요")
        args.checkpoint = str(config.CHECKPOINT_DIR / args.run_name / f"stage{ckpt_stage}_best.pt")

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = torch.load(args.checkpoint, map_location=device, weights_only=False)
    cfg = ckpt["config"]
    model = DefSegModel(backbone_name=cfg["backbone"]).to(device)
    model.load_trainable_state_dict(ckpt["model_state"])
    print(f"loaded {args.checkpoint} "
          f"(epoch={ckpt.get('epoch','?')}, val_acc={ckpt.get('val_acc','?')}, "
          f"miou={ckpt.get('miou','?')})")

    img_size = args.img_size or cfg.get("img_size", config.IMG_SIZE)
    img_size = round_to_patch(img_size, model.patch_size)

    loader, desc = build_val_loader(data_stage, img_size, args.batch_size,
                                    args.num_workers, split=args.split)
    tag = (f"ckpt=stage{ckpt_stage} | data=stage{data_stage} (CROSS-EVAL)"
           if cross else f"stage{ckpt_stage}")
    print(f"{tag} | {desc} | img_size={img_size}")

    cm = accumulate_confusion(model, loader, device, config.N_CLASSES)
    m = per_class_metrics(cm)
    names = config.ORNL_CLASS_NAMES

    # ----- output dir -----
    if args.out_dir is not None:
        out_dir = Path(args.out_dir)
    else:
        run = args.run_name or "adhoc"
        sub = "confusion" if args.split == "val" else f"confusion_{args.split}"
        stage_dir = f"stage{ckpt_stage}_model_on_stage{data_stage}_data"
        out_dir = config.FIGURE_DIR / run / stage_dir / sub
    out_dir.mkdir(parents=True, exist_ok=True)

    np.save(out_dir / "confusion_counts.npy", cm)
    save_csv_matrix(cm, out_dir / "confusion_matrix.csv", names)
    save_metrics_csv(cm, m, out_dir / "per_class_metrics.csv", names)
    save_heatmap(cm, out_dir / "confusion_row_normalized.png", names)

    # ----- console report -----
    valid_iou = [m["iou"][c] for c in range(config.N_CLASSES) if m["support"][c] > 0 and m["iou"][c] == m["iou"][c]]
    miou = float(np.mean(valid_iou)) if valid_iou else float("nan")
    total = m["support"].sum()
    print("\n=== Per-class metrics (sorted by GT pixel support) ===")
    print(f"{'#':>2} {'class':<22} {'support':>13} {'frac%':>7} {'IoU':>7} {'prec':>7} {'recall':>7}  top-confusion")
    tc = top_confusions_per_class(cm)
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

    print(f"\nsaved → {out_dir}")
    print("  - confusion_counts.npy / confusion_matrix.csv")
    print("  - per_class_metrics.csv")
    print("  - confusion_row_normalized.png")


if __name__ == "__main__":
    main()
