"""v4 DefSegModel 의 confusion matrix 평가 (학습 없음, 1-pass) — 10-class.

v4 는 cross-validation 이 없고 stage 2 도 단일 ckpt 라 fold/cv 옵션 없음.
평가 대상은 ORNL v4 cached split (Stage 1/2 모두 동일).

평가 모드:
  - --stage 1  : stage1_best.pt
  - --stage 2  : stage2_best.pt   (default)

출력 (기본 figures/<run>/v4/<tag>/confusion/):
    confusion_counts.npy
    confusion_matrix.csv
    per_class_metrics.csv
    confusion_row_normalized.png

사용:
    python -m DefSeg_AM.v4.inference.confusion --run-name <run> --stage 2
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
from .. import config_v4 as cfg


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
    """loader 1-pass → flat confusion counts 누적 (rows=GT, cols=Pred)."""
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
        idx = g * n_classes + p
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
    support = cm.sum(axis=1)
    pred_sum = cm.sum(axis=0)
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
    out = []
    for i in range(cm.shape[0]):
        row = cm[i].astype(np.float64).copy()
        support = row.sum()
        row[i] = -1
        j = int(row.argmax())
        frac = (cm[i, j] / support) if support > 0 else float("nan")
        out.append((i, j, frac))
    return out


def merge_candidate_pairs(cm: np.ndarray, top_k: int = 8) -> list[tuple[int, int, float, int]]:
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
    n = cm.shape[0]
    row_sum = cm.sum(axis=1, keepdims=True)
    norm = np.divide(cm, row_sum, out=np.zeros_like(cm, dtype=np.float64),
                     where=row_sum > 0)
    fig, ax = plt.subplots(figsize=(10, 8.5))
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
                        color="white" if v < 0.6 else "black", fontsize=7)
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="fraction of GT-class pixels")
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


def dump_all(cm: np.ndarray, out_dir: Path, names: list[str], title: str | None = None) -> dict:
    out_dir.mkdir(parents=True, exist_ok=True)
    m = per_class_metrics(cm)
    np.save(out_dir / "confusion_counts.npy", cm)
    save_csv_matrix(cm, out_dir / "confusion_matrix.csv", names)
    save_metrics_csv(cm, m, out_dir / "per_class_metrics.csv", names)
    save_heatmap(cm, out_dir / "confusion_row_normalized.png", names, title=title)
    return m


def report(cm: np.ndarray, names: list[str], n_classes: int) -> None:
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
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ckpt["config"]
    model = DefSegModel(
        backbone_name=c["backbone"],
        n_classes=c.get("n_classes", cfg.N_CLASSES_V4),
    ).to(device)
    model.load_trainable_state_dict(ckpt["model_state"])
    img_size = c.get("img_size", cfg.IMG_SIZE)
    print(f"loaded {ckpt_path.name}  "
          f"(epoch={ckpt.get('epoch','?')}, val_acc={ckpt.get('val_acc','?')}, "
          f"miou={ckpt.get('miou','?')}, img_size={img_size})")
    return model, img_size


def ornl_loader(split: str, img_size: int, batch_size: int, num_workers: int):
    from ..data.data_ornl_v4 import DefSegORNLCachedDatasetV4
    ds = DefSegORNLCachedDatasetV4(split, img_size, training=False)
    builds = "Build 1" if split == "val" else "Builds 2-5"
    desc = f"ORNL v4 cached {split} ({builds}) — {len(ds)} layers"
    loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                        num_workers=num_workers, pin_memory=True, drop_last=False)
    return loader, desc


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, default="vits14_dpt_dual_sz1036_11cls_v4")
    ap.add_argument("--stage", type=int, default=2, choices=[1, 2])
    ap.add_argument("--split", type=str, default="val", choices=["val", "train"],
                    help="val=Build 1, train=Builds 2-5")
    ap.add_argument("--batch-size", type=int, default=2)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--img-size", type=int, default=None,
                    help="미지정 시 ckpt config 의 img_size 사용")
    ap.add_argument("--max-batches", type=int, default=None,
                    help="스모크용: loader 앞 N 배치만")
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_dir = cfg.CHECKPOINT_DIR / args.run_name
    names = cfg.ORNL_CLASS_NAMES_V4
    n_classes = cfg.N_CLASSES_V4
    fig_root = cfg.FIGURE_DIR / args.run_name / "v4"

    ckpt = run_dir / ("stage1_best.pt" if args.stage == 1 else "stage2_best.pt")
    if not ckpt.exists():
        ap.error(f"ckpt not found: {ckpt}")

    model, ck_size = load_model(ckpt, device)
    img_size = round_to_patch(args.img_size or ck_size, model.patch_size)
    loader, desc = ornl_loader(args.split, img_size, args.batch_size, args.num_workers)
    print(f"stage{args.stage} | {desc} | img_size={img_size}")
    cm = accumulate_confusion(model, loader, device, n_classes, max_batches=args.max_batches)
    cm = cm.reshape(n_classes, n_classes).cpu().numpy()
    sub = "confusion" if args.split == "val" else f"confusion_{args.split}"
    out_dir = Path(args.out_dir) if args.out_dir else fig_root / f"stage{args.stage}" / sub
    dump_all(cm, out_dir, names, title=f"v4 stage{args.stage} ({args.split})")
    report(cm, names, n_classes)
    print(f"\nsaved → {out_dir}")


if __name__ == "__main__":
    main()
