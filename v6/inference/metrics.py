"""v6 공용 지표 — confusion matrix 누적과 거기서 나오는 수치 · 파일 (v5 사본 + PLAN_v6 §6.2 의 지표).

학습 중 검증 (training/train.py), 사람 GT 채점 (inference/evaluate.py), 기준선 채점 (inference/baselines.py) 이 함께 쓴다.

confusion matrix 는 항상 rows = 정답, cols = 예측.

v5 에 없던 것:
  - miss_false_alarm      놓침 (결함 정답을 정상으로 예측) · 잘못 짚음 (정상 정답을 결함으로 예측) 의 비율
  - pair_confusion        정의가 맞닿은 클래스 쌍 (IS↔PS, RS↔RH, SE↔DB, SP↔OM) 의 쌍 안 혼동률
  - merged_iou            여러 클래스를 하나로 합쳐 본 IoU (예: IS ∪ PS)
  - 기준선용 "대응 없음" 열: 옛 11-class 모델의 Swelling · Under Melting 예측은 새 체계에 갈 곳이 없다.
    그런 예측을 담는 열을 하나 더 둔 (C+1) × (C+1) 행렬을 쓰고 (정답 쪽 행은 비어 있다), 위 함수들은 none_id 로 그 열을 받는다
"""
from __future__ import annotations

import csv
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from .. import config_v6 as cfg


# ---------------------------------------------------------------------------
# Confusion accumulation
# ---------------------------------------------------------------------------

def confusion_from_pred(
    pred: torch.Tensor, label: torch.Tensor, n_classes: int,
) -> torch.Tensor:
    """(…) pred, (…) label → flat (n_classes²,) long counts. IGNORE 화소는 뺀다."""
    valid = label != cfg.IGNORE_INDEX
    g = label[valid].reshape(-1).long()
    p = pred[valid].reshape(-1).long()
    return torch.bincount(g * n_classes + p, minlength=n_classes * n_classes)


@torch.no_grad()
def accumulate_confusion(
    model,
    loader: DataLoader,
    device: torch.device,
    n_classes: int,
    log_every: int | None = None,
    max_batches: int | None = None,
) -> np.ndarray:
    """loader 1-pass → (n_classes, n_classes) int64 confusion counts."""
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
        pred = model(img0, img1).argmax(dim=1)
        cm += confusion_from_pred(pred, label, n_classes)
        if log_every and step % log_every == 0:
            print(f"  [{step:4d}/{n_batches}] confusion 누적 …", flush=True)
    return cm.reshape(n_classes, n_classes).cpu().numpy()


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


def summarize(cm: np.ndarray, defect_eval_classes: list[int] | None = None) -> dict:
    """confusion matrix → 요약 수치.

    Returns dict:
        acc          : 전체 화소 정확도
        miou         : 정답 화소가 있는 클래스의 IoU 평균
        defect_miou  : defect_eval_classes 중 정답 화소가 있는 클래스의 IoU 평균
                       (None 이면 결함 클래스 2..9 전체에서 고른다). 해당 클래스가 없으면 nan.
        iou          : (n_classes,) 클래스별 IoU. 정답 화소가 없는 클래스는 nan.
        defect_classes: defect_miou 에 들어간 클래스 번호
    """
    m = per_class_metrics(cm)
    support = m["support"]
    iou = np.where(support > 0, m["iou"], np.nan)
    total = support.sum()
    acc = float(np.diag(cm).sum() / total) if total > 0 else float("nan")
    with_support = [c for c in range(cm.shape[0]) if support[c] > 0]
    miou = float(np.mean([iou[c] for c in with_support])) if with_support else float("nan")
    cand = cfg.DEFECT_IDS_V6 if defect_eval_classes is None else defect_eval_classes
    used = [int(c) for c in cand if support[c] > 0]
    defect_miou = float(np.mean([iou[c] for c in used])) if used else float("nan")
    return {
        "acc": acc, "miou": miou, "defect_miou": defect_miou,
        "iou": iou, "defect_classes": used,
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
# v6 — 놓침 · 잘못 짚음, 혼동 쌍, 합친 클래스 (PLAN_v6 §6.2)
# ---------------------------------------------------------------------------

def _ratio(num: float, den: float) -> float:
    return float(num) / float(den) if den > 0 else float("nan")


def miss_false_alarm(
    cm: np.ndarray,
    normal_ids: list[int] | None = None,
    defect_ids: list[int] | None = None,
    none_id: int | None = None,
) -> dict:
    """놓침과 잘못 짚음 — IoU 가 낮은 까닭이 어느 쪽인지 가린다.

    miss         : 결함 정답 화소 가운데 정상 (Powder · Printed) 으로 예측한 비율
    to_none      : 결함 정답 화소 가운데 "대응 없음" 으로 예측한 비율 (none_id 가 있을 때. 화면에는 미레이블로 나오므로 놓친 것과 같다)
    wrong_defect : 결함 정답 화소 가운데 다른 결함으로 예측한 비율
    false_alarm  : 정상 정답 화소 가운데 결함으로 예측한 비율
    """
    normal = list(cfg.NORMAL_IDS_V6 if normal_ids is None else normal_ids)
    defect = list(cfg.DEFECT_IDS_V6 if defect_ids is None else defect_ids)
    cm = np.asarray(cm, dtype=np.float64)
    d_total = cm[defect].sum()
    n_total = cm[normal].sum()
    d_to_normal = cm[np.ix_(defect, normal)].sum()
    d_correct = sum(cm[c, c] for c in defect)
    d_to_none = cm[defect, none_id].sum() if none_id is not None else 0.0
    d_to_defect = cm[np.ix_(defect, defect)].sum() - d_correct
    return {
        "defect_px": int(d_total), "normal_px": int(n_total),
        "miss": _ratio(d_to_normal, d_total),
        "to_none": _ratio(d_to_none, d_total),
        "wrong_defect": _ratio(d_to_defect, d_total),
        "correct": _ratio(d_correct, d_total),
        "false_alarm": _ratio(cm[np.ix_(normal, defect)].sum(), n_total),
    }


def pair_confusion(cm: np.ndarray, a: int, b: int) -> dict:
    """클래스 쌍 (a, b) 의 혼동.

    within_pair : 정답도 예측도 이 둘 가운데 하나인 화소에서, 둘을 맞바꾼 비율  (a→b + b→a) / (a→a + a→b + b→a + b→b)
    a_as_b      : a 정답 화소 전체 가운데 b 로 예측한 비율 (b_as_a 도 같다)
    """
    cm = np.asarray(cm, dtype=np.float64)
    aa, ab, ba, bb = cm[a, a], cm[a, b], cm[b, a], cm[b, b]
    return {
        "a": int(a), "b": int(b),
        "within_pair": _ratio(ab + ba, aa + ab + ba + bb),
        "a_as_b": _ratio(ab, cm[a].sum()),
        "b_as_a": _ratio(ba, cm[b].sum()),
        "a_px": int(cm[a].sum()), "b_px": int(cm[b].sum()),
    }


def all_pair_confusions(cm: np.ndarray) -> dict[str, dict]:
    return {name: pair_confusion(cm, a, b) for name, a, b in cfg.CONFUSION_PAIRS}


def merged_iou(cm: np.ndarray, ids: list[int]) -> float:
    """ids 의 클래스들을 한 클래스로 합쳤을 때의 IoU (정답 화소가 없으면 nan)."""
    cm = np.asarray(cm, dtype=np.float64)
    ids = list(ids)
    tp = cm[np.ix_(ids, ids)].sum()
    gt = cm[ids].sum()
    pred = cm[:, ids].sum()
    return _ratio(tp, gt + pred - tp) if gt > 0 else float("nan")


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
    # matplotlib 은 여기서만 쓴다 — 학습 프로세스 (DDP rank) 가 import 하지 않게 함수 안에서 부른다
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

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
    ax.set_title(title or "Row-normalized confusion")
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
