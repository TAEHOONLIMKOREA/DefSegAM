"""v5 검증 fold 요약 — 곡선을 모아 표 · 그림 · 권장 길이를 낸다 (학습 없음, GPU 불필요). PLAN_v5 §6.1.

읽는 것 (fold 마다, 학습이 남긴 파일):
    figures/<run>/v5/cv/<fold>/curve.csv        step 별 검증 지표 (학습용 가중치 · EMA 사본)
    figures/<run>/v5/cv/<fold>/run.json         학습 · 검증 표본, 채점 클래스
    figures/<run>/v5/cv/<fold>/eval_native_last/metrics.json   (있으면) native 해상도 채점
    figures/<run>/v5/cv/<fold>/ref_v4/metrics.json             (있으면) 같은 표본에서의 v4

쓰는 것 (figures/<run>/v5/cv/):
    summary.csv, summary.md, summary.json, curves.png

권장 길이: loso fold 들 (둘 이상) 의 EMA 결함 mIoU 를 step 별로 평균한 곡선의 최고점 P 를 찾는다.
  - 최고점이 마지막 10% 구간에 있으면 아직 오르는 중일 수 있으므로 줄이지 않는다.
  - 끝 값이 최고점의 95% 이상이면 (최고점 뒤로 거의 내려가지 않았으면) 과적합으로 보지 않고 줄이지 않는다.
  - 그 밖에는 TOTAL_STEPS ≈ 1.5 × P 를 제안한다 (cosine 일정이 끝까지 내려가도록 여유를 둔 어림값).
loso fold 가 하나뿐이면 평균 곡선이 없어 권장 길이를 내지 않는다.

사용 (CWD = 3DP_VPPM/):
    python -m DefSeg_AM.v5.scripts.summarize_cv
    python -m DefSeg_AM.v5.scripts.summarize_cv --img-size 224        # --quick 스모크 결과
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .. import config_v5 as cfg

# 그림 색 — dataviz 기준 팔레트의 categorical 1 · 2 번 (light). 검증: validate_palette --pairs all 통과
# (CVD ΔE 24.7, normal ΔE 33.6, 둘 다 surface 대비 3:1 이상).
C_EMA = "#2a78d6"        # EMA 사본 (저장 · 배포되는 가중치)
C_RAW = "#eb6834"        # 학습용 가중치
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def load_fold(fold_dir: Path) -> dict | None:
    """fold 폴더 → dict. 곡선이 비어 있으면 None."""
    curve_path = fold_dir / "curve.csv"
    run_path = fold_dir / "run.json"
    if not (curve_path.exists() and run_path.exists()):
        return None
    with curve_path.open() as fh:
        rows = list(csv.DictReader(fh))
    rows = [r for r in rows if r.get("ema_defect_miou") not in (None, "") or r.get("ema_miou") not in (None, "")]
    if not rows:
        return None
    meta = json.loads(run_path.read_text())
    steps = np.array([int(r["step"]) for r in rows])
    out = {
        "fold": fold_dir.name,
        "meta": meta,
        "steps": steps,
        "ema_defect": np.array([_f(r.get("ema_defect_miou")) for r in rows]),
        "raw_defect": np.array([_f(r.get("raw_defect_miou")) for r in rows]),
        "ema_miou": np.array([_f(r.get("ema_miou")) for r in rows]),
        "raw_miou": np.array([_f(r.get("raw_miou")) for r in rows]),
        "ema_acc": np.array([_f(r.get("ema_acc")) for r in rows]),
        "ema_iou_final": [_f(rows[-1].get(f"ema_iou_{c}")) for c in range(cfg.N_CLASSES_V5)],
        "native": None, "native_ref_v4": None,
    }
    for key, sub in (("native", "eval_native_last"), ("native_ref_v4", "ref_v4")):
        p = fold_dir / sub / "metrics.json"
        if p.exists():
            out[key] = json.loads(p.read_text())["overall"]
    return out


def fold_row(f: dict) -> dict:
    """fold 하나의 요약 수치."""
    meta = f["meta"]
    sel = f["ema_defect"]
    if np.all(np.isnan(sel)):           # 채점할 결함 클래스가 없는 fold
        sel = f["ema_miou"]
    i_best = int(np.nanargmax(sel))
    val_sources = sorted({k.split("/")[0] for k in meta["val_samples"]})
    return {
        "fold": f["fold"],
        "val_sources": val_sources,
        "n_train": len(meta["train_samples"]),
        "n_val": len(meta["val_samples"]),
        "defect_classes": [cfg.CLASS_ABBRS_V5[c] for c in meta["defect_eval_classes"]],
        "total_steps": int(f["steps"][-1]),
        "best_step": int(f["steps"][i_best]),
        "best_ema_defect_miou": float(sel[i_best]),
        "final_ema_defect_miou": float(f["ema_defect"][-1]),
        "final_raw_defect_miou": float(f["raw_defect"][-1]),
        "final_ema_miou": float(f["ema_miou"][-1]),
        "final_ema_acc": float(f["ema_acc"][-1]),
        "native_defect_miou": (f["native"] or {}).get("defect_miou"),
        "native_miou": (f["native"] or {}).get("miou"),
        "native_ref_v4_defect_miou": (f["native_ref_v4"] or {}).get("defect_miou"),
        "native_ref_v4_miou": (f["native_ref_v4"] or {}).get("miou"),
    }


def mean_loso_curve(folds: list[dict]) -> tuple[np.ndarray, np.ndarray, list[str]] | None:
    """loso fold 들의 EMA 결함 mIoU 를 공통 step 에서 평균."""
    loso = [f for f in folds if f["fold"].startswith("loso_") and not np.all(np.isnan(f["ema_defect"]))]
    if len(loso) < 2:
        return None
    common = set(loso[0]["steps"].tolist())
    for f in loso[1:]:
        common &= set(f["steps"].tolist())
    if not common:
        return None
    steps = np.array(sorted(common))
    vals = []
    for f in loso:
        lut = dict(zip(f["steps"].tolist(), f["ema_defect"].tolist()))
        vals.append([lut[s] for s in steps.tolist()])
    return steps, np.nanmean(np.array(vals), axis=0), [f["fold"] for f in loso]


# 끝 값이 최고점의 (1 − PLATEAU_TOL) 이상이면 최고점 뒤의 하락을 과적합으로 보지 않는다.
# 첫 실제 실행 (2026-11-17, loso_L21, 10,000 step) 에서 EMA 곡선은 최고점 (step 5,750, 0.1843) 뒤로
# 0.179–0.184 사이를 오르내리다 0.1792 로 끝났다 (최고점의 97.2%). 이런 곡선에 1.5 × P 를 그대로 내면
# 근거 없이 길이를 바꾸게 된다.
PLATEAU_TOL = 0.05


def recommend(steps: np.ndarray, mean: np.ndarray) -> dict:
    """평균 곡선 → 권장 TOTAL_STEPS · EMA decay."""
    total = int(steps[-1])
    i_peak = int(np.nanargmax(mean))
    peak_step = int(steps[i_peak])
    peak, final = float(mean[i_peak]), float(mean[-1])
    ratio = final / peak if peak > 0 else float("nan")
    spacing = int(steps[1] - steps[0]) if len(steps) > 1 else total
    late = peak_step >= 0.9 * total
    flat = (not late) and peak > 0 and final >= (1.0 - PLATEAU_TOL) * peak
    where = (
        f"평균 곡선의 최고점은 step {peak_step:,} (결함 mIoU {peak:.4f}) 이고 끝 (step {total:,}) 에서는 "
        f"{final:.4f} 다 (최고점의 {100 * ratio:.1f}%)."
    )
    if late:
        rec_steps = total
        note = (
            f"평균 곡선의 최고점이 마지막 10% 구간 (step {peak_step:,} / {total:,}) 에 있다. "
            f"아직 오르는 중일 수 있으므로 줄이지 않는다 — {total:,} step 그대로 두거나 더 길게 한 번 확인한다."
        )
    elif flat:
        rec_steps = total
        note = (
            f"{where} 최고점 뒤로 {100 * PLATEAU_TOL:.0f}% 넘게 내려가지 않았으므로 과적합으로 보지 않는다 — "
            f"{total:,} step 그대로 둔다."
        )
    else:
        rec_steps = int(min(total, math.ceil(1.5 * peak_step / spacing) * spacing))
        note = (
            f"{where} 최고점 뒤로 {100 * PLATEAU_TOL:.0f}% 넘게 내려갔다 (과적합). "
            f"권장 TOTAL_STEPS ≈ 1.5 × {peak_step:,} = {rec_steps:,}. "
            f"길이를 바꾸면 검증 fold 도 모두 그 길이로 다시 돌려야 이 표와 곡선이 그 길이의 결과가 된다 "
            f"(일부만 다시 돌리면 길이가 섞인다)."
        )
    # EMA 규칙: 시상수 1 / (1 − decay) ≤ 총 step / 10
    rec_decay = min(cfg.EMA_DECAY, 1.0 - 10.0 / max(rec_steps, 20))
    return {
        "peak_step": peak_step, "peak_value": peak, "final_value": final,
        "final_over_peak": None if ratio != ratio else float(ratio),
        "run_total_steps": total, "recommended_total_steps": rec_steps,
        "recommended_ema_decay": round(rec_decay, 4),
        "peak_in_last_10pct": bool(late), "kept_because_flat": bool(flat), "note": note,
    }


# ---------------------------------------------------------------------------
# 그림 — fold 마다 한 칸 (small multiples), 칸마다 EMA · 학습용 가중치 두 선
# ---------------------------------------------------------------------------

def _compact(v: float, _pos=None) -> str:
    if v >= 1000:
        s = f"{v / 1000:.1f}".rstrip("0").rstrip(".")
        return f"{s}K"
    return f"{int(v)}"


def plot_curves(folds: list[dict], mean_curve, rec: dict | None, out_path: Path, run_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D
    from matplotlib.ticker import FuncFormatter, MaxNLocator

    panels: list[tuple[str, dict | None]] = [("fold", f) for f in folds]
    if mean_curve is not None:
        panels.append(("mean", None))
    n = len(panels)
    ncols = min(3, n)
    nrows = int(math.ceil(n / ncols))

    y_max = 0.0
    for f in folds:
        for k in ("ema_defect", "raw_defect"):
            if not np.all(np.isnan(f[k])):
                y_max = max(y_max, float(np.nanmax(f[k])))
    y_top = min(1.0, max(0.2, math.ceil((y_max + 0.03) * 10) / 10))
    total = max(int(f["steps"][-1]) for f in folds)

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(5.4 * ncols, 3.5 * nrows + 0.9),
        sharex=True, sharey=True, squeeze=False, facecolor=SURFACE,
    )

    def style(ax, col: int, row: int) -> None:
        ax.set_facecolor(SURFACE)
        ax.grid(True, axis="y", color=GRID, linewidth=0.75, linestyle="-")
        ax.set_axisbelow(True)
        for side in ("top", "right", "left"):
            ax.spines[side].set_visible(False)
        ax.spines["bottom"].set_color(AXIS)
        ax.spines["bottom"].set_linewidth(0.75)
        ax.tick_params(axis="both", colors=INK_MUTED, labelsize=8, length=0)
        ax.set_xlim(0, total * 1.16)             # 오른쪽 여백 = 끝점 라벨 자리
        ax.set_ylim(0, y_top)
        ax.xaxis.set_major_locator(MaxNLocator(nbins=5, integer=True))
        ax.xaxis.set_major_formatter(FuncFormatter(_compact))
        ax.yaxis.set_major_locator(MaxNLocator(nbins=5))
        if col == 0:
            ax.set_ylabel("defect mIoU (held-out human GT)", color=INK_SECONDARY, fontsize=8)
        if row == nrows - 1:
            ax.set_xlabel("optimizer step", color=INK_SECONDARY, fontsize=8)

    def end_dot(ax, x, y, color) -> None:
        ax.plot([x], [y], marker="o", markersize=6.5, markerfacecolor=color,
                markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="none", zorder=5)

    for k, (kind, f) in enumerate(panels):
        row, col = divmod(k, ncols)
        ax = axes[row][col]
        style(ax, col, row)
        if kind == "fold":
            st = f["steps"]
            ax.plot(st, f["raw_defect"], color=C_RAW, linewidth=1.5,
                    solid_joinstyle="round", solid_capstyle="round", zorder=3)
            ax.plot(st, f["ema_defect"], color=C_EMA, linewidth=1.5,
                    solid_joinstyle="round", solid_capstyle="round", zorder=4)
            r = fold_row(f)
            y_end = f["ema_defect"][-1]
            if y_end == y_end:
                end_dot(ax, st[-1], y_end, C_EMA)
                ax.annotate(f"{y_end:.3f}", (st[-1], y_end), xytext=(7, 0), textcoords="offset points",
                            va="center", ha="left", fontsize=8, color=INK_PRIMARY)
            if r["best_step"] != int(st[-1]) and not np.all(np.isnan(f["ema_defect"])):
                yb = r["best_ema_defect_miou"]
                end_dot(ax, r["best_step"], yb, C_EMA)
                ax.annotate(f"best {yb:.3f} @ {r['best_step']:,}", (r["best_step"], yb),
                            xytext=(0, 9), textcoords="offset points",
                            va="bottom", ha="center", fontsize=8, color=INK_PRIMARY)
            val_src = ", ".join(cfg.SOURCE_SHORT.get(s, s) for s in r["val_sources"])
            ax.set_title(f"{f['fold']}", loc="left", fontsize=10, color=INK_PRIMARY, pad=18)
            ax.text(0.0, 1.03,
                    f"val: {val_src} ({r['n_val']} layers) · classes: {' '.join(r['defect_classes']) or '-'}",
                    transform=ax.transAxes, fontsize=7.5, color=INK_SECONDARY, va="bottom", ha="left")
        else:
            steps, mean, names = mean_curve
            ax.plot(steps, mean, color=C_EMA, linewidth=1.5,
                    solid_joinstyle="round", solid_capstyle="round", zorder=4)
            i_peak = int(np.nanargmax(mean))
            end_dot(ax, steps[i_peak], mean[i_peak], C_EMA)
            ax.annotate(f"peak {mean[i_peak]:.3f} @ {int(steps[i_peak]):,}", (steps[i_peak], mean[i_peak]),
                        xytext=(0, 9), textcoords="offset points",
                        va="bottom", ha="center", fontsize=8, color=INK_PRIMARY)
            ax.set_title("LOSO mean (EMA)", loc="left", fontsize=10, color=INK_PRIMARY, pad=18)
            sub = f"mean of {len(names)} leave-one-source-out folds"
            if rec is not None:
                sub += f" · suggested total steps {rec['recommended_total_steps']:,}"
            ax.text(0.0, 1.03, sub, transform=ax.transAxes, fontsize=7.5,
                    color=INK_SECONDARY, va="bottom", ha="left")

    for k in range(n, nrows * ncols):
        row, col = divmod(k, ncols)
        axes[row][col].set_visible(False)
    # 숨긴 칸 위의 칸은 x 눈금 라벨을 직접 켠다
    for col in range(ncols):
        shown = [row for row in range(nrows) if axes[row][col].get_visible()]
        if shown and shown[-1] != nrows - 1:
            ax = axes[shown[-1]][col]
            ax.tick_params(labelbottom=True)
            ax.set_xlabel("optimizer step", color=INK_SECONDARY, fontsize=8)

    handles = [
        Line2D([0], [0], color=C_EMA, linewidth=1.5, label="EMA copy (saved weights)"),
        Line2D([0], [0], color=C_RAW, linewidth=1.5, label="training weights (raw)"),
    ]
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.012, 0.955),
                     ncol=2, frameon=False, fontsize=8.5, handlelength=1.8)
    for t in leg.get_texts():
        t.set_color(INK_PRIMARY)
    fig.suptitle(f"v5 validation folds — {run_name}", x=0.012, ha="left",
                 fontsize=11.5, color=INK_PRIMARY, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=144, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 표
# ---------------------------------------------------------------------------

def _n(v, nd: int = 4) -> str:
    return "–" if v is None or (isinstance(v, float) and v != v) else f"{v:.{nd}f}"


def write_tables(folds: list[dict], rows: list[dict], rec: dict | None, out_dir: Path, run_name: str) -> str:
    fields = [
        "fold", "val_sources", "n_train", "n_val", "defect_classes", "total_steps",
        "best_step", "best_ema_defect_miou", "final_ema_defect_miou", "final_raw_defect_miou",
        "final_ema_miou", "final_ema_acc",
        "native_defect_miou", "native_miou", "native_ref_v4_defect_miou", "native_ref_v4_miou",
    ]
    with (out_dir / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            rr = dict(r)
            rr["val_sources"] = " ".join(r["val_sources"])
            rr["defect_classes"] = " ".join(r["defect_classes"])
            w.writerow(rr)

    img = folds[0]["meta"]["config"]["img_size"]
    lines = [f"# v5 검증 fold 요약 — `{run_name}`", ""]
    lines += [
        f"결함 mIoU 는 그 fold 에서 학습 화소와 검증 화소가 모두 있는 결함 클래스의 IoU 평균이다 (떼어 둔 사람 GT, {img}²).",
        "native 열은 같은 표본을 원본 해상도로 채점한 값이고, v4 열은 같은 표본에서의 v4 `stage2_best.pt` 다",
        "(v4 는 이 표본들을 학습에 썼으므로 v4 쪽에 유리한 비교다).",
        "",
        "| fold | 검증 소스 | 학습 / 검증 레이어 | 채점 결함 클래스 | best step | best (EMA) | 끝 (EMA) | 끝 (학습용) | native (v5) | native (v4, 참고) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['fold']}` | {', '.join(cfg.SOURCE_SHORT.get(s, s) for s in r['val_sources'])} "
            f"| {r['n_train']} / {r['n_val']} | {' '.join(r['defect_classes']) or '–'} "
            f"| {r['best_step']:,} | {_n(r['best_ema_defect_miou'])} | {_n(r['final_ema_defect_miou'])} "
            f"| {_n(r['final_raw_defect_miou'])} | {_n(r['native_defect_miou'])} | {_n(r['native_ref_v4_defect_miou'])} |"
        )
    lines += ["", "## 클래스별 IoU — 마지막 step, EMA 사본", "",
              "| fold | " + " | ".join(cfg.CLASS_ABBRS_V5) + " |",
              "|---|" + "---:|" * cfg.N_CLASSES_V5]
    for f in folds:
        meta = f["meta"]
        cells = []
        for c in range(cfg.N_CLASSES_V5):
            v = f["ema_iou_final"][c]
            if meta["class_val_px"][c] == 0:
                cells.append("–")
            elif meta["class_train_px"][c] == 0:
                cells.append(f"{_n(v, 3)} †")
            else:
                cells.append(_n(v, 3))
        lines.append(f"| `{f['fold']}` | " + " | ".join(cells) + " |")
    lines += ["", "– 검증 화소 없음. † 학습 화소가 0 인 클래스 (그 소스에만 있는 클래스 — 결함 mIoU 에서 뺀다)."]
    if rec is not None:
        lines += ["", "## 권장 길이", "", rec["note"],
                  f"그 길이에서 EMA 규칙 (시상수 ≤ 총 step / 10) 을 지키는 decay 는 {rec['recommended_ema_decay']} 이하다."]
    else:
        n_loso = sum(1 for f in folds if f["fold"].startswith("loso_"))
        lines += ["", "## 권장 길이", "",
                  f"아직 내지 않았다 — 검증 step 이 겹치는 loso fold 가 둘 이상 있어야 평균 곡선을 낸다 (지금 loso fold {n_loso}개). "
                  "그 전에는 위 표의 best step 과 `curves.png` 를 본다."]
    text = "\n".join(lines) + "\n"
    (out_dir / "summary.md").write_text(text)
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--backbone", type=str, default=cfg.DINO_BACKBONE)
    ap.add_argument("--img-size", type=int, default=cfg.IMG_SIZE)
    ap.add_argument("--no-plot", action="store_true")
    args = ap.parse_args()

    run_name = args.run_name or cfg.default_run_name(args.backbone, args.img_size)
    cv_dir = cfg.FIGURE_DIR / run_name / "v5" / "cv"
    if not cv_dir.is_dir():
        ap.error(f"검증 fold 결과가 없다: {cv_dir}")

    order = {name: i for i, name in enumerate(cfg.CV_FOLDS)}
    fold_dirs = sorted([d for d in cv_dir.iterdir() if d.is_dir()], key=lambda d: order.get(d.name, 999))
    folds = [f for f in (load_fold(d) for d in fold_dirs) if f is not None]
    if not folds:
        ap.error(f"곡선이 있는 fold 가 없다: {cv_dir}")

    rows = [fold_row(f) for f in folds]
    mean_curve = mean_loso_curve(folds)
    rec = recommend(mean_curve[0], mean_curve[1]) if mean_curve is not None else None

    text = write_tables(folds, rows, rec, cv_dir, run_name)
    with (cv_dir / "summary.json").open("w") as fh:
        json.dump({
            "run_name": run_name, "folds": rows,
            "loso_mean": None if mean_curve is None else {
                "folds": mean_curve[2], "steps": mean_curve[0].tolist(),
                "ema_defect_miou": [None if v != v else float(v) for v in mean_curve[1]],
            },
            "recommendation": rec,
        }, fh, ensure_ascii=False, indent=1)
    if not args.no_plot:
        plot_curves(folds, mean_curve, rec, cv_dir / "curves.png", run_name)
    print(text)
    print(f"saved → {cv_dir}/{{summary.csv, summary.md, summary.json" + ("" if args.no_plot else ", curves.png") + "}")


if __name__ == "__main__":
    main()
