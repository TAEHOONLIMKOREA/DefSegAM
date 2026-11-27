"""v6 검증 fold 요약 — 곡선과 채점 결과를 모아 표 · 그림을 낸다 (학습 없음, GPU 불필요). PLAN_v6 §7.1 의 읽는 순서를 따른다.

읽는 것 (fold 마다):
    figures/<run>/v6/cv/<fold>/curve.csv                        step 별 검증 지표 (학습용 가중치 · EMA 사본, 학습 해상도)
    figures/<run>/v6/cv/<fold>/run.json                         학습 · 검증 표본, 클래스별 화소 수, 배선 확인 결과
    figures/<run>/v6/cv/<fold>/eval_{native,resized}_last/metrics.json      (있으면) v6 `last.pt` 의 채점
    figures/<run>/v6/cv/<fold>/ref_copy_resized/metrics.json                (있으면) R1 복사 기준선
    figures/<run>/v6/cv/<fold>/ref_v5_{native,resized}/metrics.json         (있으면) R2 — 같은 fold 의 v5 모델을 새 id 로 옮긴 것
    figures/<run>/v6/cv/<fold>/ref_v4_{native,resized}/metrics.json         (있으면) R3 — v4 `stage2_best.pt` 를 새 id 로 옮긴 것
    figures/<v5 run>/v5/cv/<fold>/curve.csv                     (있으면) 같은 이름의 v5 fold 곡선 — 클래스별 곡선 그림의 참고선

쓰는 것:
    figures/<run>/v6/cv/{summary.md, summary.csv, summary.json, curves.png}
    figures/<run>/v6/cv/<fold>/class_curves.png                 클래스별 IoU 곡선 (v6 · v5 참고선 · 복사 기준선)

권장 길이는 v5 와 같은 규칙이다: loso fold 들 (둘 이상) 의 EMA 결함 mIoU 를 step 별로 평균한 곡선의 최고점 P 를 찾는다.
  - 최고점이 마지막 10% 구간에 있으면 아직 오르는 중일 수 있으므로 줄이지 않는다.
  - 끝 값이 최고점의 95% 이상이면 과적합으로 보지 않고 줄이지 않는다.
  - 그 밖에는 TOTAL_STEPS ≈ 1.5 × P 를 제안한다.
inlayer fold 의 곡선은 길이를 정하는 근거로 쓰지 않는다 (낙관적인 값이다 — PLAN_v6 §7.1).

사용 (CWD = 3DP_VPPM/):
    python -m DefSeg_AM.v6.scripts.summarize_cv
    python -m DefSeg_AM.v6.scripts.summarize_cv --img-size 224        # --quick 스모크 결과
"""
from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import numpy as np

from .. import config_v6 as cfg

# 그림 색 — dataviz 기준 팔레트 (light). EMA · 학습용 가중치는 categorical 1 · 2 번이고 (v5 의 같은 그림과 같은 색.
# 기준 팔레트 문서의 검증값: CVD ΔE 24.7, normal ΔE 33.6, 둘 다 surface 대비 3:1 이상),
# 클래스별 곡선은 "하나가 주인공이고 나머지는 배경" 이므로 v6 만 색 (1번), 참고선 (v5) 은 회색이다.
C_EMA = "#2a78d6"        # EMA 사본 (저장 · 배포되는 가중치)
C_RAW = "#eb6834"        # 학습용 가중치
C_REF = "#898781"        # 참고선 (v5 의 같은 fold) — 강조하지 않는 회색
SURFACE = "#fcfcfb"
INK_PRIMARY = "#0b0b0b"
INK_SECONDARY = "#52514e"
INK_MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"

N_CLS = cfg.N_CLASSES_V6
AB = cfg.CLASS_ABBRS_V6

# 새 id → 옛 11-class id. 정의가 그대로인 클래스만 v5 의 곡선을 참고선으로 그린다 (PLAN_v6 §7.1-2).
# IS (옛 IS = 새 IS ∪ PS) · SE (Swelling 이 합쳐졌다) · SP · OM (Soot 가 들어왔다) 은 정답이 달라 같은 양이 아니다.
SAME_DEFINITION_OLD_ID = {0: 0, 1: 1, 4: 3, 5: 2, 9: 6}

# 채점 결과 폴더 (fold 폴더 안)
EVAL_DIRS = {
    "v6_native": "eval_native_last", "v6_resized": "eval_resized_last",
    "copy_resized": "ref_copy_resized",
    "v5_native": "ref_v5_native", "v5_resized": "ref_v5_resized",
    "v4_native": "ref_v4_native", "v4_resized": "ref_v4_resized",
}


def _f(x) -> float:
    try:
        return float(x)
    except (TypeError, ValueError):
        return float("nan")


def _read_curve(path: Path) -> list[dict]:
    with path.open() as fh:
        rows = list(csv.DictReader(fh))
    return [r for r in rows if r.get("ema_defect_miou") not in (None, "") or r.get("ema_miou") not in (None, "")]


def load_fold(fold_dir: Path) -> dict | None:
    """fold 폴더 → dict. 곡선이 비어 있으면 None."""
    curve_path = fold_dir / "curve.csv"
    run_path = fold_dir / "run.json"
    if not (curve_path.exists() and run_path.exists()):
        return None
    rows = _read_curve(curve_path)
    if not rows:
        return None
    meta = json.loads(run_path.read_text())
    out = {
        "fold": fold_dir.name,
        "dir": fold_dir,
        "meta": meta,
        "steps": np.array([int(r["step"]) for r in rows]),
        "ema_defect": np.array([_f(r.get("ema_defect_miou")) for r in rows]),
        "raw_defect": np.array([_f(r.get("raw_defect_miou")) for r in rows]),
        "ema_miou": np.array([_f(r.get("ema_miou")) for r in rows]),
        "raw_miou": np.array([_f(r.get("raw_miou")) for r in rows]),
        "ema_acc": np.array([_f(r.get("ema_acc")) for r in rows]),
        "ema_iou": np.array([[_f(r.get(f"ema_iou_{c}")) for c in range(N_CLS)] for r in rows]),   # (T, C)
        "evals": {},
        "v5_curve": None,
    }
    for key, sub in EVAL_DIRS.items():
        p = fold_dir / sub / "metrics.json"
        if p.exists():
            out["evals"][key] = json.loads(p.read_text())
    # 같은 이름의 v5 fold 곡선 (참고선). 학습 해상도가 같을 때만 견줄 수 있다
    v5_dir = cfg.FIGURE_DIR / cfg.V5_RUN_NAME / "v5" / "cv" / fold_dir.name
    if (v5_dir / "curve.csv").exists() and (v5_dir / "run.json").exists():
        v5_meta = json.loads((v5_dir / "run.json").read_text())
        same_val = sorted(v5_meta.get("val_samples", [])) == sorted(meta.get("val_samples", []))
        if v5_meta["config"]["img_size"] == meta["config"]["img_size"] and same_val:
            v5_rows = _read_curve(v5_dir / "curve.csv")
            if v5_rows:
                out["v5_curve"] = {
                    "steps": np.array([int(r["step"]) for r in v5_rows]),
                    "ema_iou": {new: np.array([_f(r.get(f"ema_iou_{old}")) for r in v5_rows])
                                for new, old in SAME_DEFINITION_OLD_ID.items()},
                }
    return out


def _ov(f: dict, key: str) -> dict | None:
    """fold 의 채점 결과 하나의 overall (없으면 None)."""
    doc = f["evals"].get(key)
    return None if doc is None else doc["overall"]


def fold_row(f: dict) -> dict:
    """fold 하나의 요약 수치."""
    meta = f["meta"]
    sel = f["ema_defect"]
    if np.all(np.isnan(sel)):           # 채점할 결함 클래스가 없는 fold
        sel = f["ema_miou"]
    i_best = int(np.nanargmax(sel))
    val_groups = sorted({k.split("/")[0] for k in meta["val_samples"]})
    wiring = meta.get("wiring_check")

    def g(key: str, field: str):
        ov = _ov(f, key)
        return None if ov is None else ov.get(field)

    return {
        "fold": f["fold"],
        "val_groups": val_groups,
        "n_train": len(meta["train_samples"]),
        "n_val": len(meta["val_samples"]),
        "defect_classes": [AB[c] for c in meta["defect_eval_classes"]],
        "wiring_match": None if wiring is None else bool(wiring["match"]),
        "total_steps": int(f["steps"][-1]),
        "best_step": int(f["steps"][i_best]),
        "best_ema_defect_miou": float(sel[i_best]),
        "final_ema_defect_miou": float(f["ema_defect"][-1]),
        "final_raw_defect_miou": float(f["raw_defect"][-1]),
        "final_ema_miou": float(f["ema_miou"][-1]),
        "final_ema_acc": float(f["ema_acc"][-1]),
        "native_defect_miou": g("v6_native", "defect_miou"),
        "native_miou": g("v6_native", "miou"),
        "resized_defect_miou": g("v6_resized", "defect_miou"),
        "copy_resized_defect_miou": g("copy_resized", "defect_miou"),
        "v5_native_defect_miou": g("v5_native", "defect_miou"),
        "v5_resized_defect_miou": g("v5_resized", "defect_miou"),
        "v4_native_defect_miou": g("v4_native", "defect_miou"),
        "v4_resized_defect_miou": g("v4_resized", "defect_miou"),
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


# 끝 값이 최고점의 (1 − PLATEAU_TOL) 이상이면 최고점 뒤의 하락을 과적합으로 보지 않는다 (v5 와 같은 값 —
# v5 loso_L21 의 EMA 곡선은 최고점 뒤로 0.179–0.184 사이를 오르내리다 최고점의 97.2% 로 끝났다).
PLATEAU_TOL = 0.05


def recommend(steps: np.ndarray, mean: np.ndarray) -> dict:
    """평균 곡선 → 권장 TOTAL_STEPS · EMA decay (v5 와 같은 규칙)."""
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
# 그림 1 — fold 마다 한 칸 (small multiples), 칸마다 EMA · 학습용 가중치 두 선 (v5 와 같은 그림)
# ---------------------------------------------------------------------------

def _compact(v: float, _pos=None) -> str:
    if v >= 1000:
        s = f"{v / 1000:.1f}".rstrip("0").rstrip(".")
        return f"{s}K"
    return f"{int(v)}"


def _style_axis(ax, total: int, y_top: float, show_ylabel: bool, show_xlabel: bool, ylabel: str) -> None:
    from matplotlib.ticker import FuncFormatter, MaxNLocator

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
    if show_ylabel:
        ax.set_ylabel(ylabel, color=INK_SECONDARY, fontsize=8)
    if show_xlabel:
        ax.set_xlabel("optimizer step", color=INK_SECONDARY, fontsize=8)


def _end_dot(ax, x, y, color) -> None:
    ax.plot([x], [y], marker="o", markersize=6.5, markerfacecolor=color,
            markeredgecolor=SURFACE, markeredgewidth=1.5, linestyle="none", zorder=5)


def plot_curves(folds: list[dict], mean_curve, rec: dict | None, out_path: Path, run_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

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

    for k, (kind, f) in enumerate(panels):
        row, col = divmod(k, ncols)
        ax = axes[row][col]
        _style_axis(ax, total, y_top, col == 0, row == nrows - 1, "defect mIoU (held-out human GT)")
        if kind == "fold":
            st = f["steps"]
            ax.plot(st, f["raw_defect"], color=C_RAW, linewidth=1.5,
                    solid_joinstyle="round", solid_capstyle="round", zorder=3)
            ax.plot(st, f["ema_defect"], color=C_EMA, linewidth=1.5,
                    solid_joinstyle="round", solid_capstyle="round", zorder=4)
            r = fold_row(f)
            y_end = f["ema_defect"][-1]
            if y_end == y_end:
                _end_dot(ax, st[-1], y_end, C_EMA)
                ax.annotate(f"{y_end:.3f}", (st[-1], y_end), xytext=(7, 0), textcoords="offset points",
                            va="center", ha="left", fontsize=8, color=INK_PRIMARY)
            if r["best_step"] != int(st[-1]) and not np.all(np.isnan(f["ema_defect"])):
                yb = r["best_ema_defect_miou"]
                _end_dot(ax, r["best_step"], yb, C_EMA)
                ax.annotate(f"best {yb:.3f} @ {r['best_step']:,}", (r["best_step"], yb),
                            xytext=(0, 9), textcoords="offset points",
                            va="bottom", ha="center", fontsize=8, color=INK_PRIMARY)
            val_src = ", ".join(cfg.SOURCE_SHORT.get(s, s) for s in r["val_groups"])
            ax.set_title(f"{f['fold']}", loc="left", fontsize=10, color=INK_PRIMARY, pad=18)
            ax.text(0.0, 1.03,
                    f"val: {val_src} ({r['n_val']} layers) · classes: {' '.join(r['defect_classes']) or '-'}",
                    transform=ax.transAxes, fontsize=7.5, color=INK_SECONDARY, va="bottom", ha="left")
        else:
            steps, mean, names = mean_curve
            ax.plot(steps, mean, color=C_EMA, linewidth=1.5,
                    solid_joinstyle="round", solid_capstyle="round", zorder=4)
            i_peak = int(np.nanargmax(mean))
            _end_dot(ax, steps[i_peak], mean[i_peak], C_EMA)
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
    fig.suptitle(f"v6 validation folds — {run_name}", x=0.012, ha="left",
                 fontsize=11.5, color=INK_PRIMARY, y=0.995)
    fig.tight_layout(rect=[0, 0, 1, 0.93])
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=144, facecolor=SURFACE)
    plt.close(fig)


# ---------------------------------------------------------------------------
# 그림 2 — 클래스마다 한 칸: v6 의 EMA IoU 곡선 + (정의가 그대로인 클래스는) v5 의 곡선 + 복사 기준선
# ---------------------------------------------------------------------------

def plot_class_curves(f: dict, out_path: Path, run_name: str) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.lines import Line2D

    meta = f["meta"]
    st = f["steps"]
    total = int(st[-1])
    ncols, nrows = 5, 2
    copy = _ov(f, "copy_resized")
    v5 = f["v5_curve"]
    val_px = meta["class_val_px"]
    val_layers = meta.get("class_val_layers") or [None] * N_CLS
    train_px = meta["class_train_px"]

    fig, axes = plt.subplots(
        nrows, ncols, figsize=(3.5 * ncols, 3.2 * nrows + 1.0),
        sharex=True, sharey=True, squeeze=False, facecolor=SURFACE,
    )
    for c in range(N_CLS):
        row, col = divmod(c, ncols)
        ax = axes[row][col]
        _style_axis(ax, total, 1.0, col == 0, row == nrows - 1, f"IoU (held-out human GT, {meta['config']['img_size']}²)")
        ax.set_title(f"{c} {cfg.CLASS_NAMES_V6[c]}", loc="left", fontsize=9.5, color=INK_PRIMARY, pad=18)
        if val_px[c] == 0:
            sub = "no validation pixels"
        else:
            n_l = "" if val_layers[c] is None else f" in {val_layers[c]} layer{'' if val_layers[c] == 1 else 's'}"
            sub = f"val {val_px[c]:,} px{n_l}" + ("" if train_px[c] > 0 else " · no training pixels")
            # 복사 기준선의 값은 부제에 적는다 (그림 안에 적으면 곡선과 겹친다). 선은 범례가 설명한다
            if copy is not None and copy["iou"][c] is not None:
                sub += f" · copy baseline {copy['iou'][c]:.2f}"
        ax.text(0.0, 1.03, sub, transform=ax.transAxes, fontsize=7.5, color=INK_SECONDARY, va="bottom", ha="left")
        if val_px[c] == 0:
            continue
        # 복사 기준선 (수평 hairline)
        if copy is not None and copy["iou"][c] is not None:
            ax.axhline(copy["iou"][c], color=AXIS, linewidth=0.9, zorder=2)
        # v5 참고선 (정의가 그대로인 클래스만)
        if v5 is not None and c in v5["ema_iou"] and not np.all(np.isnan(v5["ema_iou"][c])):
            ax.plot(v5["steps"], v5["ema_iou"][c], color=C_REF, linewidth=1.3,
                    solid_joinstyle="round", solid_capstyle="round", zorder=3)
        y = f["ema_iou"][:, c]
        ax.plot(st, y, color=C_EMA, linewidth=1.5, solid_joinstyle="round", solid_capstyle="round", zorder=4)
        if y[-1] == y[-1]:
            _end_dot(ax, st[-1], y[-1], C_EMA)
            ax.annotate(f"{y[-1]:.3f}", (st[-1], y[-1]), xytext=(7, 0), textcoords="offset points",
                        va="center", ha="left", fontsize=8, color=INK_PRIMARY)

    handles = [Line2D([0], [0], color=C_EMA, linewidth=1.5, label="v6 (EMA copy)")]
    if v5 is not None:
        handles.append(Line2D([0], [0], color=C_REF, linewidth=1.3,
                              label="v5, same fold (EMA) — classes whose definition did not change"))
    if copy is not None:
        handles.append(Line2D([0], [0], color=AXIS, linewidth=0.9,
                              label="copy baseline (nearest training layer of the same source)"))
    leg = fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.008, 0.955),
                     ncol=len(handles), frameon=False, fontsize=8.5, handlelength=1.8)
    for t in leg.get_texts():
        t.set_color(INK_PRIMARY)
    fig.suptitle(f"v6 per-class IoU during training — {run_name} / {f['fold']}", x=0.008, ha="left",
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


def _pct(v, nd: int = 1) -> str:
    return "–" if v is None or (isinstance(v, float) and v != v) else f"{100 * v:.{nd}f}%"


def _iou_cells(ov: dict | None, meta: dict, field: str = "iou") -> list[str]:
    """클래스 10개의 칸. 검증 화소가 없는 클래스는 –, 학습 화소가 없는 클래스는 †."""
    if ov is None:
        return ["–"] * N_CLS
    cells = []
    for c in range(N_CLS):
        v = ov[field][c]
        if meta["class_val_px"][c] == 0 or v is None:
            cells.append("–")
        elif meta["class_train_px"][c] == 0:
            cells.append(f"{_n(v, 3)} †")
        else:
            cells.append(_n(v, 3))
    return cells


def _ref_name(doc: dict | None, fallback: str) -> str:
    """기준선 ckpt 의 짧은 이름 — `checkpoints/<…_vN>/<나머지>` → "vN `<나머지>`" (예: v5 `cv/inlayer/last.pt`)."""
    if doc is None:
        return fallback
    parts = Path(doc.get("checkpoint", "")).parts
    if "checkpoints" in parts and len(parts) > parts.index("checkpoints") + 2:
        i = parts.index("checkpoints")
        return f"{parts[i + 1].rsplit('_', 1)[-1]} `{'/'.join(parts[i + 2:])}`"
    return f"{fallback} `{'/'.join(parts[-2:])}`" if parts else fallback


def fold_section(f: dict) -> list[str]:
    """fold 하나의 표들 (PLAN_v6 §7.1 의 읽는 순서)."""
    meta = f["meta"]
    img = meta["config"]["img_size"]
    ev = f["evals"]
    lines = [f"## `{f['fold']}`", ""]

    # 1. 배선
    w = meta.get("wiring_check")
    if w is None:
        wiring = "배선 대조: 대조할 표가 없다 (고정 목록 fold 가 아니거나 스모크 실행이다)."
    elif w["match"]:
        wiring = f"배선 대조: 클래스별 화소 수 · 레이어 수가 계획의 표 (`splits/{w['split_file']}`) 와 **같다**."
    else:
        wiring = (
            f"배선 대조: 클래스별 화소 수가 계획의 표 (`splits/{w['split_file']}`) 와 **다르다** — "
            f"학습 {w.get('diff_train_px')} · 검증 {w.get('diff_val_px')}. 데이터셋을 다른 옵션으로 다시 변환했다면 정상이다."
        )
    conv = (meta["config"].get("dataset") or {}).get("conversion") or {}
    lines += [
        f"학습 {len(meta['train_samples'])} · 검증 {len(meta['val_samples'])} 레이어. {wiring}",
        f"데이터셋 변환: {conv.get('created_at', '?')}, 옵션 {json.dumps(conv.get('options', {}), ensure_ascii=False)}.",
        "",
    ]

    head = "| | " + " | ".join(AB) + " | 결함 mIoU |"
    rule = "|---|" + "---:|" * (N_CLS + 1)

    def row(label: str, key: str, field: str = "iou", with_miou: bool = True) -> str | None:
        ov = _ov(f, key)
        if ov is None:
            return None
        tail = _n(ov.get("defect_miou")) if with_miou else ""
        return f"| {label} | " + " | ".join(_iou_cells(ov, meta, field)) + f" | {tail} |"

    r2 = _ref_name(ev.get("v5_native") or ev.get("v5_resized"), "v5")
    r3 = _ref_name(ev.get("v4_native") or ev.get("v4_resized"), "v4")

    # 2. 학습 해상도 — 복사 기준선과 견준다
    lines += [f"### 클래스별 IoU — {img}² (학습 중 검증과 같은 해상도. 복사 기준선과는 여기서 견준다)", "", head, rule]
    for label, key in (
        ("v6 `last` (EMA)", "v6_resized"),
        ("R1 복사 기준선", "copy_resized"),
        (f"R2 {r2} → 새 id", "v5_resized"),
        (f"R3 {r3} → 새 id", "v4_resized"),
    ):
        r = row(label, key)
        if r is not None:
            lines.append(r)
    if _ov(f, "v6_resized") is None:
        ema_last = {"iou": [None if v != v else float(v) for v in f["ema_iou"][-1]],
                    "defect_miou": float(f["ema_defect"][-1])}
        lines.append("| v6 마지막 step (EMA, 곡선에서) | " + " | ".join(_iou_cells(ema_last, meta)) + f" | {_n(ema_last['defect_miou'])} |")
    lines.append("")

    # 3. 원본 해상도 — 주 지표
    if any(k in ev for k in ("v6_native", "v5_native", "v4_native")):
        lines += ["### 클래스별 IoU — 원본 해상도 (서빙과 같은 경로)", "", head, rule]
        for label, key, field, with_miou in (
            ("v6 `last` (EMA)", "v6_native", "iou", True),
            ("v6 재현율", "v6_native", "recall", False),
            ("v6 정밀도", "v6_native", "precision", False),
            (f"R2 {r2} → 새 id", "v5_native", "iou", True),
            (f"R3 {r3} → 새 id", "v4_native", "iou", True),
        ):
            r = row(label, key, field, with_miou)
            if r is not None:
                lines.append(r)
        lines.append("")

    notes = ["– 검증 화소 없음. † 학습 화소가 0 인 클래스 (결함 mIoU 에서 뺀다)."]
    if "v5_native" in ev or "v4_native" in ev or "v5_resized" in ev or "v4_resized" in ev:
        union = []
        for label, key in (("v6", "v6_native"), ("R2", "v5_native"), ("R3", "v4_native")):
            ov = _ov(f, key)
            if ov is not None and ov.get("is_ps_union_iou") is not None:
                union.append(f"{label} {_n(ov['is_ps_union_iou'], 3)}")
        notes.append(
            "R2 · R3 는 옛 11-class 모델의 예측을 새 id 로 옮긴 것이다 (Swelling 예측 → SE). 옛 모델은 PS 를 낼 수 없고 옛 IS 예측이 새 IS 로 옮겨지므로 "
            "IS · PS 와 결함 mIoU 는 그대로 견주지 않는다"
            + (f" — IS 와 PS 를 합쳐 본 IoU (원본 해상도): {', '.join(union)}." if union else ".")
            + " 옛 모델은 Soot 를 미레이블로 두고 배워 새 정답의 SP 에서는 낮게 나온다 (SP 는 R1 과 견준다). "
            "R3 (v4) 는 이 검증 레이어를 학습에 썼다."
        )
    lines += notes + [""]

    # 4. 놓침 · 잘못 짚음
    mfa_rows = []
    for label, key in (("v6 `last`", "v6_native"), ("R2", "v5_native"), ("R3", "v4_native")):
        ov = _ov(f, key)
        if ov is None:
            continue
        m = ov["miss_false_alarm"]
        mfa_rows.append(
            f"| {label} | {_pct(m['correct'])} | {_pct(m['miss'])} | {_pct(m['wrong_defect'])} | {_pct(m['to_none'])} | {_pct(m['false_alarm'], 2)} |"
        )
    if mfa_rows:
        lines += [
            "### 놓침과 잘못 짚음 — 원본 해상도", "",
            "| | 결함 정답 → 맞음 | → 정상 (놓침) | → 다른 결함 | → 대응 없음 | 정상 정답 → 결함 (잘못 짚음) |",
            "|---|---:|---:|---:|---:|---:|",
            *mfa_rows, "",
        ]

    # 5. 혼동 쌍
    ov6 = _ov(f, "v6_native") or _ov(f, "v6_resized")
    if ov6 is not None:
        res_name = "원본 해상도" if _ov(f, "v6_native") is not None else f"{img}²"
        lines += [
            f"### 혼동 쌍 — v6 `last`, {res_name}", "",
            "쌍 안 혼동률 = 정답도 예측도 그 둘 가운데 하나인 화소에서 둘을 맞바꾼 비율. 앞 → 뒤 = 앞 클래스 정답 화소 전체 가운데 뒤 클래스로 예측한 비율.",
            "",
            "| 쌍 | 쌍 안 혼동률 | 앞 → 뒤 | 뒤 → 앞 | 앞 / 뒤 정답 화소 |",
            "|---|---:|---:|---:|---:|",
        ]
        for name, d in ov6["pairs"].items():
            lines.append(
                f"| {name.replace('-', ' ↔ ')} | {_pct(d['within_pair'])} | {_pct(d['a_as_b'])} | {_pct(d['b_as_a'])} | {d['a_px']:,} / {d['b_px']:,} |"
            )
        lines.append("")

    # 6. 소스별
    doc6 = ev.get("v6_native")
    if doc6 is not None and doc6.get("by_group"):
        lines += [
            "### 소스별 IoU — v6 `last`, 원본 해상도", "",
            "합친 값은 원본 화소가 많은 소스의 값이 된다. 클래스가 여러 소스에 있으면 이 표로 읽는다.", "",
            "| 소스 | " + " | ".join(AB) + " |",
            "|---|" + "---:|" * N_CLS,
        ]
        for g, ov in doc6["by_group"].items():
            cells = ["–" if (ov["iou"][c] is None or ov["support_px"][c] == 0) else _n(ov["iou"][c], 3) for c in range(N_CLS)]
            lines.append(f"| {cfg.SOURCE_SHORT.get(g, g)} | " + " | ".join(cells) + " |")
        lines.append("")
    lines += [f"클래스별 곡선: `{f['fold']}/class_curves.png`.", ""]
    return lines


def write_tables(folds: list[dict], rows: list[dict], rec: dict | None, out_dir: Path, run_name: str) -> str:
    fields = [
        "fold", "val_groups", "n_train", "n_val", "defect_classes", "wiring_match", "total_steps",
        "best_step", "best_ema_defect_miou", "final_ema_defect_miou", "final_raw_defect_miou",
        "final_ema_miou", "final_ema_acc",
        "native_defect_miou", "native_miou", "resized_defect_miou", "copy_resized_defect_miou",
        "v5_native_defect_miou", "v5_resized_defect_miou", "v4_native_defect_miou", "v4_resized_defect_miou",
    ]
    with (out_dir / "summary.csv").open("w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=fields)
        w.writeheader()
        for r in rows:
            rr = dict(r)
            rr["val_groups"] = " ".join(r["val_groups"])
            rr["defect_classes"] = " ".join(r["defect_classes"])
            w.writerow(rr)

    img = folds[0]["meta"]["config"]["img_size"]
    lines = [f"# v6 검증 fold 요약 — `{run_name}`", ""]
    lines += [
        f"결함 mIoU 는 그 fold 에서 학습 화소와 검증 화소가 모두 있는 결함 클래스의 IoU 평균이다 (떼어 둔 사람 GT, 곡선의 값은 {img}²).",
        "inlayer fold 의 값은 낙관적이다 — 검증 레이어와 같은 빌드의 다른 레이어가 학습에 있다. 성능으로 옮겨 적지 않고 복사 기준선과 함께 읽는다 (PLAN_v6 §7.1).",
        "",
        f"| fold | 검증 소스 | 학습 / 검증 레이어 | 채점 결함 클래스 | best step | best (EMA) | 끝 (EMA) | 끝 (학습용) | 원본 해상도 (v6 `last`) | 복사 기준선 ({img}²) |",
        "|---|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    for r in rows:
        lines.append(
            f"| `{r['fold']}` | {', '.join(cfg.SOURCE_SHORT.get(s, s) for s in r['val_groups'])} "
            f"| {r['n_train']} / {r['n_val']} | {' '.join(r['defect_classes']) or '–'} "
            f"| {r['best_step']:,} | {_n(r['best_ema_defect_miou'])} | {_n(r['final_ema_defect_miou'])} "
            f"| {_n(r['final_raw_defect_miou'])} | {_n(r['native_defect_miou'])} | {_n(r['copy_resized_defect_miou'])} |"
        )
    lines.append("")
    for f in folds:
        lines += fold_section(f)

    if rec is not None:
        lines += ["## 권장 길이", "", rec["note"],
                  f"그 길이에서 EMA 규칙 (시상수 ≤ 총 step / 10) 을 지키는 decay 는 {rec['recommended_ema_decay']} 이하다."]
    else:
        n_loso = sum(1 for f in folds if f["fold"].startswith("loso_"))
        lines += ["## 권장 길이", "",
                  f"내지 않았다 — 검증 step 이 겹치는 loso fold 가 둘 이상 있어야 평균 곡선을 낸다 (지금 loso fold {n_loso}개). "
                  "inlayer 의 곡선은 길이를 정하는 근거로 쓰지 않는다. 길이 (10,000 step) 는 v5 의 두 fold 에서 확인한 값이다 (PLAN_v6 §5.2)."]
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
    cv_dir = cfg.FIGURE_DIR / run_name / "v6" / "cv"
    if not cv_dir.is_dir():
        ap.error(f"검증 fold 결과가 없다: {cv_dir}")

    order = {name: i for i, name in enumerate(cfg.CV_FOLDS)}
    fold_dirs = sorted([d for d in cv_dir.iterdir() if d.is_dir()], key=lambda d: order.get(d.name, 999))
    folds = [f for f in (load_fold(d) for d in fold_dirs) if f is not None]
    if not folds:
        ap.error(f"곡선이 있는 fold 가 없다 (학습이 끝난 fold 가 없다): {cv_dir}")

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
        for f in folds:
            plot_class_curves(f, f["dir"] / "class_curves.png", run_name)
    print(text)
    print(f"saved → {cv_dir}/{{summary.csv, summary.md, summary.json"
          + ("" if args.no_plot else ", curves.png, <fold>/class_curves.png") + "}")


if __name__ == "__main__":
    main()
