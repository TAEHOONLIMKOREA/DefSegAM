"""v8 — 기준선 (보정 없음) 과 v8 의 실행을 fold 마다 한 표로 모으고 PLAN_v8 §4.4 의 판정을 낸다. GPU 를 쓰지 않는다.

읽는 것: figures/<run>/v6/cv/<fold>/eval_<해상도>_<which>/metrics.json   — v6 의 evaluate 가 쓴 것 그대로
         기준선 = v7 의 꼬리표 `N` 실행 (`…_10cls_v7-N-s<seed>`. `inlayer` seed 42 는 v6 의 실행) — v7 의 비교 스크립트가 모으는 것과 같다
         v8     = `…_10cls_v8-<꼬리표>-s<seed>`
쓰는 것: figures/v8_compare/sz<img>/{<fold>.md, summary.json}

판정 (PLAN_v8 §4.4 — 문턱은 제안값):
  재는 값 = 결함 mIoU (원본 해상도, last.pt)
  기준선  = 꼬리표 N 의 seed 평균, σ = 그 표준편차 (seed 가 둘 이상일 때만)
  효과:   결함 mIoU ≥ 기준선 + max(0.02, 2σ)
  부작용: Powder 또는 Printed 의 IoU 가 0.01 넘게 내려감 / 잘못 짚은 비율이 기준선 + 1.0%p 초과 /
          결함 클래스 하나의 IoU 가 그 클래스의 max(0.03, 2σ) 넘게 내려감

사용 (CWD = 3DP_VPPM/):
    python -m DefSeg_AM.v8.scripts.compare_runs
    python -m DefSeg_AM.v8.scripts.compare_runs --resolution resized
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from ...v6 import config_v6 as v6cfg
from ...v7 import config_v7 as v7cfg
from ...v7.scripts.compare_runs import collect as collect_v7
from .. import config_v8 as cfg

EFFECT_MIN, POW_PRT_MAX_DROP, FA_MAX_RISE, CLASS_MIN_DROP = 0.02, 0.01, 0.01, 0.03
AB = v6cfg.CLASS_ABBRS_V6
DEFECTS = list(v6cfg.DEFECT_IDS_V6)
RUN_RE = re.compile(r"_10cls_v8-(?P<variant>.+)-s(?P<seed>\d+)$")
BASE = "N"


def _nan(v) -> float:
    return float("nan") if v is None else float(v)


def _load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    m = json.loads(path.read_text(encoding="utf-8"))
    o = m["overall"]
    return {
        "iou": [_nan(v) for v in o["iou"]],
        "support": [int(v) for v in o["support_px"]],
        "defect_miou": _nan(o["defect_miou"]),
        "miss": _nan(o["miss_false_alarm"]["miss"]),
        "false_alarm": _nan(o["miss_false_alarm"]["false_alarm"]),
        "by_group": {g: _nan(d["defect_miou"]) for g, d in m.get("by_group", {}).items()},
        "n_layers": int(m["n_layers"]), "path": str(path),
    }


def collect(fold: str, img_size: int, resolution: str, which: str) -> list[dict]:
    """그 fold 의 실행 목록 — 기준선 (N) 먼저, 그 뒤 v8 의 꼬리표 · seed 순."""
    tag = f"eval_{resolution}_{which}"
    runs = []
    for r in collect_v7(fold, img_size, resolution, which, True):
        if r["variant"] != v7cfg.DEFAULT_VARIANT:
            continue                                    # v7 의 다른 꼬리표 (생성 데이터를 넣은 실행) 는 여기서 견주지 않는다
        m = _load(Path(r["path"]))
        if m is not None:
            runs.append(dict(m, variant=BASE, seed=r["seed"], run=r["run"], norm_set=None))
    for d in sorted(v6cfg.FIGURE_DIR.glob(f"*_dpt_dual_sz{img_size}_10cls_v8-*-s*")):
        mt = RUN_RE.search(d.name)
        if not mt:
            continue
        fdir = d / "v6" / "cv" / fold
        m = _load(fdir / tag / "metrics.json")
        if m is None:
            continue
        rec = fdir / cfg.RUN_NORM_JSON
        ns = json.loads(rec.read_text(encoding="utf-8")).get("norm_set") if rec.is_file() else None
        runs.append(dict(m, variant=mt["variant"], seed=int(mt["seed"]), run=d.name, norm_set=ns))
    return sorted(runs, key=lambda r: (r["variant"] != BASE, r["variant"], r["seed"]))


def _mean_std(vals: list[float]) -> tuple[float, float | None]:
    a = np.array(vals, dtype=np.float64)
    a = a[np.isfinite(a)]
    if not len(a):
        return float("nan"), None
    return float(a.mean()), (float(a.std(ddof=1)) if len(a) >= 2 else None)


def judge(fold: str, runs: list[dict]) -> dict:
    base = [r for r in runs if r["variant"] == BASE]
    out = {"fold": fold, "n_baseline": len(base), "variants": {}}
    if not base:
        out["note"] = "기준선 (꼬리표 N) 이 없다 — 판정하지 않는다"
        return out
    bM, sM = _mean_std([r["defect_miou"] for r in base])
    b_iou = [_mean_std([r["iou"][c] for r in base]) for c in range(v6cfg.N_CLASSES_V6)]
    b_fa, _ = _mean_std([r["false_alarm"] for r in base])
    b_miss, _ = _mean_std([r["miss"] for r in base])
    groups = sorted({g for r in base for g in r["by_group"]})
    b_grp = {g: _mean_std([r["by_group"].get(g, float("nan")) for r in base]) for g in groups}
    thr = max(EFFECT_MIN, 2 * sM) if sM is not None else EFFECT_MIN
    out["baseline"] = {"defect_miou": bM, "sigma": sM, "threshold": thr, "false_alarm": b_fa, "miss": b_miss,
                       "seeds": [r["seed"] for r in base], "iou": [m for m, _ in b_iou], "iou_sigma": [s for _, s in b_iou],
                       "by_group": {g: m for g, (m, _) in b_grp.items()}, "by_group_sigma": {g: s for g, (_, s) in b_grp.items()}}
    if sM is None:
        out["note"] = "기준선의 seed 가 하나뿐이라 σ 를 모른다 — 문턱은 제안값 (0.02) 만 썼다. 작은 차이는 읽지 않는다"
    for v in sorted({r["variant"] for r in runs} - {BASE}):
        rs = [r for r in runs if r["variant"] == v]
        vM = float(np.nanmean([r["defect_miou"] for r in rs]))
        v_iou = [float(np.nanmean([r["iou"][c] for r in rs])) if any(r["iou"][c] == r["iou"][c] for r in rs) else float("nan")
                 for c in range(v6cfg.N_CLASSES_V6)]
        side = []
        for c in (0, 1):
            d = v_iou[c] - b_iou[c][0]
            if d < -POW_PRT_MAX_DROP:
                side.append(f"{AB[c]} IoU {d:+.3f}")
        d_fa = float(np.nanmean([r["false_alarm"] for r in rs])) - b_fa
        if d_fa > FA_MAX_RISE:
            side.append(f"잘못 짚은 비율 {100 * d_fa:+.2f}%p")
        for c in DEFECTS:
            if b_iou[c][0] != b_iou[c][0] or v_iou[c] != v_iou[c]:
                continue
            lim = max(CLASS_MIN_DROP, 2 * b_iou[c][1]) if b_iou[c][1] is not None else CLASS_MIN_DROP
            d = v_iou[c] - b_iou[c][0]
            if d < -lim:
                side.append(f"{AB[c]} IoU {d:+.3f}")
        effect = bool(vM - bM >= thr)
        verdict = ("부작용" + (" (효과의 문턱은 넘었다)" if effect else "")) if side else ("효과" if effect else "문턱 안 (차이 없음)")
        out["variants"][v] = {
            "defect_miou": vM, "delta": vM - bM, "effect": effect, "verdict": verdict, "side_effects": side,
            "seeds": [r["seed"] for r in rs], "norm_sets": sorted({str(r["norm_set"]) for r in rs}),
            "delta_miss": float(np.nanmean([r["miss"] for r in rs])) - b_miss, "delta_false_alarm": d_fa,
            "iou": v_iou, "delta_iou": [v_iou[c] - b_iou[c][0] for c in range(v6cfg.N_CLASSES_V6)],
            "by_group": {g: float(np.nanmean([r["by_group"].get(g, float("nan")) for r in rs])) for g in groups},
        }
    return out


def _f(v, nd=3) -> str:
    return "–" if v is None or v != v else f"{v:.{nd}f}"


def _pm(v, nd=3) -> str:
    return "–" if v is None or v != v else f"{v:+.{nd}f}"


def write_fold(fold: str, runs: list[dict], j: dict, out_dir: Path, resolution: str, which: str) -> str:
    L = [f"# v8 실행 비교 — `{fold}` ({'원본 해상도' if resolution == 'native' else '모델 입력 크기'}, `{which}.pt`)", "",
         "기준선 = 보정 없이 학습한 실행 (꼬리표 N). v8 의 실행은 보정 세트의 사진으로 학습하고 같은 보정을 건 사진으로 채점한 것이다. "
         "어느 실행도 생성 데이터를 학습하지 않았다. 판정의 문턱은 제안값이다 (PLAN_v8 §4.4)."]
    if j.get("note"):
        L += ["", f"**주의**: {j['note']}"]
    L += ["", "## 실행별", "",
          "| 꼬리표 | seed | 보정 세트 | 결함 mIoU | " + " | ".join(AB) + " | 놓친 비율 | 잘못 짚은 비율 | run |",
          "|---|---:|---|---:|" + "---:|" * len(AB) + "---:|---:|---|"]
    for r in runs:
        L.append(f"| {r['variant']} | {r['seed']} | {r['norm_set'] or '–'} | {_f(r['defect_miou'])} | " + " | ".join(_f(v) for v in r["iou"])
                 + f" | {100 * r['miss']:.1f}% | {100 * r['false_alarm']:.2f}% | `{r['run']}` |")
    if "baseline" in j:
        b = j["baseline"]
        L += ["", "## 판정", "",
              f"기준선 (N, seed {b['seeds']}): 결함 mIoU = {_f(b['defect_miou'])}, σ = {_f(b['sigma'], 4)}, 문턱 = +{_f(b['threshold'])} "
              f"(효과라고 보는 값: {_f(b['defect_miou'] + b['threshold'])} 이상). 놓친 비율 {100 * b['miss']:.1f}%, 잘못 짚은 비율 {100 * b['false_alarm']:.2f}%.", ""]
        if j["variants"]:
            L += ["| 꼬리표 | seed | 결함 mIoU | 기준선과의 차 | 놓친 비율의 차 | 잘못 짚은 비율의 차 | 판정 |", "|---|---|---:|---:|---:|---:|---|"]
            for v, d in j["variants"].items():
                verdict = d["verdict"] + (f" — {', '.join(d['side_effects'])}" if d["side_effects"] else "")
                L.append(f"| {v} | {d['seeds']} | {_f(d['defect_miou'])} | {_pm(d['delta'])} | {100 * d['delta_miss']:+.1f}%p | "
                         f"{100 * d['delta_false_alarm']:+.2f}%p | **{verdict}** |")
            vs = list(j["variants"])
            L += ["", "## 클래스별 IoU — 기준선의 평균 (σ) 과 그와의 차", "",
                  "| 클래스 | 기준선 (σ) | " + " | ".join(f"{v} | 차" for v in vs) + " |", "|---|---:|" + "---:|---:|" * len(vs)]
            for c in range(v6cfg.N_CLASSES_V6):
                if b["iou"][c] != b["iou"][c]:
                    continue
                L.append(f"| {AB[c]} | {_f(b['iou'][c])} ({_f(b['iou_sigma'][c], 4)}) | "
                         + " | ".join(f"{_f(j['variants'][v]['iou'][c])} | {_pm(j['variants'][v]['delta_iou'][c])}" for v in vs) + " |")
            L += ["", "## 소스별 결함 mIoU — 기준선의 평균 (σ) 과 그와의 차", "",
                  "| 소스 | 기준선 (σ) | " + " | ".join(f"{v} | 차" for v in vs) + " |", "|---|---:|" + "---:|---:|" * len(vs)]
            for g, m in b["by_group"].items():
                cells = []
                for v in vs:
                    x = j["variants"][v]["by_group"].get(g, float("nan"))
                    cells.append(f"{_f(x)} | {_pm(x - m) if x == x and m == m else '–'}")
                L.append(f"| {v6cfg.SOURCE_SHORT.get(g, g)} | {_f(m)} ({_f(b['by_group_sigma'][g], 4)}) | " + " | ".join(cells) + " |")
            L += ["", "평균만 보지 않는다 — 소스마다 · 클래스마다의 차를 그 σ 와 견준다 (σ 의 두 배 안쪽은 seed 의 흔들림과 가를 수 없다)."]
        else:
            L.append("기준선과 견줄 v8 의 실행이 아직 없다.")
    text = "\n".join(L) + "\n"
    (out_dir / f"{fold}.md").write_text(text, encoding="utf-8")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=["inlayer", "loso_L21"], help="판정 fold — inlayer 와 loso_L21 (PLAN_v8 §4.3)")
    ap.add_argument("--img-size", type=int, default=v6cfg.IMG_SIZE)
    ap.add_argument("--resolution", choices=["native", "resized"], default="native", help="주 지표는 원본 해상도 (native)")
    ap.add_argument("--which", default="last")
    args = ap.parse_args()

    out_dir = cfg.COMPARE_FIG_DIR / f"sz{args.img_size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for fold in args.folds:
        runs = collect(fold, args.img_size, args.resolution, args.which)
        if not runs:
            print(f"[compare] {fold}: 채점된 실행이 없다")
            continue
        j = judge(fold, runs)
        summary[fold] = j
        print(write_fold(fold, runs, j, out_dir, args.resolution, args.which))
    (out_dir / "summary.json").write_text(json.dumps(summary, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"[compare] → {out_dir}")


if __name__ == "__main__":
    main()
