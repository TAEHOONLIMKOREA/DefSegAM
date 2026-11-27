"""v7 — 실행 여럿 (꼬리표 · seed) 을 fold 마다 한 표로 모으고 PLAN_v7 §4.3 의 판정을 낸다. GPU 를 쓰지 않는다.

읽는 것: figures/<run>/v6/cv/<fold>/eval_<해상도>_<which>/metrics.json   — v6 의 evaluate 가 쓴 것 그대로
         run = <backbone>_dpt_dual_sz<img>_10cls_v7-<꼬리표>-s<seed>
         `inlayer` 의 기준선 (꼬리표 N) seed 42 가 없으면 v6 의 P0 (…_10cls_v6) 를 그 자리에 놓는다 — v7 의 학습 스크립트가
         합성 없이 v6 와 같은 loss 를 낸다는 것을 스모크로 확인했기 때문이다 (PLAN_v7 §5.3). `--no-v6-p0` 로 끈다.
쓰는 것: figures/v7_compare/sz<img>/{<fold>.md, summary.json}

판정 (PLAN_v7 §4.3 — 문턱은 제안값):
  T = 대상 클래스 (RS · RH · SE · DB 가운데 그 fold 에서 채점되는 것) 의 IoU 평균, 원본 해상도, last.pt
  기준선 = 꼬리표 N 의 seed 평균, σ = 그 표준편차 (seed 가 둘 이상일 때만)
  효과:   T ≥ 기준선 + max(0.03, 2σ)
  부작용: Powder 또는 Printed IoU 가 0.01 넘게 내려감 / 잘못 짚은 비율이 기준선 + 1.0%p 초과 /
          합성에 넣지 않은 클래스 (IS · PS · SP) 의 IoU 가 그 클래스의 max(0.03, 2σ) 넘게 내려감

사용 (CWD = 3DP_VPPM/):
    python -m DefSeg_AM.v7.scripts.compare_runs
    python -m DefSeg_AM.v7.scripts.compare_runs --resolution resized
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import numpy as np

from ...v6 import config_v6 as v6cfg
from .. import config_v7 as cfg

TARGET = (4, 5, 6, 9)            # RS · RH · SE · DB — 합성 레이어에 넣는 결함 (PLAN_v7 §2.2)
WATCH = (2, 3, 7)                # IS · PS · SP — 합성에 넣지 않은 클래스. 내려가면 부작용 (OM 은 검증 화소가 적어 뺐다)
T_MIN, POW_PRT_MAX_DROP, FA_MAX_RISE, WATCH_MIN_DROP = 0.03, 0.01, 0.01, 0.03
AB = v6cfg.CLASS_ABBRS_V6
RUN_RE = re.compile(r"_10cls_v7-(?P<variant>.+)-s(?P<seed>\d+)$")


def _load(path: Path) -> dict | None:
    if not path.is_file():
        return None
    m = json.loads(path.read_text(encoding="utf-8"))
    o = m["overall"]
    return {
        "iou": [np.nan if v is None else float(v) for v in o["iou"]],
        "recall": [np.nan if v is None else float(v) for v in o["recall"]],
        "precision": [np.nan if v is None else float(v) for v in o["precision"]],
        "support": [int(v) for v in o["support_px"]],
        "defect_miou": float(o["defect_miou"]) if o["defect_miou"] is not None else np.nan,
        "miss": float(o["miss_false_alarm"]["miss"]),
        "false_alarm": float(o["miss_false_alarm"]["false_alarm"]),
        "n_layers": int(m["n_layers"]), "ckpt_step": m.get("ckpt_step"), "path": str(path),
    }


def collect(fold: str, img_size: int, resolution: str, which: str, use_v6_p0: bool) -> list[dict]:
    """그 fold 의 실행 목록 — [{variant, seed, run, ...metrics}]."""
    tag = f"eval_{resolution}_{which}"
    runs = []
    for d in sorted(v6cfg.FIGURE_DIR.glob(f"*_dpt_dual_sz{img_size}_10cls_v7-*-s*")):
        mt = RUN_RE.search(d.name)
        if not mt:
            continue
        m = _load(d / "v6" / "cv" / fold / tag / "metrics.json")
        if m is not None:
            runs.append(dict(m, variant=mt["variant"], seed=int(mt["seed"]), run=d.name))
    have_n42 = any(r["variant"] == cfg.DEFAULT_VARIANT and r["seed"] == v6cfg.RANDOM_SEED for r in runs)
    if use_v6_p0 and not have_n42:
        v6_run = v6cfg.default_run_name(v6cfg.DINO_BACKBONE, img_size)
        m = _load(v6cfg.FIGURE_DIR / v6_run / "v6" / "cv" / fold / tag / "metrics.json")
        if m is not None:
            runs.append(dict(m, variant=cfg.DEFAULT_VARIANT, seed=v6cfg.RANDOM_SEED, run=v6_run + "  (v6 의 실행)"))
    return sorted(runs, key=lambda r: (r["variant"] != cfg.DEFAULT_VARIANT, r["variant"], r["seed"]))


def target_classes(runs: list[dict]) -> list[int]:
    """대상 클래스 가운데 이 fold 에서 채점되는 것 (검증 화소가 있고 IoU 가 나온 것)."""
    return [c for c in TARGET if all(r["support"][c] > 0 and r["iou"][c] == r["iou"][c] for r in runs)]


def t_value(r: dict, tc: list[int]) -> float:
    return float(np.mean([r["iou"][c] for c in tc])) if tc else float("nan")


def _mean_std(vals: list[float]) -> tuple[float, float | None]:
    a = np.array(vals, dtype=np.float64)
    return float(np.nanmean(a)), (float(np.nanstd(a, ddof=1)) if len(a) >= 2 else None)


def judge(fold: str, runs: list[dict]) -> dict:
    tc = target_classes(runs)
    base = [r for r in runs if r["variant"] == cfg.DEFAULT_VARIANT]
    out = {"fold": fold, "target_classes": [AB[c] for c in tc], "n_baseline": len(base), "variants": {}}
    if not base:
        out["note"] = "기준선 (꼬리표 N) 이 없다 — 판정하지 않는다"
        return out
    bT, sT = _mean_std([t_value(r, tc) for r in base])
    b_iou = [_mean_std([r["iou"][c] for r in base]) for c in range(v6cfg.N_CLASSES_V6)]
    b_fa, _ = _mean_std([r["false_alarm"] for r in base])
    b_miss, _ = _mean_std([r["miss"] for r in base])
    thr = max(T_MIN, 2 * sT) if sT is not None else T_MIN
    out["baseline"] = {"T": bT, "sigma_T": sT, "threshold": thr, "false_alarm": b_fa, "miss": b_miss,
                       "seeds": [r["seed"] for r in base]}
    if sT is None:
        out["note"] = "기준선의 seed 가 하나뿐이라 σ 를 모른다 — 문턱은 제안값 (0.03) 만 썼다. 작은 차이는 읽지 않는다"
    for v in sorted({r["variant"] for r in runs} - {cfg.DEFAULT_VARIANT}):
        rs = [r for r in runs if r["variant"] == v]
        vT = float(np.mean([t_value(r, tc) for r in rs]))
        side = []
        for c in (0, 1):
            d = float(np.mean([r["iou"][c] for r in rs])) - b_iou[c][0]
            if d < -POW_PRT_MAX_DROP:
                side.append(f"{AB[c]} IoU {d:+.3f}")
        d_fa = float(np.mean([r["false_alarm"] for r in rs])) - b_fa
        if d_fa > FA_MAX_RISE:
            side.append(f"잘못 짚은 비율 {100 * d_fa:+.2f}%p")
        for c in WATCH:
            if b_iou[c][0] != b_iou[c][0]:
                continue
            lim = max(WATCH_MIN_DROP, 2 * b_iou[c][1]) if b_iou[c][1] is not None else WATCH_MIN_DROP
            d = float(np.nanmean([r["iou"][c] for r in rs])) - b_iou[c][0]
            if d < -lim:
                side.append(f"{AB[c]} IoU {d:+.3f}")
        verdict = "부작용" if side else ("효과" if vT - bT >= thr else "문턱 안 (차이 없음)")
        out["variants"][v] = {
            "T": vT, "delta_T": vT - bT, "verdict": verdict, "side_effects": side, "seeds": [r["seed"] for r in rs],
            "delta_miss": float(np.mean([r["miss"] for r in rs])) - b_miss, "delta_false_alarm": d_fa,
            "delta_recall": {AB[c]: float(np.nanmean([r["recall"][c] for r in rs]) - np.nanmean([b["recall"][c] for b in base])) for c in tc},
            "delta_precision": {AB[c]: float(np.nanmean([r["precision"][c] for r in rs]) - np.nanmean([b["precision"][c] for b in base])) for c in tc},
        }
    return out


def _f(v, nd=3) -> str:
    return "–" if v is None or v != v else f"{v:.{nd}f}"


def write_fold(fold: str, runs: list[dict], j: dict, out_dir: Path, resolution: str, which: str) -> str:
    tc = [AB.index(a) for a in j["target_classes"]]
    L = [f"# v7 실행 비교 — `{fold}` ({'원본 해상도' if resolution == 'native' else '모델 입력 크기'}, `{which}.pt`)", ""]
    L.append(f"대상 클래스 (T 에 넣는 것): {' · '.join(j['target_classes']) or '없음'}.  T = 그 IoU 의 평균. 판정의 문턱은 제안값이다 (PLAN_v7 §4.3).")
    if j.get("note"):
        L += ["", f"**주의**: {j['note']}"]
    L += ["", "## 실행별", "",
          "| 꼬리표 | seed | T | 결함 mIoU | " + " | ".join(AB) + " | 놓친 비율 | 잘못 짚은 비율 | run |",
          "|---|---:|---:|---:|" + "---:|" * len(AB) + "---:|---:|---|"]
    for r in runs:
        L.append(f"| {r['variant']} | {r['seed']} | {_f(t_value(r, tc))} | {_f(r['defect_miou'])} | "
                 + " | ".join(_f(v) for v in r["iou"])
                 + f" | {100 * r['miss']:.1f}% | {100 * r['false_alarm']:.2f}% | `{r['run']}` |")
    if "baseline" in j:
        b = j["baseline"]
        L += ["", "## 판정", "",
              f"기준선 (N, seed {b['seeds']}): T = {_f(b['T'])}, σ = {_f(b['sigma_T'], 4)}, 문턱 = +{_f(b['threshold'])}. "
              f"놓친 비율 {100 * b['miss']:.1f}%, 잘못 짚은 비율 {100 * b['false_alarm']:.2f}%.", ""]
        if j["variants"]:
            L += ["| 꼬리표 | seed | T | 기준선과의 차 | 놓친 비율의 차 | 잘못 짚은 비율의 차 | 재현율의 차 (대상 클래스) | 정밀도의 차 | 판정 |",
                  "|---|---|---:|---:|---:|---:|---|---|---|"]
            for v, d in j["variants"].items():
                rec = " · ".join(f"{k} {x:+.3f}" for k, x in d["delta_recall"].items())
                pre = " · ".join(f"{k} {x:+.3f}" for k, x in d["delta_precision"].items())
                verdict = d["verdict"] + (f" — {', '.join(d['side_effects'])}" if d["side_effects"] else "")
                L.append(f"| {v} | {d['seeds']} | {_f(d['T'])} | {d['delta_T']:+.3f} | {100 * d['delta_miss']:+.1f}%p | "
                         f"{100 * d['delta_false_alarm']:+.2f}%p | {rec} | {pre} | **{verdict}** |")
            L += ["", "RS · RH 는 재현율과 정밀도를 따로 읽는다 (사람 주석이 줄의 일부만 칠한다 — PLAN_v7 §4.3 의 \"RS 를 읽는 법\")."]
        else:
            L.append("기준선과 견줄 실행이 아직 없다.")
    text = "\n".join(L) + "\n"
    (out_dir / f"{fold}.md").write_text(text, encoding="utf-8")
    return text


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=["inlayer", "loso_L21"],
                    help="v7 의 판정 fold — inlayer 와 loso_L21 (PLAN_v7 §7 의 V12)")
    ap.add_argument("--img-size", type=int, default=v6cfg.IMG_SIZE)
    ap.add_argument("--resolution", choices=["native", "resized"], default="native", help="주 지표는 원본 해상도 (native)")
    ap.add_argument("--which", default="last")
    ap.add_argument("--no-v6-p0", action="store_true", help="inlayer 의 N · seed 42 자리에 v6 의 P0 를 놓지 않는다")
    args = ap.parse_args()

    out_dir = v6cfg.FIGURE_DIR / "v7_compare" / f"sz{args.img_size}"
    out_dir.mkdir(parents=True, exist_ok=True)
    summary = {}
    for fold in args.folds:
        runs = collect(fold, args.img_size, args.resolution, args.which, use_v6_p0=not args.no_v6_p0)
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
