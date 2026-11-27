"""0단계 — 학습 없이 재는 민감도 (PLAN_v8 §4.1). GPU 1대.

이미 있는 **기준선 모델** (보정 없이 학습한 것 — `inlayer` seed 42 는 v6 의 실행, 그 밖은 `…_v7-N-s<seed>`) 에
검증 레이어의 사진을 바꿔 넣고 결함 mIoU (원본 해상도, `last.pt`) 를 잰다. 채점은 v6 의 `evaluate` 와 같은 함수로 한다.

  0-1  밝기를 통째로 옮긴 사진 (−40 … +60 단계)                         → 모델이 밝기에 얼마나 민감한가
  0-1b 보정 세트 (기본 `b115`) 의 값으로 소스마다 옮긴 사진              → 다시 학습하지 않고 밝기만 맞춰 넣으면 어떻게 되는가
  0-2  가짜 조명 (기울어진 면 + 넓은 얼룩 하나) 을 곱한 사진             → 모델이 조명 얼룩에 얼마나 민감한가
  0-3  0-2 의 사진에서 조명 지도를 다시 구해 나눈 사진                    → 조명 제거가 제 일을 하는가
       · 한 소스의 레이어 전부에 같은 가짜 조명을 얹고 (장비 하나의 조명을 흉내 낸다), 그 소스의 도포 직후 사진 전부로 지도를 구한다 (라벨 없이)
       · "안전 장치 켬" = 실제로 쓸 방식 (통과한 소스만 나눈다),  "안전 장치 끔" = 모든 소스를 나눈다 (지도가 틀린 소스에서 무슨 일이 나는지 본다)
       · 가짜 조명 0% = 원본 사진에서 구한 지도로 나눈 것 (추론할 때만 조명을 걷어낸 것)

출력: figures/v8_check/sz<img>_s<seed>/{results.json, summary.md, curves.png, fake_light.png}

사용 (CWD = 3DP_VPPM/) — 보통은 compose 로 돌린다 (`v8/docker`, DEFSEG_V8_PHASE=check):
    python -m DefSeg_AM.v8.scripts.sensitivity
    python -m DefSeg_AM.v8.scripts.sensitivity --folds loso_L21 --quick
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ...common.models.model import round_to_patch
from ...v6 import config_v6 as v6cfg
from ...v6.data.data_gt_v6 import enumerate_samples_v6, load_image_uint8, load_sample_native
from ...v6.inference.evaluate import load_model, predict_pair, select_samples, summary_json
from ...v6.inference.metrics import confusion_from_pred
from ...v7 import config_v7 as v7cfg
from ...v7.scripts.compare_runs import collect as collect_baselines
from .. import config_v8 as cfg
from ..data import photo_norm as pn

SHIFTS = (-40, -20, 0, 20, 40, 60)               # 0-1: 더하는 단계
LIGHT_P2P = (5, 10, 20, 40)                      # 0-2 · 0-3: 가짜 조명의 최대 − 최소 (%)
C = v6cfg.N_CLASSES_V6
AB = v6cfg.CLASS_ABBRS_V6


# ---------------------------------------------------------------------------
# 가짜 조명
# ---------------------------------------------------------------------------

def fake_light(hw: tuple[int, int], p2p_pct: float) -> np.ndarray:
    """기울어진 면 + 넓은 얼룩 하나 → (H, W) float32, 평균 1, 최대 − 최소 = p2p_pct %. 사진 크기와 상관없이 같은 무늬다."""
    h, w = hw
    u = np.linspace(-1.0, 1.0, w, dtype=np.float64)[None, :]
    v = np.linspace(-1.0, 1.0, h, dtype=np.float64)[:, None]
    plane = 0.8 * u + 0.6 * v                                                    # −1.4 … +1.4
    blob = np.exp(-((u - 0.35) ** 2 + (v + 0.25) ** 2) / (2 * 0.55 ** 2))        # 가운데에서 비낀 넓은 얼룩
    f = 0.5 * plane / 2.8 + 0.5 * blob
    f = (f - f.mean()) / (f.max() - f.min())
    return (1.0 + (p2p_pct / 100.0) * f).astype(np.float32)


def lit(img: np.ndarray, light: np.ndarray | None) -> np.ndarray:
    """사진 × 가짜 조명 → uint8 (카메라가 저장한 것처럼 반올림하고 자른다)."""
    if light is None:
        return img
    return np.clip(np.rint(img.astype(np.float32) * light), 0, 255).astype(np.uint8)


# ---------------------------------------------------------------------------
# 기준선 모델 · 채점
# ---------------------------------------------------------------------------

def baseline_run(fold: str, seed: int, img_size: int, which: str) -> str:
    """보정 없이 학습한 실행의 이름. v7 의 `N` 실행, 없고 seed 가 42 면 v6 의 실행 (같은 계산이다 — v7/docker/README 의 스모크)."""
    v7 = v7cfg.default_run_name(v6cfg.DINO_BACKBONE, img_size, v7cfg.DEFAULT_VARIANT, seed)
    if (v6cfg.run_ckpt_dir(v7, fold) / f"{which}.pt").is_file():
        return v7
    v6 = v6cfg.default_run_name(v6cfg.DINO_BACKBONE, img_size)
    if seed == v6cfg.RANDOM_SEED and (v6cfg.run_ckpt_dir(v6, fold) / f"{which}.pt").is_file():
        return v6
    raise SystemExit(f"FATAL: fold '{fold}' · seed {seed} 의 기준선 ckpt ({which}.pt) 가 없다 — {v7} 도 {v6} 도 아니다")


@torch.no_grad()
def score(model, layers: list[dict], transform, img_size: int, device, defect_eval) -> dict:
    """검증 레이어에 transform 을 건 사진을 넣어 채점 → {"overall": 요약, "by_group": {소스: 요약}} (원본 해상도)."""
    total = np.zeros((C, C), dtype=np.int64)
    by: dict[str, np.ndarray] = {}
    for L in layers:
        a0, a1 = transform(L)
        pred = predict_pair(model, a0, a1, img_size, device, resolutions=("native",))["native"]
        gt = torch.from_numpy(L["gt"].astype(np.int64)).to(device)
        cm = confusion_from_pred(pred, gt, C).reshape(C, C).cpu().numpy()
        total += cm
        by[L["group"]] = by.get(L["group"], 0) + cm
        del pred, gt
    return {"overall": summary_json(total, defect_eval), "by_group": {g: summary_json(m, defect_eval) for g, m in by.items()}}


def _brief(res: dict) -> dict:
    """요약에서 표에 쓰는 것만."""
    o = res["overall"]
    return {
        "defect_miou": o["defect_miou"], "miou": o["miou"], "iou": o["iou"],
        "miss": o["miss_false_alarm"]["miss"], "false_alarm": o["miss_false_alarm"]["false_alarm"],
        "by_group_defect_miou": {g: d["defect_miou"] for g, d in res["by_group"].items()},
    }


# ---------------------------------------------------------------------------
# 0-3: 가짜 조명을 얹은 소스에서 지도를 다시 구한다 (라벨 없이)
# ---------------------------------------------------------------------------

def estimate_maps(samples_by_group: dict[str, list], levels: tuple[int, ...]) -> dict[str, dict[int, dict]]:
    """{소스: {가짜 조명의 크기: summarize_map 의 결과 + 얹은 지도와의 일치}}. 크기 0 은 원본 사진이다."""
    out: dict[str, dict[int, dict]] = {}
    for g, ss in samples_by_group.items():
        cells = {p: [] for p in levels}
        lights: dict[tuple, dict[int, np.ndarray]] = {}
        for s in ss:
            img1 = load_image_uint8(s.img1_path)
            if img1.shape not in lights:
                lights[img1.shape] = {p: (fake_light(img1.shape, p) if p > 0 else None) for p in levels}
            for p in levels:
                cells[p].append(pn.layer_cells(lit(img1, lights[img1.shape][p])))
        out[g] = {p: pn.summarize_map(pn.pooled_cells(cells[p]), len(ss)) for p in levels}
        # 얹은 지도를 되찾았는가 — 구한 지도에는 그 소스에 원래 있던 무늬도 들어 있으므로, 원본 사진의 지도 (0%) 로 나눈 것을 얹은 지도와 견준다
        base = pn.surface_at(out[g][0]["coef"], (cfg.GRID, cfg.GRID)) if 0 in out[g] and out[g][0]["coef"] is not None else None
        for p in levels:
            d = out[g][p]
            if p > 0 and d["coef"] is not None:
                est = pn.surface_at(d["coef"], (cfg.GRID, cfg.GRID)).astype(np.float64)
                if base is not None:
                    est = est / base
                est = est / est.mean()
                inj = pn.cell_grid(fake_light((cfg.GRID * 8, cfg.GRID * 8), p)).astype(np.float64)
                inj = inj / inj.mean()
                d["corr_with_injected"] = float(np.corrcoef(est.ravel(), inj.ravel())[0, 1])
                d["rms_vs_injected_pct"] = float(np.sqrt(np.mean((est / inj - 1.0) ** 2)) * 100)
                d["injected_sd_pct"] = float(inj.std() * 100)
        print(f"[check]   지도 추정 {v6cfg.SOURCE_SHORT.get(g, g):5s} (n={len(ss)}): "
              + "  ".join(f"{p}%→크기 {out[g][p]['map_sd_pct']:.1f}% 설명 {out[g][p]['map_r2']:.2f} {'통과' if out[g][p]['use_map'] else '걸림'}"
                          for p in levels if out[g][p]["coef"] is not None), flush=True)
    return out


def source_norm(name: str, d: dict, n_layers: int) -> pn.SourceNorm:
    return pn.SourceNorm(name=name, n_layers=n_layers, brightness=float("nan"), shift=0.0, coef=d["coef"], use_map=d["use_map"])


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--folds", nargs="+", default=["inlayer", "loso_L21"], choices=list(v6cfg.CV_FOLDS))
    ap.add_argument("--seed", type=int, default=v6cfg.RANDOM_SEED, help="기준선 모델의 seed")
    ap.add_argument("--img-size", type=int, default=v6cfg.IMG_SIZE)
    ap.add_argument("--which", type=str, default="last", choices=["last", "best"])
    ap.add_argument("--norm-set", type=str, default=cfg.DEFAULT_SET_NAME[cfg.MODE_B],
                    help="0-1b 에 쓸 보정 세트 (그 세트의 소스별 더한 값을 읽는다). 없으면 0-1b 를 건너뛴다")
    ap.add_argument("--quick", action="store_true", help="스모크: fold 마다 앞 2 레이어, 조건 몇 개만")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    img_size = round_to_patch(args.img_size, 14)
    shifts = (0, 40) if args.quick else SHIFTS
    levels = (20,) if args.quick else LIGHT_P2P
    out_dir = cfg.CHECK_FIG_DIR / (f"sz{img_size}_s{args.seed}" + ("_quick" if args.quick else ""))
    out_dir.mkdir(parents=True, exist_ok=True)

    # 0-1b 에 쓸 소스별 더한 값
    src_shift: dict[str, float] | None = None
    try:
        nroot = cfg.resolve_norm_root(args.norm_set)
        ndoc = json.loads((nroot / cfg.NORM_JSON).read_text(encoding="utf-8"))
        src_shift = {k: float(d["shift"]) for k, d in ndoc["sources"].items()}       # 통계의 값 (세트에 건 값이 아니라 — raw 세트여도 같다)
    except FileNotFoundError:
        print(f"[check] 보정 세트 '{args.norm_set}' 가 없다 — 0-1b (소스마다 밝기를 맞춰 넣기) 는 건너뛴다")

    all_samples = enumerate_samples_v6()
    by_group_all: dict[str, list] = {}
    for s in all_samples:
        by_group_all.setdefault(s.group, []).append(s)

    doc = {"img_size": img_size, "seed": args.seed, "which": args.which, "quick": args.quick,
           "shifts": list(shifts), "light_p2p_pct": list(levels), "folds": {}}
    map_cache: dict[str, dict[int, dict]] = {}
    t_all = time.time()
    for fold in args.folds:
        run = baseline_run(fold, args.seed, img_size, args.which)
        fig_dir = v6cfg.run_fig_dir(run, fold)
        model, _ = load_model(v6cfg.run_ckpt_dir(run, fold) / f"{args.which}.pt", device)
        val, defect_eval, _ = select_samples(all_samples, fold, None, fold_run_json=fig_dir / "run.json", img_size=img_size)
        if args.quick:
            val = val[:2]
        layers = []
        for s in val:
            i0, i1, gt = load_sample_native(s)
            layers.append({"sample": s, "group": s.group, "i0": i0, "i1": i1, "gt": gt})
        groups = sorted({L["group"] for L in layers})
        print(f"[check] === {fold}: 기준선 {run}/{args.which}.pt, 검증 {len(layers)} 레이어, 소스 {[v6cfg.SOURCE_SHORT.get(g, g) for g in groups]} ===", flush=True)

        def run_cond(name: str, transform) -> dict:
            t0 = time.time()
            b = _brief(score(model, layers, transform, img_size, device, defect_eval))
            print(f"[check]   {name:34s} 결함 mIoU {b['defect_miou']:.4f}  놓침 {100 * b['miss']:.1f}%  잘못 짚음 {100 * b['false_alarm']:.2f}%"
                  f"  ({time.time() - t0:.0f}s)", flush=True)
            return b

        F: dict = {"run": run, "n_layers": len(layers), "groups": groups,
                   "defect_classes": [AB[c] for c in (defect_eval or v6cfg.DEFECT_IDS_V6)]}

        # ── 0-1 밝기를 통째로 옮긴다 ──
        F["shift"] = {}
        for d in shifts:
            F["shift"][str(d)] = run_cond(f"0-1 밝기 {d:+d}", lambda L, d=d: (pn.apply_norm(L["i0"], shift=d), pn.apply_norm(L["i1"], shift=d)))
        ref = F["shift"]["0"]
        # 저장해 둔 채점과 같은가 (보정 없는 사진 = v6 의 evaluate 가 낸 값)
        stored = fig_dir / f"eval_native_{args.which}" / "metrics.json"
        if stored.is_file() and not args.quick:
            sm = json.loads(stored.read_text())["overall"]["defect_miou"]
            F["stored_defect_miou"] = sm
            if abs(sm - ref["defect_miou"]) > 2e-3:
                raise SystemExit(f"FATAL: 보정 없는 사진의 결함 mIoU ({ref['defect_miou']:.6f}) 가 저장해 둔 채점 ({sm:.6f}) 과 다르다 — 채점 경로가 다르다")
            print(f"[check]   보정 없는 사진의 점수 {ref['defect_miou']:.6f} — 저장해 둔 채점 ({stored.parent.name}) {sm:.6f} (차 {ref['defect_miou'] - sm:+.1e})")

        # ── 0-1b 소스마다 밝기를 맞춰 넣는다 (다시 학습하지 않은 모델에) ──
        if src_shift is not None:
            F["per_source_shift"] = dict(
                run_cond("0-1b 소스마다 115 로", lambda L: (pn.apply_norm(L["i0"], shift=src_shift[L["group"]]),
                                                        pn.apply_norm(L["i1"], shift=src_shift[L["group"]]))),
                shifts={v6cfg.SOURCE_SHORT.get(g, g): src_shift[g] for g in groups},
            )

        # ── 0-2 가짜 조명을 얹는다 ──
        light_cache: dict[tuple, np.ndarray] = {}

        def light_for(shape, p):
            if (shape, p) not in light_cache:
                light_cache[(shape, p)] = fake_light(shape, p)
            return light_cache[(shape, p)]

        F["light"] = {}
        for p in levels:
            F["light"][str(p)] = run_cond(f"0-2 가짜 조명 {p}%", lambda L, p=p: (lit(L["i0"], light_for(L["i0"].shape, p)),
                                                                           lit(L["i1"], light_for(L["i1"].shape, p))))

        # ── 0-3 지도를 다시 구해 나눈다 ──
        lv3 = (0,) + tuple(levels)
        for g in groups:
            if g not in map_cache:
                map_cache.update(estimate_maps({g: by_group_all[g]}, (0,) + tuple(LIGHT_P2P if not args.quick else levels)))
        F["removal"] = {"gates_on": {}, "gates_off": {}, "maps": {
            v6cfg.SOURCE_SHORT.get(g, g): {str(p): {k: v for k, v in map_cache[g][p].items() if k != "coef"} for p in lv3} for g in groups}}
        norm_maps: dict[tuple, np.ndarray] = {}

        def removed(L, p, gates: bool):
            d = map_cache[L["group"]][p]
            out = []
            for a in (L["i0"], L["i1"]):
                x = lit(a, light_for(a.shape, p)) if p > 0 else a
                if d["coef"] is not None and (d["use_map"] or not gates):
                    key = (L["group"], p, a.shape)
                    if key not in norm_maps:
                        norm_maps[key] = source_norm(L["group"], d, len(by_group_all[L["group"]])).map_for(a.shape)
                    x = pn.apply_norm(x, shift=0.0, light=norm_maps[key])
                out.append(x)
            return out[0], out[1]

        for p in lv3:
            for key, gates in (("gates_on", True), ("gates_off", False)):
                F["removal"][key][str(p)] = run_cond(f"0-3 조명 {p}% → 나눔 (안전 장치 {'켬' if gates else '끔'})",
                                                     lambda L, p=p, gates=gates: removed(L, p, gates))
            norm_maps.clear()

        # seed 사이의 흩어짐 (기준선 N 의 seed 들)
        base = [r for r in collect_baselines(fold, img_size, "native", args.which, True) if r["variant"] == v7cfg.DEFAULT_VARIANT]
        vals = [r["defect_miou"] for r in base]
        F["baseline_seeds"] = {"seeds": [r["seed"] for r in base], "defect_miou": vals,
                               "mean": float(np.mean(vals)) if vals else None,
                               "sigma": float(np.std(vals, ddof=1)) if len(vals) >= 2 else None}
        doc["folds"][fold] = F
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()

    doc["wall_minutes"] = (time.time() - t_all) / 60.0
    (out_dir / "results.json").write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    text = write_summary(doc, out_dir)
    print(text)
    plot(doc, out_dir)
    print(f"[check] {doc['wall_minutes']:.1f}분 → {out_dir}")


# ---------------------------------------------------------------------------
# 표 · 그림
# ---------------------------------------------------------------------------

def _d(v, ref) -> str:
    return "–" if v is None or ref is None else f"{v - ref:+.3f}"


def write_summary(doc: dict, out_dir: Path) -> str:
    L = [f"# v8 0단계 — 학습 없이 잰 민감도 (기준선 모델 seed {doc['seed']}, `{doc['which']}.pt`, 원본 해상도)", "",
         "보정 없이 학습한 모델에 검증 레이어의 사진을 바꿔 넣고 결함 mIoU 를 쟀다. 정답은 그대로다. \"차\" 는 사진을 바꾸지 않았을 때와의 차다.",
         "seed 사이의 σ 는 같은 설정을 seed 만 바꿔 학습한 기준선들의 결함 mIoU 의 표준편차다 — 이보다 작은 차는 읽지 않는다.", ""]
    if doc.get("quick"):
        L += ["**스모크 (`--quick`) 의 값이다 — 레이어 2 장, 조건 일부. 점수에 뜻이 없다.**", ""]
    for fold, F in doc["folds"].items():
        ref = F["shift"]["0"]["defect_miou"]
        bs = F["baseline_seeds"]
        srcs = [v6cfg.SOURCE_SHORT.get(g, g) for g in F["groups"]]
        L += [f"## `{fold}` — 검증 {F['n_layers']} 레이어 ({' · '.join(srcs)})", "",
              f"기준선 `{F['run']}`. 사진을 바꾸지 않았을 때 결함 mIoU {ref:.3f}. "
              + (f"seed 사이의 σ = {bs['sigma']:.4f} (seed {bs['seeds']})." if bs.get("sigma") is not None else "seed 사이의 σ 는 모른다."), ""]

        def row(name, b):
            per = " | ".join("–" if b["by_group_defect_miou"].get(g) is None else f"{b['by_group_defect_miou'][g]:.3f}" for g in F["groups"])
            return (f"| {name} | {b['defect_miou']:.3f} | {_d(b['defect_miou'], ref)} | {100 * b['miss']:.1f}% | {100 * b['false_alarm']:.2f}% | "
                    f"{b['iou'][0]:.3f} | {b['iou'][1]:.3f} | {per} |")

        head = ["| 넣은 사진 | 결함 mIoU | 차 | 놓친 비율 | 잘못 짚은 비율 | Powder IoU | Printed IoU | " + " | ".join(f"{s} 결함 mIoU" for s in srcs) + " |",
                "|---|---:|---:|---:|---:|---:|---:|" + "---:|" * len(srcs)]
        L += ["### 0-1 밝기를 통째로 옮긴 것", ""] + head
        for d, b in F["shift"].items():
            L.append(row(f"{int(d):+d} 단계" if int(d) else "그대로", b))
        if "per_source_shift" in F:
            sh = " · ".join(f"{k} {v:+.0f}" for k, v in F["per_source_shift"]["shifts"].items())
            L.append(row(f"소스마다 115 로 ({sh})", F["per_source_shift"]))
        L += ["", "### 0-2 가짜 조명을 얹은 것 (걷어내지 않음)", ""] + head
        for p, b in F["light"].items():
            L.append(row(f"최대 − 최소 {p}%", b))
        L += ["", "### 0-3 조명 지도를 다시 구해 나눈 것", "",
              "\"안전 장치 켬\" 은 실제로 쓸 방식이다 (통과한 소스만 나눈다). \"끔\" 은 모든 소스를 나눈 것이다. 0% 는 원본 사진에서 구한 지도로 나눈 것이다.", ""] + head
        for p in F["removal"]["gates_on"]:
            L.append(row(f"조명 {p}% → 나눔, 안전 장치 켬", F["removal"]["gates_on"][p]))
            L.append(row(f"조명 {p}% → 나눔, 안전 장치 끔", F["removal"]["gates_off"][p]))
        L += ["", "소스마다 다시 구한 지도 (라벨 없이, 그 소스의 도포 직후 사진 전부로):", "",
              "| 소스 | 얹은 조명 (그 표준편차) | 구한 지도의 크기 | 면이 설명하는 정도 | 안전 장치 (S1 · S2 · S3) | 얹은 지도와의 상관 | 얹은 지도와의 차 (RMS) |",
              "|---|---:|---:|---:|---|---:|---:|"]
        for s, per in F["removal"]["maps"].items():
            for p, d in per.items():
                g = d["gates"]
                gate = "".join("O" if g[k] else "X" for k in ("S1_layers", "S2_size", "S3_smooth")) + (" → 통과" if d["use_map"] else " → 걸림")
                sd = "–" if d["map_sd_pct"] is None else f"{d['map_sd_pct']:.1f}%"
                r2 = "–" if d["map_r2"] is None else f"{d['map_r2']:.2f}"
                co = "–" if d.get("corr_with_injected") is None else f"{d['corr_with_injected']:.2f}"
                rm = "–" if d.get("rms_vs_injected_pct") is None else f"{d['rms_vs_injected_pct']:.1f}%"
                inj = "" if d.get("injected_sd_pct") is None else f" ({d['injected_sd_pct']:.1f}%)"
                L.append(f"| {s} | {p}%{inj} | {sd} | {r2} | {gate} | {co} | {rm} |")
        L.append("")
    L += ["- 가짜 조명 = 기울어진 면 + 넓은 얼룩 하나 (평균 1). 한 소스의 레이어 전부에 같은 무늬를 곱했다. 크기는 최대 − 최소다 (표준편차는 그 1/4 쯤).",
          "- \"얹은 지도와의 상관 · 차\" 는 구한 지도를 그 소스의 원본 사진에서 구한 지도 (0%) 로 나눈 것을 얹은 지도와 견준 값이다 "
          "(구한 지도 = 얹은 것 × 원래 있던 것). 얹은 무늬의 얼룩은 2차식이 아니라서 차가 0 이 되지는 않는다.",
          "- 그림: `curves.png` (결함 mIoU 의 변화), `fake_light.png` (얹은 무늬)."]
    text = "\n".join(L) + "\n"
    (out_dir / "summary.md").write_text(text, encoding="utf-8")
    return text


def plot(doc: dict, out_dir: Path) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    folds = list(doc["folds"])
    f, axes = plt.subplots(len(folds), 2, figsize=(11, 3.8 * len(folds)), squeeze=False)
    for i, fold in enumerate(folds):
        F = doc["folds"][fold]
        ref = F["shift"]["0"]["defect_miou"]
        sig = F["baseline_seeds"].get("sigma")
        ax = axes[i, 0]
        xs = [int(k) for k in F["shift"]]
        ax.plot(xs, [F["shift"][str(k)]["defect_miou"] for k in xs], "o-", color="#1f77b4", label="whole-photo shift")
        if "per_source_shift" in F:
            ax.axhline(F["per_source_shift"]["defect_miou"], color="#d62728", ls="--", lw=1, label="per-source shift to 115 (no retraining)")
        ax.set_xlabel("brightness shift added to both photos (levels)")
        ax = axes[i, 1]
        xs = [int(k) for k in F["light"]]
        ax.plot([0] + xs, [ref] + [F["light"][str(k)]["defect_miou"] for k in xs], "o-", color="#7f7f7f", label="fake lighting, not removed")
        x3 = [int(k) for k in F["removal"]["gates_on"]]
        ax.plot(x3, [F["removal"]["gates_on"][str(k)]["defect_miou"] for k in x3], "s-", color="#2ca02c", label="removed (gates on)")
        ax.plot(x3, [F["removal"]["gates_off"][str(k)]["defect_miou"] for k in x3], "^--", color="#ff7f0e", label="removed (gates off)")
        ax.set_xlabel("fake lighting, max − min (%)")
        for ax in axes[i]:
            if sig is not None:
                ax.axhspan(ref - 2 * sig, ref + 2 * sig, color="k", alpha=0.08, label="unchanged ± 2σ (seeds)")
            ax.axhline(ref, color="k", lw=0.8)
            ax.set_ylabel("defect mIoU (native)")
            ax.set_title(f"{fold} — baseline seed {doc['seed']}, {F['n_layers']} validation layers", fontsize=10)
            ax.grid(alpha=0.3)
            ax.legend(fontsize=8)
    f.tight_layout()
    f.savefig(out_dir / "curves.png", dpi=120)
    plt.close(f)

    lv = doc["light_p2p_pct"]
    f, axes = plt.subplots(1, len(lv), figsize=(3.2 * len(lv), 3.4), squeeze=False)
    for j, p in enumerate(lv):
        m = fake_light((256, 256), p)
        im = axes[0, j].imshow(m, cmap="RdBu_r", vmin=0.8, vmax=1.2)
        axes[0, j].set_title(f"max − min {p}%  (sd {m.std() * 100:.1f}%)", fontsize=9)
        axes[0, j].set_xticks([]); axes[0, j].set_yticks([])
    f.colorbar(im, ax=axes.ravel().tolist(), shrink=0.8, label="multiplier")
    f.savefig(out_dir / "fake_light.png", dpi=110, bbox_inches="tight")
    plt.close(f)


if __name__ == "__main__":
    main()
