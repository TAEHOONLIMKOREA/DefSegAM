"""ORNL 빌드의 무작위 레이어에서 DSCNN · v4 · v5 · v6 · v7 의 예측을 나란히 본다 (참고용 — PLAN_v7 §4.4).

ORNL 빌드에는 사람 GT 가 없다. 여기서 나오는 것은 정확도가 아니라 **모델끼리의 비교**다. 판정에는 쓰지 않는다.
    열:  visible/0 | visible/1 | DSCNN (HDF5) | v4 | v5 | v6 | v7

v6 의 `inference/compare_ornl.py` 가 하는 일에 v7 의 열 하나를 더한 것이다. 레이어 뽑기 · 읽기 · 타일 · 쪽 그리기 · 수치는 그 모듈의 함수를 그대로 부른다.
  - v7 = `--variant` · `--seed` · `--fold` 로 고른 실행의 ckpt (기본: 꼬리표 W, seed 42, inlayer).
  - v6 = **같은 fold · 같은 seed 의 기준선** (사람 GT 만으로 학습한 것). inlayer · seed 42 면 v6 의 P0 다. 그래야 두 열의 차이가 합성 데이터 하나다.
    (`--v6-checkpoint` 로 바꿀 수 있다 — 예: v6 의 전수 학습 모델.)
  - v4 는 v6 의 비교와 같은 ckpt 다. v5 = 같은 fold 의 모델 — Soot 를 Spatter 로 배운 실행 (`…_11cls_v5-soot2sp`) 이 있으면 그것을, 없으면 기존 v5 를 쓴다
    (`--v5-checkpoint` 로 바꾼다. v6 의 비교가 쓰던 기존 v5 의 전수 학습 모델은 2026-11-23 에 지웠다).
    입력 밝기는 v6 의 비교와 같다 (v4 는 레이어마다 늘린 사진, v5 · v6 · v7 은 원본 밝기). TTA · 후처리 없음.
  - 레이어는 v6 의 비교와 같은 규칙 · 같은 seed 로 뽑는다 — 빌드마다 4 장이면 v6 의 100 레이어 가운데 빌드마다 앞의 4 장과 같다.

출력 (figures/<v7 run>/v6/<cv/<fold> | full>/ornl_random<N>_<which>/):
    pages/page_XX_<build>.png   비교 그림 — 한 쪽에 한 빌드
    layers.json                  뽑은 레이어
    summary.md · summary.json    모델마다 클래스별 예측 화소 비율, CAD 파트 마스크와의 겹침, RS 선의 가로 길이, v7 과 다른 모델의 일치

사용 (CWD = 3DP_VPPM/. compose 로는 DEFSEG_V7_PHASE=ornl_compare):
    python -m DefSeg_AM.v7.scripts.compare_ornl_v7                       # W · seed 42 · inlayer, 빌드마다 4 레이어
    python -m DefSeg_AM.v7.scripts.compare_ornl_v7 --fold loso_L21 --n-per-build 2
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ...common.models.model import round_to_patch
from ...v6 import config_v6 as v6cfg
from ...v6.inference import compare_ornl as co
from ...v6.inference.baselines import NONE_ID, old_to_new_lut
from ...v6.inference.evaluate import load_model, predict_pair
from .. import config_v7 as cfg

AB = v6cfg.CLASS_ABBRS_V6
NC = v6cfg.N_CLASSES_V6


def baseline_ckpt(backbone: str, img_size: int, fold: str | None, seed: int, which: str) -> Path | None:
    """같은 fold · 같은 seed 의 기준선 (합성 없음). v7 의 N 실행이 있으면 그것, seed 42 는 v6 의 실행도 본다."""
    cands = [v6cfg.run_ckpt_dir(cfg.default_run_name(backbone, img_size, "N", seed), fold) / f"{which}.pt"]
    if seed == v6cfg.RANDOM_SEED:
        cands.append(v6cfg.run_ckpt_dir(v6cfg.default_run_name(backbone, img_size), fold) / f"{which}.pt")
    for p in cands:
        if p.is_file():
            return p
    return None


def main():
    ap = argparse.ArgumentParser(description="ORNL 빌드의 무작위 레이어에서 DSCNN · v4 · v5 · v6 · v7 예측 비교 (참고용)")
    ap.add_argument("--variant", type=str, default="W", help="v7 실행의 꼬리표")
    ap.add_argument("--seed", type=int, default=v6cfg.RANDOM_SEED, help="v7 실행의 seed")
    ap.add_argument("--fold", type=str, default="inlayer", help="v7 실행의 fold. 'full' 이면 전수 학습 모델")
    ap.add_argument("--which", type=str, default="last", choices=["last", "best"])
    ap.add_argument("--backbone", type=str, default=v6cfg.DINO_BACKBONE)
    ap.add_argument("--img-size", type=int, default=v6cfg.IMG_SIZE)
    ap.add_argument("--v6-checkpoint", type=str, default=None, help="기본: 같은 fold · 같은 seed 의 기준선")
    ap.add_argument("--v5-checkpoint", type=str, default=None,
                    help="기본: 같은 fold 의 v5 — Soot 를 Spatter 로 배운 실행 (…_11cls_v5-soot2sp) 이 있으면 그것, 없으면 기존 v5")
    ap.add_argument("--v4-checkpoint", type=str, default=str(v6cfg.CHECKPOINT_DIR / v6cfg.V4_RUN_NAME / "stage2_best.pt"))
    ap.add_argument("--builds", type=str, nargs="+", default=list(v6cfg.ORNL_BUILD_FILES), choices=list(v6cfg.ORNL_BUILD_FILES))
    ap.add_argument("--n-per-build", type=int, default=4)
    ap.add_argument("--layer-seed", type=int, default=v6cfg.RANDOM_SEED, help="레이어를 뽑는 seed (v6 의 비교와 같은 값이 기본)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold = None if args.fold == "full" else args.fold
    run7 = cfg.default_run_name(args.backbone, args.img_size, args.variant, args.seed)
    ckpt7 = v6cfg.run_ckpt_dir(run7, fold) / f"{args.which}.pt"
    if not ckpt7.is_file():
        ap.error(f"v7 ckpt 없음: {ckpt7}")
    ckpt6 = Path(args.v6_checkpoint) if args.v6_checkpoint else baseline_ckpt(args.backbone, args.img_size, fold, args.seed, args.which)
    if ckpt6 is None or not ckpt6.is_file():
        ap.error(f"v6 (기준선) ckpt 를 찾지 못했다 — --v6-checkpoint 로 준다 (fold={args.fold}, seed={args.seed})")
    if args.v5_checkpoint:
        ckpt5 = Path(args.v5_checkpoint)
    else:                                                            # compare_gt_v7 과 같은 규칙 (기존 v5 의 전수 학습 모델은 2026-11-23 에 지웠다)
        sub = Path("full") if fold is None else Path("cv") / fold
        cands = [v6cfg.CHECKPOINT_DIR / f"{v6cfg.V5_RUN_NAME}-soot2sp" / sub / f"{args.which}.pt",
                 v6cfg.CHECKPOINT_DIR / v6cfg.V5_RUN_NAME / sub / f"{args.which}.pt"]
        ckpt5 = next((p for p in cands if p.is_file()), cands[-1])

    m7, c7 = load_model(ckpt7, device)
    m6, c6 = load_model(ckpt6, device)
    img_size = round_to_patch(args.img_size, m7.patch_size)
    old, old_cfg, checkpoints = {}, {}, {"v7": co._rel(ckpt7), "v6": co._rel(ckpt6)}
    for name, p in (("v4", Path(args.v4_checkpoint)), ("v5", ckpt5)):
        if p.exists():
            old[name], ck_old = load_model(p, device, n_classes=v6cfg.N_CLASSES_V4)
            old_cfg[name] = ck_old.get("config", {}) or {}
            checkpoints[name] = co._rel(p)
        else:
            print(f"[compare7] {name} ckpt 없음 ({p}) — 그 열을 뺀다")
    models = ["dscnn"] + [m for m in ("v4", "v5") if m in old] + ["v6", "v7"]
    lut = old_to_new_lut("se", device)
    palette = torch.tensor(list(v6cfg.DIVA_COLORS) + [v6cfg.UNLABELED_COLOR], dtype=torch.float32, device=device)
    synth = ((c7.get("config", {}) or {}).get("synth") or {}).get("name") or "none"      # 학습 스크립트가 ckpt 에 적은 합성 세트의 이름

    builds = {}
    for b in args.builds:
        t0 = time.time()
        builds[b] = co.pick_layers(b, args.n_per_build, args.layer_seed)
        print(f"[compare7] {b}: {builds[b]['n_layers']} 레이어 가운데 {builds[b]['layers']}  ({time.time() - t0:.0f}s)")
    items = [(b, li) for b, info in builds.items() for li in info["layers"]]
    n_layers = len(items)

    out_dir = Path(args.out_dir) if args.out_dir else v6cfg.run_fig_dir(run7, fold) / f"ornl_random{n_layers}_{args.which}"
    (out_dir / "pages").mkdir(parents=True, exist_ok=True)
    for p in sorted((out_dir / "pages").glob("page_*.png")):         # 이 스크립트가 앞서 만든 쪽만 지운다
        p.unlink()
    (out_dir / "layers.json").write_text(json.dumps({"seed": args.layer_seed, "builds": builds}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[compare7] {n_layers} 레이어 | 모델 {models} | v7={checkpoints['v7']} | v6={checkpoints['v6']} | out={out_dir}")

    info = dict(co.MODEL_INFO)
    base_tag = "baseline, GT only" if fold is not None else "full model, GT only"
    info["v6"] = ("DefSeg-AM v6", f"{base_tag} ({v6cfg.run_tag(fold)})", "v6")
    info["v7"] = (f"DefSeg-AM v7-{args.variant}", f"GT + synthetic set {synth}", "v7")
    if "v5" in old:                                # v5 의 열 — 어느 ckpt 인지 (fold · Soot 를 Spatter 로 배웠는지) 를 머리글에 적는다
        p5 = ckpt5
        where = "full" if p5.parent.name == "full" else f"cv/{p5.parent.name}"
        soot = ", Soot → Spatter" if old_cfg["v5"].get("soot_mode") == "spatter" else ""
        info["v5"] = ("DefSeg-AM v5", f"GT only, 11 classes{soot} ({where})", "v5")
    columns = [("visible/0", "after melting"), ("visible/1", "after spreading")] + [info[m][:2] for m in models]
    co.PAGE_NOTE = (                                                    # 쪽 아래의 설명 — v6 모듈의 글을 이 그림에 맞게 바꾼다 (실행 중에만)
        f"No human ground truth exists for these builds — columns 3–{len(columns)} are all model predictions. "
        "Old classes are shown with pbf-2026.11 ids: Swelling → SE, old Incomplete Spreading → IS, Misprint / Under Melting → no class.",
        f"v6 and v7 differ in one thing: v7 also trained on synthetic layers (same fold, same seed {args.seed}). Inputs: v5 · v6 · v7 get raw brightness, "
        "v4 the per-layer stretched photos. No TTA, no post-processing. Photos are 3×3 averages, contrast-stretched for display only.",
    )

    loader = DataLoader(co.OrnlLayers(items), batch_size=None, shuffle=False, num_workers=args.workers,
                        collate_fn=co._identity, prefetch_factor=2 if args.workers > 0 else None)
    counts = np.zeros((n_layers, len(models), co.N_BINS), dtype=np.int64)
    cross = np.zeros((n_layers, len(models), NC, co.N_BINS), dtype=np.int64)      # v7 × 모델
    part = np.zeros((n_layers, len(models), 2, co.N_BINS), dtype=np.int64)
    rs_rows = np.zeros((n_layers, len(models), 3), dtype=np.float64)
    page_rows, page_no, t0 = [], 0, time.time()
    n_pages = sum(1 for info_b in builds.values() if info_b["layers"])
    for i, item in enumerate(loader):
        b, li = item["build"], int(item["layer"])
        raw0, raw1 = item["raw0"], item["raw1"]
        d = torch.from_numpy(item["dscnn"].astype(np.int64)).to(device)
        pred = {"dscnn": torch.where(d < 0, torch.full_like(d, NONE_ID), d)}
        if "v4" in old:
            pred["v4"] = lut[predict_pair(old["v4"], item["str0"], item["str1"], img_size, device, ("native",))["native"]]
        if "v5" in old:
            pred["v5"] = lut[predict_pair(old["v5"], raw0, raw1, img_size, device, ("native",))["native"]]
        pred["v6"] = predict_pair(m6, raw0, raw1, img_size, device, ("native",))["native"]
        pred["v7"] = predict_pair(m7, raw0, raw1, img_size, device, ("native",))["native"]
        on_part = torch.from_numpy(item["part"]).to(device).long()
        for k, m in enumerate(models):
            counts[i, k] = torch.bincount(pred[m].reshape(-1), minlength=co.N_BINS).cpu().numpy()
            cross[i, k] = co._bincount2(pred["v7"], pred[m], NC, co.N_BINS)
            part[i, k] = co._bincount2(on_part, pred[m], 2, co.N_BINS)
            rs_rows[i, k] = co.rs_row_stats(pred[m])
        tiles = [co.tile_gray(co.display_u8(raw0), device), co.tile_gray(co.display_u8(raw1), device)]
        tiles += [co.tile_label(pred[m], palette) for m in models]
        page_rows.append({"build": b, "label": f"{b} · layer {li}", "tiles": tiles})
        if i + 1 == n_layers or items[i + 1][0] != b:                      # 한 쪽에 한 빌드
            page_no += 1
            title = (f"ORNL {b} — random layers, DSCNN vs DefSeg-AM v4 · v5 · v6 · v7 "
                     f"(v7 = {args.variant}, seed {args.seed}, {v6cfg.run_tag(fold)} {args.which})    page {page_no} / {n_pages}")
            co.render_page(page_rows, columns, title, out_dir / "pages" / f"page_{page_no:02d}_{b}.png")
            page_rows = []
        print(f"[compare7] {i + 1}/{n_layers}  {b} layer {li}  ({time.time() - t0:.0f}s)")

    # ----- 수치 -----
    tot = counts.sum(axis=(0, 2), keepdims=False)                           # 모델마다의 화소 수
    share = counts.sum(axis=0) / tot[:, None]                               # (모델, 11)
    doc = {
        "note": "사람 GT 가 없는 사진이다 — 모델끼리의 비교이지 점수가 아니다.",
        "n_layers": n_layers, "builds": {b: v["layers"] for b, v in builds.items()}, "layer_seed": args.layer_seed,
        "models": models, "checkpoints": checkpoints, "img_size": img_size,
        "v7": {"variant": args.variant, "seed": args.seed, "fold": args.fold, "synth_set": synth, "step": c7.get("step")},
        "class_share": {m: {**{AB[c]: float(share[k, c]) for c in range(NC)}, "no_class": float(share[k, NONE_ID])} for k, m in enumerate(models)},
        "defect_share": {m: float(share[k, 2:NC].sum()) for k, m in enumerate(models)},
        "cad": {m: co.part_overlap(part[:, k].sum(axis=0)) for k, m in enumerate(models)},
        "rs": {m: co.rs_summary(rs_rows[:, k]) for k, m in enumerate(models)},
        "v7_vs": {m: co.agreement(cross[:, k].sum(axis=0)) for k, m in enumerate(models) if m != "v7"},
        "by_build": {},
    }
    bi = np.array([b for b, _ in items])
    for b in builds:
        sel = bi == b
        s = counts[sel].sum(axis=0)
        s = s / s.sum(axis=1, keepdims=True)
        doc["by_build"][b] = {m: {AB[c]: float(s[k, c]) for c in range(2, NC)} for k, m in enumerate(models)}
    (out_dir / "summary.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")

    pct = lambda v: "–" if v is None else (f"{100 * v:.2f}" if v >= 0.0005 else ("0" if v == 0 else "<0.05"))
    f3 = lambda v: "–" if v is None else f"{v:.3f}"
    names = {"dscnn": "DSCNN", "v4": "v4", "v5": "v5", "v6": "v6 (기준선)", "v7": f"v7-{args.variant}"}
    L = [f"# ORNL 무작위 {n_layers} 레이어 — DSCNN · v4 · v5 · v6 · v7 의 예측 비교", "",
         "사람 GT 가 없는 사진이다. 아래는 **모델끼리의 비교**이지 점수가 아니다. 판정에는 쓰지 않는다 (PLAN_v7 §4.4).", "",
         f"- v7 = `{checkpoints['v7']}` (꼬리표 {args.variant}, seed {args.seed}, 합성 세트 {synth})",
         f"- v6 = `{checkpoints['v6']}` — 같은 fold · 같은 seed 의 기준선 (사람 GT 만). 두 모델의 차이는 합성 데이터 하나다",
         f"- 레이어: 빌드마다 {args.n_per_build} 장 (seed {args.layer_seed}) — " + ", ".join(f"{b} {v['layers']}" for b, v in builds.items()),
         "- B3 · B4 · B5 는 합성 세트의 배경으로 쓴 빌드다 (CAD 배치와 사진 가장자리). B1 · B2 는 어느 단계에서도 쓰지 않았다", "",
         "## 클래스별 예측 화소 비율 (%)", "",
         "| 모델 | " + " | ".join(AB) + " | 대응 없음 | 결함 합 |", "|---|" + "---:|" * (NC + 2)]
    for k, m in enumerate(models):
        L.append(f"| {names[m]} | " + " | ".join(pct(share[k, c]) for c in range(NC)) + f" | {pct(share[k, NONE_ID])} | {pct(share[k, 2:NC].sum())} |")
    L += ["", "## 빌드별 — v6 와 v7 의 결함 클래스 비율 (%)", "", "| 빌드 | 모델 | " + " | ".join(AB[2:]) + " |", "|---|---|" + "---:|" * (NC - 2)]
    for b in builds:
        for m in ("v6", "v7"):
            L.append(f"| {b} | {names[m]} | " + " | ".join(pct(doc['by_build'][b][m][AB[c]]) for c in range(2, NC)) + " |")
    L += ["", "## CAD 파트 마스크와의 겹침 · RS 선", "",
          "| 모델 | 파트 IoU | 파트 밖을 Printed 로 낸 비율 (%) | RS 가 있는 행에서 덮은 폭 (중앙값) | RS 행 수 (레이어 중앙값) | 끝까지 이은 행 |", "|---|---:|---:|---:|---:|---:|"]
    for m in models:
        c, r = doc["cad"][m], doc["rs"][m]
        L.append(f"| {names[m]} | {f3(c['part_iou'])} | {pct(c['off_part_printed_rate'])} | {f3(r['row_cover_median'])} | {r['rows_median']:.0f} | {r['full_width_rows']} |")
    L += ["", f"## v7 과 다른 모델의 일치 (화소 비율, 클래스별은 두 예측 사이의 IoU)", "",
          "| 견준 모델 | 화소 일치 | " + " | ".join(AB) + " |", "|---|---:|" + "---:|" * NC]
    for m in models:
        if m == "v7":
            continue
        a = doc["v7_vs"][m]
        L.append(f"| {names[m]} | {a['pixel_agreement']:.4f} | " + " | ".join(f3(v) for v in a["iou"]) + " |")
    L += ["", "그림: `pages/page_XX_<빌드>.png` — 한 줄이 한 레이어, 열은 용융 직후 · 도포 직후 사진과 다섯 모델의 예측이다."]
    (out_dir / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"[compare7] 끝 → {out_dir}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
