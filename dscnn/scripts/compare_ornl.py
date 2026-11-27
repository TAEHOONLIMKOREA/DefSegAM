"""ORNL 빌드의 무작위 레이어에서 ORNL 의 DSCNN 예측 · **DSCNN 재구현** · v4 · v5 · v6 · v7 을 나란히 본다 (참고용).

v7 의 `scripts/compare_ornl_v7.py` 가 그리는 그림에 **DSCNN 재구현의 열 하나**를 더한 것이다.
    열:  visible/0 | visible/1 | DSCNN (ORNL, HDF5) | DSCNN 재구현 | v4 | v5 | v6 | v7

ORNL 빌드에는 사람 GT 가 없다. 여기서 나오는 것은 정확도가 아니라 **모델끼리의 비교**다.
"DSCNN" 이 두 열이다 — 서로 다른 모델이다:
  - DSCNN (ORNL)   = ORNL 이 HDF5 에 넣어 배포한 예측 (`slices/segmentation_results`). 그 모델의 가중치는 공개되지 않았다.
  - DSCNN 재구현    = Scime et al. (2020) 의 구조를 다시 짜서 사람 GT 만으로 학습한 이 폴더의 모델 (`--tag` · `--seed` · `--fold` 로 고른 실행).

레이어 뽑기 · 읽기 · 타일 · 쪽 그리기 · 수치는 v6 의 `inference/compare_ornl.py` 의 함수를 그대로 부른다 (같은 seed 면 v7 의 비교와 같은 레이어다).
v4 · v5 · v6 · v7 의 ckpt 와 입력은 v7 의 비교와 같다 (v6 = 같은 fold · 같은 seed 의 기준선, v5 = Soot 를 Spatter 로 배운 실행이 있으면 그것).

DSCNN 재구현의 입력 (ORNL 빌드는 학습에 없던 소스다):
  - 원본 밝기, 원본 해상도 (1842²) 를 타일로 잘라 넣는다. 타일 한 변 = `config_dscnn.TILE_SIZE_ORNL` (178 — 같은 장비인 L21 의 225 를 사진 크기의 비로 옮긴 값).
  - ū (빼는 평균 밝기) = **그 빌드에서 뽑은 레이어의 사진 평균**. 채점 스크립트가 학습에 없던 소스에 쓰는 규칙과 같다 (`evaluate.resolve_mean_and_tile`).

출력 (기본: v7 실행의 폴더, 기존 비교 옆 — figures/<v7 run>/v6/cv/<fold>/ornl_random<N>_<which>_dscnn_reimpl_<꼬리표>/):
    pages/page_XX_<build>.png   비교 그림 — 한 쪽에 한 빌드
    layers.json                  뽑은 레이어
    summary.md · summary.json    모델마다 클래스별 예측 화소 비율, CAD 파트 마스크와의 겹침, RS 선, 모델 사이의 일치

사용 (CWD = 3DP_VPPM/. compose 로는 DSCNN_PHASE=ornl_compare):
    python -m DefSeg_AM.dscnn.scripts.compare_ornl --tag sqrtw           # v6 의 가중치로 학습한 실행, 빌드마다 4 레이어
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
from ...v7 import config_v7 as v7cfg
from ...v7.scripts.compare_ornl_v7 import baseline_ckpt
from .. import config_dscnn as cfg
from ..inference.predict import load_dscnn, photo_mean, predict_photos

AB = v6cfg.CLASS_ABBRS_V6
NC = v6cfg.N_CLASSES_V6
RE = "dscnn_re"                                   # DSCNN 재구현의 열 이름 ("dscnn" 은 ORNL 의 예측)


def main():
    ap = argparse.ArgumentParser(description="ORNL 무작위 레이어에서 DSCNN (ORNL) · DSCNN 재구현 · v4 · v5 · v6 · v7 예측 비교 (참고용)")
    ap.add_argument("--fold", type=str, default="inlayer", choices=list(v6cfg.CV_FOLDS))
    ap.add_argument("--seed", type=int, default=cfg.RANDOM_SEED, help="DSCNN 재구현 실행의 seed")
    ap.add_argument("--tag", type=str, default="", help="DSCNN 재구현 실행의 꼬리표")
    ap.add_argument("--which", type=str, default="last", choices=["last", "best"])
    ap.add_argument("--dscnn-checkpoint", type=str, default=None)
    ap.add_argument("--tile-size", type=int, default=cfg.TILE_SIZE_ORNL, help="DSCNN 재구현의 타일 한 변 (ORNL 사진 1842²)")
    ap.add_argument("--backbone", type=str, default=v6cfg.DINO_BACKBONE)
    ap.add_argument("--img-size", type=int, default=v6cfg.IMG_SIZE)
    ap.add_argument("--ref-seed", type=int, default=v6cfg.RANDOM_SEED, help="v6 (기준선) · v7 실행의 seed")
    ap.add_argument("--v7-variant", type=str, default="W", help="v7 실행의 꼬리표. 그 실행이 없으면 v7 의 열을 뺀다")
    ap.add_argument("--v6-checkpoint", type=str, default=None, help="기본: 같은 fold · 같은 seed 의 기준선")
    ap.add_argument("--v5-checkpoint", type=str, default=None, help="기본: 같은 fold 의 v5 — Soot 를 Spatter 로 배운 실행이 있으면 그것")
    ap.add_argument("--v4-checkpoint", type=str, default=str(v6cfg.CHECKPOINT_DIR / v6cfg.V4_RUN_NAME / "stage2_best.pt"))
    ap.add_argument("--builds", type=str, nargs="+", default=list(v6cfg.ORNL_BUILD_FILES), choices=list(v6cfg.ORNL_BUILD_FILES))
    ap.add_argument("--n-per-build", type=int, default=4)
    ap.add_argument("--layer-seed", type=int, default=v6cfg.RANDOM_SEED, help="레이어를 뽑는 seed (v6 · v7 의 비교와 같은 값이 기본)")
    ap.add_argument("--workers", type=int, default=6)
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fold = args.fold
    run_d = cfg.default_run_name(args.seed, args.tag)
    ckpt_d = Path(args.dscnn_checkpoint) if args.dscnn_checkpoint else cfg.run_ckpt_dir(run_d, fold) / f"{args.which}.pt"
    if not ckpt_d.is_file():
        ap.error(f"DSCNN 재구현의 ckpt 없음: {ckpt_d}")
    md, cd = load_dscnn(ckpt_d, device)
    cd_cfg = cd.get("config", {}) or {}
    cw = str(cd_cfg.get("class_weight", "median"))                   # 학습 스크립트가 적는 글: "median frequency (no clip)" | "sqrt inverse frequency (clip 10)"
    weight_name = "paper class weights" if cw.startswith("median") else ("v6 class weights" if cw.startswith("sqrt") else cw)

    ckpt6 = Path(args.v6_checkpoint) if args.v6_checkpoint else baseline_ckpt(args.backbone, args.img_size, fold, args.ref_seed, args.which)
    if ckpt6 is None or not ckpt6.is_file():
        ap.error(f"v6 (기준선) ckpt 를 찾지 못했다 — --v6-checkpoint 로 준다 (fold={fold}, seed={args.ref_seed})")
    m6, _ = load_model(ckpt6, device)
    img_size = round_to_patch(args.img_size, m6.patch_size)
    checkpoints = {RE: co._rel(ckpt_d), "v6": co._rel(ckpt6)}

    run7 = v7cfg.default_run_name(args.backbone, args.img_size, args.v7_variant, args.ref_seed)
    ckpt7 = v6cfg.run_ckpt_dir(run7, fold) / f"{args.which}.pt"
    m7, synth = None, "none"
    if ckpt7.is_file():
        m7, c7 = load_model(ckpt7, device)
        synth = ((c7.get("config", {}) or {}).get("synth") or {}).get("name") or "none"
        checkpoints["v7"] = co._rel(ckpt7)
    else:
        print(f"[ornl-dscnn] v7 ckpt 없음 ({ckpt7}) — 그 열을 뺀다")

    if args.v5_checkpoint:
        ckpt5 = Path(args.v5_checkpoint)
    else:                                                            # compare_gt · compare_ornl_v7 과 같은 규칙
        cands = [v6cfg.CHECKPOINT_DIR / f"{v6cfg.V5_RUN_NAME}-soot2sp" / "cv" / fold / f"{args.which}.pt",
                 v6cfg.CHECKPOINT_DIR / v6cfg.V5_RUN_NAME / "cv" / fold / f"{args.which}.pt"]
        ckpt5 = next((p for p in cands if p.is_file()), cands[-1])
    old, old_cfg = {}, {}
    for name, p in (("v4", Path(args.v4_checkpoint)), ("v5", ckpt5)):
        if p.exists():
            old[name], ck_old = load_model(p, device, n_classes=v6cfg.N_CLASSES_V4)
            old_cfg[name] = ck_old.get("config", {}) or {}
            checkpoints[name] = co._rel(p)
        else:
            print(f"[ornl-dscnn] {name} ckpt 없음 ({p}) — 그 열을 뺀다")
    models = ["dscnn", RE] + [m for m in ("v4", "v5") if m in old] + ["v6"] + (["v7"] if m7 is not None else [])
    lut = old_to_new_lut("se", device)
    palette = torch.tensor(list(v6cfg.DIVA_COLORS) + [v6cfg.UNLABELED_COLOR], dtype=torch.float32, device=device)

    builds = {}
    for b in args.builds:
        t0 = time.time()
        builds[b] = co.pick_layers(b, args.n_per_build, args.layer_seed)
        print(f"[ornl-dscnn] {b}: {builds[b]['n_layers']} 레이어 가운데 {builds[b]['layers']}  ({time.time() - t0:.0f}s)")
    items = [(b, li) for b, info in builds.items() for li in info["layers"]]
    n_layers = len(items)

    if args.out_dir:
        out_dir = Path(args.out_dir)
    else:                                                            # 기존 비교 (v7 실행의 폴더) 옆에 둔다. v7 이 없으면 DSCNN 실행의 폴더
        name = f"ornl_random{n_layers}_{args.which}_dscnn_reimpl_{args.tag or 'paperw'}"
        out_dir = (v6cfg.run_fig_dir(run7, fold) if m7 is not None else cfg.run_fig_dir(run_d, fold)) / name
    (out_dir / "pages").mkdir(parents=True, exist_ok=True)
    for p in sorted((out_dir / "pages").glob("page_*.png")):         # 이 스크립트가 앞서 만든 쪽만 지운다
        p.unlink()
    (out_dir / "layers.json").write_text(json.dumps({"seed": args.layer_seed, "builds": builds}, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[ornl-dscnn] {n_layers} 레이어 | 모델 {models} | DSCNN 재구현={checkpoints[RE]} | out={out_dir}")

    info = dict(co.MODEL_INFO)
    info["dscnn"] = ("DSCNN (ORNL)", "predictions stored in the ORNL HDF5 files", "dscnn")
    info[RE] = ("DSCNN (re-impl.)", f"our re-implementation, GT only, {weight_name}", RE)
    info["v6"] = ("DefSeg-AM v6", f"baseline, GT only ({v6cfg.run_tag(fold)})", "v6")
    info["v7"] = (f"DefSeg-AM v7-{args.v7_variant}", f"GT + generated layers ({synth})", "v7")
    if "v5" in old:
        soot = ", Soot → Spatter" if old_cfg["v5"].get("soot_mode") == "spatter" else ""
        info["v5"] = ("DefSeg-AM v5", f"GT only, 11 classes{soot} (cv/{ckpt5.parent.name})", "v5")
    columns = [("visible/0", "after melting"), ("visible/1", "after spreading")] + [info[m][:2] for m in models]
    co.PAGE_NOTE = (                                                 # 쪽 아래의 설명 — v6 모듈의 글을 이 그림에 맞게 바꾼다 (실행 중에만)
        f"No human ground truth exists for these builds — columns 3–{len(columns)} are all model predictions. The two DSCNN columns are different models: "
        "column 3 = the predictions ORNL shipped in the HDF5 files (their network, weights not public); "
        "column 4 = the DSCNN network of Scime et al. (2020) re-implemented here and trained from scratch on the human GT only.",
        "Old classes are shown with pbf-2026.11 ids: Swelling → SE, old Incomplete Spreading → IS, Misprint / Under Melting → no class. "
        f"Inputs: DSCNN re-impl. reads raw brightness at native resolution in {args.tile_size}-px tiles (mean taken from this build's sampled layers); "
        "v5 · v6 · v7 raw brightness at 1036²; v4 the per-layer stretched photos.",
        "No TTA, no post-processing. Photos are 3×3 averages, contrast-stretched for display only; each label pixel shows the most common non-powder class "
        "of its 3×3 block (Powder if under 3 of 9 pixels).",
    )

    loader = DataLoader(co.OrnlLayers(items), batch_size=None, shuffle=False, num_workers=args.workers,
                        collate_fn=co._identity, prefetch_factor=2 if args.workers > 0 else None)
    nm = len(models)
    counts = np.zeros((n_layers, nm, co.N_BINS), dtype=np.int64)
    cross_re = np.zeros((n_layers, nm, NC, co.N_BINS), dtype=np.int64)       # DSCNN 재구현 × 모델
    cross_v7 = np.zeros((n_layers, nm, NC, co.N_BINS), dtype=np.int64)       # v7 × 모델
    part = np.zeros((n_layers, nm, 2, co.N_BINS), dtype=np.int64)
    rs_rows = np.zeros((n_layers, nm, 3), dtype=np.float64)
    means: dict[str, float] = {}
    n_pages = sum(1 for info_b in builds.values() if info_b["layers"])
    buf, page_no, t0 = [], 0, time.time()
    for i, item in enumerate(loader):
        buf.append(item)
        b = item["build"]
        if i + 1 < n_layers and items[i + 1][0] == b:               # 한 빌드를 다 모은 뒤에 돈다 (ū 를 그 빌드의 사진에서 낸다)
            continue
        mean_b = photo_mean([x[k] for x in buf for k in ("raw0", "raw1")])
        means[b] = mean_b
        page_rows = []
        for j, it in enumerate(buf):
            idx = i + 1 - len(buf) + j
            li = int(it["layer"])
            raw0, raw1 = it["raw0"], it["raw1"]
            d = torch.from_numpy(it["dscnn"].astype(np.int64)).to(device)
            pred = {"dscnn": torch.where(d < 0, torch.full_like(d, NONE_ID), d)}
            pred[RE] = predict_photos(md, raw0, raw1, t=args.tile_size, mean=mean_b, device=device, key=f"{b}/{li}")
            if "v4" in old:
                pred["v4"] = lut[predict_pair(old["v4"], it["str0"], it["str1"], img_size, device, ("native",))["native"]]
            if "v5" in old:
                pred["v5"] = lut[predict_pair(old["v5"], raw0, raw1, img_size, device, ("native",))["native"]]
            pred["v6"] = predict_pair(m6, raw0, raw1, img_size, device, ("native",))["native"]
            if m7 is not None:
                pred["v7"] = predict_pair(m7, raw0, raw1, img_size, device, ("native",))["native"]
            on_part = torch.from_numpy(it["part"]).to(device).long()
            for k, m in enumerate(models):
                counts[idx, k] = torch.bincount(pred[m].reshape(-1), minlength=co.N_BINS).cpu().numpy()
                cross_re[idx, k] = co._bincount2(pred[RE], pred[m], NC, co.N_BINS)
                if m7 is not None:
                    cross_v7[idx, k] = co._bincount2(pred["v7"], pred[m], NC, co.N_BINS)
                part[idx, k] = co._bincount2(on_part, pred[m], 2, co.N_BINS)
                rs_rows[idx, k] = co.rs_row_stats(pred[m])
            tiles = [co.tile_gray(co.display_u8(raw0), device), co.tile_gray(co.display_u8(raw1), device)]
            tiles += [co.tile_label(pred[m], palette) for m in models]
            page_rows.append({"build": b, "label": f"{b} · layer {li}", "tiles": tiles})
            print(f"[ornl-dscnn] {idx + 1}/{n_layers}  {b} layer {li}  ū={mean_b:.1f}  ({time.time() - t0:.0f}s)")
        page_no += 1
        title = (f"ORNL {b} — random layers: DSCNN (ORNL) · DSCNN re-impl. · DefSeg-AM " + " · ".join(m for m in models if m.startswith("v"))
                 + f"    ({v6cfg.run_tag(fold)} {args.which})    page {page_no} / {n_pages}")
        co.render_page(page_rows, columns, title, out_dir / "pages" / f"page_{page_no:02d}_{b}.png")
        buf = []

    # ----- 수치 -----
    tot = counts.sum(axis=(0, 2))                                            # 모델마다의 화소 수
    share = counts.sum(axis=0) / tot[:, None]                                # (모델, 11)
    doc = {
        "note": "사람 GT 가 없는 사진이다 — 모델끼리의 비교이지 점수가 아니다. dscnn = ORNL 이 배포한 예측, dscnn_re = DSCNN 재구현 (사람 GT 만).",
        "n_layers": n_layers, "builds": {b: v["layers"] for b, v in builds.items()}, "layer_seed": args.layer_seed,
        "models": models, "checkpoints": checkpoints, "img_size": img_size,
        "dscnn_re": {"run": run_d, "fold": fold, "step": cd.get("step"), "class_weight": cd_cfg.get("class_weight"),
                     "tile_size": args.tile_size, "mean_by_build": means},
        "class_share": {m: {**{AB[c]: float(share[k, c]) for c in range(NC)}, "no_class": float(share[k, NONE_ID])} for k, m in enumerate(models)},
        "defect_share": {m: float(share[k, 2:NC].sum()) for k, m in enumerate(models)},
        "cad": {m: co.part_overlap(part[:, k].sum(axis=0)) for k, m in enumerate(models)},
        "rs": {m: co.rs_summary(rs_rows[:, k]) for k, m in enumerate(models)},
        "dscnn_re_vs": {m: co.agreement(cross_re[:, k].sum(axis=0)) for k, m in enumerate(models) if m != RE},
        "by_build": {},
    }
    if m7 is not None:
        doc["v7"] = {"variant": args.v7_variant, "seed": args.ref_seed, "fold": fold, "synth_set": synth}
        doc["v7_vs"] = {m: co.agreement(cross_v7[:, k].sum(axis=0)) for k, m in enumerate(models) if m != "v7"}
    bi = np.array([b for b, _ in items])
    for b in builds:
        s = counts[bi == b].sum(axis=0)
        s = s / s.sum(axis=1, keepdims=True)
        doc["by_build"][b] = {m: {AB[c]: float(s[k, c]) for c in range(2, NC)} for k, m in enumerate(models)}
    (out_dir / "summary.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")

    pct = lambda v: "–" if v is None else (f"{100 * v:.2f}" if v >= 0.0005 else ("0" if v == 0 else "<0.05"))
    f3 = lambda v: "–" if v is None else f"{v:.3f}"
    names = {"dscnn": "DSCNN (ORNL 배포 예측)", RE: "DSCNN 재구현", "v4": "v4", "v5": "v5", "v6": "v6 (기준선)", "v7": f"v7-{args.v7_variant}"}
    L = [f"# ORNL 무작위 {n_layers} 레이어 — DSCNN (ORNL) · DSCNN 재구현 · " + " · ".join(m for m in models if m.startswith("v")) + " 의 예측 비교", "",
         "사람 GT 가 없는 사진이다. 아래는 **모델끼리의 비교**이지 점수가 아니다.", "",
         "\"DSCNN\" 이 둘이다. **DSCNN (ORNL)** 은 ORNL 이 HDF5 에 넣어 배포한 예측이고 (가중치 비공개), **DSCNN 재구현** 은 같은 논문의 구조를 다시 짜서 사람 GT 만으로 학습한 모델이다.", "",
         f"- DSCNN 재구현 = `{checkpoints[RE]}` ({weight_name}, step {cd.get('step')}). 원본 밝기 · 원본 해상도, 타일 {args.tile_size}, "
         "ū 는 빌드마다 뽑은 레이어의 사진 평균 — " + ", ".join(f"{b} {v:.1f}" for b, v in means.items()),
         f"- v6 = `{checkpoints['v6']}` — 같은 fold 의 기준선 (사람 GT 만)"]
    if m7 is not None:
        L.append(f"- v7 = `{checkpoints['v7']}` (꼬리표 {args.v7_variant}, seed {args.ref_seed}, 생성 세트 {synth})")
    for m in ("v5", "v4"):
        if m in checkpoints:
            L.append(f"- {m} = `{checkpoints[m]}`")
    L += [f"- 레이어: 빌드마다 {args.n_per_build} 장 (seed {args.layer_seed}) — " + ", ".join(f"{b} {v['layers']}" for b, v in builds.items()), "",
          "## 클래스별 예측 화소 비율 (%)", "",
          "| 모델 | " + " | ".join(AB) + " | 대응 없음 | 결함 합 |", "|---|" + "---:|" * (NC + 2)]
    for k, m in enumerate(models):
        L.append(f"| {names[m]} | " + " | ".join(pct(share[k, c]) for c in range(NC)) + f" | {pct(share[k, NONE_ID])} | {pct(share[k, 2:NC].sum())} |")
    per_build = [m for m in ("dscnn", RE, "v6") if m in models]
    L += ["", "## 빌드별 — 결함 클래스 비율 (%)", "", "| 빌드 | 모델 | " + " | ".join(AB[2:]) + " |", "|---|---|" + "---:|" * (NC - 2)]
    for b in builds:
        for m in per_build:
            L.append(f"| {b} | {names[m]} | " + " | ".join(pct(doc['by_build'][b][m][AB[c]]) for c in range(2, NC)) + " |")
    L += ["", "## CAD 파트 마스크와의 겹침 · RS 선", "",
          "| 모델 | 파트 IoU | 파트 밖을 Printed 로 낸 비율 (%) | RS 가 있는 행에서 덮은 폭 (중앙값) | RS 행 수 (레이어 중앙값) | 끝까지 이은 행 |", "|---|---:|---:|---:|---:|---:|"]
    for m in models:
        c, r = doc["cad"][m], doc["rs"][m]
        L.append(f"| {names[m]} | {f3(c['part_iou'])} | {pct(c['off_part_printed_rate'])} | {f3(r['row_cover_median'])} | {r['rows_median']:.0f} | {r['full_width_rows']} |")

    def agree_table(title: str, key: str, ref: str):
        rows = ["", title, "", "| 견준 모델 | 화소 일치 | " + " | ".join(AB) + " |", "|---|---:|" + "---:|" * NC]
        for m in models:
            if m == ref:
                continue
            a = doc[key][m]
            rows.append(f"| {names[m]} | {a['pixel_agreement']:.4f} | " + " | ".join(f3(v) for v in a["iou"]) + " |")
        return rows

    L += agree_table("## DSCNN 재구현과 다른 모델의 일치 (화소 비율, 클래스별은 두 예측 사이의 IoU)", "dscnn_re_vs", RE)
    if m7 is not None:
        L += agree_table("## v7 과 다른 모델의 일치", "v7_vs", "v7")
    L += ["", f"그림: `pages/page_XX_<빌드>.png` — 한 줄이 한 레이어, 열은 용융 직후 · 도포 직후 사진과 {nm} 모델의 예측이다."]
    (out_dir / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"[ornl-dscnn] 끝 → {out_dir}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
