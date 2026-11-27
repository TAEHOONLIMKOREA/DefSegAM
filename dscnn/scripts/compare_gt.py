"""사람 GT 의 검증 레이어에서 **DSCNN 재구현** 과 DefSeg-AM v4 · v5 · v6 · v7 의 예측을 정답과 나란히 본다.

    열:  visible/0 | visible/1 | 사람 GT | DSCNN (재구현, GT 만) | v4 | v5 | v6 | v7

`v7/scripts/compare_gt_v7.py` 에 DSCNN 의 열을 더한 것이다 (그림을 그리는 함수는 그쪽 것을 불러 쓴다).
여기의 DSCNN 은 ORNL 이 배포한 예측이 아니라, 논문의 구조를 다시 짜서 **같은 사람 GT · 같은 fold** 로 학습한 모델이다 (dscnn/training/train.py).

  - 레이어: DSCNN 실행의 fold 가 떼어 둔 검증 레이어 (inlayer 면 22 장). DSCNN · v5 · v6 · v7 은 같은 fold 로 배워 이 레이어를 보지 않았다. **v4 는 학습에 썼다.**
  - 그림은 모두 1036² 로 맞춘다 (소스마다 원본 크기가 다르다). DSCNN 의 예측은 원본 해상도에서 낸 것을 NEAREST 로 옮긴다.
  - 점수는 두 해상도로 낸다: 원본 해상도 (주 지표 — DefSeg-AM 은 logits 를 원본 크기로 올려 argmax, DSCNN 은 원래 원본 해상도) 와 1036².

출력 (figures/<dscnn run>/cv/<fold>/gt_val<N>_<which>/):
    pages/page_XX.png            비교 그림 — 한 줄이 한 레이어
    summary.md · summary.json    모델마다 클래스별 IoU (두 해상도), 놓침 · 잘못 짚음, 레이어별 found

사용 (CWD = 3DP_VPPM/. compose 로는 DSCNN_PHASE=compare):
    python -m DefSeg_AM.dscnn.scripts.compare_gt --fold inlayer
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch

from ...common.data.image_utils import resize_image_uint8, resize_label
from ...common.models.model import round_to_patch
from ...v6 import config_v6 as v6cfg
from ...v6.data.data_gt_v6 import enumerate_samples_v6, load_sample_native
from ...v6.inference import compare_ornl as co
from ...v6.inference.baselines import NONE_ID, old_to_new_lut
from ...v6.inference.evaluate import load_model, predict_pair, select_samples
from ...v7 import config_v7 as v7cfg
from ...v7.scripts.compare_gt_v7 import render_page, scores, stamp
from ...v7.scripts.compare_ornl_v7 import baseline_ckpt

from .. import config_dscnn as cfg
from ..data.tiles import make_layer
from ..inference.evaluate import resolve_mean_and_tile
from ..inference.predict import load_dscnn, predict_layer

AB = v6cfg.CLASS_ABBRS_V6
NC = v6cfg.N_CLASSES_V6


def main():
    ap = argparse.ArgumentParser(description="사람 GT 의 검증 레이어에서 DSCNN 재구현 · v4 · v5 · v6 · v7 을 정답과 나란히")
    ap.add_argument("--fold", type=str, default="inlayer", choices=list(v6cfg.CV_FOLDS))
    ap.add_argument("--seed", type=int, default=cfg.RANDOM_SEED, help="DSCNN 실행의 seed")
    ap.add_argument("--tag", type=str, default="", help="DSCNN 실행의 꼬리표")
    ap.add_argument("--which", type=str, default="last", choices=["last", "best"])
    ap.add_argument("--dscnn-checkpoint", type=str, default=None)
    ap.add_argument("--backbone", type=str, default=v6cfg.DINO_BACKBONE)
    ap.add_argument("--img-size", type=int, default=v6cfg.IMG_SIZE)
    ap.add_argument("--ref-seed", type=int, default=v6cfg.RANDOM_SEED, help="v6 (기준선) · v7 실행의 seed")
    ap.add_argument("--v7-variant", type=str, default="W", help="v7 실행의 꼬리표. 그 실행이 없으면 v7 의 열을 뺀다")
    ap.add_argument("--v6-checkpoint", type=str, default=None)
    ap.add_argument("--v5-checkpoint", type=str, default=None, help="기본: 같은 fold 의 v5 — Soot 를 Spatter 로 배운 실행이 있으면 그것")
    ap.add_argument("--v4-checkpoint", type=str, default=str(v6cfg.CHECKPOINT_DIR / v6cfg.V4_RUN_NAME / "stage2_best.pt"))
    ap.add_argument("--rows-per-page", type=int, default=5)
    ap.add_argument("--max-layers", type=int, default=None, help="스모크용: 앞 N 레이어만")
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_d = cfg.default_run_name(args.seed, args.tag)
    ckpt_d = Path(args.dscnn_checkpoint) if args.dscnn_checkpoint else cfg.run_ckpt_dir(run_d, args.fold) / f"{args.which}.pt"
    if not ckpt_d.is_file():
        ap.error(f"DSCNN ckpt 없음: {ckpt_d} — DSCNN_PHASE=cv 로 먼저 학습한다")
    dscnn, ck_d = load_dscnn(ckpt_d, device)
    dcfg = ck_d.get("config", {}) or {}

    ckpt6 = Path(args.v6_checkpoint) if args.v6_checkpoint else baseline_ckpt(args.backbone, args.img_size, args.fold, args.ref_seed, args.which)
    ckpt7 = v6cfg.run_ckpt_dir(v7cfg.default_run_name(args.backbone, args.img_size, args.v7_variant, args.ref_seed), args.fold) / f"{args.which}.pt"
    if args.v5_checkpoint:
        ckpt5 = Path(args.v5_checkpoint)
    else:
        cands = [v6cfg.CHECKPOINT_DIR / f"{v6cfg.V5_RUN_NAME}-soot2sp" / "cv" / args.fold / f"{args.which}.pt",
                 v6cfg.CHECKPOINT_DIR / v6cfg.V5_RUN_NAME / "cv" / args.fold / f"{args.which}.pt"]
        ckpt5 = next((p for p in cands if p.is_file()), cands[-1])

    nets, net_cfg, checkpoints = {}, {}, {"dscnn": co._rel(ckpt_d)}
    for name, p, ncls in (("v4", Path(args.v4_checkpoint), v6cfg.N_CLASSES_V4), ("v5", ckpt5, v6cfg.N_CLASSES_V4),
                          ("v6", ckpt6, NC), ("v7", ckpt7, NC)):
        if p is not None and p.is_file():
            nets[name], ck = load_model(p, device, n_classes=ncls)
            net_cfg[name] = ck.get("config", {}) or {}
            checkpoints[name] = co._rel(p)
        else:
            print(f"[gt-dscnn] {name} ckpt 없음 ({p}) — 그 열을 뺀다")
    models = ["dscnn"] + [m for m in ("v4", "v5", "v6", "v7") if m in nets]
    img_size = round_to_patch(args.img_size, next(iter(nets.values())).patch_size) if nets else args.img_size
    lut = old_to_new_lut("se", device)
    palette = torch.tensor(list(v6cfg.DIVA_COLORS) + [v6cfg.UNLABELED_COLOR], dtype=torch.float32, device=device)

    fig_dir = cfg.run_fig_dir(run_d, args.fold)
    samples, _, _ = select_samples(enumerate_samples_v6(), args.fold, None, fold_run_json=fig_dir / "run.json", img_size=img_size)
    if args.max_layers is not None:
        samples = samples[: args.max_layers]
    n = len(samples)
    out_dir = Path(args.out_dir) if args.out_dir else fig_dir / f"gt_val{n}_{args.which}"
    (out_dir / "pages").mkdir(parents=True, exist_ok=True)
    for p in sorted((out_dir / "pages").glob("page_*.png")):
        p.unlink()
    print(f"[gt-dscnn] {n} 레이어 (fold {args.fold} 의 검증) | 모델 {models} | out={out_dir}")

    soot = ", Soot → Spatter" if net_cfg.get("v5", {}).get("soot_mode") == "spatter" else ""
    gen_set = ((net_cfg.get("v7", {}) or {}).get("synth") or {}).get("name") or "none"
    info = {
        "dscnn": ("DSCNN (re-impl.)", f"Scime 2020 network, GT only (cv/{args.fold})"),
        "v4": ("DefSeg-AM v4", "serving model — TRAINED ON THESE LAYERS"),
        "v5": ("DefSeg-AM v5", f"GT only, 11 classes{soot} (cv/{args.fold})"),
        "v6": ("DefSeg-AM v6", f"GT only (cv/{args.fold})"),
        "v7": (f"DefSeg-AM v7-{args.v7_variant}", f"GT + generated layers ({gen_set})"),
    }
    columns = [("visible/0", "after melting"), ("visible/1", "after spreading"), ("Human GT", "pbf-2026.11 labels")] + [info[m] for m in models]
    notes = (
        f"Validation layers of fold '{args.fold}': DSCNN (re-implemented) · v5 · v6 · v7 were trained on the same human-GT layers and never saw these; v4 did (reference only). "
        f"All tiles shown at {img_size}².",
        "DSCNN here is NOT ORNL's released model: the network of Scime et al. (2020) re-implemented and trained from scratch on the human GT only, predicting at native resolution.",
        "Numbers on each prediction: found = share of GT defect pixels given the right class; false alarm = share of GT Powder / Printed pixels predicted as a defect. No TTA, no post-processing.",
    )

    cms = {r: {m: np.zeros((NC, co.N_BINS), dtype=np.int64) for m in models} for r in ("native", "resized")}
    per_layer, page_rows, page_no, t0 = [], [], 0, time.time()
    n_pages = int(np.ceil(n / args.rows_per_page))
    cache: dict[str, tuple[float, int]] = {}
    for i, s in enumerate(samples):
        i0, i1, ann = load_sample_native(s)
        gt_n = torch.from_numpy(ann.astype(np.int64)).to(device)
        gt_r = torch.from_numpy(resize_label(ann, img_size).astype(np.int64)).to(device)
        if s.group not in cache:
            def photos(_g=s.group):
                out = []
                for x in samples:
                    if x.group == _g:
                        a, b, _ = load_sample_native(x)
                        out += [a, b]
                return out
            cache[s.group] = resolve_mean_and_tile(s.group, dcfg, photos)
        mean, t = cache[s.group]
        pred_n, pred_r = {}, {}
        pd = predict_layer(dscnn, make_layer(i0, i1, None, key=s.key, group=s.group, t=t, mean=mean, device=device))
        pred_n["dscnn"] = pd
        pred_r["dscnn"] = torch.from_numpy(resize_label(pd.cpu().numpy(), img_size).astype(np.int64)).to(device)
        for m in models[1:]:
            both = predict_pair(nets[m], i0, i1, img_size, device, ("native", "resized"))
            old = m in ("v4", "v5")
            pred_n[m] = lut[both["native"]] if old else both["native"]
            pred_r[m] = lut[both["resized"]] if old else both["resized"]

        r0, r1 = resize_image_uint8(i0, img_size), resize_image_uint8(i1, img_size)
        valid_r = gt_r >= 0
        gt_show = torch.where(valid_r, gt_r, torch.full_like(gt_r, NONE_ID))
        tiles = [co.tile_gray(co.display_u8(r0), device), co.tile_gray(co.display_u8(r1), device), co.tile_label(gt_show, palette)]
        rec = {"key": s.key, "classes": [AB[c] for c in range(2, NC) if bool((gt_r == c).any())]}
        valid_n = gt_n >= 0
        for m in models:
            cms["native"][m] += co._bincount2(gt_n[valid_n], pred_n[m][valid_n], NC, co.N_BINS)
            cm = co._bincount2(gt_r[valid_r], pred_r[m][valid_r], NC, co.N_BINS)
            cms["resized"][m] += cm
            sc = scores(cm)
            rec[m] = {"found": sc["found"], "false_alarm": sc["false_alarm"]}
            txt = (f"found {100 * sc['found']:.0f}%" if sc["found"] is not None else "no GT defect") + f" · false alarm {100 * (sc['false_alarm'] or 0):.1f}%"
            tiles.append(stamp(co.tile_label(pred_r[m], palette), txt))
        per_layer.append(rec)
        page_rows.append({"label": f"{v6cfg.SOURCE_SHORT.get(s.group, s.group)} · {s.layer}", "tiles": tiles})
        if len(page_rows) == args.rows_per_page or i + 1 == n:
            page_no += 1
            title = (f"Human-GT validation layers ({args.fold}) — ground truth vs DSCNN (re-implemented, GT only) and DefSeg-AM "
                     f"{' · '.join(m for m in models[1:])}    page {page_no} / {n_pages}")
            render_page(page_rows, columns, title, notes, out_dir / "pages" / f"page_{page_no:02d}.png")
            page_rows = []
        print(f"[gt-dscnn] {i + 1}/{n}  {s.key}  ({time.time() - t0:.0f}s)")
        del pred_n, pred_r, pd

    res = {r: {m: scores(cms[r][m]) for m in models} for r in cms}
    doc = {"n_layers": n, "fold": args.fold, "img_size": img_size, "models": models, "checkpoints": checkpoints,
           "dscnn": {"run": run_d, "step": ck_d.get("step"), "weights": ck_d.get("weights"), "tile_size": dcfg.get("tile_size"),
                     "epochs": dcfg.get("epochs"), "total_steps": dcfg.get("total_steps"), "parameters": dcfg.get("parameters")},
           "v5_soot_mode": net_cfg.get("v5", {}).get("soot_mode", "ignore"),
           "scores": res, "per_layer": per_layer,
           "note": "native = 원본 해상도 (주 지표), resized = 1036². v4 는 이 레이어를 학습에 썼다. v5 는 PS 를 낼 수 없다 (클래스 체계에 없다)."}
    (out_dir / "summary.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")

    f3 = lambda v: "–" if v is None else f"{v:.3f}"
    pc = lambda v: "–" if v is None else f"{100 * v:.1f}%"
    names = {"dscnn": "DSCNN 재구현 (GT 만)", "v4": "v4 (이 레이어로 학습함)", "v5": "v5" + (" (Soot → Spatter)" if soot else ""),
             "v6": "v6", "v7": f"v7-{args.v7_variant}"}
    L = [f"# 사람 GT 검증 {n} 레이어 — DSCNN 재구현과 DefSeg-AM 의 비교 ({args.fold})", "",
         "DSCNN 재구현 = Scime et al. (2020) 의 구조를 다시 짜서 **같은 사람 GT · 같은 fold** 로 무작위 초기값에서 학습한 모델 (ORNL 이 배포한 DSCNN 의 예측이 아니다).",
         "DSCNN · v5 · v6 · v7 은 이 레이어를 학습에서 보지 않았고, **v4 는 학습에 썼다** (참고). TTA · 후처리 없음.", "",
         *[f"- {names[m]} = `{checkpoints[m]}`" for m in models], ""]
    for r, head in (("native", "원본 해상도 (주 지표)"), ("resized", f"{img_size}²")):
        L += [f"## 클래스별 IoU — {head}", "", "| 모델 | " + " | ".join(AB) + " | 결함 mIoU |", "|---|" + "---:|" * (NC + 1)]
        for m in models:
            L.append(f"| {names[m]} | " + " | ".join(f3(v) for v in res[r][m]["iou"]) + f" | {f3(res[r][m]['defect_miou'])} |")
        L += ["", f"## 결함을 찾은 정도 — {head}", "", "| 모델 | 맞힘 (found) | 정상으로 냄 (놓침) | 잘못 짚음 (정상 → 결함) | 화소 정확도 |", "|---|---:|---:|---:|---:|"]
        for m in models:
            L.append(f"| {names[m]} | {pc(res[r][m]['found'])} | {pc(res[r][m]['miss'])} | {pc(res[r][m]['false_alarm'])} | {pc(res[r][m]['pixel_acc'])} |")
        L.append("")
    L += ["맞힘 = 정답이 결함인 화소 가운데 같은 클래스로 낸 비율. v5 는 PS 를 IS 로 내므로 (PS 가 클래스 체계에 없다) 맞힘이 낮게 나온다.", "",
          f"## 레이어별 — 결함을 맞힌 비율 ({img_size}²)", "", "| 레이어 | 정답에 있는 결함 | " + " | ".join(names[m] for m in models) + " |", "|---|---|" + "---:|" * len(models)]
    for r in per_layer:
        L.append(f"| {r['key']} | {' · '.join(r['classes']) or '없음'} | " + " | ".join(pc(r[m]["found"]) for m in models) + " |")
    L += ["", "그림: `pages/page_XX.png` — 한 줄이 한 레이어, 열은 사진 두 장 · 정답 · 모델들의 예측이다."]
    (out_dir / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:34]))
    print(f"[gt-dscnn] 끝 → {out_dir}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
