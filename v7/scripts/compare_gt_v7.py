"""사람 GT 의 검증 레이어에서 v4 · v5 · v6 · v7 의 예측을 **정답과 나란히** 본다.

    열:  visible/0 | visible/1 | 사람 GT | v4 | v5 | v6 | v7

`compare_ornl_v7.py` 의 짝이다. 그쪽은 정답이 없는 ORNL 사진에서 모델끼리 견주고, 이쪽은 정답이 있는 레이어에서 견준다.
DSCNN 의 열은 없다 — DSCNN 의 예측은 ORNL 빌드의 HDF5 에만 들어 있고, 사람 GT 데이터셋에는 주석 (정답) 만 있다.

  - 레이어: v7 실행의 fold 가 떼어 둔 검증 레이어 전부 (inlayer 면 22 장). v5 · v6 · v7 은 같은 fold 의 모델이라 이 레이어를 학습에서 보지 않았다.
    **v4 는 이 레이어를 학습에 썼다** — 참고로만 본다.
  - v6 = 같은 fold · 같은 seed 의 기준선 (사람 GT 만). v7 과의 차이는 생성 모델이 그린 레이어를 함께 배웠는가 하나다.
  - v5 = 같은 fold 의 모델. Soot 를 Spatter 로 배운 실행 (`…_11cls_v5-soot2sp`) 이 있으면 그것을, 없으면 기존 v5 를 쓴다 (`--v5-checkpoint` 로 바꾼다).
  - 모두 1036² 에서 견준다 (모델의 입력 크기. 소스마다 원본 크기가 달라 한 그림에 놓으려는 것이다). 옛 11-class 예측은 새 id 로 옮긴다.
  - 예측 칸의 왼쪽 위 숫자: found = 정답이 결함인 화소 가운데 같은 클래스로 맞힌 비율, false alarm = 정답이 Powder · Printed 인 화소 가운데 결함으로 낸 비율.

출력 (figures/<v7 run>/v6/cv/<fold>/gt_val<N>_<which>/):
    pages/page_XX.png            비교 그림 — 한 줄이 한 레이어
    summary.md · summary.json    모델마다 클래스별 IoU (1036²), 놓침 · 잘못 짚음, 레이어별 found

사용 (CWD = 3DP_VPPM/. compose 로는 DEFSEG_V7_PHASE=gt_compare):
    python -m DefSeg_AM.v7.scripts.compare_gt_v7                         # W · seed 42 · inlayer
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw

from ...common.data.image_utils import resize_image_uint8, resize_label
from ...common.models.model import round_to_patch
from ...v6 import config_v6 as v6cfg
from ...v6.data.data_gt_v6 import enumerate_samples_v6, load_sample_native
from ...v6.inference import compare_ornl as co
from ...v6.inference.baselines import NONE_ID, old_to_new_lut
from ...v6.inference.evaluate import load_model, predict_pair, select_samples
from .. import config_v7 as cfg
from .compare_ornl_v7 import baseline_ckpt

AB = v6cfg.CLASS_ABBRS_V6
NC = v6cfg.N_CLASSES_V6
INK, INK2 = (11, 11, 11), (82, 81, 78)


def render_page(rows: list[dict], columns: list[tuple[str, str]], title: str, notes: tuple[str, ...], out_path: Path) -> None:
    """rows = [{"label": str, "tiles": [uint8 (T, T) | (T, T, 3), …]}] → PNG 한 장. (v6 의 compare_ornl.render_page 와 같은 배치, 범례의 글만 다르다.)"""
    t = rows[0]["tiles"][0].shape[0]
    n_col = len(columns)
    gap, left, right = 8, 62, 16
    title_h, head_h, legend_h, note_h = 62, 74, 100, 26 * len(notes) + 22
    w = left + n_col * t + (n_col - 1) * gap + right
    h = title_h + head_h + len(rows) * t + (len(rows) - 1) * gap + legend_h + note_h
    page = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(page)
    d.text((left, 12), title, font=co._font(30, True), fill=INK)
    f_main, f_sub = co._font(26, True), co._font(18)
    for c, (main, sub) in enumerate(columns):
        cx = left + c * (t + gap) + t // 2
        d.text((cx, title_h + 20), main, font=f_main, fill=INK, anchor="mm")
        d.text((cx, title_h + 52), sub, font=f_sub, fill=INK2, anchor="mm")
    f_row = co._font(26, True)
    for r, row in enumerate(rows):
        y = title_h + head_h + r * (t + gap)
        for c, tile in enumerate(row["tiles"]):
            page.paste(Image.fromarray(tile).convert("RGB"), (left + c * (t + gap), y))
        tmp = Image.new("RGB", (t, 44), (255, 255, 255))
        ImageDraw.Draw(tmp).text((t // 2, 22), row["label"], font=f_row, fill=INK, anchor="mm")
        page.paste(tmp.rotate(90, expand=True), (8, y))
    entries = [(v6cfg.DIVA_COLORS[c], f"{AB[c]}  {v6cfg.CLASS_NAMES_V6[c]}") for c in range(NC)]
    entries.append((v6cfg.UNLABELED_COLOR, "unlabeled (GT)  /  no class (old model)"))
    f_leg = co._font(22)
    y0 = title_h + head_h + len(rows) * t + (len(rows) - 1) * gap + 16
    per_row = 6
    cell = (w - left - right) // per_row
    for i, (color, name) in enumerate(entries):
        x = left + (i % per_row) * cell
        y = y0 + (i // per_row) * 38
        d.rectangle([x, y, x + 26, y + 26], fill=tuple(color), outline=(120, 120, 120))
        d.text((x + 36, y + 13), name, font=f_leg, fill=INK, anchor="lm")
    f_note = co._font(19)
    for i, line in enumerate(notes):
        d.text((left, y0 + 84 + i * 26), line, font=f_note, fill=INK2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    page.save(out_path)


def stamp(tile: np.ndarray, text: str) -> np.ndarray:
    """예측 칸의 왼쪽 위에 숫자를 적는다 (흰 바탕)."""
    im = Image.fromarray(tile).convert("RGB")
    d = ImageDraw.Draw(im)
    f = co._font(20, True)
    box = d.textbbox((8, 6), text, font=f)
    d.rectangle([box[0] - 5, box[1] - 4, box[2] + 5, box[3] + 4], fill=(255, 255, 255), outline=(150, 150, 150))
    d.text((8, 6), text, font=f, fill=INK)
    return np.asarray(im)


def scores(cm: np.ndarray) -> dict:
    """cm[정답 (10), 예측 (11)] → 클래스별 IoU · 재현율 · 정밀도, 결함 mIoU, 놓침 · 잘못 짚음."""
    row, col = cm.sum(axis=1), cm.sum(axis=0)[:NC]
    iou, rec, pre = [], [], []
    for c in range(NC):
        union = row[c] + col[c] - cm[c, c]
        iou.append(float(cm[c, c] / union) if row[c] > 0 and union > 0 else None)
        rec.append(float(cm[c, c] / row[c]) if row[c] > 0 else None)
        pre.append(float(cm[c, c] / col[c]) if col[c] > 0 else None)
    d_px, n_px = float(cm[2:].sum()), float(cm[:2].sum())
    have = [iou[c] for c in range(2, NC) if iou[c] is not None]
    return {
        "iou": iou, "recall": rec, "precision": pre, "support_px": [int(v) for v in row],
        "defect_miou": float(np.mean(have)) if have else None,
        "found": float(sum(cm[c, c] for c in range(2, NC)) / d_px) if d_px else None,
        "miss": float(cm[2:, :2].sum() / d_px) if d_px else None,
        "false_alarm": float(cm[:2, 2:].sum() / n_px) if n_px else None,
        "pixel_acc": float(np.trace(cm[:, :NC]) / cm.sum()),
    }


def main():
    ap = argparse.ArgumentParser(description="사람 GT 의 검증 레이어에서 v4 · v5 · v6 · v7 예측을 정답과 나란히")
    ap.add_argument("--variant", type=str, default="W", help="v7 실행의 꼬리표")
    ap.add_argument("--seed", type=int, default=v6cfg.RANDOM_SEED)
    ap.add_argument("--fold", type=str, default="inlayer", choices=list(v6cfg.CV_FOLDS))
    ap.add_argument("--which", type=str, default="last", choices=["last", "best"])
    ap.add_argument("--backbone", type=str, default=v6cfg.DINO_BACKBONE)
    ap.add_argument("--img-size", type=int, default=v6cfg.IMG_SIZE)
    ap.add_argument("--v6-checkpoint", type=str, default=None, help="기본: 같은 fold · 같은 seed 의 기준선")
    ap.add_argument("--v5-checkpoint", type=str, default=None, help="기본: 같은 fold 의 v5 — Soot 를 Spatter 로 배운 실행이 있으면 그것")
    ap.add_argument("--v4-checkpoint", type=str, default=str(v6cfg.CHECKPOINT_DIR / v6cfg.V4_RUN_NAME / "stage2_best.pt"))
    ap.add_argument("--rows-per-page", type=int, default=5)
    ap.add_argument("--max-layers", type=int, default=None, help="스모크용: 앞 N 레이어만")
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run7 = cfg.default_run_name(args.backbone, args.img_size, args.variant, args.seed)
    ckpt7 = v6cfg.run_ckpt_dir(run7, args.fold) / f"{args.which}.pt"
    if not ckpt7.is_file():
        ap.error(f"v7 ckpt 없음: {ckpt7}")
    ckpt6 = Path(args.v6_checkpoint) if args.v6_checkpoint else baseline_ckpt(args.backbone, args.img_size, args.fold, args.seed, args.which)
    if ckpt6 is None or not ckpt6.is_file():
        ap.error("v6 (기준선) ckpt 를 찾지 못했다 — --v6-checkpoint 로 준다")
    if args.v5_checkpoint:
        ckpt5 = Path(args.v5_checkpoint)
    else:
        cands = [v6cfg.CHECKPOINT_DIR / f"{v6cfg.V5_RUN_NAME}-soot2sp" / "cv" / args.fold / f"{args.which}.pt",
                 v6cfg.CHECKPOINT_DIR / v6cfg.V5_RUN_NAME / "cv" / args.fold / f"{args.which}.pt"]
        ckpt5 = next((p for p in cands if p.is_file()), cands[-1])

    m7, c7 = load_model(ckpt7, device)
    m6, _ = load_model(ckpt6, device)
    img_size = round_to_patch(args.img_size, m7.patch_size)
    old, old_cfg, checkpoints = {}, {}, {"v7": co._rel(ckpt7), "v6": co._rel(ckpt6)}
    for name, p in (("v4", Path(args.v4_checkpoint)), ("v5", ckpt5)):
        if p.is_file():
            old[name], ck = load_model(p, device, n_classes=v6cfg.N_CLASSES_V4)
            old_cfg[name] = ck.get("config", {}) or {}
            checkpoints[name] = co._rel(p)
        else:
            print(f"[gt7] {name} ckpt 없음 ({p}) — 그 열을 뺀다")
    models = [m for m in ("v4", "v5") if m in old] + ["v6", "v7"]
    lut = old_to_new_lut("se", device)
    palette = torch.tensor(list(v6cfg.DIVA_COLORS) + [v6cfg.UNLABELED_COLOR], dtype=torch.float32, device=device)
    gen_set = ((c7.get("config", {}) or {}).get("synth") or {}).get("name") or "none"

    fig_dir = v6cfg.run_fig_dir(run7, args.fold)
    samples, _, _ = select_samples(enumerate_samples_v6(), args.fold, None, fold_run_json=fig_dir / "run.json", img_size=img_size)
    if args.max_layers is not None:
        samples = samples[: args.max_layers]
    n = len(samples)
    out_dir = Path(args.out_dir) if args.out_dir else fig_dir / f"gt_val{n}_{args.which}"
    (out_dir / "pages").mkdir(parents=True, exist_ok=True)
    for p in sorted((out_dir / "pages").glob("page_*.png")):
        p.unlink()
    print(f"[gt7] {n} 레이어 (fold {args.fold} 의 검증) | 모델 {models} | {checkpoints} | out={out_dir}")

    soot = ", Soot → Spatter" if old_cfg.get("v5", {}).get("soot_mode") == "spatter" else ""
    info = {
        "v4": ("DefSeg-AM v4", "serving model — TRAINED ON THESE LAYERS"),
        "v5": ("DefSeg-AM v5", f"GT only, 11 classes{soot} (cv/{args.fold})"),
        "v6": ("DefSeg-AM v6", f"baseline, GT only (cv/{args.fold})"),
        "v7": (f"DefSeg-AM v7-{args.variant}", f"GT + generated layers ({gen_set})"),
    }
    columns = [("visible/0", "after melting"), ("visible/1", "after spreading"), ("Human GT", "pbf-2026.11 labels")] + [info[m] for m in models]
    notes = (
        f"Validation layers of fold '{args.fold}': v5 · v6 · v7 never saw them in training; v4 did (reference only). All tiles at {img_size}² (the model input size), "
        "old 11-class predictions shown with pbf-2026.11 ids.",
        "Numbers on each prediction: found = share of GT defect pixels given the right class; false alarm = share of GT Powder / Printed pixels predicted as a defect. "
        "No TTA, no post-processing. Photos are contrast-stretched for display only.",
    )

    cms = {m: np.zeros((NC, co.N_BINS), dtype=np.int64) for m in models}
    per_layer, page_rows, page_no, t0 = [], [], 0, time.time()
    n_pages = int(np.ceil(n / args.rows_per_page))
    for i, s in enumerate(samples):
        i0, i1, ann = load_sample_native(s)
        gt = torch.from_numpy(resize_label(ann, img_size).astype(np.int64)).to(device)          # -1 = 미레이블
        valid = gt >= 0
        pred = {}
        if "v4" in old:
            pred["v4"] = lut[predict_pair(old["v4"], i0, i1, img_size, device, ("resized",))["resized"]]
        if "v5" in old:
            pred["v5"] = lut[predict_pair(old["v5"], i0, i1, img_size, device, ("resized",))["resized"]]
        pred["v6"] = predict_pair(m6, i0, i1, img_size, device, ("resized",))["resized"]
        pred["v7"] = predict_pair(m7, i0, i1, img_size, device, ("resized",))["resized"]

        r0, r1 = resize_image_uint8(i0, img_size), resize_image_uint8(i1, img_size)
        gt_show = torch.where(valid, gt, torch.full_like(gt, NONE_ID))
        tiles = [co.tile_gray(co.display_u8(r0), device), co.tile_gray(co.display_u8(r1), device), co.tile_label(gt_show, palette)]
        rec = {"key": s.key, "defect_px": int((gt >= 2).sum()), "classes": [AB[c] for c in range(2, NC) if bool((gt == c).any())]}
        for m in models:
            cm = co._bincount2(gt[valid], pred[m][valid], NC, co.N_BINS)
            cms[m] += cm
            sc = scores(cm)
            rec[m] = {"found": sc["found"], "false_alarm": sc["false_alarm"]}
            txt = (f"found {100 * sc['found']:.0f}%" if sc["found"] is not None else "no GT defect") + f" · false alarm {100 * (sc['false_alarm'] or 0):.1f}%"
            tiles.append(stamp(co.tile_label(pred[m], palette), txt))
        per_layer.append(rec)
        short = v6cfg.SOURCE_SHORT.get(s.group, s.group)
        page_rows.append({"label": f"{short} · {s.layer}", "tiles": tiles})
        if len(page_rows) == args.rows_per_page or i + 1 == n:
            page_no += 1
            title = (f"Human-GT validation layers ({args.fold}) — ground truth vs DefSeg-AM v4 · v5 · v6 · v7 "
                     f"(v7 = {args.variant}, seed {args.seed}, {args.which})    page {page_no} / {n_pages}")
            render_page(page_rows, columns, title, notes, out_dir / "pages" / f"page_{page_no:02d}.png")
            page_rows = []
        print(f"[gt7] {i + 1}/{n}  {s.key}  ({time.time() - t0:.0f}s)")

    res = {m: scores(cms[m]) for m in models}
    doc = {"n_layers": n, "fold": args.fold, "img_size": img_size, "models": models, "checkpoints": checkpoints,
           "v7": {"variant": args.variant, "seed": args.seed, "generated_set": gen_set}, "v5_soot_mode": old_cfg.get("v5", {}).get("soot_mode", "ignore"),
           "scores": res, "per_layer": per_layer,
           "note": "1036² 채점. v4 는 이 레이어를 학습에 썼다. v5 는 PS 를 낼 수 없다 (클래스 체계에 없다)."}
    (out_dir / "summary.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")

    f3 = lambda v: "–" if v is None else f"{v:.3f}"
    pc = lambda v: "–" if v is None else f"{100 * v:.1f}%"
    names = {"v4": "v4 (이 레이어로 학습함)", "v5": "v5" + (" (Soot → Spatter)" if soot else ""), "v6": "v6 (기준선)", "v7": f"v7-{args.variant}"}
    L = [f"# 사람 GT 검증 {n} 레이어 — 정답과 v4 · v5 · v6 · v7 의 예측 ({args.fold})", "",
         f"모두 {img_size}² 에서 채점했다 (TTA · 후처리 없음). v5 · v6 · v7 은 이 레이어를 학습에서 보지 않았고, **v4 는 학습에 썼다** (참고).",
         "DSCNN 의 열은 없다 — DSCNN 의 예측은 ORNL 빌드에만 있고 사람 GT 데이터셋에는 정답만 있다.", "",
         *[f"- {names[m]} = `{checkpoints[m]}`" for m in models], "",
         "## 클래스별 IoU", "", "| 모델 | " + " | ".join(AB) + " | 결함 mIoU |", "|---|" + "---:|" * (NC + 1)]
    for m in models:
        L.append(f"| {names[m]} | " + " | ".join(f3(v) for v in res[m]["iou"]) + f" | {f3(res[m]['defect_miou'])} |")
    L += ["", "## 결함을 찾은 정도", "", "| 모델 | 맞힘 (found) | 정상으로 냄 (놓침) | 잘못 짚음 (정상 → 결함) | 화소 정확도 |", "|---|---:|---:|---:|---:|"]
    for m in models:
        L.append(f"| {names[m]} | {pc(res[m]['found'])} | {pc(res[m]['miss'])} | {pc(res[m]['false_alarm'])} | {pc(res[m]['pixel_acc'])} |")
    L += ["", "맞힘 = 정답이 결함인 화소 가운데 같은 클래스로 낸 비율. v5 는 PS 를 IS 로 내므로 (PS 가 클래스 체계에 없다) 맞힘이 낮게 나온다 — 놓친 것이 아니라 다른 이름으로 낸 것이다.", "",
          "## 레이어별 — 결함을 맞힌 비율", "", "| 레이어 | 정답에 있는 결함 | " + " | ".join(names[m] for m in models) + " |", "|---|---|" + "---:|" * len(models)]
    for r in per_layer:
        L.append(f"| {r['key']} | {' · '.join(r['classes']) or '없음'} | " + " | ".join(pc(r[m]["found"]) for m in models) + " |")
    L += ["", "그림: `pages/page_XX.png` — 한 줄이 한 레이어, 열은 사진 두 장 · 정답 · 모델들의 예측이다."]
    (out_dir / "summary.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L[:40]))
    print(f"[gt7] 끝 → {out_dir}  ({time.time() - t0:.0f}s)")


if __name__ == "__main__":
    main()
