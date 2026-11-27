"""사람 GT 를 보정해 **보정 세트**로 저장한다 (PLAN_v8 §5). CPU 만 쓴다 (125 레이어에 1 – 2분).

보정 세트 = 원본 데이터셋 (`ORNL_Data/DSCNN_Dataset_pbf-2026.11`) 과 같은 구성의 폴더다. 사진만 바뀌고 라벨은 원본 그대로다.

    cache/v8_norm/<이름>/
        visible/0/{id}.png, visible/1/{id}.png      보정한 사진 (H, W) uint8
        annotations                                  원본의 annotations 폴더로 가는 상대 링크 (--copy-annotations 면 복사)
        index.csv, classes.json, conversion.json    원본의 것 그대로
        norm.json                                   소스마다의 대표 밝기 · 옮긴 값 · 지도의 계수 · 안전 장치의 결과 · 잘린 화소
    figures/v8_norm/<이름>/
        check.md, check.json                        점검 표 (PLAN_v8 §3.4) — 라벨은 여기서만 쓴다
        maps.png                                    소스마다 칸 지도 (라벨 없이 / 라벨의 분말 칸) 와 맞춘 면
        sheet_<소스>.png                            소스마다 보정 전 · 후 사진을 나란히

보정의 종류 (`--mode`):
    raw   아무것도 하지 않는다 — 화소가 원본과 같다 (껍데기가 다른 것을 바꾸지 않는다는 스모크용)
    B     밝기 맞춤: 소스마다 (기준 밝기 − 대표 밝기) 를 더한다
    BL    밝기 맞춤 + 조명 제거: 안전 장치를 지난 소스의 사진을 그 소스의 조명 지도로 나눈 뒤 더한다

**통계는 라벨 없이 구한다** (`photo_norm`). 소스의 사진 전부가 들어간다 — 어느 fold 의 검증 레이어든 사진은 들어가고 정답은 들어가지 않는다.

사용 (CWD = 3DP_VPPM/) — 보통은 compose 로 돌린다 (`v8/docker`, DEFSEG_V8_PHASE=norm):
    python -m DefSeg_AM.v8.data.build_norm_set --mode B
    python -m DefSeg_AM.v8.data.build_norm_set --mode BL --overwrite
"""
from __future__ import annotations

import argparse
import csv
import json
import os
import shutil
import time
from collections import defaultdict
from pathlib import Path

import numpy as np
from PIL import Image

from ...v6 import config_v6 as v6cfg
from ...v6.data.data_gt_v6 import check_dataset_schema, load_image_uint8
from .. import config_v8 as cfg
from . import photo_norm as pn

POWDER_ID = 0                    # pbf-2026.11 의 Powder
POWDER_CELL_MIN = 0.98           # 점검: 칸의 화소 가운데 이만큼이 분말이면 "분말 칸"
SHEET_LAYERS = 3                 # 시트에 넣는 레이어 수 (소스마다, 고르게 뽑는다)


def _read_rows(root: Path) -> list[dict]:
    with (root / "index.csv").open(encoding="utf-8-sig", newline="") as fh:
        rows = sorted(csv.DictReader(fh), key=lambda r: r["id"])
    if not rows:
        raise RuntimeError(f"FATAL: {root / 'index.csv'} 에 표본이 없다")
    return rows


def _photo(root: Path, k: int, sid: str) -> np.ndarray:
    return load_image_uint8(root / "visible" / str(k) / f"{sid}.png")


def _pooled(stack: list[np.ndarray], min_n: int) -> np.ndarray | None:
    """레이어마다의 칸 지도 (NaN 가능) → 칸마다 중앙값. min_n 장 아래로 잡힌 칸은 NaN."""
    if len(stack) < 4:
        return None
    s = np.stack(stack)
    cnt = np.isfinite(s).sum(0)
    with np.errstate(all="ignore"):
        m = np.nanmedian(s, axis=0)
    m[cnt < min_n] = np.nan
    return m


def _sd_pct(cell_map: np.ndarray | None) -> float | None:
    """칸 지도에 (다듬지 않은) 2차식 면을 맞춰 그 표준편차를 % 로 — "분말 칸에 남은 넓은 얼룩" 의 크기."""
    if cell_map is None:
        return None
    res = pn.fit_surface(cell_map, trim=False)
    if res is None:
        return None
    surf = res[1]
    return float((surf / surf.mean()).std() * 100)


def _n(v, nd: int = 1, unit: str = "") -> str:
    return "–" if v is None or v != v else f"{v:.{nd}f}{unit}"


def _corr(a: np.ndarray | None, b: np.ndarray | None) -> float | None:
    if a is None or b is None:
        return None
    return float(np.corrcoef(a.ravel(), b.ravel())[0, 1])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--mode", required=True, choices=list(cfg.MODES), help="raw = 그대로, B = 밝기 맞춤, BL = 밝기 맞춤 + 조명 제거")
    ap.add_argument("--name", type=str, default=None, help="보정 세트의 이름 (기본: raw · b115 · bl115)")
    ap.add_argument("--src", type=str, default=None, help="원본 데이터셋 (기본: ORNL_Data/DSCNN_Dataset_pbf-2026.11)")
    ap.add_argument("--ref", type=float, default=cfg.REF_BRIGHTNESS, help="기준 밝기")
    ap.add_argument("--overwrite", action="store_true", help="같은 이름의 세트가 있으면 지우고 다시 만든다")
    ap.add_argument("--copy-annotations", action="store_true", help="라벨을 링크하지 않고 복사한다 (1.1 GB)")
    ap.add_argument("--no-check", action="store_true", help="점검 표 · 시트를 만들지 않는다")
    args = ap.parse_args()

    src = Path(args.src).resolve() if args.src else v6cfg.LEGACY_PBF_ROOT
    check_dataset_schema(src)
    name = args.name or cfg.DEFAULT_SET_NAME[args.mode]
    out = cfg.NORM_SETS_DIR / name
    fig = cfg.NORM_FIG_DIR / name
    if out.exists():
        if not args.overwrite:
            raise SystemExit(f"FATAL: {out} 가 이미 있다 — 다시 만들려면 --overwrite (compose: DEFSEG_V8_EXTRA=--overwrite)")
        if cfg.NORM_SETS_DIR.resolve() not in out.resolve().parents:
            raise SystemExit(f"FATAL: {out} 는 {cfg.NORM_SETS_DIR} 아래가 아니다 — 지우지 않는다")
        shutil.rmtree(out)
    for k in (0, 1):
        (out / "visible" / str(k)).mkdir(parents=True, exist_ok=True)

    rows = _read_rows(src)
    by: dict[str, list[dict]] = defaultdict(list)
    for r in rows:
        by[r["process_code"]].append(r)
    short = lambda s: v6cfg.SOURCE_SHORT.get(s, s)                         # noqa: E731
    print(f"[norm] {src}  →  {out}   (mode={args.mode}, 기준 밝기 {args.ref:g}, {len(rows)} 레이어, {len(by)} 소스)")

    # ── 1) 소스마다 통계 (라벨 없이) ──
    t0 = time.time()
    stats: dict[str, pn.SourceNorm] = {}
    free_maps: dict[str, np.ndarray] = {}
    for s, rs in by.items():
        acc = pn.SourceAccumulator(s, ref=args.ref)
        for r in rs:
            acc.add(_photo(src, 0, r["id"]), _photo(src, 1, r["id"]))
        stats[s] = acc.finish()
        free_maps[s] = acc.cell_map()
        st = stats[s]
        print(f"[norm]   {short(s):5s} n={st.n_layers:2d}  대표 밝기 {st.brightness:5.0f} → 옮기는 값 {st.shift:+4.0f}   "
              f"지도: 크기 {_n(st.map_sd_pct, 1)}%  설명 {_n(st.map_r2, 2)}  "
              f"안전 장치 {''.join('O' if v else 'X' for v in st.gates.values())} → {'쓴다' if st.use_map else '쓰지 않는다'}")
    print(f"[norm] 통계 {time.time() - t0:.0f}s")

    # ── 2) 사진을 보정해 저장 ──
    do_shift = args.mode in (cfg.MODE_B, cfg.MODE_BL)
    do_map = args.mode == cfg.MODE_BL
    t0 = time.time()
    applied: dict[str, dict] = {}
    for s, rs in by.items():
        st = stats[s]
        shift = st.shift if do_shift else 0.0
        use_map = bool(do_map and st.use_map)
        n_clip = n_px = 0
        maps: dict[tuple[int, int], np.ndarray] = {}
        for r in rs:
            for k in (0, 1):
                img = _photo(src, k, r["id"])
                light = None
                if use_map:
                    if img.shape not in maps:
                        maps[img.shape] = st.map_for(img.shape)
                    light = maps[img.shape]
                res, c = pn.apply_norm(img, shift=shift, light=light, return_clipped=True)
                if args.mode == cfg.MODE_RAW and not np.array_equal(res, img):
                    raise RuntimeError(f"FATAL: raw 세트인데 화소가 바뀌었다 ({r['id']})")
                Image.fromarray(res).save(out / "visible" / str(k) / f"{r['id']}.png")
                n_clip += c
                n_px += img.size
        applied[s] = {"shift": float(shift), "map": use_map, "clipped_pct": 100.0 * n_clip / max(n_px, 1)}
    print(f"[norm] 사진 {2 * len(rows)} 장 저장 {time.time() - t0:.0f}s")

    # ── 3) 라벨 · 표 ──
    if args.copy_annotations:
        shutil.copytree(src / "annotations", out / "annotations")
        ann_note = "copied"
    else:
        rel = os.path.relpath(src / "annotations", out)
        os.symlink(rel, out / "annotations", target_is_directory=True)
        ann_note = f"symlink → {rel}"
        if not (out / "annotations" / f"{rows[0]['id']}.npy").is_file():
            raise RuntimeError(f"FATAL: 라벨 링크가 깨졌다: {out / 'annotations'} → {rel}")
    for f in ("index.csv", "classes.json", "conversion.json"):
        if (src / f).is_file():
            shutil.copy2(src / f, out / f)
    conv = json.loads((src / "conversion.json").read_text(encoding="utf-8")) if (src / "conversion.json").is_file() else {}
    doc = {
        "name": name, "version": cfg.VERSION, "mode": args.mode, "ref_brightness": args.ref,
        "formula": "out = round(photo / light_map + shift), clipped to 0..255; light_map only where sources[*].applied.map is true",
        "created_at": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "source_root": str(src), "source_conversion_created_at": conv.get("created_at"),
        "n_layers": len(rows), "annotations": ann_note,
        "statistics": "label-free: median of all pixels of both photos; map from after-spreading photos (per-cell median over layers, trimmed 2nd-order surface)",
        "grid": cfg.GRID, "trim": {"rounds": cfg.TRIM_ROUNDS, "k": cfg.TRIM_K},
        "gates": {"S1_min_layers": cfg.GATE_MIN_LAYERS, "S2_min_sd_pct": cfg.GATE_MIN_SD_PCT, "S3_min_r2": cfg.GATE_MIN_R2},
        "sources": {s: dict(stats[s].to_json(), short=short(s), applied=applied[s]) for s in by},
    }
    (out / cfg.NORM_JSON).write_text(json.dumps(doc, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"[norm] saved → {out}   (라벨: {ann_note})")
    for s in by:
        a = applied[s]
        print(f"[norm]   {short(s):5s} 더한 값 {a['shift']:+4.0f}  조명 지도 {'나눔' if a['map'] else '–'}  잘린 화소 {a['clipped_pct']:.2f}%")

    if not args.no_check:
        check(src, out, fig, by, stats, free_maps, applied, name, args.mode)


# ---------------------------------------------------------------------------
# 점검 (PLAN_v8 §3.4) — 라벨은 여기서만 쓴다
# ---------------------------------------------------------------------------

def check(src: Path, out: Path, fig: Path, by: dict, stats: dict, free_maps: dict, applied: dict, name: str, mode: str) -> None:
    """소스마다: 라벨의 분말 칸으로 구한 지도와 견주고, 보정한 사진에서 분말의 밝기와 남은 얼룩을 다시 잰다. 표 · 지도 그림 · 시트를 쓴다."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig.mkdir(parents=True, exist_ok=True)
    short = lambda s: v6cfg.SOURCE_SHORT.get(s, s)                         # noqa: E731
    t0 = time.time()
    res: dict[str, dict] = {}
    panels = []
    for s, rs in by.items():
        st = stats[s]
        mn = max(3, int(np.ceil(0.2 * len(rs))))
        lab = {"before": {0: [], 1: []}, "after": {0: [], 1: []}}        # 분말 칸의 지도 (레이어마다) — 원본 / 보정한 것
        pmed = {"before": {0: [], 1: []}, "after": {0: [], 1: []}}       # 레이어마다 분말 화소의 중앙값
        pow_frac = []
        for r in rs:
            sid = r["id"]
            pw = np.load(src / "annotations" / f"{sid}.npy") == POWDER_ID
            pf = pn.cell_grid(pw)
            pow_frac.append(float(pw.mean()))
            if pw.sum() <= 1000:
                continue
            for k in (0, 1):
                for tag, root in (("before", src), ("after", out)):
                    img = _photo(root, k, sid)
                    m = float(np.median(img[pw]))
                    pmed[tag][k].append(m)
                    lab[tag][k].append(np.where(pf >= POWDER_CELL_MIN, pn.cell_grid(img) / max(m, 1.0), np.nan))
        lab_map = {t: {k: _pooled(lab[t][k], mn) for k in (0, 1)} for t in lab}
        lab_fit1 = pn.fit_surface(lab_map["before"][1], trim=False) if lab_map["before"][1] is not None else None
        free_fit = pn.fit_surface(free_maps[s], trim=True)
        o = {
            "short": short(s), "n_layers": len(rs), "powder_frac": float(np.mean(pow_frac)),
            "brightness": st.brightness, "shift_applied": applied[s]["shift"], "map_applied": applied[s]["map"],
            "clipped_pct": applied[s]["clipped_pct"],
            "free_sd_pct": st.map_sd_pct, "free_r2": st.map_r2, "gates": st.gates, "use_map": st.use_map,
            # 라벨로 본 것
            "powder_median_before": [float(np.median(pmed["before"][k])) if pmed["before"][k] else None for k in (0, 1)],
            "powder_median_after": [float(np.median(pmed["after"][k])) if pmed["after"][k] else None for k in (0, 1)],
            "powder_layer_range_after": [[float(np.min(pmed["after"][k])), float(np.max(pmed["after"][k]))] if pmed["after"][k] else None
                                         for k in (0, 1)],
            "lab_sd_pct_before": [_sd_pct(lab_map["before"][k]) for k in (0, 1)],
            "lab_sd_pct_after": [_sd_pct(lab_map["after"][k]) for k in (0, 1)],
            "corr_free_vs_label": _corr(free_fit[1] if free_fit else None, lab_fit1[1] if lab_fit1 else None),
        }
        # 같은 얼룩을 단계 (0 – 255) 로 — 더하는 보정은 분말의 밝기를 바꾸므로 % 로는 앞뒤를 견줄 수 없다
        for t in ("before", "after"):
            o[f"lab_sd_levels_{t}"] = [
                None if o[f"lab_sd_pct_{t}"][k] is None or o[f"powder_median_{t}"][k] is None
                else o[f"lab_sd_pct_{t}"][k] * o[f"powder_median_{t}"][k] / 100.0 for k in (0, 1)]
        res[s] = o
        panels.append((s, free_maps[s], free_fit[1] if free_fit else None, lab_map["before"][1], lab_fit1[1] if lab_fit1 else None))
        _sheet(src, out, fig / f"sheet_{short(s)}.png", s, rs, st, applied[s], plt)

    (fig / "check.json").write_text(json.dumps({"name": name, "mode": mode, "sources": res}, ensure_ascii=False, indent=1), encoding="utf-8")

    # ── 지도 그림 ──
    f, axes = plt.subplots(len(panels), 4, figsize=(11, 2.6 * len(panels)), squeeze=False)
    heads = ["label-free: cells (median over layers)", "label-free: fitted surface",
             "labels: powder cells only", "labels: fitted surface"]
    for i, (s, fm, ff, lm, lf) in enumerate(panels):
        for j, m in enumerate((fm, ff, lm, lf)):
            ax = axes[i, j]
            ax.set_xticks([]); ax.set_yticks([])
            if m is None:
                ax.axis("off")
                continue
            ax.imshow(m / np.nanmean(m), cmap="RdBu_r", vmin=0.85, vmax=1.15)
            if i == 0:
                ax.set_title(heads[j], fontsize=9)
        o = res[s]
        axes[i, 0].set_ylabel(f"{o['short']} (n={o['n_layers']})", fontsize=11)
        c = o["corr_free_vs_label"]
        axes[i, 1].set_xlabel(f"sd {_n(o['free_sd_pct'], 1)}%  R2 {_n(o['free_r2'], 2)}  "
                              f"{'USED' if o['map_applied'] else ('passes gates' if o['use_map'] else 'not used')}"
                              + (f"  r(labels) {c:.2f}" if c is not None else ""), fontsize=8)
    f.suptitle(f"v8 norm set '{name}' ({mode}) — broad brightness pattern per source (colour: 0.85 – 1.15 of the source mean)", fontsize=10)
    f.tight_layout(rect=(0, 0, 1, 0.985))
    f.savefig(fig / "maps.png", dpi=110)
    plt.close(f)

    # ── 표 ──
    n = _n

    L = [f"# 보정 세트 `{name}` ({mode}) 의 점검", "",
         f"원본: `{src}`.  만든 때: {time.strftime('%Y-%m-%d %H:%M')}.",
         "통계 (대표 밝기 · 조명 지도) 는 라벨 없이 사진만으로 구했다. 아래 표의 \"라벨로\" 는 점검용으로 정답의 Powder 화소를 읽어 다시 잰 것이다.", "",
         "## 밝기", "",
         "| 소스 | 레이어 | 대표 밝기 (라벨 없이) | 더한 값 | 분말의 밝기 — 원본 (용융 직후 / 도포 직후) | 분말의 밝기 — 보정 후 | 보정 후 레이어마다의 범위 (도포 직후) | 잘린 화소 |",
         "|---|---:|---:|---:|---|---|---|---:|"]
    for s, o in res.items():
        pb, pa, rg = o["powder_median_before"], o["powder_median_after"], o["powder_layer_range_after"][1]
        L.append(f"| {o['short']} | {o['n_layers']} | {o['brightness']:.0f} | {o['shift_applied']:+.0f} | {n(pb[0], 0)} / {n(pb[1], 0)} | "
                 f"{n(pa[0], 0)} / {n(pa[1], 0)} | {(n(rg[0], 0) + ' – ' + n(rg[1], 0)) if rg else '–'} | {o['clipped_pct']:.2f}% |")
    L += ["", "## 조명 지도", "",
          "| 소스 | 얼룩의 크기 (라벨 없이) | 면이 설명하는 정도 | S1 레이어 | S2 크기 | S3 매끄러움 | 지도 | 라벨로 구한 지도와의 상관 | 분말 칸의 넓은 얼룩, 단계 — 원본 (용융 직후 / 도포 직후) | 보정 후 |",
          "|---|---:|---:|:-:|:-:|:-:|---|---:|---|---|"]
    for s, o in res.items():
        g = o["gates"]
        used = "**나눴다**" if o["map_applied"] else ("통과 (이 세트에서는 쓰지 않음)" if o["use_map"] else "쓰지 않음")
        b, a = o["lab_sd_levels_before"], o["lab_sd_levels_after"]
        L.append(f"| {o['short']} | {n(o['free_sd_pct'], 1, '%')} | {n(o['free_r2'], 2)} | {'O' if g['S1_layers'] else 'X'} | {'O' if g['S2_size'] else 'X'} | "
                 f"{'O' if g['S3_smooth'] else 'X'} | {used} | {n(o['corr_free_vs_label'], 2)} | {n(b[0], 1)} / {n(b[1], 1)} | {n(a[0], 1)} / {n(a[1], 1)} |")
    L += ["",
          f"- 안전 장치의 문턱 (제안값): 레이어 {cfg.GATE_MIN_LAYERS} 장 이상 · 얼룩의 크기 {cfg.GATE_MIN_SD_PCT:g}% 이상 · 면이 칸 지도의 변화를 {cfg.GATE_MIN_R2:.0%} 이상 설명.",
          "- \"분말 칸의 넓은 얼룩\" 은 라벨의 분말 칸의 밝기에 2차식 면을 맞춘 것의 표준편차를 단계 (0 – 255) 로 적은 것이다. "
          "지도로 나눈 소스에서만 줄어야 하고, 더하기만 한 소스에서는 앞뒤가 같아야 한다. 면의 성분만 잰다 — 구역 · 줄무늬는 이 값에 다 잡히지 않는다.",
          "- 그림: `maps.png` (소스마다의 칸 지도와 면), `sheet_<소스>.png` (보정 전 · 후 사진)."]
    (fig / "check.md").write_text("\n".join(L) + "\n", encoding="utf-8")
    print("\n".join(L))
    print(f"[norm] 점검 {time.time() - t0:.0f}s → {fig}")


def _sheet(src: Path, out: Path, path: Path, s: str, rs: list[dict], st: pn.SourceNorm, applied: dict, plt) -> None:
    """소스 하나: 레이어 몇 장의 보정 전 · 후 (두 사진) + 쓴 지도."""
    pick = [rs[i] for i in sorted(set(np.linspace(0, len(rs) - 1, min(SHEET_LAYERS, len(rs)), dtype=int).tolist()))]
    f, axes = plt.subplots(len(pick), 5, figsize=(16, 3.3 * len(pick)), squeeze=False)
    heads = ["after melting — original", "after melting — corrected", "after spreading — original", "after spreading — corrected",
             "lighting map (divided)" if applied["map"] else "lighting map (estimated, NOT used)"]
    for i, r in enumerate(pick):
        shape = None
        for j, (root, k) in enumerate(((src, 0), (out, 0), (src, 1), (out, 1))):
            img = _photo(root, k, r["id"])
            shape = img.shape
            small = np.array(Image.fromarray(img).resize((512, 512), Image.BILINEAR))
            axes[i, j].imshow(small, cmap="gray", vmin=0, vmax=255)
            axes[i, j].set_xlabel(f"median {np.median(img):.0f}", fontsize=8)
        if st.coef is not None:
            m = pn.surface_at(st.coef, (96, 96))
            axes[i, 4].imshow(m, cmap="RdBu_r", vmin=0.85, vmax=1.15)
        else:
            axes[i, 4].axis("off")
        axes[i, 0].set_ylabel(f"layer {r['layer']}  ({shape[1]}×{shape[0]})", fontsize=9)
        for j in range(5):
            axes[i, j].set_xticks([]); axes[i, j].set_yticks([])
            if i == 0:
                axes[i, j].set_title(heads[j], fontsize=9)
    f.suptitle(f"{v6cfg.SOURCE_SHORT.get(s, s)} — source brightness {st.brightness:.0f}, added {applied['shift']:+.0f}, "
               f"map {'divided' if applied['map'] else 'not used'} (sd {_n(st.map_sd_pct, 1)}%, R2 {_n(st.map_r2, 2)})  — grey scale 0 – 255, not stretched",
               fontsize=10)
    f.tight_layout(rect=(0, 0, 1, 0.97))
    f.savefig(path, dpi=80)
    plt.close(f)


if __name__ == "__main__":
    main()
