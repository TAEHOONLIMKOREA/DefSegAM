"""기존 IS 판정의 검토 시트 — 사람이 `v6/legacy_review.csv` 를 확인할 때 보는 그림 (CPU, 읽기만). PLAN_v6 §2.4, D4.

판정표의 레이어마다 한 줄씩 세 칸을 그린다:
    visible/0 (용융 직후)  |  visible/1 (도포 직후)  |  visible/1 + 지금 데이터셋에 들어간 라벨
        IS (2) 분홍, PS (3) 보라 — DIVA 화면의 색. 미레이블로 둔 레이어는 원본의 기존 IS 영역을 노란 테두리로만 그린다.
PS 는 "그림자 + 경사면이 두 사진에 다 보인다" 가 기준이라 두 사진을 나란히 둔다. 사진은 보기 좋게 대비만 늘렸다 (학습 입력과 무관).

그림의 글자는 영어로만 쓴다 (컨테이너에 한글 글꼴이 없다). 판정 근거 (basis 열) 는 CSV 에서 읽는다.

판정을 고치려면: `v6/legacy_review.csv` 의 decision (IS / PS / ignore) · fill (yes / 빈칸) 을 고치고,
호스트에서 `python -m DefSeg_AM.v6.data.convert_legacy --overwrite` 로 다시 변환한 뒤 (약 3분) 이 시트를 다시 만든다.

출력: figures/<run>/dataset_checks/review_sheet_<n>_<소스>.png

사용 (CWD = 3DP_VPPM/):
    python -m DefSeg_AM.v6.scripts.review_legacy                     # 45 레이어 전부
    python -m DefSeg_AM.v6.scripts.review_legacy --uncertain-only    # 확신이 중간 이하인 레이어만 (먼저 볼 것)
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage as ndi

from .. import config_v6 as cfg
from ..data.data_gt_v6 import check_dataset_schema

PANEL_PX = 460                     # 한 칸의 긴 변 (화소)
ROWS_PER_PAGE = 6
COLOR = {
    "IS": np.array(cfg.DIVA_COLORS[2], dtype=np.float32) / 255.0,
    "PS": np.array(cfg.DIVA_COLORS[3], dtype=np.float32) / 255.0,
    "ignore": np.array([255, 221, 0], dtype=np.float32) / 255.0,
}
TITLE_COLOR = {"IS": "#b0306a", "PS": "#5b21b6", "ignore": "#555555"}
CONF_EN = {"높음": "high", "중간": "medium", "낮음": "low"}
SHORT_TO_NAME = {s["short"]: s["name"] for s in cfg.LEGACY_SOURCES}


def _stretch(a: np.ndarray) -> np.ndarray:
    a = a.astype(np.float32)
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a - lo) / max(float(hi - lo), 1e-6), 0, 1)


def _small(img: np.ndarray, mask: bool = False) -> np.ndarray:
    h, w = img.shape
    s = PANEL_PX / max(h, w)
    tw, th = max(1, round(w * s)), max(1, round(h * s))
    if mask:        # 가는 선이 사라지지 않게 면적 평균으로 줄인 뒤 조금이라도 걸친 화소를 남긴다
        return np.array(Image.fromarray((img * 255).astype(np.uint8)).resize((tw, th), Image.BOX)) > 5
    return np.array(Image.fromarray(img).resize((tw, th), Image.BILINEAR))


def load_review_rows(path: Path) -> list[dict]:
    with path.open(encoding="utf-8-sig", newline="") as fh:
        rows = list(csv.DictReader(fh))
    for r in rows:
        if r["decision"] not in COLOR:
            raise ValueError(f"{path}: decision 은 IS / PS / ignore 가운데 하나여야 한다: {r['source']}/{r['layer']} = {r['decision']!r}")
        if r["source"] not in SHORT_TO_NAME:
            raise ValueError(f"{path}: 모르는 소스 약어 {r['source']!r}")
    return rows


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--review", type=Path, default=cfg.LEGACY_REVIEW_CSV)
    ap.add_argument("--data-root", type=Path, default=cfg.LEGACY_PBF_ROOT)
    ap.add_argument("--uncertain-only", action="store_true", help="확신이 '높음' 이 아닌 IS / PS 레이어만")
    ap.add_argument("--out-dir", type=Path, default=None)
    args = ap.parse_args()

    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    root = args.data_root
    check_dataset_schema(root)
    with (root / "index.csv").open(encoding="utf-8-sig", newline="") as fh:
        index = {(r["process_code"], int(r["layer"])): r for r in csv.DictReader(fh)}
    rows = load_review_rows(args.review)
    if args.uncertain_only:
        rows = [r for r in rows if r["decision"] != "ignore" and r.get("confidence") != "높음"]
    if not rows:
        ap.error("그릴 레이어가 없다")

    out_dir = args.out_dir or (cfg.FIGURE_DIR / cfg.default_run_name() / "dataset_checks")
    out_dir.mkdir(parents=True, exist_ok=True)
    prefix = "review_sheet_uncertain" if args.uncertain_only else "review_sheet"
    for old in out_dir.glob(f"{prefix}_[0-9]*.png"):      # 판정표가 바뀌면 쪽 수가 달라진다 — 지난 그림을 남기지 않는다
        old.unlink()

    # 소스가 바뀌는 곳과 ROWS_PER_PAGE 마다 쪽을 나눈다 (소스마다 사진의 가로세로 비가 다르다)
    pages: list[list[dict]] = []
    for r in rows:
        if not pages or pages[-1][0]["source"] != r["source"] or len(pages[-1]) >= ROWS_PER_PAGE:
            pages.append([])
        pages[-1].append(r)

    for n, page in enumerate(pages, start=1):
        first = index[(SHORT_TO_NAME[page[0]["source"]], int(page[0]["layer"]))]
        aspect = int(first["height"]) / int(first["width"])
        cell_w = 3.6 if aspect <= 1.2 else 2.9          # 세로로 긴 사진 (Binder Jet) 은 칸을 좁힌다
        cell_h = cell_w * aspect + 0.85
        legend_rows = 1 if 3 * cell_w >= 10 else 3      # 좁은 쪽은 범례를 한 줄에 하나씩
        legend_h = 0.3 * legend_rows + 0.25
        fig, axes = plt.subplots(len(page), 3, figsize=(3 * cell_w, len(page) * cell_h + legend_h), dpi=100, squeeze=False)
        for k, r in enumerate(page):
            src_name = SHORT_TO_NAME[r["source"]]
            meta = index.get((src_name, int(r["layer"])))
            if meta is None:
                raise KeyError(f"판정표의 {r['source']}/{r['layer']} 가 데이터셋 {root.name} 에 없다")
            sid = meta["id"]
            i0 = np.array(Image.open(root / "visible" / "0" / f"{sid}.png"))
            i1 = np.array(Image.open(root / "visible" / "1" / f"{sid}.png"))
            lab = np.load(root / "annotations" / f"{sid}.npy")
            dec = r["decision"]
            if dec == "ignore":         # 데이터셋에는 미레이블로 들어갔다 — 원본 주석의 기존 IS 영역을 테두리로 보인다
                names = cfg.LEGACY_NATIVE_NAMES[src_name]
                is_ids = [nid for nid, name in names.items() if name in cfg.LEGACY_IS_NAMES]
                m = np.isin(np.load(cfg.PROJECT_ROOT / "ORNL_Data" / meta["orig_ann"]), is_ids)
            else:
                m = lab == cfg.LEGACY_IS_CHOICES[dec.lower()]
            g0, g1, ms = _stretch(_small(i0)), _stretch(_small(i1)), _small(m, mask=True)
            edge = ndi.binary_dilation(ms, iterations=1) & ~ndi.binary_erosion(ms, iterations=1)
            over = np.stack([g1] * 3, axis=-1)
            c = COLOR[dec]
            if dec != "ignore":
                over[ms] = 0.5 * over[ms] + 0.5 * c
            over[edge] = c
            axes[k][0].imshow(g0, cmap="gray", vmin=0, vmax=1)
            axes[k][1].imshow(g1, cmap="gray", vmin=0, vmax=1)
            axes[k][2].imshow(over)
            for ax in axes[k]:
                ax.axis("off")
            what = {"IS": "IS", "PS": "PS", "ignore": "left unlabeled"}[dec]
            conf = f", {CONF_EN.get(r.get('confidence', ''), r.get('confidence', '?'))} confidence" if dec != "ignore" else ""
            filled = " · filled from the boundary line" if (r.get("fill") or "").strip().lower() == "yes" else ""
            # 제목은 두 줄 — 칸이 좁아도 옆 칸으로 넘치지 않게
            axes[k][0].set_title(f"id {sid}  {r['source']} / {int(r['layer']):07d}\nvisible/0 (after melt)", fontsize=8.5, loc="left")
            axes[k][1].set_title("\nvisible/1 (after spread)", fontsize=8.5, loc="left")
            axes[k][2].set_title(f"-> {what}{conf}\n{100 * float(m.mean()):.2f}% of image{filled}",
                                 fontsize=8.5, loc="left", color=TITLE_COLOR[dec])
        fig.legend(
            handles=[
                Patch(facecolor=COLOR["IS"], label="2 Incomplete Spreading (IS)"),
                Patch(facecolor=COLOR["PS"], label="3 Powder Slumping (PS)"),
                Patch(facecolor="none", edgecolor=COLOR["ignore"], label="left unlabeled (-1): outline of the old IS annotation"),
            ],
            loc="upper center", ncol=3 if legend_rows == 1 else 1, fontsize=9.5, frameon=False, bbox_to_anchor=(0.5, 1.0),
        )
        fig.tight_layout(rect=(0, 0, 1, 1.0 - legend_h / fig.get_figheight()))
        out = out_dir / f"{prefix}_{n:02d}_{page[0]['source']}.png"
        fig.savefig(out)
        plt.close(fig)
        print(f"saved {out}  ({len(page)} layers: {page[0]['source']} {page[0]['layer']} – {page[-1]['layer']})")
    n_by = {d: sum(1 for r in rows if r["decision"] == d) for d in COLOR}
    print(f"{len(rows)} 레이어 · {len(pages)} 쪽 — IS {n_by['IS']}, PS {n_by['PS']}, 미레이블 {n_by['ignore']}  (판정표: {args.review})")


if __name__ == "__main__":
    main()
