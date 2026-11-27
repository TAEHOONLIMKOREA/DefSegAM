"""Q0 — 주석 없는 배경에 진짜 결함이 얼마나 있는지 사람이 보는 시트 (PLAN_GENAUG §4).

빌드마다 무작위 256×256 창 100개를 한 장에 모은다. 칸 하나 = 같은 창의 visible/0 (왼쪽) 과 visible/1 (오른쪽).
하늘색 선은 CAD 의 파트 경계다. **눈에 띄는 결함이 있는 창이 100개 가운데 5개 이하면** 그 빌드의 창에
정상 라벨 (Powder / Printed) 을 적어도 된다 — 학습 명령의 `--normal-builds` 에 넣는다 (PLAN §3.2, §3.9).

- 창은 학습 때와 같은 범위 (사진 가장자리 5% 를 뺀 안쪽) 에서 뽑는다.
- 보기 좋게 하려고 칸마다 밝기를 1–99 백분위로 늘려 그린다. 표시만 그렇고 캐시의 값은 원본 밝기다.
- 같이 나오는 CSV 에 창마다 빌드 · HDF5 레이어 · 자리가 있다. 결함이 보인 칸의 번호를 적어 두면 나중에 다시 찾을 수 있다.

실행 (CWD = 3DP_VPPM/, DefSeg_AM/venv. CPU 만 쓴다):
    python -m DefSeg_AM.genaug.checks.q0_sheet                    # 기본: B3 B4 B5
    python -m DefSeg_AM.genaug.checks.q0_sheet --builds B3 --n 100
산출물: figures/genaug/q0/<빌드>_sheet.png, <빌드>_windows.csv
"""
from __future__ import annotations

import argparse
import csv
from pathlib import Path

import numpy as np
from PIL import Image, ImageDraw
from scipy import ndimage

from ...common.utils.log import setup_logger
from .. import config_genaug as cfg
from ..data.canvas_pool import CanvasPool

_SQ = np.ones((3, 3), dtype=bool)


def _stretch(a: np.ndarray) -> np.ndarray:
    lo, hi = np.percentile(a, [1, 99])
    return np.clip((a.astype(np.float32) - lo) / max(hi - lo, 1.0) * 255.0, 0, 255).astype(np.uint8)


def make_sheet(canvas: CanvasPool, build: str, n: int, win: int, cols: int, seed: int, out_dir: Path, log) -> None:
    rng = np.random.default_rng([seed, int(build[1:])])
    S = canvas.img_size
    e = int(round(cfg.EDGE_FRAC * S))
    gap, th = 6, 12
    rows = (n + cols - 1) // cols
    cw, chh = 2 * win + 2, win + th
    sheet = Image.new("RGB", (cols * (cw + gap) + gap, rows * (chh + gap) + gap + 30), (24, 24, 24))
    d = ImageDraw.Draw(sheet)
    d.text((gap, 4), f"Q0 background check - build {build} - {n} random {win}x{win} windows - each cell: visible/0 | visible/1 - cyan: CAD part outline", fill=(255, 255, 255))
    d.text((gap, 17), "count the cells that show a real defect (spatter/soot, streak, uncovered or protruding part, debris ...). pass: <= 5 of 100.   [display: per-cell 1-99% stretch]", fill=(190, 190, 190))
    records = []
    for k in range(n):
        row = int(rng.integers(canvas.n_layers[build]))
        y0, x0 = int(rng.integers(e, S - e - win + 1)), int(rng.integers(e, S - e - win + 1))
        i0, i1, part = canvas.crop(build, row, y0, x0, win, win)
        edge = part & ~ndimage.binary_erosion(part, structure=_SQ)
        tiles = []
        for img in (i0, i1):
            rgb = np.stack([_stretch(img)] * 3, axis=-1)
            rgb[edge] = (0, 200, 255)
            tiles.append(rgb)
        cell = np.concatenate([tiles[0], np.full((win, 2, 3), 24, np.uint8), tiles[1]], axis=1)
        cx, cy = gap + (k % cols) * (cw + gap), 30 + gap + (k // cols) * (chh + gap)
        layer = int(canvas.orig_layers[build][row])
        d.text((cx, cy), f"#{k:03d}  L{layer}  y{y0} x{x0}  part {100 * part.mean():.0f}%", fill=(255, 230, 120))
        sheet.paste(Image.fromarray(cell), (cx, cy + th))
        records.append(dict(idx=k, build=build, cache_row=row, hdf5_layer=layer, y0=y0, x0=x0, part_frac=round(float(part.mean()), 4), defect_seen=""))
    out_dir.mkdir(parents=True, exist_ok=True)
    sheet.save(out_dir / f"{build}_sheet.png")
    with (out_dir / f"{build}_windows.csv").open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=list(records[0]))
        w.writeheader()
        w.writerows(records)
    log.info(f"{build}: {n} 창 → {out_dir / f'{build}_sheet.png'}  ({sheet.size[0]}×{sheet.size[1]})")


def main():
    ap = argparse.ArgumentParser(description="Q0 — 배경 점검 시트")
    ap.add_argument("--builds", nargs="+", default=list(cfg.TRAIN_BUILDS))
    ap.add_argument("--n", type=int, default=100)
    ap.add_argument("--win", type=int, default=cfg.WIN)
    ap.add_argument("--cols", type=int, default=10)
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    ap.add_argument("--out", type=Path, default=cfg.FIGURE_DIR / "genaug" / "q0")
    args = ap.parse_args()
    log = setup_logger(0, "genaug.q0")
    canvas = CanvasPool(tuple(args.builds), log=log)
    for b in args.builds:
        make_sheet(canvas, b, args.n, args.win, args.cols, args.seed, args.out, log)


if __name__ == "__main__":
    main()
