"""공용 배경 캐시 — ORNL HDF5 → 원본 밝기 사진 + CAD 파트 마스크.

규격은 rulesynth/PLAN_RULESYNTH.md §2.3 이다. GenAug 와 RuleSynth 가 같은 캐시를 읽는다.

    cache/ornl_canvas_sz1036/<빌드>/
    ├── visible_0.npy   (n, 1036, 1036) uint8   용융 직후 사진, 원본 밝기
    ├── visible_1.npy   (n, 1036, 1036) uint8   도포 직후 사진, 원본 밝기
    ├── part_mask.npy   (n, 1036, 1036) uint8   1 = 파트 자리 (slices/part_ids > 0)
    ├── meta.npz        orig_layer_idxs (HDF5 의 레이어 번호), foot_frac (파트 화소의 비율)
    └── meta.json       만든 방법 · 시각 · 읽은 파일

- 밝기: `clip(값, 0, 255)` → uint8 → BILINEAR 로 1036². 사람 GT 를 줄이는 방법과 같다.
  v1 의 이미지 캐시 (`cache/resized_sz1036/`) 는 레이어마다 1–99 백분위로 늘린 것이라 사람 GT 와 눈금이 다르다.
- 레이어: 높이 5–95% 구간 (v1 캐시와 같은 구간) 에서 5층마다 1층. 이웃한 층은 형상이 거의 같다 (층 두께 50 μm).
- 읽는 것은 `slices/camera_data/visible/{0,1}` 와 `slices/part_ids` 뿐이다. `slices/segmentation_results` (DSCNN 예측) 는 열지 않는다.
- HDF5 는 층 방향 52개씩 묶여 압축되어 있다. 52층 단위로 정렬해 읽어야 묶음 하나를 한 번만 푼다.
  묶음 하나 (사진 두 장 + 마스크) 에 약 40초가 걸려, 묶음을 여러 프로세스에 나눠 준다.
- `meta.npz` 의 `bright_frac` (RuleSynth §2.3 에 적힌 항목) 은 넣지 않았다. 정의가 그 계획의 규칙 코드에 달려 있고,
  캐시의 사진과 마스크에서 다시 계산할 수 있다.

실행 (CWD = 3DP_VPPM/, DefSeg_AM/venv. CPU 만 쓴다):
    python -m DefSeg_AM.genaug.data.build_canvas_cache                       # 다섯 빌드
    python -m DefSeg_AM.genaug.data.build_canvas_cache --builds B3 --workers 8
    python -m DefSeg_AM.genaug.data.build_canvas_cache --verify 5            # 만든 뒤 HDF5 와 다시 견준다
"""
from __future__ import annotations

import argparse
import json
import os
import time
from multiprocessing import get_context
from pathlib import Path

import h5py
import numpy as np

from ...common.data.image_utils import resize_image_uint8, resize_label
from ...common.utils.log import setup_logger
from .. import config_genaug as cfg

_KEYS = {"visible_0": "slices/camera_data/visible/0", "visible_1": "slices/camera_data/visible/1"}
_PART_KEY = "slices/part_ids"
_ARRAYS = ("visible_0", "visible_1", "part_mask")


def select_layers(n_layers: int, every: int = cfg.CANVAS_LAYER_EVERY) -> np.ndarray:
    """빌드의 레이어 수 → 캐시에 넣을 HDF5 레이어 번호 (높이 5–95% 구간에서 every 층마다)."""
    lo = int(cfg.CANVAS_LAYER_LO_FRAC * n_layers)
    hi = int(cfg.CANVAS_LAYER_HI_FRAC * n_layers)
    return np.arange(lo, hi, every, dtype=np.int32)


def raw_to_uint8(raw: np.ndarray) -> np.ndarray:
    """HDF5 의 사진 값 (float32, 0–255) → uint8. 밝기를 늘리지 않는다."""
    return np.clip(raw, 0, 255).astype(np.uint8)


def _slab_task(task):
    """52층 묶음 하나를 읽어 그 안의 고른 레이어를 캐시에 쓴다. (묶음 번호, [(행, 파트 비율)], 통계) 를 돌려준다."""
    hdf5_path, out_dir, slab, layer_rows, img_size = task
    s0 = slab * cfg.HDF5_SLAB
    stats = {"out_of_range": 0, "non_integer": 0}
    foot = []
    with h5py.File(hdf5_path, "r") as f:
        s1 = min(s0 + cfg.HDF5_SLAB, f[_PART_KEY].shape[0])
        for name, key in _KEYS.items():
            blk = f[key][s0:s1]                                   # (≤52, H, W) float32
            arr = np.lib.format.open_memmap(out_dir / f"{name}.npy.partial", mode="r+")
            for layer, row in layer_rows:
                raw = blk[layer - s0]
                stats["out_of_range"] += int(((raw < 0) | (raw > 255)).sum())
                stats["non_integer"] += int((raw != np.floor(raw)).sum())
                arr[row] = resize_image_uint8(raw_to_uint8(raw), img_size)
            arr.flush()
            del arr, blk
        blk = f[_PART_KEY][s0:s1]                                 # (≤52, H, W) uint32
        arr = np.lib.format.open_memmap(out_dir / "part_mask.npy.partial", mode="r+")
        for layer, row in layer_rows:
            m = resize_label((blk[layer - s0] > 0).astype(np.int8), img_size).astype(np.uint8)
            arr[row] = m
            foot.append((row, float(m.mean())))
        arr.flush()
        del arr, blk
    return slab, foot, stats


def is_complete(out_dir: Path, layers: np.ndarray) -> bool:
    if not all((out_dir / f"{a}.npy").is_file() for a in _ARRAYS) or not (out_dir / "meta.npz").is_file():
        return False
    try:
        m = np.load(out_dir / "meta.npz")
        return np.array_equal(m["orig_layer_idxs"], layers)
    except Exception:
        return False


def build_one(build: str, root: Path, every: int, workers: int, img_size: int, limit_slabs: int, log) -> None:
    hdf5_path = cfg.ORNL_HDF5_DIR / cfg.ORNL_BUILD_FILES[build]
    if not hdf5_path.is_file():
        raise FileNotFoundError(f"FATAL: {hdf5_path} 없음")
    with h5py.File(hdf5_path, "r") as f:
        shape = f[_KEYS["visible_0"]].shape
        if f[_KEYS["visible_1"]].shape != shape or f[_PART_KEY].shape != shape:
            raise RuntimeError(f"FATAL: {build} 의 사진과 part_ids 의 모양이 다르다")
        chunk = f[_KEYS["visible_0"]].chunks
    if chunk is not None and chunk[0] != cfg.HDF5_SLAB:
        log.warning(f"{build}: HDF5 의 층 방향 chunk 가 {chunk[0]} 이다 (설정은 {cfg.HDF5_SLAB}) — 느려질 수 있다")
    layers = select_layers(shape[0], every)
    if limit_slabs > 0:
        first = int(layers[0]) // cfg.HDF5_SLAB
        layers = layers[layers // cfg.HDF5_SLAB < first + limit_slabs]
    n = len(layers)
    out_dir = root / build
    if is_complete(out_dir, layers):
        log.info(f"{build}: 캐시가 이미 있다 ({n} 레이어) — 건너뜀")
        return
    out_dir.mkdir(parents=True, exist_ok=True)
    for a in _ARRAYS:
        np.lib.format.open_memmap(out_dir / f"{a}.npy.partial", mode="w+", dtype=np.uint8,
                                  shape=(n, img_size, img_size)).flush()
    by_slab: dict[int, list[tuple[int, int]]] = {}
    for row, layer in enumerate(layers.tolist()):
        by_slab.setdefault(layer // cfg.HDF5_SLAB, []).append((layer, row))
    tasks = [(hdf5_path, out_dir, slab, lr, img_size) for slab, lr in sorted(by_slab.items())]
    log.info(f"{build}: HDF5 {shape[0]} 레이어 → {n} 레이어 (레이어 {int(layers[0])}–{int(layers[-1])}, {every} 층마다), "
             f"묶음 {len(tasks)}개, 프로세스 {workers}개, 디스크 {3 * n * img_size * img_size / 1e9:.1f} GB")

    foot_frac = np.zeros(n, dtype=np.float32)
    total = {"out_of_range": 0, "non_integer": 0}
    t0 = time.time()
    done = 0
    with get_context("fork").Pool(workers) as pool:
        for slab, foot, stats in pool.imap_unordered(_slab_task, tasks):
            for row, frac in foot:
                foot_frac[row] = frac
            for k in total:
                total[k] += stats[k]
            done += 1
            if done % 5 == 0 or done == len(tasks):
                el = time.time() - t0
                log.info(f"  {build} 묶음 [{done:3d}/{len(tasks)}]  {el / 60:.1f}분  남은 시간 {el * (len(tasks) - done) / done / 60:.1f}분")

    for a in _ARRAYS:
        os.replace(out_dir / f"{a}.npy.partial", out_dir / f"{a}.npy")
    np.savez(out_dir / "meta.npz", orig_layer_idxs=layers, foot_frac=foot_frac)
    (out_dir / "meta.json").write_text(json.dumps({
        "format": "ornl-canvas/1",
        "spec": "rulesynth/PLAN_RULESYNTH.md §2.3",
        "built_by": "DefSeg_AM.genaug.data.build_canvas_cache",
        "created": time.strftime("%Y-%m-%dT%H:%M:%S%z"),
        "build": build,
        "hdf5": hdf5_path.name,
        "hdf5_layers": int(shape[0]),
        "native_size": [int(shape[1]), int(shape[2])],
        "img_size": img_size,
        "layers": n,
        "layer_rule": f"[int({cfg.CANVAS_LAYER_LO_FRAC}·N), int({cfg.CANVAS_LAYER_HI_FRAC}·N)) 에서 {every} 층마다",
        "brightness": "raw — clip(value, 0, 255) → uint8, BILINEAR resize (no percentile stretch)",
        "part_mask": "slices/part_ids > 0, NEAREST resize",
        "reads": [*_KEYS.values(), _PART_KEY],
        "pixels_out_of_0_255": total["out_of_range"],
        "pixels_non_integer": total["non_integer"],
        "meta_npz": ["orig_layer_idxs", "foot_frac"],
        "note": "bright_frac 는 넣지 않았다 (정의가 RuleSynth 의 규칙 코드에 있다. 캐시의 배열에서 계산할 수 있다)",
    }, indent=1, ensure_ascii=False), encoding="utf-8")
    log.info(f"{build}: 완료 {(time.time() - t0) / 60:.1f}분 — 0–255 밖의 화소 {total['out_of_range']}, 정수가 아닌 화소 {total['non_integer']}, "
             f"파트 비율 평균 {100 * foot_frac.mean():.2f}%")


def verify(build: str, root: Path, k: int, img_size: int, log) -> bool:
    """캐시의 레이어 k 개를 HDF5 에서 다시 만들어 화소 단위로 견준다."""
    out_dir = root / build
    layers = np.load(out_dir / "meta.npz")["orig_layer_idxs"]
    rows = np.random.default_rng(0).choice(len(layers), size=min(k, len(layers)), replace=False)
    arrs = {a: np.load(out_dir / f"{a}.npy", mmap_mode="r") for a in _ARRAYS}
    ok = True
    with h5py.File(cfg.ORNL_HDF5_DIR / cfg.ORNL_BUILD_FILES[build], "r") as f:
        for row in sorted(rows.tolist()):
            layer = int(layers[row])
            same = []
            for name, key in _KEYS.items():
                ref = resize_image_uint8(raw_to_uint8(f[key][layer]), img_size)
                same.append(bool(np.array_equal(ref, arrs[name][row])))
            ref = resize_label((f[_PART_KEY][layer] > 0).astype(np.int8), img_size).astype(np.uint8)
            same.append(bool(np.array_equal(ref, arrs["part_mask"][row])))
            ok &= all(same)
            m = arrs["part_mask"][row] > 0
            v0 = arrs["visible_0"][row]
            inside = float(v0[m].mean()) if m.any() else float("nan")
            log.info(f"  {build} 행 {row} (레이어 {layer}): HDF5 와 같음 {same}  파트 {100 * m.mean():.2f}%  "
                     f"visible/0 평균 — 파트 안 {inside:.1f} · 밖 {float(v0[~m].mean()):.1f}")
    return ok


def main():
    ap = argparse.ArgumentParser(description="공용 배경 캐시 (ORNL 원본 밝기 사진 + CAD 파트 마스크)")
    ap.add_argument("--builds", nargs="+", default=list(cfg.CANVAS_BUILDS), choices=list(cfg.CANVAS_BUILDS))
    ap.add_argument("--out", type=Path, default=cfg.CANVAS_ROOT)
    ap.add_argument("--every", type=int, default=cfg.CANVAS_LAYER_EVERY)
    ap.add_argument("--workers", type=int, default=6, help="빌드 하나를 읽는 프로세스 수 (프로세스당 메모리 약 1.5 GB)")
    ap.add_argument("--img-size", type=int, default=cfg.IMG_SIZE)
    ap.add_argument("--limit-slabs", type=int, default=0, help="스모크용 — 앞의 묶음 몇 개만 (--out 을 다른 곳으로 줄 것)")
    ap.add_argument("--rebuild", action="store_true")
    ap.add_argument("--verify", type=int, default=0, help="빌드마다 이 수의 레이어를 HDF5 와 다시 견준다 (만들지는 않는다)")
    args = ap.parse_args()
    log = setup_logger(rank=0, name="genaug.canvas")
    if args.limit_slabs > 0 and args.out == cfg.CANVAS_ROOT:
        raise SystemExit("FATAL: --limit-slabs 는 --out 을 캐시가 아닌 곳으로 줄 때만 쓴다")
    if args.verify > 0:
        ok = all([verify(b, args.out, args.verify, args.img_size, log) for b in args.builds])
        log.info("검증: " + ("모두 같다" if ok else "다른 레이어가 있다"))
        raise SystemExit(0 if ok else 1)
    args.out.mkdir(parents=True, exist_ok=True)
    for b in args.builds:
        if args.rebuild:
            for p in (args.out / b).glob("*"):
                p.unlink()
        build_one(b, args.out, args.every, args.workers, args.img_size, args.limit_slabs, log)


if __name__ == "__main__":
    main()
