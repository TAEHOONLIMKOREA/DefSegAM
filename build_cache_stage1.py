"""Stage 1 학습용 사전 resize+uint8+argmax cache 빌드 (1회).

기존 stage1_layer_index_{train,val}.npz (이미 필터링된 유효 layer 들) 를 받아
각 build 마다 raw HDF5 에서 chunk-read → percentile uint8 → BILINEAR resize → uint8 .npy memmap,
12 mask → argmax int8 → NEAREST resize → int8 .npy memmap 으로 저장.

학습 시엔 PIL/percentile/argmax 가 사라지고 memmap slice 한 줄만 남아 batch 당 1~2초 → 0.1초로 단축됨.

사용:
    python -m DefSeg_AM.build_cache_stage1 [--img-size 1036] [--chunk-size 50] [--rebuild]

산출:
    cache/resized_sz<IMG>/
    ├── B1/
    │   ├── visible_0.npy   (n_kept, IMG, IMG) uint8
    │   ├── visible_1.npy
    │   ├── label.npy        int8
    │   └── meta.npz         (orig_layer_idxs, defect_ratios)
    └── B2/...B5/
    + train_index.npz, val_index.npz  (aggregated entries with build/cache_row/defect_ratio)
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import h5py
import numpy as np
from PIL import Image

from . import config
from .data_ornl import (
    build_layer_index,
    ornl_image_to_uint8,
    resize_image_uint8,
    resize_label,
)
from .log import setup_logger


def cache_dir_for(img_size: int) -> Path:
    return config.CACHE_DIR / f"resized_sz{img_size}"


def build_one(
    build_id: str,
    layer_idxs_keep: list[int],
    defect_ratios_keep: list[float],
    out_dir: Path,
    img_size: int,
    chunk_size: int,
    log,
) -> None:
    hdf5_path = config.ORNL_HDF5_DIR / config.ORNL_BUILD_FILES[build_id]
    out_dir.mkdir(parents=True, exist_ok=True)

    n = len(layer_idxs_keep)
    if n == 0:
        log.warning(f"{build_id}: 0 layers to cache, skip")
        return

    v0_path = out_dir / "visible_0.npy"
    v1_path = out_dir / "visible_1.npy"
    lb_path = out_dir / "label.npy"

    v0_arr = np.lib.format.open_memmap(
        v0_path, mode="w+", dtype=np.uint8, shape=(n, img_size, img_size),
    )
    v1_arr = np.lib.format.open_memmap(
        v1_path, mode="w+", dtype=np.uint8, shape=(n, img_size, img_size),
    )
    lb_arr = np.lib.format.open_memmap(
        lb_path, mode="w+", dtype=np.int8, shape=(n, img_size, img_size),
    )

    bytes_per_layer = img_size * img_size * 3  # uint8 + uint8 + int8
    log.info(f"{build_id}: {n} layers → {n * bytes_per_layer / 1e9:.2f} GB on disk")

    t0 = time.time()
    with h5py.File(hdf5_path, "r") as f:
        vis0 = f["slices/camera_data/visible/0"]
        vis1 = f["slices/camera_data/visible/1"]
        seg = f["slices/segmentation_results"]
        H, W = vis0.shape[1:]

        for start in range(0, n, chunk_size):
            end = min(start + chunk_size, n)
            chunk_lis = layer_idxs_keep[start:end]
            lo, hi = chunk_lis[0], chunk_lis[-1] + 1
            n_chunk = end - start
            t_chunk = time.time()

            # Chunked HDF5 reads — 1 syscall per dataset slice
            v0_raw = vis0[lo:hi]                                     # (hi-lo, H, W) float32
            v1_raw = vis1[lo:hi]
            masks_raw = np.stack(                                    # (12, hi-lo, H, W) bool
                [seg[str(c)][lo:hi] for c in range(config.N_CLASSES)],
                axis=0,
            )

            # Per-layer processing (PIL resize is sequential)
            for offset, li in enumerate(chunk_lis):
                idx = li - lo
                v0_u8 = ornl_image_to_uint8(v0_raw[idx])
                v1_u8 = ornl_image_to_uint8(v1_raw[idx])
                v0_arr[start + offset] = resize_image_uint8(v0_u8, img_size)
                v1_arr[start + offset] = resize_image_uint8(v1_u8, img_size)

                lab = np.full((H, W), config.IGNORE_INDEX, dtype=np.int8)
                for c in range(config.N_CLASSES):
                    lab[masks_raw[c, idx]] = c
                lb_arr[start + offset] = resize_label(lab, img_size)

            dt_chunk = time.time() - t_chunk
            elapsed = time.time() - t0
            eta = elapsed * (n - end) / max(end, 1)
            log.info(
                f"  {build_id} [{end:5d}/{n}]  chunk {n_chunk}L in {dt_chunk:.1f}s "
                f"({dt_chunk/n_chunk*1000:.0f}ms/L)  elapsed {elapsed/60:.1f}m  ETA {eta/60:.1f}m"
            )

    v0_arr.flush(); v1_arr.flush(); lb_arr.flush()
    del v0_arr, v1_arr, lb_arr

    np.savez(
        out_dir / "meta.npz",
        orig_layer_idxs=np.array(layer_idxs_keep, dtype=np.int32),
        defect_ratios=np.array(defect_ratios_keep, dtype=np.float32),
    )
    log.info(f"{build_id}: done in {(time.time()-t0)/60:.1f} min")


def build_split(
    split_name: str,
    builds: list[str],
    cache_root: Path,
    img_size: int,
    chunk_size: int,
    rebuild: bool,
    log,
) -> None:
    """split_name='train' or 'val'."""
    # Layer index cache (= 어떤 layer 를 valid 로 볼 것인지) 는 기존 캐시 재사용
    layer_cache = config.CACHE_DIR / f"stage1_layer_index_{split_name}.npz"
    entries = build_layer_index(
        builds, cache_path=layer_cache, force_rebuild=False, verbose=False,
    )
    log.info(f"{split_name}: {len(entries)} kept entries from {len(builds)} build(s)")

    # group entries by build
    by_build: dict[str, list] = {b: [] for b in builds}
    for e in entries:
        by_build[e.build_id].append(e)

    agg_build_ids = []
    agg_cache_rows = []
    agg_defect_ratios = []

    for build_id in builds:
        bdir = cache_root / build_id
        ents = sorted(by_build[build_id], key=lambda x: x.layer_idx)
        layer_idxs = [e.layer_idx for e in ents]
        drs = [e.defect_ratio for e in ents]

        # Resume guard
        v0p = bdir / "visible_0.npy"
        v1p = bdir / "visible_1.npy"
        lbp = bdir / "label.npy"
        if (not rebuild and v0p.exists() and v1p.exists() and lbp.exists()
                and (bdir / "meta.npz").exists()):
            try:
                m = np.load(bdir / "meta.npz", allow_pickle=False)
                if len(m["orig_layer_idxs"]) == len(layer_idxs):
                    log.info(f"{build_id}: cache exists ({len(layer_idxs)} layers), skip")
                else:
                    raise ValueError("size mismatch")
            except Exception:
                log.warning(f"{build_id}: cache corrupted or stale, rebuilding")
                build_one(build_id, layer_idxs, drs, bdir, img_size, chunk_size, log)
        else:
            build_one(build_id, layer_idxs, drs, bdir, img_size, chunk_size, log)

        # aggregate
        n = len(layer_idxs)
        agg_build_ids.extend([build_id] * n)
        agg_cache_rows.extend(range(n))
        agg_defect_ratios.extend(drs)

    np.savez(
        cache_root / f"{split_name}_index.npz",
        build_ids=np.array(agg_build_ids),
        cache_rows=np.array(agg_cache_rows, dtype=np.int32),
        defect_ratios=np.array(agg_defect_ratios, dtype=np.float32),
    )
    log.info(f"{split_name}_index.npz saved ({len(agg_build_ids)} entries)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    ap.add_argument("--chunk-size", type=int, default=50,
                    help="HDF5 chunk read size (layers). 메모리 ≈ 3.4 GB/chunk @ default")
    ap.add_argument("--rebuild", action="store_true",
                    help="강제 재빌드 (기존 cache 무시)")
    ap.add_argument("--split", choices=["both", "train", "val"], default="both")
    args = ap.parse_args()

    log = setup_logger(rank=0, name="build_cache")
    cache_root = cache_dir_for(args.img_size)
    cache_root.mkdir(parents=True, exist_ok=True)
    log.info(f"cache root: {cache_root}")
    log.info(f"img_size={args.img_size}  chunk_size={args.chunk_size}  rebuild={args.rebuild}")

    t_all = time.time()
    if args.split in ("both", "train"):
        build_split("train", config.ORNL_TRAIN_BUILDS, cache_root,
                    args.img_size, args.chunk_size, args.rebuild, log)
    if args.split in ("both", "val"):
        build_split("val", config.ORNL_VAL_BUILDS, cache_root,
                    args.img_size, args.chunk_size, args.rebuild, log)
    log.info(f"ALL DONE in {(time.time()-t_all)/60:.1f} min")


if __name__ == "__main__":
    main()
