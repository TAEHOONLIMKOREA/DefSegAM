"""v2 label cache 빌드 — v1 image cache 의 label.npy 를 v2 8-class 로 재매핑.

v1 image cache (visible_0.npy, visible_1.npy) 는 그대로 사용.
v1 label.npy 의 각 픽셀 값 (0..11) 을 ORNL_12_TO_NEW_8 로 재매핑하여 label_v2.npy 로 저장.

기존 v1 cache 가 없으면 ORNL HDF5 부터 재빌드해야 함 (build_cache_stage1 사용).

사용:
    python -m DefSeg_AM.v2.data.build_cache_v2 [--img-size 1036] [--rebuild]

산출:
    cache/resized_sz1036_v2/<build>/
    ├── label_v2.npy           (n_layers, IMG, IMG) int8
    └── meta.npz               (orig_layer_idxs, defect_ratios_v2)
    cache/resized_sz1036_v2/{train,val}_index.npz
"""
from __future__ import annotations

import argparse
import time
from pathlib import Path

import numpy as np

from ...common.utils.log import setup_logger
from .. import config_v2 as cfg


def _remap_label_arr(lab_v1: np.ndarray) -> np.ndarray:
    """v1 (0..11) int8 → v2 (0..7) int8.

    IGNORE 매핑된 class 의 픽셀은 -1 로.
    """
    # numpy LUT — 효율적 매핑
    lut = np.full(13, fill_value=cfg.IGNORE_INDEX, dtype=np.int8)  # index 0..11 + 1 buffer
    for old_id, new_id in cfg.ORNL_12_TO_NEW_8.items():
        lut[old_id] = new_id
    # -1 픽셀 (원래 IGNORE) 은 lut 적용 전에 보존
    out = np.where(lab_v1 >= 0, lut[np.clip(lab_v1, 0, 11)], cfg.IGNORE_INDEX)
    return out.astype(np.int8)


def remap_one_build(
    build_id: str,
    v1_dir: Path,
    v2_dir: Path,
    rebuild: bool,
    log,
) -> dict | None:
    """한 build 의 label.npy 를 v2 로 재매핑하여 저장.

    Returns:
        meta dict: {n_layers, defect_ratios_v2 (per layer, class 2..7 픽셀 비율)} 또는 None (skip)
    """
    v1_bdir = v1_dir / build_id
    v2_bdir = v2_dir / build_id
    v2_label = v2_bdir / "label_v2.npy"
    v2_meta = v2_bdir / "meta.npz"

    v1_label = v1_bdir / "label.npy"
    v1_meta = v1_bdir / "meta.npz"

    if not v1_label.exists():
        log.warning(f"{build_id}: v1 cache 없음 ({v1_label}). build_cache_stage1 먼저 실행 필요. skip.")
        return None

    if v2_label.exists() and v2_meta.exists() and not rebuild:
        log.info(f"{build_id}: v2 label cache 이미 존재, skip (--rebuild 로 강제 가능)")
        d = np.load(v2_meta, allow_pickle=False)
        return {
            "n_layers": int(d.get("n_layers", 0)) if "n_layers" in d.files else None,
            "defect_ratios_v2": d["defect_ratios_v2"],
            "orig_layer_idxs": d["orig_layer_idxs"],
        }

    v2_bdir.mkdir(parents=True, exist_ok=True)
    log.info(f"{build_id}: load v1 label memmap …")
    lab_v1_mm = np.load(v1_label, mmap_mode="r")   # (N, H, W) int8
    n, H, W = lab_v1_mm.shape

    # 출력 memmap 할당 (덮어쓰기)
    out_mm = np.lib.format.open_memmap(
        v2_label, mode="w+", dtype=np.int8, shape=(n, H, W),
    )

    # v1 meta 의 orig_layer_idxs 그대로 carry (build_layer_index 의 결과)
    v1_m = np.load(v1_meta, allow_pickle=False)
    orig_layer_idxs = v1_m["orig_layer_idxs"]

    # chunk 단위로 처리 — RAM 부담 줄이기
    chunk = 50
    defect_pix_per_layer = np.zeros(n, dtype=np.int64)
    total_pix = H * W
    t0 = time.time()
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = np.array(lab_v1_mm[start:end])           # (chunk, H, W) int8
        block_v2 = _remap_label_arr(block)
        out_mm[start:end] = block_v2
        # v2 defect 비율 계산 (class 2..7)
        for c in cfg.DEFECT_CLASS_INDICES_V2:
            defect_pix_per_layer[start:end] += (block_v2 == c).sum(axis=(1, 2))
        log.info(f"  {build_id} [{end:5d}/{n}] elapsed {(time.time()-t0)/60:.2f}m")
    out_mm.flush()
    del out_mm, lab_v1_mm

    defect_ratios_v2 = (defect_pix_per_layer / total_pix).astype(np.float32)

    np.savez(
        v2_meta,
        orig_layer_idxs=orig_layer_idxs,
        defect_ratios_v2=defect_ratios_v2,
        n_layers=np.array(n, dtype=np.int32),
    )
    log.info(f"{build_id}: done in {(time.time()-t0)/60:.1f} min")
    return {
        "n_layers": n,
        "defect_ratios_v2": defect_ratios_v2,
        "orig_layer_idxs": orig_layer_idxs,
    }


def build_split_index(
    split: str,
    builds: list[str],
    v1_dir: Path,
    v2_dir: Path,
    rebuild: bool,
    log,
) -> None:
    """주어진 split (train/val) 의 build 들에 대해 label 재매핑 + aggregate index 저장."""
    agg_build_ids: list[str] = []
    agg_cache_rows: list[int] = []
    agg_defect_ratios: list[float] = []

    for build_id in builds:
        meta = remap_one_build(build_id, v1_dir, v2_dir, rebuild, log)
        if meta is None:
            continue
        n = meta["n_layers"]
        agg_build_ids.extend([build_id] * n)
        agg_cache_rows.extend(range(n))
        agg_defect_ratios.extend(meta["defect_ratios_v2"].tolist())

    np.savez(
        v2_dir / f"{split}_index.npz",
        build_ids=np.array(agg_build_ids),
        cache_rows=np.array(agg_cache_rows, dtype=np.int32),
        defect_ratios=np.array(agg_defect_ratios, dtype=np.float32),
    )
    log.info(f"{split}_index.npz saved ({len(agg_build_ids)} entries)")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-size", type=int, default=cfg.IMG_SIZE)
    ap.add_argument("--rebuild", action="store_true",
                    help="강제 재빌드 (기존 v2 label cache 무시)")
    ap.add_argument("--split", choices=["both", "train", "val"], default="both")
    args = ap.parse_args()

    log = setup_logger(rank=0, name="build_cache_v2")
    v1_dir = cfg.v1_cache_dir(args.img_size)
    v2_dir = cfg.v2_cache_dir(args.img_size)
    v2_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"v1 image cache: {v1_dir}")
    log.info(f"v2 label cache: {v2_dir}")

    t_all = time.time()
    if args.split in ("both", "train"):
        build_split_index("train", cfg.ORNL_TRAIN_BUILDS, v1_dir, v2_dir, args.rebuild, log)
    if args.split in ("both", "val"):
        build_split_index("val", cfg.ORNL_VAL_BUILDS, v1_dir, v2_dir, args.rebuild, log)
    log.info(f"ALL DONE in {(time.time()-t_all)/60:.1f} min")


if __name__ == "__main__":
    main()
