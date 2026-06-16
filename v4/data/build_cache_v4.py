"""v4 label cache 빌드 — v1 image cache 의 label.npy 를 v4 11-class 로 재매핑.

v1 image cache (visible_0.npy, visible_1.npy) 는 그대로 사용.
v1 label.npy 의 각 픽셀 값 (0..11) 을 ORNL_12_TO_NEW_11 로 재매핑하여 label_v4.npy 로 저장.

추가 산출 (v4 신규):
- pix_per_layer.npy : (n_layers, 11) int64  — per-layer per-class 픽셀 카운트
- rare_class_supplier.json : 각 rare class 의 supplier layer 목록

기존 v1 cache 가 없으면 ORNL HDF5 부터 재빌드해야 함 (build_cache_stage1 사용).

사용:
    python -m DefSeg_AM.v4.data.build_cache_v4 [--img-size 1036] [--rebuild]
"""
from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np

from ...common.utils.log import setup_logger
from .. import config_v4 as cfg


def _remap_label_arr(lab_v1: np.ndarray) -> np.ndarray:
    """v1 (0..11) int8 → v4 (0..10) int8.

    IGNORE 매핑된 class 의 픽셀은 -1 로.
    """
    lut = np.full(13, fill_value=cfg.IGNORE_INDEX, dtype=np.int8)
    for old_id, new_id in cfg.ORNL_12_TO_NEW_11.items():
        lut[old_id] = new_id
    out = np.where(lab_v1 >= 0, lut[np.clip(lab_v1, 0, 11)], cfg.IGNORE_INDEX)
    return out.astype(np.int8)


def remap_one_build(
    build_id: str,
    v1_dir: Path,
    v4_dir: Path,
    rebuild: bool,
    log,
) -> dict | None:
    """한 build 의 label.npy 를 v4 11-class 로 재매핑 + per-layer per-class 픽셀 카운트.

    Returns:
        meta dict: {n_layers, defect_ratios_v4, pix_per_layer (n_layers, 11)} 또는 None
    """
    v1_bdir = v1_dir / build_id
    v4_bdir = v4_dir / build_id
    v4_label = v4_bdir / "label_v4.npy"
    v4_meta = v4_bdir / "meta.npz"
    v4_pix_per_layer = v4_bdir / "pix_per_layer.npy"

    v1_label = v1_bdir / "label.npy"
    v1_meta = v1_bdir / "meta.npz"

    if not v1_label.exists():
        log.warning(f"{build_id}: v1 cache 없음 ({v1_label}). build_cache_stage1 먼저 실행 필요. skip.")
        return None

    if (v4_label.exists() and v4_meta.exists() and v4_pix_per_layer.exists() and not rebuild):
        log.info(f"{build_id}: v4 label cache 이미 존재, skip (--rebuild 로 강제 가능)")
        d = np.load(v4_meta, allow_pickle=False)
        pix = np.load(v4_pix_per_layer, allow_pickle=False)
        return {
            "n_layers": int(d.get("n_layers", 0)) if "n_layers" in d.files else None,
            "defect_ratios_v4": d["defect_ratios_v4"],
            "orig_layer_idxs": d["orig_layer_idxs"],
            "pix_per_layer": pix,
        }

    v4_bdir.mkdir(parents=True, exist_ok=True)
    log.info(f"{build_id}: load v1 label memmap …")
    lab_v1_mm = np.load(v1_label, mmap_mode="r")
    n, H, W = lab_v1_mm.shape

    out_mm = np.lib.format.open_memmap(
        v4_label, mode="w+", dtype=np.int8, shape=(n, H, W),
    )

    v1_m = np.load(v1_meta, allow_pickle=False)
    orig_layer_idxs = v1_m["orig_layer_idxs"]

    chunk = 50
    pix_per_layer = np.zeros((n, cfg.N_CLASSES_V4), dtype=np.int64)
    defect_pix_per_layer = np.zeros(n, dtype=np.int64)
    total_pix = H * W
    t0 = time.time()
    for start in range(0, n, chunk):
        end = min(start + chunk, n)
        block = np.array(lab_v1_mm[start:end])
        block_v4 = _remap_label_arr(block)
        out_mm[start:end] = block_v4
        # per-layer per-class 픽셀 카운트
        for c in range(cfg.N_CLASSES_V4):
            cnt = (block_v4 == c).sum(axis=(1, 2))
            pix_per_layer[start:end, c] = cnt
            if c in cfg.DEFECT_CLASS_INDICES_V4:
                defect_pix_per_layer[start:end] += cnt
        log.info(f"  {build_id} [{end:5d}/{n}] elapsed {(time.time()-t0)/60:.2f}m")
    out_mm.flush()
    del out_mm, lab_v1_mm

    defect_ratios_v4 = (defect_pix_per_layer / total_pix).astype(np.float32)

    np.save(v4_pix_per_layer, pix_per_layer)
    np.savez(
        v4_meta,
        orig_layer_idxs=orig_layer_idxs,
        defect_ratios_v4=defect_ratios_v4,
        n_layers=np.array(n, dtype=np.int32),
    )
    log.info(f"{build_id}: done in {(time.time()-t0)/60:.1f} min")
    return {
        "n_layers": n,
        "defect_ratios_v4": defect_ratios_v4,
        "orig_layer_idxs": orig_layer_idxs,
        "pix_per_layer": pix_per_layer,
    }


def build_split_index(
    split: str,
    builds: list[str],
    v1_dir: Path,
    v4_dir: Path,
    rebuild: bool,
    log,
) -> tuple[list[str], list[int], np.ndarray]:
    """주어진 split 의 builds 에 대해 label + pix_per_layer 재빌드 + aggregate index 저장.

    Returns: (agg_build_ids, agg_cache_rows, agg_pix_per_layer)
    """
    agg_build_ids: list[str] = []
    agg_cache_rows: list[int] = []
    agg_defect_ratios: list[float] = []
    agg_pix_list: list[np.ndarray] = []

    for build_id in builds:
        meta = remap_one_build(build_id, v1_dir, v4_dir, rebuild, log)
        if meta is None:
            continue
        n = meta["n_layers"]
        agg_build_ids.extend([build_id] * n)
        agg_cache_rows.extend(range(n))
        agg_defect_ratios.extend(meta["defect_ratios_v4"].tolist())
        agg_pix_list.append(meta["pix_per_layer"])

    agg_pix = np.concatenate(agg_pix_list, axis=0) if agg_pix_list else np.zeros((0, cfg.N_CLASSES_V4), dtype=np.int64)

    np.savez(
        v4_dir / f"{split}_index.npz",
        build_ids=np.array(agg_build_ids),
        cache_rows=np.array(agg_cache_rows, dtype=np.int32),
        defect_ratios=np.array(agg_defect_ratios, dtype=np.float32),
        pix_per_layer=agg_pix,
    )
    log.info(f"{split}_index.npz saved ({len(agg_build_ids)} entries, pix shape {agg_pix.shape})")
    return agg_build_ids, agg_cache_rows, agg_pix


def build_rare_class_supplier(
    train_build_ids: list[str],
    train_cache_rows: list[int],
    train_pix: np.ndarray,
    v4_dir: Path,
    log,
) -> None:
    """Copy-Paste 용 supplier 인덱스 작성.

    각 rare class c 에 대해 `pix[ℓ, c] ≥ CP_MIN_COMPONENT_PX` 인 layer 목록.
    Stage 1 의 train split (Builds 2-5) 만 대상. Stage 2 의 supplier 는 별도 처리
    (DSCNN_Dataset sample 단위).
    """
    out_path = v4_dir / "rare_class_supplier.json"
    supplier: dict[str, list[dict]] = {}

    rare_all = set(cfg.CP_RARE_CLASSES_S1) | set(cfg.CP_RARE_CLASSES_S2)
    for c in sorted(rare_all):
        mask = train_pix[:, c] >= cfg.CP_MIN_COMPONENT_PX
        idxs = np.where(mask)[0].tolist()
        supplier[str(c)] = [
            {
                "build_id": train_build_ids[i],
                "cache_row": int(train_cache_rows[i]),
                "pix": int(train_pix[i, c]),
            }
            for i in idxs
        ]
        log.info(f"  rare class {c} ({cfg.ORNL_CLASS_NAMES_V4[c]}): {len(idxs)} supplier layers")

    with open(out_path, "w") as f:
        json.dump({
            "class_names": cfg.ORNL_CLASS_NAMES_V4,
            "rare_classes_s1": list(cfg.CP_RARE_CLASSES_S1),
            "rare_classes_s2": list(cfg.CP_RARE_CLASSES_S2),
            "min_component_px": cfg.CP_MIN_COMPONENT_PX,
            "supplier": supplier,
        }, f, indent=2)
    log.info(f"saved rare_class_supplier → {out_path}")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--img-size", type=int, default=cfg.IMG_SIZE)
    ap.add_argument("--rebuild", action="store_true",
                    help="강제 재빌드 (기존 v4 label cache + pix_per_layer 무시)")
    ap.add_argument("--split", choices=["both", "train", "val"], default="both")
    args = ap.parse_args()

    log = setup_logger(rank=0, name="build_cache_v4")
    v1_dir = cfg.v1_cache_dir(args.img_size)
    v4_dir = cfg.v4_cache_dir(args.img_size)
    v4_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"v1 image cache: {v1_dir}")
    log.info(f"v4 label cache: {v4_dir}")

    t_all = time.time()
    train_ids, train_rows, train_pix = None, None, None
    if args.split in ("both", "train"):
        train_ids, train_rows, train_pix = build_split_index(
            "train", cfg.ORNL_TRAIN_BUILDS, v1_dir, v4_dir, args.rebuild, log,
        )
    if args.split in ("both", "val"):
        build_split_index("val", cfg.ORNL_VAL_BUILDS, v1_dir, v4_dir, args.rebuild, log)

    if train_ids is not None and len(train_ids) > 0:
        log.info("building rare_class_supplier.json …")
        build_rare_class_supplier(train_ids, train_rows, train_pix, v4_dir, log)

    log.info(f"ALL DONE in {(time.time()-t_all)/60:.1f} min")


if __name__ == "__main__":
    main()
