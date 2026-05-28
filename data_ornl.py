"""Stage 1 dataset: ORNL → DSCNN pred (12-class) 로 KD pretrain.

PLAN.md §3.1 / §3.5 / §5.1 참조. 두 가지 dataset 제공:

1. DefSegORNLDataset (원본 HDF5 직접) — 디버깅·정성 확인 용도
2. DefSegORNLCachedDataset (사전 resize+uint8 cache memmap) — 학습 진짜 사용

학습은 반드시 사전 cache 빌드 후 (2) 를 사용:
    python -m DefSeg_AM.build_cache_stage1

- 입력: visible/0, visible/1 (각 grayscale → 3-channel replicate, ImageNet 정규화)
- 라벨: slices/segmentation_results/{0..11} → argmax (큰 ID = 결함이 작은 ID 를 덮어쓰도록)
- 필터링:
    1) 안쪽 5%~95% layer 만 사용
    2) part_ids 가 전부 0 인 layer skip
    3) defect (class 2..11) 픽셀이 0 인 layer 제외
- per-layer defect ratio 를 사전 계산해서 WeightedRandomSampler 의 weight 로 사용
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import h5py
import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from . import config


# ---------------------------------------------------------------------------
# 공용 유틸 (seung_dscnn 에서 carry-over, PLAN §6.1)
# ---------------------------------------------------------------------------

def normalize_image(img_uint8: np.ndarray) -> np.ndarray:
    """uint8 (H, W) grayscale → (3, H, W) float32, ImageNet 정규화."""
    arr = img_uint8.astype(np.float32) / 255.0
    arr = np.stack([arr, arr, arr], axis=0)
    mean = np.array(config.IMAGENET_MEAN, dtype=np.float32)[:, None, None]
    std = np.array(config.IMAGENET_STD, dtype=np.float32)[:, None, None]
    return (arr - mean) / std


def resize_image_uint8(arr: np.ndarray, size: int) -> np.ndarray:
    return np.array(Image.fromarray(arr).resize((size, size), Image.BILINEAR))


def resize_label(ann: np.ndarray, size: int) -> np.ndarray:
    # int8 은 PIL 이 직접 지원 못 함 → int16 우회
    return np.array(
        Image.fromarray(ann.astype(np.int16)).resize((size, size), Image.NEAREST)
    ).astype(np.int8)


def ornl_image_to_uint8(img: np.ndarray) -> np.ndarray:
    """ORNL float32 → uint8. percentile (1, 99) 기반 per-image normalize."""
    if img.dtype == np.uint8:
        return img
    nz = img[img > 0] if (img > 0).any() else img
    lo, hi = np.percentile(nz, [1, 99])
    if hi <= lo:
        hi = lo + 1
    norm = np.clip((img - lo) / (hi - lo), 0, 1)
    return (norm * 255).astype(np.uint8)


def ornl_segmentation_argmax(
    seg_grp: h5py.Group, layer_idx: int, n_classes: int = config.N_CLASSES,
) -> np.ndarray:
    """ORNL 의 n_classes boolean mask → argmax single-label.

    겹친 픽셀은 ID 큰 쪽 (결함) 이 작은 쪽 (Powder/Printed) 을 덮어씀.
    어떤 mask 에도 속하지 않으면 -1 (IGNORE).
    """
    sample = seg_grp["0"][layer_idx]
    H, W = sample.shape
    out = np.full((H, W), config.IGNORE_INDEX, dtype=np.int8)
    for c in range(n_classes):
        m = seg_grp[str(c)][layer_idx]
        out[m] = c
    return out


# ---------------------------------------------------------------------------
# Layer index 빌드 (PLAN §3.1 "per-layer defect ratio 사전 계산")
# ---------------------------------------------------------------------------

@dataclass
class LayerEntry:
    build_id: str       # "B1".."B5"
    layer_idx: int      # HDF5 layer index
    defect_ratio: float # sum(class 2..11 mask) / (H*W) — oversampling weight 용


def build_layer_index(
    builds: list[str],
    cache_path: Path | None = None,
    force_rebuild: bool = False,
    verbose: bool = True,
) -> list[LayerEntry]:
    """주어진 빌드들의 유효 layer 를 enumerate + defect_ratio 계산.

    캐시 (npz) 가 있으면 그걸 로드, 없으면 빌드하면서 저장.
    """
    if cache_path is not None and cache_path.exists() and not force_rebuild:
        if verbose:
            print(f"[layer-index] load cache {cache_path}")
        d = np.load(cache_path, allow_pickle=False)
        # 캐시는 같은 build 조합을 가정 — 단순 sanity check
        cached_builds = set(d["build_ids"].tolist())
        if cached_builds == set(builds):
            entries = [
                LayerEntry(b, int(li), float(dr))
                for b, li, dr in zip(d["build_ids"], d["layer_idxs"], d["defect_ratios"])
            ]
            if verbose:
                print(f"[layer-index] {len(entries)} entries from cache")
            return entries
        else:
            if verbose:
                print(f"[layer-index] cache build set mismatch ({cached_builds} vs {set(builds)}) — rebuild")

    entries: list[LayerEntry] = []
    n_dropped_powderonly = 0
    n_dropped_emptyparts = 0
    for build_id in builds:
        hdf5_path = config.ORNL_HDF5_DIR / config.ORNL_BUILD_FILES[build_id]
        if verbose:
            print(f"[layer-index] scan {build_id}: {hdf5_path.name}")
        with h5py.File(hdf5_path, "r") as f:
            vis0 = f["slices/camera_data/visible/0"]
            part_ids = f["slices/part_ids"]
            seg = f["slices/segmentation_results"]
            n_layers = vis0.shape[0]
            lo = int(n_layers * config.ORNL_LAYER_LO_FRAC)
            hi = int(n_layers * config.ORNL_LAYER_HI_FRAC)

            # defect masks (class 2..11) 의 sum 을 layer 별로 계산 — chunked read
            H, W = seg["0"].shape[1:]
            total_pix = H * W
            for li in range(lo, hi):
                # part_ids check
                if part_ids[li].max() == 0:
                    n_dropped_emptyparts += 1
                    continue
                # defect pixel count
                defect_count = 0
                for c in config.DEFECT_CLASS_INDICES:
                    defect_count += int(seg[str(c)][li].sum())
                if defect_count < config.DEFECT_PIXEL_MIN:
                    n_dropped_powderonly += 1
                    continue
                entries.append(LayerEntry(
                    build_id=build_id,
                    layer_idx=li,
                    defect_ratio=defect_count / total_pix,
                ))

    if verbose:
        print(f"[layer-index] kept {len(entries)} layers "
              f"(dropped {n_dropped_emptyparts} empty-parts, "
              f"{n_dropped_powderonly} powder/printed-only)")

    if cache_path is not None:
        cache_path.parent.mkdir(parents=True, exist_ok=True)
        np.savez(
            cache_path,
            build_ids=np.array([e.build_id for e in entries]),
            layer_idxs=np.array([e.layer_idx for e in entries], dtype=np.int32),
            defect_ratios=np.array([e.defect_ratio for e in entries], dtype=np.float32),
        )
        if verbose:
            print(f"[layer-index] saved cache {cache_path}")
    return entries


# ---------------------------------------------------------------------------
# Stage 1 Dataset
# ---------------------------------------------------------------------------

class DefSegORNLDataset(Dataset):
    """ORNL HDF5 layer-wise. DSCNN pred 를 argmax 정답으로 반환.

    워커별로 HDF5 를 lazy open (h5py 는 fork-safe 아님 → __getitem__ 안에서 open).
    """

    def __init__(
        self,
        entries: list[LayerEntry],
        img_size: int = config.IMG_SIZE,
        training: bool = True,
    ):
        self.entries = entries
        self.img_size = img_size
        self.training = training
        # 워커별 HDF5 핸들 캐시 (per-process)
        self._hfile_cache: dict[str, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.entries)

    def _open(self, build_id: str) -> h5py.File:
        if build_id not in self._hfile_cache:
            self._hfile_cache[build_id] = h5py.File(
                config.ORNL_HDF5_DIR / config.ORNL_BUILD_FILES[build_id], "r",
                swmr=False,
            )
        return self._hfile_cache[build_id]

    def __getitem__(self, idx: int) -> dict:
        e = self.entries[idx]
        f = self._open(e.build_id)
        vis0 = f["slices/camera_data/visible/0"][e.layer_idx]
        vis1 = f["slices/camera_data/visible/1"][e.layer_idx]
        seg = f["slices/segmentation_results"]

        i0 = ornl_image_to_uint8(vis0)
        i1 = ornl_image_to_uint8(vis1)
        ann = ornl_segmentation_argmax(seg, e.layer_idx)

        sz = self.img_size
        i0 = resize_image_uint8(i0, sz)
        i1 = resize_image_uint8(i1, sz)
        ann = resize_label(ann, sz)

        # brightness jitter only (recoater 방향성 보존 — flip/rotation 금지)
        if self.training and np.random.random() < 0.5:
            scale = np.random.uniform(0.85, 1.15)
            i0 = np.clip(i0.astype(np.float32) * scale, 0, 255).astype(np.uint8)
            i1 = np.clip(i1.astype(np.float32) * scale, 0, 255).astype(np.uint8)

        return {
            "img0": torch.from_numpy(normalize_image(i0)),
            "img1": torch.from_numpy(normalize_image(i1)),
            "label": torch.from_numpy(ann.astype(np.int64)),
            "build_id": e.build_id,
            "layer_idx": e.layer_idx,
        }


# ---------------------------------------------------------------------------
# Stage 1 Dataset — pre-built resize cache (학습용)
# ---------------------------------------------------------------------------

class DefSegORNLCachedDataset(Dataset):
    """Sets read from `build_cache_stage1.py` 가 만든 memmap .npy.

    학습 batch 당 처리: memmap slice → np.array(copy) → ImageNet norm. PIL 호출 없음.
    Worker process 마다 memmap 을 lazy open (fork-safe).

    cache 구조 (build_cache_stage1.py 참조):
        cache_root/
        ├── {split}_index.npz  : build_ids[], cache_rows[], defect_ratios[]
        └── <build_id>/{visible_0.npy, visible_1.npy, label.npy}
    """

    def __init__(
        self,
        cache_root: Path,
        split: str,                              # 'train' or 'val'
        img_size: int = config.IMG_SIZE,
        training: bool = True,
    ):
        self.cache_root = Path(cache_root)
        self.split = split
        self.img_size = img_size
        self.training = training

        idx_path = self.cache_root / f"{split}_index.npz"
        if not idx_path.exists():
            raise FileNotFoundError(
                f"Cache index not found: {idx_path}\n"
                "Run `python -m DefSeg_AM.build_cache_stage1` first."
            )
        d = np.load(idx_path, allow_pickle=False)
        self.build_ids = d["build_ids"]                  # (N,) str
        self.cache_rows = d["cache_rows"]                # (N,) int32
        self.defect_ratios = d["defect_ratios"].astype(np.float32)
        self._memmaps: dict[str, dict[str, np.memmap]] = {}

    def __len__(self) -> int:
        return len(self.build_ids)

    def _open(self, build_id: str) -> dict[str, np.memmap]:
        if build_id not in self._memmaps:
            bdir = self.cache_root / build_id
            self._memmaps[build_id] = {
                "v0": np.load(bdir / "visible_0.npy", mmap_mode="r"),
                "v1": np.load(bdir / "visible_1.npy", mmap_mode="r"),
                "lb": np.load(bdir / "label.npy", mmap_mode="r"),
            }
        return self._memmaps[build_id]

    def __getitem__(self, idx: int) -> dict:
        build = str(self.build_ids[idx])
        row = int(self.cache_rows[idx])
        m = self._open(build)
        # np.array(...) 로 memmap slice 를 즉시 RAM 으로 복사 (DataLoader transfer 효율 ↑)
        i0 = np.array(m["v0"][row])     # (img_size, img_size) uint8
        i1 = np.array(m["v1"][row])
        ann = np.array(m["lb"][row])    # int8

        # brightness jitter (training, recoater 방향성 보존 — flip/rotation 금지)
        if self.training and np.random.random() < 0.5:
            scale = np.random.uniform(0.85, 1.15)
            i0 = np.clip(i0.astype(np.float32) * scale, 0, 255).astype(np.uint8)
            i1 = np.clip(i1.astype(np.float32) * scale, 0, 255).astype(np.uint8)

        return {
            "img0": torch.from_numpy(normalize_image(i0)),
            "img1": torch.from_numpy(normalize_image(i1)),
            "label": torch.from_numpy(ann.astype(np.int64)),
            "build_id": build,
            "cache_row": row,
        }


def estimate_class_counts_cached(
    dataset: DefSegORNLCachedDataset,
    n_sample: int = 200,
    seed: int = 0,
) -> np.ndarray:
    """Cached dataset 의 라벨 일부를 sample 해서 class count 추정.

    원본 HDF5 기반 대비 ~100× 빠름 (PIL/percentile 없음, label 이 이미 resize 됨).
    """
    rng = np.random.default_rng(seed)
    n = len(dataset)
    idxs = rng.choice(n, size=min(n_sample, n), replace=False)
    counts = np.zeros(config.N_CLASSES, dtype=np.int64)
    for i in idxs:
        build = str(dataset.build_ids[i])
        row = int(dataset.cache_rows[i])
        m = dataset._open(build)
        lab = np.array(m["lb"][row])
        valid = lab[lab >= 0]
        u, c = np.unique(valid, return_counts=True)
        for k, v in zip(u, c):
            counts[k] += v
    return counts
