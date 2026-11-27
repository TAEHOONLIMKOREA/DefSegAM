"""v5 사람 GT dataset — DSCNN_Dataset 8 소스 125 레이어 (PLAN_v5 §3–4).

  - 8 source (LPBF 6 + BJ 2, EBPBF 제외) 통합 enumerate          — v4 와 같음
  - native class → ORNL 12-class → v5 11-class (2단계 매핑)       — v4 와 같음
  - 한 번 읽어 img_size 로 줄인 뒤 RAM 에 둔다 (GTPool)           — v5 신규. step 기준 학습에서 디스크 I/O 를 없앤다
  - fold 로 학습 / 검증 분할 (소스 통째 또는 소스 안 레이어)       — v5 신규
  - 화소 수 통계는 학습 해상도 (img_size²) 기준                    — v5 변경 (v4 는 native 해상도)
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import torch
from PIL import Image
from torch.utils.data import Dataset

from ...common.data.image_utils import (
    normalize_image,
    resize_image_uint8,
    resize_label,
)
from .. import config_v5 as cfg
from .augmentation import augment_v5
from .copy_paste import CPSupplierItem


@dataclass(frozen=True)
class SampleSpecV5:
    source_name: str
    img0_path: Path
    img1_path: Path
    ann_path: Path
    mapping_key: str

    @property
    def key(self) -> str:
        """`<소스>/<레이어 파일 이름>` — fold 기록 · 평가에서 표본을 가리키는 이름."""
        return f"{self.source_name}/{self.ann_path.stem}"


def enumerate_samples_v5(
    sources: list[dict] | None = None,
    max_per_source: int | None = None,
) -> list[SampleSpecV5]:
    """8 source 의 (img0, img1, ann) 튜플 모두 나열. 소스 순서는 config, 소스 안은 파일 이름순.

    max_per_source: 스모크용 — 소스마다 앞에서 이만큼만.
    """
    if sources is None:
        sources = cfg.GT_SOURCES_V5
    samples: list[SampleSpecV5] = []
    for src in sources:
        root = src["root"]
        ann_dir = root / "annotations"
        v0_dir = root / "data" / "visible" / "0"
        v1_dir = root / "data" / "visible" / "1"
        if not ann_dir.is_dir():
            print(f"[v5/skip] {src['name']}: no annotations dir at {ann_dir}")
            continue
        n = 0
        for ann_file in sorted(ann_dir.glob("*.npy")):
            v0 = v0_dir / f"{ann_file.stem}.tif"
            v1 = v1_dir / f"{ann_file.stem}.tif"
            if not (v0.is_file() and v1.is_file()):
                continue
            samples.append(SampleSpecV5(
                source_name=src["name"],
                img0_path=v0, img1_path=v1, ann_path=ann_file,
                mapping_key=src["mapping_key"],
            ))
            n += 1
            if max_per_source is not None and n >= max_per_source:
                break
    return samples


def remap_label_v5(ann_native: np.ndarray, mapping_key: str) -> np.ndarray:
    """native class ID → v5 11-class. 2 단계 매핑 (v4 와 같은 표):
       native → ORNL 12-class (MATERIAL_TO_ORNL_V5) → v5 11-class (ORNL_12_TO_V5)
    """
    mapping = cfg.MATERIAL_TO_ORNL_V5[mapping_key]
    out = np.full(ann_native.shape, fill_value=cfg.IGNORE_INDEX, dtype=np.int8)
    for native_id, ornl_12_id in mapping.items():
        if ornl_12_id == cfg.IGNORE_INDEX:
            continue
        new_id = cfg.ORNL_12_TO_V5.get(ornl_12_id, cfg.IGNORE_INDEX)
        if new_id == cfg.IGNORE_INDEX:
            continue
        out[ann_native == native_id] = new_id
    return out


def load_image_uint8(path: Path) -> np.ndarray:
    """DSCNN_Dataset 의 visible tif → (H, W) uint8."""
    arr = np.array(Image.open(path))
    if arr.ndim != 2 or arr.dtype != np.uint8:
        raise ValueError(f"expected (H, W) uint8 image, got shape={arr.shape} dtype={arr.dtype}: {path}")
    return arr


def count_pixels(label: np.ndarray, n_classes: int = cfg.N_CLASSES_V5) -> np.ndarray:
    """(H, W) label → (n_classes,) int64. IGNORE (-1) 는 세지 않는다."""
    valid = label[label >= 0]
    return np.bincount(valid.astype(np.int64), minlength=n_classes)[:n_classes].astype(np.int64)


# ---------------------------------------------------------------------------
# GTPool — 전 표본을 한 번 읽어 img_size 로 줄여 RAM 에 둔다
# ---------------------------------------------------------------------------

class GTPool:
    """사람 GT 표본 전부를 img_size² 로 줄여 메모리에 올린다.

    v4 는 표본을 뽑을 때마다 native tif 를 다시 열어 줄였다 (epoch 당 500 표본이라 괜찮았다).
    v5 는 한 번 학습에 16 만 표본을 뽑으므로 처음에 한 번만 줄인다. 줄이는 방법은 v4 와 같다
    (이미지 BILINEAR, 라벨 NEAREST). 125 표본 × 1036² × 3 장 ≈ 0.4 GB.

    Attributes:
        i0, i1: list[(S, S) uint8]   visible/0, visible/1
        ann:    list[(S, S) int8]    v5 11-class label (IGNORE = -1)
        pix:    (N, C) int64         학습 해상도 기준 클래스별 화소 수
        pix_native: (N, C) int64     native 해상도 기준 (기록용)
        native_hw: list[(H, W)]
    """

    def __init__(self, samples: list[SampleSpecV5], img_size: int, log=None):
        self.samples = samples
        self.img_size = img_size
        n = len(samples)
        self.i0: list[np.ndarray] = []
        self.i1: list[np.ndarray] = []
        self.ann: list[np.ndarray] = []
        self.native_hw: list[tuple[int, int]] = []
        self.pix = np.zeros((n, cfg.N_CLASSES_V5), dtype=np.int64)
        self.pix_native = np.zeros((n, cfg.N_CLASSES_V5), dtype=np.int64)
        for i, s in enumerate(samples):
            ann = remap_label_v5(np.load(s.ann_path), s.mapping_key)
            self.native_hw.append((int(ann.shape[0]), int(ann.shape[1])))
            self.pix_native[i] = count_pixels(ann)
            ann_r = resize_label(ann, img_size)
            self.pix[i] = count_pixels(ann_r)
            i0 = resize_image_uint8(load_image_uint8(s.img0_path), img_size)
            i1 = resize_image_uint8(load_image_uint8(s.img1_path), img_size)
            # 원본을 실수로 고치면 다음 추출부터 오염된다 — 쓰기 금지로 잠근다
            for a in (i0, i1, ann_r):
                a.setflags(write=False)
            self.i0.append(i0)
            self.i1.append(i1)
            self.ann.append(ann_r)
            if log is not None and (i + 1) % 25 == 0:
                log.info(f"  GT pool: {i + 1}/{n} 표본 로드")

    def __len__(self) -> int:
        return len(self.samples)

    def source_names(self) -> list[str]:
        return [s.source_name for s in self.samples]


# ---------------------------------------------------------------------------
# fold 분할
# ---------------------------------------------------------------------------

def split_by_fold(
    samples: list[SampleSpecV5],
    pix: np.ndarray,
    fold_spec: tuple | None,
) -> tuple[list[int], list[int]]:
    """fold 정의 → (train_idx, val_idx). fold_spec 이 None 이면 전수 학습 (val 없음).

    ("source", name): 그 소스의 표본 전부가 val.
    ("layers", n, k): 소스마다 이름순 위치 j 에 대해 j % n == k 인 표본이 val.
        그 뒤 결함 클래스마다 (그 클래스가 있는 레이어가 2개 이상이면) val 과 train 양쪽에
        적어도 한 레이어씩 있도록 보정한다. 보정도 결정적이다 (화소 수 중앙값인 레이어를 옮긴다).
    """
    n_all = len(samples)
    if fold_spec is None:
        return list(range(n_all)), []

    kind = fold_spec[0]
    if kind == "source":
        name = fold_spec[1]
        val = {i for i, s in enumerate(samples) if s.source_name == name}
        if not val:
            raise ValueError(f"fold source '{name}' 에 표본이 없다")
    elif kind == "layers":
        n, k = int(fold_spec[1]), int(fold_spec[2])
        val = set()
        pos_in_source: dict[str, int] = {}
        for i, s in enumerate(samples):
            j = pos_in_source.get(s.source_name, 0)
            pos_in_source[s.source_name] = j + 1
            if j % n == k:
                val.add(i)
        # 클래스 보정
        for _ in range(4):
            changed = False
            for c in cfg.DEFECT_CLASS_INDICES_V5:
                has = [i for i in range(n_all) if pix[i, c] > 0]
                if len(has) < 2:
                    continue
                val_has = [i for i in has if i in val]
                train_has = [i for i in has if i not in val]
                if not val_has:
                    cand = sorted(train_has, key=lambda i: (pix[i, c], i))
                    val.add(cand[len(cand) // 2])
                    changed = True
                elif not train_has:
                    cand = sorted(val_has, key=lambda i: (pix[i, c], i))
                    val.discard(cand[len(cand) // 2])
                    changed = True
            if not changed:
                break
    else:
        raise ValueError(f"unknown fold kind: {fold_spec!r}")

    val_idx = sorted(val)
    train_idx = [i for i in range(n_all) if i not in val]
    if not train_idx:
        raise ValueError("fold 가 학습 표본을 하나도 남기지 않는다")
    return train_idx, val_idx


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DefSegGTDatasetV5(Dataset):
    """GTPool 의 일부 (indices) 를 보는 Dataset.

    training=True 면 v5 augmentation (+ Copy-Paste) 을 건다. 증강 난수는 (seed, rank, worker)
    로 고정해 같은 설정이면 같은 표본열이 나온다.
    """

    def __init__(
        self,
        pool: GTPool,
        indices: list[int],
        training: bool,
        cp_sampler=None,
        seed: int = 0,
        rank: int = 0,
    ):
        self.pool = pool
        self.indices = list(indices)
        self.training = training
        self.cp_sampler = cp_sampler
        self.seed = int(seed)
        self.rank = int(rank)
        self._rng: np.random.Generator | None = None

    def __len__(self) -> int:
        return len(self.indices)

    def _get_rng(self) -> np.random.Generator:
        if self._rng is None:
            info = torch.utils.data.get_worker_info()
            wid = info.id if info is not None else 0
            self._rng = np.random.default_rng([self.seed, self.rank, wid])
        return self._rng

    def fetch_local(self, j: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """Copy-Paste supplier 용: 이 Dataset 안의 위치 j → (i0, i1, ann). augmentation 없이 (읽기 전용)."""
        i = self.indices[int(j) % len(self.indices)]
        return self.pool.i0[i], self.pool.i1[i], self.pool.ann[i]

    def __getitem__(self, j: int) -> dict:
        i = self.indices[j]
        img0, img1, ann = self.pool.i0[i], self.pool.i1[i], self.pool.ann[i]
        if self.training:
            img0, img1, ann = augment_v5(
                img0, img1, ann, training=True,
                rng=self._get_rng(), cp_sampler=self.cp_sampler,
            )
        spec = self.pool.samples[i]
        return {
            "img0": torch.from_numpy(normalize_image(img0)),
            "img1": torch.from_numpy(normalize_image(img1)),
            "label": torch.from_numpy(ann.astype(np.int64)),
            "source": spec.source_name,
        }


def build_cp_supplier_items(
    pix_local: np.ndarray,
    rare_classes: tuple[int, ...] = cfg.CP_RARE_CLASSES,
    min_component_px: int = cfg.CP_MIN_COMPONENT_PX,
) -> dict[int, list[CPSupplierItem]]:
    """Copy-Paste supplier index — rare class 마다 그 클래스 화소가 threshold 이상인 학습 표본.

    Args:
        pix_local: (n_train, n_classes) — 학습 Dataset 의 위치 순서대로 정렬된 화소 수.

    Returns: {class_id: [CPSupplierItem(key=학습 Dataset 안의 위치), ...]}
    """
    out: dict[int, list[CPSupplierItem]] = {}
    for c in rare_classes:
        idxs = np.where(pix_local[:, c] >= min_component_px)[0]
        out[int(c)] = [
            CPSupplierItem(key=int(j), class_id=int(c), n_pixels=int(pix_local[j, c]))
            for j in idxs
        ]
    return out
