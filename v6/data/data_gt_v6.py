"""v6 사람 GT dataset — 변환해 둔 레이블 데이터셋을 읽는다 (PLAN_v6 §3, §10).

읽는 곳은 `ORNL_Data/DSCNN_Dataset_pbf-2026.11/` 다. Peregrine 주석 8 소스 125 레이어를 `pbf-2026.11` 로 옮겨
DIVA 내보내기 zip 을 푼 것과 같은 구성으로 저장한 것이다 (`data/convert_legacy.py` 가 만든다):

    visible/0/{id}.png, visible/1/{id}.png      (H, W) uint8
    annotations/{id}.npy                        (H, W) int16 — 클래스 id 0–9, 미레이블 -1
    index.csv, classes.json

v5 와 다른 점:
  - 클래스 번호를 바꾸는 단계가 로더에 없다 (v5 는 native → ORNL 12 → 11-class 를 로더가 했다). 라벨을 그대로 읽는다
  - 읽기 전에 데이터셋의 `classes.json` 이 config 의 클래스 10개와 같은지, 방향이 리코팅 축 = 가로인지 확인한다
  - fold 종류에 고정 목록 ("list") 이 더해졌다 — P0 의 inlayer 는 `splits/legacy_inlayer_v1.json` 의 22 레이어다
  - 표본에 도메인 · 그룹이 붙는다 (그룹 = Peregrine 소스. P1 에서 DIVA 공정이 더해진다)

v5 와 같은 점: 한 번 읽어 img_size 로 줄여 RAM 에 둔다 (GTPool. 이미지 BILINEAR, 라벨 NEAREST),
화소 수 통계는 학습 해상도 기준, 소스 통째 / 소스 안 레이어 fold.
"""
from __future__ import annotations

import csv
import json
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
from .. import config_v6 as cfg
from .augmentation import augment_v6
from .copy_paste import CPSupplierItem

_IMAGE_EXTS = (".png", ".tif", ".tiff", ".jpg", ".jpeg", ".bmp")


@dataclass(frozen=True)
class SampleSpecV6:
    domain: str          # cfg.DOMAIN_LEGACY (Peregrine 주석을 변환한 것). P1 에서 cfg.DOMAIN_DIVA 가 더해진다
    group: str           # sampler · 표의 묶음 단위: Peregrine 소스 이름 (index.csv 의 process_code)
    layer: int           # index.csv 의 layer (Peregrine 은 원본 파일 번호)
    sample_id: str       # 데이터셋 안의 파일 이름 (7자리 일련번호)
    img0_path: Path
    img1_path: Path
    ann_path: Path

    @property
    def key(self) -> str:
        """`<그룹>/<레이어 7자리>` — fold 기록 · 평가에서 표본을 가리키는 이름. v5 의 표본 이름과 같다."""
        return f"{self.group}/{self.layer:07d}"


# ---------------------------------------------------------------------------
# 데이터셋 읽기
# ---------------------------------------------------------------------------

def check_dataset_schema(root: Path) -> dict:
    """`classes.json` 이 config 의 pbf-2026.11 10-class 와 같은지 확인한다. 다르면 멈춘다."""
    path = root / "classes.json"
    if not path.is_file():
        raise FileNotFoundError(f"FATAL: {path} 없음 — 변환한 데이터셋이 아니다")
    meta = json.loads(path.read_text(encoding="utf-8"))
    ids = [c["id"] for c in meta.get("classes", [])]
    keys = [c["key"] for c in meta.get("classes", [])]
    if meta.get("schema") != cfg.SCHEMA_VERSION or ids != list(range(cfg.N_CLASSES_V6)) or keys != cfg.CLASS_KEYS_V6:
        raise RuntimeError(
            f"FATAL: {path} 의 클래스가 config_v6 ({cfg.SCHEMA_VERSION}) 와 다르다: schema={meta.get('schema')} keys={keys}"
        )
    if meta.get("unlabeled") != cfg.IGNORE_INDEX:
        raise RuntimeError(f"FATAL: {path} 의 미레이블 값이 {cfg.IGNORE_INDEX} 이 아니다: {meta.get('unlabeled')}")
    return meta


def _find_image(folder: Path, sample_id: str) -> Path | None:
    for ext in _IMAGE_EXTS:
        p = folder / f"{sample_id}{ext}"
        if p.is_file():
            return p
    return None


def enumerate_samples_v6(root: Path | None = None, domain: str = cfg.DOMAIN_LEGACY) -> list[SampleSpecV6]:
    """변환한 데이터셋의 표본을 `index.csv` 의 id 순으로 나열한다 (v5 의 표본 순서와 같다).

    방향이 리코팅 축 = 가로가 아닌 표본이나 두 사진이 뒤바뀐 표본 (swap_sd) 이 있으면 멈춘다 —
    돌리고 바꾸는 일은 DIVA 내보내기를 캐시로 만드는 단계 (P1) 의 몫이고, 이 로더는 하지 않는다.
    """
    root = Path(root) if root is not None else cfg.LEGACY_PBF_ROOT
    if not root.is_dir():
        raise FileNotFoundError(
            f"FATAL: {root} 없음 — Peregrine 주석을 먼저 변환한다 (호스트에서 `python -m DefSeg_AM.v6.data.convert_legacy`, CPU 약 3분)"
        )
    check_dataset_schema(root)
    with (root / "index.csv").open(encoding="utf-8-sig", newline="") as fh:
        rows = sorted(csv.DictReader(fh), key=lambda r: r["id"])
    if not rows:
        raise RuntimeError(f"FATAL: {root / 'index.csv'} 에 표본이 없다")

    samples: list[SampleSpecV6] = []
    for r in rows:
        sid = r["id"]
        axis = (r.get("recoat_axis") or "").strip()
        if axis != cfg.CANONICAL_RECOAT_AXIS:
            raise RuntimeError(
                f"FATAL: {root.name} 의 표본 {sid} 는 recoat_axis={axis!r} 다 — 이 로더는 리코팅 축이 가로인 데이터만 읽는다 (PLAN_v6 §4)"
            )
        if (r.get("swap_sd") or "0").strip() not in ("", "0"):
            raise RuntimeError(f"FATAL: {root.name} 의 표본 {sid} 는 swap_sd={r['swap_sd']!r} 다 — 이 로더는 두 사진을 바꾸지 않는다")
        ann = root / "annotations" / f"{sid}.npy"
        i0 = _find_image(root / "visible" / "0", sid)
        i1 = _find_image(root / "visible" / "1", sid)
        if not ann.is_file() or i0 is None or i1 is None:
            raise FileNotFoundError(f"FATAL: {root.name} 의 표본 {sid} 의 파일이 빠졌다 (annotations / visible/0 / visible/1)")
        samples.append(SampleSpecV6(
            domain=domain, group=r["process_code"], layer=int(r["layer"]), sample_id=sid,
            img0_path=i0, img1_path=i1, ann_path=ann,
        ))
    keys = [s.key for s in samples]
    if len(set(keys)) != len(keys):
        raise RuntimeError(f"FATAL: {root.name} 에 같은 (process_code, layer) 가 둘 이상 있다")
    return samples


def load_label(path: Path) -> np.ndarray:
    """`annotations/{id}.npy` → (H, W) int8. 값이 0–9 · -1 밖이면 멈춘다."""
    ann = np.load(path)
    if ann.ndim != 2:
        raise ValueError(f"FATAL: (H, W) 라벨이 아니다 (shape={ann.shape}): {path}")
    lo, hi = int(ann.min()), int(ann.max())
    if lo < cfg.IGNORE_INDEX or hi >= cfg.N_CLASSES_V6:
        raise ValueError(f"FATAL: 라벨 값이 0–{cfg.N_CLASSES_V6 - 1} · -1 밖이다 (min={lo}, max={hi}): {path}")
    return ann.astype(np.int8)


def load_image_uint8(path: Path) -> np.ndarray:
    """`visible/{0,1}/{id}.*` → (H, W) uint8. 8-bit 가 아닌 사진은 받지 않는다 (밝기를 임의로 늘리지 않으려는 것)."""
    img = Image.open(path)
    if img.mode in ("RGB", "RGBA", "P", "LA"):
        img = img.convert("L")
    arr = np.array(img)
    if arr.ndim != 2 or arr.dtype != np.uint8:
        raise ValueError(f"FATAL: (H, W) uint8 이미지가 아니다 (shape={arr.shape}, dtype={arr.dtype}): {path}")
    return arr


def load_sample_native(s: SampleSpecV6) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """표본 하나를 원본 해상도로 → (visible/0, visible/1, 라벨). 세 장의 크기가 다르면 멈춘다."""
    ann = load_label(s.ann_path)
    i0 = load_image_uint8(s.img0_path)
    i1 = load_image_uint8(s.img1_path)
    if i0.shape != ann.shape or i1.shape != ann.shape:
        raise ValueError(f"FATAL: {s.key} 의 두 사진과 라벨의 크기가 다르다: {i0.shape} {i1.shape} {ann.shape}")
    return i0, i1, ann


def count_pixels(label: np.ndarray, n_classes: int = cfg.N_CLASSES_V6) -> np.ndarray:
    """(H, W) label → (n_classes,) int64. 미레이블 (-1) 은 세지 않는다."""
    valid = label[label >= 0]
    return np.bincount(valid.astype(np.int64), minlength=n_classes)[:n_classes].astype(np.int64)


# ---------------------------------------------------------------------------
# GTPool — 전 표본을 한 번 읽어 img_size 로 줄여 RAM 에 둔다 (v5 와 같음)
# ---------------------------------------------------------------------------

class GTPool:
    """사람 GT 표본 전부를 img_size² 로 줄여 메모리에 올린다. 125 표본 × 1036² × 3 장 ≈ 0.4 GB.

    Attributes:
        i0, i1: list[(S, S) uint8]   visible/0, visible/1
        ann:    list[(S, S) int8]    10-class label (미레이블 = -1)
        pix:    (N, C) int64         학습 해상도 기준 클래스별 화소 수
        pix_native: (N, C) int64     native 해상도 기준 (기록용)
        unlabeled: (N,) int64        학습 해상도 기준 미레이블 화소 수
        native_hw: list[(H, W)]
    """

    def __init__(self, samples: list[SampleSpecV6], img_size: int, log=None):
        self.samples = samples
        self.img_size = img_size
        n = len(samples)
        self.i0: list[np.ndarray] = []
        self.i1: list[np.ndarray] = []
        self.ann: list[np.ndarray] = []
        self.native_hw: list[tuple[int, int]] = []
        self.pix = np.zeros((n, cfg.N_CLASSES_V6), dtype=np.int64)
        self.pix_native = np.zeros((n, cfg.N_CLASSES_V6), dtype=np.int64)
        self.unlabeled = np.zeros(n, dtype=np.int64)
        for i, s in enumerate(samples):
            i0, i1, ann = load_sample_native(s)
            self.native_hw.append((int(ann.shape[0]), int(ann.shape[1])))
            self.pix_native[i] = count_pixels(ann)
            ann_r = resize_label(ann, img_size)
            self.pix[i] = count_pixels(ann_r)
            self.unlabeled[i] = int((ann_r < 0).sum())
            i0 = resize_image_uint8(i0, img_size)
            i1 = resize_image_uint8(i1, img_size)
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

    def groups(self) -> list[str]:
        return [s.group for s in self.samples]


# ---------------------------------------------------------------------------
# fold 분할
# ---------------------------------------------------------------------------

def load_split_list(name: str) -> dict:
    """`splits/<name>` 의 고정 목록. {"val": [표본 이름, …], "expected": {…} (없을 수 있다)}."""
    path = cfg.SPLIT_DIR / name
    if not path.is_file():
        raise FileNotFoundError(f"FATAL: 분할 목록 {path} 없음")
    doc = json.loads(path.read_text(encoding="utf-8"))
    val = doc.get("val")
    if not isinstance(val, list) or not val or len(set(val)) != len(val):
        raise ValueError(f"FATAL: {path} 의 val 목록이 비었거나 중복이 있다")
    return doc


def split_by_fold(
    samples: list[SampleSpecV6],
    pix: np.ndarray,
    fold_spec: tuple | None,
    allow_missing: bool = False,
) -> tuple[list[int], list[int]]:
    """fold 정의 → (train_idx, val_idx). fold_spec 이 None 이면 전수 학습 (val 없음).

    ("list", 파일): 목록에 있는 표본이 val. 목록의 표본이 데이터셋에 없으면 멈춘다 (allow_missing 은 스모크용).
    ("source", name): 그 그룹의 표본 전부가 val.
    ("layers", n, k): 그룹마다 이름순 위치 j 에 대해 j % n == k 인 표본이 val.
        그 뒤 결함 클래스마다 (그 클래스가 있는 레이어가 2개 이상이면) val 과 train 양쪽에
        적어도 한 레이어씩 있도록 보정한다. 보정도 결정적이다 (화소 수 중앙값인 레이어를 옮긴다). v5 와 같은 규칙.
    """
    n_all = len(samples)
    if fold_spec is None:
        return list(range(n_all)), []

    kind = fold_spec[0]
    if kind == "list":
        keys = set(load_split_list(fold_spec[1])["val"])
        val = {i for i, s in enumerate(samples) if s.key in keys}
        missing = keys - {samples[i].key for i in val}
        if missing and not allow_missing:
            raise ValueError(f"FATAL: 분할 목록 {fold_spec[1]} 의 표본이 데이터셋에 없다: {sorted(missing)[:5]} …")
        if not val:
            raise ValueError(f"분할 목록 {fold_spec[1]} 의 표본이 하나도 없다")
    elif kind == "source":
        name = fold_spec[1]
        val = {i for i, s in enumerate(samples) if s.group == name}
        if not val:
            raise ValueError(f"fold source '{name}' 에 표본이 없다")
    elif kind == "layers":
        n, k = int(fold_spec[1]), int(fold_spec[2])
        val = set()
        pos_in_group: dict[str, int] = {}
        for i, s in enumerate(samples):
            j = pos_in_group.get(s.group, 0)
            pos_in_group[s.group] = j + 1
            if j % n == k:
                val.add(i)
        # 클래스 보정
        for _ in range(4):
            changed = False
            for c in cfg.DEFECT_IDS_V6:
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


def quick_subset(samples: list[SampleSpecV6], fold_spec: tuple | None, per_group: int = 3) -> list[SampleSpecV6]:
    """스모크용 — 그룹마다 per_group 레이어만 남긴다 (순서는 그대로).

    고정 목록 fold 는 목록의 레이어가 그룹마다 하나 들어가게 고른다 (앞에서부터 자르면 검증 레이어가 하나도 안 남는다).
    """
    val_keys: set[str] = set()
    if fold_spec is not None and fold_spec[0] == "list":
        val_keys = set(load_split_list(fold_spec[1])["val"])
    by_group: dict[str, list[SampleSpecV6]] = {}
    for s in samples:
        by_group.setdefault(s.group, []).append(s)
    keep: set[str] = set()
    for members in by_group.values():
        in_val = [s for s in members if s.key in val_keys][:1]
        rest = [s for s in members if s.key not in val_keys][: per_group - len(in_val)]
        keep.update(s.key for s in in_val + rest)
    return [s for s in samples if s.key in keep]


# ---------------------------------------------------------------------------
# Dataset
# ---------------------------------------------------------------------------

class DefSegGTDatasetV6(Dataset):
    """GTPool 의 일부 (indices) 를 보는 Dataset.

    training=True 면 augmentation (+ Copy-Paste) 을 건다. 증강 난수는 (seed, rank, worker)
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
            img0, img1, ann = augment_v6(
                img0, img1, ann, training=True,
                rng=self._get_rng(), cp_sampler=self.cp_sampler,
            )
        spec = self.pool.samples[i]
        return {
            "img0": torch.from_numpy(normalize_image(img0)),
            "img1": torch.from_numpy(normalize_image(img1)),
            "label": torch.from_numpy(ann.astype(np.int64)),
            "group": spec.group,
        }


def build_cp_supplier_items(
    pix_local: np.ndarray,
    classes: tuple[int, ...] = cfg.CP_CLASSES,
    min_component_px: int = cfg.CP_MIN_COMPONENT_PX,
) -> dict[int, list[CPSupplierItem]]:
    """Copy-Paste supplier index — 붙이는 클래스마다 그 클래스 화소가 threshold 이상인 학습 표본.

    Args:
        pix_local: (n_train, n_classes) — 학습 Dataset 의 위치 순서대로 정렬된 화소 수.

    Returns: {class_id: [CPSupplierItem(key=학습 Dataset 안의 위치), ...]}
    """
    out: dict[int, list[CPSupplierItem]] = {}
    for c in classes:
        idxs = np.where(pix_local[:, c] >= min_component_px)[0]
        out[int(c)] = [
            CPSupplierItem(key=int(j), class_id=int(c), n_pixels=int(pix_local[j, c]))
            for j in idxs
        ]
    return out
