"""v4 Copy-Paste augmentation (CutMix segmentation 발전형) — PLAN_v4 §6.2.

rare class object 를 source layer 에서 잘라 target sample 에 paste.
visible/0, visible/1, label 모두 동기 paste. Gaussian feathering 으로 부드러운 경계.

사용 시나리오:
- Stage 1: supplier pool = ORNL HDF5 의 train layer 중 rare class 픽셀 ≥ threshold
- Stage 2: supplier pool = DSCNN_Dataset sample 중 rare class 픽셀 ≥ threshold

CopyPasteSampler 가 supplier pool 을 관리, paste() 가 sample 단위 augmentation.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Optional

import numpy as np
from scipy.ndimage import binary_dilation, gaussian_filter, label as cc_label

from .. import config_v4 as cfg


@dataclass
class CPSupplierItem:
    """한 supplier entry — source sample 의 identifier."""
    key: object         # 임의 타입 — build_id+row 또는 DSCNN sample 인덱스 등
    class_id: int       # 어느 rare class 의 supplier 인지
    n_pixels: int       # 해당 class 픽셀 수


class CopyPasteSampler:
    """rare class 별 supplier pool 을 관리하고 random source 추출.

    fetch_fn(key) → (i0_u8, i1_u8, ann) 콜백을 받아 lazy 로 source crop 추출.
    """

    def __init__(
        self,
        items_per_class: dict[int, list[CPSupplierItem]],
        fetch_fn: Callable[[object], tuple[np.ndarray, np.ndarray, np.ndarray]],
        rare_classes: tuple[int, ...] = cfg.CP_RARE_CLASSES_S1,
        prob: float = cfg.CP_PROB,
        min_component_px: int = cfg.CP_MIN_COMPONENT_PX,
        max_objects_per_paste: int = cfg.CP_MAX_OBJECTS_PER_PASTE,
        feather_sigma: float = cfg.CP_FEATHER_SIGMA,
        bbox_max_frac: float = cfg.CP_BBOX_MAX_FRAC,
    ):
        self.items_per_class = items_per_class
        self.fetch_fn = fetch_fn
        self.rare_classes = tuple(rare_classes)
        self.prob = prob
        self.min_component_px = min_component_px
        self.max_objects_per_paste = max_objects_per_paste
        self.feather_sigma = feather_sigma
        self.bbox_max_frac = bbox_max_frac

    def _pick_source(self, rng: np.random.Generator) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
        """random rare class → random supplier 추출 → fetch.

        Returns: (i0, i1, ann, class_id) 또는 None (supplier 없음)
        """
        # rare class 중 supplier 있는 것만
        avail = [c for c in self.rare_classes if self.items_per_class.get(c)]
        if not avail:
            return None
        c = int(rng.choice(avail))
        item = self.items_per_class[c][int(rng.integers(0, len(self.items_per_class[c])))]
        i0, i1, ann = self.fetch_fn(item.key)
        return i0, i1, ann, c

    def _extract_objects(
        self,
        ann_src: np.ndarray,
        class_id: int,
        rng: np.random.Generator,
    ) -> list[tuple[slice, slice, np.ndarray]]:
        """source ann 에서 class_id 의 connected component bbox + mask 추출.

        Returns: list of (slice_y, slice_x, mask) — bbox 안의 binary mask
        """
        H, W = ann_src.shape
        max_size = int(self.bbox_max_frac * min(H, W))
        comp_mask = (ann_src == class_id)
        if not comp_mask.any():
            return []
        lab, n = cc_label(comp_mask, structure=np.ones((3, 3), dtype=np.int8))
        out: list[tuple[slice, slice, np.ndarray]] = []
        sizes = np.bincount(lab.ravel())
        valid_idx = [k for k in range(1, n + 1) if sizes[k] >= self.min_component_px]
        if not valid_idx:
            return []
        rng.shuffle(valid_idx)
        for k in valid_idx[: self.max_objects_per_paste]:
            ys, xs = np.where(lab == k)
            y0, y1 = ys.min(), ys.max() + 1
            x0, x1 = xs.min(), xs.max() + 1
            if (y1 - y0) > max_size or (x1 - x0) > max_size:
                continue
            mask = (lab[y0:y1, x0:x1] == k)
            out.append((slice(y0, y1), slice(x0, x1), mask))
        return out

    def paste(
        self,
        i0_target: np.ndarray,
        i1_target: np.ndarray,
        ann_target: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """target 에 source 의 rare class object paste.

        Args:
            i0_target, i1_target: (H, W) uint8
            ann_target:           (H, W) int8

        Returns:
            (i0_aug, i1_aug, ann_aug) — paste 적용된 결과
        """
        if rng.random() >= self.prob:
            return i0_target, i1_target, ann_target

        src = self._pick_source(rng)
        if src is None:
            return i0_target, i1_target, ann_target
        i0_src, i1_src, ann_src, class_id = src

        # shape mismatch 시 source 를 target shape 로 resize (드물 — 같은 build_cache 라 보통 일치)
        if i0_src.shape != i0_target.shape:
            from PIL import Image
            H, W = i0_target.shape
            i0_src = np.array(Image.fromarray(i0_src).resize((W, H), Image.BILINEAR))
            i1_src = np.array(Image.fromarray(i1_src).resize((W, H), Image.BILINEAR))
            ann_src = np.array(Image.fromarray(ann_src.astype(np.uint8))
                               .resize((W, H), Image.NEAREST)).astype(np.int8)

        objs = self._extract_objects(ann_src, class_id, rng)
        if not objs:
            return i0_target, i1_target, ann_target

        H, W = i0_target.shape
        out_i0 = i0_target.copy()
        out_i1 = i1_target.copy()
        out_ann = ann_target.copy()

        for sl_y, sl_x, mask in objs:
            h, w = mask.shape
            # random paste 위치
            y0 = int(rng.integers(0, H - h + 1))
            x0 = int(rng.integers(0, W - w + 1))
            # source crop
            src_i0 = i0_src[sl_y, sl_x]
            src_i1 = i1_src[sl_y, sl_x]
            # alpha mask (Gaussian feathering)
            alpha = gaussian_filter(mask.astype(np.float32), sigma=self.feather_sigma)
            # clip 0..1, mask 가장자리 점진적 blend
            alpha = np.clip(alpha, 0.0, 1.0)
            # paste image (alpha blending)
            tgt_y, tgt_x = slice(y0, y0 + h), slice(x0, x0 + w)
            out_i0[tgt_y, tgt_x] = (
                src_i0.astype(np.float32) * alpha
                + out_i0[tgt_y, tgt_x].astype(np.float32) * (1.0 - alpha)
            ).clip(0, 255).astype(np.uint8)
            out_i1[tgt_y, tgt_x] = (
                src_i1.astype(np.float32) * alpha
                + out_i1[tgt_y, tgt_x].astype(np.float32) * (1.0 - alpha)
            ).clip(0, 255).astype(np.uint8)
            # paste label (binary — feathering 적용 안 함)
            out_ann[tgt_y, tgt_x] = np.where(mask, class_id, out_ann[tgt_y, tgt_x])

        return out_i0, out_i1, out_ann


def load_supplier_index(json_path: Path) -> dict[int, list[CPSupplierItem]]:
    """build_cache_v4 가 만든 rare_class_supplier.json 로드.

    각 entry 의 key = (build_id, cache_row) tuple.
    """
    with open(json_path) as f:
        data = json.load(f)
    out: dict[int, list[CPSupplierItem]] = {}
    for c_str, items in data["supplier"].items():
        c = int(c_str)
        out[c] = [
            CPSupplierItem(
                key=(it["build_id"], int(it["cache_row"])),
                class_id=c,
                n_pixels=int(it["pix"]),
            )
            for it in items
        ]
    return out
