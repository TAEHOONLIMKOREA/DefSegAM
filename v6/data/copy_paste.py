"""v6 Copy-Paste augmentation — v5 방식 + 위치 규칙 (PLAN_v6 §5.5).

드문 클래스의 객체를 다른 학습 레이어에서 잘라 target 표본에 붙인다.
visible/0, visible/1, label 을 함께 붙이고, 경계는 Gaussian feathering 으로 섞는다.
supplier pool 은 **학습 분할의 사람 GT** 뿐이다 (검증용으로 뺀 레이어는 들어가지 않는다).

v5 와 다른 점 — 위치 규칙.
v4 · v5 는 아무 자리에나 붙였다. 새 체계에서는 그러면 정의가 깨진다 (Printed 위에 붙인 Spatter 는 정의상 Over Melting 이다).
그래서 붙이는 클래스마다 "밑에 있어야 하는 클래스" 를 두고, 객체 mask 의 90% 이상이 그 클래스의 화소 위에 놓일 때만 붙인다:

    SP (7) · DB (9) · IS (2) · PS (3)   → Powder (0) 위
    OM (8) · SE (6)                     → Printed (1) 위
    RS (4) · RH (5)                     → 제한 없음

자리를 10번 뽑아도 조건을 못 채우면 그 객체는 붙이지 않는다. 표는 config_v6 의 CP_REQUIRED_UNDER 다.
밑의 라벨은 그때까지 붙인 것이 반영된 target 라벨이다 (먼저 붙인 객체 위에 겹쳐 붙지 않는다).

자리를 뽑는 법 (CP_ANCHOR_ON_REQUIRED): 제한이 있는 클래스는 밑에 있어야 하는 클래스의 화소 하나를 고르고
객체의 무게중심을 거기에 놓은 뒤 90% 조건을 검사한다. 이미지 전체에서 고르게 뽑으면 Printed 가 화소의 1할쯤이라
SE · OM 이 거의 붙지 않는다.
"""
from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from typing import Callable, Optional

import numpy as np
from scipy.ndimage import gaussian_filter, label as cc_label

from .. import config_v6 as cfg


@dataclass
class CPSupplierItem:
    """한 supplier entry — source sample 의 identifier."""
    key: object         # 학습 Dataset 안의 위치 (int)
    class_id: int       # 어느 클래스의 supplier 인지
    n_pixels: int       # 해당 class 픽셀 수


class CopyPasteSampler:
    """붙이는 클래스별 supplier pool 을 관리하고 random source 추출.

    fetch_fn(key) → (i0_u8, i1_u8, ann) 콜백을 받아 source 를 가져온다.

    counts: 이 객체가 (이 프로세스에서) 다룬 객체 수 — {"objects": 클래스별 고른 객체, "pasted": 클래스별 붙인 객체}.
            DataLoader worker 마다 따로 세므로 학습 로그에는 쓰지 않는다 (점검 스크립트용).
    """

    def __init__(
        self,
        items_per_class: dict[int, list[CPSupplierItem]],
        fetch_fn: Callable[[object], tuple[np.ndarray, np.ndarray, np.ndarray]],
        classes: tuple[int, ...] = cfg.CP_CLASSES,
        prob: float = cfg.CP_PROB,
        min_component_px: int = cfg.CP_MIN_COMPONENT_PX,
        max_objects_per_paste: int = cfg.CP_MAX_OBJECTS_PER_PASTE,
        feather_sigma: float = cfg.CP_FEATHER_SIGMA,
        bbox_max_frac: float = cfg.CP_BBOX_MAX_FRAC,
        required_under: dict[int, int] | None = None,
        region_min_frac: float = cfg.CP_REGION_MIN_FRAC,
        position_tries: int = cfg.CP_POSITION_TRIES,
        anchor_on_required: bool = cfg.CP_ANCHOR_ON_REQUIRED,
    ):
        self.items_per_class = items_per_class
        self.fetch_fn = fetch_fn
        self.classes = tuple(classes)
        self.prob = prob
        self.min_component_px = min_component_px
        self.max_objects_per_paste = max_objects_per_paste
        self.feather_sigma = feather_sigma
        self.bbox_max_frac = bbox_max_frac
        self.required_under = dict(cfg.CP_REQUIRED_UNDER if required_under is None else required_under)
        self.region_min_frac = region_min_frac
        self.position_tries = position_tries
        self.anchor_on_required = anchor_on_required
        self.counts = {"objects": Counter(), "pasted": Counter()}

    def _pick_source(self, rng: np.random.Generator) -> Optional[tuple[np.ndarray, np.ndarray, np.ndarray, int]]:
        """random class → random supplier 추출 → fetch.

        Returns: (i0, i1, ann, class_id) 또는 None (supplier 없음)
        """
        avail = [c for c in self.classes if self.items_per_class.get(c)]
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

    def _place(
        self,
        mask: np.ndarray,
        class_id: int,
        ann: np.ndarray,
        rng: np.random.Generator,
    ) -> Optional[tuple[int, int]]:
        """객체 (bbox 안의 mask) 를 붙일 자리 (y0, x0) 를 고른다. 위치 규칙을 못 채우면 None.

        ann 은 지금까지 붙인 것이 반영된 target 라벨이다.
        """
        H, W = ann.shape
        h, w = mask.shape
        req = self.required_under.get(class_id)
        if req is None:                                   # 제한 없는 클래스 — v5 와 같이 아무 자리
            return int(rng.integers(0, H - h + 1)), int(rng.integers(0, W - w + 1))

        need = self.region_min_frac * float(mask.sum())
        anchors = None
        if self.anchor_on_required:
            anchors = np.flatnonzero(ann == req)
            if anchors.size == 0:
                return None
            ys, xs = np.nonzero(mask)
            cy, cx = int(round(float(ys.mean()))), int(round(float(xs.mean())))
        for _ in range(self.position_tries):
            if anchors is not None:
                ay, ax = divmod(int(anchors[int(rng.integers(0, anchors.size))]), W)
                y0 = min(max(ay - cy, 0), H - h)
                x0 = min(max(ax - cx, 0), W - w)
            else:
                y0 = int(rng.integers(0, H - h + 1))
                x0 = int(rng.integers(0, W - w + 1))
            under = ann[y0:y0 + h, x0:x0 + w][mask]
            if float((under == req).sum()) >= need:
                return y0, x0
        return None

    def paste(
        self,
        i0_target: np.ndarray,
        i1_target: np.ndarray,
        ann_target: np.ndarray,
        rng: np.random.Generator,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
        """target 에 source 의 객체를 붙인다. 입력 배열은 고치지 않는다.

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

        # shape mismatch 시 source 를 target shape 로 resize (같은 pool 이라 보통 일치)
        if i0_src.shape != i0_target.shape:
            from PIL import Image
            H, W = i0_target.shape
            i0_src = np.array(Image.fromarray(i0_src).resize((W, H), Image.BILINEAR))
            i1_src = np.array(Image.fromarray(i1_src).resize((W, H), Image.BILINEAR))
            ann_src = np.array(Image.fromarray(ann_src.astype(np.int16))
                               .resize((W, H), Image.NEAREST)).astype(np.int8)

        objs = self._extract_objects(ann_src, class_id, rng)
        if not objs:
            return i0_target, i1_target, ann_target

        out_i0 = out_i1 = out_ann = None
        for sl_y, sl_x, mask in objs:
            self.counts["objects"][class_id] += 1
            pos = self._place(mask, class_id, ann_target if out_ann is None else out_ann, rng)
            if pos is None:                               # 위치 규칙을 못 채웠다 — 이 객체는 붙이지 않는다
                continue
            if out_ann is None:
                out_i0 = i0_target.copy()
                out_i1 = i1_target.copy()
                out_ann = ann_target.copy()
            y0, x0 = pos
            h, w = mask.shape
            # source crop
            src_i0 = i0_src[sl_y, sl_x]
            src_i1 = i1_src[sl_y, sl_x]
            # alpha mask (Gaussian feathering) — mask 가장자리 점진적 blend
            alpha = gaussian_filter(mask.astype(np.float32), sigma=self.feather_sigma)
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
            self.counts["pasted"][class_id] += 1

        if out_ann is None:                               # 하나도 못 붙였다
            return i0_target, i1_target, ann_target
        return out_i0, out_i1, out_ann
