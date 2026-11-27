"""DSCNN 의 입력 — 레이어를 원본 해상도 타일로 자르고 stack A · B · C · D 를 만든다 (논문 3.3 절).

레이어 하나 (사진 두 장 + 라벨) 는 GPU 에 올려 둔다 (`Layer`). 타일은 그 위에서 잘라 낸다 — DataLoader 를 쓰지 않는다.

    A  레이어 전체를 bilinear 로 32² 로 줄이고 대칭으로 2 화소 덧댐 (36²). 레이어마다 한 번 만든다
    B  원본 해상도 타일 T² + 둘레 5 화소 (사진의 가장자리에서는 대칭으로 접어 채운다)
    C  타일을 bilinear 로 128² 로 줄인 것 + 둘레 2 화소 (= 원본에서 2·T/128 화소. 같은 간격으로 132² 을 뽑는다)
    D  베드 중심 기준 (x, y) ∈ [−1, 1]

한 묶음 (`make_stacks`) 의 타일은 T 가 같아야 한다. 학습의 한 배치는 여러 소스의 타일을 섞으므로 T 가 같은 것끼리 나눠 부른다 (training/train.py).

밝기는 소스 (장비 · 재료) 마다 정해 둔 평균 ū 를 빼기만 한다 (논문 식 1 — 사진마다 따로 맞추지 않는다, 8-bit 단위 그대로).
증강 (학습): 밝기 이동은 A · B · C 에, 잡음은 A · C 에만.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import numpy as np
import torch
import torch.nn.functional as F

from .. import config_dscnn as cfg


@dataclass
class Layer:
    key: str
    group: str
    t: int                       # 타일 한 변 (원본 해상도)
    mean: float                  # 이 소스의 ū
    hw: tuple[int, int]
    pad: int                     # img 에 덧댄 화소 수
    img: torch.Tensor            # (2, H + 2·pad, W + 2·pad) uint8 — 대칭으로 덧댄 사진 두 장 (visible/0, visible/1)
    a: torch.Tensor              # (2, 36, 36) float32 — stack A (ū 를 뺀 것)
    label: torch.Tensor | None   # (H, W) int8, 미레이블 -1


def regional_margin(t: int) -> int:
    """stack C 의 둘레 2 화소가 원본에서 차지하는 폭 (올림) + bilinear 가 읽는 이웃 1 화소."""
    return int(math.ceil(cfg.REGIONAL_PAD * t / cfg.REGIONAL_SIZE)) + 1


def image_pad(t: int) -> int:
    return max(cfg.LOCAL_PAD, regional_margin(t))


def global_stack(i0: np.ndarray, i1: np.ndarray, mean: float) -> torch.Tensor:
    """레이어 전체 → (2, 36, 36) float32 (CPU). bilinear 로 32² (TensorFlow · OpenCV 의 bilinear 처럼 미리 흐리지 않는다) → 대칭 2 화소."""
    x = torch.from_numpy(np.stack([i0, i1]).astype(np.float32))[None]
    s = F.interpolate(x, size=(cfg.GLOBAL_SIZE, cfg.GLOBAL_SIZE), mode="bilinear", align_corners=False)[0].numpy()
    p = cfg.GLOBAL_PAD
    s = np.pad(s, ((0, 0), (p, p), (p, p)), mode="symmetric")
    return torch.from_numpy(s - np.float32(mean))


def make_layer(
    i0: np.ndarray, i1: np.ndarray, ann: np.ndarray | None, *,
    key: str, group: str, t: int, mean: float, device: torch.device,
) -> Layer:
    """원본 해상도의 사진 두 장 (uint8) 과 라벨 → GPU 에 올린 Layer."""
    h, w = i0.shape
    if i1.shape != (h, w) or (ann is not None and ann.shape != (h, w)):
        raise ValueError(f"FATAL: {key} 의 사진 · 라벨의 크기가 다르다")
    if h < t or w < t:
        raise ValueError(f"FATAL: {key} 의 사진 ({h}×{w}) 이 타일 ({t}) 보다 작다")
    p = image_pad(t)
    img = np.pad(np.stack([i0, i1]), ((0, 0), (p, p), (p, p)), mode="symmetric")
    return Layer(
        key=key, group=group, t=int(t), mean=float(mean), hw=(h, w), pad=p,
        img=torch.from_numpy(np.ascontiguousarray(img)).to(device),
        a=global_stack(i0, i1, mean).to(device),
        label=None if ann is None else torch.from_numpy(ann.astype(np.int8)).to(device),
    )


def tile_grid(h: int, w: int, t: int) -> np.ndarray:
    """고정 격자의 타일 왼쪽 위 좌표 (n, 2) = (y0, x0). 나누어 떨어지지 않으면 오른쪽 · 아래의 마지막 타일을 가장자리에 붙인다.

    순서는 위 → 아래, 왼쪽 → 오른쪽이다. 다시 이어 붙일 때 이 순서로 쓰면 가장자리 타일이 겹친 곳을 덮어쓴다 (논문).
    """
    def starts(n: int) -> list[int]:
        s = list(range(0, n - t + 1, t))
        if s[-1] + t < n:
            s.append(n - t)
        return s
    return np.array([(y, x) for y in starts(h) for x in starts(w)], dtype=np.int64)


_GRID_CACHE: dict[tuple[int, str], torch.Tensor] = {}


def _regional_grid(t: int, device: torch.device) -> torch.Tensor:
    """(1, 132, 132, 2) — (T + 2M)² 로 자른 조각에서 stack C 의 132² 을 읽는 자리 (grid_sample, align_corners=False)."""
    k = (t, str(device))
    if k not in _GRID_CACHE:
        m_int = regional_margin(t)
        scale = t / cfg.REGIONAL_SIZE
        n = cfg.REGIONAL_SIZE + 2 * cfg.REGIONAL_PAD
        i = torch.arange(n, dtype=torch.float32, device=device)
        pos = m_int - cfg.REGIONAL_PAD * scale + (i + 0.5) * scale - 0.5       # 조각 안의 화소 좌표
        norm = (pos + 0.5) / (t + 2 * m_int) * 2.0 - 1.0
        gy, gx = torch.meshgrid(norm, norm, indexing="ij")
        _GRID_CACHE[k] = torch.stack([gx, gy], dim=-1)[None]
    return _GRID_CACHE[k]


def make_stacks(
    layers: list[Layer], items: np.ndarray, *,
    noise_var: torch.Tensor | None = None, shift: torch.Tensor | None = None,
    generator: torch.Generator | None = None, with_a: bool = True, with_label: bool = True,
) -> dict[str, torch.Tensor]:
    """타일 묶음 → {"a", "b", "c", "d", "label"}.

    Args:
        items: (N, 3) = (layers 안의 위치, y0, x0). 한 묶음의 타일은 T 가 같아야 한다 (같은 소스).
        noise_var: (N,) 밝기 0–1 기준의 잡음 분산 (0 이면 없음). A · C 에만 넣는다.
        shift: (N,) 동적 범위에 대한 밝기 이동 (±0.1). A · B · C 에 넣는다.
    """
    first = layers[int(items[0, 0])]
    t, dev = first.t, first.img.device
    lp, m = cfg.LOCAL_PAD, regional_margin(t)
    bs, cs, ds, labs, means, a_s = [], [], [], [], [], []
    ar = torch.arange(t, dtype=torch.float32, device=dev) + 0.5
    for li, y0, x0 in items.tolist():
        L = layers[li]
        if L.t != t:
            raise ValueError("한 묶음에 타일 크기가 다른 레이어가 섞였다")
        p, (h, w) = L.pad, L.hw
        bs.append(L.img[:, p + y0 - lp: p + y0 + t + lp, p + x0 - lp: p + x0 + t + lp])
        cs.append(L.img[:, p + y0 - m: p + y0 + t + m, p + x0 - m: p + x0 + t + m])
        xs = ((x0 + ar) / w * 2.0 - 1.0)[None, :].expand(t, t)
        ys = ((y0 + ar) / h * 2.0 - 1.0)[:, None].expand(t, t)
        ds.append(torch.stack([xs, ys]))
        means.append(L.mean)
        if with_a:
            a_s.append(L.a)
        if with_label:
            labs.append(L.label[y0: y0 + t, x0: x0 + t])
    n = len(bs)
    mean = torch.tensor(means, dtype=torch.float32, device=dev)[:, None, None, None]
    b = torch.stack(bs).float() - mean
    c_src = torch.stack(cs).float() - mean
    c = F.grid_sample(c_src, _regional_grid(t, dev).expand(n, -1, -1, -1), mode="bilinear", padding_mode="border", align_corners=False)
    out = {"b": b, "c": c, "d": torch.stack(ds)}
    if with_a:
        out["a"] = torch.stack(a_s)
    if shift is not None:
        s = (shift.to(dev).float() * cfg.DYNAMIC_RANGE)[:, None, None, None]
        out["b"] = out["b"] + s
        out["c"] = out["c"] + s
        if with_a:
            out["a"] = out["a"] + s
    if noise_var is not None:
        sd = (noise_var.to(dev).float().sqrt() * cfg.DYNAMIC_RANGE)[:, None, None, None]
        out["c"] = out["c"] + torch.randn(out["c"].shape, device=dev, generator=generator) * sd
        if with_a:
            out["a"] = out["a"] + torch.randn(out["a"].shape, device=dev, generator=generator) * sd
    if with_label:
        out["label"] = torch.stack(labs).long()
    return out


class TileBank:
    """학습 타일의 목록 — 주석이 MIN_LABELED_FRAC 이상인 타일만.

    Attributes:
        items:  (n, 3) int64 = (layers 안의 위치, y0, x0)
        pix:    (n, C) int64  타일의 클래스별 화소 수
        group:  list[str]     타일의 소스
    """

    def __init__(self, layers: list[Layer], n_classes: int = cfg.N_CLASSES, min_frac: float = cfg.MIN_LABELED_FRAC):
        items, pix, group = [], [], []
        self.n_grid = 0
        for li, L in enumerate(layers):
            if L.label is None:
                raise ValueError(f"{L.key}: 라벨이 없는 레이어는 학습 타일이 될 수 없다")
            lab = L.label
            for y0, x0 in tile_grid(*L.hw, L.t).tolist():
                self.n_grid += 1
                tile = lab[y0: y0 + L.t, x0: x0 + L.t]
                valid = tile[tile >= 0].long()
                if valid.numel() < min_frac * L.t * L.t:
                    continue
                items.append((li, y0, x0))
                pix.append(torch.bincount(valid, minlength=n_classes)[:n_classes].cpu().numpy())
                group.append(L.group)
        if not items:
            raise ValueError("FATAL: 학습 타일이 하나도 없다")
        self.items = np.array(items, dtype=np.int64)
        self.pix = np.stack(pix).astype(np.int64)
        self.group = group
        self.layers = layers
        self.tile_t = np.array([layers[int(li)].t for li in self.items[:, 0]], dtype=np.int64)     # 타일마다의 T

    def __len__(self) -> int:
        return len(self.items)

    def class_px(self) -> np.ndarray:
        return self.pix.sum(axis=0)

    def epoch_batches(self, rng: np.random.Generator) -> list[tuple[np.ndarray, np.ndarray]]:
        """한 epoch = (타일 × 증강 9 가지) 를 한 번씩. 전부 섞은 뒤 차례로 배치에 담는다 — **한 배치에 여러 소스가 섞인다.**

        배치는 타일이 BATCH_MAX_TILES 장이 되거나 화소 수가 BATCH_PIXEL_BUDGET 을 넘기 직전에 끊는다.
        소스를 섞는 까닭: 한 배치를 한 소스로만 채우면 BatchNorm 이 학습 때는 그 소스의 통계를, 추론 때는 여러 소스가 섞인 이동 평균을 써서 어긋난다.

        Returns: [(타일 번호 (n,), 증강 번호 (n,) 0–8), …]
        """
        n_combo = len(cfg.NOISE_VARS) * len(cfg.SHIFT_FRACS)
        tile = np.repeat(np.arange(len(self)), n_combo)
        combo = np.tile(np.arange(n_combo), len(self))
        order = rng.permutation(len(tile))
        tile, combo = tile[order], combo[order]
        px = (self.tile_t[tile] ** 2).tolist()
        batches, s, n = [], 0, len(tile)
        while s < n:
            e, acc = s, 0
            while e < n and e - s < cfg.BATCH_MAX_TILES and (e == s or acc + px[e] <= cfg.BATCH_PIXEL_BUDGET):
                acc += px[e]
                e += 1
            batches.append((tile[s:e], combo[s:e]))
            s = e
        return batches

    def steps_per_epoch(self) -> int:
        """한 epoch 의 step 수 (섞는 순서에 따라 조금 달라지므로 정해 둔 순서로 한 번 담아 본 값)."""
        return len(self.epoch_batches(np.random.default_rng(0)))


def combo_to_aug(combo: np.ndarray) -> tuple[torch.Tensor, torch.Tensor]:
    """증강 번호 0–8 → (잡음 분산 (n,), 밝기 이동 (n,))."""
    nv = np.array(cfg.NOISE_VARS, dtype=np.float32)[combo // len(cfg.SHIFT_FRACS)]
    sh = np.array(cfg.SHIFT_FRACS, dtype=np.float32)[combo % len(cfg.SHIFT_FRACS)]
    return torch.from_numpy(nv), torch.from_numpy(sh)


def median_frequency_weights(class_px: np.ndarray) -> np.ndarray:
    """논문 식 6: w_k = Median({f}) / f_k. 화소가 없는 클래스는 0 (중앙값에서도 뺀다). 상한을 두지 않는다."""
    px = np.asarray(class_px, dtype=np.float64)
    f = px / max(px.sum(), 1.0)
    have = f > 0
    w = np.zeros_like(f)
    w[have] = np.median(f[have]) / f[have]
    return w.astype(np.float32)
