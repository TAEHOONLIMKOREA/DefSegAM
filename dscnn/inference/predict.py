"""DSCNN 추론 — 레이어를 타일로 잘라 원본 해상도의 예측을 다시 이어 붙인다 (논문 3.3 절).

global 갈래는 레이어마다 한 번만 돈다. 타일은 격자 순서로 쓰므로 가장자리에 붙인 타일이 겹친 곳을 덮어쓴다.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .. import config_dscnn as cfg
from ..data.tiles import Layer, make_layer, make_stacks, tile_grid
from ..models.dscnn import DSCNN


def photo_mean(photos: list[np.ndarray]) -> float:
    """사진들의 화소 평균 (ū). 크기가 다른 사진이 섞여도 화소 수로 가중한다."""
    total = sum(float(p.sum(dtype=np.float64)) for p in photos)
    count = sum(int(p.size) for p in photos)
    return total / max(count, 1)


def load_dscnn(ckpt_path: Path, device: torch.device) -> tuple[DSCNN, dict]:
    """ckpt → (모델 (eval), ckpt dict). ckpt["config"] 에 소스별 ū 와 타일 크기가 들어 있다."""
    ckpt = torch.load(ckpt_path, map_location=device, weights_only=False)
    c = ckpt.get("config", {}) or {}
    if c.get("arch") != "dscnn":
        raise ValueError(f"{ckpt_path}: DSCNN ckpt 가 아니다 (arch={c.get('arch')})")
    model = DSCNN(n_classes=c["n_classes"], in_ch=c.get("in_channels", cfg.IN_CHANNELS), fc_relu=bool(c.get("fc_relu", False))).to(device)
    model.load_state_dict(ckpt["model_state"])
    model.eval()
    print(f"loaded {ckpt_path}  (DSCNN, weights={ckpt.get('weights', '?')}, step={ckpt.get('step', '?')}, fold={c.get('fold')})")
    return model, ckpt


@torch.no_grad()
def predict_layer(model: DSCNN, layer: Layer, pixel_budget: int = cfg.INFER_PIXEL_BUDGET) -> torch.Tensor:
    """Layer → (H, W) long 예측 (원본 해상도, GPU)."""
    h, w = layer.hw
    t = layer.t
    grid = tile_grid(h, w, t)
    items = np.concatenate([np.zeros((len(grid), 1), dtype=np.int64), grid], axis=1)
    out = torch.full((h, w), -1, dtype=torch.long, device=layer.img.device)
    g = model.global_features(layer.a[None])
    step = max(1, pixel_budget // (t * t))
    for s in range(0, len(items), step):
        part = items[s: s + step]
        st = make_stacks([layer], part, with_a=False, with_label=False)
        pred = model(st["b"], st["c"], st["d"], g=g).argmax(dim=1)
        for k, (_, y0, x0) in enumerate(part.tolist()):
            out[y0: y0 + t, x0: x0 + t] = pred[k]
    if bool((out < 0).any()):
        raise RuntimeError(f"{layer.key}: 타일이 덮지 못한 화소가 있다")
    return out


@torch.no_grad()
def predict_photos(
    model: DSCNN, i0: np.ndarray, i1: np.ndarray, *, t: int, mean: float, device: torch.device, key: str = "",
) -> torch.Tensor:
    """원본 해상도의 사진 두 장 → (H, W) long 예측."""
    layer = make_layer(i0, i1, None, key=key, group="", t=t, mean=mean, device=device)
    return predict_layer(model, layer)
