"""학습된 생성 모델을 불러와 창을 그리는 함수 — 합성 세트를 만드는 단계가 쓰는 입구.

    model, diff, ck = load_generator(ckpt_path, device)
    g0, g1 = fill_windows(model, diff, i0, i1, region, label)          # 모두 numpy, uint8

그릴 자리 밖의 화소는 입력과 한 값도 다르지 않다 (모델의 출력이 아니라 입력을 그대로 돌려준다).
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .. import config_genaug as cfg
from ..models.diffusion import Diffusion, sample_fill
from ..models.unet import build_unet


def load_generator(path: Path | str, device: torch.device | str, use_ema: bool = True):
    """ckpt (`last.pt` 또는 `ema_step_*.pt`) → (모델, 확산 과정, ckpt 사전). 기본은 EMA 사본을 쓴다."""
    ck = torch.load(path, map_location="cpu", weights_only=False)
    model = build_unet(ck["model_cfg"])
    key = "ema" if (use_ema and "ema" in ck) else "model"
    model.load_state_dict(ck[key])
    model.to(device).eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model, Diffusion(ck["config"]["T"], device), ck


def to_uint8(x: torch.Tensor) -> np.ndarray:
    """[-1, 1] 텐서 → uint8."""
    return ((x.float().clamp(-1, 1) + 1.0) * 127.5).round().clamp(0, 255).to(torch.uint8).cpu().numpy()


@torch.no_grad()
def fill_windows(model, diff: Diffusion, i0: np.ndarray, i1: np.ndarray, region: np.ndarray, label: np.ndarray,
                 init0: np.ndarray | None = None, init1: np.ndarray | None = None,
                 steps: int = cfg.SAMPLE_STEPS, guidance: float = cfg.SAMPLE_CFG, strength: float = 1.0,
                 seed: int | None = None, batch: int = 32, amp: bool = False):
    """창 N 개를 그린다.

    i0, i1   (N, H, W) uint8   원본 창 (visible/0, visible/1)
    region   (N, H, W) bool    그릴 자리
    label    (N, H, W) uint8   라벨 맵 (0–9, "모름" = 10). 그릴 자리 밖의 값은 쓰지 않는다
    init0/1  (N, H, W) uint8   강도 < 1 일 때의 출발 그림 (진짜 조각을 붙인 창). 없으면 원본 창
    돌려주는 값: (g0, g1) — (N, H, W) uint8. 그릴 자리 밖은 i0, i1 과 같다
    """
    dev = next(model.parameters()).device
    gen = torch.Generator(device=dev).manual_seed(int(seed)) if seed is not None else None
    out0, out1 = np.empty_like(i0), np.empty_like(i1)
    for a in range(0, len(i0), batch):
        sl = slice(a, a + batch)
        x = torch.from_numpy(np.stack([i0[sl], i1[sl]], axis=1).astype(np.float32) / 127.5 - 1.0).to(dev)
        reg = torch.from_numpy(region[sl][:, None].astype(np.float32)).to(dev)
        lab = torch.from_numpy(label[sl].astype(np.int64)).to(dev)
        init = None
        if init0 is not None and init1 is not None:
            init = torch.from_numpy(np.stack([init0[sl], init1[sl]], axis=1).astype(np.float32) / 127.5 - 1.0).to(dev)
        g = to_uint8(sample_fill(model, diff, x, reg, lab, steps=steps, guidance=guidance, strength=strength,
                                 init=init, generator=gen, amp=amp))
        out0[sl] = np.where(region[sl], g[:, 0], i0[sl])
        out1[sl] = np.where(region[sl], g[:, 1], i1[sl])
    return out0, out1
