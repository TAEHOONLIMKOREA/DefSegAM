"""레이어 한 장을 통째로 그린다 — 라벨 맵 + 사진 가장자리 → 1036² 사진 두 장 (v7/PLAN_v7.md §2.1).

진짜 화소는 사진 가장자리 (5%, 52 px) 뿐이다. 안쪽은 256 창을 이어 그린다: 창 하나를 그릴 때 이미 그려진 화소
(진짜 가장자리와 먼저 그린 이웃 창) 를 "아는 화소" 로 주고 나머지를 그린다. 창은 이웃과 64 px 이상 겹친다.

그리는 순서
  spiral  진짜 가장자리에 닿은 바깥 창부터 안쪽으로 (기본값). 어느 창도 진짜 화소에서 두 창 넘게 떨어지지 않고,
          마지막 창 (가운데) 은 사방이 이미 그려진 채로 메운다
  raster  왼쪽 위에서 줄을 따라. 2026-11-18 의 점검이 쓴 순서다

여러 레이어를 한 묶음으로 그린다 (창의 자리가 같으므로 창마다 레이어를 묶는다). 잡음은 레이어마다 따로 뽑아
묶음을 어떻게 짜도 한 레이어의 잡음이 같다.
"""
from __future__ import annotations

import math

import numpy as np
import torch

from .. import config_genaug as cfg
from ..models.diffusion import Diffusion, sample_fill
from ..sampling.fill import to_uint8

ORDERS = ("spiral", "raster")


def frame_px(img_size: int) -> int:
    """사진 가장자리의 진짜 화소 폭."""
    return int(round(cfg.EDGE_FRAC * img_size))


def window_positions(img_size: int, edge: int, win: int = cfg.WIN, stride: int = cfg.SET_STRIDE,
                     overlap_frame: int = cfg.BORDER) -> list[int]:
    """창의 왼쪽 위 좌표 (한 축). 첫 창과 끝 창은 진짜 가장자리를 overlap_frame 만큼 물고, 그 사이를 고르게 나눈다."""
    first, last = edge - overlap_frame, img_size - edge + overlap_frame - win
    if last <= first:
        return [first]
    n = int(math.ceil((last - first) / stride)) + 1
    return [int(round(v)) for v in np.linspace(first, last, n)]


def window_order(n: int, order: str = cfg.SET_ORDER) -> list[tuple[int, int]]:
    """n × n 창을 그리는 순서 — (행, 열) 의 목록."""
    if order == "raster":
        return [(i, j) for i in range(n) for j in range(n)]
    if order != "spiral":
        raise ValueError(f"order 는 {ORDERS} 가운데 하나다: {order!r}")
    out: list[tuple[int, int]] = []
    for r in range((n + 1) // 2):                  # 바깥 고리부터. 고리 안에서는 시계 방향
        a, b = r, n - 1 - r
        if a == b:
            out.append((a, a))
            break
        out += [(a, j) for j in range(a, b + 1)]
        out += [(i, b) for i in range(a + 1, b + 1)]
        out += [(b, j) for j in range(b - 1, a - 1, -1)]
        out += [(i, a) for i in range(b - 1, a, -1)]
    return out


def owner_map(img_size: int, stride: int = cfg.SET_STRIDE, order: str = cfg.SET_ORDER, win: int = cfg.WIN) -> np.ndarray:
    """화소마다 그 화소를 그린 창의 번호 (그린 순서, 0 부터). 진짜 가장자리는 -1. 이음매를 재는 점검이 쓴다."""
    E = frame_px(img_size)
    own = np.full((img_size, img_size), -2, dtype=np.int16)
    own[:E], own[-E:], own[:, :E], own[:, -E:] = -1, -1, -1, -1
    pos = window_positions(img_size, E, win, stride)
    for k, (iy, ix) in enumerate(window_order(len(pos), order)):
        sub = own[pos[iy]:pos[iy] + win, pos[ix]:pos[ix] + win]
        sub[sub == -2] = k
    assert int(own.min()) >= -1
    return own


@torch.no_grad()
def draw_layers(model, diff: Diffusion, real0: np.ndarray, real1: np.ndarray, labels: np.ndarray, seeds: list[int],
                *, steps: int = cfg.SAMPLE_STEPS, guidance: float = cfg.SAMPLE_CFG, spacing: str = cfg.SET_SPACING,
                stride: int = cfg.SET_STRIDE, order: str = cfg.SET_ORDER, eta: float = 0.0, amp: bool = True) -> np.ndarray:
    """레이어 B 장을 그린다.

    real0, real1  (B, S, S) uint8   그 레이어의 진짜 사진. **가장자리 (frame_px) 의 화소만 쓴다**
    labels        (B, S, S) uint8   라벨 맵 (0–9, "모름" = 10)
    seeds         길이 B             레이어마다의 난수 seed
    돌려주는 값: (B, 2, S, S) uint8 — 가장자리는 진짜 화소 그대로다
    """
    dev = next(model.parameters()).device
    B, S = labels.shape[0], labels.shape[-1]
    W = cfg.WIN
    E = frame_px(S)
    frame = torch.ones((S, S), dtype=torch.bool, device=dev)
    frame[E:S - E, E:S - E] = False
    real = torch.from_numpy(np.stack([real0, real1], axis=1).astype(np.float32) / 127.5 - 1.0).to(dev)
    out = torch.zeros((B, 2, S, S), device=dev)
    out[:, :, frame] = real[:, :, frame]                      # 진짜 화소는 여기서만 들어온다
    del real
    done = frame.clone()
    lab = torch.from_numpy(labels.astype(np.int64)).to(dev)
    gens = [torch.Generator(device=dev).manual_seed(int(s)) for s in seeds]
    pos = window_positions(S, E, W, stride)
    for iy, ix in window_order(len(pos), order):
        y, x = pos[iy], pos[ix]
        region = (~done[y:y + W, x:x + W]).float()
        if float(region.sum()) == 0:
            continue
        region = region[None, None].expand(B, 1, W, W).contiguous()
        out[:, :, y:y + W, x:x + W] = sample_fill(
            model, diff, out[:, :, y:y + W, x:x + W].clone(), region, lab[:, y:y + W, x:x + W],
            steps=steps, guidance=guidance, generator=gens, amp=amp, spacing=spacing, eta=eta,
        )
        done[y:y + W, x:x + W] = True
    if not bool(done.all()):
        raise RuntimeError("창이 안쪽을 다 덮지 못했다 — window_positions 를 확인한다")
    res = to_uint8(out)
    f = frame.cpu().numpy()
    assert np.array_equal(res[:, 0][:, f], real0[:, f]) and np.array_equal(res[:, 1][:, f], real1[:, f])
    return res
