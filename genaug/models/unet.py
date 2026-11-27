"""결함 인페인팅 확산 모델의 U-Net (PLAN_GENAUG §3.2).

확산 모델에 흔히 쓰는 구성이다 (DDPM / ADM 계열): ResBlock (GroupNorm → SiLU → Conv) + 시간 임베딩,
해상도를 절반씩 줄였다 되돌리는 U 자 구조와 건너뛰기 연결, 낮은 해상도 (16², 8²) 에서의 self-attention.

입력 16 채널 = 잡음 섞인 창 2 + 가린 창 2 + 그릴 자리 1 + 라벨 맵 11.  출력 2 채널 (v-prediction).
채널 수는 위쪽 (256²) 을 좁게, 아래쪽을 넓게 잡았다 — 256² 에서 128 채널을 쓰는 원래 구성은 메모리를 많이 쓴다.
"""
from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def timestep_embedding(t: torch.Tensor, dim: int, max_period: float = 10000.0) -> torch.Tensor:
    """(B,) 정수 시각 → (B, dim) sin/cos 임베딩."""
    half = dim // 2
    freqs = torch.exp(-math.log(max_period) * torch.arange(half, dtype=torch.float32, device=t.device) / half)
    args = t.float()[:, None] * freqs[None]
    emb = torch.cat([torch.cos(args), torch.sin(args)], dim=-1)
    if dim % 2:
        emb = F.pad(emb, (0, 1))
    return emb


def group_norm(ch: int) -> nn.GroupNorm:
    groups = 32
    while ch % groups:
        groups //= 2
    return nn.GroupNorm(groups, ch, eps=1e-6)


def zero_init(m: nn.Module) -> nn.Module:
    for p in m.parameters():
        nn.init.zeros_(p)
    return m


class ResBlock(nn.Module):
    def __init__(self, cin: int, cout: int, tdim: int, dropout: float):
        super().__init__()
        self.norm1 = group_norm(cin)
        self.conv1 = nn.Conv2d(cin, cout, 3, padding=1)
        self.temb = nn.Linear(tdim, cout)
        self.norm2 = group_norm(cout)
        self.drop = nn.Dropout(dropout)
        self.conv2 = zero_init(nn.Conv2d(cout, cout, 3, padding=1))
        self.skip = nn.Conv2d(cin, cout, 1) if cin != cout else nn.Identity()

    def forward(self, x: torch.Tensor, temb: torch.Tensor) -> torch.Tensor:
        h = self.conv1(F.silu(self.norm1(x)))
        h = h + self.temb(temb)[:, :, None, None]
        h = self.conv2(self.drop(F.silu(self.norm2(h))))
        return self.skip(x) + h


class AttnBlock(nn.Module):
    def __init__(self, ch: int, heads: int):
        super().__init__()
        assert ch % heads == 0
        self.heads = heads
        self.norm = group_norm(ch)
        self.qkv = nn.Conv2d(ch, ch * 3, 1)
        self.proj = zero_init(nn.Conv2d(ch, ch, 1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        B, C, H, W = x.shape
        q, k, v = self.qkv(self.norm(x)).reshape(B, 3, self.heads, C // self.heads, H * W).unbind(1)
        q, k, v = (t.transpose(-1, -2) for t in (q, k, v))        # (B, heads, HW, C/heads)
        h = F.scaled_dot_product_attention(q, k, v)
        h = h.transpose(-1, -2).reshape(B, C, H, W)
        return x + self.proj(h)


class Stage(nn.Module):
    """ResBlock 하나 (+ 그 해상도에서 attention)."""

    def __init__(self, cin: int, cout: int, tdim: int, dropout: float, attn: bool, heads: int):
        super().__init__()
        self.res = ResBlock(cin, cout, tdim, dropout)
        self.attn = AttnBlock(cout, heads) if attn else None

    def forward(self, x, temb):
        x = self.res(x, temb)
        return self.attn(x) if self.attn is not None else x


class Downsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, stride=2, padding=1)

    def forward(self, x, temb=None):
        return self.conv(x)


class Upsample(nn.Module):
    def __init__(self, ch: int):
        super().__init__()
        self.conv = nn.Conv2d(ch, ch, 3, padding=1)

    def forward(self, x):
        return self.conv(F.interpolate(x, scale_factor=2.0, mode="nearest"))


class FillUNet(nn.Module):
    def __init__(self, in_ch: int, out_ch: int, base: int, mult: tuple[int, ...], n_res: int,
                 attn_res: tuple[int, ...], heads: int, dropout: float, win: int):
        super().__init__()
        self.cfg = dict(in_ch=in_ch, out_ch=out_ch, base=base, mult=tuple(mult), n_res=n_res,
                        attn_res=tuple(attn_res), heads=heads, dropout=dropout, win=win)
        if win % (2 ** (len(mult) - 1)):
            raise ValueError(f"창 크기 {win} 이 {2 ** (len(mult) - 1)} 의 배수가 아니다")
        self.t_in = max(base, 128)                 # 시각의 sin/cos 임베딩 크기
        tdim = max(base * 4, 512)
        self.time = nn.Sequential(nn.Linear(self.t_in, tdim), nn.SiLU(), nn.Linear(tdim, tdim))
        self.inp = nn.Conv2d(in_ch, base, 3, padding=1)

        # 내려가는 쪽
        self.down = nn.ModuleList()
        skip_ch = [base]
        ch, res = base, win
        for level, m in enumerate(mult):
            cout = base * m
            for _ in range(n_res):
                self.down.append(Stage(ch, cout, tdim, dropout, res in attn_res, heads))
                ch = cout
                skip_ch.append(ch)
            if level != len(mult) - 1:
                self.down.append(Downsample(ch))
                skip_ch.append(ch)
                res //= 2

        # 가운데
        self.mid1 = ResBlock(ch, ch, tdim, dropout)
        self.mid_attn = AttnBlock(ch, heads)
        self.mid2 = ResBlock(ch, ch, tdim, dropout)

        # 올라가는 쪽
        self.up = nn.ModuleList()
        self.up_sample = nn.ModuleList()
        for level, m in reversed(list(enumerate(mult))):
            cout = base * m
            stages = nn.ModuleList()
            for _ in range(n_res + 1):
                stages.append(Stage(ch + skip_ch.pop(), cout, tdim, dropout, res in attn_res, heads))
                ch = cout
            self.up.append(stages)
            if level != 0:
                self.up_sample.append(Upsample(ch))
                res *= 2
            else:
                self.up_sample.append(nn.Identity())
        assert not skip_ch

        self.out_norm = group_norm(ch)
        self.out = zero_init(nn.Conv2d(ch, out_ch, 3, padding=1))

    def forward(self, x: torch.Tensor, t: torch.Tensor) -> torch.Tensor:
        temb = self.time(timestep_embedding(t, self.t_in))
        h = self.inp(x)
        hs = [h]
        for layer in self.down:
            h = layer(h, temb)
            hs.append(h)
        h = self.mid2(self.mid_attn(self.mid1(h, temb)), temb)
        for stages, up in zip(self.up, self.up_sample):
            for st in stages:
                h = st(torch.cat([h, hs.pop()], dim=1), temb)
            h = up(h)
        return self.out(F.silu(self.out_norm(h)))


def build_unet(model_cfg: dict) -> FillUNet:
    return FillUNet(**model_cfg)


def count_params(m: nn.Module) -> int:
    return sum(p.numel() for p in m.parameters())
