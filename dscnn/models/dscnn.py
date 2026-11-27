"""DSCNN (Dynamic Segmentation CNN) — Scime et al., Additive Manufacturing 36 (2020) 101453 의 표 3 – 6 을 그대로 옮긴 것.

갈래 셋이 나란히 돈 뒤 화소마다 이어 붙여 1×1 CONV 로 분류한다.

    stack A (N, 2·n_cam, 36, 36)        레이어 전체를 32² 로 줄여 2 화소 덧댄 것   → global     → (N, 16)        → T×T 로 펼침
    stack B (N, 2·n_cam, T+10, T+10)    원본 해상도 타일 + 둘레 5 화소              → localization → (N, 128, T, T)
    stack C (N, 2·n_cam, 132, 132)      타일을 128² 로 줄여 2 화소 덧댄 것          → regional U-Net → (N, 32, 128, 128) → bilinear 로 T×T
    stack D (N, 2, T, T)                베드 중심 기준 (x, y) ∈ [−1, 1]
    concat [128, 16, 32, 2] = 178 → 1×1 CONV → n_classes (ReLU 없음)

표에 적힌 대로 옮긴 세부:
  - regional 의 내려가는 쪽만 BatchNorm 을 쓴다 (CONV → ReLU → BN 순서). 올라가는 쪽 · global · localization 은 쓰지 않는다
  - regional 은 4×4×512 를 FC 로 512-벡터 하나까지 줄였다가 (Drop50) 다시 펼친다. 올라가는 한 단계 = 최근접 확대 → 1×1 CONV → skip 이어 붙임 → 3×3 CONV
  - 첫 CONV 뒤의 crop 은 덧댄 화소를 떼는 것이다 ("valid" 와 같다). localization 은 11×11 CONV 하나 (padding 없음 → T+10 이 T 가 된다)
  - FC 뒤에는 ReLU 를 적지 않았다 — 표가 CONV 는 "CONV + ReLU", FC 는 "FC" 로만 적는다 (fc_relu=True 로 바꿀 수 있다)
  - 초기값: CONV N(0, 0.05²), FC N(0, 0.004²), bias 0.1

파라미터 수 (n_cam = 1): 13,997,328 + 179·n_classes (BN 은 채널마다 4개 — γ · β · 평균 · 분산). 논문이 적은 13,996,612 + 179·(n_classes + 1) 과 537 차이.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config_dscnn as cfg

N_FEATURES = 178          # 128 (localization) + 16 (global) + 32 (regional) + 2 (좌표)


class _Down(nn.Module):
    """CONV + ReLU + BN (표 3 의 내려가는 쪽 한 줄)."""

    def __init__(self, c_in: int, c_out: int, k: int):
        super().__init__()
        self.conv = nn.Conv2d(c_in, c_out, k, padding=k // 2)
        self.bn = nn.BatchNorm2d(c_out, momentum=cfg.BN_MOMENTUM)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.bn(F.relu(self.conv(x)))


class _Up(nn.Module):
    """최근접 확대 → 1×1 CONV + ReLU → skip 이어 붙임 → 3×3 CONV + ReLU (BN 없음)."""

    def __init__(self, c_in: int, c_out: int):
        super().__init__()
        self.reduce = nn.Conv2d(c_in, c_out, 1)
        self.fuse = nn.Conv2d(2 * c_out, c_out, 3, padding=1)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="nearest")
        x = F.relu(self.reduce(x))
        return F.relu(self.fuse(torch.cat([x, skip], dim=1)))


class RegionalUNet(nn.Module):
    """표 3. (N, C, 132, 132) → (N, 32, 128, 128)."""

    def __init__(self, in_ch: int, fc_relu: bool = False):
        super().__init__()
        self.pad = cfg.REGIONAL_PAD
        self.d1 = _Down(in_ch, 32, 5)      # 132 → crop 128
        self.d2 = _Down(32, 64, 5)         # 64
        self.d3 = _Down(64, 128, 5)        # 32
        self.d4 = _Down(128, 256, 5)       # 16
        self.d5 = _Down(256, 512, 3)       # 8
        self.fc = nn.Linear(4 * 4 * 512, 512)
        self.fc_relu = fc_relu
        self.drop = nn.Dropout(0.5)
        self.u5 = _Up(512, 512)            # 1 → 8
        self.u4 = _Up(512, 256)            # 16
        self.u3 = _Up(256, 128)            # 32
        self.u2 = _Up(128, 64)             # 64
        self.u1 = _Up(64, 32)              # 128

    def forward(self, c: torch.Tensor) -> torch.Tensor:
        p = self.pad
        s1 = self.d1(c)[:, :, p:-p, p:-p]                  # (N, 32, 128, 128)
        s2 = self.d2(F.max_pool2d(s1, 2))                  # (N, 64, 64, 64)
        s3 = self.d3(F.max_pool2d(s2, 2))                  # (N, 128, 32, 32)
        s4 = self.d4(F.max_pool2d(s3, 2))                  # (N, 256, 16, 16)
        s5 = self.d5(F.max_pool2d(s4, 2))                  # (N, 512, 8, 8)
        v = self.fc(F.max_pool2d(s5, 2).flatten(1))        # (N, 512)
        if self.fc_relu:
            v = F.relu(v)
        x = self.drop(v)[:, :, None, None]                 # (N, 512, 1, 1)
        x = self.u5(x, s5)
        x = self.u4(x, s4)
        x = self.u3(x, s3)
        x = self.u2(x, s2)
        return self.u1(x, s1)                              # (N, 32, 128, 128)


class GlobalCNN(nn.Module):
    """표 4. (N, C, 36, 36) → (N, 16)."""

    def __init__(self, in_ch: int, fc_relu: bool = False):
        super().__init__()
        self.pad = cfg.GLOBAL_PAD
        self.c1 = nn.Conv2d(in_ch, 32, 5, padding=2)
        self.c2 = nn.Conv2d(32, 64, 5, padding=2)
        self.c3 = nn.Conv2d(64, 128, 5, padding=2)
        self.fc1 = nn.Linear(4 * 4 * 128, 256)
        self.fc2 = nn.Linear(256, 16)
        self.fc_relu = fc_relu
        self.drop1 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.5)

    def forward(self, a: torch.Tensor) -> torch.Tensor:
        p = self.pad
        x = F.relu(self.c1(a))[:, :, p:-p, p:-p]           # 32
        x = F.relu(self.c2(F.max_pool2d(x, 2)))            # 16
        x = F.relu(self.c3(F.max_pool2d(x, 2)))            # 8
        x = self.fc1(F.max_pool2d(x, 2).flatten(1))        # 4·4·128 → 256
        if self.fc_relu:
            x = F.relu(x)
        x = self.fc2(self.drop1(x))
        if self.fc_relu:
            x = F.relu(x)
        return self.drop2(x)                               # (N, 16)


class DSCNN(nn.Module):
    def __init__(self, n_classes: int = cfg.N_CLASSES, in_ch: int = cfg.IN_CHANNELS, fc_relu: bool = False):
        super().__init__()
        self.n_classes = n_classes
        self.in_ch = in_ch
        self.regional = RegionalUNet(in_ch, fc_relu)
        self.glob = GlobalCNN(in_ch, fc_relu)
        self.local = nn.Conv2d(in_ch, 128, 2 * cfg.LOCAL_PAD + 1)      # 11×11, padding 없음 (표 5)
        self.classify = nn.Conv2d(N_FEATURES, n_classes, 1)            # 표 6
        self.reset_parameters()

    def reset_parameters(self) -> None:
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.normal_(m.weight, 0.0, cfg.INIT_CONV_STD)
                nn.init.constant_(m.bias, cfg.INIT_BIAS)
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0.0, cfg.INIT_FC_STD)
                nn.init.constant_(m.bias, cfg.INIT_BIAS)

    def global_features(self, a: torch.Tensor) -> torch.Tensor:
        """stack A → (N, 16). 한 레이어의 모든 타일에 같은 값이므로 추론에서는 레이어마다 한 번만 낸다."""
        return self.glob(a)

    def forward(
        self, b: torch.Tensor, c: torch.Tensor, d: torch.Tensor,
        a: torch.Tensor | None = None, g: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """b (N, C, T+10, T+10), c (N, C, 132, 132), d (N, 2, T, T), a (N, C, 36, 36) 또는 g (N | 1, 16) → logits (N, n_classes, T, T)."""
        if g is None:
            if a is None:
                raise ValueError("stack A (a) 나 미리 낸 global 특징 (g) 가운데 하나가 있어야 한다")
            g = self.glob(a)
        return self.head(b, d, self.regional(c), g)

    def head(self, b: torch.Tensor, d: torch.Tensor, reg: torch.Tensor, g: torch.Tensor) -> torch.Tensor:
        """localization + 이어 붙임 + 분류. reg = regional 의 출력 (N, 32, 128, 128), g = global 의 출력 (N | 1, 16).

        regional · global 은 타일 크기와 무관하게 한 묶음으로 돌 수 있다 (입력이 늘 132² · 36²). 타일 크기가 다른 소스를 한 배치에 섞을 때
        그 둘을 먼저 한꺼번에 돌리고, 이 함수만 타일 크기가 같은 것끼리 부른다 (학습 스크립트).
        """
        loc = F.relu(self.local(b))                                     # (N, 128, T, T)
        t_hw = loc.shape[-2:]
        reg = F.interpolate(reg, size=t_hw, mode="bilinear", align_corners=False)                # (N, 32, T, T)
        if g.shape[0] != loc.shape[0]:
            g = g.expand(loc.shape[0], -1)
        glo = g[:, :, None, None].expand(-1, -1, *t_hw)                 # (N, 16, T, T)
        return self.classify(torch.cat([loc, glo, reg, d], dim=1))


def count_parameters(model: nn.Module) -> dict[str, int]:
    """학습되는 값 + BN 의 이동 평균 · 분산 (논문이 세는 방식)."""
    learn = sum(p.numel() for p in model.parameters())
    stats = sum(b.numel() for n, b in model.named_buffers() if n.endswith(("running_mean", "running_var")))
    return {"learnable": learn, "bn_stats": stats, "total": learn + stats}
