"""DefSegModel: DINOv2 frozen + DPT-style multi-scale decoder.

PLAN.md §1.3 / §1.4 / §4 참조.

Architecture:
    img0, img1 (B, 3, H, W)
        ↓ DINOv2 (frozen, dual forward) ×2
    f0_s1..s4, f1_s1..s4  (B, D, Hp, Wp) each, from blocks [2,5,8,11]
        ↓ per-stage fusion: concat(f0, f1, f1-f0) → 1x1 conv → 256 ch
    4 fused tensors at patch grid resolution
        ↓ Reassemble: 4x↑ / 2x↑ / identity / 2x↓
    s1=(296,296)  s2=(148,148)  s3=(74,74)  s4=(37,37)        (for H=1036)
        ↓ Fusion blocks (top-down s4→s1)
    output at 4x s1 resolution
        ↓ Head (3x3 conv, ReLU, 1x1 conv → n_classes)
        ↓ bilinear interpolate to input (H, W)
    logits (B, n_classes, H, W)
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F

from .. import config


def round_to_patch(size: int, patch: int) -> int:
    """size 를 patch 의 배수로 내림 (학습/추론 input size 자동 보정용)."""
    return (size // patch) * patch


# ---------------------------------------------------------------------------
# DPT building blocks
# ---------------------------------------------------------------------------

class ResidualConvUnit(nn.Module):
    """3x3 conv → ReLU → 3x3 conv → residual add. DPT 표준 유닛."""

    def __init__(self, features: int):
        super().__init__()
        self.conv1 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.conv2 = nn.Conv2d(features, features, kernel_size=3, padding=1)
        self.relu = nn.ReLU(inplace=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        out = self.relu(x)
        out = self.conv1(out)
        out = self.relu(out)
        out = self.conv2(out)
        return out + x


class FeatureFusionBlock(nn.Module):
    """Top-down fusion: prev → (optional skip add) → resConv → ×2 upsample → 1x1."""

    def __init__(self, features: int):
        super().__init__()
        self.res_skip = ResidualConvUnit(features)   # skip 처리용 (있을 때만 호출)
        self.res_out = ResidualConvUnit(features)
        self.out_conv = nn.Conv2d(features, features, kernel_size=1)

    def forward(
        self,
        prev: torch.Tensor,
        skip: torch.Tensor | None = None,
    ) -> torch.Tensor:
        if skip is not None:
            prev = prev + self.res_skip(skip)
        out = self.res_out(prev)
        out = F.interpolate(out, scale_factor=2.0, mode="bilinear", align_corners=True)
        out = self.out_conv(out)
        return out


# ---------------------------------------------------------------------------
# Main model
# ---------------------------------------------------------------------------

class DefSegModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = config.DINO_BACKBONE,
        n_classes: int = config.N_CLASSES,
        decoder_channels: int = config.DECODER_CHANNELS,
        intermediate_layers: tuple[int, ...] = config.INTERMEDIATE_LAYERS,
        head_dropout: float = 0.1,
    ):
        super().__init__()
        self.backbone_name = backbone_name
        self.n_classes = n_classes
        self.decoder_channels = decoder_channels
        self.intermediate_layers = tuple(intermediate_layers)

        # 1) DINOv2 backbone (frozen)
        self.backbone = torch.hub.load(
            "facebookresearch/dinov2", backbone_name,
            trust_repo=True, verbose=False,
        )
        self.embed_dim = self.backbone.embed_dim
        self.patch_size = self.backbone.patch_size
        self.backbone.eval()
        for p in self.backbone.parameters():
            p.requires_grad = False

        # 2) Per-stage dual fusion: (f0, f1, f1-f0) concat → 1x1 conv → decoder_channels
        #    각 stage 별로 별도 conv 4개 (channel mixing 이 stage 마다 다르게 학습됨)
        n_stages = len(self.intermediate_layers)
        self.fuse_proj = nn.ModuleList([
            nn.Conv2d(3 * self.embed_dim, decoder_channels, kernel_size=1)
            for _ in range(n_stages)
        ])

        # 3) Reassemble — 4 stages 각각 다른 해상도로 변환.
        #    s1 = 4× ↑  / s2 = 2× ↑  / s3 = identity  / s4 = 2× ↓
        self.reassemble = nn.ModuleList([
            nn.ConvTranspose2d(decoder_channels, decoder_channels, kernel_size=4, stride=4),
            nn.ConvTranspose2d(decoder_channels, decoder_channels, kernel_size=2, stride=2),
            nn.Identity(),
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, stride=2, padding=1),
        ])

        # 4) Fusion blocks (top-down: s4 → s3 → s2 → s1) — 4개
        self.fusion_blocks = nn.ModuleList([
            FeatureFusionBlock(decoder_channels) for _ in range(n_stages)
        ])

        # 5) Head: 3x3 conv → ReLU → dropout → 1x1 conv → n_classes
        self.head = nn.Sequential(
            nn.Conv2d(decoder_channels, decoder_channels, kernel_size=3, padding=1),
            nn.ReLU(inplace=True),
            nn.Dropout2d(p=head_dropout),
            nn.Conv2d(decoder_channels, n_classes, kernel_size=1),
        )

    # ------------------------------------------------------------------
    # Backbone helper (frozen, no grad)
    # ------------------------------------------------------------------
    @torch.no_grad()
    def _multi_scale_features(self, img: torch.Tensor) -> list[torch.Tensor]:
        """(B, 3, H, W) → list of n_stages × (B, D, Hp, Wp).

        get_intermediate_layers(reshape=True) 가 직접 spatial map 으로 reshape 해줌.
        """
        B, _, H, W = img.shape
        assert H % self.patch_size == 0 and W % self.patch_size == 0, (
            f"H={H} W={W} must be multiples of patch_size={self.patch_size}"
        )
        feats = self.backbone.get_intermediate_layers(
            img,
            n=list(self.intermediate_layers),
            reshape=True,
            return_class_token=False,
            norm=True,
        )
        return list(feats)  # tuple → list

    # ------------------------------------------------------------------
    # Forward
    # ------------------------------------------------------------------
    def forward(self, img0: torch.Tensor, img1: torch.Tensor) -> torch.Tensor:
        f0 = self._multi_scale_features(img0)
        f1 = self._multi_scale_features(img1)
        assert len(f0) == len(self.fuse_proj)

        # Per-stage dual fusion + projection to decoder_channels
        fused = []
        for s in range(len(f0)):
            joint = torch.cat([f0[s], f1[s], f1[s] - f0[s]], dim=1)
            fused.append(self.fuse_proj[s](joint))

        # Reassemble to multi-scale resolutions
        reasm = [self.reassemble[s](fused[s]) for s in range(len(fused))]
        # reasm = [s1 (highest res), s2, s3, s4 (lowest res)]

        # Top-down fusion: start from s4, progressively add s3/s2/s1
        out = self.fusion_blocks[3](reasm[3])                       # s4 → 2× ↑
        out = self.fusion_blocks[2](out, reasm[2])                  # + s3 → 2× ↑
        out = self.fusion_blocks[1](out, reasm[1])                  # + s2 → 2× ↑
        out = self.fusion_blocks[0](out, reasm[0])                  # + s1 → 2× ↑

        # Head + final upsample to input resolution
        out = self.head(out)
        out = F.interpolate(out, size=img0.shape[-2:], mode="bilinear", align_corners=False)
        return out

    # ------------------------------------------------------------------
    # Trainable parameter / state-dict helpers (backbone 제외, checkpoint 작게)
    # ------------------------------------------------------------------
    def trainable_modules(self) -> list[tuple[str, nn.Module]]:
        return [
            ("fuse_proj", self.fuse_proj),
            ("reassemble", self.reassemble),
            ("fusion_blocks", self.fusion_blocks),
            ("head", self.head),
        ]

    def trainable_parameters(self):
        params = []
        for _, m in self.trainable_modules():
            params += list(m.parameters())
        return params

    def trainable_state_dict(self) -> dict:
        sd = {}
        for name, m in self.trainable_modules():
            for k, v in m.state_dict().items():
                sd[f"{name}.{k}"] = v
        return sd

    def load_trainable_state_dict(self, sd: dict) -> None:
        for name, m in self.trainable_modules():
            prefix = f"{name}."
            sub = {k[len(prefix):]: v for k, v in sd.items() if k.startswith(prefix)}
            if sub:
                m.load_state_dict(sub)
