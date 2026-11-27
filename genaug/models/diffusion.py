"""확산 과정 — 학습 손실과 그리기 (PLAN_GENAUG §3.2, §3.5).

- 잡음 일정: cosine (Nichol & Dhariwal), 1,000 단계
- 모델이 맞히는 것: v = sqrt(ᾱ)·잡음 − sqrt(1−ᾱ)·원본  (v-prediction). 잡음이 아주 많거나 적은 단계에서도 값의 크기가 고르다
- 손실: v 의 제곱 오차. 그릴 자리 안을 주로 보고 (밖은 입력으로 주어진 화소라 가중 0.05), 단계별 가중은 min-SNR
- 그리기: DDIM (무작위성 없음). 매 단계마다 **그릴 자리 밖의 화소를 원본으로 되돌려 놓는다** — 끝난 뒤 그 화소는 원본과 한 값도 다르지 않다
- guidance: 라벨 맵을 준 예측과 "모름" 으로 뗀 예측의 차이를 배율만큼 키운다
- 강도 s: 1 이면 잡음에서 시작해 처음부터 그리고, 1 보다 작으면 출발 그림 (붙인 조각) 에 잡음을 그만큼만 섞어 거기서부터 그린다
"""
from __future__ import annotations

import math

import torch
import torch.nn.functional as F

from .. import config_genaug as cfg


class Diffusion:
    def __init__(self, T: int = cfg.T_STEPS, device: torch.device | str = "cpu"):
        self.T = T
        s = 0.008
        x = torch.linspace(0, T, T + 1, dtype=torch.float64)
        f = torch.cos(((x / T) + s) / (1 + s) * math.pi * 0.5) ** 2
        betas = (1 - f[1:] / f[:-1]).clamp(max=0.999)
        ac = torch.cumprod(1 - betas, dim=0)                       # ᾱ_t,  t = 0 … T-1
        self.sqrt_ac = ac.sqrt().float().to(device)
        self.sqrt_1mac = (1 - ac).sqrt().float().to(device)
        self.snr = (ac / (1 - ac)).float().to(device)

    def to(self, device) -> "Diffusion":
        self.sqrt_ac, self.sqrt_1mac, self.snr = (t.to(device) for t in (self.sqrt_ac, self.sqrt_1mac, self.snr))
        return self

    @staticmethod
    def _b(v: torch.Tensor) -> torch.Tensor:
        return v[:, None, None, None]

    def q_sample(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        """원본에 단계 t 만큼의 잡음을 섞는다."""
        return self._b(self.sqrt_ac[t]) * x0 + self._b(self.sqrt_1mac[t]) * noise

    def v_target(self, x0: torch.Tensor, t: torch.Tensor, noise: torch.Tensor) -> torch.Tensor:
        return self._b(self.sqrt_ac[t]) * noise - self._b(self.sqrt_1mac[t]) * x0

    def x0_eps_from_v(self, x_t: torch.Tensor, t: torch.Tensor, v: torch.Tensor):
        a, s = self._b(self.sqrt_ac[t]), self._b(self.sqrt_1mac[t])
        return a * x_t - s * v, s * x_t + a * v

    def loss_weight(self, t: torch.Tensor, gamma: float = cfg.MIN_SNR_GAMMA) -> torch.Tensor:
        """min-SNR 가중 (v-prediction 꼴): min(SNR, γ) / (SNR + 1)."""
        snr = self.snr[t]
        return torch.clamp(snr, max=gamma) / (snr + 1.0)


def model_input(x_t: torch.Tensor, x_clean: torch.Tensor, region: torch.Tensor, label: torch.Tensor) -> torch.Tensor:
    """모델에 넣는 16 채널 — 잡음 섞인 창, 그릴 자리를 지운 원본, 그릴 자리, 라벨 맵 (one-hot)."""
    onehot = F.one_hot(label.long(), cfg.N_LABEL_CH).permute(0, 3, 1, 2).to(x_t.dtype)
    return torch.cat([x_t, x_clean * (1.0 - region), region, onehot], dim=1)


def training_loss(model, diff: Diffusion, x0: torch.Tensor, region: torch.Tensor, label: torch.Tensor,
                  cond_drop: float = cfg.COND_DROP, generator: torch.Generator | None = None):
    """한 배치의 손실. (평균 손실, 표본별 손실, 뽑은 단계) 를 돌려준다."""
    B = x0.shape[0]
    t = torch.randint(0, diff.T, (B,), device=x0.device, generator=generator)
    noise = torch.randn(x0.shape, device=x0.device, generator=generator, dtype=x0.dtype)
    x_t = diff.q_sample(x0, t, noise)
    if cond_drop > 0:                                               # 일부 표본은 라벨 맵을 전부 "모름" 으로
        drop = torch.rand(B, device=x0.device, generator=generator) < cond_drop
        label = torch.where(drop[:, None, None], torch.full_like(label, cfg.UNKNOWN), label)
    v = model(model_input(x_t, x0, region, label), t)
    target = diff.v_target(x0, t, noise)
    w = region + cfg.OUTSIDE_LOSS_WEIGHT * (1.0 - region)           # (B, 1, H, W)
    per = ((v.float() - target.float()) ** 2 * w).sum(dim=(1, 2, 3)) / (w.sum(dim=(1, 2, 3)) * x0.shape[1])
    return (per * diff.loss_weight(t)).mean(), per.detach(), t


SPACINGS = ("uniform", "quad")


def sample_steps(t_start: int, steps: int, spacing: str = "uniform", device=None) -> torch.Tensor:
    """그릴 때 밟는 단계 (잡음이 많은 쪽 → 0).

    uniform  고르게 나눈다 (기본값 — 학습 중의 표본 시트와 2026-11-18 까지의 점검이 쓴 배치)
    quad     잡음이 적은 구간에 몬다 (간격이 제곱으로 준다). 같은 50 단계로 도포 직후 사진의 고운 결이 더 산다
             (창 전체를 그렸을 때 결의 세기가 진짜의 0.52–0.73 → 0.77–0.81, 2026-11-18 측정)
    """
    n = max(1, min(int(steps), t_start + 1))
    if spacing == "uniform":
        ts = torch.linspace(t_start, 0, n, device=device)
    elif spacing == "quad":
        ts = (torch.linspace(1, 0, n, device=device) ** 2) * t_start
    else:
        raise ValueError(f"spacing 은 {SPACINGS} 가운데 하나다: {spacing!r}")
    return ts.round().long().unique_consecutive()


@torch.no_grad()
def sample_fill(model, diff: Diffusion, x_clean: torch.Tensor, region: torch.Tensor, label: torch.Tensor,
                steps: int = cfg.SAMPLE_STEPS, guidance: float = cfg.SAMPLE_CFG, strength: float = 1.0,
                init: torch.Tensor | None = None, generator: torch.Generator | list | tuple | None = None,
                amp: bool = False, spacing: str = "uniform", eta: float = 0.0) -> torch.Tensor:
    """그릴 자리를 그린다.

    x_clean  (B, 2, H, W)  [-1, 1]  원본 창. 그릴 자리 밖의 화소만 쓴다
    region   (B, 1, H, W)  0/1      그릴 자리
    label    (B, H, W)     0 … 10   라벨 맵 (그릴 자리 밖은 "모름")
    init     (B, 2, H, W)           강도 < 1 일 때의 출발 그림 (그릴 자리에 붙인 조각). 없으면 x_clean 을 쓴다
    generator                        난수 발생기 하나, 또는 표본마다 하나씩 (길이 B 의 목록 — 묶음을 어떻게 짜도 표본의 잡음이 같다)
    spacing                          단계의 배치 (`sample_steps`)
    eta                              단계마다 새 잡음을 섞는 정도 (DDIM 의 η). 0 = 섞지 않는다 (기본값), 1 = DDPM 처럼 섞는다
    돌려주는 값: (B, 2, H, W) — 그릴 자리 밖은 x_clean 과 같다
    """
    dev = x_clean.device
    known = x_clean * (1.0 - region)
    label = torch.where(region[:, 0] > 0.5, label, torch.full_like(label, cfg.UNKNOWN))
    null = torch.full_like(label, cfg.UNKNOWN)
    strength = float(min(max(strength, 0.0), 1.0))
    if strength <= 0.0:
        start = init if init is not None else x_clean
        return region * start + (1.0 - region) * x_clean
    t_start = max(int(round(diff.T * strength)) - 1, 0)
    ts = sample_steps(t_start, steps, spacing, dev)
    per_sample = isinstance(generator, (list, tuple))
    if per_sample and len(generator) != x_clean.shape[0]:
        raise ValueError(f"표본마다의 난수 발생기는 {x_clean.shape[0]} 개여야 한다: {len(generator)}")

    def rnd():
        if per_sample:
            return torch.stack([torch.randn(x_clean.shape[1:], device=dev, generator=g, dtype=x_clean.dtype) for g in generator])
        return torch.randn(x_clean.shape, device=dev, generator=generator, dtype=x_clean.dtype)

    B = x_clean.shape[0]
    tb = torch.full((B,), int(ts[0]), device=dev, dtype=torch.long)
    if strength < 1.0:
        start = region * (init if init is not None else x_clean) + (1.0 - region) * x_clean
        x = diff.q_sample(start, tb, rnd())
    else:
        x = region * rnd() + (1.0 - region) * diff.q_sample(x_clean, tb, rnd())

    use_cfg = guidance != 1.0
    for i, t in enumerate(ts.tolist()):
        tb = torch.full((B,), t, device=dev, dtype=torch.long)
        with torch.autocast(device_type=dev.type, dtype=torch.bfloat16, enabled=amp and dev.type == "cuda"):
            if use_cfg:
                inp = torch.cat([model_input(x, x_clean, region, label), model_input(x, x_clean, region, null)])
                v_c, v_u = model(inp, torch.cat([tb, tb])).float().chunk(2)
                v = v_u + guidance * (v_c - v_u)
            else:
                v = model(model_input(x, x_clean, region, label), tb).float()
        x0, _ = diff.x0_eps_from_v(x, tb, v)
        x0 = x0.clamp(-1.0, 1.0)
        if i + 1 < len(ts):
            a_t, s_t = diff._b(diff.sqrt_ac[tb]), diff._b(diff.sqrt_1mac[tb])
            eps = (x - a_t * x0) / s_t
            tp = torch.full((B,), int(ts[i + 1]), device=dev, dtype=torch.long)
            a_p, s_p = diff._b(diff.sqrt_ac[tp]), diff._b(diff.sqrt_1mac[tp])
            if eta > 0.0:                                                                   # DDIM, η > 0 — 새 잡음 σ·z 를 섞는다
                sigma = eta * (s_p / s_t) * torch.sqrt((1.0 - (a_t / a_p) ** 2).clamp(min=0.0))
                x = a_p * x0 + torch.sqrt((s_p ** 2 - sigma ** 2).clamp(min=0.0)) * eps + sigma * rnd()
            else:
                x = a_p * x0 + s_p * eps                                                    # DDIM, η = 0
            x = region * x + (1.0 - region) * diff.q_sample(x_clean, tp, rnd())             # 아는 화소를 되돌려 놓는다
        else:
            x = x0
    return region * x + known
