"""DDP-호환 WeightedRandomSampler.

PyTorch 표준 DistributedSampler 는 가중치 없는 uniform — defect-ratio 기반
oversampling 을 DDP 와 결합하려면 직접 구현 필요.

각 rank 는 매 epoch 마다 epoch+rank 기반 seed 로 독립 multinomial sampling.
WeightedRandomSampler(replacement=True) 와 동일한 분포, rank 마다 다른 sample.
"""
from __future__ import annotations

import torch
from torch.utils.data import Sampler


class DistributedWeightedSampler(Sampler[int]):
    def __init__(
        self,
        weights,
        num_samples_total: int,
        num_replicas: int = 1,
        rank: int = 0,
        replacement: bool = True,
        seed: int = 0,
    ):
        self.weights = torch.as_tensor(weights, dtype=torch.double)
        self.num_samples = num_samples_total // num_replicas
        self.num_replicas = num_replicas
        self.rank = rank
        self.replacement = replacement
        self.seed = seed
        self.epoch = 0

    def __iter__(self):
        g = torch.Generator()
        g.manual_seed(self.seed + self.epoch * 1000 + self.rank)
        idx = torch.multinomial(
            self.weights, self.num_samples,
            replacement=self.replacement, generator=g,
        )
        return iter(idx.tolist())

    def __len__(self) -> int:
        return self.num_samples

    def set_epoch(self, epoch: int) -> None:
        self.epoch = epoch
