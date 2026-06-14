"""DDP helper — v1·v2 공유.

torchrun 멀티 GPU 환경 초기화 및 rank/all-reduce 유틸. v1·v2 의
train_stage1 / train_stage2 가 함께 임포트한다.
"""
from __future__ import annotations

import os

import torch
import torch.distributed as dist


def init_distributed() -> tuple[int, int, int]:
    """torchrun 환경이면 DDP init, 아니면 single-process. Returns (rank, world_size, local_rank)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def is_main(rank: int) -> bool:
    return rank == 0


def unwrap(m):
    return m.module if hasattr(m, "module") else m


def reduce_counts(world_size: int, *tensors: torch.Tensor) -> None:
    if world_size > 1:
        for t in tensors:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)
