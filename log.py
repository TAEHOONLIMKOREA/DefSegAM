"""Timestamped, rank-aware logger.

사용:
    from .log import setup_logger, get_logger
    log = setup_logger(rank=rank)
    log.info("hello")
    # → 17:23:45.001 [r0] hello

DDP 환경:
    - rank 0 만 INFO 로그를 stdout 으로 출력
    - 다른 rank 는 WARNING 이상만 출력 (rank prefix 로 식별 가능)
"""
from __future__ import annotations

import logging
import sys


def setup_logger(rank: int = 0, name: str = "defseg") -> logging.Logger:
    """timestamp + rank prefix + INFO/DEBUG 분리."""
    fmt = logging.Formatter(
        f"%(asctime)s.%(msecs)03d [r{rank}] %(levelname).1s %(message)s",
        datefmt="%H:%M:%S",
    )
    h = logging.StreamHandler(sys.stdout)
    h.setFormatter(fmt)

    logger = logging.getLogger(name)
    logger.handlers = [h]
    # rank 0 만 verbose, 나머지는 WARNING+ (DDP race 회피)
    logger.setLevel(logging.INFO if rank == 0 else logging.WARNING)
    logger.propagate = False
    return logger


def get_logger(name: str = "defseg") -> logging.Logger:
    """이미 setup 된 logger 재취득."""
    return logging.getLogger(name)
