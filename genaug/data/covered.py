"""덮임 판정 — 도포 직후 사진에서 파트 자리가 주변 분말과 구별되지 않으면 덮인 것으로 본다.

`scripts/synth_prototype/covered_rule.py` 의 `covered_mask` 를 값 그대로 옮긴 사본이다 (버전 · 계획 폴더끼리 임포트하지 않는다).
사람 GT 125 레이어에서의 검증 (rulesynth/PLAN_RULESYNTH.md §4.1): 이 판정이 고른 화소의 0.24% 만 결함이었다.

쓰임 (PLAN_GENAUG §3.2 의 학습 자료 ②): 주석 없는 ORNL 창의 라벨 맵에서, 파트 자리 가운데 "확실히 덮인" 곳만 Printed 로 적는다.
드러나 보이는 곳은 진짜 Super Elevation 일 수 있으므로 "모름" 으로 둔다.
"""
from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, uniform_filter

from .. import config_genaug as cfg


def covered_mask(
    i1: np.ndarray,
    foot: np.ndarray,
    tau: float = cfg.COVERED_TAU,
    win: int = cfg.COVERED_WIN,
    grow: int = cfg.COVERED_GROW,
) -> np.ndarray:
    """파트 자리 `foot` 가운데 "확실히 덮인" 화소.

    i1   = visible/1 (도포 직후, uint8, **원본 밝기**)
    foot = 파트 자리 (bool)
    """
    x = i1.astype(np.float32)
    powder = ~binary_dilation(foot, iterations=3)                 # 파트 가장자리를 피한 분말 화소
    m = powder.astype(np.float32)
    cnt = uniform_filter(m, win, mode="nearest")
    mu = uniform_filter(x * m, win, mode="nearest") / np.maximum(cnt, 1e-3)
    var = uniform_filter(x * x * m, win, mode="nearest") / np.maximum(cnt, 1e-3) - mu * mu
    sd = np.sqrt(np.maximum(var, 0.0))
    ok = cnt > 0.05                                               # 주변에 분말이 충분히 있는 곳만 판정
    z = np.abs(x - mu) / np.maximum(sd, 2.0)
    exposed = (z > tau) & foot
    exposed = binary_dilation(exposed, iterations=grow)
    return binary_erosion(foot, iterations=2) & ~exposed & ok
