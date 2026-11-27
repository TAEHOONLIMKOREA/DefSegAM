"""v8 설정 — v6 에 더하는 값만 둔다 (PLAN_v8 §1, §3, §5).

v8 = v6 + 입력 사진의 전처리 둘 (소스별 밝기 맞춤, 조명 지도 제거). 클래스 · 모델 · 길이 · 증강 · 손실 · 채점의 값은 **v6 의 config 를 그대로 쓴다**.
여기에는 전처리의 값 · 보정 세트의 자리 · run 이름만 있다.

이 폴더는 저장소의 규칙 ("버전끼리 임포트하지 않는다") 의 예외다 (PLAN_v8 §7 의 D1) — 기준선이 v7 의 `N` 실행이라서
같은 학습 코드 (v7 의 train, v6 의 evaluate) 를 불러야 전처리만 다른 비교가 된다. **생성 데이터는 쓰지 않는다.**
"""
from __future__ import annotations

from pathlib import Path

from ..v6 import config_v6 as v6cfg

VERSION = "v8"

# ── 밝기 맞춤 (PLAN_v8 §1, §3.1) ──
REF_BRIGHTNESS = 115.0           # 기준 밝기 (제안값). 소스의 대표 밝기 (중앙값) 를 이 값으로 옮긴다

# ── 조명 지도 (PLAN_v8 §3.2) ──
GRID = 96                        # 사진을 GRID × GRID 칸으로 줄여 지도를 구한다
TRIM_ROUNDS = 3                  # 면에서 많이 벗어난 칸을 빼고 다시 맞추는 횟수
TRIM_K = 2.0                     # 벗어났다고 보는 문턱 — 잔차가 TRIM_K × (잔차의 MAD 로 잰 표준편차) 를 넘는 칸
MAP_CLIP = (0.5, 2.0)            # 지도 값의 한계 — 이 밖으로 나가는 지도는 쓰지 않는다 (맞춤이 틀린 것이다)

# ── 안전 장치 (PLAN_v8 §3.3) — 셋을 모두 지나야 그 소스에 지도를 쓴다. 문턱은 제안값 ──
GATE_MIN_LAYERS = 10             # S1: 레이어 수
GATE_MIN_SD_PCT = 1.0            # S2: 얼룩의 크기 (지도의 표준편차, %)
GATE_MIN_R2 = 0.70               # S3: 면이 칸 지도의 변화를 설명하는 정도

# ── 보정의 종류 — 보정 세트를 만들 때 고른다 ──
MODE_RAW = "raw"                 # 아무것도 하지 않는다 (원본 화소 그대로 — "껍데기가 다른 것을 바꾸지 않는다" 를 확인하는 스모크용)
MODE_B = "B"                     # 밝기 맞춤만
MODE_BL = "BL"                   # 밝기 맞춤 + 조명 제거 (안전 장치를 지난 소스만)
MODES = (MODE_RAW, MODE_B, MODE_BL)
DEFAULT_SET_NAME = {MODE_RAW: "raw", MODE_B: "b115", MODE_BL: "bl115"}      # 꼬리표 (= 보정의 종류) → 보정 세트의 이름

# ── 자리 ──
NORM_SETS_DIR = v6cfg.OUTPUT_DIR / "cache" / "v8_norm"          # 보정 세트 (보정한 사진 + 원본과 같은 라벨)
NORM_FIG_DIR = v6cfg.FIGURE_DIR / "v8_norm"                     # 보정 세트의 점검 표 · 시트
CHECK_FIG_DIR = v6cfg.FIGURE_DIR / "v8_check"                   # 0단계 (학습 없는 민감도 점검)
COMPARE_FIG_DIR = v6cfg.FIGURE_DIR / "v8_compare"               # 기준선과 v8 실행의 비교 표
NORM_JSON = "norm.json"                                         # 보정 세트 안의 기록 (소스마다의 값 · 지도 · 안전 장치)
RUN_NORM_JSON = "norm_set.json"                                 # 실행의 그림 폴더에 남기는 기록 (어느 보정 세트로 학습했는가)


def resolve_norm_root(name_or_path: str) -> Path:
    """`--norm-set` 의 값 → 보정 세트 폴더. 폴더 경로를 주거나 `cache/v8_norm/` 아래의 이름을 준다."""
    p = Path(name_or_path)
    if p.is_dir() and (p / NORM_JSON).is_file():
        return p.resolve()
    q = NORM_SETS_DIR / name_or_path
    if q.is_dir() and (q / NORM_JSON).is_file():
        return q
    raise FileNotFoundError(
        f"FATAL: 보정 세트 '{name_or_path}' 없음 — 폴더 경로도 아니고 {NORM_SETS_DIR} 아래에도 없다 "
        f"(v8/docker 의 DEFSEG_V8_PHASE=norm 이 만든다)"
    )


def default_run_name(backbone: str = v6cfg.DINO_BACKBONE, img_size: int = v6cfg.IMG_SIZE,
                     variant: str = MODE_B, seed: int = v6cfg.RANDOM_SEED) -> str:
    """`<backbone>_dpt_dual_sz<img>_10cls_v8-<꼬리표>-s<seed>` (예: vitb14_reg_dpt_dual_sz1036_10cls_v8-B-s42).

    ckpt 와 그림 폴더는 v6 의 함수 (`run_ckpt_dir` · `run_fig_dir`) 로 만든다 — v6 의 채점 · 요약이 `--run-name` 만으로 찾게.
    그래서 그림은 `figures/<run>/v6/…` 에 쌓인다 (v7 과 같다).
    """
    base = v6cfg.default_run_name(backbone, img_size)
    if not base.endswith("_v6"):
        raise RuntimeError(f"v6 의 run 이름 규칙이 바뀌었다: {base}")
    return f"{base[:-3]}_v8-{variant}-s{int(seed)}"
