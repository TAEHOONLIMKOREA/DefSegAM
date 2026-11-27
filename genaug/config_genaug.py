"""GenAug 설정 — 결함 인페인팅 확산 모델 (PLAN_GENAUG.md §3.2).

버전 폴더 (v1–v6) 를 임포트하지 않는다. 의존은 `common` 뿐이다.

읽는 자료:
  - 사람 GT: `ORNL_Data/DSCNN_Dataset_pbf-2026.11/` (v6 가 변환해 둔 10-class 레이블 데이터셋)
  - 주석 없는 사진: `cache/ornl_canvas_sz1036/` (공용 배경 캐시 — 원본 밝기 사진 + CAD 파트 마스크)
DSCNN 예측 (`slices/segmentation_results`, v1 캐시의 `label.npy` · `defect_ratios`) 은 어디서도 읽지 않는다.
"""
from __future__ import annotations

from pathlib import Path

from ..common.config import (  # noqa: F401  (re-export)
    CACHE_DIR,
    CHECKPOINT_DIR,
    FIGURE_DIR,
    IMG_SIZE,
    ORNL_BUILD_FILES,
    ORNL_HDF5_DIR,
    PROJECT_ROOT,
)

HERE = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# 클래스 — pbf-2026.11 (10-class). 데이터셋의 classes.json 과 대조한다 (data/gt_pool.py)
# ---------------------------------------------------------------------------
SCHEMA_VERSION = "pbf-2026.11"
CLASS_KEYS = [
    "powder", "printed", "incomplete_spreading", "powder_slumping", "recoater_streaking",
    "recoater_hopping", "super_elevation", "spatter", "over_melting", "debris",
]
CLASS_ABBRS = ["Pow", "Prt", "IS", "PS", "RS", "RH", "SE", "SP", "OM", "DB"]
N_CLASSES = len(CLASS_KEYS)          # 10
IGNORE_INDEX = -1                    # 미레이블
POWDER, PRINTED = 0, 1
UNKNOWN = N_CLASSES                  # 라벨 맵의 "모름" 채널 (10). 미레이블 화소 · 그릴 자리 밖 · 조건을 뗀 표본
N_LABEL_CH = N_CLASSES + 1           # 11

# 생성 대상 — 창에 들어가는 작은 객체로 된 결함 (PLAN §3.6): DB · SE · SP · RH · OM
GEN_CLASSES = (9, 6, 7, 5, 8)

# 표시용 색 (시트의 윤곽선). DIVA 의 색과 같을 필요는 없다
CLASS_COLORS = [
    (120, 120, 120), (255, 255, 255), (0, 170, 255), (0, 90, 200), (255, 220, 0),
    (255, 140, 0), (255, 0, 255), (0, 220, 120), (255, 60, 60), (170, 110, 255),
]

# ---------------------------------------------------------------------------
# 사람 GT 와 보면 안 되는 레이어
# ---------------------------------------------------------------------------
GT_ROOT = PROJECT_ROOT / "ORNL_Data" / f"DSCNN_Dataset_{SCHEMA_VERSION}"
HOLDOUT_FILE = HERE / "splits" / "holdout_v1.json"
# holdout 종류 → 뺀 뒤 남아야 하는 레이어 수. 다르면 멈춘다
#   both    = inlayer 의 검증 22 + L21 의 20 (4 겹침) 을 뺀 87 — 두 판정 fold 에 모두 쓸 수 있다 (PLAN §5.1)
#   inlayer = inlayer 의 검증 22 만 뺀 103 — inlayer 만으로 판정할 때 (PLAN §8 의 G7 의 대안)
HOLDOUT_EXPECTED_LAYERS = {"both": 87, "inlayer": 103}
DEFAULT_HOLDOUT = "both"

MIN_OBJECT_PX = 30                   # 객체로 세는 최소 크기 (v5 Copy-Paste 와 같다)

# ---------------------------------------------------------------------------
# 공용 배경 캐시 — rulesynth/PLAN_RULESYNTH.md §2.3 의 규격
# ---------------------------------------------------------------------------
CANVAS_ROOT = CACHE_DIR / f"ornl_canvas_sz{IMG_SIZE}"
CANVAS_BUILDS = ("B1", "B2", "B3", "B4", "B5")      # 캐시는 다섯 빌드를 모두 만든다
CANVAS_LAYER_LO_FRAC = 0.05                          # v1 캐시와 같은 높이 구간
CANVAS_LAYER_HI_FRAC = 0.95
CANVAS_LAYER_EVERY = 5                               # 5층마다 1층
HDF5_SLAB = 52                                       # HDF5 의 층 방향 chunk 크기 — 이 단위로 읽어야 한 번만 푼다
PART_EDGE_PX = 3                                     # CAD 와 사진의 어긋남 — 파트 경계 ±3 px 는 라벨을 주지 않는다

# 생성 모델 학습에 쓰는 빌드 (PLAN §1.1, §8 의 G6). B1 은 비교 빌드로 남기고 B2 는 보류
TRAIN_BUILDS = ("B3", "B4", "B5")
# 라벨 맵에 Powder / Printed 를 적어 주는 빌드 — Q0 (사람이 보는 배경 점검) 을 통과한 빌드만 (PLAN §3.2, §4).
# Q0 을 보기 전의 잠정값이다. 학습 명령의 --normal-builds 로 바꾼다 ("none" = 적지 않는다)
NORMAL_LABEL_BUILDS = ("B3",)
EDGE_FRAC = 0.05                                     # 사진 가장자리 5% 는 창을 놓지 않는다 (판 밖 · 모서리의 밝은 얼룩)

# 덮임 판정 (scripts/synth_prototype/covered_rule.py 의 값)
COVERED_TAU = 3.0
COVERED_WIN = 61
COVERED_GROW = 3
COVERED_PAD = 64                                     # 창 둘레에 이만큼 더 잘라 판정한다 (지역 통계의 가장자리 효과)

# ---------------------------------------------------------------------------
# 창 · 그릴 자리
# ---------------------------------------------------------------------------
WIN = 256
BORDER = 16                          # "창 전체 그리기" 에서 진짜로 남기는 테두리
RING_PX = 6                          # 결함 둘레의 띠 (그리지만 라벨은 주지 않는 곳)

P_LABELED = 0.5                      # 학습 창 가운데 사람 GT 의 몫. 나머지는 주석 없는 ORNL 창
P_DEFECT_CENTERED = 0.7              # 사람 GT 창 가운데 결함 객체를 가운데 둔 것의 몫
COND_DROP = 0.15                     # 라벨 맵을 전부 "모름" 으로 바꾸는 비율 (guidance 용)
GAIN_RANGE = (0.85, 1.15)            # 밝기 흔들기 (두 사진에 같은 배율)

# 그릴 자리의 종류와 비율. 합 = 1
#   object = 결함 객체 + 띠,  box = 무작위 사각형,  blob = 무작위 덩어리,
#   full = 테두리만 남긴 창 전체,  full0 = 테두리도 없는 창 전체 (나중에 창을 이어 그릴 때 쓴다)
REGION_MIX_DEFECT = {"object": 0.50, "box": 0.15, "blob": 0.10, "full": 0.20, "full0": 0.05}
REGION_MIX_OTHER = {"box": 0.30, "blob": 0.25, "full": 0.35, "full0": 0.10}

# ---------------------------------------------------------------------------
# 모델 · 확산 과정 · 학습
# ---------------------------------------------------------------------------
IMG_CH = 2                           # visible/0, visible/1
MODEL_IN_CH = IMG_CH + IMG_CH + 1 + N_LABEL_CH      # 잡음 섞인 창 2 + 가린 창 2 + 그릴 자리 1 + 라벨 맵 11 = 16
BASE_CH = 64
CH_MULT = (1, 2, 4, 4, 8, 8)         # 해상도 256 → 128 → 64 → 32 → 16 → 8 에서 채널 64 · 128 · 256 · 256 · 512 · 512.
                                     # 256² 에서 128 채널을 쓰는 원래 구성 (base 128, mult 1·1·2·2·4·4) 은 batch 16 도 32 GB 에 안 들어간다 (2026-11-17 측정)
N_RES_BLOCKS = 2
ATTN_RESOLUTIONS = (16, 8)
ATTN_HEADS = 4
DROPOUT = 0.0

T_STEPS = 1000                       # 확산 단계 수 (cosine 일정, v-prediction)
MIN_SNR_GAMMA = 5.0
OUTSIDE_LOSS_WEIGHT = 0.05           # 그릴 자리 밖의 손실 가중 (입력으로 주어진 화소라 작게)

TOTAL_STEPS = 100_000                # 출발값 (제안값). 표본 그림과 관문으로 정한다
BATCH_SIZE = 16                      # GPU 당. bfloat16 로 23 GB 를 쓴다 (2026-11-17 측정, RTX 5090). 32 는 들어가지 않는다
USE_AMP = True                       # bfloat16 autocast — 같은 batch 에서 1.45배 빠르고 손실 값은 fp32 와 같았다
LR = 1e-4
WEIGHT_DECAY = 0.0
WARMUP_STEPS = 1000
GRAD_CLIP_NORM = 1.0
EMA_DECAY = 0.9999
LOG_EVERY = 100
SAMPLE_EVERY = 5000
SAVE_EVERY = 5000
KEEP_EVERY = 25_000                  # 이 간격의 EMA 사본은 지우지 않고 남긴다
SEED = 42

SAMPLE_STEPS = 50                    # 그릴 때의 단계 수 (DDIM)
SAMPLE_CFG = 2.0                     # guidance 배율

RUN_NAME = "fill_w256"

# ---------------------------------------------------------------------------
# 합성 세트 — 라벨 맵으로 그린 레이어 (synth/, v7/PLAN_v7.md §2)
# ---------------------------------------------------------------------------
SETS_DIR = CACHE_DIR / "genaug" / "sets"            # 세트 하나 = 사람 GT 데이터셋과 같은 구성의 폴더 (v7 이 읽는다)
SET_NAME = "w600_v1"
SET_LAYERS = 600
SET_BUILDS = TRAIN_BUILDS                            # 배경 (CAD 배치 + 사진 가장자리) 으로 쓰는 빌드 — 생성 모델이 본 빌드
SET_SEED = 20261119
SET_CLASSES = (4, 5, 6, 9)                           # 넣는 결함 — RS · RH · SE · DB (PLAN_v7 §1.3 의 인식 시험을 통과한 것)
SET_GROUP_PREFIX = "synth_"                          # index.csv 의 process_code = synth_<빌드> (v7 의 config_v7.SYNTH_GROUP_PREFIX 와 같아야 한다)
SET_SPACING = "quad"                                 # 그리는 단계의 배치 (models/diffusion.py 의 sample_steps)
SET_STRIDE = 192                                     # 창을 놓는 간격의 상한 (창 256 → 이웃과 64 px 이상 겹친다)
SET_ORDER = "spiral"                                 # 창을 그리는 순서 — spiral = 진짜 가장자리에서 안쪽으로, raster = 왼쪽 위부터 줄 따라
SET_BATCH = 8                                        # 한 번에 그리는 레이어 수 (GPU 당)


def set_dir(name: str) -> Path:
    return SETS_DIR / name


def set_fig_dir(name: str) -> Path:
    return FIGURE_DIR / "genaug" / "sets" / name


def ckpt_dir(run_name: str) -> Path:
    return CHECKPOINT_DIR / "genaug" / run_name


def fig_dir(run_name: str) -> Path:
    return FIGURE_DIR / "genaug" / run_name
