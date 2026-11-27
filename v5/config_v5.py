"""DefSeg-AM v5 — v4 의 11-class 그대로 + 사람 GT 전용 단일 단계 학습.

설계는 [PLAN_v5.md](PLAN_v5.md) 참조.

핵심 변경 (vs v4):
- Stage 1 (ORNL HDF5 의 DSCNN 예측 라벨로 하는 KD) 이 없다. decoder 는 무작위 초기값에서 출발
- 학습 데이터 = DSCNN_Dataset 의 사람 주석 8 소스 125 레이어뿐 (v4 Stage 2 와 같은 데이터 · 같은 매핑)
- 길이를 epoch 이 아니라 optimizer step 으로 고정, EMA decay 0.9999 → 0.999
- 검증 · ckpt 선정을 ORNL Build 1 (DSCNN 예측) 이 아니라 떼어 둔 사람 GT 로
- Sampler: 소스 균등 × class-aware 의 곱 → 소스 균등과 클래스 균등의 혼합 + 표본당 상한

클래스 · backbone · 입력 크기 · augmentation · Copy-Paste · loss · 후처리는 v4 와 같다.
"""
from __future__ import annotations

import os
from pathlib import Path

from ..common import config as _common
from ..common.config import (  # re-export 공통 path/상수
    PROJECT_ROOT,
    ORNL_HDF5_DIR,
    ORNL_BUILD_FILES,
    DSCNN_ROOT,
    IGNORE_INDEX,
    INTERMEDIATE_LAYERS,
    DECODER_CHANNELS,
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ORNL_LAYER_LO_FRAC,
    ORNL_LAYER_HI_FRAC,
    N_INFER_LAYERS,
    NUM_WORKERS,
)

# ============================================================================
# v5 — Backbone (v4 와 같음)
# ============================================================================
DINO_BACKBONE = "dinov2_vitb14_reg"   # 86 M frozen + Registers 4 token


# ============================================================================
# v5 — 11 class (v4 와 같은 이름 · 같은 번호, PLAN_v5 §2)
# ============================================================================
CLASS_NAMES_V5 = [
    "Powder",                # 0
    "Printed",               # 1
    "Recoater Hopping",      # 2
    "Recoater Streaking",    # 3
    "Incomplete Spreading",  # 4
    "Swelling",              # 5
    "Debris",                # 6
    "Super-Elevation",       # 7
    "Spatter",               # 8
    "Over Melting",          # 9
    "Under Melting",         # 10
]
N_CLASSES_V5 = len(CLASS_NAMES_V5)  # 11

# 로그 · 표에 쓰는 짧은 이름
CLASS_ABBRS_V5 = ["Pow", "Prt", "RH", "RS", "IS", "Sw", "DB", "SE", "SP", "OM", "UM"]

# ORNL 12-class ID → v5 11-class ID (-1 = IGNORE). v4 의 ORNL_12_TO_NEW_11 과 같다.
ORNL_12_TO_V5: dict[int, int] = {
    0:  0,    # Powder
    1:  1,    # Printed
    2:  2,    # Recoater Hopping
    3:  3,    # Recoater Streaking
    4:  4,    # Incomplete Spreading
    5:  5,    # Swelling
    6:  6,    # Debris
    7:  7,    # Super-Elevation
    8:  8,    # Spatter
    9: -1,    # Misprint              → IGNORE
    10: 9,    # Over Melting
    11: 10,   # Under Melting
}

# Powder/Printed 가 아닌 "결함" 인덱스
DEFECT_CLASS_INDICES_V5 = list(range(2, N_CLASSES_V5))  # 2..10


# ============================================================================
# v5 — 사람 GT: DSCNN_Dataset native → ORNL 12-class 매핑 (v4 와 같음)
# ============================================================================
MATERIAL_TO_ORNL_V5: dict[str, dict[int, int]] = dict(_common.MATERIAL_TO_ORNL)

MATERIAL_TO_ORNL_V5["v2021_BJ"] = {
    0: 0,   # Un-Fused Powder    → ORNL Powder
    1: 1,   # Printed Material   → ORNL Printed
    2: 3,   # Recoater Streaking → ORNL Recoater Streaking
    3: 4,   # Powder Short Feed  → ORNL Incomplete Spreading
    4: -1,  # (no examples)
    5: 6,   # Debris             → ORNL Debris
}
MATERIAL_TO_ORNL_V5["v2022_BJ_H13"] = {
    0: 0, 1: 1,
    2: 3,
    3: 6,
    4: 4,
    5: -1, 6: -1, 7: -1,
}


# ----------------------------------------------------------------------------
# Soot 주석을 어디로 보내는가 (2026-11-20 에 더한 선택지)
#   "ignore"   지금까지의 v5 — Soot 는 미레이블이다 (기본값. 이 값이면 v5 의 계산은 전과 같다)
#   "spatter"  Soot 를 Spatter 로 배운다. v6 가 Soot 를 Spatter (SP) 로 읽으므로, v5 와 v6 를 Spatter 에서 견주려고 둔 것이다.
#              v5 는 이 주석을 미레이블로 두고 배워서 Soot 처럼 생긴 영역에 Spatter 를 내지 않는다 (v6/RESULTS_v6.md §1.2).
# 환경변수 DEFSEG_V5_SOOT 로 고른다 — 학습 · 채점 · 요약이 같은 매핑을 쓰게 하려고 인자가 아니라 환경변수로 뒀다.
# "spatter" 면 run 이름에 꼬리 (-soot2sp) 가 붙어 기존 v5 의 산출물과 섞이지 않는다.
# ----------------------------------------------------------------------------
SOOT_MODE = (os.environ.get("DEFSEG_V5_SOOT", "ignore").strip().lower() or "ignore")
SOOT_NATIVE_ID = 8                         # native 주석에서 Soot 의 번호 (아래 네 매핑에서 같다)
SOOT_MAPPING_KEYS = ("v2021_LPBF", "17-4_PH_Stainless_Steel", "Inconel_718_2", "Maraging_Steel")
_ORNL12_SPATTER = 8
if SOOT_MODE == "spatter":
    for _k in SOOT_MAPPING_KEYS:
        _m = dict(MATERIAL_TO_ORNL_V5[_k])          # 새 dict — common 의 표를 건드리지 않는다
        if _m.get(SOOT_NATIVE_ID) != IGNORE_INDEX:
            raise RuntimeError(f"{_k} 의 native {SOOT_NATIVE_ID} 가 미레이블이 아니다 — Soot 가 아닌 것을 바꾸려 한다")
        _m[SOOT_NATIVE_ID] = _ORNL12_SPATTER
        MATERIAL_TO_ORNL_V5[_k] = _m
elif SOOT_MODE != "ignore":
    raise ValueError(f"DEFSEG_V5_SOOT 는 ignore | spatter 가운데 하나다: {SOOT_MODE!r}")
RUN_SUFFIX = "" if SOOT_MODE == "ignore" else "-soot2sp"


# ============================================================================
# v5 — 사람 GT 8 소스 (LPBF 6 + BJ 2, EBPBF 제외; v3/v4 와 같음)
# ============================================================================
GT_SOURCES_V5 = [
    *_common.DSCNN_TRAIN_SOURCES,
    {
        "name": "v2021_BJ",
        "root": DSCNN_ROOT / "Peregrine Dataset v2021-03" / "Binder Jet",
        "mapping_key": "v2021_BJ",
    },
    {
        "name": "v2022_BJ_H13",
        "root": DSCNN_ROOT / "Peregrine Dataset v2022-10.1/Binder_Jet/ExOne_M-Flex/H13_Steel/training",
        "mapping_key": "v2022_BJ_H13",
    },
]

# 소스 약어 (fold 이름 · 표에 쓴다. docs/DEFECT_TAXONOMY.md §5.2 와 같다)
SOURCE_SHORT = {
    "v2021_LPBF":       "L21",
    "v2021_BJ":         "BJ21",
    "v2022_17-4PH":     "17-4",
    "v2022_GammaPrint": "GP",
    "v2022_Inc718_1":   "I1",
    "v2022_Inc718_2":   "I2",
    "v2022_Maraging":   "MS",
    "v2022_BJ_H13":     "H13",
}


# ============================================================================
# v5 — 검증 fold (PLAN_v5 §4)
# ============================================================================
# ("source", <소스 이름>)  : 그 소스 전체를 검증용으로 뺀다 (leave-one-source-out)
# ("layers", n, k)        : 소스마다 레이어를 이름순으로 세어 n 개 중 k 번째를 뺀다 (소스 안 분할)
CV_FOLDS: dict[str, tuple] = {
    f"loso_{short}": ("source", name) for name, short in SOURCE_SHORT.items()
}
CV_FOLDS["inlayer"] = ("layers", 5, 4)

# compose 의 cv phase 가 기본으로 도는 fold.
# loso 4개는 둘 이상의 소스에 있는 결함 6종 (RS · IS · Swelling · Debris · SE · Spatter) 을 한 번 이상 채점한다.
DEFAULT_CV_FOLDS = ("loso_L21", "loso_MS", "loso_I2", "loso_GP", "inlayer")


# ============================================================================
# v5 — Augmentation 상수 (v4 와 같음)
# ============================================================================
DSCNN_NOISE_SIGMA_PCT_CHOICES = (0.0, 0.01, 0.1)
DSCNN_INTENSITY_SHIFT_PCT_CHOICES = (0.0, +0.10, -0.10)
ENABLE_FLIP_LR = True
ENABLE_FLIP_UD = True
ENABLE_ROT180 = True
ENABLE_CYCLIC_SHIFT = True
CYCLIC_SHIFT_MAX_FRAC = 0.25
CYCLIC_SHIFT_PROB = 0.5
ENABLE_BRIGHTNESS_JITTER = True
BRIGHTNESS_JITTER_RANGE = (0.85, 1.15)

# Copy-Paste (v4 Stage 2 와 같음 — supplier 는 학습 분할의 사람 GT)
CP_ENABLE = True
CP_PROB = 0.5
CP_RARE_CLASSES = (2, 4, 5, 6, 7, 9, 10)   # Hopping, Inc.Spr., Swelling, Debris, SE, Over Melt, Under Melt
CP_MIN_COMPONENT_PX = 30
CP_MAX_OBJECTS_PER_PASTE = 3
CP_FEATHER_SIGMA = 5
CP_BBOX_MAX_FRAC = 0.3


# ============================================================================
# v5 — Sampler (PLAN_v5 §5.4)
# ============================================================================
# p_i = MIX × (소스 균등) + (1 − MIX) × (클래스 균등),  그 뒤 한 표본의 몫을 MAX_REPEAT / N 으로 제한
SAMPLER_SOURCE_MIX = 0.5
SAMPLER_MAX_REPEAT = 8.0


# ============================================================================
# v5 — 학습 hyper-params (단일 단계, step 기준 — PLAN_v5 §5)
# ============================================================================
TOTAL_STEPS = 10_000             # 상한. 검증 fold 의 곡선을 보고 확정한다
EVAL_EVERY = 250                 # 검증 주기 (step)
WARMUP_STEPS = 300
BATCH_SIZE = 4                   # per-GPU. 4-GPU DDP → effective 16
LR = 2.83e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
CLASS_WEIGHT_CLIP = 10.0
EMA_DECAY = 0.999                # 시상수 1,000 step ≤ TOTAL_STEPS / 10
RANDOM_SEED = 42

# ORNL Build 1 참고 평가 (DSCNN 예측과의 일치도 — 학습 · 선정에는 쓰지 않는다)
REF_EVAL_BUILD = "2021-07-13 TCR Phase 1 Build 1"


# ============================================================================
# v5 — Output paths
# ============================================================================
OUTPUT_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURE_DIR = OUTPUT_DIR / "figures"
CACHE_DIR = OUTPUT_DIR / "cache"

# 참고 평가 (confusion phase) 만 캐시를 읽는다: 이미지는 v1, 라벨은 v4 캐시 (클래스 공간이 같다).
V1_CACHE_DIR_NAME = "resized_sz{img_size}"
V4_CACHE_DIR_NAME = "resized_sz{img_size}_v4"


def v1_cache_dir(img_size: int = IMG_SIZE) -> Path:
    return CACHE_DIR / V1_CACHE_DIR_NAME.format(img_size=img_size)


def v4_cache_dir(img_size: int = IMG_SIZE) -> Path:
    return CACHE_DIR / V4_CACHE_DIR_NAME.format(img_size=img_size)


def default_run_name(backbone: str = DINO_BACKBONE, img_size: int = IMG_SIZE) -> str:
    """`<backbone>_dpt_dual_sz<img>_11cls_v5` (예: vitb14_reg_dpt_dual_sz1036_11cls_v5).

    DEFSEG_V5_SOOT=spatter 면 끝에 `-soot2sp` 가 붙는다 (Soot 를 Spatter 로 배운 실행).
    """
    short = backbone[len("dinov2_"):] if backbone.startswith("dinov2_") else backbone
    return f"{short}_dpt_dual_sz{img_size}_{N_CLASSES_V5}cls_v5{RUN_SUFFIX}"


def run_tag(fold: str | None) -> str:
    """전수 학습은 `full`, 검증 fold 는 `cv/<fold>`."""
    return "full" if fold is None else f"cv/{fold}"


def run_ckpt_dir(run_name: str, fold: str | None = None) -> Path:
    return CHECKPOINT_DIR / run_name / run_tag(fold)


def run_fig_dir(run_name: str, fold: str | None = None) -> Path:
    return FIGURE_DIR / run_name / "v5" / run_tag(fold)


# ============================================================================
# v5 — Inference 휴리스틱 후처리 (v4 와 같음)
# ============================================================================
# 규칙 1: SE/Swelling 이 Printed 에서 멀면 → Debris
PP_SE_SOURCE_CLASSES = (5, 7)            # Swelling, Super-Elevation
PP_SE_TARGET_CLASS = 6                   # Debris
PP_SE_FAR_DISTANCE_PX = 100
PP_SE_MIN_COMPONENT_PX = 30

# 규칙 2: Printed 위의 Spatter (overlap≥50%) → Over Melting
PP_PS_SOURCE_CLASS = 8                   # Spatter
PP_PS_TARGET_CLASS = 9                   # Over Melting
PP_PS_OVERMELT_OVERLAP_FRAC = 0.5
PP_PS_MIN_COMPONENT_PX = 30
PP_PS_PRINTED_DILATE_PX = 10
