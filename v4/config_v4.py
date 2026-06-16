"""DefSeg-AM v4 — 11-class + class-aware sampler + Copy-Paste + ViT-B/14_reg.

v4 의 설계 결정은 [PLAN_v4.md](PLAN_v4.md) 참조.

핵심 변경 (vs v3):
- 11-class (v3 의 10 + Under Melting 신규)
- Backbone: dinov2_vits14 → dinov2_vitb14_reg
- Per-GPU batch 2 → 4, LR 2e-4 → 2.83e-4 (sqrt scaling)
- Sampler: defect_ratios ** 0.5 → class-aware (inverse-frequency, α=1.0)
- Copy-Paste augmentation (rare class object 합성)
- 후처리 규칙 2 변경: Printed 위 Spatter (overlap≥50%) → Over Melting
"""
from __future__ import annotations

from pathlib import Path

from ..common import config as _v1
from ..common.config import (  # re-export 공통 path/상수
    PROJECT_ROOT,
    ORNL_HDF5_DIR,
    ORNL_BUILD_FILES,
    ORNL_TRAIN_BUILDS,
    ORNL_VAL_BUILDS,
    DSCNN_ROOT,
    IGNORE_INDEX,
    INTERMEDIATE_LAYERS,
    DECODER_CHANNELS,
    IMG_SIZE,
    IMAGENET_MEAN,
    IMAGENET_STD,
    ORNL_LAYER_LO_FRAC,
    ORNL_LAYER_HI_FRAC,
    DEFECT_PIXEL_MIN,
    N_INFER_LAYERS,
    NUM_WORKERS,
)

# ============================================================================
# v4 — Backbone (변경: vits14 → vitb14_reg)
# ============================================================================
DINO_BACKBONE = "dinov2_vitb14_reg"   # 86 M frozen + Registers 4 token


# ============================================================================
# v4 — 11 class (PLAN_v4 §2)
# ============================================================================
ORNL_CLASS_NAMES_V4 = [
    "Powder",                # 0
    "Printed",               # 1
    "Recoater Hopping",      # 2
    "Recoater Streaking",    # 3
    "Incomplete Spreading",  # 4  (v3 에서 제거됐다 v4 복원)
    "Swelling",              # 5
    "Debris",                # 6
    "Super-Elevation",       # 7
    "Spatter",               # 8
    "Over Melting",          # 9
    "Under Melting",         # 10 (v4 신규 학습 대상)
]
N_CLASSES_V4 = len(ORNL_CLASS_NAMES_V4)  # 11

# v1 ORNL 12-class ID → v4 11-class ID (-1 = IGNORE)
ORNL_12_TO_NEW_11: dict[int, int] = {
    0:  0,    # Powder
    1:  1,    # Printed
    2:  2,    # Recoater Hopping
    3:  3,    # Recoater Streaking
    4:  4,    # Incomplete Spreading
    5:  5,    # Swelling
    6:  6,    # Debris
    7:  7,    # Super-Elevation
    8:  8,    # Spatter
    9: -1,    # Misprint              → IGNORE (유일)
    10: 9,    # Over Melting
    11: 10,   # Under Melting         (v4 신규)
}

# Powder/Printed 가 아닌 "결함" 인덱스 (sampler weight 계산용)
DEFECT_CLASS_INDICES_V4 = list(range(2, N_CLASSES_V4))  # 2..10


# ============================================================================
# v4 — DSCNN_Dataset native → v1 12-class 매핑 (v3 의 것 그대로 import)
# ============================================================================
MATERIAL_TO_ORNL_V4: dict[str, dict[int, int]] = dict(_v1.MATERIAL_TO_ORNL)

# Binder Jet 2 source 매핑 (v2/v3 와 동일)
MATERIAL_TO_ORNL_V4["v2021_BJ"] = {
    0: 0,   # Un-Fused Powder    → ORNL Powder
    1: 1,   # Printed Material   → ORNL Printed
    2: 3,   # Recoater Streaking → ORNL Recoater Streaking
    3: 4,   # Powder Short Feed  → ORNL Incomplete Spreading
    4: -1,  # (no examples)
    5: 6,   # Debris             → ORNL Debris
}
MATERIAL_TO_ORNL_V4["v2022_BJ_H13"] = {
    0: 0, 1: 1,
    2: 3,
    3: 6,
    4: 4,
    5: -1, 6: -1, 7: -1,
}


# ============================================================================
# v4 — DSCNN_Dataset 8 source (LPBF 6 + BJ 2, EBPBF 제외; v3 와 동일)
# ============================================================================
DSCNN_TRAIN_SOURCES_V4 = [
    *_v1.DSCNN_TRAIN_SOURCES,
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


# ============================================================================
# v4 — Augmentation 상수 (PLAN_v4 §6)
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

# v4 신규 — Copy-Paste (CutMix segmentation 발전형)
CP_ENABLE = True
CP_PROB = 0.5
# Stage 별 rare class 분리 (PLAN_v4 §6.2)
CP_RARE_CLASSES_S1 = (5, 7)                          # Swelling, SE
CP_RARE_CLASSES_S2 = (2, 4, 5, 6, 7, 9, 10)          # 위 + Hopping, Inc.Spr., Debris, Over Melt, Under Melt
CP_MIN_COMPONENT_PX = 30
CP_MAX_OBJECTS_PER_PASTE = 3
CP_FEATHER_SIGMA = 5
CP_BBOX_MAX_FRAC = 0.3


# ============================================================================
# v4 — Class-Aware Sampler (PLAN_v4 §4)
# ============================================================================
CLASS_AWARE_ALPHA = 1.0
CLASS_AWARE_EPS = 1e-3


# ============================================================================
# v4 — Stage 1 (KD pretrain) hyper-params
# ============================================================================
S1_EPOCHS = 30
S1_BATCH_SIZE = 4                # per-GPU. 4-GPU DDP → effective 16
S1_LR = 2.83e-4                  # sqrt scaling: 2e-4 × √2
S1_WEIGHT_DECAY = 1e-4
S1_FOCAL_GAMMA = 2.0
S1_OVERSAMPLE_EPS = 1e-3
S1_WARMUP_STEPS = 200
S1_GRAD_CLIP_NORM = 1.0
S1_CLASS_WEIGHT_CLIP = 10.0

EMA_DECAY = 0.9999
HARD_BOOTSTRAP_LAMBDA = 0.8


# ============================================================================
# v4 — Stage 2 hyper-params
# ============================================================================
S2_EPOCHS = 50
S2_BATCH_SIZE = 4
S2_LR = 2.83e-4
S2_WEIGHT_DECAY = 1e-4
S2_WARMUP_STEPS = 50
S2_GRAD_CLIP_NORM = 1.0
S2_CLASS_WEIGHT_CLIP = 10.0

S2_REPLICATE_FACTOR = 4
S2_RANDOM_SEED = 42
S2_EVAL_BUILD = "2021-07-13 TCR Phase 1 Build 1"
S2_EVAL_N_LAYERS = 200


# ============================================================================
# v4 — Output paths
# ============================================================================
OUTPUT_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURE_DIR = OUTPUT_DIR / "figures"
CACHE_DIR = OUTPUT_DIR / "cache"

V4_CACHE_DIR_NAME = "resized_sz{img_size}_v4"
V1_CACHE_DIR_NAME = "resized_sz{img_size}"


def v1_cache_dir(img_size: int = IMG_SIZE) -> Path:
    return CACHE_DIR / V1_CACHE_DIR_NAME.format(img_size=img_size)


def v4_cache_dir(img_size: int = IMG_SIZE) -> Path:
    return CACHE_DIR / V4_CACHE_DIR_NAME.format(img_size=img_size)


# ============================================================================
# v4 — Inference 휴리스틱 후처리 (PP_)
# ============================================================================
# 규칙 1: SE/Swelling 이 Printed 에서 멀면 → Debris (v3 와 동일 동작, ID 시프트)
PP_SE_SOURCE_CLASSES = (5, 7)            # Swelling, Super-Elevation
PP_SE_TARGET_CLASS = 6                   # Debris
PP_SE_FAR_DISTANCE_PX = 100
PP_SE_MIN_COMPONENT_PX = 30

# 규칙 2: Printed 위의 Spatter (overlap≥50%) → Over Melting (신규)
# pred 가 mutual exclusive (한 픽셀당 한 클래스) 이므로 Printed mask 를 약간
# dilation 시킨 후 Spatter component 와의 overlap 측정.
PP_PS_SOURCE_CLASS = 8                   # Spatter
PP_PS_TARGET_CLASS = 9                   # Over Melting
PP_PS_OVERMELT_OVERLAP_FRAC = 0.5
PP_PS_MIN_COMPONENT_PX = 30
PP_PS_PRINTED_DILATE_PX = 10             # Printed mask 의 "주변" 정의 (boundary 가 가까운 Spatter)
