"""DefSeg-AM v3 — 10-class + 강화 augmentation + 단일 ckpt 학습 + 확장 PP.

v3 의 설계 결정은 [PLAN_v3.md](PLAN_v3.md) 참조.

핵심 변경:
- 10-class (Recoater Hopping/Streaking 분리 복원, Incomplete Spreading 복원)
- Augmentation = flip LR/UD + 180° rot + cyclic shift (1-px uniform) + DSCNN
- Stage 2: DSCNN_Dataset 8 source 전부 train, ORNL Build 1 로 평가 (CV 없음)
- 후처리: 부품에서 멀리 떨어진 SE/Swelling → Debris
"""
from __future__ import annotations

from pathlib import Path

# 공통 상수 그대로 재사용 (common)
from ..common import config as _v1
from ..common.config import (  # re-export 공통 path/상수
    PROJECT_ROOT,
    ORNL_HDF5_DIR,
    ORNL_BUILD_FILES,
    ORNL_TRAIN_BUILDS,
    ORNL_VAL_BUILDS,
    DSCNN_ROOT,
    IGNORE_INDEX,
    DINO_BACKBONE,
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
# v3 — 10 class (PLAN_v3 §2)
# ============================================================================
ORNL_CLASS_NAMES_V3 = [
    "Powder",               # 0
    "Printed",              # 1
    "Recoater Hopping",     # 2  (v1 old 2)
    "Recoater Streaking",   # 3  (v1 old 3)
    "Incomplete Spreading", # 4  (v1 old 4)
    "Swelling",             # 5  (v1 old 5)
    "Debris",               # 6  (v1 old 6)
    "Super-Elevation",      # 7  (v1 old 7)
    "Spatter",              # 8  (v1 old 8)
    "Over Melting",         # 9  (v1 old 10)
]
N_CLASSES_V3 = len(ORNL_CLASS_NAMES_V3)  # 10

# v1 ORNL 12-class ID → v3 10-class ID (-1 = IGNORE)
ORNL_12_TO_NEW_10: dict[int, int] = {
    0:  0,    # Powder
    1:  1,    # Printed
    2:  2,    # Recoater Hopping     (분리 유지)
    3:  3,    # Recoater Streaking
    4:  4,    # Incomplete Spreading (v2 에서 제거 → v3 복원)
    5:  5,    # Swelling
    6:  6,    # Debris
    7:  7,    # Super-Elevation
    8:  8,    # Spatter
    9: -1,    # Misprint             → IGNORE
    10: 9,    # Over Melting
    11:-1,    # Under Melting        → IGNORE
}

# Powder/Printed 가 아닌 "결함" 인덱스 (sampler oversample 의 defect_ratio 계산용)
DEFECT_CLASS_INDICES_V3 = list(range(2, N_CLASSES_V3))  # 2..9


# ============================================================================
# v3 — DSCNN_Dataset native → v1 12-class 매핑 (+ BJ 2)
# ============================================================================
# data_dscnn_v3.remap_label_v3 가 이걸 사용한 후 ORNL_12_TO_NEW_10 으로 변환.
MATERIAL_TO_ORNL_V3: dict[str, dict[int, int]] = dict(_v1.MATERIAL_TO_ORNL)

# Binder Jet 2 source 매핑 (v2 와 동일)
MATERIAL_TO_ORNL_V3["v2021_BJ"] = {
    0: 0,   # Un-Fused Powder    → ORNL Powder
    1: 1,   # Printed Material   → ORNL Printed
    2: 3,   # Recoater Streaking → ORNL Recoater Streaking
    3: 4,   # Powder Short Feed  → ORNL Incomplete Spreading (v3 복원)
    4: -1,  # (no examples)
    5: 6,   # Debris             → ORNL Debris
}
MATERIAL_TO_ORNL_V3["v2022_BJ_H13"] = {
    0: 0,   # Powder
    1: 1,   # Printed
    2: 3,   # Roller Streaking   → ORNL Recoater Streaking
    3: 6,   # Debris             → ORNL Debris
    4: 4,   # Short Feed         → ORNL Incomplete Spreading (v3 복원)
    5: -1,  # Cornrows           → IGNORE
    6: -1,  # Exposed Part       → IGNORE (불확실)
    7: -1,  # Misprint           → IGNORE
}


# ============================================================================
# v3 — DSCNN_Dataset 8 source 정의 (LPBF 6 + BJ 2, EBPBF 제외; v2 와 동일)
# ============================================================================
DSCNN_TRAIN_SOURCES_V3 = [
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
# v3 — Augmentation 상수 (PLAN_v3 §3 / DSCNN_Summary §6)
# ============================================================================
# Gaussian noise — % of 8-bit dynamic range (=255). DSCNN 원본 0.01% / 0.1%
DSCNN_NOISE_SIGMA_PCT_CHOICES = (0.0, 0.01, 0.1)

# Mean intensity shift — ±10% of DR. DSCNN 원본
DSCNN_INTENSITY_SHIFT_PCT_CHOICES = (0.0, +0.10, -0.10)

# Flip — LR / UD 독립 50%
ENABLE_FLIP_LR = True
ENABLE_FLIP_UD = True

# Rotation — 180° 만 (90/270 제외 — recoater 축 비대칭)
ENABLE_ROT180 = True

# Cyclic shift (np.roll) — 1-px uniform sampling
ENABLE_CYCLIC_SHIFT = True
CYCLIC_SHIFT_MAX_FRAC = 0.25  # shift 범위 = ±(img_size * frac)
CYCLIC_SHIFT_PROB = 0.5       # shift 적용 확률

# Brightness multiplicative jitter (v1/v2 유지)
ENABLE_BRIGHTNESS_JITTER = True
BRIGHTNESS_JITTER_RANGE = (0.85, 1.15)


# ============================================================================
# v3 — Stage 1 (KD pretrain) hyper-params
# ============================================================================
S1_EPOCHS = 30
S1_BATCH_SIZE = 2                # per-GPU. 4-GPU DDP → effective 8
S1_LR = 2e-4                     # sqrt scaling (1e-4 × √4)
S1_WEIGHT_DECAY = 1e-4
S1_FOCAL_GAMMA = 2.0
S1_OVERSAMPLE_POWER = 0.5
S1_OVERSAMPLE_EPS = 1e-3
S1_WARMUP_STEPS = 200
S1_GRAD_CLIP_NORM = 1.0
S1_CLASS_WEIGHT_CLIP = 10.0

# EMA weight saving (DSCNN_Summary §5.6)
EMA_DECAY = 0.9999

# Hard-bootstrapping loss (DSCNN_Summary §5.2, Eq.4) — Stage 1 옵션
HARD_BOOTSTRAP_LAMBDA = 0.8


# ============================================================================
# v3 — Stage 2 (DSCNN_Dataset 전수 finetune) hyper-params
# ============================================================================
S2_EPOCHS = 50
S2_BATCH_SIZE = 2                # per-GPU
S2_LR = 2e-4                     # sqrt scaling
S2_WEIGHT_DECAY = 1e-4
S2_WARMUP_STEPS = 50
S2_GRAD_CLIP_NORM = 1.0
S2_CLASS_WEIGHT_CLIP = 10.0

# Replicate factor — Stage 2 의 epoch 당 effective sample 수를 K 배
S2_REPLICATE_FACTOR = 4

# Stage 2 random seed (augmentation + sampler)
S2_RANDOM_SEED = 42

# Stage 2 평가 — 매 epoch 마다 ORNL Build 1 의 일부 layer 로 mIoU 측정
S2_EVAL_BUILD = "2021-07-13 TCR Phase 1 Build 1"
S2_EVAL_N_LAYERS = 200           # Build 1 layer 균등 샘플링 수


# ============================================================================
# v3 — Output paths
# ============================================================================
OUTPUT_DIR = Path(__file__).resolve().parents[1]   # = DefSeg_AM/
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURE_DIR = OUTPUT_DIR / "figures"
CACHE_DIR = OUTPUT_DIR / "cache"

# v3 의 label cache 만 별도 디렉터리 (image 는 v1 그대로 재사용)
V3_CACHE_DIR_NAME = "resized_sz{img_size}_v3"
V1_CACHE_DIR_NAME = "resized_sz{img_size}"


def v1_cache_dir(img_size: int = IMG_SIZE) -> Path:
    return CACHE_DIR / V1_CACHE_DIR_NAME.format(img_size=img_size)


def v3_cache_dir(img_size: int = IMG_SIZE) -> Path:
    return CACHE_DIR / V3_CACHE_DIR_NAME.format(img_size=img_size)


# ============================================================================
# v3 — Inference 휴리스틱 후처리 (PP_)
# ============================================================================
# PP_SE_SWELLING_FAR — SE/Swelling 의 component 가 Printed 에서 멀면 Debris(6)
# 모든 임계치는 "실 결함이 사라지지 않도록" 보수적.

PP_SE_SWELLING_FAR_DISTANCE_PX = 100
PP_SE_SWELLING_MIN_COMPONENT_PX = 30
PP_SE_SWELLING_SOURCE_CLASSES = (5, 7)   # 5=Swelling, 7=Super-Elevation (v3 인덱스)
PP_SE_SWELLING_TARGET_CLASS = 6          # 6=Debris (v3 인덱스)
