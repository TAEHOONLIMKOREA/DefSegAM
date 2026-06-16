"""DefSeg-AM v2 — 8-class + DSCNN-style augmentation + cross-validation.

v1 의 [DefSeg_AM/config.py](../config.py) 를 base 로 import 한 뒤, v2 전용 상수를
이 파일에서 정의/override. 직접 v1 모듈을 수정하지 않음.

자세한 설계: [DefSeg_AM/PLAN_v2.md](../PLAN_v2.md), DSCNN 원본: [DSCNN_Summary.md](../paper/DSCNN/DSCNN_Summary.md).
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
# v2 — 8 class (PLAN_v2 §2)
# ============================================================================
# 사용자 지정 순서:
#   0=Powder, 1=Printed, 2=Recoater Disturbance,
#   3=Swelling, 4=Spatter, 5=Super-Elevation, 6=Over Melting, 7=Debris
ORNL_CLASS_NAMES_V2 = [
    "Powder",               # 0
    "Printed",              # 1
    "Recoater Disturbance", # 2  (v1 old 2 Hopping + 3 Streaking)
    "Swelling",             # 3  (v1 old 5)
    "Spatter",              # 4  (v1 old 8)
    "Super-Elevation",      # 5  (v1 old 7)
    "Over Melting",         # 6  (v1 old 10)
    "Debris",               # 7  (v1 old 6)
]
N_CLASSES_V2 = len(ORNL_CLASS_NAMES_V2)  # 8

# v1 ORNL 12-class ID → v2 8-class ID (-1 = IGNORE)
ORNL_12_TO_NEW_8: dict[int, int] = {
    0:  0,    # Powder              → Powder
    1:  1,    # Printed             → Printed
    2:  2,    # Recoater Hopping    → Recoater Disturbance  (통합)
    3:  2,    # Recoater Streaking  → Recoater Disturbance  (통합)
    4: -1,    # Incomplete Spreading → IGNORE              (제거)
    5:  3,    # Swelling            → Swelling
    6:  7,    # Debris              → Debris
    7:  5,    # Super-Elevation     → Super-Elevation
    8:  4,    # Spatter             → Spatter
    9: -1,    # Misprint            → IGNORE              (제거)
    10: 6,    # Over Melting        → Over Melting
    11:-1,    # Under Melting       → IGNORE              (제거)
}

# Powder/Printed 가 아닌 "결함" 인덱스 (sampler oversample 의 defect_ratio 계산용)
DEFECT_CLASS_INDICES_V2 = list(range(2, N_CLASSES_V2))  # 2..7


# ============================================================================
# v2 — DSCNN_Dataset native (재료) → v1 12-class 매핑 + BJ 2 source 추가
# ============================================================================
# v1 의 LPBF 6 source 그대로 가져온 뒤 BJ 2 source 추가.
# data_dscnn_v2.remap_label_v2 가 이걸 사용한 후 다시 ORNL_12_TO_NEW_8 로 변환.

# v1 의 6 LPBF mapping 그대로 import
MATERIAL_TO_ORNL_V2: dict[str, dict[int, int]] = dict(_v1.MATERIAL_TO_ORNL)

# v2 신규 — Binder Jet 2 source 매핑
# v2021_BJ (ExOne Innovent / Innovent+, readme v2021-03 의 BJ section 6 클래스)
MATERIAL_TO_ORNL_V2["v2021_BJ"] = {
    0: 0,   # Un-Fused Powder    → ORNL Powder
    1: 1,   # Printed Material    → ORNL Printed
    2: 3,   # Recoater Streaking  → ORNL Recoater Streaking (→ v2 Recoater Disturbance via 8-map)
    3: -1,  # Powder Short Feed   → IGNORE (= Incomplete Spreading, v2 에서 제거)
    4: -1,  # (no examples)
    5: 6,   # Debris              → ORNL Debris
}
# v2022_BJ_H13 (ExOne M-Flex / H13 Steel — readme v2022-10.1 §5.1 의 8 클래스)
MATERIAL_TO_ORNL_V2["v2022_BJ_H13"] = {
    0: 0,   # Powder
    1: 1,   # Printed
    2: 3,   # Roller Streaking    → ORNL Recoater Streaking
    3: 6,   # Debris              → ORNL Debris
    4: -1,  # Short Feed          → IGNORE
    5: -1,  # Cornrows            → IGNORE
    6: -1,  # Exposed Part (super-elevation 비슷하나 불확실)
    7: -1,  # Misprint            → IGNORE (v2 제거)
}


# ============================================================================
# v2 — DSCNN_Dataset 8 source 정의 (LPBF 6 + BJ 2, EBPBF 제외)
# ============================================================================
DSCNN_TRAIN_SOURCES_V2 = [
    # v1 의 6 LPBF source 그대로 (mapping_key 도 그대로)
    *_v1.DSCNN_TRAIN_SOURCES,
    # 신규 BJ 2 source
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

# 8-fold leave-one-source-out 의 fold id (0..7) 별 source 이름
DSCNN_CV_SOURCE_NAMES = [s["name"] for s in DSCNN_TRAIN_SOURCES_V2]
N_FOLDS = len(DSCNN_CV_SOURCE_NAMES)  # 8


# ============================================================================
# v2 — Augmentation 상수 (PLAN_v2 §4 / DSCNN_Summary §6)
# ============================================================================
# Gaussian noise — % of 8-bit dynamic range (=255). DSCNN 원본 0.01% / 0.1%
DSCNN_NOISE_SIGMA_PCT_CHOICES = (0.0, 0.01, 0.1)

# Mean intensity shift — ±10% of DR. DSCNN 원본
DSCNN_INTENSITY_SHIFT_PCT_CHOICES = (0.0, +0.10, -0.10)

# D4 group rotation/flip — v2 신규 (Recoater 통합 덕에 가능)
ENABLE_D4_AUGMENTATION = True

# Cyclic shift (np.roll) — 사용자 신규 제안
ENABLE_CYCLIC_SHIFT = True
CYCLIC_SHIFT_MAX_FRAC = 0.25  # shift 범위 = ±(img_size * frac)

# Brightness multiplicative jitter (v1 그대로 유지)
ENABLE_BRIGHTNESS_JITTER = True
BRIGHTNESS_JITTER_RANGE = (0.85, 1.15)


# ============================================================================
# v2 — Stage 1 (KD pretrain) hyper-params
# ============================================================================
S1_EPOCHS = 30
S1_BATCH_SIZE = 2
S1_LR = 1e-4
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
HARD_BOOTSTRAP_LAMBDA = 0.8  # 1.0 = standard CE, < 1.0 = model self-trust


# ============================================================================
# v2 — Stage 2 (Cross-validation finetune) hyper-params
# ============================================================================
S2_EPOCHS = 50
S2_BATCH_SIZE = 2
S2_LR = 1e-4
S2_WEIGHT_DECAY = 1e-4
S2_WARMUP_STEPS = 50
S2_GRAD_CLIP_NORM = 1.0
S2_CLASS_WEIGHT_CLIP = 10.0


# ============================================================================
# v2 — Output paths (v1 와 디렉터리 분리)
# ============================================================================
OUTPUT_DIR = Path(__file__).resolve().parents[1]  # = DefSeg_AM/
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"        # 기존 디렉터리. run_name 으로 분리.
FIGURE_DIR = OUTPUT_DIR / "figures"
CACHE_DIR = OUTPUT_DIR / "cache"

# v2 의 label cache 만 별도 디렉터리 (image 는 v1 와 공유)
V2_CACHE_DIR_NAME = "resized_sz{img_size}_v2"   # label 만 저장
V1_CACHE_DIR_NAME = "resized_sz{img_size}"       # image 는 v1 그대로 재사용


def v1_cache_dir(img_size: int = IMG_SIZE) -> Path:
    """v1 의 image cache 디렉터리 (visible_0.npy, visible_1.npy 가 있는 곳)."""
    return CACHE_DIR / V1_CACHE_DIR_NAME.format(img_size=img_size)


def v2_cache_dir(img_size: int = IMG_SIZE) -> Path:
    """v2 의 label cache 디렉터리 (label_v2.npy 가 있는 곳)."""
    return CACHE_DIR / V2_CACHE_DIR_NAME.format(img_size=img_size)


# Cross-validation 결과 요약 파일
CV_SUMMARY_FILE = "cv_summary.json"


# ============================================================================
# v2 — Inference 휴리스틱 후처리 (PP_)
# ============================================================================
# 두 규칙. 모든 임계치는 "실 결함이 사라지지 않도록" 보수적으로 설정.
#
# 1) PP_STATIC_OUTSIDE_POWDER — 카메라 setup 의 정적 영역 (빌드플레이트 / fixture /
#    챔버 frame) 이 결함으로 오분류되는 케이스 제거.
#    조건 (AND, 픽셀 단위): |visible/1 − visible/0| ≤ T_static
#                       AND  픽셀이 powder ROI 의 바깥
#    → 해당 픽셀이 결함 (class 2..7) 으로 분류돼 있으면 IGNORE_INDEX 로.
#
# 2) PP_SE_SWELLING_FAR_FROM_PART — Super-Elevation / Swelling 의 connected
#    component 가 Printed (part) 영역으로부터 충분히 멀리 있으면 Debris (7) 로.
#    부품에서 떨어진 곳의 raised feature 는 사실상 떨어진 입자 (debris).

# ---- 공통 ----
# uint8 정적 임계 : |i1 − i0| ≤ 이 값이면 "정적" 픽셀.
# 보수적 = 작게 → 강한 일치만 정적으로 인정 → 환원 픽셀 적음.
PP_STATIC_DIFF_THRESHOLD = 5

# ---- 규칙 1 : powder ROI 정의 ----
# Powder ROI = pred ∈ {Powder(0), Printed(1)} 의 largest connected component
# + closing(구멍 메우기) + dilation. 보수적 = ROI 를 크게 잡아 부품 가장자리의
# 진짜 결함이 ROI 밖으로 새어 IGNORE 되지 않도록 한다.
PP_POWDER_ROI_USE_PRINTED = True    # True 면 Powder ∪ Printed, False 면 Powder 만
PP_POWDER_ROI_CLOSING_PX = 5        # 작은 구멍 메우기
PP_POWDER_ROI_DILATE_PX = 40        # 보수적 dilation (px)

# ---- 규칙 2 : SE/Swelling → Debris ----
# Printed mask 에서 component 의 nearest pixel 까지 거리 (px). 이 값을 넘어야
# Debris 로 재분류. 보수적 = 크게 → 부품에서 확실히 멀리 있을 때만 변경.
PP_SE_SWELLING_FAR_DISTANCE_PX = 100
# 너무 작은 component 는 노이즈로 보고 변경 대상에서 제외 (보수적 안전장치)
PP_SE_SWELLING_MIN_COMPONENT_PX = 30
# 재분류 대상 / 결과 클래스
PP_SE_SWELLING_SOURCE_CLASSES = (3, 5)   # 3=Swelling, 5=Super-Elevation
PP_SE_SWELLING_TARGET_CLASS = 7          # 7=Debris
