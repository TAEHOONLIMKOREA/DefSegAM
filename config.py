"""DefSeg-AM: DINOv2 + DPT decoder, 2-stage segmentation.

Stage 1 = ORNL HDF5 segmentation_results (DSCNN pred) 으로 KD pretrain.
Stage 2 = DSCNN_Dataset annotations (human GT) 로 finetune.

자세한 설계는 PLAN.md 참고.
"""
from __future__ import annotations

from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[1]

# === ORNL Co-Registered HDF5 (Stage 1 KD pretrain) ===
ORNL_HDF5_DIR = (
    PROJECT_ROOT / "ORNL_Data"
    / "Co-Registered In-Situ and Ex-Situ Dataset"
    / "[baseline] (Peregrine v2023-11)"
)
ORNL_BUILD_FILES = {
    "B1": "2021-07-13 TCR Phase 1 Build 1.hdf5",
    "B2": "2021-04-16 TCR Phase 1 Build 2.hdf5",
    "B3": "2021-04-28 TCR Phase 1 Build 3.hdf5",
    "B4": "2021-08-03 TCR Phase 1 Build 4.hdf5",
    "B5": "2021-08-23 TCR Phase 1 Build 5.hdf5",
}
# Stage 1 split: Build 1 = val, 나머지 train (seung_dscnn 와 동일하게 Build 1 을 정성 비교 build 로 유지)
ORNL_TRAIN_BUILDS = ["B2", "B3", "B4", "B5"]
ORNL_VAL_BUILDS = ["B1"]

# === DSCNN_Dataset (Stage 2 GT finetune) ===
DSCNN_ROOT = PROJECT_ROOT / "ORNL_Data" / "DSCNN_Dataset"

# === ORNL 12 클래스 (HDF5 slices/segmentation_results.attrs['class_names']) ===
ORNL_CLASS_NAMES = [
    "Powder",             # 0
    "Printed",            # 1
    "Recoater Hopping",   # 2
    "Recoater Streaking", # 3
    "Incomplete Spreading", # 4
    "Swelling",           # 5
    "Debris",             # 6
    "Super-Elevation",    # 7
    "Spatter",            # 8
    "Misprint",           # 9
    "Over Melting",       # 10
    "Under Melting",      # 11
]
N_CLASSES = len(ORNL_CLASS_NAMES)
IGNORE_INDEX = -1

# Powder/Printed 만 있는 layer 를 가려내기 위한 "결함" 인덱스 (PLAN §3.1)
DEFECT_CLASS_INDICES = list(range(2, N_CLASSES))  # 2..11

# === DSCNN_Dataset 재료별 native ID → ORNL 12-class 재매핑 ===
# seung_dscnn/config.py 의 MATERIAL_TO_ORNL 와 동일 (검증된 매핑, PLAN §3.2 참조).
MATERIAL_TO_ORNL: dict[str, dict[int, int]] = {
    "v2021_LPBF": {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
        8: -1,  # Soot
    },
    "17-4_PH_Stainless_Steel": {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4, 5: 5, 6: 6, 7: 7,
        8: -1,  # Soot
        9: 10,  # Excessive Melting → Over Melting
        10: -1, # Crashing
        11: 9,  # Misprint
    },
    "GammaPrint-700": {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4,
        5: 6,   # Debris
        6: 5,   # Edge Swelling → Swelling
        7: 7,
        8: 8,   # Spatter on Powder
        9: -1, 10: -1, 11: -1,
        12: 10, # Excessive Melting → Over Melting
        13: 9,  # Misprint
        14: 11, # Localized Dark Regions → Under Melting
    },
    "Inconel_718_1": {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4,
        5: 6, 6: 5, 7: 7, 8: 8,
        9: -1, 10: -1, 11: -1,
        12: 10, 13: 9, 14: 11,
    },
    "Inconel_718_2": {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4,
        5: 5, 6: 6, 7: 7, 8: -1,
        9: 10, 10: -1, 11: 9,
    },
    "Maraging_Steel": {
        0: 0, 1: 1, 2: 2, 3: 3, 4: 4,
        5: 5, 6: 6, 7: 7,
        8: -1,  # Soot
        9: 9,   # Misprint
    },
}

# === Stage 2 학습 소스 (seung_dscnn 와 동일; LPBF 만, EBPBF/BJ 제외) ===
DSCNN_TRAIN_SOURCES = [
    {
        "name": "v2021_LPBF",
        "root": DSCNN_ROOT / "Peregrine Dataset v2021-03" / "Laser Powder Bed Fusion",
        "mapping_key": "v2021_LPBF",
    },
    {
        "name": "v2022_17-4PH",
        "root": DSCNN_ROOT / "Peregrine Dataset v2022-10.1/Laser_Powder_Bed_Fusion/EOS_M290/17-4_PH_Stainless_Steel/training",
        "mapping_key": "17-4_PH_Stainless_Steel",
    },
    {
        "name": "v2022_GammaPrint",
        "root": DSCNN_ROOT / "Peregrine Dataset v2022-10.1/Laser_Powder_Bed_Fusion/EOS_M290/GammaPrint-700/training",
        "mapping_key": "GammaPrint-700",
    },
    {
        "name": "v2022_Inc718_1",
        "root": DSCNN_ROOT / "Peregrine Dataset v2022-10.1/Laser_Powder_Bed_Fusion/EOS_M290/Inconel_718_1/training",
        "mapping_key": "Inconel_718_1",
    },
    {
        "name": "v2022_Inc718_2",
        "root": DSCNN_ROOT / "Peregrine Dataset v2022-10.1/Laser_Powder_Bed_Fusion/EOS_M290/Inconel_718_2/training",
        "mapping_key": "Inconel_718_2",
    },
    {
        "name": "v2022_Maraging",
        "root": DSCNN_ROOT / "Peregrine Dataset v2022-10.1/Laser_Powder_Bed_Fusion/AddUp_FormUp_350/Maraging_Steel/training",
        "mapping_key": "Maraging_Steel",
    },
]
DSCNN_VAL_SOURCE_NAMES = ["v2022_Maraging"]

# === Backbone / Decoder ===
DINO_BACKBONE = "dinov2_vits14"  # embed_dim=384, patch=14, 12 blocks
INTERMEDIATE_LAYERS = (2, 5, 8, 11)  # 0-based, 즉 3/6/9/12 번째 block (DPT 4-stage 입력)
DECODER_CHANNELS = 256

# === 입력 ===
IMG_SIZE = 1036  # = 74 patches × 14 (DINOv2 patch_size)
IMAGENET_MEAN = [0.485, 0.456, 0.406]
IMAGENET_STD = [0.229, 0.224, 0.225]

# === Layer 필터링 (Stage 1, PLAN §3.1) ===
ORNL_LAYER_LO_FRAC = 0.05  # 빌드 안쪽 5%
ORNL_LAYER_HI_FRAC = 0.95  # ~ 95%
# "Powder/Printed only" layer 제외 — defect mask 합이 이 이하면 drop
DEFECT_PIXEL_MIN = 1  # 결함 픽셀이 단 1개라도 있으면 keep (정확히 0인 것만 drop)

# === Stage 1 (KD pretrain) ===
# 안정성 fix (이전 run NaN 발생 → 5가지 적용):
#   - lr 5e-4 → 1e-4 (작은 decoder + Focal + α=clip 조합 안정)
#   - warmup 200 step (학습 초반 grad explosion 방지)
#   - grad_clip max_norm=1.0 (FP32 raw grad 크기 제한)
#   - class_weight clip 50 → 10 (rare class 의 loss 폭주 방지)
#   - AMP off (FP16 overflow 위험 제거)
S1_EPOCHS = 30
S1_BATCH_SIZE = 2
S1_LR = 1e-4
S1_WEIGHT_DECAY = 1e-4
S1_FOCAL_GAMMA = 2.0
S1_OVERSAMPLE_POWER = 0.5
S1_OVERSAMPLE_EPS = 1e-3
S1_WARMUP_STEPS = 200            # linear warmup 0 → lr 동안 step 수
S1_GRAD_CLIP_NORM = 1.0          # gradient L2-norm clip
S1_CLASS_WEIGHT_CLIP = 10.0      # sqrt-inv weight 의 max 값

# === Stage 2 (GT finetune) ===
S2_EPOCHS = 50
S2_BATCH_SIZE = 2
S2_LR = 1e-4
S2_WEIGHT_DECAY = 1e-4
S2_WARMUP_STEPS = 50             # 데이터 작아서 warmup 도 짧게
S2_GRAD_CLIP_NORM = 1.0
S2_CLASS_WEIGHT_CLIP = 10.0

# === 추론/시각화 ===
N_INFER_LAYERS = 12  # ORNL 비교 추론 시 균등 추출할 layer 개수
NUM_WORKERS = 4

# === 출력 경로 ===
OUTPUT_DIR = Path(__file__).resolve().parent
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURE_DIR = OUTPUT_DIR / "figures"
CACHE_DIR = OUTPUT_DIR / "cache"
