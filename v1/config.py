"""DefSeg-AM v1 설정 — 공유 상수(common) + v1 전용 하이퍼파라미터.

DINOv2 + DPT decoder, 2-stage segmentation (12-class).
Stage 1 = ORNL HDF5 segmentation_results (DSCNN pred) 으로 KD pretrain.
Stage 2 = DSCNN_Dataset annotations (human GT) 로 finetune.

경로/ORNL 라벨공간/DSCNN 매핑/backbone 등 공유 상수는 [../common/config.py](../common/config.py).
v2 (8-class + cross-val + DSCNN aug) 는 [../v2/](../v2/) 참고.
"""
from __future__ import annotations

# 공유 상수 전부 re-export (PROJECT_ROOT, ORNL_*, DSCNN_*, DINO_*, IMG_SIZE,
# IMAGENET_*, N_CLASSES, IGNORE_INDEX, OUTPUT_DIR/CHECKPOINT_DIR/... 등)
from ..common.config import *  # noqa: F401,F403

# === Stage 2 val split (v1 전용; v2 는 8-fold CV 사용) ===
DSCNN_VAL_SOURCE_NAMES = ["v2022_Maraging"]

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
