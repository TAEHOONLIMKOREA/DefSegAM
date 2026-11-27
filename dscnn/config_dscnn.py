"""DSCNN 재구현 — 사람 GT 만으로 학습해 DefSeg-AM (v4 – v7) 과 같은 검증 레이어에서 견준다.

버전이 아니다 (`genaug/` 처럼 따로 둔 실험 폴더). 클래스 · 데이터셋 · fold · 채점은 **v6 의 것을 임포트해** 그대로 쓴다
(`pbf-2026.11` 10-class, `ORNL_Data/DSCNN_Dataset_pbf-2026.11`, `inlayer` 22 레이어, v6 의 confusion matrix 채점).
달라지는 것은 모델과 그 모델의 학습 방식뿐이다.

출처: Scime, Siddel, Baird, Paquit, *Additive Manufacturing* 36 (2020) 101453 (`paper/DSCNN/` 의 PDF).
아래 값 가운데 "논문" 이라고 적은 것은 그 논문의 표 3 – 6 과 3.3 · 3.7 절에서 옮긴 것이고,
"정함" 이라고 적은 것은 논문이 우리 상황 (여러 장비를 한 모델로, 2022 판 데이터의 해상도) 을 다루지 않아 이 구현에서 고른 값이다.
"""
from __future__ import annotations

from pathlib import Path

from ..v6 import config_v6 as v6cfg

# ============================================================================
# 클래스 · 데이터 — v6 그대로
# ============================================================================
N_CLASSES = v6cfg.N_CLASSES_V6            # 10
IGNORE_INDEX = v6cfg.IGNORE_INDEX         # -1
N_CAM = 1                                 # 카메라 한 대 → 입력 채널 2 (visible/0 용융 직후, visible/1 도포 직후). 논문의 2·n_cam
IN_CHANNELS = 2 * N_CAM

# ============================================================================
# 입력 구성 (논문 3.3 절)
# ============================================================================
GLOBAL_SIZE = 32          # 논문: 레이어 전체를 32² 로 줄인다 (stack A)
GLOBAL_PAD = 2            # 논문: 대칭으로 2 화소 덧대 36²
REGIONAL_SIZE = 128       # 논문: 타일을 128² 로 줄인다 (stack C)
REGIONAL_PAD = 2          # 논문: 2 화소 덧대 132²
LOCAL_PAD = 5             # 논문: 원본 해상도 타일에 둘레 5 화소 (stack B, T+10)
MIN_LABELED_FRAC = 0.10   # 논문: 주석이 10% 에 못 미치는 타일은 학습에서 뺀다
DYNAMIC_RANGE = 255.0     # 8-bit

# 타일 한 변 T (원본 해상도의 화소). 논문은 장비마다 값을 준다: EOS M290 128 (보정 후 875²), ConceptLaser M2 225 (2325²),
# ExOne Innovent 450 (1922×4731), ExOne M-Flex 400 (1581×2346). 2022 판 데이터는 같은 장비라도 사진 크기가 달라서
# **베드에서 차지하는 비율이 같도록** 긴 변의 비로 옮겼다 (정함). AddUp FormUp 350 은 논문에 없어 M2 와 M290 사이의 비율로 골랐다 (정함).
TILE_SIZE: dict[str, int] = {
    "v2021_LPBF": 225,          # ConceptLaser M2, 2325² — 논문 값 그대로
    "v2022_17-4PH": 112,        # EOS M290, 766²  ← 128 × 766 / 875
    "v2022_Inc718_2": 112,      # EOS M290, 764²
    "v2022_GammaPrint": 292,    # EOS M290, 1997² ← 128 × 1997 / 875
    "v2022_Inc718_1": 292,      # EOS M290, 1997²
    "v2022_Maraging": 225,      # AddUp FormUp 350, 1834² — 논문에 없는 장비 (베드의 12%)
    "v2021_BJ": 450,            # ExOne Innovent, 약 1930×4760 — 논문 값 그대로
    "v2022_BJ_H13": 794,        # ExOne M-Flex, 2910×4656 ← 400 × 4656 / 2346
}
# ORNL Co-Registered 빌드 (ConceptLaser M2, 1842²) — 참고 비교용. 225 × 1842 / 2325
TILE_SIZE_ORNL = 178


def tile_size_for(group: str) -> int:
    if group not in TILE_SIZE:
        raise KeyError(f"FATAL: 소스 {group!r} 의 타일 크기가 config_dscnn.TILE_SIZE 에 없다")
    return TILE_SIZE[group]


# ============================================================================
# 학습 (논문 3.7 절)
# ============================================================================
LR = 1e-4                 # 논문: Adam step size 0.0001
ADAM_BETAS = (0.9, 0.999) # 논문
ADAM_EPS = 1e-4           # 논문
EMA_DECAY = 0.9999        # 논문: 저장하는 가중치는 EMA (decay 0.9999)
INIT_CONV_STD = 0.05      # 논문: CONV 가중치 N(0, 0.05²)
INIT_FC_STD = 0.004       # 논문: FC 가중치 N(0, 0.004²)
INIT_BIAS = 0.1           # 논문: bias 0.1
BN_MOMENTUM = 0.01        # TensorFlow 의 기본 decay 0.99 에 해당 (논문은 "TensorFlow 구현 그대로" 라고만 적는다)

# 증강 (논문): 잡음 (없음 · 분산 0.01% · 0.1%) × 밝기 이동 (없음 · +10% · −10%) = 9 가지. 타일마다 9 가지를 모두 쓴다.
#   잡음은 stack A · C 에만 넣고 B 에는 넣지 않는다. 회전 · 반전은 쓰지 않는다 (논문이 명시).
#   "분산이 동적 범위의 0.01% · 0.1%" 는 밝기를 0–1 로 놓았을 때의 분산 1e-4 · 1e-3 으로 읽었다 → σ = 2.55 · 8.06 단계 (정함)
NOISE_VARS = (0.0, 1e-4, 1e-3)
SHIFT_FRACS = (0.0, +0.10, -0.10)

# 한 배치에 여러 소스의 타일을 섞는다 (정함 — 논문은 장비마다 모델을 따로 둔다). 소스를 섞지 않으면 BatchNorm 이 학습 때는 한 소스의 통계를,
# 추론 때는 여러 소스가 섞인 이동 평균을 써서 어긋난다 (2026-11-23 에 시험 실행으로 확인 — BJ21 의 화소 정확도가 0.66 ↔ 0.84 로 갈렸다).
# 타일 수는 논문의 범위 (15 – 50) 를 따르되 화소 수가 BATCH_PIXEL_BUDGET 을 넘기 전에 끊는다 (GPU 메모리). 평균 40 장쯤 된다.
BATCH_MAX_TILES = 50
BATCH_PIXEL_BUDGET = 15 * 450 * 450       # ExOne Innovent (T 450) 에서 15 장

# 길이: 논문은 "epoch 100 쯤" 이라고 적는다. 여기서 한 epoch = 학습 타일 × 증강 9 가지를 한 번씩 (논문: 타일마다 9 번 복제).
# inlayer fold 에서 1 epoch 은 약 1,500 step 이다 (학습 스크립트가 시작할 때 정확한 수를 적는다).
EPOCHS = 100
EVAL_EVERY = 2_000        # step
RANDOM_SEED = 42

# 추론: 한 번에 넣는 타일의 화소 수 (메모리)
INFER_PIXEL_BUDGET = 24 * 450 * 450

# ============================================================================
# 산출물
# ============================================================================
OUTPUT_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURE_DIR = OUTPUT_DIR / "figures"


def default_run_name(seed: int = RANDOM_SEED, tag: str = "") -> str:
    """`dscnn_native_10cls_gt[-<tag>]-s<seed>`."""
    return f"dscnn_native_{N_CLASSES}cls_gt{('-' + tag) if tag else ''}-s{seed}"


def run_tag(fold: str | None) -> str:
    return "full" if fold is None else f"cv/{fold}"


def run_ckpt_dir(run_name: str, fold: str | None = None) -> Path:
    return CHECKPOINT_DIR / run_name / run_tag(fold)


def run_fig_dir(run_name: str, fold: str | None = None) -> Path:
    return FIGURE_DIR / run_name / run_tag(fold)
