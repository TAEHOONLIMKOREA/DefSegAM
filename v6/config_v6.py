"""DefSeg-AM v6 — DIVA 레이블 체계 `pbf-2026.11` (10-class) · 사람 GT 전용 학습.

설계는 [PLAN_v6.md](PLAN_v6.md) 참조.

앞쪽은 **Peregrine 주석 (`DSCNN_Dataset`) 을 새 체계의 레이블 데이터셋으로 변환하는 데 쓰는 값**이고
(클래스 · 경로 · 소스 목록 · native 이름 표 · 매핑), 뒤쪽은 **학습 · 평가 값**이다 (PLAN_v6 §5, 부록 A).

학습 방식은 v5 와 같다 (사람 GT 전용, decoder 무작위 초기값, step 기준 단일 단계, EMA 0.999).
v5 와 달라지는 것은 클래스 체계 (10-class, 변환해 둔 데이터셋을 읽는다) 와 Copy-Paste 의 위치 규칙이다.

클래스의 단일 출처는 DIVA 의 `backend/modules/Labeling/schema_pbf_2026_11.json` 이고,
`v6/schema_pbf_2026_11.json` 은 그 사본이다 (DIVA commit fd1db13b, 2026-11-09). 사본이 안내서
(`docs/DEFSEGAM_CLASS_UPDATE.md` 부록) 의 값과 다르면 import 할 때 멈춘다.
"""
from __future__ import annotations

import json
from pathlib import Path

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
# v6 — 클래스 (pbf-2026.11, PLAN_v6 §2.1)
# ============================================================================
SCHEMA_FILE = Path(__file__).resolve().parent / "schema_pbf_2026_11.json"
with open(SCHEMA_FILE, encoding="utf-8") as _fp:
    SCHEMA = json.load(_fp)

SCHEMA_VERSION = SCHEMA["version"]                              # "pbf-2026.11"
CLASS_NAMES_V6 = [c["name"] for c in SCHEMA["classes"]]         # index = class id
CLASS_KEYS_V6 = [c["key"] for c in SCHEMA["classes"]]
CLASS_ABBRS_V6 = ["Pow", "Prt"] + [c["abbr"] for c in SCHEMA["classes"][2:]]   # 로그 · 표에 쓰는 짧은 이름
N_CLASSES_V6 = len(CLASS_NAMES_V6)                              # 10
NORMAL_IDS_V6 = [c["id"] for c in SCHEMA["classes"] if c["group"] == "normal"]   # [0, 1]
DEFECT_IDS_V6 = [c["id"] for c in SCHEMA["classes"] if c["group"] == "defect"]   # [2 .. 9]
DIVA_COLORS = [tuple(c["color"]) for c in SCHEMA["classes"]]    # DIVA 화면의 표시 색 (RGB). 초안이라 바뀔 수 있다
UNLABELED_COLOR = tuple(SCHEMA["unlabeled"]["color"])

# 사본이 안내서 부록의 값과 같은지 대조한다. 다르면 사본이나 DIVA 쪽이 바뀐 것이므로 매핑부터 다시 본다.
_EXPECTED_KEYS = (
    "powder", "printed", "incomplete_spreading", "powder_slumping", "recoater_streaking",
    "recoater_hopping", "super_elevation", "spatter", "over_melting", "debris",
)
if SCHEMA_VERSION != "pbf-2026.11" or tuple(CLASS_KEYS_V6) != _EXPECTED_KEYS:
    raise RuntimeError(f"FATAL: {SCHEMA_FILE.name} 이 pbf-2026.11 의 클래스 10개와 다르다: {SCHEMA_VERSION} {CLASS_KEYS_V6}")
if [c["id"] for c in SCHEMA["classes"]] != list(range(10)) or SCHEMA["unlabeled"]["export_value"] != IGNORE_INDEX:
    raise RuntimeError(f"FATAL: {SCHEMA_FILE.name} 의 class id 또는 미레이블 값이 0–9 / -1 이 아니다")


# ============================================================================
# v6 — Peregrine 주석 8 소스 (LPBF 6 + BJ 2, EBPBF 제외; v3–v5 와 같은 목록 · 같은 순서)
#   장비 · 재료는 각 판의 readme.pdf, 리코터 방향은 v2022 의 info.txt (v2021 에는 메타가 없다)
# ============================================================================
_V21 = DSCNN_ROOT / "Peregrine Dataset v2021-03"
_V22 = DSCNN_ROOT / "Peregrine Dataset v2022-10.1"

LEGACY_SOURCES = [
    {"name": "v2021_LPBF", "short": "L21", "root": _V21 / "Laser Powder Bed Fusion",
     "release": "Peregrine v2021-03", "process": "LPBF", "machine": "ConceptLaser M2", "mc_id": "ConceptLaser_M2", "material": "Stainless Steel 316L"},
    {"name": "v2022_17-4PH", "short": "17-4", "root": _V22 / "Laser_Powder_Bed_Fusion/EOS_M290/17-4_PH_Stainless_Steel/training",
     "release": "Peregrine v2022-10.1", "process": "LPBF", "machine": "EOS M290", "mc_id": "EOS_M290", "material": "17-4 PH Stainless Steel"},
    {"name": "v2022_GammaPrint", "short": "GP", "root": _V22 / "Laser_Powder_Bed_Fusion/EOS_M290/GammaPrint-700/training",
     "release": "Peregrine v2022-10.1", "process": "LPBF", "machine": "EOS M290", "mc_id": "EOS_M290", "material": "GammaPrint-700"},
    {"name": "v2022_Inc718_1", "short": "I1", "root": _V22 / "Laser_Powder_Bed_Fusion/EOS_M290/Inconel_718_1/training",
     "release": "Peregrine v2022-10.1", "process": "LPBF", "machine": "EOS M290", "mc_id": "EOS_M290", "material": "Inconel 718"},
    {"name": "v2022_Inc718_2", "short": "I2", "root": _V22 / "Laser_Powder_Bed_Fusion/EOS_M290/Inconel_718_2/training",
     "release": "Peregrine v2022-10.1", "process": "LPBF", "machine": "EOS M290", "mc_id": "EOS_M290", "material": "Inconel 718"},
    {"name": "v2022_Maraging", "short": "MS", "root": _V22 / "Laser_Powder_Bed_Fusion/AddUp_FormUp_350/Maraging_Steel/training",
     "release": "Peregrine v2022-10.1", "process": "LPBF", "machine": "AddUp FormUp 350", "mc_id": "AddUp_FormUp_350", "material": "Maraging Steel"},
    {"name": "v2021_BJ", "short": "BJ21", "root": _V21 / "Binder Jet",
     "release": "Peregrine v2021-03", "process": "BJ", "machine": "ExOne Innovent / Innovent+", "mc_id": "ExOne_Innovent", "material": "SiC / B4C / H13 Steel"},
    {"name": "v2022_BJ_H13", "short": "H13", "root": _V22 / "Binder_Jet/ExOne_M-Flex/H13_Steel/training",
     "release": "Peregrine v2022-10.1", "process": "BJ", "machine": "ExOne M-Flex", "mc_id": "ExOne_M-Flex", "material": "H13 Steel"},
]

# 모든 소스의 주석이 리코팅 축 = 이미지 가로다 (PLAN_v6 §4). DIVA 내보내기 (기본 vertical) 와 다르므로 index.csv 에 적는다.
LEGACY_RECOAT_AXIS = "horizontal"


# ============================================================================
# v6 — native id → 이름 (소스별). v2022 는 info.txt, v2021 은 readme.pdf 의 표를 그대로 옮겼다.
#   변환할 때 v2022 소스는 info.txt 와 다시 대조한다 (다르면 FATAL).
# ============================================================================
_NATIVE_V21_LPBF = {
    0: "Un-Fused Powder", 1: "Printed Material", 2: "Recoater Hopping", 3: "Recoater Streaking",
    4: "Incomplete Spreading", 5: "Swelling", 6: "Debris", 7: "Super-Elevation", 8: "Soot",
}
_NATIVE_V21_BJ = {
    0: "Un-Fused Powder", 1: "Printed Material", 2: "Recoater Streaking", 3: "Powder Short Feed",
    4: "no examples", 5: "Debris",
}
_NATIVE_EOS_A = {   # 17-4 PH, Inconel 718 (2)
    0: "Powder", 1: "Printed", 2: "Recoater Hopping", 3: "Recoater Streaking", 4: "Incomplete Spreading",
    5: "Edge Swelling", 6: "Debris", 7: "Super-Elevation", 8: "Soot", 9: "Excessive Melting",
    10: "Crashing", 11: "Misprint",
}
_NATIVE_EOS_B = {   # GammaPrint-700, Inconel 718 (1) — optical tomography 가 있는 쪽
    0: "Powder", 1: "Printed", 2: "Recoater Hopping", 3: "Recoater Streaking", 4: "Incomplete Spreading",
    5: "Debris", 6: "Edge Swelling", 7: "Super-Elevation", 8: "Spatter on Powder", 9: "Localized Bright Spot",
    10: "Mounding Powder", 11: "Stripe Boundary", 12: "Excessive Melting", 13: "Misprint",
    14: "Localized Dark Regions",
}
_NATIVE_MS = {
    0: "Powder", 1: "Printed", 2: "Recoater Hopping", 3: "Recoater Streaking", 4: "Incomplete Spreading",
    5: "Swelling", 6: "Debris", 7: "Super-Elevation", 8: "Soot", 9: "Misprint",
}
_NATIVE_H13 = {
    0: "Powder", 1: "Printed", 2: "Roller Streaking", 3: "Debris", 4: "Short Feed", 5: "Cornrows",
    6: "Exposed Part", 7: "Misprint",
}
LEGACY_NATIVE_NAMES: dict[str, dict[int, str]] = {
    "v2021_LPBF": _NATIVE_V21_LPBF,
    "v2022_17-4PH": _NATIVE_EOS_A,
    "v2022_GammaPrint": _NATIVE_EOS_B,
    "v2022_Inc718_1": _NATIVE_EOS_B,
    "v2022_Inc718_2": _NATIVE_EOS_A,
    "v2022_Maraging": _NATIVE_MS,
    "v2021_BJ": _NATIVE_V21_BJ,
    "v2022_BJ_H13": _NATIVE_H13,
}
NATIVE_UNLABELED = -1          # 주석 파일 안의 미레이블 값 (두 판 모두)


# ============================================================================
# v6 — native 이름 → 새 id (PLAN_v6 §2.3). 번호가 아니라 이름으로 옮긴다 (소스마다 번호가 다르다)
# ============================================================================
LEGACY_NAME_TO_V6: dict[str, int] = {
    "Powder": 0, "Un-Fused Powder": 0,
    "Printed": 1, "Printed Material": 1,
    "Recoater Streaking": 4, "Roller Streaking": 4,
    "Recoater Hopping": 5,
    "Super-Elevation": 6,
    "Spatter on Powder": 7,
    "Soot": 7,                    # D10 — 다만 비어 있지 않은 파트 마스크 안쪽은 LEGACY_SOOT_IN_PART
    "Excessive Melting": 8,       # 가이드 30–32쪽이 Over Melting 으로 읽는다
    "Debris": 9,
}
LEGACY_SOOT_NAME = "Soot"
LEGACY_SOOT_IN_PART = 8           # D10 — 파트 위의 거뭇한 자국은 Over Melting

# D3 — Swelling 을 어디로 보낼지. "se" = Super Elevation (권장안), "ignore" = 미레이블
LEGACY_SWELLING_NAMES = ("Swelling", "Edge Swelling")
LEGACY_SWELLING_CHOICES = {"se": 6, "ignore": IGNORE_INDEX}
LEGACY_SWELLING_DEFAULT = "se"

# D4 — 기존 IS 는 새 기준의 IS 와 PS 가 섞여 있다. 검토 전에는 전부 미레이블, 검토 표가 있으면 레이어 단위로 2 / 3 / -1
LEGACY_IS_NAMES = ("Incomplete Spreading", "Powder Short Feed", "Short Feed")
LEGACY_IS_CHOICES = {"is": 2, "ps": 3, "ignore": IGNORE_INDEX}
# 레이어 단위 판정표 (열: source, layer, decision = IS / PS / ignore). 있으면 변환할 때 기본으로 반영한다 (PLAN_v6 §2.4)
LEGACY_REVIEW_CSV = Path(__file__).resolve().parent / "legacy_review.csv"
# 판정표의 fill 열이 켜진 레이어는 원본이 덮이지 않은 영역 대신 그 경계선만 칠한 것이다 (17-4 · GP).
# 선이 이미지 가장자리와 함께 가두는 쪽 전체를 같은 클래스로 칠한다 (legacy_map.enclosed_side, PLAN_v6 §2.4)
LEGACY_FILL_SEAL_FRAC = 0.01      # 선의 끝이 가장자리에 못 미치는 틈을 막는 거리 — 짧은 변의 이 비율 (최소 3 px)
LEGACY_FILL_MAX_FRAC = 0.5        # 채울 영역이 이미지의 이 비율을 넘으면 가둔 쪽을 잘못 고른 것으로 보고 멈춘다

# 새 체계에 대응이 없어 미레이블로 두는 이름 → 까닭
LEGACY_NAME_IGNORED: dict[str, str] = {
    "Exposed Part": "D11 보류 — 정의는 SE 와 맞지만 Binder Jet 이고 가이드에 선례가 없다",
    "Stripe Boundary": "새 체계에 없다 (레이저 stripe 경계)",
    "Localized Bright Spot": "새 체계에 없다 (optical tomography 에서 보이는 것)",
    "Localized Dark Regions": "새 체계에 없다 (v4 · v5 에서는 Under Melting 이었다)",
    "Cornrows": "새 체계에 없다",
    "Misprint": "새 체계에 없다 (v4 부터 뺐다)",
    "Crashing": "새 체계에 없다",
    "Mounding Powder": "새 체계에 없다",
    "no examples": "v2021 Binder Jet 의 빈 번호",
}

# 모든 native 이름은 위 네 묶음 가운데 정확히 한 곳에 있어야 한다.
for _src, _names in LEGACY_NATIVE_NAMES.items():
    for _nid, _name in _names.items():
        _hits = (_name in LEGACY_NAME_TO_V6) + (_name in LEGACY_SWELLING_NAMES) + (_name in LEGACY_IS_NAMES) + (_name in LEGACY_NAME_IGNORED)
        if _hits != 1:
            raise RuntimeError(f"FATAL: {_src} 의 native 이름 {_name!r} (id {_nid}) 의 갈 곳이 {_hits}개다 — config_v6 의 매핑 표를 고친다")


# ============================================================================
# v6 — 변환한 데이터셋의 위치 (DIVA 내보내기 zip 을 푼 것과 같은 구성. ORNL_Data 는 compose 에서 read-only 로 보인다)
# ============================================================================
LEGACY_PBF_ROOT = PROJECT_ROOT / "ORNL_Data" / f"DSCNN_Dataset_{SCHEMA_VERSION}"
LEGACY_AGENCY = "ORNL"            # index.csv 의 agency — DIVA 의 기관 이름 자리
LEGACY_STATUS = "converted"       # index.csv 의 status — DIVA 의 labeled / reviewed 와 구별한다


# ############################################################################
# 여기부터 학습 · 평가 값 (PLAN_v6 §5–§8, 부록 A)
# ############################################################################

# ============================================================================
# v6 — Backbone · 방향 규약
# ============================================================================
DINO_BACKBONE = "dinov2_vitb14_reg"   # 86 M frozen + Registers 4 token (v4 · v5 와 같음)

# 학습 · 추론의 모든 이미지는 리코팅 축 = 이미지 가로다 (PLAN_v6 §4). 변환한 Peregrine 데이터셋은 이미 이 방향이다.
CANONICAL_RECOAT_AXIS = "horizontal"

# 소스 약어 (fold 이름 · 표에 쓴다)
SOURCE_SHORT = {s["name"]: s["short"] for s in LEGACY_SOURCES}

# 표본의 도메인. sampler 가 도메인 사이의 비중을 따로 정한다 (PLAN_v6 §5.4). DIVA 내보내기 ("diva") 는 P1 부터다.
DOMAIN_LEGACY = "legacy"
DOMAIN_DIVA = "diva"


# ============================================================================
# v6 — 옛 11-class (v4 · v5) → 새 id. 기준선 채점 (R2 · R3 · B0 · B1) 에만 쓴다 (PLAN_v6 §6.3)
# ============================================================================
V4_CLASS_NAMES = list(SCHEMA["model_maps"]["defseg-v4"]["class_names"])
N_CLASSES_V4 = len(V4_CLASS_NAMES)                              # 11
# None = 새 체계에 대응이 없다 (Swelling, Under Melting). DIVA 는 이 예측을 미레이블로 들여온다
V4_TO_NEW: dict[int, int | None] = {
    int(k): (None if v == SCHEMA["unlabeled"]["id"] else int(v))
    for k, v in SCHEMA["model_maps"]["defseg-v4"]["ids"].items()
}
V4_SWELLING_ID = V4_CLASS_NAMES.index("Swelling")               # 5 — "Swelling → SE" 변형에서 SE (6) 로 옮긴다
V4_SWELLING_AS = 6
if V4_TO_NEW != {0: 0, 1: 1, 2: 5, 3: 4, 4: 2, 5: None, 6: 9, 7: 6, 8: 7, 9: 8, 10: None}:
    raise RuntimeError(f"FATAL: {SCHEMA_FILE.name} 의 defseg-v4 매핑이 안내서 부록과 다르다: {V4_TO_NEW}")

# 기준선으로 채점하는 옛 모델의 run 이름 (ckpt 는 checkpoints/<run>/… 에 있다)
V5_RUN_NAME = "vitb14_reg_dpt_dual_sz1036_11cls_v5"
V4_RUN_NAME = "vitb14_reg_dpt_dual_sz1036_11cls_v4"


# ============================================================================
# v6 — 검증 fold (PLAN_v6 §3.3, §7.1)
# ============================================================================
# ("list", <파일>)        : splits/<파일> 의 목록에 있는 레이어가 검증 (고정 목록 — 매핑 결정이 바뀌어도 흔들리지 않는다)
# ("layers", n, k)        : 소스마다 레이어를 이름순으로 세어 n 개 중 k 번째를 뺀다 (소스 안 분할, v5 와 같은 규칙)
# ("source", <소스 이름>) : 그 소스 전체를 검증용으로 뺀다 (leave-one-source-out)
SPLIT_DIR = Path(__file__).resolve().parent / "splits"

CV_FOLDS: dict[str, tuple] = {
    # P0 · P1 — v5 의 inlayer fold 와 같은 22 레이어 (v5 의 규칙 ("layers", 5, 4) 가 낸 목록을 그대로 고정한 것)
    "inlayer": ("list", "legacy_inlayer_v1.json"),
}
# inlayer 5-fold 로 넓힐 때 (§7.1 "넓히기") — 뺄 자리를 바꾼 나머지 네 fold
for _k in range(4):
    CV_FOLDS[f"inlayer_k{_k}"] = ("layers", 5, _k)
# 본 적 없는 장비 조건 (필요할 때만). 이름은 v5 와 같다
for _s in LEGACY_SOURCES:
    CV_FOLDS[f"loso_{_s['short']}"] = ("source", _s["name"])

# compose 의 cv phase 가 기본으로 도는 fold
DEFAULT_CV_FOLDS = ("inlayer",)


# ============================================================================
# v6 — Augmentation 상수 (v4 · v5 와 같음)
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

# Copy-Paste (PLAN_v6 §5.5) — v5 방식 (supplier 는 학습 분할의 GT) + 위치 규칙
CP_ENABLE = True
CP_PROB = 0.5
CP_CLASSES = (5, 6, 8, 9)            # RH · SE · OM · DB. RS (4) · SP (7) · IS (2) · PS (3) 는 넣지 않는다
CP_MIN_COMPONENT_PX = 30
CP_MAX_OBJECTS_PER_PASTE = 3
CP_FEATHER_SIGMA = 5
CP_BBOX_MAX_FRAC = 0.3
# 위치 규칙: 객체 mask 의 CP_REGION_MIN_FRAC 이상이 아래 클래스의 화소 위에 놓여야 붙인다 (target 라벨 기준).
# 표에 없는 클래스 (RS 4 · RH 5) 는 제한이 없다. 자리를 CP_POSITION_TRIES 번 뽑아도 못 채우면 그 객체는 붙이지 않는다.
CP_REQUIRED_UNDER: dict[int, int] = {
    2: 0, 3: 0, 7: 0, 9: 0,          # IS · PS · SP · DB → Powder 위
    6: 1, 8: 1,                      # SE · OM → Printed 위
}
CP_REGION_MIN_FRAC = 0.9
CP_POSITION_TRIES = 10
# 자리를 뽑는 법. True 면 제한이 있는 클래스는 "필요한 클래스의 화소 하나" 를 골라 객체의 중심을 거기에 놓고 검사한다.
# False 면 v5 처럼 이미지 전체에서 고르게 뽑아 검사한다 (Printed 는 화소의 1할쯤이라 SE · OM 이 거의 붙지 않는다).
CP_ANCHOR_ON_REQUIRED = True


# ============================================================================
# v6 — Sampler (PLAN_v6 §5.4)
# ============================================================================
# 도메인 d 안에서: q_i = MIX × (그룹 균등) + (1 − MIX) × (클래스 균등), 그 뒤 한 표본의 몫을 MAX_REPEAT / N_d 로 제한
# 도메인 사이:     p_i = share[d] × q_i,  share[diva] = min(DOMAIN_SHARE_DIVA_MAX, N_diva / N)
SAMPLER_SOURCE_MIX = 0.5
SAMPLER_MAX_REPEAT = 8.0
DOMAIN_SHARE_DIVA_MAX = 0.7


# ============================================================================
# v6 — 학습 hyper-params (단일 단계, step 기준 — v5 의 두 fold 에서 확인한 값, PLAN_v6 §5.2)
# ============================================================================
TOTAL_STEPS = 10_000
EVAL_EVERY = 250                 # 검증 주기 (step)
WARMUP_STEPS = 300
BATCH_SIZE = 4                   # per-GPU. 4-GPU DDP → effective 16
LR = 2.83e-4
WEIGHT_DECAY = 1e-4
GRAD_CLIP_NORM = 1.0
CLASS_WEIGHT_CLIP = 10.0
EMA_DECAY = 0.999                # 시상수 1,000 step ≤ TOTAL_STEPS / 10
RANDOM_SEED = 42

# ORNL figure (참고용 — 학습 · 선정에는 쓰지 않는다)
REF_INFER_BUILD = "2021-07-13 TCR Phase 1 Build 1"


# ============================================================================
# v6 — 평가에서 따로 보는 것 (PLAN_v6 §6.2)
# ============================================================================
# 혼동 쌍: 정의가 맞닿아 있는 클래스 둘 (이름, a, b)
CONFUSION_PAIRS = (("IS-PS", 2, 3), ("RS-RH", 4, 5), ("SE-DB", 6, 9), ("SP-OM", 7, 8))


# ============================================================================
# v6 — Output paths (v5 와 같은 배치)
# ============================================================================
OUTPUT_DIR = Path(__file__).resolve().parents[1]
CHECKPOINT_DIR = OUTPUT_DIR / "checkpoints"
FIGURE_DIR = OUTPUT_DIR / "figures"


def default_run_name(backbone: str = DINO_BACKBONE, img_size: int = IMG_SIZE) -> str:
    """`<backbone>_dpt_dual_sz<img>_10cls_v6` (예: vitb14_reg_dpt_dual_sz1036_10cls_v6)."""
    short = backbone[len("dinov2_"):] if backbone.startswith("dinov2_") else backbone
    return f"{short}_dpt_dual_sz{img_size}_{N_CLASSES_V6}cls_v6"


def run_tag(fold: str | None) -> str:
    """전수 학습은 `full`, 검증 fold 는 `cv/<fold>`."""
    return "full" if fold is None else f"cv/{fold}"


def run_ckpt_dir(run_name: str, fold: str | None = None) -> Path:
    return CHECKPOINT_DIR / run_name / run_tag(fold)


def run_fig_dir(run_name: str, fold: str | None = None) -> Path:
    return FIGURE_DIR / run_name / "v6" / run_tag(fold)


# ============================================================================
# v6 — Inference 휴리스틱 후처리 (v4 · v5 의 두 규칙을 새 id 로, PLAN_v6 §8). 주 지표는 후처리 없이 낸다
# ============================================================================
# 규칙 1: Printed 에서 먼 SE 성분 → Debris (제자리의 파트 / 떨어져 나온 조각)
PP_SE_SOURCE_CLASSES = (6,)              # Super Elevation
PP_SE_TARGET_CLASS = 9                   # Debris
PP_SE_FAR_DISTANCE_PX = 100
PP_SE_MIN_COMPONENT_PX = 30

# 규칙 2: Printed 와 50% 이상 겹치는 Spatter 성분 → Over Melting (분말 위 / 파트 위)
PP_SP_SOURCE_CLASS = 7                   # Spatter
PP_SP_TARGET_CLASS = 8                   # Over Melting
PP_SP_OVERMELT_OVERLAP_FRAC = 0.5
PP_SP_MIN_COMPONENT_PX = 30
PP_SP_PRINTED_DILATE_PX = 10
