# DefSeg-AM **v4** 실험 계획

> v3 ([PLAN_v3.md](../v3/PLAN_v3.md)) 의 후속.
> 핵심 변경:
> - **클래스 11개** (v3 의 10 에 **Under Melting 신규 추가**, Incomplete Spreading / Swelling / SE 는 v3 처럼 분리 유지. Misprint 만 IGNORE)
> - **백본 업그레이드**: DINOv2 ViT-S/14 → **ViT-B/14 with Registers**
> - **Class-aware sampling**: rare class 풍부한 layer 위주로 oversample
> - **Copy-Paste augmentation**: rare class object 를 다른 layer 에 합성 (CutMix 의 segmentation 발전형)
> - **후처리 규칙 1개 추가**: Printed 위의 Spatter → Over Melting

---

## 1. v3 → v4 변경 요약

| 영역 | v3 | v4 |
|---|---|---|
| 클래스 수 | 10 | **11** |
| 클래스 변경 | — | **Under Melting 신규 추가** (v3 에서 IGNORE 였음). Incomplete Spreading / Swelling / SE 분리 유지. Misprint 만 IGNORE |
| **Backbone** | dinov2_vits14 (22 M) | **dinov2_vitb14_reg (86 M + Registers)** |
| **Decoder** | DPT (13.4 M trainable) | DPT (14.6 M trainable, channel 호환 자동) |
| Augmentation pipeline | flip(LR/UD)+rot180+cyclic+DSCNN | **동일** + **Copy-Paste (rare class)** |
| Sampler | `defect_ratios ** 0.5` (defect 전체 비율) | **class-aware: rare class 픽셀 풍부한 layer weighted** |
| 휴리스틱 후처리 | SE/Swelling far → Debris (1 규칙) | 동일 + **Printed 위 Spatter → Over Melting (2 규칙)** |
| 학습 hyperparameter | epoch 30/50, batch 2, lr 2e-4, K=4 | epoch/K 동일, **batch 2 → 4 (per-GPU)**, **lr 2e-4 → 2.83e-4** (sqrt scaling) |
| Effective batch (4-GPU DDP) | 8 | **16** |
| 4-GPU DDP | torchrun nproc=4 | **동일** |
| Stage 1 학습 시간 | ~3 h 20 m | ~3-4 h (backbone forward ↑ 와 batch 4 의 step 절반 ↓ 가 상쇄) |

---

## 2. 클래스 정의 (11개)

### 2.1 클래스 목록

| ID | Name | 정의 |
|---:|---|---|
| 0 | Powder | 이상 또는 프린트된 파트가 없는 분말 베드 영역 |
| 1 | Printed | 이상이 감지되지 않은 프린트 영역 |
| 2 | Recoater Hopping | 리코터가 표면 아래 파트에 충돌할 때 발생하는 물결무늬 |
| 3 | Recoater Streaking | 리코터 손상 또는 큰 입자 끌림으로 인한 줄무늬 |
| 4 | Incomplete Spreading | 분말 베드에 불충분한 분말 도포 |
| 5 | Swelling | 분말 위로 돌출된 프린트 재료의 변형/뒤틀림 |
| 6 | Debris | 분말 베드의 소-중형 교란 (포괄적 클래스) |
| 7 | Super-Elevation | 프린트 영역 위의 분말 커버리지 부족 |
| 8 | Spatter | 용접 풀에서 튀어나와 분말 베드에 착지한 비산물 |
| 9 | Over Melting | 고에너지 밀도 공정 파라미터로 용융된 영역 |
| 10 | **Under Melting** | **고에너지 부족으로 충분히 용융되지 않은 영역** (신규) |

**변경 (vs v3)**:
- **신규 클래스 `Under Melting (10)`** — ORNL HDF5 의 native 11 을 학습 대상으로
  부활. 총 28.3 M 픽셀 (B2 에 99.9% 집중) — 분포 편향 크지만 픽셀 수 자체는 충분
- 그 외 모든 클래스는 v3 와 동일 (Incomplete Spreading / Swelling / SE 분리 유지)
- IGNORE 처리되는 native class 는 Misprint (9) 만

**v1 ORNL native 12 → v4 11 매핑** (`ORNL_12_TO_NEW_11_V4`):

```
0  Powder              → 0   Powder
1  Printed             → 1   Printed
2  Recoater Hopping    → 2   Recoater Hopping
3  Recoater Streaking  → 3   Recoater Streaking
4  Incomplete Spreading→ 4   Incomplete Spreading
5  Swelling            → 5   Swelling
6  Debris              → 6   Debris
7  Super-Elevation     → 7   Super-Elevation
8  Spatter             → 8   Spatter
9  Misprint            → -1  IGNORE
10 Over Melting        → 9   Over Melting
11 Under Melting       → 10  Under Melting   ← 신규 학습 대상
```

→ ORNL 12-class 의 11개가 v4 에 1:1 매핑되며, Misprint 만 IGNORE. v3 보다 학습
대상 클래스가 1개 더 많아 라벨 정보 활용도 ↑.

### 2.2 DSCNN_Dataset native → v4 매핑

원칙: native → ORNL 12-class → v4 11-class (2-step). `MATERIAL_TO_ORNL_V4` 는
v3 의 `MATERIAL_TO_ORNL_V3` 를 그대로 import. 그 뒤 `ORNL_12_TO_NEW_11_V4` 적용.

DSCNN_Dataset 의 사람 라벨에 Incomplete Spreading / Under Melting 이 있는지는
build_cache_v4 단계에서 확인. 일반적으로 DSCNN_Dataset 의 일부 source
(예: v2021_BJ 의 Powder Short Feed) 가 Incomplete Spreading 에 매핑됨 — Stage 2
에서도 학습 데이터 일부 존재. Under Melting 은 거의 없으므로 **Stage 1 (ORNL HDF5)
의 28 M 픽셀이 사실상 유일한 학습 소스**.

---

## 3. 동기 — Class 불균형 + 빌드 의존 결함

ORNL Baseline (Peregrine v2023-11) 의 픽셀 분포 (B1-B5 합산, v4 매핑 적용 후):

| v4 ID | Class | Total (px) | 집중 빌드 | 점유율 |
|---:|---|---:|---|---:|
| 0 | Powder | 15.33 G | 균등 | — |
| 1 | Printed | 705 M | 균등 | — |
| 2 | Recoater Hopping | 25 | B4 만 | **사실상 없음** |
| 3 | Recoater Streaking | 187 M | B2 + B5 균등 | — |
| 4 | Incomplete Spreading | 867 K | B1+B2+B3+B4 균등 (B5 적음) | 약간 편향 |
| 5 | Swelling | 10 M | B2 (57%) | 편향 |
| 6 | Debris | 1.8 M | B3 (90%) | 강한 편향 |
| 7 | Super-Elevation | 268 K | B5 (98%) | 극단 편향 |
| 8 | Spatter | 667 M | B4 (62%) | 편향 |
| 9 | Over Melting | 394 K | B2 (99.8%) | 극단 편향 |
| 10 | **Under Melting (신규)** | **28 M** | **B2 (99.9%)** | **극단 편향** |

→ v4 의 rare class 우선순위 (학습 데이터 부족 정도):
1. **Recoater Hopping (25 픽셀)** — 사실상 학습 불가, Copy-Paste 필수
2. **Super-Elevation (268 K, B5 만)** — Copy-Paste 필요
3. **Over Melting (394 K, B2 만)** — B2 분포 편향, Copy-Paste 권장
4. **Incomplete Spreading (867 K)** — 빌드별 균등하지만 픽셀 수 매우 적음, Copy-Paste 권장
5. **Debris (1.8 M, B3 만)** — Copy-Paste 권장
6. **Under Melting (28 M, B2 만)** — 픽셀 수 충분하지만 build 편향 → sampler 만으로
   부족할 수 있음, Copy-Paste 권장
7. **Swelling (10 M, B2 57%)** — 픽셀 풍부, sampler 만으로도 가능
8. Recoater Streaking / Spatter — common 으로 간주

→ **class-aware sampling + Copy-Paste 의 결합** 으로 모든 rare class 학습 빈도 ↑.

---

## 4. 핵심 알고리즘 — Class-Aware Layer Sampling

### 4.1 Per-layer per-class 픽셀 카운트 사전 계산

각 layer ℓ 에 대해 클래스별 픽셀 카운트 벡터 `pix[ℓ, c]` (shape:
`(N_layers, N_CLASSES_V4=8)`).

- Stage 1: ORNL HDF5 의 v4 label cache (`label_v4.npy`) 1-pass — chunk 단위
  카운트 후 `pix_per_layer.npy` 저장. build_cache_v4 단계에서 한 번만.
- Stage 2: DSCNN_Dataset (img0, img1, ann) 별 GT label 1-pass — sample 수 적어
  비용 무시.

### 4.2 Class importance weight (class 단위)

```
total_pix[c] = Σ_ℓ pix[ℓ, c]
class_w[c]   = 1 / max(total_pix[c], eps) ** α
class_w[c]  /= sum(class_w)          # 정규화 (단순 비교용)
```

rare class 일수록 `class_w[c]` 큼. `α` 는 강조 강도.

### 4.3 Layer weight 계산

```
layer_w[ℓ] = Σ_c class_w[c] × pix[ℓ, c]  + eps
```

해석: rare class 픽셀 많은 layer 가 큰 weight.
common class (Powder/Printed) 픽셀은 contribution 작음.

### 4.4 Class focus power α

v3 의 `defect_ratios ** 0.5` 의 일반화. v4 default:

| α | 효과 |
|---:|---|
| 0.5 | sqrt-inverse (v3 의 oversample_power 와 같은 수준) |
| **1.0** | **inverse-frequency (v4 default)** — rare class 강한 강조 |
| 1.5 | 매우 강한 강조 (common class 학습 후퇴 위험) |

→ v4 default: **α = 1.0**.

### 4.5 DistributedWeightedSampler 통합

기존 `common/data/samplers.py::DistributedWeightedSampler` 의 `weights` 인자에
`layer_w` 를 그대로 전달. 코드 변화 최소.

---

## 5. Stage 2 의 class-aware sampler (DSCNN_Dataset)

v3 의 source 균등 (`compute_source_balanced_weights`) 정책 유지 + class-aware
weight 를 **곱** 으로 결합:

```
src_w[i]         = 1 / (n_src × n_in_source[src(i)])   # v3 와 동일
class_w[c]       = 1 / max(total_pix[c], eps) ** α
sample_pix[i, c] = sample i 의 GT 클래스 c 픽셀 수
sample_cls_w[i]  = Σ_c class_w[c] × sample_pix[i, c]
combined_w[i]    = src_w[i] × sample_cls_w[i] + eps
```

→ "source 균등 + 그 source 내에서 rare class 풍부한 sample 우선".

`replicate_factor K=4`, `S2_RANDOM_SEED=42` 는 v3 와 동일.

---

## 6. Augmentation pipeline (v3 base + Copy-Paste 추가)

### 6.1 v3 와 동일한 부분

- flip LR / UD (각 독립 50%)
- 180° rotation (50%)
- cyclic shift ±IMG/4 (1-px uniform, prob 0.5)
- DSCNN Gaussian noise / mean intensity shift
- brightness multiplicative jitter

### 6.2 신규 — Copy-Paste augmentation (rare class)

**동기**: class-aware sampler 가 "rare class supplier layer 가 자주 등장" 은 해결하지만,
*supplier 자체가 절대 부족한 경우* (Recoater Hopping 25 픽셀, Super-Elevation B5
의존 등) 는 sampling 으로 못 풂. **합성으로 데이터 자체를 늘림**.

CutMix (Yun et al. 2019) 의 segmentation 발전형인
[Copy-Paste (Ghiasi et al. 2020, "Simple Copy-Paste is a Strong Data
Augmentation Method for Instance Segmentation")](https://arxiv.org/abs/2012.07177) 채택.

**알고리즘** (training time, sample 단위):

```
1. 현재 sample (img0_T, img1_T, ann_T) — "target"
2. 확률 CP_PROB (default 0.5) 로:
   a. rare class supplier pool 에서 source sample 1 개 추출
   b. source 의 GT label 에서 rare class connected component 들 찾기
   c. 각 component 면적 ≥ CP_MIN_COMPONENT_PX 인 것만 사용
   d. 최대 CP_MAX_OBJECTS_PER_PASTE 개 random 선택
   e. 각 object 의 bbox + binary mask 를 추출
   f. random shift 위치 (target 안 random pixel)
   g. visible/0, visible/1 둘 다 동시에 paste
   h. Gaussian feathering (σ = CP_FEATHER_SIGMA) 으로 부드러운 경계
   i. label 도 동일 위치에 paste
3. paste 후 §6.1 의 기존 augmentation 적용
```

**Rare class 정의 — Stage 별로 다르게**:

| ID | Class | Pixel total (Builds 2-5) | **Stage 1** | **Stage 2** | 동기 |
|---:|---|---:|:---:|:---:|---|
| 5 | Swelling | 10 M | ✅ | ✅ | B2 57% 편향, Stage 1 의 핵심 학습 대상 |
| 7 | Super-Elevation | 268 K | ✅ | ✅ | B5 98% 의존, 핵심 학습 대상 |
| 2 | Recoater Hopping | 25 | ✗ | ✅ | Stage 1 train 에 사실상 없어 paste source 부재 |
| 4 | Incomplete Spreading | 867 K | ✗ | ✅ | 빌드 균등하지만 픽셀 적음, Stage 2 의 DSCNN 사람 라벨에 의존 |
| 6 | Debris | 1.8 M | ✗ | ✅ | B3 의존이지만 Stage 1 에서는 학습 우선순위 낮음 |
| 9 | Over Melting | 394 K | ✗ | ✅ | B2 의존이지만 Stage 1 에서는 학습 우선순위 낮음 |
| 10 | Under Melting | 28 M | ✗ | ✅ | B2 의존이지만 Stage 1 에서는 우선순위 낮음 (단 DSCNN_Dataset 에 라벨 없으면 Stage 2 도 효과 미미) |

→ Stage 1 의 `CP_RARE_CLASSES_S1 = (5, 7)` — Swelling + Super-Elevation.
   Stage 2 의 `CP_RARE_CLASSES_S2 = (2, 4, 5, 6, 7, 9, 10)` — Hopping, Inc.Spr.,
   Swelling, Debris, SE, Over Melt, Under Melt.

이렇게 분리한 이유: Stage 1 의 핵심은 사용자 결정 ("SE, Swelling 만 Stage 1
에서도 증강하고 나머지는 Stage 2 에서만") 반영. 나머지 rare class 들은
DSCNN_Dataset 의 사람 라벨에서 (가능한 경우) 더 신뢰성 있게 학습.

**Source supplier pool**: 학습 시작 시 `pix_per_layer` 로부터 각 rare class 마다
"해당 class 픽셀 ≥ CP_MIN_COMPONENT_PX 인 layer" 목록 사전 계산.

**Pasting 디테일**:

- **위치**: target image 의 random (x, y). 도메인 sanity (Powder 위 SE 비현실) 무시 — 모델은 패턴만 학습
- **두 visible 채널 동기**: `(visible/0, visible/1)` 페어를 source 에서 같이 잘라 같은 위치에 paste — 후처리 규칙 1 (정적성) 의 의미 보존
- **Feathering**: bbox 가장자리에서 Gaussian (σ=`CP_FEATHER_SIGMA`=5 px) 으로 알파 blending — sharp 경계 회피
- **Label**: feather 영향 없이 binary paste (segmentation label 은 분리)
- **IGNORE 픽셀**: source 의 IGNORE 영역은 target 의 원본 label 유지

**상수** (보수적 default — 실 데이터 너무 망가뜨리지 않도록):

| 상수 | 값 | 의미 |
|---|---|---|
| `CP_ENABLE` | True | 켜기 |
| `CP_PROB` | 0.5 | sample 당 적용 확률 |
| `CP_RARE_CLASSES_S1` | **(5, 7)** | Stage 1: Swelling + Super-Elevation |
| `CP_RARE_CLASSES_S2` | **(2, 4, 5, 6, 7, 9, 10)** | Stage 2: Hopping, Inc.Spr., Swelling, Debris, SE, Over Melt, Under Melt |
| `CP_MIN_COMPONENT_PX` | 30 | 너무 작은 component 무시 |
| `CP_MAX_OBJECTS_PER_PASTE` | 3 | 한 sample 당 paste 할 object 최대 |
| `CP_FEATHER_SIGMA` | 5 | Gaussian 알파 blending 폭 (px) |
| `CP_BBOX_MAX_FRAC` | 0.3 | source component bbox 가 image 의 30% 초과면 paste 안 함 (너무 큰 object 보호) |

### 6.3 적용 순서

매 sample 당:
1. **Copy-Paste** (확률 `CP_PROB`)
2. cyclic shift (확률 0.5)
3. flip LR, flip UD (각 50%)
4. 180° rotation (50%)
5. DSCNN Gaussian noise / intensity shift
6. brightness multiplicative jitter

Copy-Paste 가 가장 먼저 적용되어 기존 기하 augmentation 이 합성된 object 도 함께
변환 → 합성된 object 가 다양한 위치·방향으로 학습됨.

### 6.4 사전 계산 산출물

`build_cache_v4` 단계에서 `pix_per_layer.npy` 외 추가로:

- `rare_class_supplier.json`: 각 rare class c 마다 `{layer_idx, component_count, build_id}` 목록
- 학습 시작 시 build 별 visible/0, visible/1 memmap 만 열어두면 random source crop 가능

---

## 7. 학습 단계

### 7.1 Stage 1 — KD pretrain

- 데이터: ORNL Build 2-5 train, B1 val (v3 동일)
- 라벨: v4 11-class (위 §2.1 매핑)
- **Backbone**: dinov2_vitb14_reg (86 M, frozen) ← v3 의 vits14 (22 M) 에서 업그레이드
- **Decoder**: DPT (14.6 M trainable) — backbone embed_dim 384→768 으로 입력 channel
  자동 확장. decoder output channel 은 그대로 256
- 손실: focal loss + sqrt-inv class weight (v3 동일)
- EMA 0.9999, **per-GPU batch 4** (effective 16), **lr 2.83e-4** (= 2e-4 × √2, sqrt scaling),
  30 epoch, warmup 200, AdamW
- **Sampler**: class-aware (위 §4) — α=1.0
- **Augmentation**: §6.3 — Copy-Paste 는 `CP_RARE_CLASSES_S1 = (5, 7)` (Swelling + Super-Elevation)
- 4-GPU DDP
- 메모리: ViT-S batch 2 = 11.8 GB → ViT-B batch 4 ≈ 20 GB per GPU (실측 기준, 32 GB 의 65%)

### 7.2 Stage 2 — DSCNN_Dataset GT 전수 + ORNL Build 1 평가

- v3 와 동일: DSCNN 8 source 전부 train + 매 epoch ORNL Build 1 200 layer eval
- Backbone / decoder: §7.1 과 동일 (Stage 1 best ckpt 에서 init)
- **Sampler**: source 균등 × class-aware (위 §5) — 곱
- **Augmentation**: §6.3 — Copy-Paste 는 `CP_RARE_CLASSES_S2 = (2, 4, 5, 6, 7, 9, 10)` (Hopping, Inc.Spr., Swelling, Debris, SE, Over Melting, Under Melting).
  Stage 2 의 supplier pool 은 DSCNN_Dataset 자체 sample (8 source GT) 에서 추출
- epoch 50, **per-GPU batch 4** (effective 16), **lr 2.83e-4**, K=4, seed=42, EMA on

### 7.3 Class weight (loss)

`sqrt_inv_class_weight` (clip 10) — v3 동일. class-aware sampler 가 이미
rare class 노출 빈도를 높이므로 loss weight 추가 강화는 불필요.

---

## 8. 휴리스틱 후처리 (2 규칙)

학습/Loss 와 무관. Inference 직전 적용. CLI: `--postprocess`.

### 8.1 규칙 1 — 부품에서 먼 SE/Swelling → Debris (v3 와 동일 동작)

- 대상: Super-Elevation (v4 ID 7), Swelling (v4 ID 5) 의 connected component
- 면적 ≥ `PP_SE_MIN_COMPONENT_PX` (= 30)
- Printed (1) mask 의 nearest pixel 까지 **최소** 거리 ≥
  `PP_SE_FAR_DISTANCE_PX` (= 100 px)
- 조건 모두 만족 시 → Debris (v4 ID 6)

> v3 와 동일 패턴. 코드 상수:
> `PP_SE_SOURCE_CLASSES = (5, 7)` (Swelling, SE),
> `PP_SE_TARGET_CLASS = 6` (Debris).

### 8.2 규칙 2 (신규) — Printed 영역 위의 Spatter → Over Melting

도메인 가설: Spatter (비산물) 는 보통 Powder 위에 착지. 만약 Spatter 가 Printed
영역 위에 직접 분포한다면, 그 영역은 Spatter 보다는 **강한 melt pool 이 Printed
표면을 녹인 결과 (Over Melting)** 일 가능성이 높음.

**알고리즘** (connected component 단위):

1. Spatter mask (pred == 8) 의 connected component 추출
2. 각 component 마다:
   - 면적 ≥ `PP_PS_MIN_COMPONENT_PX` (작은 노이즈 무시)
   - **Printed (pred == 1) mask 와의 overlap 비율** 계산
     = `(component ∩ printed) 픽셀 수` / `component 픽셀 수`
   - overlap 비율 ≥ `PP_PS_OVERMELT_OVERLAP_FRAC` (= "component 의 대부분이 Printed 위") 면 변경 대상
3. 조건 만족 component 의 모든 픽셀 → Over Melting (v4 ID 9)

**상수** (보수적 — 실 Spatter 가 사라지지 않도록):

| 상수 | 값 | 의미 |
|---|---|---|
| `PP_PS_OVERMELT_OVERLAP_FRAC` | **0.5** | component 의 50% 이상이 Printed 위면 Over Melting |
| `PP_PS_MIN_COMPONENT_PX` | 30 | 작은 노이즈 무시 |
| `PP_PS_TARGET_CLASS` | 9 | Over Melting |
| `PP_PS_SOURCE_CLASS` | 8 | Spatter |

**보수성 근거**:
- overlap 50% 임계 → "절반 이상이 Printed 위" 일 때만 변경. 진짜 Powder 위 Spatter
  (대부분 component 면적이 Powder 영역) 는 안 건드림
- component 단위 판정 → 산발적인 픽셀 노이즈가 아닌 의미 있는 spatial cluster 만
- 두 visible 채널 정보는 사용 안 함 (Printed mask 자체로 충분)

### 8.3 적용 순서

1. 규칙 1 (Super-Elev → Debris)
2. 규칙 2 (Printed 위 Spatter → Over Melting)

순서 의미: 규칙 1 은 Super-Elev 위치 기반, 규칙 2 는 Spatter↔Printed 의 overlap
기반 — 서로 영향 없음. 순서 무관하지만 명확성 위해 1 → 2.

---

## 9. 폴더 구조

```
DefSeg_AM/v4/
├── PLAN_v4.md
├── README.md
├── __init__.py
├── config_v4.py              # 11-class 상수 + ClassAware_* + CP_* + PP_PS_* + BACKBONE
├── data/
│   ├── __init__.py
│   ├── augmentation.py        # v3 base + Copy-Paste 통합
│   ├── copy_paste.py          # Copy-Paste 알고리즘 (source crop, feathering, paste)
│   ├── data_ornl_v4.py        # v3 dataset + per-layer per-class 카운트 로딩 + Copy-Paste supplier
│   ├── data_dscnn_v4.py       # v3 dataset + class-aware combined weight 헬퍼 + Copy-Paste supplier
│   ├── build_cache_v4.py      # 11-class 재매핑 + pix_per_layer.npy + rare_class_supplier.json
│   └── sampler_helpers.py     # class_w / layer_w 계산 함수
├── models/
│   └── losses_v4.py           # v3 import 재export
├── training/
│   ├── train_stage1.py        # v3 base, sampler weight 만 class-aware
│   └── train_stage2.py        # v3 base, sampler weight 만 결합형
├── inference/
│   ├── infer.py               # 11-class palette
│   ├── postprocess.py         # 규칙 1 + 규칙 2 (Printed↔Spatter)
│   └── confusion.py
├── scripts/
│   ├── run_build_cache_v4.sh
│   ├── run_stage1_v4.sh
│   ├── run_stage2_v4.sh
│   └── run_all_v4.sh
└── docker/
    ├── Dockerfile
    ├── docker-compose.yml
    └── README.md
```

---

## 10. 산출물 / 캐시

- **v4 label cache**: `DefSeg_AM/cache/resized_sz1036_v4/<build>/label_v4.npy` (별도 매핑)
- **v4 픽셀 통계**: `DefSeg_AM/cache/resized_sz1036_v4/<build>/pix_per_layer.npy`
  shape `(n_layers, 11)`, int64
- **Copy-Paste supplier 인덱스**: `DefSeg_AM/cache/resized_sz1036_v4/rare_class_supplier.json`
  각 rare class 마다 supplier layer 목록
- **image cache 는 v1 재사용**: `cache/resized_sz1036/<build>/visible_{0,1}.npy`
- **ckpt**: `DefSeg_AM/checkpoints/vitb14_reg_dpt_dual_sz1036_11cls_v4/{stage1_best, stage2_best}.pt`
  ※ backbone prefix `vitb14_reg_` + 클래스 수 `11cls` + `_v4`
- **figure**: `DefSeg_AM/figures/vitb14_reg_dpt_dual_sz1036_11cls_v4/v4/...`

---

## 11. 평가

### 11.1 정량

- Stage 1 / Stage 2: ORNL Build 1 confusion + per-class IoU + mIoU
- **rare class 중심 비교**: v3 vs v4 같은 평가셋에서
  - Super-Elev (v4 통합) 의 IoU 가 v3 의 Swelling+SE 각각보다 향상됐는지
  - Debris / Over Melting 의 prediction 자체가 나타나는지 (v3 는 거의 0)
- **B5 보조 평가** (선택): Super-Elev 평가가 의미 있는 build 1 외 inference

### 11.2 정성

- v1 / v2 / v3 / v4 비교 figure — 기존 `compare_v1_v2_v3.py` 확장
- 후처리 전/후 figure 비교

### 11.3 v3 vs v4 ablation 문서

`RESULTS_v3_vs_v4.md` — 같은 평가셋, 같은 모델, sampler + 클래스 정의만 다른
비교.

---

## 12. 확정된 hyperparameter 요약

| 항목 | 값 | 출처 |
|---|---|---|
| 클래스 수 | **11** (v3 의 10 에 + Under Melting 신규 추가) | §2 |
| **Backbone** | **dinov2_vitb14_reg** (86 M frozen + Registers 4 token) | §1, §7.1 |
| Decoder | DPT (14.6 M trainable) | §7.1 |
| Augmentation pipeline | §6.3 (v3 base + Copy-Paste) | §6 |
| **Copy-Paste** | rare class object 합성 — Stage 별 다른 set | §6.2 |
| `CP_PROB` | 0.5 | §6.2 |
| `CP_RARE_CLASSES_S1` | **(5, 7)** — Swelling, Super-Elevation | §6.2 |
| `CP_RARE_CLASSES_S2` | **(2, 4, 5, 6, 7, 9, 10)** — Hopping, Inc.Spr., Swelling, Debris, SE, Over Melt, Under Melt | §6.2 |
| `CP_MIN_COMPONENT_PX` | 30 | §6.2 |
| `CP_MAX_OBJECTS_PER_PASTE` | 3 | §6.2 |
| `CP_FEATHER_SIGMA` | 5 (px) | §6.2 |
| `CP_BBOX_MAX_FRAC` | 0.3 | §6.2 |
| **Sampler (Stage 1)** | class-aware: `Σ_c (1/total_pix[c])^α × pix[ℓ, c]` | §4 |
| **Sampler α** | **1.0** (inverse-frequency) | §4.4 |
| **Sampler (Stage 2)** | source 균등 × class-aware (곱) | §5 |
| Replicate factor K (Stage 2) | 4 | v3 동일 |
| Random seed | 42 | v3 동일 |
| Epoch | 30 (S1), 50 (S2) | v3 동일 |
| **Per-GPU batch** | **4** (effective 16 with 4-GPU DDP) | §7 |
| **LR** | **2.83e-4** (= 2e-4 × √2, sqrt scaling) | §7 |
| EMA decay | 0.9999 | v3 동일 |
| Loss | focal + sqrt-inv class weight clip 10 | v3 동일 |
| 4-GPU DDP | torchrun nproc=4 | v3 동일 |
| **후처리 규칙 1** | SE/Swelling → Debris (far from Printed) | v3 와 동일 동작 |
| **후처리 규칙 2** | Printed 위 Spatter → Over Melting (overlap ≥ 50%) | **신규 — overlap 기반** |
| `PP_SE_SOURCE_CLASSES` | (5, 7) — Swelling, Super-Elevation | §8.1 |
| `PP_SE_TARGET_CLASS` | 6 — Debris | §8.1 |
| `PP_SE_FAR_DISTANCE_PX` | 100 | v3 동일 |
| `PP_SE_MIN_COMPONENT_PX` | 30 | v3 동일 |
| `PP_PS_SOURCE_CLASS` | 8 — Spatter | §8.2 |
| `PP_PS_TARGET_CLASS` | 9 — Over Melting | §8.2 |
| `PP_PS_OVERMELT_OVERLAP_FRAC` | 0.5 | §8.2 (신규) |
| `PP_PS_MIN_COMPONENT_PX` | 30 | §8.2 (신규) |

---

## 13. 미해결 / 추후 결정

학습 시작 전에 결정한 default 는 §12 hyperparameter 요약 참조. 학습 후 결과에
따라 조정 가능한 항목만 아래 정리.

1. **`PP_PS_OVERMELT_OVERLAP_FRAC` (= 0.5)**: 후처리 규칙 2 의 overlap 임계가
   적정한지 — Build 1 inference figure 보고 ±0.1 조정 가능
   - false positive (정상 Spatter 가 Over Melting 으로 잘못 변경) 많으면 0.6
   - 효과 미미 (실제 Over Melting 케이스가 안 잡힘) 면 0.4
2. **Sampler oversample 후 common class 후퇴**: Stage 1 학습 결과에서 Powder
   IoU < 0.95 면 sampler α 를 0.7 로 낮추고 재학습
3. **CP_PROB (= 0.5) 의 자연성**: train loss 가 매우 noisy 하면 0.3 으로 낮추는
   옵션. 단 우선은 0.5 그대로
4. **Stage 1 의 `CP_RARE_CLASSES_S1 = (5, 7)` 효과**:
   Stage 1 후 Swelling / Super-Elevation IoU 가 의미 있게 오르는지 확인. 효과
   미미하면 Stage 1 에서도 Inc.Spr./Debris/Hopping/Under Melting 추가
5. **Under Melting (10) 의 DSCNN_Dataset 학습 가능성**: DSCNN_Dataset 8 source 의
   사람 라벨에 Under Melting 이 있는지 build_cache_v4 단계에서 확인 필요. 없으면
   Stage 2 의 Under Melting Copy-Paste 가 무의미 → S2 의 rare class set 에서 10 제거
   (Stage 1 ORNL HDF5 의 B2 28M 픽셀이 유일한 학습 소스)
6. **Under Melting 의 B2 의존**: 99.9% B2 분포 → Stage 1 의 class-aware sampler 가
   B2 layer 만 압도적으로 추출하게 됨. 다른 빌드 layer 가 거의 안 등장하면 generalize
   문제 가능 — Powder/Printed/Spatter 등 다른 클래스 학습이 후퇴할 수 있음
7. **Incomplete Spreading (4) 의 학습 가능성**: 867 K 픽셀 (모든 빌드에 균등) +
   DSCNN_Dataset 의 일부 source (예: Powder Short Feed) 에도 분포. v3 에서는 IGNORE
   였지만 v4 에서는 학습 대상 — Stage 1/2 의 IoU 가 의미 있게 나오는지 확인
