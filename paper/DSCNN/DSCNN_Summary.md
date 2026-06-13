# DSCNN 논문 상세 요약

> **원본**: Scime, L.; Siddel, D.; Baird, S.; Paquit, V. *"Layer-wise anomaly detection and classification
> for powder bed additive manufacturing processes: A machine-agnostic algorithm for real-time pixel-wise
> semantic segmentation"*, **Additive Manufacturing 36 (2020) 101453**.
> DOI: [10.1016/j.addma.2020.101453](https://doi.org/10.1016/j.addma.2020.101453)
>
> 본문 출처: 본 폴더의 [DSCNN 원본 PDF](.) (21 pages, 텍스트 추출 가능).
> 본 요약은 학습/추론 관련 디테일을 그대로 보존하기 위해 원문 인용을 다수 포함합니다.

---

## 0. 키워드

semantic segmentation, layer-wise anomaly detection, powder bed additive manufacturing, machine-agnostic,
multi-scale CNN, transfer learning, U-Net, real-time

## 1. 알고리즘 목표 (Section 1.4)

저자가 명시한 4 가지 요구사항:

1. **Real-time** : 고해상도에서도 실시간 예측 가능
2. **Native resolution pixel-wise segmentation**: 카메라 원 해상도에서 픽셀 단위 라벨
3. **Machine-agnostic transfer learning**: 한 AM 머신에서 학습한 지식을 다른 머신으로 빠르게 전이
4. **Multi-modal sensor fusion**: 여러 카메라/센서를 자연스럽게 융합

→ DSCNN 의 모든 아키텍처 결정은 위 4 가지를 만족시키기 위한 것.

---

## 2. 분류 대상 anomaly (Section 1.2, 2)

### 2.1 라벨 클래스

본문에서 정의된 표면 가시 anomaly:

| Anomaly | 정의 | 시각 특징 |
|---|---|---|
| **Recoater Hopping** | rigid recoater blade 가 powder surface 아래의 part 와 부딪침 | recoater 진행 방향에 **수직** 한 반복적 줄무늬 |
| **Recoater Streaking** | recoater 손상 또는 powder bed 의 큰 contaminant 끌림 | recoater 진행 방향에 **평행** 한 줄 |
| **Incomplete Spreading (Short Feed)** | 분말이 충분히 도포되지 않음 | 부분적으로 powder 없는 영역 |
| **Debris** | 분말 베드의 소·중형 교란 (포괄 클래스) | 작은 점/얼룩 |
| **Spatter / Soot** | 용융 시 ejecta. soot 는 저해상도에서 보이는 spatter | 흩뿌려진 입자 |
| **Swelling / Super-Elevation** | 부품이 분말 위로 돌출 (warping, 열응력) | 부품 가장자리/내부의 변형 |
| **Misprint** | 의도된 part 형상 외부에 인쇄된 재료 | 의도하지 않은 위치의 fused 영역 |
| **Part Damage** | 부품 손상 (직접 학습 안 함, **heuristic 으로만 예측**) | — |

### 2.2 머신/재료 데이터

본 논문은 6 개 머신 × 3 기술 (L-PBF, EB-PBF, BJ) 에서 검증.
본 프로젝트 (DefSeg-AM) 는 그 중 **ConceptLaser M2 (L-PBF, SS316L)** 만 사용.

### 2.3 라벨링 가이드라인 (Section 3.2)

저자가 따른 7 가지 규칙:

1. **모든 anomaly class 가 표현됨**. class 마다 최소 **100,000 픽셀** 라벨링
2. **다양성 > 수량**: rare event 가 잘 표현되는 게 단순 수량보다 중요
3. 단일 build 만 사용하지 말 것 — 여러 build 의 다양성 확보
4. **다중 라벨 픽셀**: super-elevation > incomplete spreading 처럼 더 심각한 결함 우선
5. 모든 카메라 (post-spread + post-fusion) 의 이미지를 동시 고려해서 라벨
6. powder bed 의 **모든 영역** 에서 라벨
7. 최소 **3 layer** 이상 라벨링 후 학습 시작 (transfer learning 결합 시)

> "Training a CNN (Section 3.7) requires a significant amount of ground truth data, typically on the order
> of **10⁵ – 10⁷ targets** (labeled samples) [44]."

---

## 3. 입력 구성: Image Stacks A, B, C, D (Section 3.3)

DSCNN 의 입력은 **4 개 image stack** 으로 구성됨. 각 stack 이 다른 size scale 의 정보를 담음.

| Stack | 크기 (pixel) | 채널 수 | 정보 source | 역할 |
|---|---|---|---|---|
| **A** | 32×32 (padded 36×36) | `2·n_cam` | 전체 layer 이미지를 32×32 로 bilinear resize | **글로벌** context (전체 powder bed 상태) |
| **B** | `T+10` × `T+10` (native tile + 5 padding) | `2·n_cam` | 원 해상도 tile (5 pixel symmetric pad) | **로컬** (recoater streaking 등 fine detail) |
| **C** | 132×132 | `2·n_cam` | tile 을 128×128 로 bilinear resize 후 2-pad | **regional** (mid-scale 형태 정보) |
| **D** | T × T | 2 | (x, y) 좌표 맵 | **위치 정보** (절대 좌표) |

여기서:
- **T** = native tile size. 머신별로 다름: EOS M290 = 128, ConceptLaser M2 = 225, Renishaw AM250 = 200, ExOne M-Flex = 400, ExOne Innovent = 450, Arcam Q10 = 200 (pixel 단위)
- **`n_cam`** = 카메라 수. 채널은 `2·n_cam` = (post-spread + post-fusion) × `n_cam`
- 각 layer 이미지를 grid of tiles 로 자르고, 끝부분은 image border 에 pin (overlap 발생 시 마지막 tile 이 overwrite)

### 정규화 (Eq. 1)

**Non-canonical (이미지 단위가 아닌 global 단위 정규화)**:

```
u_normalized = u − ū
```

`u` 는 pixel intensity, `ū` 는 calibration 시점에 결정된 **global mean intensity**.
8-bit 입력 가정. 출력은 32-bit float.

> 이유: "the absolute intensity of an individual pixel encodes valuable information"
> 일반 이미지 분류와 달리 절대 intensity 가 의미 있어서 image-wise 정규화 안 함.

---

## 4. DSCNN 아키텍처 (Section 3.4)

세 개의 **parallel leg** + concat + classification.

```
Stack A (36×36)  ─→ [Global CNN]       ─┐
Stack B (T+10)   ─→ [Localization CNN] ─┼─→ Concat + (x,y) ─→ Classification (1×1 conv) ─→ Softmax
Stack C (132)    ─→ [Regional U-Net]   ─┘
```

총 학습 가능 파라미터: **13,996,612 + 179·(n_classes + 1)** ≈ 14M.

### 4.1 Regional U-Net (Stack C → 128×128 tile features)

**가장 깊은 leg.** U-Net 구조 (Ronneberger et al. [35]).

| 단계 | Layer | 출력 채널 | 비고 |
|---|---|---|---|
| Encoder | 5×5 CONV + ReLU + BN | 32 (입력 `2·n_cam`) | 첫 conv, padding 제거 (crop) |
|  | MaxPool 2×2 | 32 | |
|  | 5×5 CONV + ReLU + BN | 64 | |
|  | MaxPool 2×2 | 64 | |
|  | 5×5 CONV + ReLU + BN | 128 | |
|  | MaxPool 2×2 | 128 | |
|  | 5×5 CONV + ReLU + BN | 256 | |
|  | MaxPool 2×2 | 256 | |
|  | 3×3 CONV + ReLU + BN | 512 | |
|  | MaxPool 2×2 | 512 | |
| Bottleneck | **FC** (Drop50) | 512 | **Non-canonical**: U-Net 의 bottleneck 을 spatial dim 없이 1 vector 로 collapse |
| Decoder (5단) | NN upsample → 1×1 CONV → skip concat → 3×3 CONV | 256 → 128 → 64 → 32 → 8 | |
| Final | BL upsample → T × T | 1 | bilinear 로 native tile size 까지 |

> "Each expansion step is similar, but not identical, to the corresponding steps in [35]" — 차이점:
> 1. **NN upsample → 1×1 CONV** (channel 압축) → skip concat → 3×3 CONV → ReLU
> 2. **Batch normalization 은 decoder 에서 사용 안 함** (encoder 만)
> 3. 마지막은 BL upsample (NN 이 아님)

**Padding**: tile 진입 시 ±5 pixel symmetric pad → 첫 conv 후 `crop` 으로 제거.
"empirically found to remove edge effects at the borders between tiles"

### 4.2 Global CNN (Stack A → tile features)

**얕은 leg, 적은 파라미터.** Stack A 의 32×32 image 에서 global context 추출.

- 학습 데이터의 unique sample 이 적음 (수십~수백 layer 단위) → overfit 방지 위해 small network
- 출력은 single feature vector → bilinear upsample → T × T
- **upsample 직전 FC layer** 로 채널 256 → **16** 로 줄임 ("emphasizing the more local information")
- **Dropout 두 번** 포함 (overfit 방지)

### 4.3 Localization CNN (Stack B → tile features)

**Machine-agnostic 의 핵심.** Stack B 의 arbitrary 사이즈 입력을 받아 native resolution 으로 분류 가능하게 함.

- **단일 CONV + ReLU** layer (pooling 없음, deeper layer 없음)
- 출력: per-pixel feature vector (각 pixel 위치마다 local context 인코딩)
- 임의 spatial dim 입력 가능 (FC 없음) → resolution-agnostic

### 4.4 Concat + Pixel-wise Classification (Section 3.5)

```
[U-Net out (128)] + [Global out (16)] + [Localization out (32)] + [(x,y) coord (2)] = 178 ch
       ↓
1×1 CONV (178 → n_classes), no ReLU
       ↓
Softmax (Eq. 3)
```

178 = 128 + 16 + 32 + 2. **(x, y) coordinate 를 channel 로 명시 concat** 하는 게 핵심 — pixel 의
**절대 위치 정보** 를 모델이 활용 가능 (build plate 의 좌표가 의미 있음).

**Softmax (Eq. 3)**:

```
q_j({p}, j) = exp(p_j) / Σ_{k=1..n_classes} exp(p_k)
```

여기서 `{p}` = 1×1 CONV 직후의 raw logits (arbitrary magnitude),
`q_j ∈ (0, 1]` 는 j 클래스의 pseudo-probability. Σq = 1.

### 4.5 Spatial dim 공식 (Eq. 2)

각 CONV layer 의 출력 spatial size:

```
W_o = ⌊(W_i − F + 2P) / S⌋ + 1
```

- `W_i`, `W_o`: 입력/출력 spatial size
- `F`: kernel size
- `P`: padding (TensorFlow 자동 처리, stride 1 일 때 `W_o = W_i`)
- `S`: stride

---

## 5. Loss Functions (Section 3.7)

DSCNN 은 **두 가지 loss** 를 모두 구현. 주로 cross-entropy 사용, hard-bootstrapping 은 초기 탐색용.

### 5.1 Cross-Entropy (주력)

TensorFlow 표준 cross-entropy. 한 pixel 의 one-hot target `{t}` 와 예측 `{q}` 사이:

```
E({q}, {t}) = − Σ_{k=1..n_classes}  t_k · log(q_k + ε)
```

`ε = 1×10⁻⁸` (numerical stability).

### 5.2 Hard-Bootstrapping Loss (Eq. 4, Reed et al. [53] 기반)

Noisy label 대응용. 모델이 "confident" 할 때는 자기 예측을 신뢰, 자신 없을 때는 GT 따름.

```
E = − Σ_{k=1..n_classes}  (λ · t_k + (1 − λ) · z_k) · log(q_k + ε)
```

- `λ ∈ [0, 1]`: trust coefficient (GT 신뢰도). λ=1 이면 standard CE.
- `{z}`: `{q}` 의 one-hot encoding (즉 argmax 후 one-hot)
- 직관: 모델이 confident 한 mislabel pixel 의 gradient 가 작아져서 noisy GT 영향 ↓

> "this loss function incorporates a consistency check on the ground truth labels; it gives the algorithm
> permission to fit more loosely to the training data if intra-class inconsistencies are extant."
>
> 주의: λ 가 너무 작으면 "the model may ignore the ground truths completely, effectively constructing its
> own self-reinforcing version of reality" → λ 조심.

### 5.3 Class-Wise Balancing Weight (Eq. 6)

극심한 class imbalance 해결책. pixel-wise loss `{E}` 에 **곱** 으로 적용:

```
w_k = Median({f}) / f_k
```

- `f_k`: class k 의 train data 등장 빈도
- `{f}`: 전체 class 별 빈도 set
- median 으로 정규화 → 어떤 class 는 weight > 1, 어떤 class 는 < 1

효과:
- 없으면 → "rarer anomaly classes (e.g. recoater streaking and porosity) were simply not learned"
- 있으면 → "even extremely rare anomaly classes, comprising only **0.1%** of the training dataset, are effectively learned"

### 5.4 Unlabeled Pixel 처리

```
if pixel is unlabeled: loss_at_pixel = 0
```

- 부분 라벨링 (한 tile 의 일부만 라벨) 도 허용
- 단 **최소 10% 라벨된 tile** 만 학습에 포함 (안정성)

### 5.5 Optimizer (Adam)

**TensorFlow Adam** 사용 (sparse gradient 적합).

| Hyperparam | 값 |
|---|---|
| Adam step size (lr) | **0.0001** (= 1e-4) |
| `β₁` (first moment decay) | **0.9** |
| `β₂` (second moment decay) | **0.999** |
| Adam ε | **1×10⁻⁴** |
| 갱신 식 (Eq. 5) | `Ω_{i+1} = Ω_i − η · ∇E(Ω_i)` (개념적) |

> "Overall training behavior and final algorithm performance appear relatively insensitive to the choice
> of Adam parameters" — Adam param 변경에 둔감.

### 5.6 Exponential Moving Average (EMA)

학습 종료 시 final weight 는 EMA 적용:

> "The learnable parameters saved in the final DSCNNs correspond to the **exponential moving average** values
> at the end of training with a **decay rate of 0.9999**."

→ 마지막 step weight 가 아니라 학습 후반의 평균값 저장.

---

## 6. Data Augmentation (Section 3.7)

> "CNNs are vulnerable to learning texture-responsive features even when morphological-responsive features
> would be more robust ... DSCNN was failing when altered process parameters significantly modified the
> textural differences between powder and part pixels."

### 6.1 Gaussian Noise (핵심)

Stack **A 와 C 에만** 추가 (B 에는 안 함):

```
noise ~ N(0, σ²)
σ² ∈ {0.01% · DR², 0.1% · DR²}    # DR = input data 의 dynamic range
```

- 두 가지 variance level: 0.01% / 0.1% of dynamic range
- **Stack B 는 noise 안 추가** — 이유: "to ensure preservation of some textural features (which may be
  important for classification) while forcing at least part of the DSCNN to learn low-frequency
  morphological features"

### 6.2 Mean Intensity Shift

```
shift ∈ {0, +10% · DR, −10% · DR}    # all stacks A, B, C
```

- **±10% of dynamic range** 만큼 평행 이동
- A, B, C 모든 stack 에 적용

### 6.3 Augmentation 조합 — 9× tile duplication

3 (noise: none, 0.01%, 0.1%) × 3 (shift: none, +10%, -10%) = **9 조합** (none 포함).

```
each training tile → duplicated 9 times during training
```

### 6.4 명시적 금지

> "Other canonical data augmentation techniques such as **image rotation are not applicable** for these
> datasets as they **may alter the ground truth classifications**."

→ Recoater streaking/hopping 이 방향성 라벨이라 rotation/flip 시 의미 깨짐.

---

## 7. Training Procedure (Section 3.7)

| 항목 | 값 |
|---|---|
| CONV kernel weight init | N(0, 0.05²) zero-centered normal |
| FC layer weight init | N(0, 0.004²) |
| Bias init | constant 0.1 |
| Mini-batch size | **15 ~ 50 tiles** (머신별로 다름) |
| Targets per mini-batch | `n_t = n_batch · T²` (T = tile size) |
| Epoch 수 | **~ 100** (모든 머신) |
| Shuffle | 매 epoch 마다 mini-batch 간 random shuffle |
| From-scratch 학습 시간 | **약 2일** |
| Transfer learning 시간 | **2시간 이내** ("less than two hours") |
| EMA decay | 0.9999 |

> Initialization parameter 에는 둔감하다고 명시: "performance of the DSCNN is not particularly sensitive
> to the choice of the parameters of the initialization distributions."

---

## 8. Transfer Learning (Section 3.9)

### 8.1 절차

1. Source 머신 (data 많음, e.g. ConceptLaser M2) 으로 DSCNN 학습 (random init)
2. Target 머신 (data 적음) 의 DSCNN 을 source 의 weight 로 초기화
3. **마지막 classification layer 만 random init** (anomaly class 수가 달라서)
4. **BatchNorm 통계 모두 reset** (다른 머신은 통계 분포 다름)
5. 모든 layer free to be re-learned (선택적으로 lower layer freeze 가능 — overfit 방지)

### 8.2 카메라 수 다른 경우

- Source: 1 카메라 (α)
- Target: 2 카메라 (α, β)
- Target 의 `ω_2α`, `ω_2β` 모두 source 의 `ω_1α` 로 초기화
- 첫 CONV 의 출력 채널은 kernel 수에 의해 고정 (camera 수에 invariant) → 깊은 layer 는 그대로 transfer

### 8.3 한계

- Tile resize 시 small-scale info 손실 가능
- Localization CNN 의 receptive field 가 native resolution 에서 너무 작아질 수 있음
- 1 MP → 50 MP 같은 큰 해상도 차이는 challenging

---

## 9. Heuristics (Section 3.8) — 후처리

DSCNN 의 raw prediction 에 도메인 지식으로 후처리.

**EOS M290 의 경우** (heuristic 활성 + reported testing performance 에 포함):

1. **debris 검출이 expected part location 위** → switch to part_damage
2. **part / super-elevation / swelling 검출이 part 위치에서 >1100 μm 멀리** → switch to debris
   (이유: 이 세 클래스는 정의상 fused material 없이 발생 불가)

**ConceptLaser M2 의 경우**:

3. **part 검출이 part 위치에서 >350 μm 멀리** → switch to misprint
   (등록 정확도가 높아 tighter tolerance 가능)

> CAD geometry 는 inference 시 후처리용으로만 활용. DSCNN 입력 channel 로 넣는 건 고려됐지만
> "rejected due to concerns regarding overfitting during training and a reduction in model generalizability."

---

## 10. 평가 지표 (Section 4.1)

저자는 IoU 같은 정교한 metric 보다 **직접 pixel-wise prediction accuracy** 를 선호:

> "the authors feel that simple, direct comparison of the pixel-wise predictions to the pixel-wise labels
> quantifies the DSCNN performance in the most meaningful terms."

Split:
- **Training**: Table 2 의 라벨링된 데이터
- **Validation**: training set 의 random **10% tile-wise withhold**
- **Testing**: **fully annotated** layer (학습 중 한 번도 안 본 layer). 단 layer 단위라 머신별로 수량 차이.

testing accuracy 가 validation 보다 보통 낮음:
- training set 은 인간 labeler 가 ambiguous pixel 제외하는 경향
- testing set 은 의도적으로 모든 pixel 포함 (전체 layer 라벨링) → 어려움

---

## 11. 본 프로젝트 (DefSeg-AM) 와 비교

| 항목 | DSCNN 원본 | DefSeg-AM | 차이/일치 |
|---|---|---|---|
| 클래스 수 | 머신별 가변 (8~9) | **12** (ORNL Peregrine v2023-11 출력) | DSCNN 자체는 머신마다 다른 class set; ORNL 의 production 구현은 12-class 통합 |
| Backbone | DSCNN 자체 (U-Net + 2 parallel CNN) | **DINOv2 ViT-S/14 frozen** | 완전히 다름. DSCNN 은 from-scratch 학습 가능, 우리는 pretrained ViT |
| Decoder | NN upsample + 1×1 conv + skip + 3×3 conv | **DPT-style multi-scale** | 둘 다 multi-scale 활용 |
| Input | 4 image stack (A, B, C, D) + (x,y) | **dual visible (after melt + after spread)** | 우리는 (x,y) 없음 |
| Loss (Stage 1) | Cross-entropy + class balance weight (median/freq) | **Focal Loss γ=2 + sqrt-inv α (clip 10)** | Focal 은 hard-bootstrapping 의 다른 접근 — "어렵게 학습" |
| Loss (Stage 2) | (전이학습 시 동일 CE) | **표준 CE + sqrt-inv α** | 같은 계열 |
| Optimizer | Adam lr=1e-4, β=(0.9, 0.999), ε=1e-4 | **AdamW lr=1e-4 (warmup) + Cosine** | weight decay 추가 (AdamW), warmup 추가 |
| Augmentation | Gaussian noise (σ²=0.01%, 0.1% of DR) on A, C + ±10% intensity shift on A, B, C | **brightness jitter ±15%** | ⚠️ **noise augmentation 미적용** — 추가 검토 필요 |
| Rotation/Flip 금지 | ✅ 명시 | ✅ 적용 | 일치 |
| Class balance | `w_k = median(f) / f_k`, clip 없음 | `w_k = sqrt((1-f)/f)`, **clip 10** | 우리가 더 보수적 (clip) |
| Mini-batch | 15~50 tiles | 2 layers (full image) | 우리는 cache 의 1036×1036 full-image batch |
| Epochs | ~100 | 30 (Stage 1) / 50 (Stage 2) | DSCNN 이 더 길게 |
| Param init | CONV N(0, 0.05²), FC N(0, 0.004²) | timm default (Xavier/Kaiming) | 큰 영향 없음 (DSCNN 도 둔감) |
| **EMA save** | decay 0.9999 | ❌ 미적용 | **추가 검토 가치 있음** |
| Heuristic post-process | EOS M290, CL M2 만 활용 | ❌ 미적용 | CAD 미사용 |
| Transfer learning | Cross-machine (M2 → others) | ❌ 단일 머신 (ORNL Build 1~5) | 우리 use case 다름 |

## 12. DSCNN 원본 모방해서 본 프로젝트에 추가하면 좋을 것

우선순위 순:

### A. Gaussian noise augmentation (가장 중요)

DSCNN 의 핵심 기법이고 본 프로젝트 미적용. 30분 작업.

```python
# data_ornl.py / data_dscnn.py 의 __getitem__ 에 추가
if self.training:
    sigma_pct = np.random.choice([0.0, 0.01, 0.1]) / 100.0   # DSCNN 원본 3 levels
    if sigma_pct > 0:
        noise_std = 255 * sigma_pct
        i0 = np.clip(i0.astype(np.float32) + np.random.randn(*i0.shape) * noise_std, 0, 255).astype(np.uint8)
        i1 = np.clip(i1.astype(np.float32) + np.random.randn(*i1.shape) * noise_std, 0, 255).astype(np.uint8)
```

### B. EMA weight saving

학습 후반 weight 의 평균값을 save → val_acc 안정화 가능. PyTorch 의 `torch.optim.swa_utils.AveragedModel` 사용:

```python
ema_model = torch.optim.swa_utils.AveragedModel(
    model, multi_avg_fn=torch.optim.swa_utils.get_ema_multi_avg_fn(0.9999)
)
# 각 step 마다
ema_model.update_parameters(model)
# 학습 끝나면 ema_model 의 weight 저장
```

### C. (x, y) coordinate channel

DSCNN 의 image stack D 와 동등. ORNL build plate 의 절대 좌표가 의미 있을 수 있음
(코너에서 spatter 가 다르게 보이는 등). DPT decoder 의 head 직전에 `(x, y) grid` 를 concat.

### D. Hard-bootstrapping loss 시도

DSCNN pred 가 noisy label 이라는 점에서 (Stage 1 의 핵심 가정) hard-bootstrapping 이 효과 클 수 있음.
λ ≈ 0.8 권장. 다만 Focal Loss 와 동시 적용 시 수치적으로 복잡 — Focal 만 유지하거나 Focal 끄고 시도.

### E. Class balance weight 공식을 DSCNN 식으로 교체

우리는 `sqrt((1-f)/f)` 사용 중. DSCNN 은 `median(f)/f_k`. 더 극단적 (rare class 에 더 큰 weight). clip
없이 적용 시 폭주 위험 → 우리 sqrt+clip 도 합리적이지만, DSCNN 방식도 ablation 가치 있음.

---

## 13. 결론적 메모

- DSCNN 은 **"machine-agnostic"** 을 명시 목표로 디자인. 우리 DefSeg-AM 은 ORNL 단일 머신 한정 — 따라서
  DSCNN 의 transfer-learning 메커니즘은 모방 불필요
- DSCNN 의 **multi-scale (global + regional + local) parallel leg** 아이디어는 우리 DPT decoder 의
  multi-scale fusion 으로 충분히 대체됨
- **augmentation (특히 noise)** 만큼은 명백히 적용 가치 있음 → 다음 학습 사이클에서 추가 권장
- DSCNN 의 **(x, y) coordinate channel** 은 흥미로운 차별점 — ORNL data 의 build plate 위치 정보가
  결함 학습에 도움될 수 있음. ablation 단위로 시도 가치

---

> 작성: 2026-06-13. 원본 PDF 텍스트 직접 추출 후 정리.
> 각 인용은 원본의 Section/Equation 번호와 함께 표기.
