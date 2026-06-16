# DefSeg-AM **v3** 실험 계획

> v2 ([PLAN_v2.md](../v2/PLAN_v2.md)) 의 후속. v2 가 8-class + D4 augmentation +
> 8-fold cross-validation 이었다면, v3 는 **10-class (Recoater 분리 복원 +
> Incomplete Spreading 복원)** + **on-image 도메인 augmentation** + **DSCNN_Dataset
> GT 전수 학습 (단일 ckpt) + ORNL Build 1 로 평가** + **SE/Swelling → Debris
> 휴리스틱 후처리** 가 핵심.

---

## 1. v2 → v3 변경 요약

| 영역 | v2 | v3 |
|---|---|---|
| 클래스 수 | 8 (Recoater Hopping/Streaking 통합, 3개 제거) | **10** (Recoater 분리 복원, Incomplete Spreading 복원) |
| Augmentation | D4 group (4 rot × 2 flip) + cyclic shift | **flip(LR+UD) + 180° rot + cyclic shift** + DSCNN noise/intensity/brightness |
| Stage 2 CV | leave-one-source-out 8-fold | **CV 없음** — DSCNN_Dataset 8 source 전부 train, 평가는 ORNL Build 1 로 |
| Stage 2 ckpt | fold 별 8 개 | **단일** `stage2_best.pt` |
| Stage 2 augmentation | 기본 1회 | **DSCNN_Dataset 은 작은 shift step 으로 더 많이 증강** (epoch 당 effective sample ↑) |
| 휴리스틱 후처리 | (없음) → 사용자 요청으로 v2 에 추가됐던 SE/Swelling far → Debris | 단일 규칙 — Printed 에서 멀면 Debris(6) |

---

## 2. 클래스 정의 (10개)

### 2.1 클래스 목록

| ID | Name | 정의 |
|---:|------|------|
| 0  | Powder | 이상 또는 프린트된 파트가 없는 분말 베드 영역 |
| 1  | Printed | 이상이 감지되지 않은 프린트 영역 |
| 2  | Recoater Hopping | 리코터가 표면 아래 파트에 충돌할 때 발생하는 물결무늬 |
| 3  | Recoater Streaking | 리코터 손상 또는 큰 입자 끌림으로 인한 줄무늬 |
| 4  | Incomplete Spreading | 분말 베드에 불충분한 분말 도포 |
| 5  | Swelling | 분말 위로 돌출된 프린트 재료의 변형/뒤틀림 |
| 6  | Debris | 분말 베드의 소-중형 교란 (포괄적 클래스) |
| 7  | Super-Elevation | 프린트 영역 위의 분말 커버리지 부족 |
| 8  | Spatter | 용접 풀에서 튀어나와 분말 베드에 착지한 비산물 |
| 9  | Over Melting | 고에너지 밀도 공정 파라미터로 용융된 영역 |

**제거된 클래스** (vs v1 ORNL 12-class): `Misprint(9)`, `Under Melting(11)`
— v2 와 동일 사유 (라벨 매우 희소 + intra-class noise 큼).

### 2.2 v1 ORNL 12-class → v3 10-class 매핑 (`ORNL_12_TO_NEW_10`)

```
0  Powder              → 0   Powder
1  Printed             → 1   Printed
2  Recoater Hopping    → 2   Recoater Hopping       (분리 유지)
3  Recoater Streaking  → 3   Recoater Streaking
4  Incomplete Spreading→ 4   Incomplete Spreading   (v2 에서 제거 → v3 복원)
5  Swelling            → 5   Swelling
6  Debris              → 6   Debris
7  Super-Elevation     → 7   Super-Elevation
8  Spatter             → 8   Spatter
9  Misprint            → -1  IGNORE
10 Over Melting        → 9   Over Melting
11 Under Melting       → -1  IGNORE
```

### 2.3 DSCNN_Dataset native → v3 매핑

원칙은 v2 와 동일. `MATERIAL_TO_ORNL_V3` 는 v2 의 매핑을 그대로 import 한 뒤
v3 10-class 인덱스 (`ORNL_12_TO_NEW_10`) 를 한 번 더 적용. 라벨 없는 픽셀은
그대로 `-1` IGNORE.

---

## 3. Data Augmentation

### 3.1 DSCNN 원본 유지

- **Gaussian Noise**: σ ∈ {0, 0.01%, 0.1%} of DR (8-bit 기준 0, 0.026, 0.255)
- **Mean Intensity Shift**: ±10% of DR 또는 0 (3-way choice)
- **Brightness multiplicative jitter**: ×U(0.85, 1.15) — v1/v2 의 추가 augmentation 유지

### 3.2 기하 변환 (v2 D4 → v3 단순화)

| 변환 | v2 | v3 | 비고 |
|---|---|---|---|
| 좌우 flip (LR) | ✓ | **✓** | 분말 베드 좌우 대칭 |
| 상하 flip (UD) | ✓ | **✓** | 사용자 결정 — LR/UD 모두 적용 |
| 0° / 180° rotation | ✓ | **✓** | 분말 베드 180° 대칭 |
| 90° / 270° rotation | ✓ | **✗** | 사용자 결정 — recoater 진행축 비대칭이라 제외 |
| Cyclic shift (np.roll) | ±IMG/4 | **±IMG/4** | 잘려나간 영역은 반대편에서 채움 — 잔류 결함이 patch boundary 에 가도 학습 신호 보존 |

**v3 적용 순서** (training=True 시 1 sample 당):
1. `np.roll` shift ((dy, dx) ∈ ±IMG/4, 50% 확률)
2. flip LR (50% 확률), flip UD (50% 확률) — 독립
3. 180° rotation (50% 확률)
4. DSCNN noise σ choice
5. DSCNN intensity shift choice
6. brightness jitter

label 은 1-3 만 동일하게 적용 (4-6 은 이미지 강도만 변경).

### 3.3 Stage 2 (DSCNN_Dataset) 추가 증강 — "중복되지 않는 선에서 더 많이"

DSCNN_Dataset 은 8 source 합쳐도 sample 수가 적음 (수백 수준). 같은 augmentation
파이프라인이지만 **effective dataset 크기를 epoch 당 늘리는** 방식:

- **micro-shift dense sampling**: cyclic shift `(dy, dx) ∈ [-IMG/4, +IMG/4]` 의
  **1-px 균등** (uniform integer) sampling → effective unique shift 가
  `(IMG/2)² ≈ 270k` 가지. gaussian 가중 없이 단순 uniform.
- **DataLoader replicate factor `K = 4`**: `Stage2Dataset` 의 `__len__` 을
  `K × N` 으로 늘려 한 epoch 에 같은 GT sample 을 K 번 다른 augmentation 으로
  본다. epoch 당 step 수 4 배 → EMA(0.9999) 가 의미 있는 단위로 update.
- **batch_size 유지** — gradient 안정성. K 배만큼 epoch 당 step 수 증가.

### 3.4 Inference 시

모든 augmentation OFF. (옵션) Test-Time Augmentation 은 v2 의 D4 8-way 가 아니라
**flip(LR) × flip(UD) × 180° rot = 8-way** 변환 셋의 일관 평균 (v3 학습 augmentation
과 일치하는 변환만).

---

## 4. 학습 단계

### 4.1 Stage 1 — KD pretrain (변경 최소)

- 데이터: ORNL HDF5 5 builds (Build 2-5 train, Build 1 val) — v1/v2 와 동일
- 라벨: v3 10-class
- 손실: focal loss (γ=2) + sqrt-inv class weight + (옵션) hard-bootstrap λ=0.8
- EMA decay 0.9999 (v2 와 동일)
- epoch 30, batch 2, lr 1e-4, warmup 200, AdamW
- val 데이터 유지 — Build 1 로 best ckpt 선정

### 4.2 Stage 2 — DSCNN_Dataset GT 전수 학습 (단일 ckpt, ORNL Build 1 평가)

- 데이터: DSCNN_TRAIN_SOURCES_V3 = v2 와 동일한 8 source (LPBF 6 + BJ 2, EBPBF
  제외) — **8 source 전부 train**, DSCNN_Dataset 안에서 val 분리하지 않음
- 라벨 매핑: §2.3 의 2-step 매핑
- Stage 1 best ckpt 에서 init
- augmentation: §3.3 (1-px uniform shift, K=4 replicate factor) — v2 대비 강화
- epoch 50, batch 2, lr 1e-4, warmup 50, EMA on
- random seed = 42 고정
- **평가용 val = ORNL HDF5 Build 1** (Stage 1 의 val 과 동일). DSCNN_Dataset 의
  GT 는 한 픽셀도 학습에서 빠지지 않음 → 사용자 의도 만족. 평가 데이터는 학습과
  다른 분포(ORNL native, multi-class argmax) 이므로 cross-domain generalization
  측정이 됨
- Best ckpt 선정: ORNL Build 1 의 mIoU 가 가장 높은 epoch 의 EMA weight
  (`stage2_best.pt`)

### 4.3 손실 — class weight

`sqrt_inv_class_weight` 가 v3 10-class 의 빈도 분포에 따라 자동 조정. rare class
가 너무 dominant 한 weight 를 받지 않도록 `S1_CLASS_WEIGHT_CLIP=10` 로 cap.

---

## 5. 휴리스틱 후처리 (사용자 요청)

학습/Loss 와는 무관 — Inference (시각화·평가) 직전에만 적용.
구현: `v3/inference/postprocess.py`. 상수: `v3/config_v3.py` 의 `PP_*` 그룹.
모든 임계치는 **실 결함이 사라지지 않도록 보수적** default.

### 5.1 부품에서 먼 SE/Swelling → Debris

- 대상: Super-Elevation (7), Swelling (5) 의 connected component
- 면적 ≥ `PP_SE_SWELLING_MIN_COMPONENT_PX` (= 30)
- Printed (1) mask 의 nearest pixel 까지 **최소** 거리 ≥
  `PP_SE_SWELLING_FAR_DISTANCE_PX` (= 100 px)
- 위 조건 모두 만족하면 Debris (6) 으로 재분류

> 거리 계산은 `scipy.ndimage.distance_transform_edt` 를 Printed mask 의 complement
> 에 적용해 O(HW) 로 처리. component 내 **최소** 거리 사용 → 부품에 조금이라도
> 닿으면 변경 안 함 (보수적).

CLI: `python -m DefSeg_AM.v3.inference.infer ... --postprocess`.
출력 디렉터리에 `_pp` suffix.

---

## 6. 폴더 구조

```
DefSeg_AM/v3/
├── PLAN_v3.md                ← 이 파일
├── README.md                 (v3 사용법)
├── __init__.py
├── config_v3.py              # 10-class, augmentation 상수, PP_*
├── data/
│   ├── __init__.py
│   ├── augmentation.py       # flip LR/UD + 180° rot + cyclic + DSCNN
│   ├── data_ornl_v3.py       # Stage 1 dataset
│   ├── data_dscnn_v3.py      # Stage 2 dataset (CV 없음, replicate factor)
│   ├── build_cache_v3.py     # label_v3.npy
│   └── samplers.py           # (공통 import)
├── models/
│   ├── __init__.py
│   └── losses_v3.py          # focal/bootstrap import + 10-class weight helper
├── training/
│   ├── train_stage1.py       # ORNL HDF5 (변경 최소)
│   └── train_stage2.py       # DSCNN_Dataset 전수 학습 (val 없음)
├── inference/
│   ├── infer.py              # --postprocess, --tta
│   ├── postprocess.py        # SE/Swelling 이 부품에서 멀면 Debris 로 재분류
│   └── confusion.py
├── scripts/
│   ├── run_build_cache_v3.sh
│   ├── run_stage1_v3.sh
│   ├── run_stage2_v3.sh
│   └── run_all_v3.sh
└── docker/
    ├── Dockerfile
    ├── docker-compose.yml
    └── README.md
```

---

## 7. 산출물 / 캐시

- **v3 label cache**: `DefSeg_AM/cache/resized_sz1036_v3/<build>/`
  - `label_v3.npy` (n_layers, IMG, IMG) int8, 값 ∈ {-1, 0..9}
  - `meta.npz` (orig_layer_idxs, defect_ratios_v3)
- **image cache 는 v1 그대로 재사용** (`resized_sz1036/<build>/visible_{0,1}.npy`)
- **ckpt**: `DefSeg_AM/checkpoints/vits14_dpt_dual_sz1036_10cls_v3/`
  - `stage1_best.pt`
  - `stage2_best.pt` (ORNL Build 1 mIoU 기준 best 의 EMA weight)
- **figure**: `DefSeg_AM/figures/vits14_dpt_dual_sz1036_10cls_v3/v3/`
  - `stage1/inference/`, `stage1_pp/inference/`
  - `stage2/inference/`, `stage2_pp/inference/`

---

## 8. 평가

### 8.1 정량

- Stage 1: ORNL Build 1 val 의 confusion + per-class IoU + mIoU
- Stage 2: ORNL Build 1 의 confusion + per-class IoU + mIoU — Stage 1 best vs
  Stage 2 best 비교 (DSCNN_Dataset finetune 의 효과 측정)
- ORNL Build 1 평가는 분포 다른 데이터셋이므로 **cross-domain generalization**
  지표가 됨

### 8.2 정성

- Build 1 의 대표 layer 4-panel figure (visible/0, visible/1, GT, pred)
- `--postprocess` 적용 전·후 비교 — 빌드플레이트 false positive 가 IGNORE/Powder
  로 변하는지, 먼 SE/Swelling 이 Debris 가 되는지

### 8.3 v1/v2 와의 비교

`RESULTS_v1_vs_v2_vs_v3.md` 에 mIoU / per-class IoU / 정성 한 페이지 비교.

---

## 9. 확정된 Stage 2 hyperparameter 요약

| 항목 | 값 | 출처 |
|---|---|---|
| 데이터 분리 | **DSCNN_Dataset 8 source 전부 train**, ORNL Build 1 로 평가 | §4.2 |
| Replicate factor K | **4** | §3.3 |
| Cyclic shift 분포 | **1-px uniform** ([-IMG/4, +IMG/4]) | §3.2 / §3.3 |
| Random seed | **42** (고정) | §4.2 |
| Epoch | 50 | §4.2 |
| Batch size | 2 | §4.2 |
| Learning rate | 1e-4 | §4.2 |
| Warmup steps | 50 | §4.2 |
| EMA decay | 0.9999 | v2 와 동일 |
| Best ckpt 선정 기준 | ORNL Build 1 mIoU 최고 epoch 의 EMA weight | §4.2 |
