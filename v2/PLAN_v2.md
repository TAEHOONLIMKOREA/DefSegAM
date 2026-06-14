# DefSeg-AM v2 실험 계획

> **버전**: v2 (2026-06-14 작성)
> **기준**: 기존 [PLAN.md](PLAN.md) (v1) 의 모델 구조 유지, **클래스 / 데이터 / 학습 정책** 4 가지 변경.
> **참고 문헌**: [paper/DSCNN/DSCNN_Summary.md](paper/DSCNN/DSCNN_Summary.md) 의 모든 학습 디테일 모방.

---

## 1. 변경 요약 (v1 → v2)

| 항목 | v1 (기존) | v2 (이번) |
|---|---|---|
| **클래스 수** | 12 (ORNL Peregrine 표준) | **8** (rare/ambiguous class 제거 + recoater 통합) |
| **Stage 1 데이터** | ORNL HDF5 `slices/segmentation_results` (DSCNN model pred, KD pretrain) | 동일 (ORNL HDF5 KD pretrain), 단 8-class 로 재매핑 |
| **Stage 2 데이터** | DSCNN_Dataset 의 LPBF 5 source (EBPBF 제외) | **EBPBF 제외 모든 source (LPBF + Binder Jet 포함, 7 source)** |
| **Stage 2 평가** | v2022_Maraging 만 val (고정) | **8-fold cross-validation** (모든 source 가 한 번씩 val) |
| **Augmentation** | brightness jitter ±15% | **DSCNN 원본 (Gaussian noise + intensity shift) + D4 rotation/flip + Cyclic shift** |
| **Loss** | Focal (S1) + CE (S2) + sqrt-inv α | + **DSCNN 원본의 hard-bootstrapping loss (옵션)** + **class balance weight 공식 변경 (median/freq)** |
| **EMA weight** | 미적용 | **EMA decay 0.9999** (DSCNN 표준) |
| **모델 구조** | DINOv2 ViT-S/14 frozen + DPT decoder | **동일 유지** |

---

## 2. 클래스 정의 (8개)

### 2.1 새 클래스 목록

사용자 지정 순서 그대로:

| ID | 이름 | 의미 |
|---|---|---|
| 0 | Powder | 분말 베드 (정상 미융합 영역) |
| 1 | Printed | 정상 인쇄 영역 |
| **2** | **Recoater Disturbance** | recoater 시스템 관련 모든 표면 줄무늬 (Hopping + Streaking 통합) |
| 3 | Swelling | 부품의 분말 위 돌출 / 뒤틀림 |
| 4 | Spatter | 용융 풀에서 튄 비산물 |
| 5 | Super-Elevation | 부품 영역 위의 분말 커버리지 부족 |
| 6 | Over Melting | 과용융 (keyhole porosity 동반) |
| 7 | Debris | 분말 베드의 소-중형 교란 |

### 2.2 v1 (12) → v2 (8) 매핑

```python
# config.py 의 새 매핑 (적용 위치: ornl_segmentation_argmax, remap_label)
ORNL_12_TO_NEW_8 = {
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
N_CLASSES_V2 = 8
IGNORE_INDEX = -1
DEFECT_CLASS_INDICES_V2 = [2, 3, 4, 5, 6, 7]  # Powder/Printed 제외
```

### 2.3 v2 의 정당성

- **제거된 3 class**:
  - **Incomplete Spreading**: v1 Stage 1 의 IoU = 0.0016 (학습 거의 안 됨, DSCNN teacher 라벨 자체가 noisy)
  - **Misprint**: v1 IoU = 0.0945 (rare 하고 part_damage 와 시각 유사)
  - **Under Melting**: v1 IoU = 0.0000 (GammaPrint-700 의 OT 라벨로만 정의되어 ORNL 에선 거의 없음)
- **합쳐진 2 class**:
  - Recoater Hopping (수직 줄) + Streaking (평행 줄) → **방향 무관 "Recoater Disturbance"**
  - 결정적 효과: **합치면 90° rotation 으로도 라벨 의미 보존** → D4 augmentation 가능 (§4.2)

---

## 3. 데이터 — Cross-Validation 설계

### 3.1 사용 가능한 source (EBPBF 제외)

DSCNN_Dataset (ORNL Peregrine 출력) 의 **LPBF + Binder Jet** 만 사용. EBPBF (Electron Beam) 는 명시적 제외.

| ID | source 폴더 | 머신 | 재료 | native classes | 추정 layer 수 |
|---|---|---|---|---|---|
| **S1** | v2021-03/Laser Powder Bed Fusion/ | ConceptLaser M2 | SS 316L | 9 (incl. Soot) | ~20 |
| **S2** | v2022-10.1/Laser_Powder_Bed_Fusion/EOS_M290/17-4_PH_Stainless_Steel/training | EOS M290 | 17-4 PH SS | 12 | 14 |
| **S3** | v2022-10.1/.../EOS_M290/GammaPrint-700/training | EOS M290 | GammaPrint-700 | 15 (incl. OT-only) | 9 |
| **S4** | v2022-10.1/.../EOS_M290/Inconel_718_1/training | EOS M290 | Inconel 718 (TI-NIR+VL) | 15 | 5 |
| **S5** | v2022-10.1/.../EOS_M290/Inconel_718_2/training | EOS M290 | Inconel 718 (VL only) | 12 | 24 |
| **S6** | v2022-10.1/.../AddUp_FormUp_350/Maraging_Steel/training | AddUp FormUp 350 | Maraging Steel | 10 | 26 |
| **S7** | v2021-03/Binder Jet/ | ExOne Innovent | SiC/B4C/H13 mix | 6 (incl. no-example) | ~20 |
| **S8** | v2022-10.1/Binder_Jet/ExOne_M-Flex/H13_Steel/training | ExOne M-Flex | H13 Tool Steel | 8 | 7 |

총 **8 source, ~125 layer**.

### 3.2 Cross-Validation 정책

**8-fold leave-one-source-out (LOSO)**:

```
fold 1: train = {S2,S3,S4,S5,S6,S7,S8}, val = {S1}
fold 2: train = {S1,S3,S4,S5,S6,S7,S8}, val = {S2}
...
fold 8: train = {S1,S2,S3,S4,S5,S6,S7}, val = {S8}
```

각 fold 마다:
1. Stage 1 (ORNL KD pretrain) — fold 무관하게 1 회만 (val build 만 B1 고정)
2. Stage 2 (DSCNN_Dataset finetune) — **fold 별로 Stage 1 ckpt 에서 시작**

→ Stage 1 = 1 회, Stage 2 = 8 회. 최종 8 개 `stage2_best_fold{k}.pt` 생성.

### 3.3 평가 방법

- 각 fold 의 val source 에서 per-class IoU + pixel acc 측정
- **8 fold 평균** = 최종 cross-validation 성능
- per-source breakdown 도 보고 (어느 머신/재료 domain 이 어려운지)

### 3.4 v1 (ORNL HDF5) 의 역할 — Stage 1 의 KD pretrain

Stage 1 은 v1 과 동일하게 **ORNL HDF5 의 DSCNN model pred** 를 teacher 로 사용. 단 **8-class 로 매핑**:

```python
# data_ornl.py:ornl_segmentation_argmax 수정
def ornl_segmentation_argmax_v2(seg_grp, layer_idx):
    out = np.full((H, W), IGNORE_INDEX, dtype=np.int8)
    for c_old in range(12):  # ORNL HDF5 는 12 채널
        mask = seg_grp[str(c_old)][layer_idx]
        new_id = ORNL_12_TO_NEW_8[c_old]
        if new_id >= 0:
            out[mask] = new_id    # IGNORE 인 class (=Incomplete/Misprint/UnderMelt) 는 적용 안 함
    return out
```

→ Stage 1 train = ORNL Build 2,3,4,5, val = ORNL Build 1 (기존과 동일). 단 출력 8-class.

---

## 4. Data Augmentation — 종합 정책

### 4.1 DSCNN 원본의 augmentation (정확히 모방)

#### Gaussian Noise (DSCNN_Summary §6.1)

```python
# 두 visible 채널 모두 적용 (DSCNN 의 stack A, C 대응)
sigma_pct = np.random.choice([0.0, 0.01, 0.1]) / 100.0
if sigma_pct > 0:
    DR = 255  # 8-bit dynamic range
    noise_std = DR * sigma_pct
    i0 = np.clip(i0.astype(np.float32) + np.random.randn(*i0.shape) * noise_std, 0, 255).astype(np.uint8)
    i1 = np.clip(i1.astype(np.float32) + np.random.randn(*i1.shape) * noise_std, 0, 255).astype(np.uint8)
```

3 levels: none / 0.01% / 0.1% of dynamic range.

#### Mean Intensity Shift (DSCNN_Summary §6.2)

```python
shift_pct = np.random.choice([0.0, +0.10, -0.10])
if shift_pct != 0:
    shift = 255 * shift_pct
    i0 = np.clip(i0.astype(np.float32) + shift, 0, 255).astype(np.uint8)
    i1 = np.clip(i1.astype(np.float32) + shift, 0, 255).astype(np.uint8)
```

3 levels: none / +10% / -10%.

### 4.2 D4 Group Rotation + Flip (새로 추가 — Recoater 통합 덕에 가능)

```python
# 4 rotations
k = np.random.randint(4)   # 0, 90, 180, 270 degrees
if k > 0:
    i0 = np.rot90(i0, k=k).copy()
    i1 = np.rot90(i1, k=k).copy()
    ann = np.rot90(ann, k=k).copy()

# Horizontal flip (50% prob)
if np.random.random() < 0.5:
    i0 = i0[:, ::-1].copy()
    i1 = i1[:, ::-1].copy()
    ann = ann[:, ::-1].copy()
```

→ D4 dihedral group (8 unique transforms).

### 4.3 Cyclic Shift (사용자 신규 제안)

이미지/라벨을 random pixel offset 만큼 평행 이동, **사라진 영역은 반대쪽에서 채움** (torus topology):

```python
# x, y 축 각각 random shift
dx = np.random.randint(-img_size // 4, img_size // 4 + 1)
dy = np.random.randint(-img_size // 4, img_size // 4 + 1)
if dx != 0 or dy != 0:
    i0 = np.roll(i0, shift=(dy, dx), axis=(0, 1))
    i1 = np.roll(i1, shift=(dy, dx), axis=(0, 1))
    ann = np.roll(ann, shift=(dy, dx), axis=(0, 1))
```

**주의점**:
- shift 범위는 `±img_size/4` (즉 ±259 pixels for 1036) 정도. 너무 크면 unrealistic
- 라벨도 함께 roll (이게 핵심 — `__getitem__` 안에서 image+label 동시 처리)
- 일부 build plate 경계 영역이 반대쪽으로 가는 게 약간 unrealistic 이지만, augmentation 으로는 acceptable
- 결함 위치가 plate 위에서 어디든 발생할 수 있음을 학습 → **positional invariance** 향상

### 4.4 기존 brightness jitter (유지)

```python
if np.random.random() < 0.5:
    scale = np.random.uniform(0.85, 1.15)  # ±15% multiplicative
    i0 = np.clip(i0.astype(np.float32) * scale, 0, 255).astype(np.uint8)
    i1 = np.clip(i1.astype(np.float32) * scale, 0, 255).astype(np.uint8)
```

이건 DSCNN 의 intensity shift (±10% additive) 와 약간 다른 효과 — 곱셈은 명암비도 바꿈. 둘 다 유지 (조합 효과).

### 4.5 결합 — 학습 1 sample 당 적용 순서

```
load (i0, i1, ann)
  ↓
D4 rotation (4 옵션)
  ↓
Horizontal flip (2 옵션)
  ↓
Cyclic shift (random dx, dy)
  ↓
Gaussian noise (3 levels)  — image only
  ↓
Mean intensity shift (3 levels)  — image only
  ↓
Brightness jitter (multiplicative, ±15%)  — image only
  ↓
Normalize (ImageNet mean/std)
```

**총 가능 조합**: 4 × 2 × (shift continuous) × 3 × 3 × (jitter continuous) = sample 마다 다른 augment.
유효 dataset 크기: 원본 × 36+ (continuous part 제외) = 36× 가량.

### 4.6 Inference 시

- Augmentation 없음 (deterministic)
- 단, **Test-Time Augmentation (TTA)** 옵션: D4 group 8 변형 추론 후 inverse-rotation 으로 정렬한 logits 평균 → 더 robust 예측 (선택)

---

## 5. 학습 단계

### 5.1 Stage 1 — KD pretrain (변경 최소)

| 항목 | 값 |
|---|---|
| Train 데이터 | ORNL HDF5 Build 2,3,4,5 (변경 없음) |
| Val 데이터 | ORNL HDF5 Build 1 (변경 없음) |
| Teacher | ORNL `slices/segmentation_results` (DSCNN model pred), **8-class 재매핑** |
| 출력 dim | **8** |
| Loss | Focal Loss γ=2.0 + α (sqrt-inv weight, clip=10) |
| Optimizer | AdamW lr=1e-4, weight_decay=1e-4 |
| Scheduler | Warmup 200 step + Cosine annealing |
| Grad clip | max_norm=1.0 |
| AMP | OFF (FP32, Blackwell 호환) |
| **EMA** (신규) | decay=0.9999, ckpt 저장 시 EMA weight 사용 |
| Epochs | 30 |
| Sampler | WeightedRandomSampler (defect_ratio^0.5 oversampling, defect 인덱스는 v2 의 [2..7]) |
| Cache | `cache/resized_sz1036/` 재사용 (label 만 v2 재매핑 — 캐시 재빌드 필요) |
| Augmentation | §4 의 전체 정책 |

→ 1 회만 실행. 결과: `checkpoints/<run_name>/stage1_best.pt` (8-class output).

### 5.2 Stage 2 — Cross-validation finetune (대폭 변경)

#### 5.2.1 데이터

7 train source + 1 val source (fold 별로 바뀜). 각 source 의 모든 layer 사용 (train/val 분리는 source level).

#### 5.2.2 Native class → 8-class 매핑

각 source 의 native class 를 v2 의 8-class 로 매핑. 자료는 DSCNN_Dataset 의 v2021/v2022 readme PDF 와 기존 [seung_dscnn/config.py:41-99](../seung_dscnn/config.py#L41-L99) 의 `MATERIAL_TO_ORNL` 를 참고하여 **2 단계 매핑** (native → ORNL 12 → v2 8):

```python
# config.py 의 새 함수 (data_dscnn.py 에서 사용)
def remap_label_v2(ann_native: np.ndarray, mapping_key: str) -> np.ndarray:
    out = np.full_like(ann_native, fill_value=IGNORE_INDEX, dtype=np.int8)
    # 1단계: native → ORNL 12-class (기존 MATERIAL_TO_ORNL 그대로)
    for native_id, ornl_12_id in MATERIAL_TO_ORNL[mapping_key].items():
        if ornl_12_id < 0:
            continue
        # 2단계: ORNL 12 → v2 8
        new_id = ORNL_12_TO_NEW_8[ornl_12_id]
        if new_id < 0:
            continue
        out[ann_native == native_id] = new_id
    return out
```

**신규 BJ source 의 mapping** (v2021 BJ, v2022 BJ ExOne M-Flex) 추가:

```python
MATERIAL_TO_ORNL.update({
    # v2021 Binder Jet (ExOne Innovent variants, readme 의 BJ section 기준)
    "v2021_BJ": {
        0: 0,   # Un-Fused Powder → Powder
        1: 1,   # Printed Material
        2: 3,   # Recoater Streaking → ORNL_3 (= v2 Recoater Disturbance after stage2 mapping)
        3: -1,  # Powder Short Feed → IGNORE (= Incomplete Spreading, v2 에서 제거)
        4: -1,  # (no examples)
        5: 6,   # Debris → ORNL_6 (= v2 Debris)
    },
    # v2022 Binder Jet (ExOne M-Flex H13)
    "v2022_BJ_H13": {
        0: 0, 1: 1,
        2: 3,    # Roller Streaking → Recoater Streaking → v2 Recoater Disturbance
        3: 6,    # Debris
        4: -1,   # Short Feed → IGNORE
        5: -1,   # Cornrows → IGNORE
        6: -1,   # Exposed Part (super-elevation 비슷하나 불확실) → IGNORE
        7: -1,   # Misprint → IGNORE (v2 제거)
    },
})
```

(BJ 매핑은 readme PDF 의 native class 정의 정확 확인 후 fine-tune. v2 작성 시점은 우선 보수적으로 ambiguous 한 건 IGNORE.)

#### 5.2.3 학습 절차

```
for fold_k in range(8):
    train_sources = [S for S in SOURCES_8 if S != fold_k]
    val_sources = [SOURCES_8[fold_k]]
    
    # Stage 1 best ckpt 에서 init
    model.load_trainable_state_dict(stage1_best.pt)
    
    # 학습
    train(model, train_sources, val_sources, epochs=50, lr=1e-4, ...)
    
    # save
    torch.save(model_ema, f"stage2_best_fold{fold_k}.pt")
```

| 항목 | 값 |
|---|---|
| Train sources | 7 (한 source 빠짐) |
| Val source | 1 (rotating) |
| Loss | **표준 CE** + α (sqrt-inv weight, clip=10, fold 별 재계산) |
| Optimizer | AdamW lr=1e-4, weight_decay=1e-4 |
| Scheduler | Warmup 50 step + Cosine |
| Grad clip | max_norm=1.0 |
| EMA | decay=0.9999 |
| Epochs | 50 |
| Augmentation | §4 의 전체 정책 |
| Init | `stage1_best.pt` 의 `trainable_state_dict` |
| Output dim | 8 |

#### 5.2.4 결과 ckpt 구조

```
checkpoints/vits14_dpt_dual_sz1036_8cls_v2/
├── stage1_best.pt                  # 1회 (KD pretrain)
├── stage2_best_fold0_S1.pt         # fold 0: val=S1 (v2021_LPBF)
├── stage2_best_fold1_S2.pt         # fold 1: val=S2 (v2022_17-4PH)
├── stage2_best_fold2_S3.pt         # ...
├── stage2_best_fold3_S4.pt
├── stage2_best_fold4_S5.pt
├── stage2_best_fold5_S6.pt
├── stage2_best_fold6_S7.pt         # val=v2021_BJ
├── stage2_best_fold7_S8.pt         # val=v2022_BJ_H13
└── cv_summary.json                 # 8 fold 의 per-class IoU 집계
```

---

## 6. DSCNN 원본 디테일 추가 적용 (DSCNN_Summary §12 의 5개 권장)

| 추가 | 상태 |
|---|---|
| **A. Gaussian noise augmentation** | ✅ §4.1 |
| **B. EMA weight saving (0.9999)** | ✅ §5.1, §5.2 |
| C. (x, y) coordinate channel | ⚠️ **선택**. ablation 으로 진행 가능. 현재 v2 default 에서는 미적용 (DPT decoder 의 spatial info 가 이미 conv 로 학습됨) |
| **D. Hard-bootstrapping loss** | ⚠️ **Stage 1 의 옵션** (`--use-hard-bootstrap` flag). λ=0.8 권장. teacher pred 가 noisy 라는 점에서 valuable. Stage 2 는 hand-labeled GT 라 적용 안 함. |
| **E. Class balance weight: median/freq** | ⚠️ **ablation 옵션**. 기본은 sqrt-inv (안정). DSCNN 원본 식 (`w_k = median(f) / f_k`) 도 옵션으로 추가. |

### 6.1 EMA 구현 (PyTorch 2.9+)

```python
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn

ema_model = AveragedModel(model, multi_avg_fn=get_ema_multi_avg_fn(0.9999))

# training loop
for step in ...:
    loss.backward()
    optim.step()
    ema_model.update_parameters(model)

# 학습 끝나면 EMA 의 weight 를 best ckpt 로 save
torch.save({
    "model_state": unwrap(ema_model).module.trainable_state_dict(),
    ...
}, ckpt_path)
```

### 6.2 Hard-bootstrapping Loss 구현 (Stage 1 옵션)

```python
def hard_bootstrap_loss(logits, target, lambda_trust=0.8, ignore_index=-1):
    """Eq.4 of DSCNN paper.
    E = -Σ (λ·t_k + (1-λ)·z_k) · log(q_k + ε)
    where z_k = onehot(argmax(q)).
    """
    B, C, H, W = logits.shape
    valid = (target != ignore_index)
    log_q = F.log_softmax(logits, dim=1)             # (B, C, H, W)
    q = log_q.exp()
    z = F.one_hot(q.argmax(dim=1), num_classes=C).permute(0, 3, 1, 2).float()
    t = F.one_hot(target.clamp(min=0), num_classes=C).permute(0, 3, 1, 2).float()
    mix = lambda_trust * t + (1 - lambda_trust) * z   # (B, C, H, W)
    nll = -(mix * log_q).sum(dim=1)                   # (B, H, W)
    return nll[valid].mean()
```

---

## 7. 코드 영향 — 수정 파일 목록

| 파일 | 변경 |
|---|---|
| **config.py** | `ORNL_CLASS_NAMES` (8), `N_CLASSES=8`, `ORNL_12_TO_NEW_8`, `DEFECT_CLASS_INDICES` (=[2..7]), `MATERIAL_TO_ORNL` 에 BJ 2개 추가 |
| **data_ornl.py** | `ornl_segmentation_argmax` 가 12→8 매핑, `__getitem__` 에 §4 의 D4/cyclic shift/noise/intensity augmentation 추가 |
| **data_dscnn.py** | `remap_label_v2` (2단계 매핑), `enumerate_samples` 에 BJ 2 source 추가, `__getitem__` 에 동일 augmentation |
| **losses.py** | `hard_bootstrap_loss` 신규, `sqrt_inv_class_weight` 와 `median_inv_class_weight` 두 옵션 |
| **build_cache_stage1.py** | label 채널만 12→8 재매핑하여 cache 다시 build (image cache 는 재사용 가능) |
| **model.py** | `N_CLASSES=8` 로 head 출력 자동 변경. 코드 변경 없음. |
| **train_stage1.py** | `--use-hard-bootstrap`, `--ema-decay` 플래그 추가. EMA 통합. |
| **train_stage2.py** | **Cross-validation loop** 추가 (`--fold` 인자 또는 외부 shell 에서 8회 호출), val source 별 mIoU 집계, fold 별 ckpt save |
| **infer.py** | `--stage 2 --fold k` 로 특정 fold ckpt 로드, ensemble 옵션 (`--ensemble` 로 8 ckpt 평균) |
| **run_stage1.sh** | `run_name` 에 `_8cls_v2` suffix, `--use-hard-bootstrap` 추가 |
| **run_stage2.sh** | for loop 으로 8 fold 순회 |
| **run_build_cache.sh** | 동일 (cache 의 label 채널만 재빌드 필요) |

---

## 8. 폴더 / 산출물 구조

```
DefSeg_AM/
├── PLAN.md                              # v1 (기존, 보존)
├── PLAN_v2.md                           # 본 문서
├── config.py                            # 8-class 로 변경
├── ...
├── cache/
│   ├── resized_sz1036/                  # image cache 재사용
│   └── stage1_layer_index_*_v2.npz      # v2 의 defect index (class 2~7 기준 재계산)
├── checkpoints/vits14_dpt_dual_sz1036_8cls_v2/
│   ├── stage1_best.pt
│   ├── stage2_best_fold{0..7}_S{1..8}.pt
│   └── cv_summary.json
└── figures/vits14_dpt_dual_sz1036_8cls_v2/
    ├── stage1/inference/...
    ├── stage2/inference/fold0/...
    ├── stage2/inference/fold1/...
    └── stage2/cv_summary_plot.png       # 8 fold 의 per-class IoU bar chart
```

---

## 9. 예상 시간 / 디스크

| 단계 | 시간 (1 GPU, FP32) | 디스크 |
|---|---|---|
| 캐시 label 재빌드 | ~30 분 | 기존 +0 (덮어씀) |
| Stage 1 학습 (30 epoch) | ~12 시간 (기존과 유사) | ckpt 51 MB |
| Stage 2 × 8 fold (각 50 epoch) | 각 ~20분 × 8 = ~3시간 | ckpt 51 MB × 8 = **408 MB** |
| **총** | **~16 시간** | ~460 MB |

(데이터 augmentation 36× 효과로 학습 자체는 더 빠르게 수렴할 가능성도 있음.)

---

## 10. 평가 / 보고

### 10.1 정량

각 fold 마다:
- per-class IoU (8개)
- mIoU (8 valid 평균)
- per-class precision, recall, F1
- confusion matrix

8 fold 평균 + std → 최종 cross-validation 결과.

### 10.2 정성

- 각 fold 의 val source 에서 대표 layer (4~6개) 의 4-panel PNG (visible/0, visible/1, GT, prediction)
- 8 fold ensemble vs single fold 비교 (`infer.py --ensemble`)
- Stage 1 only (KD) vs Stage 1 + Stage 2 (GT) 비교

### 10.3 v1 (12-class) 와의 비교

같은 ORNL Build 1 의 동일 layer 에서:
- v1 의 12-class prediction
- v2 의 8-class prediction
- → "통합/제거된 class 가 어떻게 표현되는지" 정성 비교

---

## 11. 실험 우선순위 / 단계

### Phase 1 — 코드 수정 + 캐시 재빌드 (Day 1, ~4 시간)

1. `config.py`: 8-class + BJ mapping
2. `data_ornl.py` / `data_dscnn.py`: augmentation 추가, label 매핑
3. `losses.py`: hard_bootstrap_loss
4. `train_stage1.py` / `train_stage2.py`: EMA, cross-val loop
5. `build_cache_stage1.py`: 캐시 label 재빌드 (30분)
6. Smoke test (`--quick` 모드)

### Phase 2 — Stage 1 (Day 2, ~12 시간)

```bash
bash DefSeg_AM/run_stage1.sh
# 8-class 출력 + EMA + augmentation 종합
```

### Phase 3 — Stage 2 × 8 fold (Day 3, ~3 시간)

```bash
for k in 0 1 2 3 4 5 6 7; do
    FOLD=$k bash DefSeg_AM/run_stage2_fold.sh
done
```

### Phase 4 — 평가 (Day 3, ~1 시간)

- 8 fold 의 ckpt 로 ensemble inference
- 정성 비교 (v1 vs v2)
- DEBUG_HISTORY.md 에 v2 결과 추가

---

## 12. 미해결 사항 / 추후 결정

1. **BJ source 의 native class mapping** — readme PDF 의 정확한 class 정의 확인 필요. v2 작성 시 일부 ambiguous 한 건 보수적으로 IGNORE.
2. **Stage 1 의 Hard-bootstrapping λ 값** — 0.8 권장이지만 ablation 시도 가치 있음.
3. **Test-Time Augmentation (TTA)** — D4 group 8 변형 ensemble. 기본은 미적용, 선택.
4. **(x, y) coordinate channel** — ablation 으로 별도 실험 가능 (v2 default 미적용).
5. **Cyclic shift 의 shift 범위** — 현재 `±img_size/4`. 너무 크면 build plate 의 경계가 깨질 수 있음 → 1차 학습 후 정성 검토.

---

> 작성: 2026-06-14. v1 (PLAN.md) 의 모델 구조 유지 + 클래스/데이터/학습 정책 4 가지 변경.
> 코드 구현 진행 시 본 문서를 reference 로 한 단계씩 진행 권장.
