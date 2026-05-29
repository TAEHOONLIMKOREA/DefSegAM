# DefSeg-AM 구현 계획

> **목표**: ORNL L-PBF 의 layer-wise powder bed 이미지에 대해
> **DINOv2 (frozen) backbone + DPT-style multi-scale decoder + classifier** 로
> 12-class 결함 segmentation 모델을 학습한다.
>
> 학습은 **2-stage**:
>   - **Stage 1 (KD pretrain)**: ORNL Co-Registered HDF5 의
>     `slices/segmentation_results` (DSCNN 모델이 예측한 12-class boolean mask)
>     를 hard pseudo-label 로 두고 ORNL 도메인에서 대규모 사전학습.
>   - **Stage 2 (GT finetune)**: DSCNN_Dataset 의 `training/annotations/*.npy`
>     (사람이 직접 라벨링한 native class) 를 ORNL 12-class 공간으로 재매핑하여
>     소규모 진짜 GT 로 fine-tuning.
>
> 모델/코드 패턴은 [seung_dscnn/](../seung_dscnn) 를 참고하되, 입력 도메인·학습 순서·디코더가 다르다.

---

## 1. 핵심 설계 결정

### 1.1 2-Stage 학습 — KD pretrain → GT finetune (사용자 요구)

| Stage | 데이터 소스 | 라벨 형태 | 규모 | 목적 |
|---|---|---|---|---|
| **S1** | `ORNL_Data/Co-Registered .../[baseline] (Peregrine v2023-11)/*.hdf5` 의 `slices/segmentation_results/{0..11}` | DSCNN 모델 예측 (12 boolean mask) → **argmax** 후 hard CE | 5 빌드 × 수천 layer = 수만 장 (라벨 무료) | 도메인 사전학습 — head 가 ORNL 픽셀 통계에 적응 |
| **S2** | `ORNL_Data/DSCNN_Dataset/Peregrine v2021-03/Laser Powder Bed Fusion/` + `v2022-10.1/Laser_Powder_Bed_Fusion/EOS_M290 & AddUp_FormUp_350/.../training/annotations/*.npy` | 사람 GT (재료별 native ID) → **ORNL 12-class 재매핑** → hard CE | 6 source × 수십 layer = ~80~90 장 | 진짜 GT 로 마무리 — 라벨 잡음 제거 |

> S1 → S2 전환 시 backbone 는 frozen 유지, head/decoder/classifier 의 weight 만 그대로 이어받고 optimizer state 는 reset (LR schedule 새로 시작).

### 1.2 입력 모달리티 — Dual visible (seung_dscnn 와 동일)

- **`visible/0` (after melt)** + **`visible/1` (after spread)** 두 채널을 동시에 입력.
- 두 이미지를 **각각 DINOv2 로 forward** → 각 scale 마다 patch token feature `(f0, f1)` 획득.
- Fusion: `(f0, f1, f1 - f0)` concat → 디코더 입력 채널 = `3 × D_embed`.
- Stage 1, Stage 2 둘 다 입력 dual 동일 (DSCNN_Dataset 도 `data/visible/0`·`data/visible/1` 둘 다 존재).

### 1.3 Backbone — DINOv2 ViT-S/14 frozen

| 항목 | 값 |
|---|---|
| Backbone | `dinov2_vits14` (embed_dim=384, patch_size=14, 12 blocks) — `torch.hub.load("facebookresearch/dinov2", ...)` |
| 학습 | **Frozen** (`requires_grad=False`, 항상 `.eval()`) — Stage 1/2 모두 |
| 입력 해상도 | **1036×1036** (= 74×74 patches, 14의 배수) |
| Intermediate layer 추출 | block index `[2, 5, 8, 11]` (0-based, 즉 3/6/9/12 번째 block) — DPT 의 4-stage 입력 |
| 추출 인터페이스 | `backbone.get_intermediate_layers(img, n=[2,5,8,11], return_class_token=False, norm=True)` |

> 추후 ablation 으로 `vitb14` / `vitl14` 를 시도할 수 있으나 1차는 `vits14` 로 고정 (메모리·속도 보수적).

### 1.4 Decoder — DPT-style multi-scale fusion

- DINOv2 4-stage intermediate token `(B, 384, 74, 74)` × 4 개를 받아 **Reassemble** 로 서로 다른 해상도로 변환 → **Fusion blocks** 로 top-down 으로 합치며 progressive upsample → **Classifier head** 가 12-channel logits 출력.

| 컴포넌트 | 출력 해상도 (1036 입력 기준) | 구성 |
|---|---|---|
| Reassemble s1 (block 3) | (256, 296, 296) | 1×1 conv (proj) → ConvTranspose2d(stride=4) — 4× upsample |
| Reassemble s2 (block 6) | (256, 148, 148) | 1×1 conv → ConvTranspose2d(stride=2) — 2× upsample |
| Reassemble s3 (block 9) | (256, 74, 74)  | 1×1 conv (identity) |
| Reassemble s4 (block 12)| (256, 37, 37)  | 1×1 conv → Conv2d(stride=2) — 2× downsample |
| Fusion blocks (s4→s1) | 각각 (256, ...) | 잔차 conv (3×3 BN ReLU 2회) + bilinear ×2 upsample + 위 stage 와 add |
| Head | (12, 1036, 1036) | 3×3 conv → 256 → ReLU → 1×1 conv → 12 → bilinear ×2 upsample (마지막) |

- **Dual fusion**: 각 stage 의 `(f0_s, f1_s, f1_s - f0_s)` → 1×1 conv 로 `3×D → 256` 채널 압축 후 Reassemble 로 넘김.
- DPT 원 구현 ([Intel-ISL/DPT](https://github.com/isl-org/DPT)) 의 ResidualConvUnit / FeatureFusionBlock 패턴을 가져오되, 본 프로젝트는 외부 mmsegmentation 의존을 피하기 위해 **자체 구현** (~150 LOC).
- 학습 파라미터 ≈ Reassemble + Fusion + Head = **수 M 수준** (backbone 제외).

### 1.5 클래스 공간 — ORNL 12-class 고정

- 출력 차원은 **ORNL HDF5 의 12 class** ([seung_dscnn/config.py:21-34](../seung_dscnn/config.py#L21-L34) 와 동일):

  ```
  0 Powder, 1 Printed, 2 Recoater Hopping, 3 Recoater Streaking,
  4 Incomplete Spreading, 5 Swelling, 6 Debris, 7 Super-Elevation,
  8 Spatter, 9 Misprint, 10 Over Melting, 11 Under Melting
  ```

- **Stage 2 의 DSCNN_Dataset 재매핑**도 같은 12 class 로. 매핑 표는 [seung_dscnn/config.py:41-99](../seung_dscnn/config.py#L41-L99) `MATERIAL_TO_ORNL` 를 그대로 가져온다 (검증된 매핑).
- 매핑 불가능한 native class (e.g. v2021 LPBF 의 Soot, GammaPrint 의 Localized Bright Spot) → `-1` (IGNORE_INDEX, CE 계산 시 무시).
- **EBPBF/BJ 제외** (사용자 요구; LPBF 만, seung_dscnn 정책 계승).

---

## 2. 전체 아키텍처

```
                                ┌──────────────────────────────────────────────┐
                                │  visible/0 (after melt)    visible/1 (after spread)
                                │     │                          │
                                │     ▼                          ▼
                                │  [DINOv2 ViT-S/14 frozen]   [DINOv2 ViT-S/14 frozen]
                                │     │ (4 intermediate)        │ (4 intermediate)
                                │     ▼                          ▼
                                │   f0_s1..s4 (B,384,Hp,Wp)    f1_s1..s4
                                └──────┬──────────────────────────┬─────────────┘
                                       │ per-stage fusion         │
                                       └─→ (f0, f1, f1-f0) concat ──→ 1x1 conv → 256ch
                                                                       │
                                                                       ▼ (4 stages)
                                                    ┌──────────────────────────────────┐
                                                    │  DPT Reassemble (4 resolutions)  │
                                                    │  s1: 4×↑ , s2: 2×↑ , s3: id, s4: 2×↓
                                                    └──────────────┬───────────────────┘
                                                                   │
                                                                   ▼
                                                    ┌──────────────────────────────────┐
                                                    │  Top-down Fusion blocks (s4→s1)  │
                                                    │  잔차 + bilinear ×2 + add        │
                                                    └──────────────┬───────────────────┘
                                                                   │
                                                                   ▼ (B, 256, ~1/2 input)
                                                    ┌──────────────────────────────────┐
                                                    │  Classifier Head                 │
                                                    │  3x3 conv 256 → 256              │
                                                    │  1x1 conv 256 → 12               │
                                                    │  bilinear ×2 → (B,12,1036,1036)  │
                                                    └──────────────┬───────────────────┘
                                                                   │
                                ┌──────────────────────────────────┴─────────────┐
                                │                                                │
                          ┌─────▼─────┐                                    ┌────▼─────┐
                          │ Stage 1   │                                    │ Stage 2  │
                          │ CE loss vs│                                    │ CE loss  │
                          │ DSCNN     │                                    │ vs human │
                          │ argmax    │                                    │ GT (re-  │
                          │ (ORNL hdf5)│                                   │ mapped)  │
                          └───────────┘                                    └──────────┘
```

---

## 3. 데이터 파이프라인

### 3.1 Stage 1 입력 — ORNL Co-Registered HDF5 (DSCNN pred)

- **Path**: `ORNL_Data/Co-Registered In-Situ and Ex-Situ Dataset/[baseline] (Peregrine v2023-11)/{2021-04-16 TCR Phase 1 Build 2, ..., 2021-08-23 TCR Phase 1 Build 5}.hdf5` (5 빌드)
- **Per-layer 입력**:
  - `slices/camera_data/visible/0[layer]` → (1842, 1842) float32 → percentile-normalize → uint8 → resize 1036
  - `slices/camera_data/visible/1[layer]` → 동일 처리
- **Per-layer 라벨 (KD 정답)**:
  - `slices/segmentation_results/{0..11}[layer]` → 12 boolean mask, shape (1842, 1842)
  - **argmax 규약** (seung_dscnn/infer_ornl.py:68-76 `_ornl_gt_argmax` 와 동일): 큰 class ID (= 결함) 가 작은 ID (= Powder/Printed) 를 덮어쓰도록 `for c in range(12): out[mask_c] = c` 순차 적용 → (1842, 1842) int8
  - 어떤 mask 에도 속하지 않는 pixel = `-1` (IGNORE)
  - NEAREST resize → 1036
- **Layer 필터링** (class imbalance 완화의 첫 단계):
  - 빌드 시작/끝 (powder layer 만) 제외: 안쪽 5%~95% 구간만 사용 (seung_dscnn/infer_ornl.py:136-140 `select_default_layers` 와 동일 룰).
  - `part_ids` 가 전부 0 인 layer (= as-yet-empty) 도 skip.
  - **Powder/Printed only layer 강제 제외**: DSCNN pred 의 12-class 중 **class 2~11 (Powder/Printed 제외한 결함 10종)** boolean mask 의 합이 0 인 layer 는 학습 데이터에서 전부 drop. 정보량 0 + class imbalance 의 주범.
- **Per-layer defect ratio 사전 계산** (oversampling 용):
  - 살아남은 layer 각각에 대해 `defect_ratio = sum(class 2..11 mask) / total_pixels` 를 한 번 계산하여 캐시 → `DefSeg-AM/cache/stage1_layer_index.npz` (`build_id, layer_idx, defect_ratio` 컬럼).
  - 캐시 빌드는 `data_ornl.py` 의 `build_layer_index()` 가 담당, 첫 학습 시 1회 실행 (수 분 예상).
  - `WeightedRandomSampler` 의 weight 로 사용 (§5.1 참고).
- **Split**:
  - Train: Build 2, 3, 4, 5
  - Val: **Build 1** (`2021-07-13 TCR Phase 1 Build 1`) — seung_dscnn 의 매 epoch 비교 추론 빌드와 동일하게 두어 정성 비교 가능.
- **데이터셋 클래스**: `DefSegORNLDataset` — layer index 를 sample 단위로 enumerate, lazy HDF5 open per worker (multi-process safety).

### 3.2 Stage 2 입력 — DSCNN_Dataset annotations (human GT)

- **Sources** (seung_dscnn/config.py:103-140 의 `TRAIN_SOURCES` 6 개를 그대로 계승):

  | Name | Root | Mapping key | n_classes (native) |
  |---|---|---|---|
  | v2021_LPBF | `DSCNN_Dataset/Peregrine Dataset v2021-03/Laser Powder Bed Fusion/` | v2021_LPBF | 9 |
  | v2022_17-4PH | `Peregrine Dataset v2022-10.1/Laser_Powder_Bed_Fusion/EOS_M290/17-4_PH_Stainless_Steel/training/` | 17-4_PH_Stainless_Steel | 12 |
  | v2022_GammaPrint | `.../EOS_M290/GammaPrint-700/training/` | GammaPrint-700 | 15 |
  | v2022_Inc718_1 | `.../EOS_M290/Inconel_718_1/training/` | Inconel_718_1 | 15 |
  | v2022_Inc718_2 | `.../EOS_M290/Inconel_718_2/training/` | Inconel_718_2 | 12 |
  | v2022_Maraging | `.../AddUp_FormUp_350/Maraging_Steel/training/` | Maraging_Steel | 10 |

- **Per-layer 입력**:
  - `data/visible/0/{stem}.tif` → uint8 grayscale → resize 1036
  - `data/visible/1/{stem}.tif` → 동일
- **Per-layer 라벨**:
  - `annotations/{stem}.npy` → (H, W) int (native class) → `MATERIAL_TO_ORNL[mapping_key]` 로 재매핑 → 12-class int8 (`-1` = IGNORE) → NEAREST resize → 1036
  - 재매핑 로직: [seung_dscnn/data.py:56-65](../seung_dscnn/data.py#L56-L65) `remap_label` 를 그대로 import (혹은 복사).
- **Split**: seung_dscnn 정책 유지 — **v2022_Maraging 을 validation source** 로, 나머지 5 개를 train.
- **데이터셋 클래스**: `DefSegDSCNNDataset` — `SampleSpec` enumerate 후 layer-단위 indexing (seung_dscnn/data.py:30-53 와 동일 구조).

### 3.3 Augmentation — recoater 방향성 보존

- **flip / rotation 금지** (seung_dscnn/data.py:115 의 룰 계승; recoater streaking/hopping 이 방향성 있음 → 좌우 반전 시 라벨 의미 깨짐).
- **Brightness jitter** 만 가볍게 (±15%, p=0.5).
- 추후 ablation 으로 random crop 검토 가능 (계획 단계에서는 미적용).

### 3.4 정규화

- DINOv2 = ImageNet pretrained → **ImageNet mean/std** (`[0.485, 0.456, 0.406]` / `[0.229, 0.224, 0.225]`) 적용.
- Grayscale → 3-channel 로 replicate 후 정규화 (seung_dscnn/data.py:68-74 `_normalize_image`).

### 3.5 ORNL float32 → uint8 변환 (Stage 1 전용)

- ORNL HDF5 의 `visible/*` 는 float32. percentile (1, 99) 기반 per-image normalize → uint8 (seung_dscnn/infer_ornl.py:57-65 `_ornl_image_to_uint8`).
- DSCNN_Dataset 의 `.tif` 는 이미 uint8 → 그대로 사용.

---

## 4. 모델 상세

### 4.1 `DefSegModel` (model.py)

```python
class DefSegModel(nn.Module):
    def __init__(
        self,
        backbone_name: str = "dinov2_vits14",
        n_classes: int = 12,
        decoder_channels: int = 256,
        intermediate_layers: tuple[int, ...] = (2, 5, 8, 11),  # 0-based
    ):
        # 1. DINOv2 hub load, freeze, .eval()
        # 2. Per-stage fusion 1x1 conv: 3*D -> decoder_channels (×4)
        # 3. Reassemble: 4 stages (4x↑, 2x↑, id, 2x↓)
        # 4. Fusion blocks: 4개 (ResidualConvUnit ×2 + bilinear ×2 upsample + skip add)
        # 5. Head: conv3x3 -> ReLU -> conv1x1 -> n_classes -> bilinear x2 upsample
```

- **trainable_parameters()** / **trainable_state_dict()**: backbone 제외한 모든 모듈 (fusion + reassemble + fusion_blocks + head) — checkpoint 에는 학습 가능한 파라미터만 저장 (seung_dscnn/model.py:208-218 패턴).

### 4.2 Backbone 호출

- DINOv2 v2 의 `get_intermediate_layers(x, n=[2,5,8,11], return_class_token=False, norm=True, reshape=True)` 사용 → 바로 `(B, D, Hp, Wp)` × 4 반환 (reshape=True).
- `torch.no_grad()` 로 감싸기 (gradient 흐름 차단).

### 4.3 Reassemble & Fusion 구현 메모

- DPT 의 표준 구현 그대로:
  - **ResidualConvUnit**: `Conv3x3 → ReLU → Conv3x3 → ReLU + skip add`
  - **FeatureFusionBlock**: `ResidualConvUnit(prev) → add(skip) → ResidualConvUnit → bilinear ×2 upsample`
- 입력/출력 텐서 shape 를 `assert` 로 검증 (1036 입력 가정).
- BatchNorm 대신 그냥 conv + ReLU — batch=2 정도라 BN 통계 불안.

---

## 5. 학습 절차

### 5.1 Stage 1 (KD pretrain)

| 항목 | 값 |
|---|---|
| Optimizer | AdamW, lr=5e-4, weight_decay=1e-4 |
| Scheduler | CosineAnnealingLR (T_max = epochs × len(train_loader)) |
| **Loss** | **Focal Loss (γ=2)** + α (class weight) + `ignore_index=-1`. CE 가 아닌 Focal — 잘 맞추는 pixel (Powder/Printed) 의 loss 가 `(1-pt)^γ` 로 자동 down-weight → gradient 가 hard pixel (rare defect) 로 집중. |
| α (class weight) | sqrt-inverse-frequency, clip 50 (seung_dscnn/data.py:158-173 동일) — Focal Loss 의 α 자리에 사용. Stage 1 데이터 기준으로 학습 시작 시 1회 계산. |
| **Sampling** | `WeightedRandomSampler(weights = defect_ratio^0.5 + 1e-3, replacement=True)`. defect-rich layer 가 epoch 당 더 자주 뽑히게 (가벼운 oversampling — `^0.5` 로 완만하게). `defect_ratio` 는 §3.1 의 사전 캐시 사용. |
| Batch size | 2 (1036 입력 두 장 × ViT-S/14 frozen — V100/A100 16GB 기준 안전) |
| Epochs | 30 (대규모 데이터, 사전학습 단계) |
| Mixed precision | `torch.amp.autocast(device_type='cuda')` — 메모리 절감 |
| Checkpoint | `DefSeg-AM/checkpoints/<run_name>/stage1_best.pt` (val_acc 기준 best) |

#### Focal Loss 구현 (참고)

```python
def focal_loss(logits, target, gamma=2.0, alpha_weight=None, ignore_index=-1):
    ce = F.cross_entropy(
        logits, target, weight=alpha_weight,
        ignore_index=ignore_index, reduction='none',
    )                                       # (B, H, W)
    pt = torch.exp(-ce)                     # 정답 class 의 확률
    valid = (target != ignore_index)
    fl = (1 - pt) ** gamma * ce
    return fl[valid].mean()
```

### 5.2 Stage 2 (GT finetune)

| 항목 | 값 |
|---|---|
| 초기화 | Stage 1 best checkpoint 의 `trainable_state_dict` 를 그대로 load |
| Optimizer | AdamW, **lr=1e-4** (Stage 1 의 1/5; 진짜 GT 라 과적합 위험) |
| Scheduler | CosineAnnealingLR |
| **Loss** | **표준 `CrossEntropy(weight=α, ignore_index=-1)`** — Focal 미사용. Stage 2 는 데이터가 작고 (~80 layer) 사람이 결함 중심으로 선별해서 라벨링한 깨끗한 GT 라, Focal 의 `(1-pt)^γ` down-weight 가 오히려 underfitting 위험. seung_dscnn 와 동일한 CE + sqrt-inv weight 로 보수적 finetune. |
| α (class weight) | sqrt-inverse-frequency, clip 50 — Stage 2 데이터로 재계산 |
| Sampling | 기본 `RandomSampler` (uniform). Stage 2 는 데이터가 ~80 layer 로 작아 oversampling 효과 제한적 + 각 layer 가 이미 사람이 라벨링한 결함 중심이므로 layer 단위 oversampling 불필요. |
| Batch size | 2 |
| Epochs | 50 (소규모 데이터, 충분히 수렴할 때까지) |
| Checkpoint | `DefSeg-AM/checkpoints/<run_name>/stage2_best.pt` |

### 5.3 Per-epoch ORNL 정성 비교 (선택)

- seung_dscnn 처럼 매 epoch 끝에 `2021-07-13 TCR Phase 1 Build 1` 의 12 layer 균등 추출하여 4-panel PNG 저장: `[visible/0, visible/1, DSCNN pred (= S1 정답), our prediction]`.
- Stage 2 의 경우 `[visible/0, visible/1, S1 정답, our prediction]` 유지 — Stage 2 GT 는 DSCNN_Dataset 영역이라 ORNL layer 와 직접 비교 불가.

### 5.4 Metrics

- **Pixel accuracy** (IGNORE 제외)
- **Per-class IoU** + **mIoU** (12 class)
- **Confusion matrix** (validation 끝에 1회 저장)

---

## 6. 폴더 / 파일 구조

```
DefSeg-AM/
├── PLAN.md                       # 본 문서
├── README.md                     # (구현 후 작성) 실행법, 결과 요약
├── __init__.py                   # 패키지화 (빈 파일)
│
├── config.py                     # 경로·하이퍼파라미터·MATERIAL_TO_ORNL·ORNL_CLASS_NAMES
├── data_ornl.py                  # Stage 1 데이터셋: DefSegORNLDataset (HDF5 layer-wise)
├── data_dscnn.py                 # Stage 2 데이터셋: DefSegDSCNNDataset (재료별 native → ORNL remap)
├── model.py                      # DefSegModel (DINOv2 + DPT decoder + classifier)
├── losses.py                     # class weight 계산 + CE wrapper (선택, train.py 안에 둬도 됨)
├── train_stage1.py               # Stage 1 KD pretrain 진입점
├── train_stage2.py               # Stage 2 GT finetune 진입점 (--init_from stage1_best.pt)
├── infer.py                      # 학습된 모델로 ORNL HDF5 4-panel 비교 PNG 생성 (seung_dscnn/infer_ornl.py 포팅)
│
├── run_stage1.sh                 # nohup 백그라운드 학습 스크립트
├── run_stage2.sh
│
├── checkpoints/                  # (gitignore) <run_name>/stage1_best.pt, stage2_best.pt
└── figures/                      # (gitignore) <run_name>/{stage1,stage2}/epoch_NNN/layerXXXX.png
```

### 6.1 seung_dscnn 와의 코드 공유

- `MATERIAL_TO_ORNL`, `ORNL_CLASS_NAMES`, `remap_label`, `_ornl_image_to_uint8`, `_ornl_gt_argmax`, `_normalize_image`, `select_default_layers` → seung_dscnn 에서 **그대로 import** 하거나 (cross-package import) **복사** (독립성 유지).
- 1차는 **복사** (DefSeg-AM 을 self-contained 로) — seung_dscnn 수정이 본 프로젝트에 영향 주지 않도록.

---

## 7. 평가 / 비교 실험

### 7.1 비교군

| Model | 학습 데이터 | 비고 |
|---|---|---|
| **seung_dscnn (baseline)** | DSCNN_Dataset GT 만 (1-stage) | 기존 결과 |
| **DefSeg-AM S1 only** | ORNL DSCNN pred 만 | KD 만의 효과 확인 |
| **DefSeg-AM S2 only** | DSCNN_Dataset GT 만 (no KD pretrain) | DPT decoder 자체 효과 (vs seung_dscnn 의 simple head) |
| **DefSeg-AM S1 → S2 (full)** | KD pretrain → GT finetune | 본 제안 — 모든 컴포넌트 |

### 7.2 정량 (validation)

- Stage 1 val = ORNL Build 1 의 DSCNN pred (= self-distillation 의 한계 측정)
- Stage 2 val = v2022_Maraging 의 human GT
- 보고: pixel acc, per-class IoU, mIoU, confusion matrix

### 7.3 정성

- ORNL Build 1 의 동일 layer set 에 대해 4 모델의 prediction PNG 를 같은 자리에 배치하여 비교.

---

## 8. 실행 정책

- **호스트 venv 직접 실행** (seung_dscnn 와 동일 — docker compose 정책 §3 적용 안 됨; standalone 모듈).
- 풀런은 사용자가 직접 실행, 본 PLAN 구현 단계에서는 `--quick` 스모크 (2 epoch, batch 1, 224 입력) 까지만 Claude 가 검증.

```bash
# Stage 1
CUDA_VISIBLE_DEVICES=1 nohup ./venv/bin/python -m DefSeg-AM.train_stage1 \
    --epochs 30 --batch-size 2 --img-size 1036 \
    --run-name vits14_dpt_dual \
    > DefSeg-AM/stage1.log 2>&1 &

# Stage 2 (Stage 1 best 자동 load)
CUDA_VISIBLE_DEVICES=1 nohup ./venv/bin/python -m DefSeg-AM.train_stage2 \
    --epochs 50 --batch-size 2 --img-size 1036 \
    --run-name vits14_dpt_dual \
    > DefSeg-AM/stage2.log 2>&1 &
```

> 주의: `DefSeg-AM` 의 하이픈 때문에 Python 패키지로는 `-m DefSeg-AM.train_stage1` 직접 호출이 안 된다. 구현 시 폴더명을 `DefSeg_AM` (underscore) 로 바꾸거나, `python DefSeg-AM/train_stage1.py` 로 직접 실행하거나, 패키지 진입을 `runpy` 로 우회. **구현 시 사용자에게 확인 필요**.

---

## 9. seung_dscnn 대비 차이점 요약

| 항목 | seung_dscnn | DefSeg-AM |
|---|---|---|
| Backbone | DINOv2 또는 DINOv1 선택 가능 | DINOv2 ViT-S/14 고정 (확장 여지) |
| Decoder | simple 4-layer conv head (late/early upsample 옵션) | **DPT-style multi-scale fusion** |
| 학습 데이터 | DSCNN_Dataset GT 만 (1-stage) | **ORNL DSCNN pred (S1) → DSCNN_Dataset GT (S2)** 2-stage |
| ORNL 평가 | 학습은 DSCNN_Dataset, ORNL 은 정성 비교만 | ORNL 이 학습에 직접 포함 (S1) |
| 입력 | dual visible/0+1 (concat_diff) | dual visible/0+1 (per-stage concat_diff) — 동일 |
| Class 공간 | ORNL 12-class | ORNL 12-class — 동일 |

---

## 10. 구현 순서 (예상)

1. `config.py` — 경로·상수·매핑 정의 (seung_dscnn 에서 carry-over).
2. `data_ornl.py` — Stage 1 데이터셋 + DataLoader smoke test.
3. `data_dscnn.py` — Stage 2 데이터셋 + DataLoader smoke test.
4. `model.py` — DefSegModel (DPT decoder 자체 구현) + forward shape 검증.
5. `train_stage1.py` — 학습 루프 + `--quick` 스모크 (224 입력, 2 epoch).
6. `train_stage2.py` — Stage 1 checkpoint load + finetune.
7. `infer.py` — ORNL 4-panel 시각화 (seung_dscnn/infer_ornl.py 포팅).
8. `run_stage1.sh`, `run_stage2.sh` — nohup 백그라운드 스크립트.
9. `README.md` — 실행법·결과 요약 (학습 완료 후).

---

## 11. 미해결 / 추후 결정 사항

- **패키지 이름의 하이픈 문제**: `DefSeg-AM` 폴더명 그대로 가면 `python -m DefSeg-AM.xxx` 가 import 불가. 구현 시 `DefSeg_AM` 으로 rename 하거나 직접 `python DefSeg-AM/xxx.py` 로 실행하는 방식 확정 필요.
- **DPT decoder 의 정확한 stride / channel 설계**는 1036 입력에 맞춰 구현 시 미세 조정 (특히 stage s1 의 4× upsample 후 296 vs s2 의 148 의 정확한 align).
- **Stage 1 의 batch 구성**: 5 빌드 × 수천 layer 면 1 epoch 가 매우 오래 걸릴 수 있음 → 필요 시 `--steps-per-epoch` 로 sub-sample 도입 검토.
- **Class imbalance 추가 대책**: §3.1 의 defect-free layer 필터링 + §5.1 의 Focal Loss + light oversampling 으로 1차 대응. 그래도 rare class (Spatter, Under Melting 등) 의 mIoU 가 0 에 머무를 경우 → **Dice Loss 추가** (`loss = focal + 0.5 * dice`) 또는 **oversampling weight 강화** (`^0.5` → `^1.0`) 시도.
- **Stage 2 만으로 finetune 충분한지 vs joint training**: 일단 sequential (S1 → S2) 로 가되, ablation 으로 `joint(S1 loss + α·S2 loss)` 비교 가능.
