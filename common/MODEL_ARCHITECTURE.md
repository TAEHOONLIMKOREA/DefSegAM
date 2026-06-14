# DefSeg-AM 모델 구조 (DefSegModel)

[models/model.py](../models/model.py) 의 `DefSegModel` 구조도.
모든 텐서 크기는 **입력 1036×1036** 기준이며 `(C, H, W)` 형식 (배치 B 생략).

- Backbone: **DINOv2 ViT-S/14 (frozen)** — `embed_dim D=384`, `patch=14`
- patch grid: `1036 / 14 = 74` → backbone feature 는 모두 `74×74`
- decoder 채널: `256`, 출력 클래스: `12`

---

## 1. 전체 흐름 (high-level)

```
        img0 (after melt)              img1 (after spread)
        (3, 1036, 1036)                (3, 1036, 1036)
              │                              │
              ▼                              ▼
     ┌──────────────────┐          ┌──────────────────┐
     │ DINOv2 ViT-S/14  │          │ DINOv2 ViT-S/14  │   ※ 같은 backbone, 가중치 공유
     │   (FROZEN)       │          │   (FROZEN)       │   ※ no_grad, .eval()
     └──────────────────┘          └──────────────────┘
        │ blocks [2,5,8,11]            │ blocks [2,5,8,11]
        │ 4-stage intermediate         │
        ▼                              ▼
   f0_s1..s4                       f1_s1..s4          각 (384, 74, 74)
        └──────────────┬───────────────┘
                       ▼   per-stage dual fusion (stage마다 독립)
            concat(f0, f1, f1−f0)  →  (1152, 74, 74)
                       │  1×1 conv (fuse_proj)
                       ▼
              fused s1..s4  →  (256, 74, 74) × 4
                       │  Reassemble (stage별 해상도 변환)
                       ▼
        s1(256,296,296) s2(256,148,148) s3(256,74,74) s4(256,37,37)
                       │  Top-down Fusion blocks (s4→s3→s2→s1)
                       ▼
                  (256, 592, 592)
                       │  Head (3×3 → ReLU → Dropout → 1×1)
                       ▼
                  (12, 592, 592)
                       │  bilinear interpolate → 입력 크기
                       ▼
                logits (12, 1036, 1036)
```

---

## 2. 텐서 크기 추적 (단계별)

| 단계 | 연산 | 출력 shape | 비고 |
|---|---|---|---|
| 입력 | `img0`, `img1` | `(3, 1036, 1036)` ×2 | dual visible |
| Backbone | DINOv2 `get_intermediate_layers([2,5,8,11])` | `(384, 74, 74)` × 4 × 2 | **frozen** |
| Fusion(concat) | `cat(f0, f1, f1−f0)` | `(1152, 74, 74)` × 4 | `3×384=1152` |
| Fusion(proj) | `fuse_proj` 1×1 conv | `(256, 74, 74)` × 4 | stage별 conv 4개 |
| Reassemble s1 | `ConvTranspose2d(k=4, s=4)` | `(256, 296, 296)` | 4× ↑ |
| Reassemble s2 | `ConvTranspose2d(k=2, s=2)` | `(256, 148, 148)` | 2× ↑ |
| Reassemble s3 | `Identity` | `(256, 74, 74)` | 그대로 |
| Reassemble s4 | `Conv2d(k=3, s=2, p=1)` | `(256, 37, 37)` | 2× ↓ |
| Fusion fb[3] | `FFB(s4)` | `(256, 74, 74)` | 37 → ×2 |
| Fusion fb[2] | `FFB(out, +s3)` | `(256, 148, 148)` | 74 → ×2 |
| Fusion fb[1] | `FFB(out, +s2)` | `(256, 296, 296)` | 148 → ×2 |
| Fusion fb[0] | `FFB(out, +s1)` | `(256, 592, 592)` | 296 → ×2 |
| Head | 3×3 → ReLU → Drop → 1×1 | `(12, 592, 592)` | 채널 256→12 |
| Upsample | `F.interpolate(bilinear)` | `(12, 1036, 1036)` | 입력 크기로 복원 |

> Reassemble 의 4가지 해상도(296/148/74/37)는 top-down fusion이 `s4→s1`로 ×2씩 올라갈 때
> 각 단계의 skip과 **해상도가 정확히 맞도록** 역으로 설계된 값이다.

---

## 3. Per-stage Dual Fusion 상세

각 stage `s` 에서 두 이미지의 feature와 그 **차이**를 합쳐 256채널로 압축.

```
   f0[s] (384,74,74)  ─┐
   f1[s] (384,74,74)  ─┼─ concat(dim=채널) ─→ joint (1152, 74, 74)
   f1[s]−f0[s] (384)  ─┘                          │
                                                  │  fuse_proj[s] : Conv2d(1152, 256, 1×1)
                                                  ▼
                                          fused[s] (256, 74, 74)

   · f1−f0 = melt↔spread 변화량 (결함 단서) 을 명시적으로 주입
   · 1×1 conv = 픽셀별 FC (74×74 위치에 동일 W 공유) → 채널 1152→256 압축+혼합
   · stage 4개가 각자 별도 fuse_proj 를 가짐 (얕은/깊은 feature 특성이 달라서)
```

---

## 4. Reassemble — 4개 해상도로 펼치기

DINOv2는 모든 stage가 같은 74×74지만, DPT 디코더는 **서로 다른 해상도의 피라미드**가 필요하다.
그래서 stage별로 업/다운샘플:

```
   fused s1 (256,74,74) ──ConvTranspose2d(k4,s4)──→ (256, 296, 296)   4× 확대 (가장 세밀)
   fused s2 (256,74,74) ──ConvTranspose2d(k2,s2)──→ (256, 148, 148)   2× 확대
   fused s3 (256,74,74) ──Identity──────────────→ (256,  74,  74)    유지
   fused s4 (256,74,74) ──Conv2d(k3,s2,p1)───────→ (256,  37,  37)    2× 축소 (가장 거침)
```

---

## 5. Top-down Feature Fusion (s4 → s1)

가장 거친 s4부터 시작해 한 단계씩 올라가며(×2 upsample) skip을 더한다.

```
            reasm s4 (37×37)
                  │
            ┌─────▼──────┐
            │  fb[3]     │  skip 없음
            │  RCU → ↑2  │
            └─────┬──────┘
                  ▼ (74×74)         + reasm s3 (74×74)
            ┌─────▼──────┐◀─────────────┘
            │  fb[2]     │
            │ +skip→RCU→↑2
            └─────┬──────┘
                  ▼ (148×148)       + reasm s2 (148×148)
            ┌─────▼──────┐◀─────────────┘
            │  fb[1]     │
            │ +skip→RCU→↑2
            └─────┬──────┘
                  ▼ (296×296)       + reasm s1 (296×296)
            ┌─────▼──────┐◀─────────────┘
            │  fb[0]     │
            │ +skip→RCU→↑2
            └─────┬──────┘
                  ▼ (592×592) → Head
```

### FeatureFusionBlock (FFB) 내부

```
   prev ──────────────────────────┐
                                   ▼
   skip ──[ResidualConvUnit]──→ (＋) add        ※ skip 있을 때만
                                   │
                                   ▼
                        [ResidualConvUnit]
                                   │
                          bilinear ×2 upsample
                                   │
                            1×1 conv (out_conv)
                                   ▼
                                 out
```

### ResidualConvUnit (RCU)

```
   x ──┬──────────────────────────────────────┐
       ▼                                       │
   ReLU → Conv3×3 → ReLU → Conv3×3 ──→ (＋) add ◀┘
                                        ▼
                                       out
   (DPT 표준 잔차 유닛; 채널 수 256 유지)
```

---

## 6. Head (분류기)

```
   (256, 592, 592)
        │  Conv2d 3×3 (256→256)
        │  ReLU
        │  Dropout2d(p=0.1)
        │  Conv2d 1×1 (256→12)     ← 픽셀별 12-class logits
        ▼
   (12, 592, 592)
        │  F.interpolate(bilinear → 1036×1036)
        ▼
   (12, 1036, 1036)   = 픽셀마다 12개 class score
```

---

## 7. 학습 대상 vs Frozen

| 모듈 | 학습? | 역할 |
|---|---|---|
| `backbone` (DINOv2 ViT-S/14) | ❌ **Frozen** | 사전학습 feature 추출기 (no_grad, eval) |
| `fuse_proj` (1×1 conv ×4) | ✅ | dual feature 융합 + 채널 압축 |
| `reassemble` (4) | ✅ | 다중 해상도 변환 |
| `fusion_blocks` (FFB ×4) | ✅ | top-down 융합 |
| `head` | ✅ | 12-class 분류 |

- 체크포인트에는 **학습 대상(`fuse_proj`/`reassemble`/`fusion_blocks`/`head`)만** 저장
  (`trainable_state_dict()`), backbone은 제외해 용량을 줄인다.
- **Stage 1 → Stage 2** 전환 시 이 trainable weight만 이어받는다
  (`load_trainable_state_dict()`).

---

## 8. forward 한눈에 (코드 대응)

```python
f0 = backbone(img0)            # 4 × (384,74,74)   no_grad
f1 = backbone(img1)            # 4 × (384,74,74)
for s in range(4):             # per-stage dual fusion
    joint   = cat(f0[s], f1[s], f1[s]-f0[s])   # (1152,74,74)
    fused[s]= fuse_proj[s](joint)              # (256,74,74)
reasm = [reassemble[s](fused[s]) for s in 0..3]  # 296/148/74/37

out = fusion_blocks[3](reasm[3])               # s4        → 74
out = fusion_blocks[2](out, reasm[2])          # +s3       → 148
out = fusion_blocks[1](out, reasm[1])          # +s2       → 296
out = fusion_blocks[0](out, reasm[0])          # +s1       → 592
out = head(out)                                # (12,592,592)
out = interpolate(out, 1036)                   # (12,1036,1036)
```
```
입력 dual → DINOv2(frozen)×2 → 차이융합 → 다중해상도 → top-down 융합 → 분류 → 원본 크기
```
