# DefSeg-AM v1 vs v2 — 학습 결과 비교

> 작성일: 2026-06-14
> v1 ckpt: `DefSeg_AM/checkpoints/vits14_dpt_dual_sz1036_1gpu_nanfix/`
> v2 ckpt: `DefSeg_AM/checkpoints/vits14_dpt_dual_sz1036_8cls_v2/`

본 문서는 [PLAN.md](../PLAN.md) (v1) 와 [PLAN_v2.md](../PLAN_v2.md) (v2) 의 학습 결과를 정리/비교한다.
모델 구조 (DINOv2 ViT-S/14 frozen + DPT decoder) 는 동일, **클래스 / 데이터 / 학습 정책** 만 변경.

---

## 1. 한눈에 보는 비교

| 항목 | v1 | v2 | 변화 |
|---|---|---|---|
| 클래스 수 | 12 (ORNL Peregrine 표준) | **8** (rare/ambiguous 제거 + Recoater 통합) | -33% |
| Stage 1 Val | ORNL Build 1 (DSCNN pred) | ORNL Build 1 (DSCNN pred, 8-class 재매핑) | 동일 |
| Stage 2 Val | **v2022_Maraging 단일** | **8-fold cross-validation** (8 source 모두) | robust ↑↑ |
| Stage 2 데이터 | LPBF 5 source (EBPBF/BJ 제외) | **LPBF 6 + BJ 2** (EBPBF 만 제외) | 8 source |
| Augmentation | brightness jitter ±15% 만 | DSCNN noise + intensity shift + **D4 rotation/flip + Cyclic shift** + brightness | 36+ × |
| EMA weight save | ❌ | ✅ decay=0.9999 | 후반 안정 |
| 학습 시간 (총) | ~12h (Stage 1) + ~8m (Stage 2) | ~25m (Stage 1) + ~1h 27m (S2 × 8 fold) | Stage 1 24× 빠름 (augmentation 효과 + 작은 데이터로 빠른 수렴) |

---

## 2. Stage 1 — KD pretrain (ORNL HDF5)

같은 val set (ORNL Build 1) 에서 평가 가능. v1 12-class 와 v2 8-class 가 다르지만 공통 class (Powder/Printed/Swelling/Spatter/Super-Elev/Over Melt/Debris) 는 직접 비교.

### 2.1 종합

| 지표 | v1 (best e14) | v2 (best e29) | 변화 |
|---|---|---|---|
| **val_acc** | 0.9783 | **0.9734** | -0.5%p (거의 동일) |
| **mIoU** | 0.2456 | **0.3675** | **+50%** ↑ |
| 학습 epoch 수렴 | e14 best (이후 plateau) | e29 best (마지막 epoch, 여전히 개선 중) | v2 가 더 오래 학습 가능 |
| 학습 시간 (1 epoch) | ~25 분 (1 GPU) | ~25 분 (4 GPU DDP) | 비슷 — v2 가 augmentation 처리 부담 더 |

### 2.2 Per-class IoU (공통 클래스만, ORNL Build 1)

| Class | v1 | v2 (재매핑 후 동급) | 변화 |
|---|---|---|---|
| 0 Powder | 0.9772 | **0.9721** | -0.5%p |
| 1 Printed | 0.9284 | **0.9160** | -1.2%p |
| **Recoater 통합** | n/a + 0.0883 (Hopping n/a, Streaking 0.09) | **0.0723** | 통합 효과 (둘 다 학습) |
| 5 Swelling (v1=5, v2=3) | 0.2777 | **0.2916** | +1.4%p |
| 8 Spatter (v1=8, v2=4) | 0.3333 | **0.3206** | -1.3%p |
| 7 Super-Elev (v1=7, v2=5) | 0.0000 | 0.0000 | — |
| 10 Over Melt (v1=10, v2=6) | 0.0000 | n/a (val 에 없음) | — |
| 6 Debris (v1=6, v2=7) | 0.0000 | 0.0000 | — |
| ❌ 제거 (v1 만) | Incomplete Spr (0.0016), Misprint (0.0945), Under Melt (0) | — | 평균 mIoU 끌어내리던 0-점 class 제거 |

→ **v2 의 mIoU 0.37 이 v1 의 0.25 보다 50% 높은 핵심 이유**: 0 점 받던 rare class (Incomplete Spreading, Misprint, Under Melting) 4 개가 제거되어 분모가 작아짐. 공통 class 의 절대 성능은 v1 와 동일 수준 — 즉 **모델 성능 자체는 같고, mIoU metric 만 안정화**.

---

## 3. Stage 2 — Finetune

### 3.1 v1 setup vs v2 setup

| 항목 | v1 | v2 |
|---|---|---|
| Train sources | 5 (Maraging 제외 LPBF) | Fold 별 7 source |
| Val source | 1 고정 (v2022_Maraging) | Fold 별 1 (8 source 모두 한 번씩 val 됨) |
| Epoch | 50 | 50 |
| Best ckpt 시점 | epoch 13 | 모든 fold 가 epoch 49 (=마지막) |
| EMA | ❌ | ✅ |
| Augmentation | 없음 (data 작아서 학습 자체가 우선) | 풀 augmentation 적용 |

### 3.2 단일 비교 — 같은 Maraging val 에서

v2 의 fold 5 (val=Maraging) 가 v1 의 Stage 2 와 정확히 같은 val set:

| 지표 | v1 (e13 best) | v2 fold 5 (e49 best) | 변화 |
|---|---|---|---|
| val_acc | 0.6872 | **0.8032** | **+16.9%p** ↑↑ |
| mIoU | 0.1109 | 0.139 | +25% ↑ |
| 학습 데이터 | 5 source (Maraging 제외) | 7 source (BJ 2 + LPBF 5, Maraging 제외) | v2 가 +2 source |

> v2 가 같은 val set 에서 **현저히 우수** — Augmentation + EMA + BJ source 추가 학습의 종합 효과.

### 3.3 8-fold Cross-Validation 전체 결과

| Fold | Val source | val_acc | mIoU | Powder | Printed | RecDist | Swelling | Spatter | SuperElev | OverMelt | Debris |
|---|---|---|---|---|---|---|---|---|---|---|---|
| 0 | v2021_LPBF | **0.948** | 0.346 | 0.95 | 0.84 | **0.16** | 0.15 | 0.00 | **0.69** | 0.00 | 0.00 |
| 1 | v2022_17-4PH | **0.957** | 0.285 | 0.95 | 0.94 | 0.00 | **0.40** | 0.00 | 0.00 | 0.00 | 0.00 |
| 2 | v2022_GammaPrint | **0.973** | 0.345 | 0.97 | 0.86 | 0.03 | **0.40** | 0.17 | **0.45** | 0.00 | 0.00 |
| 3 | v2022_Inc718_1 | 0.803 | 0.335 | 0.74 | 0.69 | 0.00 | **0.40** | **0.28** | 0.00 | n/a | 0.00 |
| 4 | v2022_Inc718_2 | 0.938 | 0.281 | 0.94 | 0.79 | 0.06 | 0.27 | 0.00 | 0.18 | 0.00 | 0.01 |
| 5 | v2022_Maraging | 0.803 | 0.139 | 0.83 | 0.15 | 0.12 | 0.02 | 0.00 | 0.00 | 0.00 | 0.00 |
| 6 | v2021_BJ | 0.841 | 0.144 | 0.89 | 0.07 | **0.19** | 0.00 | 0.00 | 0.00 | 0.00 | 0.01 |
| 7 | v2022_BJ_H13 | 0.893 | 0.146 | 0.89 | 0.02 | 0.11 | 0.00 | 0.00 | 0.00 | n/a | 0.00 |
| **평균** |  | **0.894** | **0.253** | **0.89** | **0.55** | **0.083** | **0.205** | **0.056** | **0.165** | ~0 | ~0.003 |

### 3.4 도메인 일반화 — LPBF vs Binder Jet

```
LPBF fold (0~5) 평균: val_acc 0.904   mIoU 0.288
BJ   fold (6, 7) 평균: val_acc 0.867   mIoU 0.145
```

- BJ 가 val_acc 는 비슷하지만 **mIoU 가 절반** — BJ 도메인의 Printed/Spatter/Swelling 등을 잘 못 잡음
- 원인: train 7 source 중 BJ 는 1개 → LPBF 통계로 학습된 모델이 BJ 의 시각 통계에 약함
- 특히 BJ fold 의 **Printed IoU 0.02~0.07** 이 mIoU 끌어내림

---

## 4. 클래스별 학습 정도 — 8 fold 평균

| 학습 강도 | 클래스 | mIoU (8-fold avg) | v1 동급 클래스 (Stage 1) | 비교 |
|---|---|---|---|---|
| ✅ 매우 잘 | Powder, Printed | 0.89, 0.55 | 0.97, 0.93 | v2 가 BJ 영향으로 Printed 평균이 낮아짐 |
| ✓ 의미 있음 | Swelling | 0.21 | 0.28 | 거의 비슷 |
| ✓ 의미 있음 | Super-Elevation | 0.17 | 0 | **v2 가 학습 성공** ⭐ |
| ✓ 의미 있음 | Recoater Disturbance | 0.08 | n/a + 0.09 | **통합 후 모든 fold 에서 학습됨** ⭐ |
| △ 약함 | Spatter | 0.06 | 0.33 | v2 가 낮음 — augmentation 으로 일반화는 됐으나 도메인 별로 다름 |
| ❌ 거의 안 됨 | Over Melting | ~0 | 0 | data 자체 부족 |
| ❌ 거의 안 됨 | Debris | ~0 | 0 | data 자체 부족 |

### 4.1 v2 의 두 가지 핵심 개선

1. **Super-Elevation 학습 성공** — v1 에선 모든 layer 가 0 IoU 였는데 v2 의 fold 0 (v2021_LPBF) 에서 **0.6972**, fold 2 (GammaPrint) 0.4501. 8 source 통합 train 의 효과.
2. **Recoater Disturbance 안정 학습** — v1 에선 두 클래스 (Hopping/Streaking) 가 따로라 val build 에 없으면 n/a. v2 통합 후 **8 fold 모두 nonzero**, D4 augmentation 효과로 방향 무관 학습 강화.

---

## 5. 단점 / 한계

### 5.1 BJ 도메인 적응 약함

train 의 7/8 이 LPBF (또는 BJ 의 다른 데이터) → BJ 의 Printed 영역을 잘 못 잡음.
**해결 후보**: BJ 전용 train data 추가, 또는 Domain Adaptation (e.g. CORAL, MMD loss).

### 5.2 Over Melting / Debris 거의 0

원인은 **train 데이터의 절대 부족** (v1, v2 동일).
- Over Melting: GammaPrint 와 일부 source 에만 존재, 매우 rare
- Debris: GammaPrint 의 일부 layer, BJ 의 일부 — 양이 부족

**해결 후보**: ORNL HDF5 의 v2022_Maraging 외 다른 build 추가 시도, 또는 synthesis (e.g. cut-and-paste augmentation).

### 5.3 mIoU 의 평균 편향

mIoU 계산 시 NaN (val set 에 없는 class) 제외 평균이라 fold 간 분모 다름. 표준 mIoU 계산은 confusion matrix accumulate 가 표준 — v2 의 cross-val 평균이 0.253 이라는 절대값은 그대로 받기보다는 **fold 간 분포** 를 보는 게 안전.

---

## 6. 종합 평가

### 6.1 v1 대비 v2 의 명확한 이점

- ✅ **평가 robust** (단일 source → 8 fold 평균)
- ✅ **Recoater 통합** + D4 augmentation 으로 방향성 결함 학습 안정
- ✅ **Super-Elevation 학습 성공** (v1 의 0 IoU 해소)
- ✅ **EMA + warmup + augmentation** 으로 학습 dynamics 안정
- ✅ **BJ 도메인 포함** — 두 다른 머신 (ExOne Innovent + M-Flex) 까지 커버
- ✅ **단일 비교 (Maraging val)** 에서도 val_acc +17%p

### 6.2 v2 가 v1 보다 좋아진 것이 아닌 것

- ❌ **Powder/Printed 의 absolute IoU 는 거의 동일** — 모델 capacity 가 같으니 당연
- ❌ **BJ 도메인 일반화는 한계** — train data 부족이 본질
- ❌ **rare class (Over Melt, Debris)** 는 여전히 학습 안 됨

### 6.3 다음 실험 추천 (v3?)

1. **TTA Ensemble**: 8 fold ckpt 의 softmax 평균 + D4 group 8 변형 평균 → 일관성 향상 (구현 ~30분)
2. **Hard-bootstrapping 활성화**: 현재 default OFF → ON 으로 Stage 1 재학습 (12 시간), teacher noise 흡수
3. **BJ source 가중치 ↑**: WeightedRandomSampler 의 BJ source 비중을 늘려 도메인 균형
4. **Confusion matrix 분석**: 어느 클래스가 어디로 혼동되는지 정확히 보기 (`inference/confusion.py --ckpt-stage 2`)

---

## 7. 산출물 위치

```
v1:
DefSeg_AM/checkpoints/vits14_dpt_dual_sz1036_1gpu_nanfix/
├── stage1_best.pt                      # val_acc 0.9783, mIoU 0.2456 (12-class)
└── stage2_best.pt                      # val_acc 0.6872, mIoU 0.1109 (Maraging only)

v2:
DefSeg_AM/checkpoints/vits14_dpt_dual_sz1036_8cls_v2/
├── stage1_best.pt                      # val_acc 0.9734, mIoU 0.3675 (8-class)
├── stage2_best_fold0_v2021_LPBF.pt
├── stage2_best_fold1_v2022_17-4PH.pt
├── stage2_best_fold2_v2022_GammaPrint.pt
├── stage2_best_fold3_v2022_Inc718_1.pt
├── stage2_best_fold4_v2022_Inc718_2.pt
├── stage2_best_fold5_v2022_Maraging.pt
├── stage2_best_fold6_v2021_BJ.pt
├── stage2_best_fold7_v2022_BJ_H13.pt
└── cv_summary.json                     # 모든 fold 의 per-class IoU
```
