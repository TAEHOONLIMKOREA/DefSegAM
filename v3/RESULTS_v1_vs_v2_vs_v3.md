# DefSeg-AM v1 vs v2 vs v3 — 학습 결과 비교

> 작성일: 2026-06-16
> v1 ckpt: `DefSeg_AM/checkpoints/vits14_dpt_dual_sz1036_1gpu_nanfix/`
> v2 ckpt: `DefSeg_AM/checkpoints/vits14_dpt_dual_sz1036_8cls_v2/`
> v3 ckpt: `DefSeg_AM/checkpoints/vits14_dpt_dual_sz1036_10cls_v3/`

본 문서는 [PLAN.md (v1)](../PLAN.md), [PLAN_v2.md (v2)](../v2/PLAN_v2.md), [PLAN_v3.md (v3)](PLAN_v3.md) 의 학습 결과를 비교한다. 모델 구조 (DINOv2 ViT-S/14 frozen + DPT decoder) 는 세 버전 모두 동일, **클래스 / 데이터 / 학습 정책** 만 변경.

세 버전이 서로 다른 native class 어휘를 갖기 때문에, 공정한 비교를 위해 **공통 8-class 어휘** 로 reduce 한 뒤 같은 평가셋 (ORNL Build 1) 에서 평가했다.

---

## 1. 한눈에 보는 비교

| 항목 | v1 | v2 | v3 |
|---|---|---|---|
| 클래스 수 | 12 (ORNL 표준) | 8 (Recoater 통합, 3개 제거) | **10** (Recoater 분리 복원, Incomplete Spreading 복원) |
| Stage 1 데이터 | ORNL Build 2-5 train, B1 val | 동일 (라벨만 8-class) | 동일 (라벨만 10-class) |
| Stage 2 정책 | LPBF 5 source, 1 source val | leave-one-source-out 8-fold | DSCNN 8 source 전수 train + ORNL B1 평가 |
| Stage 2 ckpt 개수 | 1 | 8 (fold 별) | **1** |
| Augmentation | brightness jitter 만 | D4 + cyclic + DSCNN noise/intensity + brightness | flip(LR+UD) + 180° + cyclic + DSCNN |
| EMA decay | ❌ | ✅ 0.9999 | ✅ 0.9999 |
| Hard-bootstrap (옵션) | ❌ | ✅ | ✅ |
| GPU 학습 | 1 GPU | 1 GPU | **4 GPU DDP** (lr sqrt-scaled) |
| 휴리스틱 후처리 | ❌ | ❌ | ✅ SE/Swelling far → Debris |
| Stage 1 총 학습 시간 | ~12 h | ~25 m | **~3 h 20 m** (4 GPU DDP) |
| Stage 2 총 학습 시간 | ~8 m | ~1 h 27 m (8 fold) | **~15 m** (50 epoch, K=4 replicate) |

---

## 2. 공통 평가 — ORNL Build 1 (200 layer 균등 샘플링)

각 모델로 동일한 200 layer 추론 후, 모든 prediction 을 공통 8-class 로 매핑하여 confusion 누적.

### 2.1 공통 클래스 정의 + 버전별 매핑

| 공통 ID | Name | v1 (native 12) | v2 (native 8) | v3 (native 10) |
|---:|---|---|---|---|
| 0 | Powder | 0 | 0 | 0 |
| 1 | Printed | 1 | 1 | 1 |
| 2 | Recoater Disturbance | 2 + 3 | 2 | 2 + 3 |
| 3 | Swelling | 5 | 3 | 5 |
| 4 | Debris | 6 | 7 | 6 |
| 5 | Super-Elevation | 7 | 5 | 7 |
| 6 | Spatter | 8 | 4 | 8 |
| 7 | Over Melting | 10 | 6 | 9 |

매핑되지 않는 native class (v1/v3 의 Incomplete Spreading, v1 의 Misprint·Under Melting) 는 IGNORE 처리. GT 는 ORNL HDF5 의 12-class boolean mask 의 argmax (작은 ID 부터 그려서 큰 ID 가 덮음) 를 동일하게 공통으로 매핑.

### 2.2 종합 mIoU 비교 (공통 8-class, support>0 만)

| Stage | v1 | v2 | v3 | 1위 |
|---|---:|---:|---:|---|
| Stage 1 (KD pretrain) | 0.5112 | 0.5055 | 0.5028 | **v1** |
| Stage 2 (finetune) | **0.5349** | 0.5246 | 0.5132 | **v1** |
| Stage 1 → Stage 2 향상 | +0.024 | +0.019 | +0.010 | v1 |

### 2.3 클래스별 IoU (Stage 2, 공통 8-class)

| Class | GT support (px) | v1 | v2 | v3 |
|---|---:|---:|---:|---:|
| 0 Powder | 632.8 M | 0.986 | 0.975 | 0.974 |
| 1 Printed | 34.8 M | 0.926 | 0.922 | 0.922 |
| 2 Recoater Disturbance | 165 K | 0.072 | 0.043 | 0.063 |
| 3 Swelling | 170 K | 0.243 | **0.297** | 0.263 |
| 4 Debris | 0 | n/a | n/a | n/a |
| 5 Super-Elevation | 0 | n/a | n/a | n/a |
| 6 Spatter | 6.2 M | **0.447** | 0.386 | 0.343 |
| 7 Over Melting | 0 | n/a | n/a | n/a |

> Build 1 의 결함 분포 한계로 Debris / Super-Elevation / Over Melting 의 평가가 사실상 불가능 (해당 GT 픽셀 0). 평가 가능한 5개 클래스 (Powder, Printed, Recoater Disturbance, Swelling, Spatter) 의 mIoU 가 §2.2 의 값.

### 2.4 클래스별 Stage 1 vs Stage 2 변화

각 버전 내에서 finetune 효과:

| Class | v1 (S1 → S2) | v2 (S1 → S2) | v3 (S1 → S2) |
|---|---|---|---|
| Powder | 0.978 → 0.986 | 0.972 → 0.975 | 0.972 → 0.974 |
| Printed | 0.936 → 0.926 | 0.926 → 0.922 | 0.924 → 0.922 |
| Recoater Dist. | 0.072 → 0.072 | 0.047 → 0.043 | 0.065 → 0.063 |
| Swelling | 0.218 → 0.243 | 0.243 → 0.297 | 0.232 → 0.263 |
| Spatter | 0.353 → 0.447 | 0.339 → 0.386 | 0.322 → 0.343 |

Stage 2 (DSCNN_Dataset finetune) 가 모든 버전에서 **Swelling/Spatter 만 의미 있게 향상**시킴. Recoater Disturbance 는 오히려 미세 감소 (Powder 로 오분류 증가).

---

## 3. 정성 비교 — 샘플 inference figure

Build 1 의 layer 8 개 (10%-90% 구간 균등) 에서 모든 버전의 prediction 을 한 figure 에 정렬. 모든 prediction 은 공통 8-class palette 로 시각화.

각 figure 의 panel 구성 (2 행 × 5 열):
- 행 0: visible/0 (after melt), visible/1 (after spread), GT (공통 8-class), v1 Stage 1, v1 Stage 2
- 행 1: v2 Stage 1, v2 Stage 2 (fold 5 = v2022_Maraging), v3 Stage 1, v3 Stage 2, (빈 공간 — legend)

샘플 layer:

| Layer | Figure |
|---:|---|
| 357 | [layer0357.png](../figures/comparison_v1_v2_v3/2021-07-13_TCR_Phase_1_Build_1/comparison/layer0357.png) |
| 765 | [layer0765.png](../figures/comparison_v1_v2_v3/2021-07-13_TCR_Phase_1_Build_1/comparison/layer0765.png) |
| 1174 | [layer1174.png](../figures/comparison_v1_v2_v3/2021-07-13_TCR_Phase_1_Build_1/comparison/layer1174.png) |
| 1582 | [layer1582.png](../figures/comparison_v1_v2_v3/2021-07-13_TCR_Phase_1_Build_1/comparison/layer1582.png) |
| 1991 | [layer1991.png](../figures/comparison_v1_v2_v3/2021-07-13_TCR_Phase_1_Build_1/comparison/layer1991.png) |
| 2399 | [layer2399.png](../figures/comparison_v1_v2_v3/2021-07-13_TCR_Phase_1_Build_1/comparison/layer2399.png) |
| 2808 | [layer2808.png](../figures/comparison_v1_v2_v3/2021-07-13_TCR_Phase_1_Build_1/comparison/layer2808.png) |
| 3217 | [layer3217.png](../figures/comparison_v1_v2_v3/2021-07-13_TCR_Phase_1_Build_1/comparison/layer3217.png) |

---

## 4. 해석

### 4.1 정량 — v1 이 가장 높은 이유

평가셋 (Build 1, 공통 8-class) 에서 v1 Stage 2 의 mIoU 0.5349 가 v2 (0.5246) 와 v3 (0.5132) 를 앞섬. 단순히 "v1 이 좋다" 는 결론은 부정확:

1. **결함 클래스 5종 중 3종 (Debris/SE/OverMelt) 의 GT 픽셀이 0** 이라 평가에서 제외. 사실상 Powder + Printed + Recoater + Swelling + Spatter 5 개의 평균.
2. **v1 stage 2 의 train 데이터 = v2022_Maraging 까지 포함된 LPBF 5 source** 인데, ORNL Build 1 의 분포가 그 source 와 가까움 → v1 이 in-distribution 우위.
3. **v2 stage 2 는 fold 5 (v2022_Maraging) ckpt** 만 사용 (다른 fold ckpt 의 평균 ensemble 은 본 비교에 포함 안 됨). ensemble 으로 평가하면 더 올라갈 가능성.
4. **v3 stage 2** 는 DSCNN 8 source (LPBF 6 + BJ 2) 전수 train. BJ source 분포가 ORNL 과 다르므로 ORNL Build 1 평가에선 손해. 반대로 BJ 도메인엔 v3 만 학습됨.

### 4.2 클래스별 우열 (Stage 2, support>0)

| 클래스 | 1위 | 의미 |
|---|---|---|
| Powder | v1 (0.986) | 거의 동등 — backbone 효과 |
| Printed | v1 (0.926) | 거의 동등 |
| Recoater Disturbance | v1 (0.072) | 셋 다 매우 낮음 — Build 1 에 거의 없음 |
| Swelling | **v2** (0.297) | D4 augmentation 의 raised feature 학습 효과 |
| Spatter | **v1** (0.447) | v1 의 단일 source finetune 이 sharp 한 spatter boundary 학습 |

### 4.3 클래스 불균형의 영향

ORNL Build 1 의 분포가 극도로 한쪽으로 치우쳐:
- Recoater Hopping: 0 픽셀
- Debris: 0 픽셀
- Over Melting: 0 픽셀
- Super-Elevation: 2 픽셀 (사실상 0)

→ 모델이 실제로 이 클래스들을 잘 학습했는지는 **train split (Build 2-5)** 의 confusion 으로 확인해야 함:

| Class | v3 Stage 1 train (Builds 2-5) IoU |
|---|---:|
| Super-Elevation | **0.901** |
| Spatter | 0.620 |
| Debris | 0.594 |
| Swelling | 0.527 |
| Over Melting | 0.490 |
| Recoater Streaking | 0.295 |

→ Build 2-5 에서는 v3 가 결함 클래스를 충분히 학습. **val (Build 1) 의 분포가 evaluation bias** 를 만듦.

### 4.4 학습 구조의 효율 — v3 의 정성적 우위

정량 mIoU 외의 측면에서 v3 의 명확한 강점:

| 측면 | v1 | v2 | v3 |
|---|---|---|---|
| Stage 1 학습 시간 | 12 h (1 GPU) | 25 m (rare class 누락) | **3 h 20 m** (4 GPU DDP, full class) |
| Stage 2 학습 시간 | 8 m | 1 h 27 m (8 fold) | **15 m** (1 ckpt) |
| Inference 단순도 | 단일 ckpt | 8 ckpt ensemble 필요 | 단일 ckpt |
| Augmentation 다양성 | 낮음 (brightness 만) | 높음 (D4+cyclic+DSCNN) | 높음 (LR+UD+rot180+cyclic+DSCNN) |
| BJ source 학습 | ❌ (제외) | ✅ | ✅ |
| 후처리 (휴리스틱) | ❌ | ❌ | ✅ |

특히 **v3 의 Stage 2 가 15 분에 단일 ckpt** 로 끝나는 것이 운영상 가장 큰 이점 — 재현/디버깅/inference deployment 가 v2 의 8-fold ensemble 보다 훨씬 단순.

---

## 5. 결론

### 정량 측면
- **Build 1 의 공통 8-class mIoU 1위는 v1 stage 2 (0.5349)**. v2 (0.5246), v3 (0.5132) 가 뒤따름. 
- 단, Build 1 의 결함 클래스 GT 가 매우 sparse (3종 0 픽셀) 라 mIoU 가 실제 모델 실력을 충분히 반영하지 못함.
- Train split (Build 2-5) 에서는 v3 의 Stage 1 train mIoU 가 0.53 으로, 모든 결함 클래스 학습이 확인됨.

### 운영 측면
- **v3 가 학습/평가/배포 모두 가장 효율적**: 4 GPU DDP, 단일 ckpt, 15분 finetune, 휴리스틱 후처리 옵션.
- v2 는 cross-validation 으로 정량 신뢰도 ↑ 하지만 ensemble inference 비용.
- v1 은 ORNL Build 1 에서 가장 높은 점수지만 BJ source / Stage 1 학습 시간 / class diversity 측면에서 한계.

### 권장
- **연구 보고용 정량 비교**: 같은 평가셋 (Build 1) 의 공통 8-class mIoU → v1 ≥ v2 ≥ v3 (소수점 차이)
- **실제 deployment**: v3 — 단일 ckpt + 4 GPU 학습 + 휴리스틱 후처리 통합
- **결함 다양성 학습**: v3 (10-class) > v2 (8-class) > v1 (12-class 이지만 일부 학습 안 됨)
- **다음 ablation 후보**: ① ORNL Build 1 외 다른 build (예: B2, B3) 로 동일 비교, ② v2 의 8-fold ensemble vs v3 single 비교, ③ v3 의 휴리스틱 후처리 on/off mIoU 영향.

---

## Appendix — 재현

```bash
# 공통 8-class 비교 inference (정성 + 정량)
PYTHONPATH=. ./DefSeg_AM/venv/bin/python -m DefSeg_AM.scripts.compare_v1_v2_v3 \
    --build "2021-07-13 TCR Phase 1 Build 1" \
    --n-layers 8 \
    --n-confusion-layers 200
```

산출물:
- `DefSeg_AM/figures/comparison_v1_v2_v3/<build>/comparison/layer*.png` — 정성 비교 figure 8개
- `DefSeg_AM/figures/comparison_v1_v2_v3/<build>/comparison_metrics.json` — confusion matrix + per-class IoU + mIoU
