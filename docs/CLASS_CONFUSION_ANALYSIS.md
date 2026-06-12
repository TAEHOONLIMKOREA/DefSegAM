# 결함 클래스 혼동 / 병합 분석 (Confusion Matrix)

> **질문**: "클래스 종류를 줄이면(병합하면) 성능이 오를까? 어떤 결함이 서로 헷갈리나?"
>
> **결론(미리)**: 혼동은 *결함↔결함*이 아니라 대부분 *결함→배경(Powder/Printed)* 이고,
> 클래스들은 train에서 이미 잘 학습된다(IoU 0.4~0.9). val mIoU가 낮은 주원인은
> "헷갈리는 결함"이 아니라 **(a) val(Build 1)에 절반의 결함 GT가 없음 + (b) train→val 일반화 격차**.
> → 병합으로 얻는 mIoU 상승은 대부분 착시. 실제 액션은 §5 참조.

분석 대상 체크포인트: `vits14_dpt_dual_sz1036_1gpu_nanfix`
(stage1 best: epoch 14, val_acc 0.978, mIoU 0.246 / stage2 best: epoch 13, val_acc 0.687)

---

## 1. 방법

[inference/confusion.py](../inference/confusion.py) — 학습 없이 val/train 1-pass로
12×12 confusion matrix 누적 → per-class IoU/precision/recall + 병합후보 pair 산출.

```bash
# 사용 (scripts/run_confusion.sh 또는 직접)
STAGE=1 bash DefSeg_AM/scripts/run_confusion.sh                    # stage1 val (Build 1)
CUDA_VISIBLE_DEVICES=1 python -m DefSeg_AM.inference.confusion \
    --run-name vits14_dpt_dual_sz1036_1gpu_nanfix --stage 1 --split train   # stage1 train (B2-5)
python -m DefSeg_AM.inference.confusion --run-name <run> --stage 2          # stage2 (Maraging GT)
```

confusion matrix: **행=GT, 열=Pred**, IGNORE(-1) 제외.
저장물: `confusion_counts.npy`, `confusion_matrix.csv`, `per_class_metrics.csv`,
`confusion_row_normalized.png` (heatmap).

---

## 2. 3개 평가 한눈에

| 평가 | 데이터 | GT 출처 | 픽셀 | mIoU |
|---|---|---|---:|---:|
| **Stage1 val** | ORNL Build 1 | DSCNN pred | 34.5억 | 0.338 |
| **Stage1 train** | ORNL Build 2–5 | DSCNN pred | 134.8억 | **0.576** |
| **Stage2 val** | DSCNN_Dataset Maraging | 사람 GT | 0.27억 | 0.186 |

> 같은 모델인데 train mIoU(0.58) ≫ val mIoU(0.34/0.25). = **일반화 격차**가 핵심.

---

## 3. Stage 1 — train(B2-5) vs val(Build 1) per-class IoU

train은 12클래스를 거의 다 포함 → 12×12 완성. val은 절반이 GT 0.

| # | 클래스 | GT(train) | **IoU train** | IoU val | recall(train) | 주 혼동(train) |
|---|---|---:|---:|---:|---:|---|
| 0 | Powder | 120.8억 | 0.950 | 0.977 | 0.95 | →Spatter |
| 8 | Spatter | 6.38억 | 0.669 | 0.333 | 0.99 | →Printed |
| 1 | Printed | 5.39억 | 0.910 | 0.928 | 0.99 | →Swelling |
| 3 | Recoater Streaking | 1.86억 | 0.416 | 0.088 | 0.99 | →Powder |
| 11 | Under Melting | 2,829만 | **0.917** | **0 (val GT 없음)** | 0.98 | →Printed |
| 5 | Swelling | 910만 | 0.568 | 0.278 | 0.76 | →Printed(0.23) |
| 6 | Debris | 176만 | **0.700** | **0 (val GT 없음)** | 0.93 | →Printed |
| 4 | Incomplete Spreading | 54만 | 0.078 | 0.002 | 0.93 | →Powder |
| 10 | Over Melting | 39만 | **0.617** | **0 (val GT 없음)** | 0.89 | →Printed |
| 7 | Super-Elevation | 27만 | **0.851** | **0 (val GT 없음)** | 0.86 | →Printed |
| 9 | Misprint | 9.8만 | 0.242 | 0.084 | 0.54 | **→Spatter(0.40)** |
| 2 | Recoater Hopping | **25** | 0.000 | n/a | 0.00 | (사실상 없음) |

**train mIoU = 0.576 / val mIoU = 0.245**

---

## 4. Stage 2 — Maraging 사람 GT (참고, 소규모 26 layers)

| # | 클래스 | GT | IoU | recall | 주 혼동 |
|---|---|---:|---:|---:|---|
| 0 | Powder | 1,491만 | 0.713 | 0.96 | →IncSpread |
| 4 | Incomplete Spreading | 941만(35%) | 0.412 | 0.43 | →Powder(0.33) |
| 2 | Recoater Hopping | 140만 | 0.000 | 0.00 | →Powder(0.99) |
| 1 | Printed | 108만 | 0.060 | 0.08 | →Powder(0.58) |
| 3 | Recoater Streaking | 24,547 | 0.092 | 0.65 | →Powder |
| 6 | Debris | 12,934 | 0.001 | 0.04 | →Powder |
| 5 | Swelling | 8,678 | 0.025 | 0.05 | →Printed |
| 7,8,9,10,11 | (나머지) | 0 | — | — | Maraging GT 없음 |

> Stage2는 데이터가 작고 도메인 시프트(Printed recall 0.08)가 심해 per-class가 noisy.
> 단, 여기서도 top-confusion은 거의 전부 **→Powder** (Stage1과 동일 패턴).

---

## 5. 핵심 발견 & 권고

### 발견
1. **혼동은 결함↔결함이 아니라 결함→배경(Powder/Printed)** — 세 평가 모두 일관. 따라서 "비슷한 결함 병합"은 이 문제를 못 고침.
2. **클래스들은 학습 가능** — train에서 Under Melting 0.92, Super-Elevation 0.85, Debris 0.70, Spatter 0.67, Over Melting 0.62, Swelling 0.57. val IoU 0은 "구분 불가"가 아니라 **val에 GT가 없거나(절반) 일반화 실패**.
3. **진짜 병목 = 일반화 격차** (train 0.58 → val 0.25), 클래스 taxonomy 아님.
4. **val split이 결함을 대표하지 못함** — Build 1 하나로는 Debris·Recoater Hopping·Over/Under Melting·Super-Elevation을 **평가조차 못 함**.

### "클래스 줄이면 성능 오르나"에 대한 답
- **숫자는 오르지만 대부분 착시.** val mIoU를 끌어내린 주범은 헷갈리는 결함이 아니라 **val에 없는 클래스**. 빼면 평균이 뛰지만 모델은 그대로.
- 데이터가 지지하는 **실제 액션**:
  - **Recoater Hopping(2) 제외** — train 전체에 픽셀 25개. 병합이 아니라 Stage1에서 drop (없는 클래스).
  - **Misprint(9) → Spatter(8) 병합 검토** — 유일하게 뚜렷한 결함↔결함 혼동(40%) + 데이터 희소(9.8만).
  - **본질 개선은 일반화** — val에 모든 결함이 포함되도록 split 재구성, overfitting 완화(정규화/증강), 그리고 결함 vs 배경 recall(Powder down-weight·Dice loss·oversampling 강화).
- 과제 자체를 단순화하려면 **"결함 vs 배경" 이진 / 물리적 coarse 그룹**이 정직한 축소 (실패 지점인 미검출을 직접 겨냥).

---

## 6. 생성물

```
figures/vits14_dpt_dual_sz1036_1gpu_nanfix/
├── stage1/confusion/          # val (Build 1)
├── stage1/confusion_train/    # train (B2-5) — 12클래스 완성
└── stage2/confusion/          # Maraging 사람 GT
        ├── confusion_counts.npy
        ├── confusion_matrix.csv
        ├── per_class_metrics.csv
        └── confusion_row_normalized.png
```

스크립트: [inference/confusion.py](../inference/confusion.py),
래퍼: [scripts/run_confusion.sh](../scripts/run_confusion.sh) (`STAGE`, `RUN_NAME`, `--split` 지원).
