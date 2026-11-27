# DefSeg-AM **v5** 실험 계획 — v4 11-class 그대로 · 사람 GT 전용 학습

> v4 ([PLAN_v4.md](../v4/PLAN_v4.md)) 의 후속. 작성일 2026-11-16.
> 상태: **구현 완료, `--quick` 스모크 통과. 풀런은 아직 돌리지 않았다** (사용자가 compose 로 실행 — [docker/README.md](docker/README.md)).
>
> 핵심 변경:
> - **DSCNN 예측을 학습에 쓰지 않는다**: Stage 1 KD 가 없고, v1–v4 ckpt 로 초기화하지 않으며, 길이와 ckpt 도 사람 GT 로 정한다
> - **학습 데이터 = 사람 GT 뿐**: `DSCNN_Dataset` 의 사람 주석 8 소스 125 레이어 (v4 Stage 2 와 같은 데이터 · 같은 매핑)
> - **단일 단계 학습**: decoder 를 무작위 초기값에서 한 번에 학습한다. 길이는 epoch 이 아니라 step 으로 정한다
> - **검증을 사람 GT 로**: 소스 하나를 통째로 떼어 두는 fold 로 길이를 정하고 일반화 성능을 본다
>
> 바꾸지 않는 것: **클래스 11개와 번호**, native → 11-class 매핑, backbone (`dinov2_vitb14_reg`), 입력 크기 1036², 증강, Copy-Paste, 손실, 후처리, ckpt 형식.
> 클래스 체계를 DIVA 의 `pbf-2026.11` 로 바꾸는 것은 v6 ([PLAN_v6.md](../v6/PLAN_v6.md)) 이고, v6 는 v5 의 학습 방식 위에 선다.

v5 가 답하려는 질문은 셋이다.

1. Stage 1 없이 사람 GT 125 레이어만으로 decoder 가 학습되는가.
2. 그러려면 몇 step 이 필요한가.
3. v4 와 견주어 클래스별로 무엇을 얻고 무엇을 잃는가.

**용어** (이 문서에서 처음 쓰는 것만)

| 말 | 뜻 |
|---|---|
| step | 이미지 한 묶음 (16장 = GPU 4대 × 4장) 을 보고 가중치를 한 번 고치는 것. 학습 길이를 이 횟수로 센다 |
| epoch | 데이터를 한 바퀴 도는 것. 데이터가 125 레이어면 한 epoch 이 8 step 도 안 되므로 v5 는 epoch 으로 길이를 세지 않는다 |
| EMA | 학습용 가중치와 따로 들고 가는 평균 사본. 매 step 뒤 학습용 가중치 쪽으로 조금씩 따라가고, 검증 · 저장 · 배포에는 이 사본을 쓴다 (§5.3) |
| fold | 사람 GT 의 일부를 검증용으로 떼어 두고 나머지로 학습하는 한 번의 실행 |
| leave-one-source-out (loso) | 소스 (장비 · 재료) 하나를 통째로 떼어 두는 fold. 본 적 없는 장비에서의 성능을 잰다 |
| sampler | 학습 때 어느 레이어를 얼마나 자주 뽑을지 정하는 규칙 |
| 결함 mIoU | 결함 클래스 (2–10) 의 IoU 평균. 그 fold 에서 학습 화소와 검증 화소가 모두 있는 클래스만 넣는다 |

---

## 0. 전제 — "정답 데이터만" 의 범위

| 자원 | v4 에서의 쓰임 | v5 |
|---|---|---|
| ORNL HDF5 `slices/segmentation_results` (DSCNN 예측 mask) | Stage 1 라벨, Stage 2 ckpt 선정 (Build 1 mIoU), Stage 1 Copy-Paste supplier, sampler 통계 | **학습 · 길이 결정 · ckpt 선정 어디에도 쓰지 않는다.** 학습이 끝난 뒤 참고 평가 (§6.4) 에서만 읽는다 |
| v1–v4 checkpoint | Stage 2 초기화 | 초기화에 쓰지 않는다. v4 ckpt 는 같은 표본에서의 비교 채점 (§6.3) 에만 쓴다 |
| `ORNL_Data/DSCNN_Dataset/**/annotations/*.npy` (사람 주석) | Stage 2 GT | **유일한 학습 데이터** |
| DINOv2 backbone | frozen | frozen (자기지도 사전학습, DSCNN 과 무관) |

- `DSCNN_Dataset` 은 폴더 이름과 달리 DSCNN 의 출력이 아니라 DSCNN 을 학습시키는 데 쓴 **사람 주석**이다.
  v4 클래스 체계로 된 사람 GT 는 이것뿐이다 (DIVA 레이블은 새 체계라 v6 에서 쓴다).
- **GT 만으로 decoder 를 처음부터 학습한 결과는 지금 없다.** v1–v4 는 모두 Stage 1 에서 출발했다.
  v4 의 `stage2_best.pt` 도 EMA (decay 0.9999) 를 1,550 step 만 돌린 평균이라, 가중치의 약 86% 가 Stage 1 에서 넘어온 초기값이다
  (0.9999^1549 ≈ 0.857). 두 ckpt 의 가중치를 직접 비교해도 차이는 상대 크기로 2.8% 다 (부록 B). 지금 서빙 중인 v4 가 이 파일이다.
  즉 v4 는 "GT 로 마무리한 모델" 이라기보다 "DSCNN 예측으로 학습한 모델을 조금 움직인 것" 이고, v5 가 GT 전용 학습의 첫 결과가 된다.

## 1. v4 → v5 변경 요약

| 영역 | v4 | v5 |
|---|---|---|
| 클래스 | 11 | **같음** (이름 · 번호 · native 매핑 모두) |
| 학습 단계 | Stage 1 KD (ORNL 의사 라벨 12,607 레이어) → Stage 2 GT (125 레이어) | **단일 단계, 사람 GT 125 레이어만** |
| decoder 초기값 | Stage 2 ← Stage 1 ckpt | 무작위 |
| 학습 길이 | S1 30 epoch (23,610 step) + S2 50 epoch (1,550 step) | 10,000 step 상한, 검증 fold 로 확정 (§5.2) |
| EMA decay | 0.9999 | 0.999 (§5.3) |
| Warmup | S2 50 step | 300 step (무작위 초기값) |
| Sampler | 소스 균등 × class-aware 의 곱, replicate K = 4 | 소스 균등과 클래스 균등의 혼합 + 표본당 상한, K 없음 (§5.4) |
| 화소 수 통계 | native 해상도 | 학습 해상도 (1036²) |
| 검증 · ckpt 선정 | ORNL Build 1 200 레이어, DSCNN 예측과의 mIoU | 떼어 둔 사람 GT 의 결함 mIoU (검증 fold). 전수 학습 모델은 고정 길이의 마지막 EMA |
| 데이터 읽기 | 표본마다 tif 를 다시 열어 줄임 | 시작할 때 한 번 줄여 RAM 에 둠 (같은 방법: 이미지 BILINEAR, 라벨 NEAREST) |
| Backbone / decoder / 입력 | `dinov2_vitb14_reg` + DPT (trainable 14.56 M), 1036² | 같음 |
| 증강 · Copy-Paste · 손실 | v4 Stage 2 | 같음 (Copy-Paste supplier 는 학습 분할의 GT) |
| 후처리 | 2 규칙 | 같음 |
| 저장 | `stage2_best.pt` (EMA 사본만) | `last.pt` (EMA) + `last_raw.pt` (학습용 가중치) |

---

## 2. 클래스 (11개, v4 와 같음) 와 GT 의 양

번호 · 이름 · 매핑은 [config_v5.py](config_v5.py) 에 v4 값을 그대로 옮겼다 (native → ORNL 12 → 11-class, Misprint 만 IGNORE).
아래는 학습 해상도 (1036²) 에서 센 화소 수다. 가운데 열은 v4 의 Stage 1 이 ORNL Build 2–5 에서 본 DSCNN 예측 라벨의 양이다.

| id | 클래스 | 약어 | 사람 GT (px / 레이어 / 소스) | v4 Stage 1 의 의사 라벨 (px) | v5 에서의 전망 |
|---:|---|---|---|---:|---|
| 0 | Powder | Pow | 98.1 M / 125 / 8 | 12,082 M | 충분 |
| 1 | Printed | Prt | 13.8 M / 124 / 8 | 539 M | 충분 |
| 2 | Recoater Hopping | RH | 1.40 M / 25 / **1** (MS) | 25 | v4 보다 나아질 여지가 크다. v4 는 이 클래스를 Stage 2 에서만 봤다 |
| 3 | Recoater Streaking | RS | 291 K / 37 / 6 | 185.9 M | GT 가 의사 라벨의 1/638 |
| 4 | Incomplete Spreading | IS | 12.5 M / 45 / 6 | 0.54 M | v4 보다 나아질 여지가 크다 (v4 의 Build 1 IoU 는 0) |
| 5 | Swelling | Sw | 90.9 K / 75 / 6 | 9.10 M | 레이어는 많지만 화소가 적다 |
| 6 | Debris | DB | 269 K / 43 / 5 | 1.76 M | 보통 |
| 7 | Super-Elevation | SE | 259 K / 10 / 3 | 0.27 M | v4 와 비슷한 양. 레이어가 10개뿐 (L21 6, GP 2, I2 2) |
| 8 | Spatter | SP | 372 K / 11 / 2 (GP, I1) | 638 M | GT 가 의사 라벨의 1/1,713. 소스가 둘뿐 |
| 9 | Over Melting | OM | 24.8 K / 11 / **1** (I2) | 0.39 M | 적다 |
| 10 | Under Melting | UM | **1.2 K / 2 / 1** (GP) | 28.3 M | **사실상 학습되지 않는다** |

- v4 가 Spatter · RS · Swelling · Under Melting 을 아는 것은 거의 전부 DSCNN 예측에서 왔다. v5 는 그 부분을 버린다.
  특히 Under Melting 은 GT 가 2 레이어 1,196 화소뿐이라, v5 는 출력 채널 10 을 갖고는 있어도 거의 내지 않을 것이다.
- 반대로 Recoater Hopping · Incomplete Spreading 은 사람 GT 에만 있다. v4 의 저장본에는 Stage 2 의 학습이 많아야 14% 반영됐으므로 (§0)
  이 두 클래스는 v5 에서 처음으로 제대로 학습된다.
- 매핑은 v4 그대로다. 따라서 v4 가 IGNORE 로 둔 Soot (L21 · 17-4 · I2 · MS) 는 v5 에서도 IGNORE 이고, 기존 IS 주석의 성격이 소스마다 다른 문제도
  그대로 물려받는다. 이 둘을 고치는 것은 v6 의 범위다 ([PLAN_v6.md](../v6/PLAN_v6.md) §2.3–2.4).

---

## 3. 데이터 — 사람 GT 8 소스 125 레이어

| 소스 (약어) | 장비 | 레이어 | 해상도 (H×W) | 결함 화소 (native) |
|---|---|---:|---|---|
| `v2021_LPBF` (L21) | ConceptLaser M2 | 20 | 2325² | IS 3.72 M, SE 963 K, DB 492 K, RS 262 K, Sw 143 K |
| `v2021_BJ` (BJ21) | Binder Jet | 20 | 4731–4876 × 1922–1981 | IS 7.89 M, RS 901 K, DB 209 K |
| `v2022_17-4PH` (17-4) | EOS M290 | 14 | 766² | IS 70 K, Sw 1.2 K |
| `v2022_GammaPrint` (GP) | EOS M290 | 9 | 1997² | SE 234 K, SP 173 K, IS 71 K, Sw 46 K, UM 4.5 K, RS 3.6 K |
| `v2022_Inc718_1` (I1) | EOS M290 | 5 | 1997² | SP 1.21 M, Sw 43 K |
| `v2022_Inc718_2` (I2) | EOS M290 | 24 | 764² | DB 72 K, RS 52 K, Sw 15 K, OM 13.5 K, SE 2.4 K |
| `v2022_Maraging` (MS) | AddUp FormUp 350 | 26 | 1834² | IS 29.5 M, RH 4.39 M, RS 79 K, DB 41 K, Sw 27 K |
| `v2022_BJ_H13` (H13) | ExOne M-Flex (Binder Jet) | 7 | 4656×2910 | IS 16.1 M, RS 153 K, DB 22 K |

- 시작할 때 125 레이어를 한 번 읽어 1036² 로 줄여 RAM 에 둔다 (`GTPool`, 프로세스당 약 0.4 GB, 20초 안팎). 줄이는 방법은 v4 와 같다.
  한 번 학습에 16 만 표본을 뽑으므로 표본마다 tif 를 다시 여는 v4 방식은 쓰지 않는다. 캐시 파일은 만들지 않는다.
- NEAREST 축소는 클래스별 면적비를 0.995–1.001 로 보존한다 (2026-11-16, 125 레이어 전수).
- sampler 와 class weight 가 쓰는 화소 수는 **학습 해상도 기준**이다. v4 는 native 해상도로 셌는데, 그러면 원본이 큰 소스 (BJ21 · H13) 가
  통계를 지배한다. 모델이 실제로 보는 것은 1036² 로 줄인 라벨이다.

---

## 4. 검증 설계 — 사람 GT 를 떼어 두는 fold

v4 는 ORNL Build 1 에서 DSCNN 예측과의 mIoU 로 ckpt 를 골랐다. v5 는 그것을 쓸 수 없으므로 사람 GT 의 일부를 떼어 검증한다.
fold 는 [config_v5.py](config_v5.py) 의 `CV_FOLDS` 에 정의되어 있고, 기본으로 도는 것은 다섯 개다.

| fold | 검증용으로 빼는 것 | 학습 / 검증 레이어 | 채점되는 결함 클래스 | 채점에서 빠지는 클래스 |
|---|---|---|---|---|
| `loso_L21` | L21 전체 | 105 / 20 | RS IS Sw DB SE | — |
| `loso_MS` | MS 전체 | 99 / 26 | RS IS Sw DB | RH (MS 에만 있다) |
| `loso_I2` | I2 전체 | 101 / 24 | RS Sw DB SE | OM (I2 에만 있다) |
| `loso_GP` | GP 전체 | 116 / 9 | RS IS Sw SE SP | UM (GP 에만 있다) |
| `inlayer` | 소스마다 레이어 5개 중 1개 | 103 / 22 | 결함 9종 전부 | — |

- **loso 4개**는 둘 이상의 소스에 있는 결함 6종 (RS · IS · Sw · DB · SE · SP) 을 한 번 이상 채점한다. 나머지 4 소스의 fold
  (`loso_BJ21`, `loso_17-4`, `loso_I1`, `loso_H13`) 도 정의되어 있고 `DEFSEG_V5_FOLDS` 로 고를 수 있다.
- **loso 의 구조적 한계**: 한 소스에만 있는 클래스 (RH, OM, UM) 는 그 소스를 빼면 학습 화소가 0 이라 채점할 수 없다. 이 셋은 loso 로는 영영 잴 수 없다.
- **`inlayer`** 는 그 빈자리를 메운다. 소스마다 이름순으로 5개 중 1개를 빼고, 결함 클래스마다 검증과 학습 양쪽에 레이어가 하나 이상 남도록 보정한다
  (결정적이다). 다만 같은 빌드의 이웃 레이어가 학습에 있으므로 **낙관적인 값**이다. "이 클래스가 배워지기는 하는가" 를 보는 용도다.
- 결함 mIoU 는 그 fold 에서 학습 화소와 검증 화소가 모두 있는 결함 클래스만 평균한다. fold 마다 클래스 구성이 다르므로 fold 끼리 값을 직접 비교하지 않는다.
- **전수 학습 모델에는 검증이 없다.** 125 레이어를 다 쓰기 때문이다. fold 에서 정한 길이만큼 돌리고 마지막 EMA 사본을 저장한다 (cosine 일정이 learning rate 를 0 까지 내린다).
  ckpt 를 고르는 단계가 없으므로 DSCNN 예측이 끼어들 틈도 없다.

---

## 5. 학습 설계

### 5.1 모델

`DefSegModel(backbone_name="dinov2_vitb14_reg", n_classes=11)` — v4 와 같은 구조 (frozen backbone ×2 → per-stage `concat(f0, f1, f1−f0)` →
DPT decoder), trainable 14.56 M. decoder 는 무작위 초기값에서 시작한다. 학습 스크립트에 `--init-from` 이 없다.

### 5.2 길이와 스케줄 — step 기준

- effective batch 16 (4 GPU × 4), AdamW, lr 2.83e-4, weight decay 1e-4, grad clip 1.0, AMP 끔 (v4 와 같음).
- warmup 300 step (learning rate 를 0 에서 서서히 올림) → cosine (끝에서 0 이 되도록 서서히 내림).
- `TOTAL_STEPS = 10,000` 은 **상한**이다 (v4 Stage 2 의 6.5배). 적정값의 근거가 없으므로 검증 fold 의 곡선을 보고 정한다.
  `summarize_cv` 가 loso fold 들 (둘 이상) 의 평균 곡선에서 최고점 P 를 찾는다. 최고점이 마지막 10% 구간에 있거나, 끝 값이 최고점의 95% 이상이면
  (최고점 뒤로 거의 내려가지 않았으면) 줄이지 않는다. 그 밖에는 `1.5 × P` 를 제안한다 — cosine 일정이 끝까지 내려갈 여유를 둔 어림값이다.
  길이를 바꾸면 검증 fold 도 모두 그 길이로 다시 돌려야 표와 곡선이 그 길이의 결과가 된다 (fold 폴더가 길이별로 나뉘지 않는다).
- 250 step 마다 검증한다. rank 0 혼자 검증 표본 전체를 한 번씩 돈다 (표본이 9–26 레이어뿐이라 `DistributedSampler` 의 패딩 중복이 지표를 흔든다).

### 5.3 EMA

EMA 사본은 매 step 뒤 `사본 ← decay × 사본 + (1 − decay) × 지금 가중치` 로 갱신한다. 사본은 학습을 시작할 때의 가중치에서 출발하며,
n step 뒤 그 출발값이 사본에 남아 있는 비율은 decay^n 이다. 1 / (1 − decay) 를 시상수라 한다.

| decay (시상수) | 1,550 step 뒤 남는 출발값 | 10,000 step 뒤 남는 출발값 |
|---|---|---|
| 0.9999 (10,000 step) — v4 | 86% | 37% |
| 0.999 (1,000 step) — v5 | 21% | 0.005% |

v4 Stage 2 는 0.9999 · 1,550 step 칸이고, 출발값은 Stage 1 결과였다. v5 의 출발값은 무작위 값이므로 남는 만큼 그대로 손실이다.

- decay 0.999. 규칙: **시상수 ≤ 총 step 의 1/10**. 길이를 줄이면 decay 도 함께 낮춘다 (예: 6,000 step → 0.998). 규칙을 어기면 학습 스크립트가 경고를 낸다.
- 매 검증에서 EMA 사본과 학습용 가중치를 **둘 다** 채점해 `curve.csv` 에 남긴다. 사본이 뒤처지면 두 곡선이 벌어져 바로 보인다.
- 전수 학습은 `last.pt` (EMA) 와 `last_raw.pt` (학습용 가중치) 를 둘 다 저장한다. v4 는 학습용 가중치를 남기지 않아 나중에 확인할 수 없었다.
- EMA 사본은 rank 0 만 갖는다 (DDP 의 모든 rank 는 같은 가중치를 가지므로 하나면 된다).

### 5.4 Sampler

```
u_i = 1 / (소스 수 × 그 소스의 레이어 수)                    # 소스 균등, 합 = 1
c_i = (1 / 클래스 수) × Σ_c pix[i, c] / total_pix[c]        # 클래스 균등, 합 = 1  (화소가 0 인 클래스는 뺀다)
p_i = 0.5 × u_i + 0.5 × c_i
p_i ← min(p_i, 8 / N) 후 남은 몫을 나머지 레이어에 비례 배분   # 한 레이어가 균등 추출의 8배를 넘지 않게
```

`c_i` 는 v4 의 class-aware weight (α = 1.0) 와 같은 양이다. 달라진 것은 `u_i` 와 **곱하지 않고 더한다**는 점과 상한이다.

| 125 레이어에서 (2026-11-16 측정) | v4 의 식 (곱) | v5 의 식 (혼합 + 상한) |
|---|---|---|
| 가장 많이 뽑히는 5 레이어의 몫 | 39% | 17% |
| 가장 적게 뽑히는 소스 | 17-4PH 0.3% (14 레이어) | H13 7.4% |
| 유효 표본 수 (1 / Σp²) | 25 | 75 |

- v4 의 곱은 드문 클래스가 몰린 몇 레이어 (GP 의 Under Melting 2장, I1 의 Spatter 3장) 에 추출을 몰아주고, 결함이 적은 소스는 거의 뽑지 않는다.
  v4 에서는 Stage 2 가 짧고 저장본에 많아야 14% 반영돼 드러나지 않았다. v5 는 이 125 레이어가 전부이므로 고쳤다.
  결함이 적은 레이어도 "정상 분말 · 정상 파트가 어떻게 보이는가" 를 가르치는 표본이다.
- v4 의 식은 화소가 0 인 클래스가 있으면 그 클래스의 weight (1 / eps) 가 정규화를 지배해 전체가 균등 추출로 무너진다.
  v4 에서는 생기지 않았지만 검증 fold 에서는 생기므로 (소스를 빼면 그 소스에만 있는 클래스가 0 이 된다) 화소가 0 인 클래스를 계산에서 뺐다.
- replicate factor K 는 없앴다. 복원 추출이고 길이를 step 으로 정하므로 필요 없다.
- 혼합 비율 0.5 와 상한 8 은 `config_v5.py` 의 `SAMPLER_SOURCE_MIX` · `SAMPLER_MAX_REPEAT` 다. 근거는 위 표의 분포이지 실험 결과가 아니다.

### 5.5 증강과 Copy-Paste

v4 Stage 2 와 같다: Copy-Paste (prob 0.5, 대상 클래스 RH · IS · Sw · DB · SE · OM · UM, 객체 최대 3개, feather σ 5, bbox 상한 0.3, 최소 30 px) →
cyclic shift → flip LR / UD → 180° 회전 → Gaussian noise · intensity shift → brightness jitter.
Copy-Paste 의 supplier 는 **학습 분할의 레이어만**이다 (검증용으로 뺀 레이어는 들어가지 않는다).
증강 난수는 (seed, rank, worker) 로 고정했다 (v4 는 매번 달랐다).

### 5.6 손실

`cross_entropy` + sqrt-inv class weight (clip 10), `ignore_index = -1` — v4 Stage 2 와 같다. v4 의 hard-bootstrap 은 의사 라벨용이었으므로 뺐다.
전수 학습의 class weight 는 Pow 0.54, Prt 2.86, RH 9.48, IS 3.03, 나머지 10 (상한) 이다.
학습 화소가 0 인 클래스는 weight 가 0 이다. ckpt 의 `config.class_train_px` 에 클래스별 학습 화소 수를 남긴다.

### 5.7 값 요약

| 항목 | 값 | 비고 |
|---|---|---|
| 클래스 수 | 11 | v4 동일 |
| Backbone / 입력 | `dinov2_vitb14_reg` frozen / 1036² | v4 동일 |
| decoder 초기값 | 무작위 | §5.1 |
| Batch | GPU 당 4 × 4 GPU = 16 | v4 동일. GPU 당 23.6 GB (측정) |
| LR / WD / warmup | 2.83e-4 / 1e-4 / 300 step, cosine | §5.2 |
| `TOTAL_STEPS` | 10,000 (상한, 검증 fold 로 확정) | §5.2 |
| 검증 주기 | 250 step | §5.2 |
| EMA decay | 0.999 | §5.3 |
| Sampler | 소스 균등 0.5 + 클래스 균등 0.5, 표본당 상한 8× | §5.4 |
| Copy-Paste | `CP_PROB` 0.5, v4 Stage 2 방식 | §5.5 |
| 손실 | CE + sqrt-inv class weight (clip 10) | §5.6 |
| Seed | 42 | |

---

## 6. 평가

### 6.1 검증 fold 의 곡선과 요약

fold 학습은 250 step 마다 떼어 둔 사람 GT 를 1036² 에서 채점해 `figures/<run>/v5/cv/<fold>/curve.csv` 에 남긴다 (학습용 가중치 · EMA 사본 각각의 정확도,
mIoU, 결함 mIoU, 클래스별 IoU). `summarize_cv` 가 이것을 모아 `figures/<run>/v5/cv/` 에 `summary.md` (표), `curves.png` (fold 별 곡선),
`summary.json` (권장 길이) 을 쓴다.

여기서 읽을 것:

- 결함 mIoU 가 0 에서 떨어져 오르는가 (질문 1).
- 평균 곡선의 최고점이 어디인가, 그 뒤로 내려가는가 — 내려가면 과적합이다 (질문 2).
- EMA 곡선이 학습용 가중치 곡선을 따라가는가.
- 클래스별 IoU: 어느 클래스가 본 적 없는 소스에서도 나오는가 (질문 3).

### 6.2 native 해상도 채점

`evaluate_gt` 가 fold 가 떼어 둔 표본을 서빙과 같은 경로로 채점한다 (원본 → 1036² → 추론 → logits 를 원본 크기로 올림 → argmax, 정답은 native 주석).
TTA · 후처리 없이 낸 값이 기본이고, `eval` phase 에 `--tta --postprocess` 를 주면 그 변형을 따로 낸다.

### 6.3 v4 와의 비교 — 같은 표본, 단 v4 쪽에 유리

클래스와 구조가 같아 v4 의 `stage2_best.pt` 를 같은 스크립트로 채점할 수 있다. `cv` phase 가 fold 마다 자동으로 낸다 (`ref_v4/`).
**v4 는 이 표본들을 Stage 2 에서 학습에 썼다.** loso fold 의 v5 는 그 소스를 본 적이 없다. 따라서:

- v5 가 v4 와 같거나 높으면 강한 근거다.
- v5 가 낮으면 판단할 수 없다 (v4 가 그 표본을 봤기 때문일 수 있다).

두 모델 모두 본 적 없는 사람 GT 는 지금 없다. DIVA 내보내기가 오면 v6 계획의 기준선 채점 (v4 · v5 를 `V4_TO_NEW` 로 옮겨 같은 test 에서) 이 그 역할을 한다.

### 6.4 ORNL Build 1 — 참고용. 어떤 결정에도 쓰지 않는다

v1–v4 는 모두 Build 1 에서 DSCNN 예측과의 일치도를 냈다. 같은 표를 v5 로도 낸다 (`confusion` phase → `figures/<run>/v5/full/build1_vs_dscnn/`).
클래스 공간이 같아 v4 의 라벨 캐시를 그대로 읽는다. v4 의 같은 표는 `figures/vitb14_reg_dpt_dual_sz1036_11cls_v4/v4/stage2/confusion/` 에 있다
(Powder 0.974, Printed 0.914, Spatter 0.327, Swelling 0.311, RS 0.125, IS 0.000).

**이 수치가 v4 보다 낮게 나와도 그것만으로 나빠졌다고 볼 수 없다.** 정답이 아니라 DSCNN 의 예측과 비교하는 값이고, v4 는 그 예측을 흉내 내도록 학습했다.
예를 들어 DSCNN 은 Build 1 의 3,218 레이어 전부에 Spatter (합 29.5 M px) 와 Swelling 을 낸다. v5 는 그것을 따라 할 이유가 없다.
`infer` phase 의 4-panel figure (visible/0 · visible/1 · DSCNN 예측 · v5 예측) 를 눈으로 보는 쪽이 더 많은 것을 알려 준다.

### 6.5 결과를 읽는 순서 (제안)

1. `curves.png` — 학습이 되는가, 어디서 꺾이는가.
2. `summary.md` 의 클래스별 IoU — 무엇이 되고 무엇이 안 되는가. §2 의 전망과 맞는가.
3. `ref_v4` 와의 차이 — §6.3 의 주의와 함께.
4. Build 1 figure — v4 figure 와 나란히.

v5 는 그 자체로 배포 판단을 내리기 위한 것이기보다, GT 전용 학습의 실제 모습을 처음 보고 v6 의 학습 설정을 정하는 실험이다.
결과는 `v5/RESULTS_v5.md` 에 정리한다.

---

## 7. 후처리

v4 의 두 규칙 그대로다 (클래스 번호가 같다). 규칙 1: Printed 에서 100 px 이상 떨어진 SE / Swelling 성분 → Debris.
규칙 2: Printed (10 px 팽창) 와 50% 이상 겹치는 Spatter 성분 → Over Melting. 주 지표는 후처리 없이 낸다.

---

## 8. 폴더 구조 · 산출물 · compose

```
DefSeg_AM/v5/
├── PLAN_v5.md
├── config_v5.py              # 클래스 (v4 와 같음), fold 정의, 학습 값, 경로
├── data/
│   ├── data_gt_v5.py         # 사람 GT: enumerate · 재매핑 · GTPool (RAM) · fold 분할 · Dataset
│   ├── sampler_helpers.py    # §5.4
│   ├── augmentation.py       # v4 사본
│   ├── copy_paste.py         # v4 사본
│   └── data_ornl_v5.py       # ORNL 캐시 읽기 — 참고 평가 전용 (§6.4)
├── models/losses_v5.py       # class weight
├── training/train.py         # 단일 단계, step 기준. --fold 로 검증 fold, 없으면 전수 학습
├── inference/
│   ├── metrics.py            # confusion matrix · 지표 · 파일 쓰기 (공용)
│   ├── evaluate_gt.py        # 사람 GT native 채점 (§6.2–6.3)
│   ├── confusion.py          # ORNL Build 1 참고 평가 (§6.4)
│   ├── infer.py              # ORNL 4-panel figure
│   ├── postprocess.py        # v4 사본
│   └── viz.py                # 팔레트 · panel (v4 와 같은 색)
├── scripts/summarize_cv.py   # 검증 fold 요약 · 권장 길이 (§6.1)
└── docker/{Dockerfile, docker-compose.yml, README.md}
```

- 의존은 `common` 뿐이다. v4 코드는 복사해서 고쳤다.
- **캐시를 만들지 않는다.** `confusion` phase 만 기존 캐시를 읽는다 (이미지는 v1, 라벨은 v4).
- **run-name**: `vitb14_reg_dpt_dual_sz1036_11cls_v5`.
  ckpt 는 `checkpoints/<run>/full/{last, last_raw}.pt`, `checkpoints/<run>/cv/<fold>/{best, last}.pt`. 그림 · 표는 `figures/<run>/v5/…`.
- **ckpt 형식은 v4 와 같다** (`model_state` = trainable weight 만, `config.backbone` · `n_classes` · `img_size`). 추가로 `weights` (ema / raw), `step`,
  `config.world_size` · `total_steps` · `class_train_px` 등을 남긴다. v4 ckpt 에는 GPU 수가 없어 step 수를 되짚을 때 추정해야 했다.
- **compose**: `DEFSEG_V5_PHASE = cv | summarize | eval | train | confusion | infer | all`, `DEFSEG_V5_FOLDS`, `DEFSEG_V5_EXTRA`.
  phase 를 주지 않으면 사용법만 찍고 끝난다. 사전조건이 없으면 FATAL 로 끝난다. 자세한 것은 [docker/README.md](docker/README.md).

---

## 9. 진행 순서

1. **(완료)** 구현 + `--quick` 스모크 (1 GPU fold, 2 GPU DDP fold, 전수 학습 → confusion → infer, native 재채점) + 실제 해상도 60 step.
2. **(사용자)** 검증 fold 하나 (`loso_L21`) 를 10,000 step 으로 → `curves.png` 를 보고 길이를 정한다. 4 GPU 로 약 2시간.
3. **(사용자)** 나머지 fold (`loso_MS loso_I2 loso_GP inlayer`) 를 정한 길이로. fold 당 약 2시간 × (길이 / 10,000).
4. **(사용자)** 전수 학습 → `confusion` → `infer`.
5. 결과를 `RESULTS_v5.md` 로 정리하고, 확정한 길이 · EMA · sampler 값을 v6 계획에 옮긴다.

---

## 10. 알려진 한계 · 위험

- **GT 가 거의 없는 클래스.** Under Melting (1.2 K px / 2 레이어) 은 학습되지 않는다고 보는 것이 맞고, Over Melting (24.8 K px, I2 뿐) 도 불안하다 (§2).
  v4 가 이 클래스들을 내던 화면에서 v5 는 다른 클래스를 내거나 아무것도 내지 않을 것이다.
- **한 소스에만 있는 클래스 (RH, OM, UM) 는 loso 로 잴 수 없다** (§4). `inlayer` 의 값은 낙관적이다.
- **125 레이어에 과적합할 수 있다.** 10,000 step 은 레이어 하나를 평균 1,280번 보는 길이다. 검증 곡선이 꺾이는 지점을 보고 길이를 줄인다.
- **v4 와의 공정한 비교가 지금은 없다** (§6.3). ORNL 수치는 DSCNN 과의 일치도일 뿐이다 (§6.4).
- **기존 주석의 문제를 그대로 물려받는다**: Soot IGNORE, IS 주석의 혼재 (§2). v6 에서 다룬다.
- **검증 fold 에서 고른 길이는 약간 낙관적이다.** 같은 fold 로 길이도 정하고 성능도 보고하기 때문이다. 고르는 값이 하나 (길이) 뿐이라 영향은 작다.
- **GPU 를 서빙과 나눠 쓴다.** 학습은 GPU 당 23.6 GB 를 쓰고, GPU 0 · 1 에는 `defsegam-api-v4` · `diva-sam-service` 가 3–4 GB 씩 올라가 있다 (32 GB 중 27 GB).
  학습 중에 DIVA 의 `/predict/batch` 가 겹치면 OOM 이 날 수 있다.
- **재개 (resume) 를 지원하지 않는다.** 중간에 끊기면 그 실행을 처음부터 다시 돌린다 (v4 와 같음).

**범위 밖**: 클래스 체계 변경과 DIVA 내보내기 (v6), `DefSegAM_API` 교체 (클래스와 ckpt 형식이 같아 파일만 바꾸면 읽히지만, 바꿀지는 결과를 보고 정한다), 논문 원고.

---

## 부록 A. 스모크에서 확인한 것 (2026-11-16)

| 실행 | 확인한 것 |
|---|---|
| `cv`, `loso_L21`, 1 GPU, `--quick` | fold 학습 → `best.pt` · `last.pt` · `curve.csv` → native 채점 (v5 + v4 참고) → 요약 |
| `cv`, `loso_MS inlayer` · `loso_GP inlayer` · `loso_I2`, 2 GPU DDP, `--quick` | DDP 에서 rank 0 검증 + barrier, rank 간 decoder 초기값 일치 점검, 소스 안 분할 fold, 요약의 평균 곡선 |
| `all`, 2 GPU DDP, `--quick` | 전수 학습 (`last.pt` · `last_raw.pt`) → Build 1 참고 평가 (2 배치) → figure (1 레이어, TTA + 후처리) |
| `eval`, `--quick --tta --postprocess --save-figs 1` | native 채점의 TTA · 후처리 · panel 저장 |
| 실제 해상도 60 step (1036², GPU 2장 × 4장, `loso_L21`) | step 당 0.67초, GPU 당 최고 23.6 GB, 검증 1회 6초. loss 2.43 → 0.52 |

`--quick` 은 224² · 6 step 이라 수치에 의미가 없다. 실제 해상도 60 step 에서 학습용 가중치의 결함 mIoU 가 0.05–0.07 까지 올랐다는 것은
"배선이 맞다" 는 뜻일 뿐 성능의 근거가 아니다.

스모크 중에 compose 에서 두 가지를 고쳤다 (v3 · v4 의 compose 를 그대로 옮기면 생기는 문제다).

- **GPU 지정이 먹지 않았다.** `deploy … count: all` 블록이 있으면 `NVIDIA_VISIBLE_DEVICES` 와 무관하게 GPU 4개가 다 보이고 프로세스가 GPU 0 부터 올라간다.
  이것을 알기 전의 스모크 다섯 번은 GPU 2 · 3 을 지정했는데도 서빙과 함께 쓰는 GPU 0 · 1 에서 돌았다. 가장 무거운 것은 실제 해상도 40 step
  (약 1분, 최고 27.3 GB / 32.6 GB) 이었고 서빙 컨테이너는 계속 healthy 였다. 블록을 빼고 `nvidia-smi -L` 로 확인한 뒤 GPU 2 · 3 에서 다시 쟀다.
- **프로젝트 네트워크를 만들 수 없었다.** 이 호스트에는 compose 네트워크가 29개 남아 있고 (27개는 컨테이너가 없다) docker 기본 주소 풀이 차서,
  새 프로젝트의 네트워크 생성이 `all predefined address pools have been fully subnetted` 로 실패한다.
  v5 compose 는 `network_mode: bridge` 로 전용 네트워크를 만들지 않는다. v3 · v4 의 compose 는 지금 이 호스트에서 같은 오류로 뜨지 않는다.

## 부록 B. 이 문서의 수치가 나온 곳

- **사람 GT 통계** (§2, §3, §4): 2026-11-16 에 `GTPool` 로 125 레이어를 전수 집계. native 화소 수는 v4 의 `compute_sample_pix_per_class` 와 값이 같다.
- **v4 Stage 1 의 의사 라벨 양** (§2): `cache/resized_sz1036_v4/train_index.npz` 의 `pix_per_layer` 합 (Build 2–5, 12,607 레이어, 1036²).
  Build 1 의 Spatter 29.5 M px 는 같은 캐시의 `val_index.npz`.
- **sampler 분포** (§5.4): 같은 날 125 레이어에서 계산. v4 열은 v4 의 `compute_sample_weights_stage2` 를 그대로 호출한 값.
- **v4 의 EMA** (§0): `v4/training/train_stage2.py` 와 `checkpoints/vitb14_reg_dpt_dual_sz1036_11cls_v4/*.pt` 의 기록
  (저장된 epoch 49 = 마지막, `ema_decay` 0.9999, `replicate_factor` 4). 125 표본 × K 4 ÷ 4 rank ÷ batch 4 = epoch 당 31 step, 50 epoch.
  실행 로그는 남아 있지 않아 GPU 수는 compose 기본값 (4) 으로 계산했다. 1 GPU 였다면 6,250 step 이고 남는 비율은 54% 다.
  두 ckpt 의 trainable weight 1,456만 개를 비교한 값은 ‖S2 − S1‖ / ‖S1‖ = 0.028, cosine 0.9996 이다.
  서빙 파일 `DefSegAM_API/assets/checkpoints/stage2_v4_11cls.pt` 는 `stage2_best.pt` 와 md5 가 같다.
- **v4 의 Build 1 표** (§6.4): `figures/vitb14_reg_dpt_dual_sz1036_11cls_v4/v4/stage2/confusion/per_class_metrics.csv`.
- **속도 · 메모리** (§5.7, §10, 부록 A): 2026-11-16 에 GPU 2 · 3 에서 `nvidia-smi` 를 2초 간격으로 기록.
