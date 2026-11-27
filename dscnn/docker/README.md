# DSCNN 재구현 — Docker 사용법

> DSCNN (Scime et al., *Additive Manufacturing* 36 (2020) 101453) 의 구조를 다시 짜서 **사람 GT 만으로** 학습하고,
> DefSeg-AM (v4 – v7) 과 **같은 검증 레이어 · 같은 지표** 로 견준다. 버전이 아니라 따로 둔 비교 실험이다 (`genaug/` 처럼).
> 클래스 (`pbf-2026.11` 10-class) · 데이터셋 · fold · 채점은 v6 의 코드를 불러 그대로 쓴다. 달라지는 것은 모델과 그 모델의 학습 방식뿐이다.
> 호스트의 `DefSeg_AM/venv` 를 read-only bind mount 하여 그대로 쓴다.

**이 DSCNN 은 ORNL 이 배포한 DSCNN 이 아니다.** ORNL 빌드의 HDF5 에 들어 있는 `segmentation_results` 는 ORNL 이 자기 데이터로 학습한 모델의 예측이고
가중치는 공개되지 않았다. 여기의 모델은 논문의 표를 보고 다시 만든 것을 우리가 가진 사람 GT 125 레이어로 무작위 초기값에서 학습한 것이다.

**용어**

| 말 | 뜻 |
|---|---|
| 타일 | 사진을 원본 해상도 그대로 잘라 낸 정사각형 조각. DSCNN 은 사진을 줄이지 않고 타일 단위로 본다. 한 변의 화소 수 T 는 장비마다 다르다 |
| step | 타일 한 묶음 (많아야 50 장, 평균 40 장쯤) 을 보고 가중치를 한 번 고치는 것 |
| epoch | 학습 타일 전부를 증강 9 가지로 한 번씩 본 것 (`inlayer` 에서 약 1,500 step) |
| fold | 사람 GT 의 일부를 검증용으로 떼어 두고 나머지로 학습하는 한 번의 실행. 기본은 v6 · v7 과 같은 `inlayer` (22 레이어를 뗀다) |
| EMA | 학습 중 가중치의 이동 평균. 마지막 값 대신 이것을 저장한다 (논문 방식, decay 0.9999) |

## 논문에서 옮긴 것과 여기서 정한 것

논문 (`paper/DSCNN/` 의 PDF, 표 3 – 6 과 3.3 · 3.7 절) 그대로:

- 구조: global (레이어 전체를 32² 로) · regional (타일을 128² 로 줄인 U-Net, 512-벡터 하나까지 줄였다 편다) · localization (원본 해상도 타일에 11×11 CONV 하나) 세 갈래 +
  화소의 (x, y) 좌표 → 178 채널 → 1×1 CONV. 파라미터 13,999,118 개 (논문의 식과 537 차이).
- 입력: 사진 두 장 (용융 직후 · 도포 직후) 을 2 채널로. 밝기는 소스마다 정한 평균을 빼기만 한다 (사진마다 따로 맞추지 않는다).
- 학습: Adam (1e-4, β 0.9 · 0.999, ε 1e-4), 스케줄 없음, EMA 0.9999. 손실은 cross-entropy × 클래스 가중치 `중앙값 빈도 / 빈도` (상한 없음), 미레이블 화소는 0.
  주석이 10% 에 못 미치는 타일은 뺀다. 초기값 CONV N(0, 0.05²) · FC N(0, 0.004²) · bias 0.1.
- 증강: 잡음 (없음 · 2 단계, global · regional 입력에만) × 밝기 이동 (없음 · ±10%) = 9 가지를 타일마다 모두. **회전 · 반전은 쓰지 않는다** (논문이 명시). Copy-Paste 도 없다.
- 길이: 100 epoch ("on the order of 100").
- 타일 크기: ConceptLaser M2 225, ExOne Innovent 450.

논문에 없어 정한 것 (`config_dscnn.py` 에 "정함" 으로 적었다):

- **한 모델로 8 소스를 함께 배운다.** 논문은 장비마다 모델을 따로 두고 전이학습으로 잇는다. v6 과 같은 조건 (한 모델 · 통합 10 클래스) 으로 견주려고 합쳤다.
  한 배치에 여러 소스의 타일을 섞는다 — 한 소스로만 채우면 BatchNorm 이 학습 때와 추론 때 다른 통계를 써서 점수가 흔들렸다 (2026-11-23 시험 실행).
  타일 크기가 소스마다 달라, 크기와 무관한 갈래 (regional · global) 는 한꺼번에 돌리고 localization 은 크기가 같은 타일끼리 돌린다.
- 2022 판 소스의 타일 크기: 같은 장비라도 사진 크기가 논문과 달라, 타일이 베드에서 차지하는 비율이 같도록 옮겼다 (EOS M290 766² → 112, 1997² → 292, ExOne M-Flex → 794). AddUp FormUp 350 은 논문에 없어 225 로 골랐다.
- 배치의 타일 수: 많아야 50 장 (논문의 범위 15 – 50) 이고, 화소 수가 예산 (T 450 인 타일 15 장 분량) 을 넘기 전에 끊는다 (GPU 메모리).
- 잡음의 크기: "분산이 동적 범위의 0.01% · 0.1%" 를 σ = 2.55 · 8.06 단계 (밝기 0–255) 로 읽었다.
- 손실의 평균: 주석이 있는 화소 수로 나눈다. FC 뒤에 ReLU 를 두지 않는다 (표가 그렇게 적는다 — `--fc-relu` 로 바꿀 수 있다).
- 검증: 논문은 타일의 10% 를 무작위로 떼지만, 여기서는 v6 과 같은 레이어를 통째로 뗀다.

**클래스 가중치는 두 가지로 돌릴 수 있다.** 기본은 논문의 `중앙값 빈도 / 빈도` 다. `--class-weight sqrt_inv` 를 주면 v6 의 가중치 (√역빈도, 상한 10) 를 쓴다 — 논문 방식이 아니다.
우리 데이터는 논문의 데이터보다 드문 클래스가 더 드물어서 (OM 이 화소의 0.016%) 논문의 가중치는 Powder 를 0.008, OM 을 37.7 로 놓는다 (4,700 배).
그러면 Powder 를 결함으로 내도 손실이 거의 늘지 않아 잘못 짚음이 많다 (아래 시험 실행).

v6 과 다른 점 (모델 말고): 증강 (v6 은 반전 · 180° 회전 · 이동 · Copy-Paste 를 쓴다), 표본 추출 (v6 은 소스 · 클래스 균등 혼합), 클래스 가중치 (v6 은 √역빈도, 상한 10),
해상도 (v6 은 1036² 로 줄여 넣는다). 모두 "그 모델의 원래 학습 방식" 을 따른 것이다 — 구조만 바꾼 비교가 아니라 **방법 전체의 비교** 다.

## 사전조건

- NVIDIA driver + nvidia-container-toolkit (host), Docker / docker compose v2, `DefSeg_AM/venv`
- `ORNL_Data/DSCNN_Dataset_pbf-2026.11` — 사람 GT (v6 의 `convert` phase 가 만든다)
- `compare` 만: 견줄 모델의 ckpt (v4 · v5 · v6 · v7 — 없는 것은 그 열만 빠진다)
- 호스트 UID/GID 매칭용 `.env`

```bash
cd DefSeg_AM/dscnn/docker
echo "UID_GID=$(id -u):$(id -g)" > .env
```

## 실행

GPU 는 한 대만 쓴다 (기본 2 번. 0 · 1 번에는 서빙 컨테이너가 있다). 생성 데이터도 DSCNN 예측도 쓰지 않는다 — 사람 GT 만이다.

```bash
cd DefSeg_AM/dscnn/docker

# 1) 학습 → 채점: inlayer fold, 100 epoch (155,000 step). RTX 5090 한 대로 2시간 16분 (2026-11-23 실측), GPU 메모리 18 GB
#    (가) 논문의 설정 그대로 → run 이름 dscnn_native_10cls_gt-s42
DSCNN_PHASE=cv docker compose up -d --build
docker compose logs -f
#    (나) 클래스 가중치만 v6 방식으로 → run 이름 dscnn_native_10cls_gt-sqrtw-s42. (가) 와 함께 돌리려면 GPU 와 프로젝트 이름 (-p) 을 달리 준다
NVIDIA_VISIBLE_DEVICES=3 DSCNN_PHASE=cv DSCNN_TAG=sqrtw DSCNN_EXTRA="--class-weight sqrt_inv" docker compose -p defseg-dscnn-b up -d --build
docker compose -p defseg-dscnn-b logs -f

# 2) 정답 · DSCNN 재구현 · v4 · v5 · v6 · v7 을 나란히 (그림 + 표). 22 레이어에 약 1분
DSCNN_PHASE=compare docker compose up --build                     # (가) 의 모델
DSCNN_PHASE=compare DSCNN_TAG=sqrtw docker compose up --build     # (나) 의 모델

# 3) ORNL 빌드의 무작위 레이어에서 ORNL 의 DSCNN 예측과 나란히 (사람 GT 없음 — 참고용. 학습 없음)
DSCNN_PHASE=ornl_compare DSCNN_TAG=sqrtw docker compose up --build

# 다른 GPU / 다른 seed / 본 적 없는 장비 fold
NVIDIA_VISIBLE_DEVICES=3 DSCNN_SEED=43 DSCNN_PHASE=cv docker compose up -d --build
DSCNN_FOLDS=loso_L21 DSCNN_PHASE=cv docker compose up -d --build

# 스모크 (30 step) / 자체 점검 (데이터 · GPU 없이 몇 초)
DSCNN_PHASE=cv DSCNN_EXTRA=--quick docker compose up --build --exit-code-from defseg-dscnn
DSCNN_PHASE=selftest docker compose up --build
```

| phase | 하는 일 |
|---|---|
| `selftest` | 파라미터 수 · 갈래의 출력 크기 · 타일 격자 · 입력 묶음 · 증강 · EMA 를 데이터 없이 확인한다 |
| `cv` | `DSCNN_FOLDS` 의 fold 마다 학습 → 채점 (원본 해상도 + 1036²) |
| `train` | 125 레이어 전수 학습 (검증 없음) |
| `eval` | 채점만 다시. 예: `DSCNN_EXTRA="--which best --save-figs 22"` |
| `compare` | 검증 레이어에서 정답 · DSCNN 재구현 · v4 · v5 · v6 · v7 을 나란히 그리고 표를 낸다 (v7 의 꼬리표는 `DSCNN_EXTRA="--v7-variant W"`) |
| `ornl_compare` | ORNL 5 빌드의 무작위 레이어 (빌드마다 4 장) 에서 DSCNN (ORNL 이 배포한 예측) · DSCNN 재구현 · v4 · v5 · v6 · v7 을 나란히 그린다. 사람 GT 가 없어 점수가 아니라 모델끼리의 비교다 (30초) |

| 환경변수 | 기본 | 뜻 |
|---|---|---|
| `DSCNN_PHASE` | (없음) | 위 표. 안 주면 사용법만 출력한다 |
| `DSCNN_FOLDS` | `inlayer` | v6 과 같은 fold 이름. 여럿이면 공백으로 띄운다 |
| `DSCNN_SEED` | `42` | 난수 출발값 (run 이름에 붙는다) |
| `DSCNN_TAG` | (없음) | run 이름에 붙는 꼬리표 — 설정을 바꾼 실행을 가를 때 |
| `DSCNN_EXTRA` | (없음) | 스크립트에 넘길 인자. 예: `--epochs 30`, `--quick` |
| `NVIDIA_VISIBLE_DEVICES` | `2` | 쓸 GPU 한 대 |

## 산출물

run 이름은 `dscnn_native_10cls_gt[-<꼬리표>]-s<seed>` 다.

```
checkpoints/<run>/cv/<fold>/{last.pt, best.pt, last_raw.pt}      last = 마지막 step 의 EMA (견줄 때 쓰는 것), best = 검증 결함 mIoU 가 가장 높았던 EMA
figures/<run>/cv/<fold>/run.json · curve.csv · result.json       설정 · 검증 곡선 (원본 해상도) · 걸린 시간
figures/<run>/cv/<fold>/eval_{native,resized}_last/              채점 — v6 의 metrics.json 과 같은 형식 (클래스별 IoU, 놓침 · 잘못 짚음, 소스별 표)
figures/<run>/cv/<fold>/gt_val22_last/                           compare — pages/page_XX.png, summary.md · summary.json
figures/<v7 run>/v6/cv/<fold>/ornl_random20_last_dscnn_reimpl_<꼬리표 | paperw>/     ornl_compare — v7 의 ORNL 비교 (ornl_random20_last) 옆에 둔다
```

## 풀런 (2026-11-23, 100 epoch)

`inlayer` fold, seed 42, 155,000 step. 실행 하나에 RTX 5090 한 대로 136분이 걸렸다. 검증 22 레이어, `last.pt` (마지막 step 의 EMA), TTA · 후처리 없음.

| 모델 | 결함 mIoU (원본 해상도) | 결함 mIoU (1036²) | 놓침 | 잘못 짚음 | 화소 정확도 | run |
|---|---:|---:|---:|---:|---:|---|
| DSCNN 재구현 (가) — 논문의 가중치 | 0.336 | 0.367 | 15.3% | 7.6% | 89.5% | `dscnn_native_10cls_gt-s42` |
| DSCNN 재구현 (나) — v6 의 가중치 | 0.398 | 0.448 | 16.1% | 3.6% | 93.2% | `dscnn_native_10cls_gt-sqrtw-s42` |
| v5 (Soot → Spatter) | 0.333 | 0.404 | 14.3% | 3.0% | 83.0% | `vitb14_reg_dpt_dual_sz1036_11cls_v5-soot2sp` |
| v6 | 0.467 | 0.533 | 11.6% | 3.0% | 94.6% | `vitb14_reg_dpt_dual_sz1036_10cls_v6` |
| v7-W | 0.479 | 0.546 | 10.8% | 3.3% | 94.4% | `vitb14_reg_dpt_dual_sz1036_10cls_v7-W-s42` |

놓침 · 잘못 짚음 · 화소 정확도는 원본 해상도의 값이다.

클래스별 IoU (원본 해상도):

| 모델 | Pow | Prt | IS | PS | RS | RH | SE | SP | OM | DB |
|---|---:|---:|---:|---:|---:|---:|---:|---:|---:|---:|
| DSCNN 재구현 (가) | 0.870 | 0.856 | 0.171 | 0.843 | 0.150 | 0.227 | 0.412 | 0.410 | 0.112 | 0.359 |
| DSCNN 재구현 (나) | 0.917 | 0.894 | 0.379 | 0.842 | 0.146 | 0.380 | 0.481 | 0.465 | 0.161 | 0.332 |
| v6 | 0.933 | 0.911 | 0.227 | 0.964 | 0.261 | 0.354 | 0.730 | 0.566 | 0.170 | 0.463 |

소스별 결함 mIoU (원본 해상도):

| 소스 | DSCNN 재구현 (가) | DSCNN 재구현 (나) | v6 |
|---|---:|---:|---:|
| L21 | 0.206 | 0.309 | 0.403 |
| 17-4 | 0.474 | 0.595 | 0.683 |
| GP | 0.210 | 0.240 | 0.494 |
| I1 | 0.751 | 0.694 | 0.622 |
| I2 | 0.329 | 0.409 | 0.531 |
| MS | 0.346 | 0.399 | 0.388 |
| BJ21 | 0.135 | 0.094 | 0.173 |
| H13 | 0.914 | 0.883 | 0.974 |

읽을 때 알아 둘 것:

- **오래 돌려도 오르지 않았다.** (나) 의 검증 점수는 epoch 13 쯤에 0.41 에 닿은 뒤 epoch 65 까지 0.38 – 0.44 사이를 오갔고, 그 뒤로는 0.39 근처로 내려갔다
  (채점 한 번과 다음 번 사이의 흔들림이 표준편차 0.013 쯤이다). 10 epoch 시험 실행 (0.410) 보다 높지 않다. 학습 손실은 0.17 → 0.04 로 끝까지 내렸으니 학습 레이어를 외우는 쪽으로 간 것이다.
  내려간 몫의 대부분은 Over Melting 이다 (0.40 → 0.16 — 검증 레이어에 5천 화소뿐인 클래스라 값이 크게 흔들린다). (가) 는 0.235 → 0.335 로 올라 epoch 60 뒤로 0.33 근처에 머물렀다.
- `best.pt` (검증 점수가 가장 높았던 step) 로 재면 (가) 0.346 / 0.366 (step 110,000), (나) 0.439 / 0.464 (step 88,000) 이다. 이 값은 **검증 레이어를 보고 고른 것**이라 DSCNN 쪽에 유리하게 높다 —
  v6 · v7 은 고르지 않은 `last.pt` 로 재므로, 같은 조건의 비교는 위 표다. 그 유리한 값으로도 v6 가 앞선다 (0.439 < 0.467).
- seed 는 하나만 돌렸다. v6 와 (나) 의 차 (0.069) 는 위의 흔들림의 다섯 배쯤이지만, seed 를 바꿨을 때의 폭은 재지 않았다.
- 채점은 `figures/<run>/cv/inlayer/eval_{native,resized}_{last,best}/`, 그림과 표는 `figures/<run>/cv/inlayer/gt_val22_last/` 에 있다.

## 시험 실행 (2026-11-23, 10 epoch — 풀런의 1/10)

길이와 설정을 정하려고 풀런 전에 `inlayer` fold 를 10 epoch (15,500 step, 14분) 만 돌린 것이다. 그때는 검증 점수가 아직 오르고 있다고 보았는데, 풀런에서 (나) 는 이 근처에서 멈췄다 (위).

| 모델 (원본 해상도) | 결함 mIoU | 놓침 | 잘못 짚음 | 화소 정확도 | run |
|---|---:|---:|---:|---:|---|
| DSCNN 재구현 — 논문의 가중치, 10 epoch | 0.235 | 12.5% | 24.3% | 75.3% | `dscnn_native_10cls_gt-pilot10-mix-s42` |
| DSCNN 재구현 — v6 의 가중치, 10 epoch | 0.410 | 9.6% | 6.1% | 91.7% | `dscnn_native_10cls_gt-pilot10-mix-sqrtw-s42` |
| v6 (10,000 step 풀런) | 0.467 | 11.6% | 3.0% | 94.6% | `vitb14_reg_dpt_dual_sz1036_10cls_v6` |

1036² 에서는 0.249 · 0.418 · 0.533 이다. 소스별로 보면 v6 가 8 소스 가운데 7 곳에서 앞선다 (I1 만 비슷하다). 합친 점수의 차가 작게 보이는 것은 화소의 38% 를 차지하는 BJ21 에서 둘 다 낮기 때문이다.
그림과 클래스별 표: `figures/dscnn_native_10cls_gt-pilot10-mix-sqrtw-s42/cv/inlayer/gt_val22_last/`.

견줄 때의 주 지표는 원본 해상도의 `last.pt` 채점이다 (v6 · v7 과 같다). DSCNN 은 원래 원본 해상도로 예측하고, DefSeg-AM 은 1036² 의 logits 를 원본 크기로 올려 채점한다.
