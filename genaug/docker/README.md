# DefSeg-AM GenAug — Docker 사용법

> GenAug = 결함 인페인팅 확산 모델로 학습 데이터를 늘리는 실험. 설계는 [../PLAN_GENAUG.md](../PLAN_GENAUG.md).
> 이 폴더로 하는 것은 셋이다 — **생성 모델을 학습시킨다, 그 모델로 합성 세트를 만든다 (`synth`), 세트를 점검한다 (`check`)**.
> 합성 세트를 분할 모델의 학습에 넣는 것은 v7 의 일이다 ([../../v7/PLAN_v7.md](../../v7/PLAN_v7.md), [../../v7/docker/README.md](../../v7/docker/README.md)).
> 호스트의 `DefSeg_AM/venv` (torch 2.9.1+cu128) 를 read-only bind mount 하여 그대로 쓴다. 새 패키지는 쓰지 않는다.

**용어**

| 말 | 뜻 |
|---|---|
| step | 창 한 묶음 (GPU 수 × 16장) 을 보고 가중치를 한 번 고치는 것 |
| EMA | 학습용 가중치와 따로 들고 가는 평균 사본. 그림을 그릴 때는 이 사본을 쓴다 |
| 창 | 1036² 사진에서 잘라 낸 256×256 조각. 생성 모델은 창 하나를 다룬다 |
| 표본 시트 | 같은 창들을 생성 모델이 어떻게 그리는지 모아 놓은 그림. 학습이 진행되는 것을 눈으로 본다 |
| 라벨 맵 | "이 화소는 Powder, 저 화소는 RS" 를 적은 지도. 생성 모델의 입력이자 합성 레이어의 정답이다 |
| 합성 레이어 · 합성 세트 | 라벨 맵으로 그린 1036² 사진 두 장과 그 라벨 맵 (사진의 바깥 5% 만 진짜 화소다). 그것을 모아 사람 GT 데이터셋과 같은 구성으로 저장한 것이 합성 세트다 |

## 사전조건

- NVIDIA driver + nvidia-container-toolkit (host), Docker / docker compose v2, `DefSeg_AM/venv`
- `ORNL_Data/DSCNN_Dataset_pbf-2026.11` — 사람 GT 를 `pbf-2026.11` 로 변환한 데이터셋 (v6 의 `convert` phase 가 만든다)
- `ORNL_Data/Co-Registered …/[baseline] (Peregrine v2023-11)/*.hdf5` — `canvas` phase 만
- `cache/ornl_canvas_sz1036/` — 공용 배경 캐시. **2026-11-17 에 다섯 빌드를 만들어 두었다** (3,167 레이어, 9.5 GB). 없으면 `canvas` phase 로 만든다
- 호스트 UID/GID 매칭용 `.env` (없으면 1000:1000 으로 돌아 산출물 쓰기에 실패한다)

```bash
cd DefSeg_AM/genaug/docker
echo "UID_GID=$(id -u):$(id -g)" > .env
```

## 순서

```bash
cd DefSeg_AM/genaug/docker

# 1) 배경 점검 시트를 본다 (Q0) — figures/genaug/q0/<빌드>_sheet.png. 이미 만들어 두었다. 다시 만들려면:
GENAUG_PHASE=q0 docker compose up --build
#    → 정상 라벨을 적어 줄 빌드를 정한다 (아래 "풀런 전에 정할 것")

# 2) 생성 모델 학습 — GPU 2 · 3, 100,000 step, 약 8시간
GENAUG_PHASE=train docker compose up -d --build
docker compose logs -f
#    중간에 끊겼으면 이어 돌린다 (5,000 step 마다 저장한다)
GENAUG_PHASE=train GENAUG_EXTRA=--resume docker compose up -d --build

# 3) 표본 시트를 다시 — 다른 창 · 다른 seed · 다른 guidance 로 (GPU 1대, 1–2분)
GENAUG_PHASE=sample NVIDIA_VISIBLE_DEVICES=2 GENAUG_EXTRA="--per-class 4 --seed 7" docker compose up --build

# 4) 합성 세트 — 라벨 맵으로 그린 레이어 600 장 → cache/genaug/sets/w600_v1/  (GPU 2 · 3 으로 70분)
GENAUG_PHASE=synth docker compose up -d --build
docker compose logs -f
#    끊겼으면 같은 명령을 다시 준다 (다 만든 묶음은 건너뛴다). 다른 세트는 이름을 바꿔서:
GENAUG_PHASE=synth GENAUG_EXTRA="--name w2000_v1 --n-layers 2000" docker compose up -d --build

# 5) 세트의 점검 (K1–K6) → figures/genaug/sets/w600_v1/  (GPU 1대, 약 10분)
GENAUG_PHASE=check NVIDIA_VISIBLE_DEVICES=2 docker compose up --build
GENAUG_PHASE=check GENAUG_SET=w2000_v1 NVIDIA_VISIBLE_DEVICES=2 docker compose up --build

docker compose down
```

- `synth` 와 `check` 를 동시에 띄우지 않는다 — 같은 서비스라 뒤의 `up` 이 앞의 컨테이너를 바꿔 버린다.

## Phase

| `GENAUG_PHASE` | 하는 일 | GPU | 걸리는 시간 (2026-11-17 측정) |
|---|---|---|---|
| (없음) | 사용법만 찍고 끝난다 | – | – |
| `canvas` | 공용 배경 캐시 — ORNL HDF5 → 원본 밝기 사진 + CAD 마스크. 이미 있는 빌드는 건너뛴다. 끝에 HDF5 와 다시 견준다 | – | 다섯 빌드를 한꺼번에 21분 (프로세스 30개) |
| `q0` | 배경 점검 시트 (빌드마다 무작위 창 100개) | – | 10초 |
| `selftest` | 배선 점검 — holdout, 학습 창, 모델, 확산 과정, "그릴 자리 밖은 그대로" | – | 1분 |
| `train` | 생성 모델 학습 | `NPROC` 대 | 100,000 step 에 약 8시간 (GPU 2대, 초당 3.55 step) |
| `sample` | 학습된 모델로 표본 시트 | 1대 | 1–2분 |
| `synth` | 합성 세트를 만든다 — 라벨 맵 (CAD 파트 배치 + 규칙대로 얹은 RS · RH · SE · DB) → 생성 모델이 창 25개를 이어 그린다 → 세트 폴더 | 보이는 GPU 전부 | 600 장에 70분 (GPU 2대, GPU 하나에서 장당 14초 — 2026-11-19) |
| `check` | 세트의 점검: K1 확대 시트 (사람이 본다), K4 창 이음매, K2 · K3 · K6 분할 모델의 인식률, K5 진짜 · 합성을 가르는 분류기와 특징 거리 | 1대 (시트 · 이음매는 CPU) | 600 장에 약 10분 (시트 15초, 이음매 1분, 인식률 모델 하나에 4분, 분류기 1분) |

## 환경변수

| 변수 | 기본값 | 뜻 |
|---|---|---|
| `GENAUG_PHASE` | (없음) | 위 표 |
| `GENAUG_EXTRA` | (없음) | 스크립트에 그대로 넘기는 인자 |
| `NVIDIA_VISIBLE_DEVICES` | `2,3` | 쓸 GPU. GPU 0 · 1 에는 서빙 컨테이너 (`defsegam-api-v4`, `diva-sam-service`) 가 있다 |
| `NPROC` | `2` | 학습 프로세스 수 = GPU 수. `NVIDIA_VISIBLE_DEVICES` 의 개수와 맞춘다 (`synth` 는 이 값과 무관하게 보이는 GPU 를 다 쓴다) |
| `GENAUG_SET` | `w600_v1` | `check` 가 볼 합성 세트의 이름 (`synth` 의 이름은 `GENAUG_EXTRA` 의 `--name`) |

`GENAUG_EXTRA` 에 넣을 수 있는 `train` 의 인자 (기본값은 `config_genaug.py`):

| 인자 | 기본값 | 뜻 |
|---|---|---|
| `--quick` | – | 스모크. 작은 모델 · 64 창 · 30 step. 산출물은 `fill_w256_quick` 에 따로 쌓인다 |
| `--resume` | – | `checkpoints/genaug/<run>/last.pt` 에서 이어 돌린다 |
| `--steps` | 100000 | 학습 길이 |
| `--normal-builds` | `B3` | 라벨 맵에 Powder / Printed 를 적어 줄 빌드. `none` = 적지 않는다 |
| `--train-builds` | `B3 B4 B5` | 주석 없는 배경으로 쓸 빌드. `none` = 사람 GT 창만 |
| `--holdout` | `both` | 보지 않을 사람 GT. `both` = 두 판정 fold 의 검증 레이어를 뺀 87 레이어로 학습, `inlayer` = 103 레이어 |
| `--run-name` | `fill_w256` | 산출물 폴더 이름 |
| `--no-amp` | – | bfloat16 을 끄고 fp32 로 (GPU 당 31 GB — 권하지 않는다) |
| `--batch-size` | 16 | GPU 당. 32 는 32 GB 에 들어가지 않는다 |

`GENAUG_EXTRA` 에 넣을 수 있는 `synth` 의 인자 (기본값은 `config_genaug.py` 의 `SET_*`):

| 인자 | 기본값 | 뜻 |
|---|---|---|
| `--quick` | – | 스모크. 6 장 · 8 단계, 세트 이름에 `_quick`. 그림의 질에는 뜻이 없다 |
| `--name` | `w600_v1` | 세트 이름 → `cache/genaug/sets/<이름>/` |
| `--n-layers` | 600 | 장 수. 늘려서 다시 돌리면 없는 장만 더 만든다 (레이어 i 는 seed 와 i 로만 정해진다) |
| `--classes` | `RS RH SE DB` | 넣을 결함 (이 넷 가운데) |
| `--builds` | `B3 B4 B5` | 배경 (CAD 파트 배치 + 사진 가장자리) 으로 쓸 빌드. 생성 모델이 본 빌드만 받는다 |
| `--guidance` | 2.0 | 라벨을 따르는 세기 |
| `--steps` · `--spacing` | 50 · `quad` | 그리는 단계 수와 배치. `quad` = 잡음이 적은 구간에 몬다 (분말의 결이 진짜의 0.45 → 0.60) |
| `--order` · `--stride` | `spiral` · 192 | 창을 그리는 순서 (바깥에서 안쪽으로 / `raster` = 줄 따라) 와 창 간격의 상한 |
| `--batch` | 8 | 한 번에 그리는 레이어 수 (GPU 당. 묶음 8 장에 3.3 GB) |
| `--seed` | 20261119 | 세트의 seed |
| `--overwrite` | – | 같은 이름의 세트를 지우고 다시 만든다 |

설정이 다른 채로 같은 이름을 주면 멈춘다 (`set_config.json` 과 견준다). 라벨 맵의 규칙은 `synth/label_map.py` 의 `Rules` 에 있다 (v7/PLAN_v7.md §2.2).

## 풀런 전에 정할 것 — 정상 라벨을 적어 줄 빌드 (Q0)

생성 모델은 "Powder 라고 적으면 이 장비의 깨끗한 분말을 그린다" 는 것을 `--normal-builds` 의 창에서 배운다.
그 빌드의 사진에 진짜 결함이 많으면 "정상" 을 그리라고 했을 때 그것도 함께 그리게 된다 (계획서 §3.2, §3.9).

- `figures/genaug/q0/B{1..5}_sheet.png` 를 연다. 칸 하나가 창 하나이고 왼쪽이 visible/0, 오른쪽이 visible/1, 하늘색 선이 CAD 의 파트 경계다.
- 계획서의 기준은 "눈에 띄는 결함이 있는 칸이 100개 가운데 5개 이하" 다.
- 정한 결과를 `GENAUG_EXTRA="--normal-builds B3"` 처럼 넘긴다. 어느 빌드도 쓰지 않기로 하면 `--normal-builds none` 이고,
  그때는 ORNL 배경의 "창 전체 그리기" (계획서 §3.9) 를 쓸 수 없다. 결함 자리만 그리는 방식은 그대로 쓸 수 있다.
- 구현하면서 시트를 일부 (빌드당 12–24칸) 들여다본 바로는 계획서의 "상태" 에 적었다. 판정은 아니다.

## 산출물

| 경로 | 내용 |
|---|---|
| `checkpoints/genaug/<run>/last.pt` | 이어 돌리기용 — 학습용 가중치 · EMA 사본 · optimizer · step (약 1.9 GB) |
| `checkpoints/genaug/<run>/ema_step_NNNNNN.pt` | EMA 사본만. 25,000 step 마다와 끝에 남긴다 (약 0.5 GB) |
| `figures/genaug/<run>/samples/step_NNNNNN.png` | 표본 시트 — 5,000 step 마다 |
| `figures/genaug/<run>/run.json` | 설정, 읽은 사람 GT 의 목록, 객체 수 |
| `figures/genaug/<run>/curve.csv` | 손실 곡선 (step, 손실, 창 종류별 오차, 속도, 메모리) |
| `figures/genaug/q0/` | 배경 점검 시트와 창 목록 |
| `cache/ornl_canvas_sz1036/<빌드>/` | 공용 배경 캐시 |
| `cache/genaug/sets/<세트>/` | 합성 세트 — `visible/{0,1}/{id}.png`, `annotations/{id}.npy` (가장자리와 파트 경계는 −1), `index.csv`, `classes.json` (사람 GT 데이터셋과 같은 구성 — v7 이 v6 의 로더로 읽는다), `meta/{id}.json` (장마다의 내력), `set_config.json`, `manifest.json` (세트 전체의 내력 · 클래스별 화소 수 · 쓸 수 있는 판정 fold) |
| `figures/genaug/sets/<세트>/` | 점검 — `overview_<빌드>.png` · `zoom_<클래스>.png` · `zoom_normal.png` (K1), `seam.json` (K4), `recognition_<모델>.json` (K2 · K3 · K6), `probe.json` (K5) |

**표본 시트를 읽는 법.** 한 줄이 창 하나다: 진짜 visible/0 · 진짜 visible/1 · 라벨 맵과 그릴 자리 (노란 선) · 그린 visible/0 · 그린 visible/1.

| 줄의 머리 | 무엇을 그렸나 |
|---|---|
| `A GT redraw` | 사람 GT 의 결함을 제자리에서 다시 그렸다. 진짜와 닮았는지 본다 |
| `B GT window` | 사람 GT 창을 라벨 맵만 보고 창 전체 다시 그렸다 |
| `C ORNL insert` | 주석 없는 ORNL 창에 결함 하나를 넣었다 — 실제로 쓸 방식 |
| `D ORNL window` | ORNL 창을 라벨 맵 (CAD 의 파트 배치 + 결함 하나) 대로 창 전체 그렸다 |

그릴 자리 밖은 원본과 한 값도 다르지 않다. 손실 값은 그림의 질을 알려 주지 않으므로 시트를 본다.

## 스모크에서 확인한 것 (2026-11-17)

| 실행 | 확인한 것 |
|---|---|
| `canvas` (묶음 2개로 먼저, 그다음 다섯 빌드 전부) | 캐시의 사진과 마스크가 HDF5 에서 다시 만든 것과 화소 단위로 같다 (빌드당 3 레이어). 0–255 밖의 값 · 정수가 아닌 값 0개 |
| `selftest` | holdout 38 레이어가 읽은 87 레이어와 겹치지 않는다. 그릴 자리 밖의 라벨이 전부 "모름". 그린 뒤 그릴 자리 밖의 화소가 그대로다 (강도 0 · 0.5 · 1) |
| `train --quick`, GPU 1대 | 학습 → 저장 → 표본 시트 |
| 실제 크기 3,000 step, GPU 2대 (`--run-name sanity`) | 초당 3.55 step, GPU 당 23.8 GB (bfloat16). 손실이 내려간다. 표본은 `figures/genaug/sanity/samples/` |
| `train --resume`, GPU 1대 (3,000 → 3,040 step) | 이어 돌린 직후의 손실이 끊기 전과 같은 수준이다 (0.003). 곡선 파일에 이어서 쓴다 |
| `sample` (`--per-class 1 --seed 7 --guidance 3`) | 저장된 모델을 불러와 다른 창으로 시트를 만든다 |
| 실제 크기, GPU 1대 | fp32 는 batch 16 에 31.4 GB · 초당 2.55 step, bfloat16 은 23.3 GB · 3.69 step. 두 쪽의 손실 값은 같았다. batch 32 는 들어가지 않는다 |

**합성 세트 쪽 (2026-11-19)**

| 실행 | 확인한 것 |
|---|---|
| `selftest` 의 6번 (CPU) | 창의 자리와 순서가 안쪽을 다 덮는다. 라벨 맵의 값이 Powder · Printed · RS · RH · SE · DB · "모름" 뿐이고, 줄 (RS · RH) 은 CAD 의 파트 밖에만 있고, 결함은 안쪽에만 놓인다. 같은 seed 는 같은 라벨 맵을 낸다 |
| `synth --quick` (GPU 2대) | 6 장 · 8 단계. 그린 뒤 가장자리 화소를 진짜 사진과 화소 단위로 견준다 (코드 안의 확인). 세트가 v7 의 로더로 읽힌다 |
| 같은 명령을 다시 / 설정을 바꿔서 / `--overwrite` | 전부 건너뛴다 / FATAL 로 멈춘다 / 지우고 다시 만든다 |
| `check` 의 네 스크립트 (quick 세트) | 시트 · 이음매 · 인식률 (모델 둘) · 분류기가 끝까지 돈다 |
| v7 의 `cv --quick` 에 quick 세트를 넣어서 | 두 판정 fold 에서 학습 → 채점 → 비교 표까지 돈다. `valid_folds` 에 없는 fold 를 주면 FATAL |
| 그리는 설정의 비교 (세트의 레이어 8 장, 같은 라벨 맵 · 같은 seed) | 단계의 배치 `quad` 가 `uniform` 보다 분말의 결을 살린다 (진짜의 0.60 대 0.45). 단계마다 잡음을 다시 섞는 것 (η 0.5 · 1) 과 guidance 를 1 로 낮추는 것은 0.60 · 0.54 · 0.65 로 차이가 작다. 그리는 순서 (`spiral` · `raster`) 는 이음매가 비슷했다. 그래서 기본값은 `quad` · η 0 · guidance 2 · `spiral` 이다 |
| 본 세트 `w600_v1` | GPU 2대로 600 장에 74분 (GPU 하나에서 장당 14초, 묶음 8 장에 3.3 GB), 1.1 GB. 점검 결과는 v7/PLAN_v7.md 의 "구현 상태 → 합성 세트 `w600_v1`" |

`--quick` 의 수치에는 뜻이 없다. 3,000 step 의 표본도 "배선이 맞다" 는 뜻일 뿐 그림의 질을 말해 주지 않는다 — 계획한 길이의 3% 다.
그 시점의 모델은 큰 모양 (파트의 윤곽, 어두운 점, SE 자리의 밝은 점) 은 그리지만 분말의 고운 결은 아직 그리지 못해, 창 전체를 그린 줄 (`D`) 이 진짜보다 매끈하다.
시험에 쓴 ckpt 는 지웠고 표본 시트만 남겼다.

## 주의

- **GPU 를 나눠 쓴다.** 학습은 GPU 당 24 GB 를 쓴다. GPU 2 · 3 에서 분할 모델 학습 (v5 · v6, GPU 4대) 이나 다른 스모크가 돌고 있으면 겹친다. `nvidia-smi` 로 먼저 본다.
- **누출 금지.** 생성 모델은 `splits/holdout_v1.json` 의 38 레이어 (inlayer 의 검증 22 + L21 의 20) 를 읽지 않는다. 읽은 레이어의 목록은 `run.json` 의 `gt_keys` 에 남는다.
- **DSCNN 예측을 읽지 않는다.** 배경 캐시는 사진과 `slices/part_ids` 만 읽어 만든다. v1 캐시의 `label.npy` · `defect_ratios` 는 쓰지 않는다.
- **합성 세트를 분할 모델의 예측으로 한 장씩 고르지 않는다.** `check` 의 인식률은 클래스를 통째로 넣을지 뺄지만 정한다 (v7/PLAN_v7.md §2.4).
- **세트를 쓸 수 있는 판정 fold 는 생성 모델의 holdout 으로 정해진다.** `--holdout both` 로 학습한 생성 모델의 세트는 `inlayer` · `loso_L21` 에 쓸 수 있고 (`manifest.json` 의 `valid_folds`), v7 의 학습은 그 밖의 fold 에 넣으면 멈춘다.
