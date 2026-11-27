# DefSeg-AM v5 — Docker 사용법

> v5 = v4 의 11-class 그대로 + **사람 GT 만으로** 단일 단계 학습 (Stage 1 KD 없음). 설계는 [../PLAN_v5.md](../PLAN_v5.md).
> 호스트의 `DefSeg_AM/venv` (torch 2.9.1+cu128) 를 read-only bind mount 하여 그대로 쓴다. 이미지에 `pip install` 안 함.

## 사전조건

- NVIDIA driver + nvidia-container-toolkit (host), Docker / docker compose v2
- `DefSeg_AM/venv`
- `ORNL_Data/DSCNN_Dataset` — 사람 GT (학습 데이터). 없으면 `cv` · `train` · `eval` 이 FATAL 로 끝난다
- `confusion` phase 만: v1 이미지 캐시 `cache/resized_sz1036/B1` + v4 라벨 캐시 `cache/resized_sz1036_v4` (클래스 공간이 같아 그대로 읽는다)
- `infer` phase 만: `ORNL_Data/Co-Registered …/[baseline] (Peregrine v2023-11)/*.hdf5`
- 호스트 UID/GID 매칭용 `.env` (없으면 1000:1000 으로 돌아 산출물 쓰기에 실패한다)

```bash
cd DefSeg_AM/v5/docker
echo "UID_GID=$(id -u):$(id -g)" > .env
```

학습과 검증은 캐시를 만들지 않는다. 사람 GT 125 레이어를 시작할 때 한 번 읽어 1036² 로 줄여 RAM 에 둔다 (프로세스당 약 0.4 GB, 20초 안팎).

## 권장 순서

```bash
cd DefSeg_AM/v5/docker

# 1) 검증 fold 하나를 먼저 — 실제 길이로 끝까지 도는지, 곡선이 어떤 모양인지 본다 (4 GPU, 약 2시간)
DEFSEG_V5_PHASE=cv DEFSEG_V5_FOLDS=loso_L21 docker compose up -d --build
docker compose logs -f
#    결과: figures/<run>/v5/cv/{summary.md, curves.png}

# 2) 나머지 fold (약 8시간) — 끝나면 summary.md 의 '권장 길이' 절이 채워진다
DEFSEG_V5_PHASE=cv DEFSEG_V5_FOLDS="loso_MS loso_I2 loso_GP inlayer" docker compose up -d --build
#    요약만 다시:  DEFSEG_V5_PHASE=summarize docker compose up --build

# 3) 전수 학습 (8 소스 125 레이어) → ORNL Build 1 참고 평가 → figure (약 2시간)
#    배포 후보 checkpoints/<run>/full/last.pt
DEFSEG_V5_PHASE=all docker compose up -d --build
#    따로 돌리려면 train → confusion → infer 순서로 하나씩

# 정리
docker compose down
```

- **한 phase 가 끝난 뒤에 다음 phase 를 띄운다.** 도는 중에 `docker compose up` 을 다시 치면 컨테이너가 교체되어 학습이 끊긴다.
- **컨테이너 로그는 다음 `up` 에서 지워진다.** 학습 로그를 파일로 따로 쓰지 않으므로, 남기려면 다음 phase 를 띄우기 전에
  `docker compose logs --no-log-prefix > ../../logs/<이름>.log` 로 받아 둔다. 수치 (`curve.csv`, `result.json`, `metrics.json`) 는 로그와 무관하게 남는다.

### 길이를 바꿀 때

바꾸지 않으면 기본값 (10,000 step, EMA decay 0.999) 으로 돈다. 바꿀 때는 `cv` · `train` · `all` 에 같은 값을 붙인다
(아래 값은 예시. EMA 규칙: `1 / (1 − decay) ≤ 총 step / 10`).

```bash
DEFSEG_V5_PHASE=all DEFSEG_V5_EXTRA="--total-steps 6000 --ema-decay 0.998" docker compose up -d --build
```

- 값은 `summary.md` 의 '권장 길이' 절에서 가져온다. 이 절은 **loso fold 가 둘 이상** 끝나야 채워진다 (fold 들의 평균 곡선으로 낸다).
  fold 하나만 돌린 뒤에는 표의 best step 과 `curves.png` 를 직접 본다.
- 평균 곡선이 최고점 뒤로 5% 넘게 내려가지 않으면 줄이라고 하지 않는다 (과적합으로 보지 않는다).
- 길이를 바꾸면 검증 fold 도 모두 그 길이로 다시 돌려야 표와 곡선이 그 길이의 결과가 된다.
  fold 폴더가 길이별로 나뉘지 않아, 일부만 다시 돌리면 한 표에 길이가 섞인다.

## Phase

`DEFSEG_V5_PHASE` 를 주지 않으면 사용법만 찍고 끝난다 (실수로 학습이 시작되지 않게).

| phase | 하는 일 | GPU |
|---|---|---|
| `cv` | `DEFSEG_V5_FOLDS` 의 fold 를 차례로 학습 → fold 마다 native 해상도 채점 (v5 + v4 참고) → 요약 | NPROC 개 |
| `summarize` | 검증 fold 요약 (`summary.{md,csv,json}`, `curves.png`) 만 다시 | 불필요 |
| `eval` | 검증 fold 의 native 채점만 다시. `DEFSEG_V5_EXTRA="--tta --postprocess"`, `--which best`, `--save-figs 6` 등 | 1개 |
| `train` | 전수 학습 → `checkpoints/<run>/full/{last.pt, last_raw.pt}` | NPROC 개 |
| `confusion` | 전수 학습 모델을 ORNL Build 1 캐시로 채점 — **정답이 아니라 DSCNN 예측과의 일치도** (v1–v4 표와 나란히 보기 위한 참고) | 1개 |
| `infer` | 전수 학습 모델로 ORNL Build 1 의 12 레이어 4-panel figure (TTA + 후처리) | 1개 |
| `all` | `train → confusion → infer`. `cv` 는 포함하지 않는다 | NPROC 개 |

## 환경변수

| 변수 | 기본 | 의미 |
|---|---|---|
| `DEFSEG_V5_PHASE` | (없음) | 위 표 |
| `DEFSEG_V5_FOLDS` | `loso_L21 loso_MS loso_I2 loso_GP inlayer` | `cv` · `eval` 이 도는 fold (공백으로 구분) |
| `DEFSEG_V5_SOOT` | `ignore` | Soot 주석을 어디로 보내는가 (2026-11-20 추가). `ignore` = 기존 v5 (미레이블). `spatter` = Spatter 로 배운다 — v6 가 Soot 를 Spatter 로 읽으므로 두 버전을 Spatter 에서 견주려는 실행이다. 이때 run 이름에 `-soot2sp` 가 붙어 (`…_11cls_v5-soot2sp`) 기존 v5 의 산출물과 섞이지 않는다. 학습 · 채점 · 요약에 같은 값을 줘야 한다. 예: `DEFSEG_V5_PHASE=cv DEFSEG_V5_FOLDS=inlayer DEFSEG_V5_SOOT=spatter docker compose up -d --build` |
| `DEFSEG_V5_EXTRA` | (empty) | `cv` · `train` 에서는 학습 스크립트 인자 (`--quick`, `--total-steps`, `--ema-decay`, `--seed`, `--no-copy-paste` …), `eval` · `confusion` · `infer` 에서는 그 스크립트 인자 |
| `NVIDIA_VISIBLE_DEVICES` | `0,1,2,3` | 사용할 GPU |
| `NPROC` | `4` | DDP world size (= GPU 개수) |
| `OMP_NUM_THREADS` | `8` | per-process CPU threads |

### Fold 이름

| 이름 | 검증용으로 빼는 것 | 채점되는 결함 클래스 |
|---|---|---|
| `loso_L21` | v2021_LPBF 전체 (20 레이어) | RS IS Sw DB SE |
| `loso_MS` | v2022_Maraging 전체 (26) | RS IS Sw DB (RH 는 이 소스에만 있어 채점 제외) |
| `loso_I2` | v2022_Inc718_2 전체 (24) | RS Sw DB SE (OM 은 이 소스에만 있어 제외) |
| `loso_GP` | v2022_GammaPrint 전체 (9) | RS IS Sw SE SP (UM 은 이 소스에만 있어 제외) |
| `inlayer` | 소스마다 레이어 5개 중 1개 (22 레이어) | 결함 9종 전부 — 같은 빌드의 이웃 레이어가 학습에 있으므로 낙관적인 값 |
| `loso_BJ21` `loso_17-4` `loso_I1` `loso_H13` | 나머지 4 소스 (기본 목록에는 없다) | RS IS DB / IS Sw / Sw SP / RS IS DB |

## GPU

- **이 compose 는 `NVIDIA_VISIBLE_DEVICES` 가 실제로 먹는다.** v3 · v4 의 compose 에 있는
  `deploy.resources.reservations.devices (count: all)` 블록을 뺐다. 그 블록이 있으면 컨테이너 안에서 GPU 4개가 다 보이고
  프로세스가 항상 GPU 0 부터 올라간다 (2026-11-16 확인).
- 학습은 GPU 당 약 23.6 GB 를 쓴다 (1036², GPU 당 4장, AMP 끔 — 2026-11-16 에 GPU 2 · 3 에서 측정).
  GPU 0 · 1 에는 서빙 컨테이너 (`defsegam-api-v4`, `diva-sam-service`) 가 3–4 GB 씩 올라가 있어 합치면 27 GB 안팎 (32 GB 중) 이다.
  학습 중에 DIVA 가 `/predict/batch` 를 크게 보내면 OOM 이 날 수 있다.
- GPU 를 줄여 돌리면 effective batch 가 `4 × NPROC` 로 줄어든다 (계획값 16). 예: `NVIDIA_VISIBLE_DEVICES=2,3 NPROC=2`.

## 소요 시간

실제로 잰 값 (2026-11-17, GPU 4장, `loso_L21`, 10,000 step): 학습 118분 (초당 1.46 step, 250 step 마다 검증 6초 포함),
그 뒤의 native 채점 (v5 + v4 참고, 20 레이어) 과 요약은 합쳐서 22초.

| 실행 | 횟수 | 기본 길이로 |
|---|---:|---|
| 검증 fold | 5 | 약 10시간 (fold 당 2시간) |
| 전수 학습 | 1 | 약 2시간 |
| confusion (Build 1 3,218 레이어) / infer | 1 / 1 | 수 분 (추정 — 실제 길이로는 아직 재지 않았다) |

## 스모크 검증 (학습 X)

```bash
# 1 GPU + quick (6 step, 224², 소스당 3 레이어) — fold 학습 → native 채점 → 요약
DEFSEG_V5_PHASE=cv DEFSEG_V5_FOLDS=loso_L21 NPROC=1 NVIDIA_VISIBLE_DEVICES=2 \
  DEFSEG_V5_EXTRA=--quick docker compose up --build

# 2 GPU DDP + quick — 전수 학습 → confusion (2 배치) → infer (1 레이어)
DEFSEG_V5_PHASE=all NPROC=2 NVIDIA_VISIBLE_DEVICES=2,3 \
  DEFSEG_V5_EXTRA=--quick docker compose up --build
```

`--quick` 산출물은 run-name 이 `vitb14_reg_dpt_dual_sz224_11cls_v5` 라 실제 결과 (`…sz1036…`) 와 섞이지 않는다.

## 산출물

```
checkpoints/vitb14_reg_dpt_dual_sz1036_11cls_v5/
├── full/{last.pt, last_raw.pt}            # 전수 학습: EMA 사본 (배포 후보) / 학습용 가중치
└── cv/<fold>/{best.pt, last.pt}           # 검증 fold
figures/vitb14_reg_dpt_dual_sz1036_11cls_v5/v5/
├── cv/{summary.md, summary.csv, summary.json, curves.png}
├── cv/<fold>/{run.json, curve.csv, result.json}
├── cv/<fold>/{eval_native_last, ref_v4}/{metrics.json, per_class_metrics.csv, confusion_*.{csv,npy,png}}
└── full/{run.json, curve.csv, result.json, build1_vs_dscnn/, inference_last_tta_pp/<build>/layerXXXX.png}
```

ckpt 형식은 v4 와 같다 (`model_state` = trainable weight 만, `config.backbone` · `n_classes` · `img_size`). 클래스 11개와 번호도 같다.
