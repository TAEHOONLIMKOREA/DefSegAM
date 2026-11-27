# DefSeg-AM v6 — Docker 사용법

> v6 = DIVA 레이블 체계 `pbf-2026.11` (10-class) + **사람 GT 만으로** 단일 단계 학습 (v5 의 학습 방식 그대로). 설계는 [../PLAN_v6.md](../PLAN_v6.md).
> 호스트의 `DefSeg_AM/venv` (torch 2.9.1+cu128) 를 read-only bind mount 하여 그대로 쓴다. 이미지에 `pip install` 안 함.
>
> 지금 있는 것은 **P0 까지**다 (Peregrine 주석만으로 학습 · 채점 · 기준선). DIVA 내보내기를 읽는 쪽 (`build_cache`, `split`) 은 내보내기가 온 뒤 P1 에서 만든다.

## 사전조건

- NVIDIA driver + nvidia-container-toolkit (host), Docker / docker compose v2
- `DefSeg_AM/venv`
- `ORNL_Data/DSCNN_Dataset_pbf-2026.11` — Peregrine 주석을 새 체계로 변환한 데이터셋 (학습 데이터). 2026-11-17 에 만들어 두었다.
  없으면 `cv` · `train` · `eval` · `baselines` · `review` 가 FATAL 로 끝난다. 다시 만드는 법은 아래 "데이터셋을 다시 변환할 때".
- 기준선 R2 · R3 (있으면 채점하고 없으면 건너뛴다): v5 의 같은 fold ckpt `checkpoints/vitb14_reg_dpt_dual_sz1036_11cls_v5/cv/<fold>/last.pt`,
  v4 의 `checkpoints/vitb14_reg_dpt_dual_sz1036_11cls_v4/stage2_best.pt`
- `infer` phase 만: `ORNL_Data/Co-Registered …/[baseline] (Peregrine v2023-11)/*.hdf5`
- 호스트 UID/GID 매칭용 `.env` (없으면 1000:1000 으로 돌아 산출물 쓰기에 실패한다)

```bash
cd DefSeg_AM/v6/docker
echo "UID_GID=$(id -u):$(id -g)" > .env
```

학습과 검증은 캐시를 만들지 않는다. 125 레이어를 시작할 때 한 번 읽어 1036² 로 줄여 RAM 에 둔다 (프로세스당 약 0.4 GB, 20초 안팎).

## P0 — 첫 실험 (PLAN_v6 §7.1)

```bash
cd DefSeg_AM/v6/docker

# inlayer fold 학습 (103 레이어) → 22 레이어 채점 → 기준선 R1–R3 → 요약.  4 GPU, 약 2시간
DEFSEG_V6_PHASE=cv DEFSEG_V6_FOLDS=inlayer docker compose up -d --build
docker compose logs -f
#   결과: figures/vitb14_reg_dpt_dual_sz1036_10cls_v6/v6/cv/{summary.md, curves.png, inlayer/class_curves.png}

# 정리
docker compose down
```

- 시작하고 1분 안에 로그의 `배선 확인` 줄을 본다. 클래스별 화소 수가 계획의 표 (`splits/legacy_inlayer_v1.json` 의 `expected`) 와 같으면
  `배선 확인: … 같다`, 다르면 경고가 나온다 (`run.json` 의 `wiring_check` 에도 남는다). 데이터셋을 다른 옵션으로 다시 변환했을 때만 달라야 한다.
- **한 phase 가 끝난 뒤에 다음 phase 를 띄운다.** 도는 중에 `docker compose up` 을 다시 치면 컨테이너가 교체되어 학습이 끊긴다.
- **컨테이너 로그는 다음 `up` 에서 지워진다.** 남기려면 다음 phase 를 띄우기 전에
  `docker compose logs --no-log-prefix > ../../logs/<이름>.log` 로 받아 둔다. 수치 (`curve.csv`, `result.json`, `metrics.json`) 는 로그와 무관하게 남는다.
- `docker compose up` 이 `Found orphan containers ([docker-defseg-am-v5-1])` 라고 알려도 그대로 둔다. v5 와 프로젝트 이름 (`docker`) 이 같아서 나오는 안내다.
  `--remove-orphans` 를 붙이면 v5 의 컨테이너와 그 로그가 지워진다.
- 같은 fold 를 다시 돌리면 그 fold 폴더의 결과를 덮어쓴다. 판정표를 고친 뒤 다시 돌릴 때처럼 앞의 결과를 남기려면 fold 폴더
  (`figures/<run>/v6/cv/inlayer`, `checkpoints/<run>/cv/inlayer`) 의 이름을 먼저 바꿔 둔다.

## 전수 학습과 ORNL 비교

```bash
cd DefSeg_AM/v6/docker

# 전수 학습 — Peregrine 주석 125 레이어 전부, 검증 없음.  4 GPU, 약 2시간 (2026-11-17 실측 114분)
DEFSEG_V6_PHASE=train docker compose up -d --build
docker compose logs --no-log-prefix > ../../logs/<이름>.log      # 다음 up 전에 로그를 받아 둔다

# ORNL 5 빌드에서 무작위 100 레이어 (빌드마다 20) — DSCNN · v4 · v5 · v6 예측을 나란히.  GPU 1장, 약 5분
DEFSEG_V6_PHASE=ornl_compare NVIDIA_VISIBLE_DEVICES=2 NPROC=1 docker compose up --build
#   결과: figures/vitb14_reg_dpt_dual_sz1036_10cls_v6/v6/full/ornl_random100_last/{overview.png, pages/page_XX_<빌드>.png, summary.png, summary.md}

# 모델을 다시 돌리지 않고 그림 · 수치만 다시 (위 실행이 남긴 preds/ 를 읽는다. GPU 를 쓰지 않는다, 약 2.5분)
DEFSEG_V6_PHASE=ornl_compare NVIDIA_VISIBLE_DEVICES=2 NPROC=1 DEFSEG_V6_EXTRA="--from-preds --workers 8" docker compose up --build

# 전수 학습 모델을 학습에 쓴 125 레이어에서 채점 (암기 정도 — 성능이 아니다).
#   GPU 로는 아직 재지 않았다 (fold 채점이 22 레이어에 12초였으니 1–2분으로 어림한다). GPU 없이 CPU 로는 10분 34초였다 (2026-11-18, 16 스레드)
DEFSEG_V6_PHASE=trainfit NVIDIA_VISIBLE_DEVICES=2 NPROC=1 docker compose up --build
DEFSEG_V6_PHASE=trainfit NVIDIA_VISIBLE_DEVICES=none NPROC=1 OMP_NUM_THREADS=16 docker compose up --build    # GPU 가 다 쓰이고 있을 때
```

- **점수가 아니다.** ORNL 빌드에는 사람 GT 가 없어 네 열 (DSCNN · v4 · v5 · v6) 이 모두 예측이다. 일치도 수치는 어떤 결정에도 쓰지 않는다 (PLAN_v6 §6.5).
  모델 예측이 아닌 기준은 CAD 파트 마스크 (`slices/part_ids`) 뿐이라 그것과의 겹침을 함께 낸다.
- 옛 예측 (DSCNN 12-class, v4 · v5 11-class) 은 새 id 로 옮겨 같은 색으로 그린다 — Swelling → SE, 옛 IS → IS, Misprint · Under Melting → 대응 없음 (검은색).
- 입력 밝기는 모델마다 학습 때 본 대로다: v5 · v6 는 원본 그대로, v4 는 레이어마다 늘린 사진. TTA · 후처리는 걸지 않는다 (`--tta` 를 주면 세 모델 모두 8-way 평균).
- 레이어는 빌드마다 seed (기본 42) 로 섞은 순서에서 CAD 파트 마스크와 두 사진이 비어 있지 않은 것을 앞에서부터 고른다. 뽑은 목록은 `layers.json` 에 남는다.
- 옵션 (`DEFSEG_V6_EXTRA`): `--n-per-build 20`, `--builds B1 B2 …`, `--seed`, `--layers-json <앞서 뽑은 layers.json>` (같은 레이어로 다시), `--fold inlayer` (fold 모델로),
  `--which`, `--tta`, `--workers 6` (HDF5 를 읽는 프로세스 수), `--no-save-preds`, `--from-preds` (앞선 실행의 `preds/` · `layers.json` · `summary.json` 으로 그림과 수치만 다시 — 모델도 GPU 도 쓰지 않는다).
- 그림의 예측 타일은 원본 (1842²) 의 3×3 칸마다 클래스 하나를 보인다 — 분말이 아닌 화소가 3개 이상이면 그 가운데 가장 많은 클래스, 아니면 분말.
  색을 평균하면 가는 Printed (파랑) 와 Spatter (자홍) 가 섞여 PS 의 범례 색 (보라) 이 되기 때문이다. 그래도 그림을 줄여 보면 얇은 고리가 보라로 보일 수 있다 (원래 크기로 보면 두 색이 따로다).
- `summary.md` 에는 클래스별 예측 화소 비율 · v6 와의 일치 · CAD 파트 마스크와의 겹침 말고도 두 가지가 더 있다: **RS 선의 가로 길이** (RS 가 있는 행에서 행 폭의 얼마를 RS 로 냈는가)
  와 **고정 위치의 Debris** (뽑힌 레이어의 절반 넘게 같은 화소에서 Debris 로 난 곳 — 카메라 쪽의 점. 레이어가 20개 넘을 때만 낸다).
- v4 · v5 의 ckpt (`…_11cls_v4/stage2_best.pt`, `…_11cls_v5/full/last.pt`) 가 없으면 그 열만 빠진다.
- ORNL HDF5 는 레이어 56개씩 묶여 압축돼 있어 한 레이어를 읽는 데 사진 한 장 1.8초, 마스크 하나 0.5초가 걸린다. 그래서 읽기를 여러 프로세스로 나눈다.

## Phase

`DEFSEG_V6_PHASE` 를 주지 않으면 사용법만 찍고 끝난다 (실수로 학습이 시작되지 않게).

| phase | 하는 일 | GPU |
|---|---|---|
| `cv` | `DEFSEG_V6_FOLDS` 의 fold 를 차례로 학습 → fold 마다 채점 (원본 해상도 + 1036²) → 기준선 R1–R3 → 요약. **P0 = `cv` + `inlayer`** | NPROC 개 |
| `summarize` | 검증 fold 요약 (`summary.{md,csv,json}`, `curves.png`, fold 마다 `class_curves.png`) 만 다시 | 불필요 |
| `eval` | 검증 fold 의 채점만 다시 → 요약. `DEFSEG_V6_EXTRA="--which best"`, `"--tta --postprocess"`, `"--save-figs 22"` 등 | 1개 |
| `baselines` | 기준선만 (→ 요약). 학습 전에도 낼 수 있다 — 2026-11-17 에 inlayer 것을 내 두었다 | 1개 |
| `review` | 기존 IS 판정의 검토 시트 (`figures/<run>/dataset_checks/review_sheet_*.png`). `DEFSEG_V6_EXTRA=--uncertain-only` 는 확신이 중간 이하인 레이어만 | 불필요 |
| `train` | 전수 학습 (Peregrine 주석 125 레이어, 검증 없음) → `checkpoints/<run>/full/{last.pt, last_raw.pt}` — 계획의 A2 | NPROC 개 |
| `trainfit` | 전수 학습 모델을 학습에 쓴 125 레이어에서 채점한다 → `figures/<run>/v6/full/trainfit_{native,resized}_last`. **암기 정도이지 성능이 아니다** — 검증을 떼지 않은 모델이 멀쩡한지, 무엇을 외우지 못했는지 보는 용도 | 1개 (없어도 된다) |
| `infer` | ORNL Build 1 의 12 레이어 4-panel figure (참고용). 기본은 전수 학습 모델, `DEFSEG_V6_EXTRA="--fold inlayer"` 로 fold 모델. `--tta --postprocess`, `--build`, `--layers` | 1개 |
| `ornl_compare` | ORNL 5 빌드에서 무작위로 뽑은 레이어 (기본 빌드마다 20 = 100) 에서 DSCNN · v4 · v5 · v6 예측을 나란히 그리고 모델끼리 견준 수치를 낸다 (참고용 — 점수가 아니다. 아래 "전수 학습과 ORNL 비교") | 1개 |
| `convert` | Peregrine 주석을 다시 변환 (아래). `ORNL_DATA_MODE=rw` 와 함께 | 불필요 |

### 기준선 (`cv` 가 끝에 함께 낸다)

| | 무엇 | 폴더 (fold 폴더 안) |
|---|---|---|
| R1 복사 기준선 | 검증 레이어마다 같은 소스의 가장 비슷한 학습 레이어의 정답을 그대로 예측으로 낸 것. 모델이 없다. 1036² 에서만 낸다. 소스를 통째로 뺀 fold (loso) 에서는 낼 수 없어 건너뛴다 | `ref_copy_resized` |
| R2 v5 | 같은 이름의 v5 fold ckpt 의 11-class 예측을 새 id 로 옮겨 채점 (Swelling 예측 → SE) | `ref_v5_{native,resized}` |
| R3 v4 | `stage2_best.pt` 를 같은 방법으로. v4 는 이 레이어들을 학습에 썼다 — 참고만 | `ref_v4_{native,resized}` |

- v6 모델의 IoU 가 R1 보다 낮은 클래스는 가장 쉬운 조건에서도 배우지 못한 것이다.
- R2 · R3 는 PS 를 낼 수 없고 Soot 를 Spatter 로 배우지 않았다. IS · PS · SP 와 결함 mIoU 는 그대로 견주지 않는다 (`summary.md` 의 각주).
- `native` = 원본 해상도 (서빙과 같은 경로 — 주 지표), `resized` = 1036² (학습 곡선 · R1 과 같은 해상도).

## 환경변수

| 변수 | 기본 | 의미 |
|---|---|---|
| `DEFSEG_V6_PHASE` | (없음) | 위 표 |
| `DEFSEG_V6_FOLDS` | `inlayer` | `cv` · `eval` · `baselines` 가 도는 fold (공백으로 구분) |
| `DEFSEG_V6_EXTRA` | (empty) | `cv` · `train` 에서는 학습 스크립트 인자 (`--quick`, `--seed`, `--no-copy-paste`, `--total-steps`, `--ema-decay` …), `eval` · `review` · `infer` · `ornl_compare` · `convert` 에서는 그 스크립트 인자 |
| `NVIDIA_VISIBLE_DEVICES` | `0,1,2,3` | 사용할 GPU |
| `NPROC` | `4` | DDP world size (= GPU 개수) |
| `OMP_NUM_THREADS` | `8` | per-process CPU threads |
| `ORNL_DATA_MODE` | `ro` | `ORNL_Data` 마운트 방식. `convert` 에서만 `rw` |

### Fold 이름

| 이름 | 검증용으로 빼는 것 | 채점되는 결함 클래스 (1036² 기준) |
|---|---|---|
| `inlayer` | `splits/legacy_inlayer_v1.json` 의 22 레이어 — v5 `inlayer` 와 같은 목록 (소스마다 5개 중 다섯째) | 8종 전부 (IS PS RS RH SE SP OM DB). 낙관적인 값 |
| `inlayer_k0` … `inlayer_k3` | 뺄 자리를 바꾼 나머지 네 fold (25–27 레이어). `inlayer` 와 합쳐 5-fold | 8종 전부 |
| `loso_L21` | v2021_LPBF 전체 (20) | IS PS RS SE SP DB |
| `loso_MS` | v2022_Maraging 전체 (26) | PS RS SE SP OM DB (RH 는 이 소스에만 있어 채점 제외) |
| `loso_I2` `loso_GP` `loso_BJ21` | 24 / 9 / 20 | RS SE OM DB / IS RS SE SP / IS PS RS DB |
| `loso_17-4` `loso_I1` `loso_H13` | 14 / 5 / 7 | PS SE SP / SE SP / PS RS DB |

## 데이터셋을 다시 변환할 때

판정표 (`v6/legacy_review.csv`) 나 매핑 (`config_v6.py`) 을 고쳤을 때만 필요하다. `ORNL_Data` 에 쓰는 일이라 마운트를 `rw` 로 바꿔 띄운다 (CPU, 약 3분).

```bash
# 기본 옵션으로 다시 (판정표 반영, Swelling → SE)
DEFSEG_V6_PHASE=convert ORNL_DATA_MODE=rw DEFSEG_V6_EXTRA=--overwrite docker compose up --build
# D3 대안 (Swelling → 미레이블) / 판정표 없이 (기존 IS 전부 미레이블)
DEFSEG_V6_PHASE=convert ORNL_DATA_MODE=rw DEFSEG_V6_EXTRA="--overwrite --swelling ignore" docker compose up --build
DEFSEG_V6_PHASE=convert ORNL_DATA_MODE=rw DEFSEG_V6_EXTRA="--overwrite --no-is-review" docker compose up --build
```

- 변환은 `<폴더>.partial` 에 다 만든 뒤 바꿔 끼운다. 원본 (`ORNL_Data/DSCNN_Dataset`) 은 읽기만 한다.
- 옵션을 바꿔 변환하면 `배선 확인` 이 경고를 낸다 (계획의 표는 기본 옵션의 값이다). 정상이다. 어떤 옵션의 데이터셋으로 학습했는지는 `run.json` 의 `config.dataset.conversion` 에 남는다.
- 컨테이너 안에는 git 이 없어 `conversion.json` 의 `code` 가 `unknown` 으로 적힌다.

## GPU

- **이 compose 는 `NVIDIA_VISIBLE_DEVICES` 가 실제로 먹는다** (v5 와 같다). v3 · v4 의 compose 에 있는
  `deploy.resources.reservations.devices (count: all)` 블록을 넣지 않았다.
- 학습은 GPU 당 약 23.6 GB 를 쓴다 (1036², GPU 당 4장, AMP 끔 — v5 에서 잰 값. 모델 구조가 같다).
  GPU 0 · 1 에는 서빙 컨테이너 (`defsegam-api-v4`, `diva-sam-service`) 가 3–4 GB 씩 올라가 있어 합치면 27 GB 안팎 (32 GB 중) 이다.
  학습 중에 DIVA 가 `/predict/batch` 를 크게 보내면 OOM 이 날 수 있다.
- GPU 를 줄여 돌리면 effective batch 가 `4 × NPROC` 로 줄어든다 (계획값 16). 예: `NVIDIA_VISIBLE_DEVICES=2,3 NPROC=2`.

## 소요 시간

2026-11-17 에 잰 값 (GPU 4장, 10,000 step, 초당 1.46 step): P0 (`cv` 의 inlayer fold 학습, 250 step 마다 검증 포함) 118.6분, 전수 학습 (`train`, 검증 없음) 114.4분.
`ornl_compare` (100 레이어, GPU 1장, `--workers 8`) 는 컨테이너가 뜨고 끝날 때까지 4분 48초다 — 레이어 고르기 71초, 추론 · 그림 205초 (첫 레이어가 나올 때까지 94초는 HDF5 읽기다).
`--from-preds` 로 그림 · 수치만 다시 만들면 CPU 로 148초다.
기준선 (R1 + 옛 모델 2개 × 22 레이어, 두 해상도) 은 GPU 1장으로 1분이 안 걸린다 (2026-11-17 에 `baselines` phase 를 실제로 돌린 값: 51초. 모델 하나의 채점이 12초).

## 스모크 검증 (학습 X)

```bash
# 1 GPU + quick (6 step, 224², 소스당 3 레이어) — fold 학습 → 채점 → 기준선 → 요약
DEFSEG_V6_PHASE=cv DEFSEG_V6_FOLDS=inlayer NPROC=1 NVIDIA_VISIBLE_DEVICES=2 \
  DEFSEG_V6_EXTRA=--quick docker compose up --build

# 2 GPU DDP + quick — 전수 학습
DEFSEG_V6_PHASE=train NPROC=2 NVIDIA_VISIBLE_DEVICES=2,3 \
  DEFSEG_V6_EXTRA=--quick docker compose up --build
```

`--quick` 산출물은 run-name 이 `vitb14_reg_dpt_dual_sz224_10cls_v6` 라 실제 결과 (`…sz1036…`) 와 섞이지 않는다. 확인한 뒤 지워도 된다.

## 산출물

```
checkpoints/vitb14_reg_dpt_dual_sz1036_10cls_v6/
├── cv/<fold>/{best.pt, last.pt}           # 검증 fold. 배포 후보 · 주 채점 대상은 last (정해진 길이의 마지막 EMA)
└── full/{last.pt, last_raw.pt}            # 전수 학습: EMA 사본 / 학습용 가중치
figures/vitb14_reg_dpt_dual_sz1036_10cls_v6/
├── dataset_checks/                        # 변환한 데이터셋의 미리보기 · 검토 시트
└── v6/
    ├── cv/{summary.md, summary.csv, summary.json, curves.png}
    ├── cv/<fold>/{run.json, curve.csv, result.json, class_curves.png}
    ├── cv/<fold>/eval_{native,resized}_last/{metrics.json, per_class_metrics.csv, confusion_*.{csv,npy,png}}
    ├── cv/<fold>/{ref_copy_resized, ref_v5_{native,resized}, ref_v4_{native,resized}}/…      # 기준선 R1–R3
    └── full/{run.json, curve.csv, result.json, inference_last…/<build>/layerXXXX.png}
        ├── trainfit_{native,resized}_last/  # trainfit: 학습 125 레이어 채점 (metrics.json, 혼동 행렬) — 암기 정도
        └── ornl_random100_last/           # ornl_compare: 무작위 100 레이어에서 DSCNN · v4 · v5 · v6
            ├── layers.json                # 뽑은 레이어 (seed · 규칙 · 빌드별 목록)
            ├── overview.png               # 먼저 볼 한 장 — 빌드마다 한 레이어 (뽑힌 것 가운데 v6 의 결함 화소 비율이 중앙값인 레이어)
            ├── pages/page_XX_<빌드>.png   # 비교 그림 — 뽑힌 레이어 전부, 한 장에 5 레이어 × 6 열
            ├── summary.{png,md,json}      # 클래스별 예측 화소 비율 · v6 와의 일치 · CAD 파트 마스크와의 겹침 · RS 선의 가로 길이 · 고정 위치의 Debris
            ├── layer_stats.csv · stats.npz
            └── preds/<빌드>_<레이어>.npz   # 원본 해상도 예측 (int8, 10 = 대응 없음) + CAD 파트 마스크
```

- `metrics.json` 에는 클래스별 IoU · 정밀도 · 재현율과 함께 놓침 · 잘못 짚음 (`miss_false_alarm`), 혼동 쌍 (`pairs`: IS↔PS, RS↔RH, SE↔DB, SP↔OM),
  IS 와 PS 를 합쳐 본 IoU (`is_ps_union_iou`), 소스별 표 (`by_group`) 가 있다.
- ckpt 형식은 v4 · v5 와 같다 (`model_state` = trainable weight 만). `config` 에 `schema` (`pbf-2026.11`), `class_keys`, `class_names`, `orientation`,
  `class_train_px`, `dataset` (변환 옵션) 이 들어 있어 서빙 쪽이 클래스 이름을 ckpt 에서 읽을 수 있다.
- ORNL figure (`infer`) 는 사진 밝기를 **원본 그대로** 넣는다. v4 · v5 의 같은 그림은 레이어마다 밝기를 늘린 입력으로 만든 것이라 나란히 볼 때 그 차이를 적어야 한다.
