# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 1. Project

L-PBF powder bed 카메라 이미지(layer 당 visible/0 melt 직후 + visible/1 도포 직후 dual 입력)에서
결함을 픽셀 단위 semantic segmentation 하는 모델. **DINOv2 (frozen, weight-shared) ×2 →
per-stage `concat(f0, f1, f1−f0)` 융합 → 자체 구현 DPT decoder → class logits**.
학습 대상은 decoder/head 뿐이고 **checkpoint 에도 trainable weight 만 저장**된다
(로드 시 backbone 은 torch.hub 로 별도 로드).

**2-Stage 학습** (v1–v4 공통. v5 부터는 Stage 1 없이 사람 GT 만으로 단일 단계 학습 — §4):
- **S1 — KD pretrain**: ORNL Co-Registered HDF5 5개 빌드. 라벨은 `slices/segmentation_results`
  (DSCNN 예측 mask) argmax hard label. Build 2-5 train / **Build 1 = 고정 val·비교 빌드**.
- **S2 — GT finetune**: DSCNN_Dataset 사람 GT (재료별 native ID → ORNL 12-class 재매핑
  `MATERIAL_TO_ORNL` → 버전별 class 재매핑). S1 ckpt 에서 decoder 만 이어받고 optimizer reset.

## 2. 실행 컨텍스트 (중요)

- **모듈 실행 시 작업 디렉터리는 항상 이 repo 의 부모(`3DP_VPPM/`)** 다.
  `python -m DefSeg_AM.v4.<module>` / `torchrun ... -m DefSeg_AM.v3.training.train_stage1` 형태로 호출.
  `common/config.py` 의 `PROJECT_ROOT = parents[2]` 가 `3DP_VPPM/` 를 가리키기 때문.
- **데이터는 repo 밖**: `3DP_VPPM/ORNL_Data/Co-Registered In-Situ and Ex-Situ Dataset/[baseline] (Peregrine v2023-11)/`
  (Stage 1 HDF5) + `3DP_VPPM/ORNL_Data/DSCNN_Dataset/` (Stage 2 GT). HDF5 키/좌표/단위가 필요하면
  baseline 디렉터리의 README.md 를 먼저 읽을 것.
- v6 용 데이터: `3DP_VPPM/ORNL_Data/DSCNN_Dataset_pbf-2026.11/` — 위 사람 GT 125 레이어를 새 클래스 체계 (`pbf-2026.11`,
  id 0–9 · 미레이블 −1) 로 옮긴 것. DIVA 내보내기 zip 과 같은 구성이다 (`visible/{0,1}/{id}.png`, `annotations/{id}.npy`,
  `index.csv`, `classes.json`). 2026-11-17 에 `v6/data/convert_legacy.py` 로 만들었다. 다시 만들 때는 `v6/docker` 에서
  `DEFSEG_V6_PHASE=convert ORNL_DATA_MODE=rw DEFSEG_V6_EXTRA=--overwrite docker compose up --build` (CPU, 약 3분 — 이 phase 만 `ORNL_Data` 를 rw 로 마운트한다).
  폴더 안의 `README.txt` 와 `conversion.json` 에 옮긴 규칙과 레이어별 화소 수가 있다.
  기존 Incomplete Spreading 주석 45 레이어는 `v6/legacy_review.csv` 의 레이어 단위 판정에 따라 IS / PS / 미레이블로 나뉜다
  (IS 9 · PS 33 · 미레이블 3 레이어. 2026-11-17 에 Claude 가 사진을 보고 정했고, L21 · 17-4 · GP 는 사용자가 준 규칙 —
  경사면이 함께 보이면 PS, 아니면 IS — 을 따랐다. 원본이 경계선만 칠한 17-4 · GP 의 7 레이어는 선이 가두는 영역 전체를 채웠다
  (판정표의 `fill` 열). 사람의 확인 전이다 — `v6/PLAN_v6.md` §2.4).
  변환 스크립트는 이 표를 기본으로 읽는다 (쓰지 않으려면 `--no-is-review`).
- venv: `DefSeg_AM/venv` (torch 2.9.1+cu128, `requirements.txt` 고정). 호스트·컨테이너 공용
  (컨테이너에 read-only bind mount, 이미지에 pip install 안 함).
- 부모 `3DP_VPPM/` 는 별개 프로젝트(VPPM, `Sources/vppm`)의 루트로 자체 CLAUDE.md 가 있다.
  이 repo(DefSeg_AM)와 데이터만 공유하고 코드는 독립.

## 3. 실행 명령

**풀런(학습 전체)은 사용자가 직접 실행한다.** Claude 는 코드/compose 작성과
스모크 검증(`--quick`, `--max-batches 2` 등)까지만.

v2·v3·v4 는 각 `vN/docker/` 의 compose 로 실행 (v1 은 `v1/scripts/run_*.sh` nohup 래퍼):

```bash
cd DefSeg_AM/v4/docker
echo "UID_GID=$(id -u):$(id -g)" > .env          # 최초 1회 (없으면 1000:1000)

DEFSEG_V4_PHASE=build_cache docker compose up -d --build   # 라벨 캐시 (~5분)
DEFSEG_V4_PHASE=stage1      docker compose up -d --build   # 4-GPU DDP 학습
DEFSEG_V4_PHASE=stage2      docker compose up -d --build
DEFSEG_V4_PHASE=confusion   docker compose up -d --build   # Build 1 confusion/IoU
DEFSEG_V4_PHASE=infer       docker compose up -d --build   # figure 생성 (TTA+후처리)
docker compose logs -f
docker compose down

# 스모크: 1 GPU + quick
DEFSEG_V4_PHASE=stage1 NPROC=1 NVIDIA_VISIBLE_DEVICES=0 DEFSEG_V4_EXTRA=--quick \
  docker compose up -d --build
```

환경변수 이름은 버전별로 `DEFSEG_V3_PHASE`/`DEFSEG_V4_PHASE` 등으로 다르다 — 각
`vN/docker/README.md` 참조. phase: `build_cache | stage1 | stage2 | all | confusion | infer`.

v5 는 phase 가 다르다: `DEFSEG_V5_PHASE = cv | summarize | eval | train | confusion | infer | all` (기본값 없음 — 안 주면 사용법만 출력).
`cv` = 검증 fold (`DEFSEG_V5_FOLDS`) 학습 → native 채점 → 요약, `train` = 8 소스 전수 학습. `v5/docker/README.md` 참조.
`DEFSEG_V5_SOOT=spatter` 를 주면 Soot 주석을 Spatter 로 배운다 (2026-11-20 추가 — 기본 `ignore` 는 기존 v5 그대로 미레이블). v6 가 Soot 를 Spatter 로 읽으므로 두 버전을 Spatter 에서 견주려는 실행이고,
run 이름에 `-soot2sp` 가 붙는다 (`vitb14_reg_dpt_dual_sz1036_11cls_v5-soot2sp`). `inlayer` fold 를 2026-11-20 에 돌렸다.

v6 는 `DEFSEG_V6_PHASE = cv | summarize | eval | baselines | review | train | trainfit | infer | ornl_compare | convert` (기본값 없음). 첫 실험 P0 는
`DEFSEG_V6_PHASE=cv DEFSEG_V6_FOLDS=inlayer` — fold 학습 → 채점 (원본 해상도 + 1036²) → 기준선 R1–R3 → 요약. `baselines` 는 학습 없이 기준선만,
`review` 는 기존 IS 판정의 검토 시트, `convert` 는 데이터셋 재변환이다. `train` 은 125 레이어 전수 학습 (`full/last.pt`),
`trainfit` 은 그 모델을 학습에 쓴 레이어에서 채점한다 (암기 정도이지 성능이 아니다 — GPU 가 없으면 `NVIDIA_VISIBLE_DEVICES=none` 으로 CPU 에서 돈다),
`ornl_compare` 는 ORNL 5 빌드에서 무작위로 뽑은 레이어 (빌드마다 20) 에서 DSCNN · v4 · v5 · v6 예측을 나란히 그린다 (사람 GT 가 없어 점수가 아니라 모델끼리의 비교다).
`DEFSEG_V6_EXTRA=--from-preds` 를 주면 모델을 다시 돌리지 않고 저장해 둔 예측 (`preds/`) 으로 그림 · 수치만 다시 만든다 (GPU 를 쓰지 않는다, 2.5분). `v6/docker/README.md` 참조.
v7 은 `DEFSEG_V7_PHASE = cv | train | eval | summarize | compare | ornl_compare | gt_compare` (기본값 없음, compose 프로젝트 이름 `defseg-v7`). 실행은 꼬리표 (`DEFSEG_V7_VARIANT`, 기본 `N` = 합성 없는 기준선) 와
seed 로 가른다 — `DEFSEG_V7_RUNS="42:inlayer 42:loso_L21"` 처럼 seed:fold 의 목록을 주면 그 순서로 돌고, 끝난 실행은 건너뛴다 (fold 의 기본값은 판정 fold 둘 — `inlayer loso_L21`).
합성 세트는 `DEFSEG_V7_SYNTH_SET=<이름>` (`cache/genaug/sets/` 아래) 과 `N` 이 아닌 꼬리표를 함께 준다 — 합성을 넣은 주 실험은
`DEFSEG_V7_PHASE=cv DEFSEG_V7_VARIANT=W DEFSEG_V7_SYNTH_SET=w600_v1 DEFSEG_V7_RUNS="42:inlayer 42:loso_L21"` 이다 (2회, 약 4시간).
`compare` 는 실행들을 fold 마다 한 표로 모아 판정을 낸다 (GPU 불필요). `ornl_compare` 는 ORNL 무작위 레이어에서 DSCNN · v4 · v5 · v6 (같은 fold · seed 의 기준선) · v7 을 나란히 그린다 (GPU 1대, 참고용 — `v7/scripts/compare_ornl_v7.py`). `gt_compare` 는 사람 GT 의 검증 레이어에서 정답과 v4 · v5 · v6 · v7 을 나란히 그린다 (`v7/scripts/compare_gt_v7.py` — DSCNN 의 열은 없다). `v7/docker/README.md` 참조.
합성 세트는 genaug 의 compose 가 만들고 점검한다: `cd genaug/docker` 에서 `GENAUG_PHASE=synth` (GPU 2 · 3 으로 600 장에 70분) · `GENAUG_PHASE=check` (K1–K6, GPU 1대 약 10분). `genaug/docker/README.md` 참조.
v8 은 `cd v8/docker` 에서 `DEFSEG_V8_PHASE = norm | check | cv | eval | compare` (기본값 없음, compose 프로젝트 이름 `defseg-v8`). 꼬리표 (`DEFSEG_V8_VARIANT`, 기본 `B`) 가 보정의 종류이자 보정 세트를 고른다 (`B` → `b115` 밝기 맞춤, `BL` → `bl115` 밝기 맞춤 + 조명 제거, `raw` → 보정 없음 — 스모크용). `norm` = 보정 세트 만들기 (CPU, 세트 하나에 약 4분 — 셋 다 2026-11-25 에 만들어 두었다), `check` = 0단계 (GPU 1대, 3분 30초 — 기준선 모델에 밝기를 옮긴 사진 · 가짜 조명을 얹은 사진을 넣어 본다), `cv` = fold 학습 → 채점 → 비교 표. 학습의 주 실행은 `DEFSEG_V8_PHASE=cv DEFSEG_V8_VARIANT=B DEFSEG_V8_RUNS="42:inlayer 42:loso_L21"` 이다 (2회, 약 4시간, GPU 4대 — 사람 GT 만 학습한다). `compare` 는 기준선 (v7 의 `N`) 과 한 표로 모은다 (GPU 불필요). `v8/docker/README.md` 참조.
DSCNN 재구현 (`dscnn/`, 아래 §4) 은 `cd dscnn/docker` 에서 `DSCNN_PHASE = selftest | cv | train | eval | compare | ornl_compare` (기본값 없음, compose 프로젝트 이름 `defseg-dscnn`, **GPU 1대** — 기본 2 번).
`cv` = fold (`DSCNN_FOLDS`, 기본 `inlayer`) 학습 → 채점 (원본 해상도 + 1036²). 100 epoch (155,000 step) 에 RTX 5090 한 대로 2시간 16분이 걸렸다.
클래스 가중치는 기본이 논문의 것이고, `DSCNN_TAG=sqrtw DSCNN_EXTRA="--class-weight sqrt_inv"` 를 주면 v6 의 것으로 돈다 (두 실행을 함께 돌리려면 `docker compose -p defseg-dscnn-b` 로 프로젝트 이름을 달리 준다). `compare` 는 검증 레이어에서 정답 · DSCNN 재구현 · v4 · v5 · v6 · v7 을 나란히 그리고 표를 낸다. `ornl_compare` 는 ORNL 무작위 레이어에서 DSCNN (ORNL 이 배포한 예측) · DSCNN 재구현 · v4 · v5 · v6 · v7 을 나란히 그린다 (`dscnn/scripts/compare_ornl.py`, 참고용 — 그림은 v7 의 비교 옆 `figures/<v7 run>/v6/cv/<fold>/ornl_random20_last_dscnn_reimpl_<꼬리표>/`). `dscnn/docker/README.md` 참조.
v5 와 v6 의 compose 는 프로젝트 이름이 같아 (`docker`) `up` 할 때 orphan 컨테이너 안내가 나온다 — `--remove-orphans` 를 붙이면 상대 버전의 컨테이너와 로그가 지워지므로 붙이지 않는다.

GPU 선택 주의 (2026-11-16 확인): v3 · v4 의 compose 는 `deploy.resources.reservations.devices (count: all)` 블록 때문에
`NVIDIA_VISIBLE_DEVICES` 를 무엇으로 주든 컨테이너에 GPU 4개가 다 보이고 프로세스가 GPU 0 부터 올라간다. v5 · v6 compose 는 그 블록을 빼서
지정이 실제로 먹는다. GPU 0·1 에는 서빙 컨테이너 (`defsegam-api-v4`, `diva-sam-service`) 가 떠 있으므로 스모크는 GPU 2·3 에서 돌린다.

버전 간 비교 figure 는 repo 루트 `scripts/` 에서:
`python -m DefSeg_AM.scripts.compare_v3_v4_multi_build` 등 (역시 CWD = `3DP_VPPM/`).

테스트/린트 설정은 현재 없다.

## 4. 버전 구조 (v1 → v6 실험 계보)

```
common/     # 모든 버전 공유: config.py(경로·ORNL 12-class·MATERIAL_TO_ORNL), models/(DefSegModel·losses),
            # data/(image_utils·DistributedWeightedSampler), training/dist_utils.py, utils/log.py
v1/         # 12-class (ORNL native), ViT-S/14, 1 GPU
v2/         # 8-class (Recoater 통합+3종 제거), 8-fold leave-one-source-out CV, D4+cyclic+DSCNN aug
v3/         # 10-class 복원, CV 없음(8 source 전수 train + Build 1 평가), 4-GPU DDP, 휴리스틱 후처리
v4/         # 11-class (+Under Melting), ViT-B/14_reg, class-aware sampler, Copy-Paste aug, 후처리 2규칙
v5/         # 11-class (v4 와 같음) + 사람 GT 전용: Stage 1 없음, decoder 무작위 초기값에서 단일 단계 (step 기준, EMA 0.999),
            # 검증은 사람 GT 를 떼어 둔 fold (leave-one-source-out). DSCNN 예측은 학습·선정에 쓰지 않고 참고 평가에서만 읽는다
v6/         # DIVA 레이블 체계 pbf-2026.11 10-class + 사람 GT 전용 (v5 의 학습 방식 그대로). 변환해 둔 데이터셋의 라벨을 그대로 읽는다 (번호를 바꾸는 단계가 로더에 없다).
            # 2026-11-17 현재 P0 까지 구현 · 스모크 확인: Peregrine 주석만으로 학습 · 채점 · 기준선 (복사 기준선, v4 · v5 예측을 새 id 로 옮겨 채점) · 요약. P0 풀런은 2026-11-17 에 끝났다 (결함 mIoU 0.533 @ 1036² · 0.467 @ 원본 해상도, 복사 기준선 0.264).
            # 전수 학습 (`full/last.pt`, 114분) 과 ORNL 5 빌드 무작위 100 레이어 비교 (`figures/<run>/v6/full/ornl_random100_last/`) 도 2026-11-18 에 끝났다. 결과와 읽는 법은 `v6/RESULTS_v6.md` (§1 P0 · §2 전수 학습 · §3 ORNL 비교).
            # 첫 실험 P0 = inlayer (v5 inlayer fold 와 같은 22 레이어를 검증으로 — v6/splits/legacy_inlayer_v1.json). v5 와 달라진 것은 클래스 체계와 Copy-Paste 의 위치 규칙.
            # DIVA 내보내기를 읽는 쪽 (diva_export · build_cache · split) 은 내보내기가 온 뒤 P1 에서 만든다 — 아직 없다
v7/         # v6 + 학습 데이터에 "합성 레이어" 도메인 하나 (PLAN_v7.md). 합성 레이어 = genaug 의 생성 모델이 라벨 맵으로 그린 사진 두 장 (대상 클래스 RS · DB · SE · RH).
            # **v6 를 복사하지 않고 임포트한다** (2026-11-18 사용자 결정 — 아래 의존 규칙의 예외). v7 에 있는 것은 더하는 것뿐이다: config_v7.py, data/synth_domain.py,
            # training/train.py (v6 의 train.py 를 복사해 고친 하나뿐인 사본), scripts/compare_runs.py, docker/. 채점 · 요약은 v6 의 모듈을 --run-name 으로 부른다.
            # 2026-11-19 현재: 합성 없는 기준선 다섯 번이 끝났다 (`inlayer` seed 42 · 43 · 44 의 T 0.453 · σ 0.0042, `loso_L21` seed 42 · 43 · 44 의 T 0.173 · σ 0.0149 — `figures/v7_compare/sz1036/`).
            # 합성 세트 `cache/genaug/sets/w600_v1` (600 장) 도 그날 만들었다 (관문 K2 – K4 통과, K1 의 사용자 확인이 남았다 — PLAN_v7 의 "구현 상태"). 합성을 넣은 풀런 (꼬리표 W, seed 42) 도 그날 두 fold 에서 끝났다: T 는 두 fold 모두 기준선과 같다 (`inlayer` 0.447, `loso_L21` 0.171). `inlayer` 에서 RS 가 올랐고 (0.260 → 0.306) DB 가 내렸으며,
            # `loso_L21` 에서는 SP 가 0.196 → 0.064 로 내려가 부작용 판정이다 (PLAN_v7 의 "구현 상태 → 주 실험의 결과"). 다음 실행은 정하지 않았다.
            # 판정은 사람 GT 의 두 fold — inlayer 와 loso_L21 — 로 한다 (2026-11-19 사용자 결정, PLAN_v7 §7 의 V12. 2026-11-18 에는 inlayer 만 쓰기로 했다가 바꿨다). 그 밖의 fold 에는 합성 세트를 넣을 수 없다
            # (세트의 manifest.json 의 valid_folds 를 학습 스크립트가 확인하고 멈춘다). genaug 를 임포트하지 않고 합성 세트 폴더만 읽는다. run 이름은 …_10cls_v7-<꼬리표>-s<seed>, 그림은 figures/<run>/v6/… 에 쌓인다 (v6 코드가 폴더를 만든다)
v8/         # v6 + 입력 사진의 전처리 둘 — 소스 (장비) 마다 밝기를 115 로 옮기는 것 (꼬리표 B) 과 조명 얼룩을 지도로 나누는 것 (꼬리표 BL) (PLAN_v8.md). 생성 데이터는 쓰지 않는다.
            # **v6 · v7 을 임포트한다** (PLAN_v8 §7 의 D1 — 사용자의 확인 전). 학습 · 채점 코드에 변환을 끼워 넣지 않고, 보정한 사진을 **보정 세트** (`cache/v8_norm/<이름>/`) 로 저장해 두고
            # 얇은 껍데기 (`v8/training/train.py`, `v8/inference/evaluate.py`) 가 v6 설정의 `LEGACY_PBF_ROOT` 를 그 프로세스 안에서만 보정 세트로 바꾼 뒤 v7 의 학습 · v6 의 채점을 그대로 부른다.
            # 통계 (소스의 화소 중앙값, 도포 직후 사진으로 구한 2차식 조명 지도) 는 라벨 없이 구한다 (`v8/data/photo_norm.py`). 조명 지도는 안전 장치 셋 (레이어 10 장 · 크기 1% · 매끄러움 0.70) 을 지난 소스에만 쓴다 — 지금은 L21 · BJ21 뿐이다.
            # 2026-11-25 에 코드 · compose · 보정 세트 셋 (`b115` · `bl115` · `raw`) · 스모크 (보정 없는 세트 `raw` 로 돌리면 v7 의 N 실행과 loss 가 같다) · 0단계 (학습 없는 민감도 점검) 를 끝냈고, 2026-11-26 에 `B` 의 풀런 (seed 42, 두 fold) 이 끝났다.
            # 0단계의 결과 (`figures/v8_check/sz1036_s42/`, PLAN_v8 §9.3): `loso_L21` 은 밝기를 −40 … +60 단계 옮겨도 결함 mIoU 가 0.157 – 0.177 (seed 의 흔들림 안) — L21 의 낮은 점수는 밝기 탓이 아니다.
            # `inlayer` 는 밝게 하면 +0.018 까지 오르지만 거의 OM 한 클래스다. 가짜 조명은 최대 − 최소 10% 까지 점수를 움직이지 않는다 — 미리 정한 규칙대로 `BL` 은 돌리지 않았다.
            # **`B` 의 결과 (PLAN_v8 §9.5, `figures/v8_compare/sz1036/`): 밝기를 맞춰도 오르지 않았다** — 결함 mIoU `inlayer` 0.467 → 0.477 (문턱 0.487 아래), `loso_L21` 0.169 → 0.161 (문턱 안).
            # `inlayer` 에서는 OM 이 오르고 (0.181 → 0.348) DB 가 내렸다 (0.466 → 0.379 — BJ21 에서 PS 와 헷갈린다. 부작용 판정). 1036² 에서는 차이가 없다. seed 43 은 돌리지 않았다 — 다음은 정하지 않았다 (권한 것은 여기서 멈추는 것)
genaug/     # 버전이 아니다 — 결함 인페인팅 확산 모델로 학습 데이터를 늘리는 실험 (PLAN_GENAUG.md). 의존은 common 뿐, 새 패키지 없음 (torch 만).
            # 2026-11-18 현재 생성 모델의 풀런까지 끝났다 (`checkpoints/genaug/fill_w256/`, 100,000 step). 결과 점검은 `figures/genaug/fill_w256/check_2026-11-18/` (저장소 코드가 아닌 스크립트로 한 것).
            # 합성 세트를 만드는 단계 (`synth/` — 라벨 맵 규칙 · 레이어 그리기 · 세트 저장) 와 점검 (`checks/{synth_sheet, seam, recognition, probe}.py`) 은 2026-11-19 에 만들었다 (GENAUG_PHASE=synth · check).
            # 방식은 "라벨 맵 → 레이어" 다 (v7/PLAN_v7.md §2): ORNL B3 · B4 · B5 의 CAD 배치 위에 RS · RH · SE · DB 를 규칙대로 얹고, 사진의 바깥 5% 만 진짜로 둔 채 안쪽을 그린다. 분할 모델 실험은 v7 계획으로 옮겼다.
            # 사람 GT 가운데 splits/holdout_v1.json 의 38 레이어 (inlayer 검증 22 + L21 20) 는 읽지 않는다. compose 는 genaug/docker (GENAUG_PHASE, 기본 GPU 2 · 3)
dscnn/      # 버전이 아니다 — **DSCNN 의 재구현** (Scime et al. 2020 의 표 3 – 6: global · regional U-Net · localization 세 갈래, 1,400만 파라미터) 을 사람 GT 만으로 학습해
            # DefSeg-AM 과 같은 검증 레이어에서 견주는 실험 (2026-11-23). ORNL 이 배포한 DSCNN 예측 (HDF5 의 segmentation_results) 과는 다른 것이다 — 그 모델의 가중치는 공개되지 않았다.
            # 클래스 · 데이터셋 · fold · 채점은 **v6 를 임포트해** 그대로 쓰고 (비교 그림은 v7 의 `compare_gt_v7` 의 함수도 부른다), 모델 · 타일 데이터 · 학습만 새로 썼다.
            # 학습 방식은 논문을 따른다 (원본 해상도 타일, Adam 1e-4, EMA 0.9999, 중앙값 빈도 가중치, 잡음 · 밝기 이동 9 가지, 회전 · 반전 없음, 100 epoch).
            # 논문에 없어 정한 것 (8 소스를 한 모델로, 한 배치에 여러 소스를 섞는 것, 2022 판 소스의 타일 크기 등) 은 `dscnn/config_dscnn.py` 의 "정함" 과 `dscnn/docker/README.md` 에 있다. compose 는 dscnn/docker (DSCNN_PHASE, GPU 1대)
            # 2026-11-23 에 `inlayer` 의 100 epoch 풀런 둘이 끝났다 (seed 42, 각 136분). 결함 mIoU (원본 해상도 / 1036², `last.pt`): 논문의 가중치 0.336 / 0.367 (`dscnn_native_10cls_gt-s42`),
            # v6 의 가중치 0.398 / 0.448 (`…-sqrtw-s42`) — v6 는 0.467 / 0.533. v6 의 가중치를 쓴 쪽은 epoch 13 쯤부터 오르지 않았다 (10 epoch 시험 실행이 0.410). 표와 읽는 법은 `dscnn/docker/README.md` 의 "풀런".
```

- 의존 방향: **vN → common 만**. 버전끼리는 서로 임포트하지 않고, 새 버전은 이전 버전
  코드를 **복사 후 수정**한다 (예: `v4/models/losses_v4.py` 는 v3 코드의 자체 사본).
  새 실험 추가 시 이 패턴을 따를 것: `vN/{config_vN.py, data/, models/, training/, inference/, scripts/, docker/}` + `PLAN_vN.md`.
  **예외 — v7 은 v6 를 임포트한다** (`from ...v6 import config_v6` 등. 2026-11-18 사용자 결정): 클래스와 학습 방식이 v6 와 같은 덧붙임이라 나머지가 v6 와 같아야 하기 때문이다.
  그래서 `v6/` 를 고치면 v7 의 동작도 바뀐다 — v6 를 고친 뒤에는 "합성 세트 없이 돌린 v7 = v6" 스모크 (`v7/docker/README.md`) 를 다시 확인한다.
  `dscnn/` 도 v6 를 임포트한다 (데이터 읽기 · fold · 채점. 같은 까닭 — 견주는 조건이 v6 와 같아야 한다). v6 · v7 은 dscnn 을 임포트하지 않는다.
  `v8/` 은 v6 와 v7 을 임포트한다 (v7 의 학습 · v6 의 채점을 그대로 부른다 — 기준선이 v7 의 `N` 실행이라서). v6 나 `v7/training/train.py` 를 고친 뒤에는 "보정 없는 세트 `raw` 로 돌린 v8 = v7 의 N" 스모크 (`v8/docker/README.md`) 도 다시 확인한다. v6 · v7 은 v8 을 임포트하지 않는다.
- 각 버전의 클래스 공간이 다르다. 재매핑 테이블은 각 `config_vN.py`
  (예: v4 의 `ORNL_12_TO_NEW_11_V4`). DSCNN GT 는 native → ORNL 12 → 버전 class 의 2-step.
  `IGNORE_INDEX = -1` 은 loss 에서 제외.
- imports 는 상대 경로 (`from ...common.models.model import DefSegModel`).

## 5. 캐시 의존 (중요)

이미지 캐시는 **v1 이 생성한 것을 v3/v4 가 재사용**하고, 버전별 build_cache 는 라벨만 재매핑한다:

```
v1 build_cache_stage1 → cache/resized_sz1036/<build>/visible_{0,1}.npy   ← 이미지 (v1·v3·v4 공용)
v3 build_cache_v3     → cache/resized_sz1036_v3/<build>/label_v3.npy     ← 라벨만
v4 build_cache_v4     → cache/resized_sz1036_v4/<build>/{label_v4, pix_per_layer}.npy
                        + rare_class_supplier.json                        ← Copy-Paste supplier
```

v1 이미지 캐시가 없으면 v3/v4 학습 불가. compose 의 `command:` 시작부에 사전조건 체크가 있어
누락 시 FATAL 로 종료한다.

v5 는 캐시를 만들지 않는다 (사람 GT 125 레이어를 시작할 때 RAM 에 올린다). `confusion` phase 만 기존 캐시를 읽는다 —
이미지는 v1, 라벨은 v4 (`resized_sz1036_v4`, 클래스 공간이 같다).

v6 도 P0 에서는 캐시를 만들지 않는다 (변환해 둔 데이터셋 125 레이어를 시작할 때 RAM 에 올린다). ORNL HDF5 는 `infer` (참고 figure) 만 읽고,
사진 밝기를 원본 그대로 넣는다 — v1 캐시와 v4 · v5 의 infer 는 레이어마다 밝기를 늘린다 (사람 GT 만으로 배운 모델에는 맞지 않는 입력이다).

`cache/ornl_canvas_sz1036/<build>/{visible_0, visible_1, part_mask}.npy` + `meta.npz` 는 주석 없는 ORNL 사진의 **공용 배경 캐시**다
(2026-11-17, `python -m DefSeg_AM.genaug.data.build_canvas_cache`, 다섯 빌드 · 5층마다 1층 · 3,167 레이어 · 9.5 GB · CPU 21분).
사진은 **원본 밝기** (v1 캐시와 다르다), 마스크는 CAD (`slices/part_ids`) 이고 DSCNN 예측은 읽지 않는다. GenAug 가 읽고, RuleSynth (`rulesynth/PLAN_RULESYNTH.md` §2.3) 가 같은 캐시를 쓰기로 되어 있다.

`cache/v8_norm/<이름>/` 는 v8 의 **보정 세트**다 (v8 의 `norm` phase 가 만들고 v8 의 학습 · 채점이 읽는다). 변환한 사람 GT 데이터셋과 같은 구성인데 사진만 보정한 것이고, `annotations` 는 원본 (`ORNL_Data/DSCNN_Dataset_pbf-2026.11/annotations`) 으로 가는 상대 링크다 — `DefSeg_AM` 과 `ORNL_Data` 가 나란히 놓인 자리에서만 풀린다 (호스트와 컨테이너 모두 그렇다). `norm.json` 에 소스마다의 대표 밝기 · 더한 값 · 조명 지도의 계수 · 안전 장치의 결과가 있다. 원본을 다시 변환하면 (`conversion.json` 의 `created_at` 이 바뀌면) 학습 · 채점이 FATAL 로 멈추므로 세트도 다시 만든다. `b115` · `bl115` · `raw` 가 있다 (2026-11-25, 각 437 MB).

`cache/genaug/sets/<이름>/` 는 **합성 세트**다 (genaug 의 `synth` phase 가 만들고 v7 이 읽는다). 구성은 변환한 사람 GT 데이터셋과 같고 (`visible/{0,1}/{id}.png`, `annotations/{id}.npy`, `index.csv`, `classes.json`)
`meta/` · `set_config.json` · `manifest.json` 이 더 있다. `w600_v1` 이 첫 세트다 (2026-11-19, 600 장 — B3 · B4 · B5 200 장씩). `smoke24_v0` 은 그 전에 스모크용으로 만든 24 장이다.

## 6. 산출물 규약

`cache/ checkpoints/ figures/ logs/ venv/` 는 gitignore 이며 전 버전이 공유하되
**run-name 으로 구분**: `<backbone>_dpt_dual_sz<img>_<n>cls_v<N>`
(예: `vitb14_reg_dpt_dual_sz1036_11cls_v4`). ckpt 는
`checkpoints/<run-name>/{stage1_best, stage2_best}.pt`, figure 는 `figures/<run-name>/...`.
v5 는 단계가 없어 `checkpoints/<run-name>/full/{last, last_raw}.pt` (전수 학습) 와 `checkpoints/<run-name>/cv/<fold>/{best, last}.pt` (검증 fold) 다.
기존 v5 (`…_11cls_v5`) 의 전수 학습 가중치는 2026-11-23 에 지웠다 (사용자 요청 — fold 의 것은 남아 있다). 그래서 v5 의 `infer` · `confusion` 과, v6 의 `ornl_compare` 를 `--from-preds` 없이 다시 돌릴 때의 v5 열은 지금 나오지 않는다.
v6 도 같은 배치다 (run-name `vitb14_reg_dpt_dual_sz1036_10cls_v6`, figure 는 `figures/<run-name>/v6/…`). fold 폴더 안의 `eval_{native,resized}_last` 가 v6 모델의 채점,
`ref_copy_resized` · `ref_v5_*` · `ref_v4_*` 가 기준선이고, `figures/<run-name>/dataset_checks/` 에 데이터셋 미리보기와 검토 시트가 있다.
v8 의 run-name 은 `…_10cls_v8-<꼬리표>-s<seed>` 이고 배치는 v7 과 같다 (`figures/<run>/v6/cv/<fold>/…`. `norm_set.json` 이 더 있다 — 어느 보정 세트로 학습했는가). **v8 의 ckpt 에는 같은 보정을 건 사진을 넣어야 한다** — v8 의 채점 껍데기는 이것을 확인하지만, 사진을 직접 읽는 v6 · v7 의 `infer` · `ornl_compare` · `gt_compare` 는 보정을 걸지 않는다. 보정 세트의 점검은 `figures/v8_norm/<세트>/`, 0단계는 `figures/v8_check/sz1036_s42/`, 비교 표는 `figures/v8_compare/sz1036/` 에 있다.
DSCNN 재구현의 run-name 은 `dscnn_native_10cls_gt[-<꼬리표>]-s<seed>` 이고 배치는 `checkpoints/<run-name>/cv/<fold>/{last, best, last_raw}.pt`,
`figures/<run-name>/cv/<fold>/{run.json, curve.csv, eval_{native,resized}_last/, gt_val22_last/}` 다 (`v6/` 폴더가 끼지 않는다. `metrics.json` 의 형식은 v6 와 같다).
`figures/<run-name>/v6/full/ornl_random100_last/` 는 전수 학습 모델의 ORNL 비교다 (`overview.png` 한 장, `pages/` 20쪽, `summary.{png,md,json}`, 원본 해상도 예측 `preds/`).

## 7. 문서 맵 — 작업 전 참조

- `docs/MODEL_ARCHITECTURE.md` — 텐서 크기 추적·FFB/RCU 내부·forward 코드 대응
- 각 `vN/PLAN_vN.md` — 실험 설계·hyperparameter 확정값·미해결 항목 (가장 상세한 스펙)
- `v3/RESULTS_v1_vs_v2_vs_v3.md`, `v2/RESULTS_v1_vs_v2.md` — 버전 간 정량 비교
- `v6/RESULTS_v6.md` — v6 의 결과 (P0 inlayer 채점, 전수 학습, ORNL 5 빌드 무작위 100 레이어에서 DSCNN · v4 · v5 와 견준 것)
- `v8/PLAN_v8.md` — 장비별 밝기 맞춤 · 조명 얼룩 제거 (2026-11-24 계획, 2026-11-25 구현). 소스별 밝기와 조명 얼룩의 측정값, 통계를 구하는 법과 안전 장치, 실행 · 판정, 정해야 할 것, 그리고 §9 에 구현 상태 · 스모크 · 0단계 (학습 없는 민감도 점검) 의 결과
- `v8/docker/README.md` — v8 의 실행법 (phase · 환경변수 · 산출물), 지금 있는 보정 세트, "보정 없는 세트 = 기준선" 스모크를 다시 돌리는 법
- `v7/PLAN_AUG.md` — 사람 GT 의 증강을 넓히는 계획 (2026-11-20, 계획만 — 코드 없음). 지금의 증강 목록, 소스별 밝기 · 결 · 해상도의 측정값, 더할 세 묶음 (밝기 · 선명도 · 기하), 실험 설계.
  **v6 의 잡음 증강은 사실상 꺼져 있다** (코드가 설정값을 한 번 더 100 으로 나눈다 — 0.5 단계짜리 변화만 준다). 기준선과 맞추려고 v6 의 코드는 고치지 않는다
- `paper/DSCNN/DSCNN_Summary.md` — DSCNN 논문의 요약 (구조 · 손실 · 증강 · 학습 설정). 같은 폴더의 PDF 는 읽힌다 (`pdftotext`).
  **`3DP_VPPM/Paper/DSCNN/` 의 PDF 는 깨져 있다** (읽으면 빈 쪽만 나온다) — 논문을 확인할 때는 이쪽 사본을 쓴다. 요약의 4.4 절은 갈래의 채널 수를 바꿔 적었다 (맞는 값: localization 128 · global 16 · regional 32)
- `dscnn/docker/README.md` — DSCNN 재구현: 논문에서 옮긴 것과 여기서 정한 것, 실행법, 산출물
- `v1/DEBUG_HISTORY.md` — 학습 중 만난 이슈/해결 기록 (NaN loss 등)
- 루트 `README.md` 는 v1/v2 까지만 반영 — v3/v4 는 각 버전 폴더 문서가 정본

## 8. 논문 작성·수정 — `defseg-paper-author` 서브에이전트 (필수 사용)

`paper/DefectSegAM_Ko/` 의 원고(Elsevier *Additive Manufacturing*, `elsarticle`, 한국어 본문)를
작성·수정·검토·번역하는 모든 요청("논문 써줘", "5장 결과 채워줘", "초록 다듬어줘", "표/그림 추가",
"참고문헌 추가", "AM 저널 형식 맞춰줘", "TODO 정리", "영문판 만들어줘" 등)은 **반드시
[.claude/agents/defseg-paper-author.md](.claude/agents/defseg-paper-author.md) 서브에이전트로 위임**한다
(명시 호출: `@agent-defseg-paper-author …`). 메인 세션에서 `paper/**/*.tex`, `refs.bib` 를 직접 편집하지 않는다.

- 사용 조건: 세션을 `DefSeg_AM/`(또는 그 하위)에서 시작해야 발견된다 — `3DP_VPPM/` 에서 시작하면 미발견.
  `.claude/agents/` 를 처음 만든 세션(2026-09-17)에서는 재시작 후 사용. 프론트매터의 자동 점검 훅은
  **DefSeg_AM 폴더의 workspace trust 를 수락한 대화형 세션에서만** 동작한다(상위 폴더 수락으로는 불충분).
- 로컬 TeX Live 2026 이 있다(2026-10-06, `~/texlive/2026`, 실행 파일은 `~/.local/bin`; sudo 없이 `tlmgr` 로 관리). 편집 후
  `latexmk -pdf main.tex` 로 컴파일해 경고(Float too large, Overfull, undefined)와 표·그림 배치를 확인한다. 산출물은 gitignore.
  같은 파일이 Overleaf(pdfLaTeX + kotex)에서도 그대로 컴파일된다. 정적 점검기도 매 편집 후 실행:
- 원고 README 가 말하는 영문 정본 `paper/DefectSegAM/` 은 **2026-09-17 현재 repo 에 존재하지 않는다** —
  한국어 폴더가 유일한 작업본이며, 영문 폴더가 생기기 전까지는 한국어 원고를 직접 수정한다.
  `paper/DefectSegAM_Ko/` 는 2026-10-06부터 git 추적 대상이다(`Figs/*.png` 와 LaTeX 빌드 산출물은 gitignore).
  편집 전 사본(`_backup/`)은 더 이상 만들지 않으며 되돌리기는 git 으로 한다. 커밋은 사용자가 요청할 때만.
- 결과 수치 출처(`figures/**/metrics.json`, `per_class_metrics.csv`, 다운스트림 `Sources/pipeline_outputs/...`)와
  저널 규정 체크리스트·원고 문체 규약·알려진 불일치 백로그는 에이전트 정의서 안에 정리되어 있다.
