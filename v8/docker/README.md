# DefSeg-AM v8 — Docker 사용법

> v8 = v6 + 입력 사진의 전처리 둘 — **소스별 밝기 맞춤** 과 **조명 지도 제거**. 설계와 판정 기준은 [../PLAN_v8.md](../PLAN_v8.md).
> **v6 · v7 의 코드를 복사하지 않고 불러 쓴다.** 학습은 v7 의 스크립트, 채점 · 요약은 v6 의 모듈 그대로이고, 달라지는 것은 모델에 들어가는 사진뿐이다.
> **생성 데이터는 쓰지 않는다** — 어느 실행도 사람 GT 만 학습한다.
> 호스트의 `DefSeg_AM/venv` 를 read-only bind mount 하여 그대로 쓴다.

**용어**

| 말 | 뜻 |
|---|---|
| 소스 | 사람 GT 의 묶음 여덟 개 (L21 · 17-4 · GP · I1 · I2 · MS · BJ21 · H13). 장비와 촬영 설정이 같은 사진들 |
| 단계 | 화소 밝기의 단위 (0 – 255) |
| 대표 밝기 | 소스의 사진 전체 (두 사진 모두) 의 화소 중앙값. 분말의 밝기와 같다 |
| 보정 세트 | 사람 GT 의 사진을 보정해 저장해 둔 폴더 (`cache/v8_norm/<이름>/`). 라벨은 원본 그대로다 |
| 꼬리표 | run 이름에 붙는 실행의 이름 = 보정의 종류. `B` = 밝기 맞춤만, `BL` = 밝기 맞춤 + 조명 제거, `raw` = 보정 없음 (스모크용) |
| 기준선 | 보정 없이 학습한 실행 — v7 의 꼬리표 `N` (`…_10cls_v7-N-s<seed>`. `inlayer` seed 42 는 v6 의 실행). 이미 있다 |
| fold | 사람 GT 의 일부를 검증용으로 떼어 두고 나머지로 학습하는 한 번의 실행. 판정 fold 는 둘이다 — `inlayer` (소스마다 레이어 일부) 와 `loso_L21` (L21 의 20 레이어를 통째로 — 본 적 없는 장비) |
| seed | 학습의 난수 출발값. 같은 설정도 seed 가 다르면 점수가 조금 다르다 |
| step | 이미지 한 묶음 (16장 = GPU 4대 × 4장) 을 보고 가중치를 한 번 고치는 것 |

## 무엇이 어디에 있나

| 파일 | 하는 일 |
|---|---|
| `v8/config_v8.py` | 기준 밝기 (115), 지도의 격자 (96), 안전 장치의 문턱, 보정 세트 · 그림의 자리, run 이름 |
| `v8/data/photo_norm.py` | 사진 묶음에서 대표 밝기와 조명 지도를 구하고 사진에 건다. **라벨을 받지 않는다** (numpy · PIL 만) |
| `v8/data/build_norm_set.py` | 사람 GT 를 보정해 보정 세트로 저장하고, 점검 표 · 시트를 만든다 (라벨은 점검에서만 읽는다) |
| `v8/data/norm_set.py` | 보정 세트를 v6 의 로더가 읽게 한다 — 이 프로세스 안에서만 v6 설정의 `LEGACY_PBF_ROOT` 를 보정 세트로 바꾼다 |
| `v8/training/train.py` | 껍데기 — 보정 세트를 사람 GT 로 삼고 v7 의 학습 (`DefSeg_AM.v7.training.train`) 을 그대로 부른다 |
| `v8/inference/evaluate.py` | 껍데기 — 같은 방식으로 v6 의 채점을 부른다. 학습에 쓴 세트와 다른 세트로 채점하려 하면 멈춘다 |
| `v8/scripts/sensitivity.py` | 0단계 — 학습 없이, 기준선 모델이 밝기 · 조명에 얼마나 민감한지 잰다 |
| `v8/scripts/compare_runs.py` | 기준선과 v8 의 실행을 fold 마다 한 표로 모으고 판정 (결함 mIoU) |

## 사전조건

- NVIDIA driver + nvidia-container-toolkit (host), Docker / docker compose v2, `DefSeg_AM/venv`
- `ORNL_Data/DSCNN_Dataset_pbf-2026.11` — 사람 GT (v6 의 `convert` phase 가 만든다)
- `check` phase 만: 기준선의 ckpt — `checkpoints/vitb14_reg_dpt_dual_sz1036_10cls_v6/cv/inlayer/last.pt`, `checkpoints/…_v7-N-s42/cv/loso_L21/last.pt`
- `compare` phase 만: 기준선의 채점 결과 (`figures/<기준선 run>/v6/cv/<fold>/eval_native_last/metrics.json`) — 2026-11-19 에 만든 것이 있다
- 호스트 UID/GID 매칭용 `.env`

```bash
cd DefSeg_AM/v8/docker
echo "UID_GID=$(id -u):$(id -g)" > .env
```

## 실행

```bash
cd DefSeg_AM/v8/docker

# 1) 보정 세트 (CPU, 세트 하나에 약 4분) → cache/v8_norm/{b115, bl115}/ , 점검은 figures/v8_norm/<세트>/
DEFSEG_V8_PHASE=norm docker compose up --build

# 2) 0단계 — 학습 없는 민감도 점검 (GPU 1대) → figures/v8_check/sz1036_s42/
DEFSEG_V8_PHASE=check NVIDIA_VISIBLE_DEVICES=2 docker compose up --build

# 3) 학습 — 판정 fold 둘을 seed 42 로. 2회, 약 4시간 (GPU 4대). 사람 GT 만 학습한다 (생성 데이터 없음)
DEFSEG_V8_PHASE=cv DEFSEG_V8_VARIANT=B DEFSEG_V8_RUNS="42:inlayer 42:loso_L21" docker compose up -d --build
docker compose logs -f
#    조명 제거까지 (꼬리표 BL) — 역시 2회, 약 4시간. 사람 GT 만 학습한다
DEFSEG_V8_PHASE=cv DEFSEG_V8_VARIANT=BL DEFSEG_V8_RUNS="42:inlayer 42:loso_L21" docker compose up -d --build

# 4) 기준선과 한 표로 모으고 판정 (GPU 불필요) → figures/v8_compare/sz1036/{inlayer, loso_L21}.md
DEFSEG_V8_PHASE=compare docker compose up --build

docker compose down
```

- 두 명령을 동시에 띄우지 않는다 — 같은 서비스라 뒤의 `up` 이 앞의 컨테이너를 바꿔 버린다. 한 줄에 seed 와 fold 를 여럿 주거나, 앞의 것이 끝난 뒤에 띄운다.
  (꼭 함께 돌려야 하면 `docker compose -p defseg-v8-b …` 로 프로젝트 이름을 달리 준다.)
- **끝난 실행은 건너뛴다** (`checkpoints/<run>/cv/<fold>/last.pt` 가 있으면). 다시 돌리려면 `DEFSEG_V8_FORCE=1`. 학습 도중에 끊긴 실행은 처음부터 다시 돈다.
- 보정 세트가 없으면 `cv` · `eval` 은 FATAL 로 멈춘다. 세트를 만든 뒤 학습 전에 `figures/v8_norm/<세트>/` 의 `check.md` 와 `sheet_<소스>.png` 를 본다.

## Phase

| `DEFSEG_V8_PHASE` | 하는 일 | GPU |
|---|---|---|
| (없음) | 사용법만 찍고 끝난다 | – |
| `norm` | 보정 세트를 만든다. `DEFSEG_V8_NORM_MODES` (기본 `B BL`) 의 종류마다 하나씩. 이미 있는 세트는 건너뛴다 (`DEFSEG_V8_EXTRA=--overwrite` 면 다시 만든다) | – |
| `check` | 0단계. 기준선 모델 (seed 42) 에 밝기를 옮긴 사진 · 가짜 조명을 얹은 사진 · 조명을 걷어낸 사진을 넣어 결함 mIoU 를 잰다 | 1대 |
| `cv` | seed × fold 마다: 학습 (v7 의 스크립트, 보정 세트의 사진) → 채점 (v6 의 `evaluate`, 같은 세트의 사진, 원본 해상도 + 1036²) → 요약 (v6 의 `summarize_cv`). 끝에 비교 표 | `NPROC` 대 (기본 4) |
| `eval` | 채점만 다시 → 요약 → 비교 표 | 1대 |
| `compare` | 기준선 (N) 과 v8 의 실행을 fold 마다 한 표로 모으고 PLAN_v8 §4.4 의 판정을 낸다 | – |

## 환경변수

| 변수 | 기본값 | 뜻 |
|---|---|---|
| `DEFSEG_V8_PHASE` | (없음) | 위 표 |
| `DEFSEG_V8_VARIANT` | `B` | 꼬리표 = 보정의 종류. `B` · `BL` · `raw`. run 이름과 보정 세트가 이것으로 정해진다 |
| `DEFSEG_V8_NORM_SET` | (꼬리표로 정한다) | 보정 세트의 이름. 비우면 `B` → `b115`, `BL` → `bl115`, `raw` → `raw` |
| `DEFSEG_V8_NORM_MODES` | `B BL` | `norm` phase 가 만들 세트의 종류 |
| `DEFSEG_V8_FOLDS` | `inlayer loso_L21` | 검증 fold |
| `DEFSEG_V8_SEEDS` | `42` | seed. 여럿이면 공백으로: `"42 43"` |
| `DEFSEG_V8_RUNS` | (없음) | `seed:fold` 의 목록. 주면 `SEEDS × FOLDS` 대신 이 목록대로, 적은 순서로 돈다 |
| `DEFSEG_V8_FORCE` | `0` | `1` 이면 끝난 실행도 다시 돈다 |
| `DEFSEG_V8_EXTRA` | (없음) | 그 phase 의 스크립트에 그대로 넘기는 인자 (`--quick`, `--total-steps 60`, `--overwrite`, `--folds loso_L21` …) |
| `NVIDIA_VISIBLE_DEVICES` | `0,1,2,3` | 쓸 GPU |
| `NPROC` | `4` | 학습 프로세스 수 (GPU 수). 4 가 아니면 effective batch 가 달라져 기준선과 견줄 수 없다 |

## 산출물

```
cache/v8_norm/<세트>/                       보정 세트 — visible/{0,1}/{id}.png (보정한 사진), annotations (원본으로 가는 상대 링크),
                                            index.csv · classes.json · conversion.json (원본의 것), norm.json (소스마다의 값 · 지도 · 안전 장치)
figures/v8_norm/<세트>/                     check.md · check.json (점검 표), maps.png (소스마다의 지도), sheet_<소스>.png (보정 전 · 후)
figures/v8_check/sz1036_s42/                0단계 — summary.md, results.json, curves.png, fake_light.png
checkpoints/<run>/cv/<fold>/{best, last}.pt run = vitb14_reg_dpt_dual_sz1036_10cls_v8-<꼬리표>-s<seed>
figures/<run>/v6/cv/<fold>/                 run.json · curve.csv · result.json (v7 의 학습이 쓴다), norm_set.json (어느 보정 세트로 학습했는가),
                                            eval_{native,resized}_last/ (v6 의 채점)
figures/v8_compare/sz1036/                  <fold>.md, summary.json
```

- 그림 쪽 경로에 `v6` 가 들어가는 것은 폴더를 v6 의 함수가 만들기 때문이다 (v7 과 같다).
- ckpt 의 `config.dataset.root` 에 보정 세트의 자리가 적힌다. **v8 의 모델에는 같은 보정을 건 사진을 넣어야 한다** — `v8/inference/evaluate.py` 가 그것을 확인한다.
  v6 의 `infer` · `ornl_compare` 처럼 사진을 직접 읽는 스크립트는 보정을 걸지 않으므로 v8 의 ckpt 에 그대로 쓰면 안 된다 (그 단계는 아직 만들지 않았다 — PLAN_v8 §5).

## 지금 있는 것 (2026-11-26)

- **보정 세트 셋**: `cache/v8_norm/{b115, bl115, raw}/` (각 437 MB). 소스마다 더한 값은 L21 +17 · 17-4 −39 · GP −14 · I1 −25 · I2 −26 · MS +60 · BJ21 +3 · H13 +15.
  `bl115` 는 거기에 더해 L21 과 BJ21 의 사진을 그 소스의 조명 지도로 나눴다 (안전 장치를 지난 소스가 이 둘뿐이다). `raw` 는 화소가 원본과 같다 — 스모크용이다.
  보정 뒤 분말의 밝기는 여덟 소스 모두 113 – 120 이다 (`figures/v8_norm/bl115/check.md`).
- **0단계의 결과**: `figures/v8_check/sz1036_s42/` (GPU 1대로 3분 30초). 읽은 것은 [../PLAN_v8.md](../PLAN_v8.md) §9.3 에 있다. 줄이면:
  - `loso_L21` 은 밝기를 −40 … +60 단계 옮겨도 결함 mIoU 가 0.157 – 0.177 로, seed 의 흔들림 안에서만 움직인다 — L21 의 점수가 낮은 까닭은 밝기 차이가 아니다.
  - `inlayer` 는 밝게 하면 +0.018 까지 오르는데 거의 OM 한 클래스다 (검증 화소가 가장 적다).
  - 가짜 조명은 최대 − 최소 10% 까지 두 fold 모두 점수를 움직이지 않는다. 이 데이터의 진짜 조명을 걷어내도 점수가 그대로다 → 미리 정한 규칙대로면 `BL` 은 돌리지 않는다.
  - 조명 제거는 얹은 지도를 상관 0.95 로 되찾지만 크기의 1/3 쯤이 남는다 (2차식 면의 한계). 모든 소스를 나누면 GP 가 크게 떨어진다 — 안전 장치는 필요하다.
- **`B` 의 풀런이 2026-11-26 에 끝났다** (seed 42, 두 fold, 119분 + 118분 — `…_10cls_v8-B-s42`). 결함 mIoU 는 `inlayer` 0.467 → 0.477, `loso_L21` 0.169 → 0.161 로 둘 다 문턱 (0.487 · 0.189) 아래다 — **밝기를 맞춰도 오르지 않았다.**
  `inlayer` 의 +0.011 은 OM 이 오르고 (0.181 → 0.348 — 잘못 짚는 화소가 절반으로) DB 가 내린 (0.466 → 0.379 — BJ21 에서 PS 와 헷갈린다) 합이고, 1036² 에서는 차이가 없다 (0.541, 기준선 0.533 – 0.545).
  표는 `figures/v8_compare/sz1036/`, 읽은 것은 [../PLAN_v8.md](../PLAN_v8.md) §9.5, 로그는 `logs/v8_cv_B_s42_2026-11-26.log`. `BL` 과 seed 43 은 돌리지 않았다.

## 스모크에서 확인한 것 (2026-11-25)

| 실행 | 확인한 것 |
|---|---|
| 보정 세트 `raw` 로 v8 의 학습 (GPU 1대, 1036², 30 step, seed 42) 을 v7 의 같은 설정 (`logs/v7_smoke_equivalence_v7side_2026-11-18.log`) 과 대조 | loss 가 모든 로그 지점에서 같다 (2.3343 → 1.2533), 검증 점수도 같다. class weight · 추출 비중 · Copy-Paste 의 조각 목록 · 배선 확인도 같다. 껍데기는 사진의 자리 말고 아무것도 바꾸지 않는다 |
| compose `cv --quick`, GPU 1대, 꼬리표 `B`, 두 fold | 학습 → 채점 (같은 보정 세트의 사진) → 요약 → 비교 표가 끝까지 돈다 |
| compose `cv --quick`, GPU 2대, 꼬리표 `BL` / 같은 명령을 한 번 더 | 두 rank 의 decoder 초기값이 같다. `norm_set.json` 이 남는다 / 건너뛴다 |
| `--synth-set` 을 준 학습 · 없는 세트 · 기준선 모델을 보정 세트로 채점 · phase 없음 | FATAL 로 멈춘다 / 사용법만 찍고 끝난다 |
| `check` 에서 사진을 바꾸지 않은 조건 | 저장해 둔 채점 (`eval_native_last`) 과 결함 mIoU 가 같다 (차 0) |
| `photo_norm` 의 통계 | 2026-11-24 에 따로 잰 값과 같다 (대표 밝기, 얼룩의 크기 — 여덟 소스 모두) |

"보정 없는 세트 = 기준선" 스모크를 다시 돌리는 법 (v6 · v7 을 고친 뒤):

```bash
DEFSEG_V8_PHASE=norm DEFSEG_V8_NORM_MODES=raw docker compose up --build          # raw 세트가 없을 때만
DEFSEG_V8_PHASE=cv NPROC=1 NVIDIA_VISIBLE_DEVICES=2 DEFSEG_V8_VARIANT=smokeraw DEFSEG_V8_NORM_SET=raw \
  DEFSEG_V8_RUNS="42:inlayer" DEFSEG_V8_EXTRA="--total-steps 30 --eval-every 15 --log-every 5" docker compose up --build
# 로그의 loss (step 1 · 5 · 10 · 15 · 20 · 25 · 30) 가 2.3343 · 2.2285 · 2.0149 · 1.6454 · 1.4867 · 1.1849 · 1.2533 이면 같다.
# 끝나면 checkpoints/…_v8-smokeraw-s42 와 figures/…_v8-smokeraw-s42 를 지운다 (비교 표에 끼지 않게).
```

`--quick` 과 30 step 의 점수에는 뜻이 없다. 스모크의 ckpt 와 그림은 지웠고 로그만 남겼다 (`logs/v8_*_2026-11-25.log`).

## 주의

- **v6 · v7 을 불러 쓴다.** `v6/` 나 `v7/training/train.py` 가 바뀌면 v8 의 동작도 바뀐다. 고친 뒤에는 아래의 "보정 없는 세트 = 기준선" 스모크를 다시 돌린다.
- **GPU 4대를 2시간씩 쓴다.** GPU 0 · 1 에는 서빙 컨테이너가 있다 (각 3 – 4 GB). 스모크와 `check` 는 GPU 2 · 3 에서 돌린다.
- **통계에 검증 레이어의 사진이 들어간다.** 대표 밝기와 조명 지도는 소스의 사진 전부로 구한다 (정답은 읽지 않는다). 새 장비의 사진을 받았을 때 하는 일과 같다 (PLAN_v8 §1).
- **원본 데이터셋을 다시 변환하면 보정 세트도 다시 만든다.** 세트가 원본의 옛 변환으로 만든 것이면 학습 · 채점이 FATAL 로 멈춘다.
- **DSCNN 예측을 읽지 않는다.** v6 와 같다.
