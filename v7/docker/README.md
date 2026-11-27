# DefSeg-AM v7 — Docker 사용법

> v7 = v6 + 합성 도메인 하나. 설계와 판정 기준은 [../PLAN_v7.md](../PLAN_v7.md).
> **v6 의 코드를 복사하지 않고 불러 쓴다.** `v7/` 에 있는 것은 학습 스크립트 (`training/train.py` — v6 의 것을 복사해 고친 하나뿐인 사본),
> 합성 세트를 붙이는 쪽 (`data/synth_domain.py`), 실행을 모으는 표 (`scripts/compare_runs.py`) 다. 채점과 요약은 v6 의 모듈을 run 이름만 바꿔 부른다.
> 호스트의 `DefSeg_AM/venv` 를 read-only bind mount 하여 그대로 쓴다.

**용어**

| 말 | 뜻 |
|---|---|
| step | 이미지 한 묶음 (16장 = GPU 4대 × 4장) 을 보고 가중치를 한 번 고치는 것 |
| fold | 사람 GT 의 일부를 검증용으로 떼어 두고 나머지로 학습하는 한 번의 실행. v7 의 판정 fold 는 둘이다 — `inlayer` (소스마다 레이어 일부를 뗀다) 와 `loso_L21` (L21 장비의 20 레이어를 통째로 뗀다 — 본 적 없는 장비) (PLAN_v7 §7 의 V12) |
| seed | 학습의 난수 출발값. 같은 설정도 seed 가 다르면 점수가 조금 다르다 |
| 꼬리표 | run 이름에 붙는 실행의 이름. `N` = 합성 없음 (기준선), `W` = 라벨 맵으로 그린 합성 레이어를 넣은 것 |
| 합성 세트 · 몫 | 생성 모델이 그린 레이어의 묶음 (`cache/genaug/sets/<이름>/`). 몫 = 학습 표본을 뽑을 때 합성 세트에서 뽑는 비율 |

## 사전조건

- NVIDIA driver + nvidia-container-toolkit (host), Docker / docker compose v2, `DefSeg_AM/venv`
- `ORNL_Data/DSCNN_Dataset_pbf-2026.11` — 사람 GT (v6 의 `convert` phase 가 만든다)
- 합성 실행만: `cache/genaug/sets/<이름>/` — 합성 세트 (genaug 의 `synth` phase 가 만든다 — [../../genaug/docker/README.md](../../genaug/docker/README.md). 아래 "지금 있는 것")
- 호스트 UID/GID 매칭용 `.env`

```bash
cd DefSeg_AM/v7/docker
echo "UID_GID=$(id -u):$(id -g)" > .env
```

## 실행

```bash
cd DefSeg_AM/v7/docker

# 1) 기준선 — 합성 없음 (v6 와 같은 학습). 한 번에 약 2시간 (GPU 4대). **2026-11-19 에 다섯 번 모두 끝났다**
#    (inlayer seed 43 · 44 — seed 42 는 v6 의 P0 로 대신한다, loso_L21 seed 42 · 43 · 44). 같은 명령을 다시 주면 전부 건너뛴다
DEFSEG_V7_PHASE=cv DEFSEG_V7_RUNS="42:loso_L21 43:loso_L21 44:loso_L21 43:inlayer 44:inlayer" docker compose up -d --build
docker compose logs -f

# 2) 합성 세트를 넣은 실행 (꼬리표 W) — 판정 fold 둘을 seed 42 로. 2회, 약 4시간 (PLAN_v7 §4.2 의 1단계)
DEFSEG_V7_PHASE=cv DEFSEG_V7_VARIANT=W DEFSEG_V7_SYNTH_SET=w600_v1 DEFSEG_V7_RUNS="42:inlayer 42:loso_L21" docker compose up -d --build
#    목록 대신 seed 만 줘도 된다 (fold 의 기본값이 판정 fold 둘이다): DEFSEG_V7_SEEDS="42"

# 3) 실행들을 한 표로 모으고 판정 (GPU 불필요) → figures/v7_compare/sz1036/{inlayer, loso_L21}.md
DEFSEG_V7_PHASE=compare docker compose up --build

docker compose down
```

- 두 명령을 동시에 띄우지 않는다 — 같은 서비스라 뒤의 `up` 이 앞의 컨테이너를 바꿔 버린다. 한 줄에 seed 와 fold 를 여럿 주거나, 앞의 것이 끝난 뒤에 띄운다.
- **끝난 실행은 건너뛴다** (`checkpoints/<run>/cv/<fold>/last.pt` 가 있으면). 중간에 끊겨도 같은 명령을 다시 띄우면 남은 것부터 돈다. 다시 돌리려면 `DEFSEG_V7_FORCE=1`.
  학습 도중에 끊긴 실행은 처음부터 다시 돈다 (이어 돌리기는 없다 — v6 와 같다).

## Phase

| `DEFSEG_V7_PHASE` | 하는 일 | GPU |
|---|---|---|
| (없음) | 사용법만 찍고 끝난다 | – |
| `cv` | seed × fold 마다: 학습 (v7 의 스크립트) → 채점 (v6 의 `evaluate`, 원본 해상도 + 1036²) → 요약 (v6 의 `summarize_cv`). 끝에 비교 표 | `NPROC` 대 (기본 4) |
| `train` | 전수 학습 (사람 GT 125 레이어 + 합성 세트). 검증 없음 | `NPROC` 대 |
| `eval` | 채점만 다시 → 요약 → 비교 표 | 1대 |
| `summarize` | 요약만 다시 | – |
| `compare` | 실행 여럿을 fold 마다 한 표로 모으고 PLAN_v7 §4.3 의 판정을 낸다 | – |
| `gt_compare` | 사람 GT 의 검증 레이어 (그 fold 가 떼어 둔 것 전부) 에서 **정답** 과 v4 · v5 · v6 · v7 의 예측을 나란히 그린다. DSCNN 의 열은 없다 (DSCNN 의 예측은 ORNL 빌드에만 있다). v5 는 Soot 를 Spatter 로 배운 실행이 있으면 그것을 쓴다. 예측 칸에 레이어마다 맞힌 비율 · 잘못 짚은 비율을 적는다. 예: `DEFSEG_V7_PHASE=gt_compare DEFSEG_V7_VARIANT=W NVIDIA_VISIBLE_DEVICES=2 DEFSEG_V7_EXTRA="--fold inlayer"` → `figures/<run>/v6/cv/<fold>/gt_val<N>_last/` (22 레이어에 20초) | 1대 |
| `ornl_compare` | ORNL 다섯 빌드의 무작위 레이어에서 DSCNN · v4 · v5 · v6 · v7 의 예측을 나란히 그린다 (사람 GT 가 없어 점수가 아니다 — 참고용). v6 열은 같은 fold · 같은 seed 의 기준선이다. 예: `DEFSEG_V7_PHASE=ornl_compare DEFSEG_V7_VARIANT=W NVIDIA_VISIBLE_DEVICES=2 DEFSEG_V7_EXTRA="--fold inlayer --n-per-build 4"` → `figures/<run>/v6/cv/<fold>/ornl_random<N>_last/` (20 레이어에 1분 30초) | 1대 |

## 환경변수

| 변수 | 기본값 | 뜻 |
|---|---|---|
| `DEFSEG_V7_PHASE` | (없음) | 위 표 |
| `DEFSEG_V7_FOLDS` | `inlayer loso_L21` | 검증 fold. v7 의 판정 fold 둘이다. 합성 세트는 그 세트의 `manifest.json` 의 `valid_folds` 에 있는 fold 에만 넣을 수 있다 (아니면 학습이 멈춘다) |
| `DEFSEG_V7_SEEDS` | `42` | seed. 여럿이면 공백으로: `"42 43 44"` |
| `DEFSEG_V7_RUNS` | (없음) | `seed:fold` 의 목록. 주면 `SEEDS × FOLDS` 대신 이 목록대로, 적은 순서로 돈다: `"42:inlayer 42:loso_L21"` |
| `DEFSEG_V7_VARIANT` | `N` | 꼬리표. `N` 은 합성 없는 기준선에만 쓴다 (합성 세트와 함께 주면 멈춘다) |
| `DEFSEG_V7_SYNTH_SET` | (없음) | 합성 세트의 이름 (`cache/genaug/sets/` 아래). 비우면 v6 와 같은 학습이다 |
| `DEFSEG_V7_SYNTH_SHARE` | `0.3` | 합성 도메인의 몫 |
| `DEFSEG_V7_FORCE` | `0` | `1` 이면 끝난 실행도 다시 돈다 |
| `DEFSEG_V7_EXTRA` | (없음) | 학습 스크립트에 그대로 넘기는 인자 (`--quick`, `--total-steps 60`, `--no-copy-paste` …) |
| `NVIDIA_VISIBLE_DEVICES` | `0,1,2,3` | 쓸 GPU |
| `NPROC` | `4` | 학습 프로세스 수 = GPU 수. 4 가 아니면 effective batch 가 달라져 기준선 · v6 와 견줄 수 없다 |

## 산출물

run 이름은 `vitb14_reg_dpt_dual_sz1036_10cls_v7-<꼬리표>-s<seed>` 다 (예: `…_v7-N-s43`, `…_v7-W-s42`).

| 경로 | 내용 |
|---|---|
| `checkpoints/<run>/cv/<fold>/{best.pt, last.pt}` | fold 모델 (EMA 사본). 채점하는 것은 `last.pt` 다 |
| `figures/<run>/v6/cv/<fold>/` | `run.json` (설정, 사람 GT 표본, 합성 표본 `synth_samples`), `curve.csv`, `result.json`, `eval_{native,resized}_last/metrics.json` |
| `figures/<run>/v6/cv/<fold>/domain_curve.csv` | 합성 실행만 — 검증 사이 구간마다 사람 GT 쪽과 합성 쪽의 loss, 뽑힌 표본 수 (rank 0 이 본 배치) |
| `figures/<run>/v6/cv/summary.md` | v6 의 요약 (run 하나의 fold 들) |
| `figures/v7_compare/sz1036/<fold>.md` | 실행 여럿을 모은 표와 판정 |

그림 폴더 경로에 `v6` 가 들어가는 것은 폴더를 v6 의 함수로 만들기 때문이다 (v6 의 채점 · 요약 스크립트가 그대로 찾게).

**비교 표를 읽는 법.** `T` 는 대상 클래스 (RS · RH · SE · DB 가운데 그 fold 에서 채점되는 것) 의 IoU 평균이다. 기준선은 꼬리표 `N` 의 seed 평균이고 σ 는 그 표준편차다.
`inlayer` 의 `N` · seed 42 가 없으면 v6 의 P0 를 그 자리에 놓는다. 판정의 문턱 (0.03, Powder · Printed 0.01, 잘못 짚은 비율 1.0%p) 은 제안값이다.
`loso_L21` 에는 RH 의 정답이 없어 T 가 RS · SE · DB 의 평균이다. 비교 표는 판정 fold 둘을 언제나 함께 만든다.

## 지금 있는 것 (2026-11-19)

- **기준선 다섯 번이 끝났다** (합성 없음): `inlayer` 의 T 0.453 (σ 0.0042) → 효과의 문턱 0.483, `loso_L21` 의 T 0.173 (σ 0.0149) → 문턱 0.203. 표는 `figures/v7_compare/sz1036/`.
- **합성 세트 `w600_v1` 이 있다** — `cache/genaug/sets/w600_v1/` (600 장, 1.1 GB. genaug 의 `synth` phase 로 만들었다). 관문 K2 – K4 를 통과했고 K1 (사람이 시트를 본다) 이 남았다.
  점검은 `figures/genaug/sets/w600_v1/`, 요약은 PLAN_v7 의 "구현 상태 → 합성 세트 `w600_v1`".
  `smoke24_v0` (24 장) 은 그 전에 스모크용으로 만든 것이다 — 실험에 쓰지 않는다.
- **합성을 넣은 실행 (꼬리표 W, seed 42) 이 두 fold 에서 끝났다** (2026-11-19): `inlayer` 는 T 0.453 → 0.447 로 차이 없음 (RS 0.260 → 0.306, DB 0.466 → 0.419),
  `loso_L21` 은 T 0.173 → 0.171 이고 SP 가 0.196 → 0.064 로 내려가 부작용 판정이다. 표는 `figures/v7_compare/sz1036/`, 읽은 것은 PLAN_v7 의 "구현 상태 → 주 실험의 결과".
- 전수 학습 모델을 v6 와 ORNL 사진에서 나란히 보는 것 (PLAN_v7 §4.4) 은 아직 없다.

## 스모크에서 확인한 것 (2026-11-18 · 19)

| 실행 | 확인한 것 |
|---|---|
| v6 의 학습과 v7 의 학습 (합성 없음) 을 같은 설정으로 (GPU 1대, 1036², 30 step, seed 42) | loss 가 모든 로그 지점에서 같다 (2.3343 → 1.2533), 검증 점수도 같다. class weight · 추출 비중 · Copy-Paste 의 조각 목록 · 배선 확인도 같다. 그래서 `inlayer` 의 기준선 seed 42 는 v6 의 P0 로 대신한다 |
| compose `cv --quick`, GPU 1대, 합성 세트 `smoke24_v0` | 학습 → 채점 (v6 의 `evaluate`) → 요약 (v6 의 `summarize_cv`) → 비교 표가 끝까지 돈다. 추출의 몫이 legacy 0.7 · synth 0.3 으로 잡힌다 |
| 실제 해상도 60 step, GPU 2대, 합성 세트 | 초당 1.41 step (처음의 느린 구간이 든 값. v6 는 1.46), GPU 당 23.6 GB (v6 와 같다). class weight · Copy-Paste 의 조각 목록 · 배선 확인이 합성 없는 실행과 같다 — 합성 화소가 통계에 들어가지 않는다. rank 0 이 본 표본 240 개 가운데 합성이 67 개 (28%) 다 |
| `DEFSEG_V7_RUNS` 에 두 실행 (`--quick`) | 목록대로 두 실행이 돈다. 같은 명령을 다시 띄우면 둘 다 건너뛴다 |
| 꼬리표 `N` + 합성 세트 / phase 없음 | FATAL 로 멈춘다 / 사용법만 찍고 끝난다 |
| 비교 표의 판정 | 지어낸 값 (seed 셋의 기준선과 두 실행) 을 넣었을 때 "효과" 와 "부작용" 이 문턱대로 나온다 |
| (2026-11-19) 합성 세트 `w600_v1`, 실제 해상도 40 step, GPU 1대 | 725 표본 (사람 GT 125 + 합성 600) 을 올리는 데 39초. 추출의 몫 legacy 0.7 · synth 0.3 (빌드마다 0.1). class weight · Copy-Paste 의 조각 목록 · 배선 확인이 합성 없는 실행과 같다 |
| (2026-11-19) `valid_folds` 에 없는 fold (`inlayer_k0`) 에 합성 세트 | FATAL 로 멈춘다 |

`--quick` 과 40 · 60 step 의 점수에는 뜻이 없다. 스모크의 ckpt 와 그림은 지웠고 로그만 남겼다 (`logs/v7_smoke_*.log`).
GPU 4대 · 10,000 step 의 실행은 2026-11-19 의 기준선 다섯 번이 처음이다 (`inlayer` seed 43 · 44, `loso_L21` seed 42 · 43 · 44. 한 번에 118 – 119분).
결과는 PLAN_v7 의 "구현 상태 → 기준선 결과" 와 `figures/v7_compare/sz1036/` 에 있다.

## 주의

- **v6 를 불러 쓴다.** `v6/` 의 코드가 바뀌면 v7 의 동작도 바뀐다. v6 를 고친 뒤에는 위의 "합성 없음 = v6" 스모크를 다시 돌리고, 그 앞뒤의 실행을 한 표에 놓지 않는다.
- **GPU 4대를 2시간씩 쓴다.** GPU 0 · 1 에는 서빙 컨테이너가 있고 (각 3–4 GB), 학습은 GPU 당 약 23.6 GB 를 쓴다. 스모크는 GPU 2 · 3 에서 돌린다.
- **누출.** 지금의 생성 모델은 판정 fold 둘의 검증 레이어 (`inlayer` 의 22 + L21 의 20, 4장이 겹친다) 를 보지 않았다. 그 밖의 fold 에 합성 세트를 넣으면 검증 레이어를 본 생성 모델의 그림으로 학습하는 것이 된다 (PLAN_v7 §2.4).
  학습 스크립트가 세트의 `manifest.json` 의 `valid_folds` 를 보고, 거기 없는 fold 면 FATAL 로 멈춘다 (전수 학습은 막지 않는다).
- **DSCNN 예측을 읽지 않는다.** v6 와 같다.
