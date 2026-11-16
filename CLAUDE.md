# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## 1. Project

L-PBF powder bed 카메라 이미지(layer 당 visible/0 melt 직후 + visible/1 도포 직후 dual 입력)에서
결함을 픽셀 단위 semantic segmentation 하는 모델. **DINOv2 (frozen, weight-shared) ×2 →
per-stage `concat(f0, f1, f1−f0)` 융합 → 자체 구현 DPT decoder → class logits**.
학습 대상은 decoder/head 뿐이고 **checkpoint 에도 trainable weight 만 저장**된다
(로드 시 backbone 은 torch.hub 로 별도 로드).

**2-Stage 학습** (모든 버전 공통):
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

버전 간 비교 figure 는 repo 루트 `scripts/` 에서:
`python -m DefSeg_AM.scripts.compare_v3_v4_multi_build` 등 (역시 CWD = `3DP_VPPM/`).

테스트/린트 설정은 현재 없다.

## 4. 버전 구조 (v1 → v4 실험 계보)

```
common/     # 모든 버전 공유: config.py(경로·ORNL 12-class·MATERIAL_TO_ORNL), models/(DefSegModel·losses),
            # data/(image_utils·DistributedWeightedSampler), training/dist_utils.py, utils/log.py
v1/         # 12-class (ORNL native), ViT-S/14, 1 GPU
v2/         # 8-class (Recoater 통합+3종 제거), 8-fold leave-one-source-out CV, D4+cyclic+DSCNN aug
v3/         # 10-class 복원, CV 없음(8 source 전수 train + Build 1 평가), 4-GPU DDP, 휴리스틱 후처리
v4/         # 11-class (+Under Melting), ViT-B/14_reg, class-aware sampler, Copy-Paste aug, 후처리 2규칙
```

- 의존 방향: **vN → common 만**. 버전끼리는 서로 임포트하지 않고, 새 버전은 이전 버전
  코드를 **복사 후 수정**한다 (예: `v4/models/losses_v4.py` 는 v3 코드의 자체 사본).
  새 실험 추가 시 이 패턴을 따를 것: `vN/{config_vN.py, data/, models/, training/, inference/, scripts/, docker/}` + `PLAN_vN.md`.
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

## 6. 산출물 규약

`cache/ checkpoints/ figures/ logs/ venv/` 는 gitignore 이며 전 버전이 공유하되
**run-name 으로 구분**: `<backbone>_dpt_dual_sz<img>_<n>cls_v<N>`
(예: `vitb14_reg_dpt_dual_sz1036_11cls_v4`). ckpt 는
`checkpoints/<run-name>/{stage1_best, stage2_best}.pt`, figure 는 `figures/<run-name>/...`.

## 7. 문서 맵 — 작업 전 참조

- `docs/MODEL_ARCHITECTURE.md` — 텐서 크기 추적·FFB/RCU 내부·forward 코드 대응
- 각 `vN/PLAN_vN.md` — 실험 설계·hyperparameter 확정값·미해결 항목 (가장 상세한 스펙)
- `v3/RESULTS_v1_vs_v2_vs_v3.md`, `v2/RESULTS_v1_vs_v2.md` — 버전 간 정량 비교
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
