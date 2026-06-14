# DefSeg-AM v2 — Docker 사용법

> 호스트의 `DefSeg_AM/venv` (torch 2.9.1+cu128 검증된 환경) 를 read-only bind mount 하여
> 컨테이너에서 그대로 사용. 이미지에 `pip install` 안 함.

## 사전조건

- NVIDIA driver + nvidia-container-toolkit (host)
- `DefSeg_AM/venv` 가 host 에서 동작 (torch 2.9.1+cu128 + cuDNN 9.10.2)
- `ORNL_Data/` (HDF5 + DSCNN_Dataset) 가 host 에 존재
- Docker / docker compose v2

## 빠른 시작

```bash
cd DefSeg_AM/v2/docker

# 1) 캐시 빌드 (label 만 8-class 재매핑, ~5분)
DEFSEG_PHASE=build_cache docker compose up -d --build
docker compose logs -f

# 2) Stage 1 학습 (~12h, 1 GPU)
DEFSEG_PHASE=stage1 docker compose up -d --build

# 3) Stage 2 — 단일 fold (~20분/fold)
DEFSEG_PHASE=stage2 DEFSEG_FOLD=0 docker compose up -d --build

# 3') Stage 2 — 8 fold 전체 (~3h, 컨테이너 안에서 loop)
DEFSEG_PHASE=stage2 DEFSEG_FOLD=all docker compose up -d --build

# 4) 전체 파이프라인 (1 + 2 + 3')
DEFSEG_PHASE=all docker compose up -d --build

# 종료 / 정리
docker compose down
```

## 환경변수

| 변수 | 값 | 기본 |
|---|---|---|
| `DEFSEG_PHASE` | `build_cache` / `stage1` / `stage2` / `infer` / `all` | `stage1` |
| `DEFSEG_FOLD` | `0..7` 또는 `all` (stage2 시) | `0` |
| `DEFSEG_EXTRA` | extra CLI 인자 (e.g. `"--use-hard-bootstrap"`, `"--quick"`) | "" |
| `NVIDIA_VISIBLE_DEVICES` | GPU index | `0` |
| `OMP_NUM_THREADS` | CPU thread/proc | `8` |

## 예시

```bash
# Stage 1 with hard-bootstrap loss (DSCNN noisy label 대응)
DEFSEG_PHASE=stage1 DEFSEG_EXTRA="--use-hard-bootstrap" docker compose up -d --build

# Quick smoke test (2 epoch, 224×224)
DEFSEG_PHASE=stage1 DEFSEG_EXTRA="--quick" docker compose up -d --build

# 다른 GPU 로 stage2 fold 3
NVIDIA_VISIBLE_DEVICES=2 DEFSEG_PHASE=stage2 DEFSEG_FOLD=3 docker compose up -d --build

# 추론 — ensemble + TTA
DEFSEG_PHASE=infer DEFSEG_EXTRA="--build '2021-07-13 TCR Phase 1 Build 1' --n-layers 10" \
    docker compose up -d --build
```

## 산출물

호스트의 `DefSeg_AM/` 폴더에 직접 저장됨 (rw bind mount):

```
DefSeg_AM/
├── cache/
│   ├── resized_sz1036/                     # v1 image cache (재사용)
│   └── resized_sz1036_v2/                  # v2 label cache (8-class)
├── checkpoints/vits14_dpt_dual_sz1036_8cls_v2/
│   ├── stage1_best.pt
│   ├── stage2_best_fold{0..7}_<src>.pt
│   └── cv_summary.json
├── figures/vits14_dpt_dual_sz1036_8cls_v2/
└── v2/
    ├── build_cache_v2.log
    ├── stage1_v2.log
    └── stage2_v2_fold{0..7}.log
```

## 디버깅

```bash
# 컨테이너 ID 확인
docker compose ps

# shell 접속
docker compose exec defseg-am-v2 bash

# 컨테이너 안에서 직접 python 호출
docker compose exec defseg-am-v2 /workspace/DefSeg_AM/venv/bin/python \
    -m DefSeg_AM.v2.training.train_stage1 --quick
```

## venv 마운트 경로 확인

```bash
docker compose exec defseg-am-v2 ls -la /workspace/DefSeg_AM/venv/bin/python
docker compose exec defseg-am-v2 /workspace/DefSeg_AM/venv/bin/python -c "import torch; print(torch.__version__)"
# 기대: 2.9.1+cu128
```
