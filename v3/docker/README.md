# DefSeg-AM v3 — Docker 사용법

> 호스트의 `DefSeg_AM/venv` (torch 2.9.1+cu128 검증된 환경) 를 read-only bind mount 하여
> 컨테이너에서 그대로 사용. 이미지에 `pip install` 안 함.

## 사전조건

- NVIDIA driver + nvidia-container-toolkit (host)
- `DefSeg_AM/venv` (torch 2.9.1+cu128 + cuDNN 9.10.2)
- `ORNL_Data/` (HDF5 + DSCNN_Dataset) 가 host 에 존재
- Docker / docker compose v2
- **4 GPU** (default. `NPROC` + `NVIDIA_VISIBLE_DEVICES` 로 조정 가능)
- 호스트 UID/GID 매칭용 `.env` (없으면 1000:1000)

```bash
echo "UID_GID=$(id -u):$(id -g)" > .env
```

## 빠른 시작

```bash
cd DefSeg_AM/v3/docker

# 1) 캐시 빌드 (label 만 10-class 재매핑, ~5분)
DEFSEG_V3_PHASE=build_cache docker compose up -d --build
docker compose logs -f

# 2) Stage 1 학습 (4-GPU DDP, ~3-4h)
DEFSEG_V3_PHASE=stage1 docker compose up -d --build

# 3) Stage 2 학습 (4-GPU DDP, ~1-2h)
DEFSEG_V3_PHASE=stage2 docker compose up -d --build

# 4) 전체 파이프라인
DEFSEG_V3_PHASE=all docker compose up -d --build

# 5) Confusion matrix (stage1 train/val + stage2 val)
DEFSEG_V3_PHASE=confusion docker compose up -d --build

# 6) Inference (figure 생성, TTA + PP 적용)
DEFSEG_V3_PHASE=infer docker compose up -d --build

# 정리
docker compose down
```

## 환경변수

| 변수 | 기본 | 의미 |
|---|---|---|
| `DEFSEG_V3_PHASE` | `stage1` | `build_cache` / `stage1` / `stage2` / `all` / `confusion` / `infer` |
| `DEFSEG_V3_EXTRA` | (empty) | CLI 인자 추가 — `--use-hard-bootstrap`, `--quick`, `--max-batches 2` 등 |
| `NVIDIA_VISIBLE_DEVICES` | `0,1,2,3` | 사용할 GPU |
| `NPROC` | `4` | DDP world size (= GPU 개수와 일치 권장) |
| `OMP_NUM_THREADS` | `8` | per-process CPU threads |

## 스모크 검증 (학습 X)

```bash
# 1 GPU + quick
DEFSEG_V3_PHASE=stage1 NPROC=1 NVIDIA_VISIBLE_DEVICES=0 \
  DEFSEG_V3_EXTRA=--quick docker compose up -d --build

# Confusion matrix 앞 2 배치만
DEFSEG_V3_PHASE=confusion DEFSEG_V3_EXTRA="--max-batches 2" \
  docker compose up -d --build
```

## v1 image cache 재사용

v3 는 v1 의 `cache/resized_sz1036/<build>/visible_{0,1}.npy` 를 그대로 재사용함.
없으면 v1 의 `build_cache_stage1` 을 먼저 실행해야 한다. v3 가 만드는 건
`cache/resized_sz1036_v3/<build>/label_v3.npy` 와 `meta.npz` 뿐.
