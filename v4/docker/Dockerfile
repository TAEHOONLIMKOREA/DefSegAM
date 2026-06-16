# syntax=docker/dockerfile:1.6
#
# DefSeg-AM v4 — 11-class + LR/UD flip + rot180 + cyclic + 전수 학습 + 4-GPU DDP.
# 호스트의 DefSeg_AM/venv 를 bind mount 하여 사용 (이미지에 pip install 안 함).
#
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HDF5_USE_FILE_LOCKING=FALSE \
    OMP_NUM_THREADS=8

# 최소 system dependency — python3 (venv 의 interpreter 호출용)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 소스 코드는 bind mount 로 들어옴 (DefSeg_AM/ 전체). 이미지에는 COPY 안 함.
# host venv 도 bind mount 로 들어옴 (/workspace/DefSeg_AM/venv).

CMD ["/workspace/DefSeg_AM/venv/bin/python", "-c", "import torch; print(torch.__version__)"]
