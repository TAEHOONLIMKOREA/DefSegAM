# syntax=docker/dockerfile:1.6
#
# DefSeg-AM GenAug — 결함 인페인팅 확산 모델 (PLAN_GENAUG.md).
# 호스트의 DefSeg_AM/venv 를 bind mount 하여 사용 (이미지에 pip install 안 함). 새 패키지는 쓰지 않는다 (torch 만).
#
FROM ubuntu:22.04

ENV DEBIAN_FRONTEND=noninteractive \
    PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    HDF5_USE_FILE_LOCKING=FALSE \
    OMP_NUM_THREADS=4

# 최소 system dependency — python3 (venv 의 interpreter 호출용)
RUN apt-get update && apt-get install -y --no-install-recommends \
        python3 ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# 소스 코드와 host venv 는 bind mount 로 들어온다. 이미지에는 COPY 안 함.

CMD ["/workspace/DefSeg_AM/venv/bin/python", "-c", "import torch; print(torch.__version__)"]
