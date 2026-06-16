#!/usr/bin/env bash
# v3 — ORNL HDF5 label cache 를 10-class 로 재빌드 (image 채널은 v1 cache 재사용)
# 사전조건: v1 image cache (cache/resized_sz1036/<build>/visible_{0,1}.npy) 존재
# 예상 시간: ~5 분 (label 만 LUT 매핑)
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE

nohup ./DefSeg_AM/venv/bin/python -u -m DefSeg_AM.v3.data.build_cache_v3 \
    --img-size 1036 \
    > DefSeg_AM/v3/build_cache_v3.log 2>&1 &
echo "PID=$!"
echo "tail -f DefSeg_AM/v3/build_cache_v3.log"
tail -f DefSeg_AM/v3/build_cache_v3.log
