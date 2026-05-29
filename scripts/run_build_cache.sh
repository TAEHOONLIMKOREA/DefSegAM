#!/usr/bin/env bash
# 1회 실행: ORNL raw HDF5 → 사전 resize+uint8+argmax cache (.npy memmap)
# 학습 진입 전에 반드시 1번 돌릴 것. 예상 시간 30~60분, 디스크 ~45GB.
set -euo pipefail
cd "$(dirname "$0")/../.."

PYTHONUNBUFFERED=1 nohup ./DefSeg_AM/venv/bin/python -u -m DefSeg_AM.data.build_cache_stage1 \
    --img-size 1036 \
    --chunk-size 50 \
    > DefSeg_AM/logs/build_cache.log 2>&1 &
echo "PID=$!"
echo "tail -f DefSeg_AM/logs/build_cache.log"
tail -f DefSeg_AM/logs/build_cache.log
