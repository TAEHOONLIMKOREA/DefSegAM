#!/usr/bin/env bash
# Stage 1: ORNL DSCNN pred 으로 KD pretrain (PLAN.md §5.1) — 1 GPU 단일 프로세스
# NaN-fix v2: AMP off + grad clip 1.0 + warmup 200 + lr 1e-4 + class weight clip 10
# 사전조건: run_build_cache.sh 가 1회 완료되어 cache/resized_sz1036/ 가 준비됨.
set -euo pipefail
cd "$(dirname "$0")/../.."

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_VISIBLE_DEVICES=0          # 첫 번째 GPU (1 GPU 모드)
export OMP_NUM_THREADS=8

nohup ./DefSeg_AM/venv/bin/python -u -m DefSeg_AM.training.train_stage1 \
    --epochs 30 \
    --batch-size 2 \
    --img-size 1036 \
    --backbone dinov2_vits14 \
    --num-workers 8 \
    --gamma 2.0 \
    --oversample-power 0.5 \
    --val-log-every 100 \
    --run-name vits14_dpt_dual_sz1036_1gpu_nanfix \
    > DefSeg_AM/logs/stage1.log 2>&1 &
echo "PID=$!"
echo "tail -f DefSeg_AM/logs/stage1.log"
tail -f DefSeg_AM/logs/stage1.log
