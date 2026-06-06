#!/usr/bin/env bash
# Stage 2: DSCNN_Dataset human GT 로 finetune (PLAN.md §5.2) — 1 GPU 단일 프로세스
# NaN-fix v2 와 동일 정책. 사전조건: checkpoints/<run_name>/stage1_best.pt 존재.
set -euo pipefail
cd "$(dirname "$0")/../.."

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_VISIBLE_DEVICES=0
export OMP_NUM_THREADS=8

nohup ./DefSeg_AM/venv/bin/python -u -m DefSeg_AM.training.train_stage2 \
    --epochs 50 \
    --batch-size 2 \
    --img-size 1036 \
    --backbone dinov2_vits14 \
    --num-workers 4 \
    --val-log-every 10 \
    --run-name vits14_dpt_dual_sz1036_1gpu_nanfix \
    > DefSeg_AM/logs/stage2.log 2>&1 &
echo "PID=$!"
echo "tail -f DefSeg_AM/logs/stage2.log"
tail -f DefSeg_AM/logs/stage2.log
