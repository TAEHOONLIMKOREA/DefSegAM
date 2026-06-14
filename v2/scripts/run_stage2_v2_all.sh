#!/usr/bin/env bash
# v2 Stage 2 — 8 fold 순차 학습 (cross-val 전체)
# 각 fold 가 끝나야 다음 fold 진행 (1 GPU 라 병렬 안 함).
set -euo pipefail
cd "$(dirname "$0")/../../.."

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

EXTRA="${V2_STAGE2_EXTRA:-}"
RUN_NAME="vits14_dpt_dual_sz1036_8cls_v2"

for FOLD in 0 1 2 3 4 5 6 7; do
    LOG="DefSeg_AM/v2/stage2_v2_fold${FOLD}.log"
    echo "===== fold ${FOLD} 시작 → ${LOG} ====="
    ./DefSeg_AM/venv/bin/python -u -m DefSeg_AM.v2.training.train_stage2 \
        --fold "${FOLD}" \
        --epochs 50 \
        --batch-size 2 \
        --img-size 1036 \
        --backbone dinov2_vits14 \
        --num-workers 4 \
        --val-log-every 10 \
        --run-name "${RUN_NAME}" \
        ${EXTRA} \
        > "${LOG}" 2>&1
    echo "===== fold ${FOLD} 완료 ====="
done
echo "ALL 8 folds done → DefSeg_AM/checkpoints/${RUN_NAME}/cv_summary.json"
