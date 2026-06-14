#!/usr/bin/env bash
# v2 Stage 2 — 단일 fold finetune (cross-val)
# Usage:  V2_FOLD=0 bash run_stage2_v2_fold.sh
# 사전조건: checkpoints/<run>/stage1_best.pt 존재 (Stage 1 완료)
set -euo pipefail
cd "$(dirname "$0")/../../.."

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"

FOLD="${V2_FOLD:-0}"
EXTRA="${V2_STAGE2_EXTRA:-}"
RUN_NAME="vits14_dpt_dual_sz1036_8cls_v2"
LOG="DefSeg_AM/v2/stage2_v2_fold${FOLD}.log"

nohup ./DefSeg_AM/venv/bin/python -u -m DefSeg_AM.v2.training.train_stage2 \
    --fold "${FOLD}" \
    --epochs 50 \
    --batch-size 2 \
    --img-size 1036 \
    --backbone dinov2_vits14 \
    --num-workers 4 \
    --val-log-every 10 \
    --run-name "${RUN_NAME}" \
    ${EXTRA} \
    > "${LOG}" 2>&1 &
echo "PID=$!  fold=${FOLD}"
echo "tail -f ${LOG}"
tail -f "${LOG}"
