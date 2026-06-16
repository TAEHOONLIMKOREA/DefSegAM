#!/usr/bin/env bash
# v4 Stage 2 — DSCNN_Dataset 전수 학습 (v4) — 4-GPU DDP
# 사전조건: checkpoints/<run>/stage1_best.pt 존재 (Stage 1 완료)
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
NPROC="${NPROC:-4}"

EXTRA="${V4_STAGE2_EXTRA:-}"
RUN_NAME="vitb14_reg_dpt_dual_sz1036_11cls_v4"
LOG="DefSeg_AM/v4/stage2_v4.log"

nohup ./DefSeg_AM/venv/bin/python -u -m torch.distributed.run \
    --nproc_per_node=${NPROC} --standalone \
    -m DefSeg_AM.v4.training.train_stage2 \
    --epochs 50 \
    --batch-size 4 \
    --img-size 1036 \
    --backbone dinov2_vitb14_reg \
    --num-workers 4 \
    --run-name "${RUN_NAME}" \
    ${EXTRA} \
    > "${LOG}" 2>&1 &
echo "PID=$!  (4-GPU DDP)"
echo "tail -f ${LOG}"
tail -f "${LOG}"
