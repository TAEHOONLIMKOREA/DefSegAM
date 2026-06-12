#!/usr/bin/env bash
# 학습된 ckpt 의 validation confusion matrix 평가 (학습 X, 1-pass).
# 결함 class 혼동/병합 후보 분석용. 사전조건: checkpoints/<run_name>/stage<N>_best.pt 존재.
#   STAGE=1 (기본): ORNL cached val (Build 1)  — cache/resized_sz1036/val_index.npz 필요
#   STAGE=2       : DSCNN_Dataset val (Maraging)
set -euo pipefail
cd "$(dirname "$0")/../.."

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"

RUN_NAME="${RUN_NAME:-vits14_dpt_dual_sz1036_1gpu_nanfix}"
STAGE="${STAGE:-1}"

./DefSeg_AM/venv/bin/python -u -m DefSeg_AM.inference.confusion \
    --run-name "$RUN_NAME" \
    --stage "$STAGE" \
    --batch-size 2 \
    --num-workers 4
