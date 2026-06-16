#!/usr/bin/env bash
# v3 Stage 1: ORNL DSCNN pred 로 KD pretrain (10-class) — 4-GPU DDP
# 사전조건: build_cache_v3 가 1회 완료되어 cache/resized_sz1036_v3/ 가 준비됨.
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
NPROC="${NPROC:-4}"

EXTRA="${V3_STAGE1_EXTRA:-}"   # e.g. "--use-hard-bootstrap" or "--quick"

nohup ./DefSeg_AM/venv/bin/python -u -m torch.distributed.run \
    --nproc_per_node=${NPROC} --standalone \
    -m DefSeg_AM.v3.training.train_stage1 \
    --epochs 30 \
    --batch-size 2 \
    --img-size 1036 \
    --backbone dinov2_vits14 \
    --num-workers 8 \
    --gamma 2.0 \
    --oversample-power 0.5 \
    --val-log-every 100 \
    --run-name vits14_dpt_dual_sz1036_10cls_v3 \
    ${EXTRA} \
    > DefSeg_AM/v3/stage1_v3.log 2>&1 &
echo "PID=$!  (4-GPU DDP)"
echo "tail -f DefSeg_AM/v3/stage1_v3.log"
tail -f DefSeg_AM/v3/stage1_v3.log
