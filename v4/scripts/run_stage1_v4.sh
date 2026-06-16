#!/usr/bin/env bash
# v4 Stage 1: ORNL DSCNN pred 로 KD pretrain (11-class) — 4-GPU DDP
# 사전조건: build_cache_v4 가 1회 완료되어 cache/resized_sz1036_v4/ 가 준비됨.
set -euo pipefail
cd "$(dirname "$0")/../../.."   # repo root

export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
export OMP_NUM_THREADS="${OMP_NUM_THREADS:-8}"
NPROC="${NPROC:-4}"

EXTRA="${V4_STAGE1_EXTRA:-}"   # e.g. "--use-hard-bootstrap" or "--quick"

nohup ./DefSeg_AM/venv/bin/python -u -m torch.distributed.run \
    --nproc_per_node=${NPROC} --standalone \
    -m DefSeg_AM.v4.training.train_stage1 \
    --epochs 30 \
    --batch-size 4 \
    --img-size 1036 \
    --backbone dinov2_vitb14_reg \
    --num-workers 8 \
   \
    --class-aware-alpha 1.0 \
    --val-log-every 100 \
    --run-name vitb14_reg_dpt_dual_sz1036_11cls_v4 \
    ${EXTRA} \
    > DefSeg_AM/v4/stage1_v4.log 2>&1 &
echo "PID=$!  (4-GPU DDP)"
echo "tail -f DefSeg_AM/v4/stage1_v4.log"
tail -f DefSeg_AM/v4/stage1_v4.log
