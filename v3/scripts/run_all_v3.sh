#!/usr/bin/env bash
# v3 전체 파이프라인 — build_cache_v3 → stage1 → stage2 (단일 ckpt)
# 4-GPU DDP. Stage 1 ≈ 3-4h, Stage 2 ≈ 1-2h (4-GPU 기준).
set -euo pipefail
HERE="$(dirname "$0")"

echo "===== [1/3] build_cache_v3 ====="
bash "${HERE}/run_build_cache_v3.sh"

echo "===== [2/3] stage1 (foreground, DDP 4-GPU) ====="
cd "${HERE}/../../.."
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC="${NPROC:-4}"

./DefSeg_AM/venv/bin/python -u -m torch.distributed.run \
    --nproc_per_node=${NPROC} --standalone \
    -m DefSeg_AM.v3.training.train_stage1 \
    --epochs 30 --batch-size 2 --img-size 1036 \
    --backbone dinov2_vits14 --num-workers 8 \
    --gamma 2.0 --oversample-power 0.5 --val-log-every 100 \
    --run-name vits14_dpt_dual_sz1036_10cls_v3 \
    ${V3_STAGE1_EXTRA:-} \
    > DefSeg_AM/v3/stage1_v3.log 2>&1
echo "stage1 done"

echo "===== [3/3] stage2 (foreground, DDP 4-GPU) ====="
./DefSeg_AM/venv/bin/python -u -m torch.distributed.run \
    --nproc_per_node=${NPROC} --standalone \
    -m DefSeg_AM.v3.training.train_stage2 \
    --epochs 50 --batch-size 2 --img-size 1036 \
    --backbone dinov2_vits14 --num-workers 4 \
    --run-name vits14_dpt_dual_sz1036_10cls_v3 \
    ${V3_STAGE2_EXTRA:-} \
    > DefSeg_AM/v3/stage2_v3.log 2>&1
echo "===== ALL DONE ====="
