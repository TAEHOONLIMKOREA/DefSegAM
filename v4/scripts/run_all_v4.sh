#!/usr/bin/env bash
# v4 전체 파이프라인 — build_cache_v4 → stage1 → stage2 (단일 ckpt)
# 4-GPU DDP. Stage 1 ≈ 3-4h, Stage 2 ≈ 1-2h (4-GPU 기준).
set -euo pipefail
HERE="$(dirname "$0")"

echo "===== [1/3] build_cache_v4 ====="
bash "${HERE}/run_build_cache_v4.sh"

echo "===== [2/3] stage1 (foreground, DDP 4-GPU) ====="
cd "${HERE}/../../.."
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0,1,2,3}"
NPROC="${NPROC:-4}"

./DefSeg_AM/venv/bin/python -u -m torch.distributed.run \
    --nproc_per_node=${NPROC} --standalone \
    -m DefSeg_AM.v4.training.train_stage1 \
    --epochs 30 --batch-size 4 --img-size 1036 \
    --backbone dinov2_vitb14_reg --num-workers 8 \
   --class-aware-alpha 1.0 --val-log-every 100 \
    --run-name vitb14_reg_dpt_dual_sz1036_11cls_v4 \
    ${V4_STAGE1_EXTRA:-} \
    > DefSeg_AM/v4/stage1_v4.log 2>&1
echo "stage1 done"

echo "===== [3/3] stage2 (foreground, DDP 4-GPU) ====="
./DefSeg_AM/venv/bin/python -u -m torch.distributed.run \
    --nproc_per_node=${NPROC} --standalone \
    -m DefSeg_AM.v4.training.train_stage2 \
    --epochs 50 --batch-size 4 --img-size 1036 \
    --backbone dinov2_vitb14_reg --num-workers 4 \
    --run-name vitb14_reg_dpt_dual_sz1036_11cls_v4 \
    ${V4_STAGE2_EXTRA:-} \
    > DefSeg_AM/v4/stage2_v4.log 2>&1
echo "===== ALL DONE ====="
