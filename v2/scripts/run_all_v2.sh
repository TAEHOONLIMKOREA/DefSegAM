#!/usr/bin/env bash
# v2 전체 파이프라인 — build_cache_v2 → stage1 → stage2 (8 fold)
set -euo pipefail
HERE="$(dirname "$0")"

echo "===== [1/3] build_cache_v2 ====="
bash "${HERE}/run_build_cache_v2.sh"

echo "===== [2/3] stage1 (foreground, ~12h) ====="
# foreground 실행 — stage2 가 stage1_best.pt 를 필요로 함
cd "${HERE}/../../.."
export PYTHONUNBUFFERED=1
export HDF5_USE_FILE_LOCKING=FALSE
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"
./DefSeg_AM/venv/bin/python -u -m DefSeg_AM.v2.training.train_stage1 \
    --epochs 30 --batch-size 2 --img-size 1036 \
    --backbone dinov2_vits14 --num-workers 8 \
    --gamma 2.0 --oversample-power 0.5 --val-log-every 100 \
    --run-name vits14_dpt_dual_sz1036_8cls_v2 \
    ${V2_STAGE1_EXTRA:-} \
    > DefSeg_AM/v2/stage1_v2.log 2>&1
echo "stage1 done"

echo "===== [3/3] stage2 (8 fold) ====="
bash "${HERE}/run_stage2_v2_all.sh"
echo "===== ALL DONE ====="
