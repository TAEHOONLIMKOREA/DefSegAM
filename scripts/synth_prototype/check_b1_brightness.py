"""v5 전수 학습 모델을 ORNL Build 1 에 넣을 때 밝기 처리에 따라 결과가 얼마나 달라지는가 (참고 평가용, 학습 없음).

같은 레이어를 (가) 늘린 밝기 — 레이어마다 1–99 백분위를 0–255 로 (v1 캐시 · v5 infer.py 가 쓰는 방식) 와
(나) 원본 밝기 — HDF5 값 그대로 두 가지로 넣어, Printed 예측을 CAD 부품 자리와 DSCNN 의 Printed 예측에 견준다.
DSCNN 예측은 v5 의 confusion phase 와 같은 뜻의 참고 비교로만 읽는다.

실행 (CWD = 3DP_VPPM/, DefSeg_AM/venv, GPU 1대):
    python -m DefSeg_AM.scripts.synth_prototype.check_b1_brightness [레이어 수=12]
"""
import sys
import h5py
import numpy as np
import torch

from DefSeg_AM.common import config as ccfg
from DefSeg_AM.common.data.image_utils import ornl_image_to_uint8, resize_label
from DefSeg_AM.v5 import config_v5 as cfg
from DefSeg_AM.v5.data.data_ornl_v5 import ornl_segmentation_argmax_v5
from DefSeg_AM.v5.inference.evaluate_gt import load_model, predict_native

N = int(sys.argv[1]) if len(sys.argv) > 1 else 12
dev = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
IMG = cfg.IMG_SIZE
model, _ = load_model(cfg.run_ckpt_dir(cfg.default_run_name(), None) / "last.pt", dev)
PART = (1, 5, 7, 9, 10)
acc = {k: dict(i_cad=0, u_cad=0, off=0, off_prt=0, i_ds=0, u_ds=0, tp=0, pred=0, gt=0) for k in ("stretched", "raw")}
with h5py.File(ccfg.ORNL_HDF5_DIR / ccfg.ORNL_BUILD_FILES["B1"], "r") as f:
    v0, v1, seg, pid = f["slices/camera_data/visible/0"], f["slices/camera_data/visible/1"], f["slices/segmentation_results"], f["slices/part_ids"]
    n = v0.shape[0]
    layers = np.linspace(int(n * cfg.ORNL_LAYER_LO_FRAC), int(n * cfg.ORNL_LAYER_HI_FRAC), N).astype(int)
    for li in layers:
        a0, a1 = v0[li], v1[li]
        foot = pid[li] > 0
        ds = ornl_segmentation_argmax_v5(seg, int(li))            # DSCNN 예측 (참고)
        for kind in ("stretched", "raw"):
            if kind == "stretched":
                i0, i1 = ornl_image_to_uint8(a0), ornl_image_to_uint8(a1)
            else:
                i0, i1 = np.clip(a0, 0, 255).astype(np.uint8), np.clip(a1, 0, 255).astype(np.uint8)
            p = predict_native(model, i0, i1, foot.shape, IMG, dev, tta=False).cpu().numpy()
            part = np.isin(p, PART); prt = p == 1
            r = acc[kind]
            r["i_cad"] += int((part & foot).sum()); r["u_cad"] += int((part | foot).sum())
            r["off"] += int((~foot).sum()); r["off_prt"] += int((prt & ~foot).sum())
            g = ds == 1; valid = ds >= 0
            r["tp"] += int((prt & g & valid).sum()); r["pred"] += int((prt & valid).sum()); r["gt"] += int(g.sum())
print(f"Build 1, {N} 레이어 {list(map(int, layers))}, 모델 = full/last.pt, TTA · 후처리 없음, native 채점")
for kind, r in acc.items():
    iou_ds = r["tp"] / max(r["pred"] + r["gt"] - r["tp"], 1)
    print(f"  {kind:9s}  CAD 부품 자리와 부품 계열 예측의 IoU {r['i_cad'] / max(r['u_cad'], 1):.3f}   부품 밖을 Printed 로 {100 * r['off_prt'] / r['off']:.2f}%   "
          f"DSCNN 의 Printed 와의 IoU {iou_ds:.3f} (정밀도 {r['tp'] / max(r['pred'], 1):.3f}, 재현율 {r['tp'] / max(r['gt'], 1):.3f})")
