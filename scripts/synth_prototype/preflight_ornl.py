"""ORNL 빌드의 주석 없는 사진 + CAD 부품 마스크에 SE 합성 규칙을 적용해 보는 사전 점검 (CPU, 학습 없음).

재료:  cache/resized_sz1036/<build>/visible_{0,1}.npy  (v1 캐시, 1036² uint8)  — 사진
       HDF5 slices/part_ids                             (CAD, 1842² uint32)     — 부품 자리 (> 0)
       DSCNN 예측 (slices/segmentation_results) 은 읽지 않는다.

보는 것:
  (1) CAD 부품 자리가 사진의 부품과 맞는가 — 실제 사진만으로 학습한 모델이 부품 계열로 예측한 화소와의 겹침
  (2) 원본 사진에서 모델이 CAD 부품 자리를 무엇으로 예측하는가 (Printed / SE / Swelling / Powder …)
  (3) CAD 부품 자리에 합성한 SE 를 두 모델이 무엇으로 예측하는가

실행 (CWD = 3DP_VPPM/, DefSeg_AM/venv):
    python -m DefSeg_AM.scripts.synth_prototype.preflight_ornl DefSeg_AM/cache/synth_prototype/ornl_raw 5 16 raw
    python -m DefSeg_AM.scripts.synth_prototype.preflight_ornl DefSeg_AM/cache/synth_prototype/ornl_stretched 5 16
    (인자: 출력 폴더, 빌드당 레이어 수, 스레드 수, 'raw' = HDF5 원본 밝기 / 생략 = v1 캐시의 늘린 밝기. GPU 를 쓰지 않고 각 약 12분)
"""
import json, sys, time
from pathlib import Path

import h5py
import numpy as np
import torch

from DefSeg_AM.scripts.synth_prototype.synth_rules import sim_super_elevation

from DefSeg_AM.common import config as ccfg
from DefSeg_AM.common.data.image_utils import resize_image_uint8, resize_label
from DefSeg_AM.v5 import config_v5 as cfg
from DefSeg_AM.v5.inference.evaluate_gt import load_model, predict_native

OUT = Path(sys.argv[1]); OUT.mkdir(parents=True, exist_ok=True)
N_PER_BUILD = int(sys.argv[2]) if len(sys.argv) > 2 else 5
torch.set_num_threads(int(sys.argv[3]) if len(sys.argv) > 3 else 16)
RAW = len(sys.argv) > 4 and sys.argv[4] == "raw"      # raw = HDF5 원본 밝기 그대로, 아니면 v1 캐시 (레이어마다 1–99 백분위로 늘린 것)
dev = torch.device("cpu")
IMG = cfg.IMG_SIZE
A = cfg.CLASS_ABBRS_V5
CACHE = Path("DefSeg_AM/cache") / f"resized_sz{IMG}"
PART_CLASSES = [1, 5, 7, 9, 10]            # Printed, Swelling, SE, OM, UM — 부품 위에서만 나오는 클래스

models = {n: load_model(cfg.run_ckpt_dir(cfg.default_run_name(), n) / "last.pt", dev)[0] for n in ("inlayer", "loso_L21")}


def predict(name, i0, i1):
    return predict_native(models[name], i0, i1, (IMG, IMG), IMG, dev, tta=False).numpy().astype(np.int8)


rng0 = np.random.default_rng(20261117)
records, arrays = [], {}
t0 = time.time()
for b in ("B2", "B3", "B4", "B5"):
    v0 = np.load(CACHE / b / "visible_0.npy", mmap_mode="r")
    v1 = np.load(CACHE / b / "visible_1.npy", mmap_mode="r")
    orig = np.load(CACHE / b / "meta.npz")["orig_layer_idxs"]
    rows = sorted(rng0.choice(len(orig), size=3 * N_PER_BUILD, replace=False).tolist())
    got = 0
    with h5py.File(ccfg.ORNL_HDF5_DIR / ccfg.ORNL_BUILD_FILES[b], "r") as f:
        pid_ds = f["slices/part_ids"]
        for r in rows:
            if got >= N_PER_BUILD:
                break
            layer = int(orig[r])
            foot_native = pid_ds[layer] > 0
            if foot_native.mean() < 0.01:
                continue
            foot = resize_label(foot_native.astype(np.int8), IMG).astype(bool)
            if RAW:
                i0 = resize_image_uint8(np.clip(f["slices/camera_data/visible/0"][layer], 0, 255).astype(np.uint8), IMG)
                i1 = resize_image_uint8(np.clip(f["slices/camera_data/visible/1"][layer], 0, 255).astype(np.uint8), IMG)
            else:
                i0, i1 = np.array(v0[r]), np.array(v1[r])
            ann = foot.astype(np.int8)                       # 규칙에는 부품 자리만 쓴다 (1 = 부품, 0 = 그 밖)
            key = f"{b}/L{layer:04d}"
            arrays[f"{key}|i0"] = i0; arrays[f"{key}|i1"] = i1; arrays[f"{key}|foot"] = foot
            base = {m: predict(m, i0, i1) for m in models}
            for m in models:
                arrays[f"{key}|pred_{m}"] = base[m]
                p = base[m]
                part_pred = np.isin(p, PART_CLASSES)
                records.append(dict(kind="orig", build=b, key=key, model=m, foot_px=int(foot.sum()),
                                    iou_foot_vs_partpred=float((foot & part_pred).sum() / max((foot | part_pred).sum(), 1)),
                                    hist_on_foot=np.bincount(p[foot].astype(np.int64), minlength=11)[:11].tolist(),
                                    hist_off_foot=np.bincount(p[~foot].astype(np.int64), minlength=11)[:11].tolist()))
            for k, zone in ((0, "blob"), (1, "band")):
                rng = np.random.default_rng([20261117, int(b[1]), layer, k])
                out = sim_super_elevation(i0, i1, ann, rng, zone=zone)
                if out is None:
                    continue
                i1n, annn, new = out
                tag = f"{key}|SE_{zone}"
                arrays[f"{tag}|i1"] = i1n; arrays[f"{tag}|new"] = new; arrays[f"{tag}|ign"] = annn == -1
                for m in models:
                    p = predict(m, i0, i1n)
                    arrays[f"{tag}|pred_{m}"] = p
                    records.append(dict(kind="sim", zone=zone, build=b, key=key, model=m, n=int(new.sum()),
                                        hist=np.bincount(p[new].astype(np.int64), minlength=11)[:11].tolist(),
                                        hist_orig=np.bincount(base[m][new].astype(np.int64), minlength=11)[:11].tolist(),
                                        # 합성하지 않은 부품 화소 가운데 예측이 SE 로 바뀐 것 (번짐)
                                        spill_se=int(((p == 7) & (base[m] != 7) & foot & ~new & (annn != -1)).sum()),
                                        untouched_foot=int((foot & ~new & (annn != -1)).sum())))
            got += 1
            print(f"{key}  foot {100 * foot.mean():.1f}%  {time.time() - t0:.0f}s", flush=True)

(OUT / "records.json").write_text(json.dumps(records))
np.savez_compressed(OUT / "arrays.npz", **arrays)


def pct(h, cs):
    h = np.asarray(h, dtype=np.float64)
    return 100.0 * h[list(cs)].sum() / max(h.sum(), 1)


print("\n== (1)(2) 원본 사진: CAD 부품 자리를 모델이 무엇으로 예측했나")
for m in models:
    g = [r for r in records if r["kind"] == "orig" and r["model"] == m]
    hon = np.sum([r["hist_on_foot"] for r in g], axis=0); hoff = np.sum([r["hist_off_foot"] for r in g], axis=0)
    print(f"{m:9s} layers {len(g)}  IoU(CAD 부품, 모델의 부품 계열 예측): mean {np.mean([r['iou_foot_vs_partpred'] for r in g]):.3f} "
          f"min {np.min([r['iou_foot_vs_partpred'] for r in g]):.3f}")
    print("   부품 자리 안:  " + "  ".join(f"{A[c]} {pct(hon, [c]):.1f}" for c in np.argsort(-hon)[:6]))
    print("   부품 자리 밖:  " + "  ".join(f"{A[c]} {pct(hoff, [c]):.1f}" for c in np.argsort(-hoff)[:6]))
    for b in ("B2", "B3", "B4", "B5"):
        gb = [r for r in g if r["build"] == b]
        hb = np.sum([r["hist_on_foot"] for r in gb], axis=0)
        print(f"     {b}: IoU {np.mean([r['iou_foot_vs_partpred'] for r in gb]):.3f}   부품 자리 안 → Prt {pct(hb, [1]):.1f}  SE {pct(hb, [7]):.1f}  Sw {pct(hb, [5]):.1f}  Pow {pct(hb, [0]):.1f}")

print("\n== (3) 합성 SE 를 무엇으로 예측했나")
for m in models:
    for zone in ("blob", "band"):
        g = [r for r in records if r["kind"] == "sim" and r["model"] == m and r["zone"] == zone]
        if not g:
            continue
        h = np.sum([r["hist"] for r in g], axis=0); ho = np.sum([r["hist_orig"] for r in g], axis=0)
        spill = sum(r["spill_se"] for r in g) / max(sum(r["untouched_foot"] for r in g), 1)
        print(f"{m:9s} {zone:5s} n={len(g):2d} px={int(h.sum()):8d}  →SE {pct(h, [7]):5.1f}  →Sw {pct(h, [5]):5.1f}  →Prt {pct(h, [1]):5.1f}  →Pow {pct(h, [0]):5.1f}  "
              f"→IS {pct(h, [4]):5.1f}  →기타 {pct(h, [2, 3, 6, 8, 9, 10]):5.1f}   | 원본의 같은 화소 →SE {pct(ho, [7]):.2f}  | 손대지 않은 부품 화소가 SE 로 바뀐 비율 {100 * spill:.2f}%")
        for b in ("B2", "B3", "B4", "B5"):
            gb = [r for r in g if r["build"] == b]
            if gb:
                hb = np.sum([r["hist"] for r in gb], axis=0)
                print(f"     {b}: n={len(gb)} →SE {pct(hb, [7]):5.1f}")
print(f"\ntotal {time.time() - t0:.0f}s  saved → {OUT}")
