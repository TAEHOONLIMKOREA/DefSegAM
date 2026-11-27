"""합성 결함이 '진짜처럼' 보이는가 — 실제 사진만으로 학습한 두 모델에게 물어본다 (CPU, 학습 없음).

모델:  inlayer/last.pt  (L21 을 포함한 8 소스의 레이어로 학습. 아래 검증 22 레이어는 본 적 없다)
       loso_L21/last.pt (L21 을 한 번도 보지 못했다)

시험 A — 두 모델이 모두 본 적 없는 L21 4 레이어:  원본 / SE 합성 4종 / Swelling 합성 2종
시험 B — inlayer 모델이 본 적 없는 나머지 18 레이어: 원본 / SE 합성 1종 / Swelling 합성 1종

적는 것: 합성으로 새로 생긴 결함 화소를 모델이 무엇으로 예측했는가, 같은 화소를 원본 사진에서는 무엇으로 예측했는가 (대조),
         그리고 같은 레이어의 실제 SE · Swelling 화소를 무엇으로 예측했는가.

실행 (CWD = 3DP_VPPM/, DefSeg_AM/venv):
    python -m DefSeg_AM.scripts.synth_prototype.eval_on_gt DefSeg_AM/cache/synth_prototype/eval_gt [threads=16]
    (GPU 를 쓰지 않는다. 16 스레드로 약 9분. 결과: <out>/records.json, <out>/arrays.npz)
"""
import json, sys, time
from pathlib import Path

import numpy as np
import torch

from DefSeg_AM.scripts.synth_prototype.synth_rules import sim_super_elevation, sim_swelling, SE, SWELL

from DefSeg_AM.v5 import config_v5 as cfg
from DefSeg_AM.v5.data.data_gt_v5 import GTPool, enumerate_samples_v5
from DefSeg_AM.v5.inference.evaluate_gt import load_model, predict_native

OUT = Path(sys.argv[1])
torch.set_num_threads(int(sys.argv[2]) if len(sys.argv) > 2 else 16)
dev = torch.device("cpu")
IMG = cfg.IMG_SIZE
A = cfg.CLASS_ABBRS_V5
run_dir = Path("DefSeg_AM/figures") / cfg.default_run_name() / "v5" / "cv"
val_keys = json.loads((run_dir / "inlayer" / "run.json").read_text())["val_samples"]

samples = enumerate_samples_v5()
idx = {s.key: i for i, s in enumerate(samples)}
val = [idx[k] for k in val_keys]
sub = [samples[i] for i in val]
t = time.time(); pool = GTPool(sub, IMG); print(f"pool {len(sub)} layers {time.time() - t:.0f}s", flush=True)

models = {}
for name in ("inlayer", "loso_L21"):
    models[name], _ = load_model(cfg.run_ckpt_dir(cfg.default_run_name(), name) / "last.pt", dev)


def predict(name, i0, i1):
    return predict_native(models[name], i0, i1, (IMG, IMG), IMG, dev, tta=False).numpy().astype(np.int8)


records = []          # dict per (layer, variant)
arrays = {}           # for the figure
t0 = time.time()
for j, s in enumerate(sub):
    src = cfg.SOURCE_SHORT[s.source_name]
    i0, i1, ann = pool.i0[j], pool.i1[j], pool.ann[j]
    is_l21 = s.source_name == "v2021_LPBF"
    use = ["inlayer", "loso_L21"] if is_l21 else ["inlayer"]
    base = {m: predict(m, i0, i1) for m in use}
    arrays[f"{s.key}|orig|i0"] = i0; arrays[f"{s.key}|orig|i1"] = i1; arrays[f"{s.key}|orig|ann"] = ann
    for m in use:
        arrays[f"{s.key}|orig|pred_{m}"] = base[m]
    # 실제 SE · Swelling
    for c in (SE, SWELL):
        g = ann == c
        if g.sum() > 0:
            for m in use:
                records.append(dict(kind="real", cls=A[c], src=src, key=s.key, model=m, n=int(g.sum()),
                                    hist=np.bincount(base[m][g].astype(np.int64), minlength=11)[:11].tolist()))
    variants = ([("SE", "blob", 0), ("SE", "blob", 1), ("SE", "band", 2), ("SE", "band", 3), ("Sw", None, 4), ("Sw", None, 5)]
                if is_l21 else [("SE", "blob", 0), ("Sw", None, 4)])
    for cls, zone, k in variants:
        rng = np.random.default_rng([20261117, val[j], k])
        out = sim_super_elevation(i0, i1, ann, rng, zone=zone) if cls == "SE" else sim_swelling(i0, i1, ann, rng)
        if out is None:
            continue
        i1n, annn, new = out
        tag = f"{s.key}|{cls}{k}"
        arrays[f"{tag}|i1"] = i1n; arrays[f"{tag}|ann"] = annn; arrays[f"{tag}|new"] = new
        for m in use:
            p = predict(m, i0, i1n)
            arrays[f"{tag}|pred_{m}"] = p
            records.append(dict(kind="sim", cls=cls, zone=zone or "", src=src, key=s.key, model=m, n=int(new.sum()),
                                hist=np.bincount(p[new].astype(np.int64), minlength=11)[:11].tolist(),
                                hist_orig=np.bincount(base[m][new].astype(np.int64), minlength=11)[:11].tolist(),
                                # 합성으로 건드리지 않은 곳에서 예측이 바뀐 화소 (부작용)
                                changed_elsewhere=int(((p != base[m]) & (annn == ann)).sum())))
    print(f"[{j + 1}/{len(sub)}] {s.key}  {time.time() - t0:.0f}s", flush=True)

OUT.mkdir(parents=True, exist_ok=True)
(OUT / "records.json").write_text(json.dumps(records, ensure_ascii=False))
np.savez_compressed(OUT / "arrays.npz", **arrays)


def frac(h, cs):
    h = np.asarray(h, dtype=np.float64)
    return float(h[list(cs)].sum() / max(h.sum(), 1))


def table(rows, title):
    print(f"\n== {title}")
    print(f"{'model':9s} {'src':5s} {'what':14s} {'layers':>6s} {'px':>9s}  {'→SE':>6s} {'→Sw':>6s} {'→Prt':>6s} {'→Pow':>6s} {'→other':>6s}   (원본 사진의 같은 화소: →SE/Sw)")
    keyf = lambda r: (r["model"], r["src"], r["kind"], r["cls"], r.get("zone", ""))
    for key in sorted({keyf(r) for r in rows}):
        g = [r for r in rows if keyf(r) == key]
        h = np.sum([r["hist"] for r in g], axis=0)
        line = (f"{key[0]:9s} {key[1]:5s} {(key[2] + ' ' + key[3] + ' ' + key[4]).strip():14s} {len(g):6d} {int(h.sum()):9d}  "
                f"{frac(h, [7]):6.3f} {frac(h, [5]):6.3f} {frac(h, [1]):6.3f} {frac(h, [0]):6.3f} {frac(h, [2, 3, 4, 6, 8, 9, 10]):6.3f}")
        if key[2] == "sim":
            ho = np.sum([r["hist_orig"] for r in g], axis=0)
            line += f"   {frac(ho, [7, 5]):.3f}"
        print(line)


table([r for r in records if r["src"] == "L21"], "시험 A — L21 4 레이어 (두 모델 모두 본 적 없는 레이어)")
table([r for r in records if r["src"] != "L21"], "시험 B — 나머지 소스 (inlayer 모델이 본 적 없는 레이어)")
print(f"\ntotal {time.time() - t0:.0f}s  saved → {OUT}")
