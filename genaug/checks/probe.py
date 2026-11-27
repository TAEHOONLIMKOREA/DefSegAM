"""K5 — 합성인 것이 분할 모델의 눈에 보이는가 (v7/PLAN_v7.md §2.6). 기록만 한다 (통과 · 탈락이 없다).

분할 모델과 같은 backbone (frozen DINOv2) 의 특징으로 **"진짜 정상 화소" 와 "합성 정상 화소" 를 가르는 선형 분류기**를 배우고,
보지 않은 레이어에서 AUC 를 잰다 (0.5 = 가르지 못한다, 1.0 = 완전히 가른다).

  - 특징: 사진 두 장을 backbone 에 넣어 얻은 14 px patch 의 마지막 단 특징을 이어 붙인 것 (분할 모델의 decoder 가 받는 것과 같은 종류)
  - 표본: 합성 레이어와 **같은 ORNL 레이어의 진짜 사진**에서 같은 자리의 patch. 합성 쪽은 Powder (또는 Printed) 만 그리라고 한 patch,
          진짜 쪽은 CAD 로 본 같은 자리 (진짜 사진에는 주석 없는 진짜 결함이 섞여 있다)
  - 나눔: 레이어 단위로 7 : 3 (같은 레이어의 patch 가 학습과 시험에 함께 들어가지 않는다)
  - 견줄 값: 진짜 사진끼리 빌드로 가른 AUC (같은 장비의 두 빌드가 이 특징으로 얼마나 갈리는가)

AUC 가 1 에 가까우면 분할 모델이 "합성 도메인" 을 따로 배울 수 있다는 뜻이다 — 합성에서 배운 것이 진짜로 덜 옮겨 갈 수 있다 (PLAN_v7 §8).
다만 AUC 는 쉽게 1 에 붙는다 (같은 장비의 두 빌드도 갈린다). 그래서 **얼마나 먼가**를 함께 잰다 — 두 patch 묶음의 특징 분포 사이의 Fréchet 거리
(특징을 표준화하고 주성분 64개로 줄인 뒤, 평균과 공분산으로 계산). 견주는 것은 셋이다:
  진짜 ↔ 합성 (같은 빌드, 같은 자리) / 진짜 빌드 ↔ 다른 진짜 빌드 / 같은 빌드의 진짜 레이어를 둘로 나눈 것끼리 (바닥값)

실행 (CWD = 3DP_VPPM/, DefSeg_AM/venv — compose 의 check phase. GPU 1대):
    python -m DefSeg_AM.genaug.checks.probe --set w600_v1
산출물: figures/genaug/sets/<세트>/probe.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np
import torch

from ...common.data.image_utils import normalize_image
from ...common.models.model import DefSegModel
from ...common.utils.log import setup_logger
from .. import config_genaug as cfg
from ..data.canvas_pool import CanvasPool
from ..sampling.preview import cad_label
from ..synth.layer import frame_px
from ..synth.set_io import pick_layers, read_layer, read_manifest, resolve_set


@torch.no_grad()
def patch_features(model: DefSegModel, i0: np.ndarray, i1: np.ndarray, device) -> torch.Tensor:
    """사진 두 장 → (Hp, Wp, 2D) — patch 마다 두 사진의 마지막 단 특징을 이어 붙인 것."""
    fs = []
    for im in (i0, i1):
        t = torch.from_numpy(normalize_image(im)).unsqueeze(0).to(device)
        fs.append(model._multi_scale_features(t)[-1][0])               # (D, Hp, Wp)
    return torch.cat(fs, dim=0).permute(1, 2, 0).float()


def pure_patches(mask: np.ndarray, p: int) -> np.ndarray:
    """(S, S) bool → (Hp, Wp) bool — patch 의 모든 화소가 True 인가."""
    n = mask.shape[0] // p
    return mask[:n * p, :n * p].reshape(n, p, n, p).all(axis=(1, 3))


def auc(score: np.ndarray, y: np.ndarray) -> float:
    order = np.argsort(score, kind="mergesort")
    rank = np.empty(len(score), dtype=np.float64)
    rank[order] = np.arange(1, len(score) + 1)
    n1, n0 = int(y.sum()), int((1 - y).sum())
    if n1 == 0 or n0 == 0:
        return float("nan")
    return float((rank[y == 1].sum() - n1 * (n1 + 1) / 2) / (n1 * n0))


def linear_probe(xtr, ytr, xte, yte, device, steps: int = 400) -> dict:
    """표준화한 특징 위의 로지스틱 회귀 (전체 배치, Adam). → 시험 AUC · 정확도."""
    xtr, xte = xtr.to(device), xte.to(device)
    mu, sd = xtr.mean(0, keepdim=True), xtr.std(0, keepdim=True) + 1e-6
    xtr, xte = (xtr - mu) / sd, (xte - mu) / sd
    yt = torch.from_numpy(ytr.astype(np.float32)).to(device)
    w = torch.zeros(xtr.shape[1], device=device, requires_grad=True)
    b = torch.zeros(1, device=device, requires_grad=True)
    opt = torch.optim.Adam([w, b], lr=1e-2, weight_decay=1e-3)
    for _ in range(steps):
        opt.zero_grad()
        loss = torch.nn.functional.binary_cross_entropy_with_logits(xtr @ w + b, yt)
        loss.backward()
        opt.step()
    with torch.no_grad():
        s = (xte @ w + b).cpu().numpy()
    return {"auc": auc(s, yte), "acc": float(((s > 0) == (yte > 0.5)).mean()), "n_train": int(len(ytr)), "n_test": int(len(yte))}


def frechet(a: torch.Tensor, b: torch.Tensor) -> float:
    """두 묶음 (n, d) 의 Fréchet 거리² = |μa − μb|² + tr(Σa + Σb − 2 (Σa Σb)^½)."""
    a, b = a.double(), b.double()
    ma, mb = a.mean(0), b.mean(0)
    ca, cb = torch.cov(a.T), torch.cov(b.T)
    ev = torch.linalg.eigvals(ca @ cb).real.clamp(min=0)          # (Σa Σb)^½ 의 trace = 고윳값의 제곱근의 합
    return float(((ma - mb) ** 2).sum() + torch.trace(ca) + torch.trace(cb) - 2.0 * ev.sqrt().sum())


def distances(real: dict, gen: dict, dim: int = 64, seed: int = 0) -> dict:
    """real · gen = {빌드: [(특징 (n, D), 레이어 번호), …]} (Powder patch). 표준화 + 주성분 dim 개 위에서 Fréchet 거리²."""
    builds = [b for b in real if len(real[b]) >= 8 and len(gen.get(b, [])) >= 8]
    if len(builds) < 1:
        return {}
    cat = lambda rows: torch.cat([f for f, _ in rows]).float()
    # 표준화와 주성분은 진짜와 합성을 합친 묶음에서 잡는다 — 진짜에서만 잡으면 합성만 벗어나는 방향이 빠져 거리가 작게 나온다
    pool = torch.cat([cat(real[b]) for b in builds] + [cat(gen[b]) for b in builds])
    mu, sd = pool.mean(0, keepdim=True), pool.std(0, keepdim=True) + 1e-6
    _, _, v = torch.pca_lowrank((pool - mu) / sd, q=dim, center=True, niter=4)
    proj = lambda x: ((x - mu) / sd) @ v
    out = {"dim": dim, "real_vs_generated": {}, "real_build_vs_build": {}, "real_within_build": {}}
    rng = np.random.default_rng(seed)
    for b in builds:
        out["real_vs_generated"][b] = frechet(proj(cat(real[b])), proj(cat(gen[b])))
        idx = rng.permutation(len(real[b]))                         # 레이어를 둘로 나눈다
        h = len(idx) // 2
        out["real_within_build"][b] = frechet(proj(cat([real[b][i] for i in idx[:h]])), proj(cat([real[b][i] for i in idx[h:]])))
    for i in range(len(builds)):
        for j in range(i + 1, len(builds)):
            out["real_build_vs_build"][f"{builds[i]}_vs_{builds[j]}"] = frechet(proj(cat(real[builds[i]])), proj(cat(real[builds[j]])))
    m = lambda d: float(np.mean(list(d.values()))) if d else float("nan")
    out["mean"] = {k: m(out[k]) for k in ("real_vs_generated", "real_build_vs_build", "real_within_build")}
    out["generated_gap_over_build_gap"] = out["mean"]["real_vs_generated"] / out["mean"]["real_build_vs_build"] if out["real_build_vs_build"] else float("nan")
    return out


def run_probe(feats: list[tuple[torch.Tensor, int, int]], device, seed: int = 0) -> dict | None:
    """feats = (특징 (n, D), 라벨 0/1, 레이어 묶음 번호) 의 목록. 묶음 단위로 7 : 3 으로 나눈다."""
    groups = sorted({g for _, _, g in feats})
    if len(groups) < 6 or len({y for _, y, _ in feats}) < 2:
        return None
    rng = np.random.default_rng(seed)
    test = set(rng.permutation(groups)[:max(2, int(round(0.3 * len(groups))))].tolist())
    tr = [(f, y) for f, y, g in feats if g not in test and len(f)]
    te = [(f, y) for f, y, g in feats if g in test and len(f)]
    if not tr or not te:
        return None
    cat = lambda rows: (torch.cat([f for f, _ in rows]), np.concatenate([np.full(len(f), y) for f, y in rows]))
    xtr, ytr = cat(tr)
    xte, yte = cat(te)
    if len(set(ytr.tolist())) < 2 or len(set(yte.tolist())) < 2:
        return None
    return linear_probe(xtr, ytr, xte, yte, device)


def main():
    ap = argparse.ArgumentParser(description="GenAug K5 — 진짜와 합성을 가르는 선형 분류기")
    ap.add_argument("--set", default=cfg.SET_NAME)
    ap.add_argument("--max-layers", type=int, default=150)
    ap.add_argument("--backbone", default="dinov2_vitb14_reg", help="분할 모델 (v6 · v7) 의 backbone")
    ap.add_argument("--per-layer", type=int, default=150, help="레이어마다 뽑는 Powder patch 수 (Printed 는 그 1/3 까지)")
    args = ap.parse_args()
    log = setup_logger(0, "genaug.probe")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = resolve_set(args.set)
    man = read_manifest(root)
    S = man["img_size"]
    E = frame_px(S)
    canvas = CanvasPool(tuple(man["builds"]))
    model = DefSegModel(backbone_name=args.backbone, n_classes=cfg.N_CLASSES).to(device).eval()
    p = model.patch_size
    inner = np.zeros((S, S), dtype=bool)
    inner[E:S - E, E:S - E] = True
    rng = np.random.default_rng(0)
    layers = pick_layers(man, args.max_layers)
    rows = {"Pow": [], "Prt": []}             # (특징, 라벨 [0 = 진짜, 1 = 합성], 레이어 번호)
    by_build = {"Pow": {}}                    # 진짜 사진의 Powder patch — 빌드끼리 가르는 대조군
    gen_build: dict = {}                      # 합성 레이어의 Powder patch (빌드별) — 거리를 재는 데 쓴다
    for k, m in enumerate(layers):
        g0, g1, ann = read_layer(root, m["id"])
        r0, r1, part = canvas.crop(m["build"], m["row"], 0, 0, S, S)
        base = cad_label(part)
        fg, fr = patch_features(model, g0, g1, device), patch_features(model, r0, r1, device)
        for name, c, cap in (("Pow", cfg.POWDER, args.per_layer), ("Prt", cfg.PRINTED, args.per_layer // 3)):
            ok = pure_patches(inner & (ann == c) & (base == c), p)      # 합성에서도 진짜 (CAD) 에서도 그 클래스인 patch
            ys, xs = np.nonzero(ok)
            if len(ys) == 0:
                continue
            sel = rng.permutation(len(ys))[:cap]
            iy, ix = torch.from_numpy(ys[sel]).to(device), torch.from_numpy(xs[sel]).to(device)
            rows[name].append((fr[iy, ix].cpu(), 0, k))
            rows[name].append((fg[iy, ix].cpu(), 1, k))
            if name == "Pow":
                by_build["Pow"].setdefault(m["build"], []).append((fr[iy, ix].cpu(), k))
                gen_build.setdefault(m["build"], []).append((fg[iy, ix].cpu(), k))
        if (k + 1) % 50 == 0:
            log.info(f"  {k + 1}/{len(layers)} 장")
    doc = {"set": man["name"], "n_layers": len(layers), "backbone": args.backbone, "patch_px": p,
           "note": "AUC of a linear classifier on frozen backbone patch features (last stage, visible/0 + visible/1), test = held-out layers. "
                   "real_vs_generated: 0 = real photo, 1 = generated (same layer, same patches). real_build_vs_build: two real builds (reference)"}
    doc["real_vs_generated"] = {n: run_probe(v, device) for n, v in rows.items()}
    builds = list(by_build["Pow"])
    ref = {}
    for a in range(len(builds)):
        for b in range(a + 1, len(builds)):
            feats = [(f, 0, g) for f, g in by_build["Pow"][builds[a]]] + [(f, 1, g) for f, g in by_build["Pow"][builds[b]]]
            ref[f"{builds[a]}_vs_{builds[b]}"] = run_probe(feats, device)
    doc["real_build_vs_build_powder"] = ref
    doc["frechet_powder"] = distances(by_build["Pow"], gen_build)
    out = cfg.set_fig_dir(man["name"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "probe.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    log.info(f"K5 진짜 · 합성을 가르는 선형 분류기 — 세트 {man['name']} 의 {len(layers)} 장, backbone {args.backbone}")
    for n, r in doc["real_vs_generated"].items():
        if r:
            log.info(f"  {n}: AUC {r['auc']:.3f}, 정확도 {r['acc']:.3f}  (patch 학습 {r['n_train']:,d} · 시험 {r['n_test']:,d})")
        else:
            log.info(f"  {n}: 표본이 모자라 재지 못했다")
    for n, r in ref.items():
        if r:
            log.info(f"  대조 — 진짜 사진의 Powder, {n}: AUC {r['auc']:.3f}")
    fd = doc["frechet_powder"]
    if fd:
        mm = fd["mean"]
        log.info(f"  거리 (Powder patch, Fréchet², 주성분 {fd['dim']}개): 진짜 ↔ 합성 {mm['real_vs_generated']:.1f} "
                 f"({', '.join(f'{b} {v:.1f}' for b, v in fd['real_vs_generated'].items())}),  진짜 빌드 ↔ 빌드 {mm['real_build_vs_build']:.1f} "
                 f"({', '.join(f'{b} {v:.1f}' for b, v in fd['real_build_vs_build'].items())}),  같은 빌드 안 {mm['real_within_build']:.1f}")
        log.info(f"  → 합성과 진짜의 거리는 빌드 사이 거리의 {fd['generated_gap_over_build_gap']:.2f}배다")
    log.info(f"→ {out / 'probe.json'}")


if __name__ == "__main__":
    main()
