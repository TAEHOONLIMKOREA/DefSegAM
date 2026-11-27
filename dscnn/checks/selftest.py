"""DSCNN 재구현의 자체 점검 — 데이터 없이 CPU 에서 몇 초 (compose 로는 DSCNN_PHASE=selftest).

  1. 파라미터 수가 논문의 수와 맞는가 (13,996,612 + 179·(n_classes + 1) 과의 차이를 적는다)
  2. 갈래마다 출력 크기가 표 3 – 6 과 같은가, 타일 크기가 달라도 도는가
  3. 타일 격자가 사진을 빠짐없이 덮는가 (가장자리에 붙인 타일 포함)
  4. stack B 의 가운데가 원본 타일과 같은가, stack C 가 타일을 128² 로 줄인 것과 같은가, stack D 의 범위가 [−1, 1] 인가
  5. 증강: 밝기 이동은 A · B · C 에, 잡음은 A · C 에만 들어가는가
  6. 레이어 추론이 모든 화소를 채우는가, EMA 사본이 BatchNorm 통계를 함께 갖는가
"""
from __future__ import annotations

import copy

import numpy as np
import torch
import torch.nn.functional as F

from .. import config_dscnn as cfg
from ..data.tiles import (
    TileBank, combo_to_aug, make_layer, make_stacks, median_frequency_weights, tile_grid,
)
from ..inference.predict import predict_layer
from ..models.dscnn import DSCNN, N_FEATURES, count_parameters
from ..training.train import Ema


def check(ok: bool, msg: str) -> None:
    print(("  ok   " if ok else "  FAIL ") + msg)
    if not ok:
        raise SystemExit(f"FATAL: {msg}")


def main():
    torch.manual_seed(0)
    rng = np.random.default_rng(0)
    dev = torch.device("cpu")

    print("[1] 파라미터 수")
    for n in (9, 10):
        m = DSCNN(n_classes=n)
        c = count_parameters(m)
        paper = 13_996_612 + 179 * (n + 1)
        print(f"     n_classes={n}: 학습 {c['learnable']:,} + BN 통계 {c['bn_stats']:,} = {c['total']:,}   논문 {paper:,}   차이 {c['total'] - paper:+,}")
        check(c["total"] == 13_997_328 + 179 * n, f"n_classes={n}: 13,997,328 + 179·n 과 같다")
        check(abs(c["total"] - paper) < 1000, "논문의 수와 1,000 안쪽으로 맞는다")

    print("[2] 출력 크기")
    m = DSCNN(n_classes=10).eval()
    check(m.classify.in_channels == N_FEATURES == 178, "이어 붙인 특징이 178 채널")
    for t in (112, 225):
        n = 2
        b = torch.randn(n, 2, t + 10, t + 10)
        c = torch.randn(n, 2, 132, 132)
        d = torch.zeros(n, 2, t, t)
        a = torch.randn(n, 2, 36, 36)
        with torch.no_grad():
            check(tuple(m.regional(c).shape) == (n, 32, 128, 128), f"T={t}: regional (N, 32, 128, 128)")
            check(tuple(m.glob(a).shape) == (n, 16), f"T={t}: global (N, 16)")
            check(tuple(F.relu(m.local(b)).shape) == (n, 128, t, t), f"T={t}: localization (N, 128, T, T)")
            out = m(b, c, d, a=a)
            check(tuple(out.shape) == (n, 10, t, t), f"T={t}: logits (N, 10, T, T)")
            g = m.global_features(a[:1])
            out2 = m(b, c, d, g=g)
            check(tuple(out2.shape) == (n, 10, t, t), f"T={t}: 미리 낸 global 특징으로도 돈다")
            out3 = m.head(b, d, m.regional(c), m.glob(a))
            check(torch.allclose(out3, out, atol=1e-5), f"T={t}: 갈래를 따로 불러도 같은 값이다")

    print("[3] 타일 격자")
    for h, w, t in ((766, 766, 112), (2325, 2325, 225), (4761, 1934, 450), (2910, 4656, 794), (450, 900, 450)):
        g = tile_grid(h, w, t)
        cover = np.zeros((h, w), dtype=np.int32)
        for y0, x0 in g.tolist():
            cover[y0: y0 + t, x0: x0 + t] += 1
        check(cover.min() >= 1 and g[:, 0].max() + t == h and g[:, 1].max() + t == w, f"{h}×{w}, T={t}: {len(g)} 타일이 빠짐없이 덮는다")

    print("[4] stack B · C · D")
    h, w, t = 300, 420, 96
    i0 = rng.integers(0, 256, (h, w), dtype=np.uint8)
    i1 = rng.integers(0, 256, (h, w), dtype=np.uint8)
    ann = rng.integers(-1, 10, (h, w)).astype(np.int8)
    L = make_layer(i0, i1, ann, key="t/0", group="g", t=t, mean=100.0, device=dev)
    grid = tile_grid(h, w, t)
    items = np.concatenate([np.zeros((len(grid), 1), dtype=np.int64), grid], axis=1)
    st = make_stacks([L], items)
    lp = cfg.LOCAL_PAD
    k = 5                                         # 안쪽 타일 하나
    y0, x0 = grid[k]
    ref = torch.from_numpy(np.stack([i0, i1])[:, y0: y0 + t, x0: x0 + t].astype(np.float32)) - 100.0
    check(torch.equal(st["b"][k][:, lp:-lp, lp:-lp], ref), "B 의 가운데 T² 가 원본 타일 − ū 와 같다")
    check(tuple(st["b"].shape[-2:]) == (t + 10, t + 10) and tuple(st["c"].shape[-2:]) == (132, 132), "B 는 (T+10)², C 는 132²")
    want = F.interpolate(ref[None], size=(128, 128), mode="bilinear", align_corners=False)[0]
    rp = cfg.REGIONAL_PAD
    # 타일의 맨 바깥 줄은 견주지 않는다 — 타일만 따로 줄이면 가장자리에서 이웃 화소를 못 읽지만, stack C 는 사진의 진짜 이웃을 읽는다
    err = (st["c"][k][:, rp + 2:-rp - 2, rp + 2:-rp - 2] - want[:, 2:-2, 2:-2]).abs().max().item()
    check(err < 1e-2, f"C 의 가운데 128² 이 타일을 bilinear 로 128² 로 옮긴 것과 같다 (최대 차 {err:.1e})")
    check(torch.equal(st["label"][k], torch.from_numpy(ann[y0: y0 + t, x0: x0 + t].astype(np.int64))), "라벨이 같은 자리다")
    d = st["d"]
    check(-1.0 < d.min().item() and d.max().item() < 1.0, "D 가 (−1, 1) 안이다")
    check(abs(d[k][0, 0, 0].item() - ((x0 + 0.5) / w * 2 - 1)) < 1e-6 and abs(d[k][1, 0, 0].item() - ((y0 + 0.5) / h * 2 - 1)) < 1e-6, "D 의 채널 0 은 x, 채널 1 은 y")
    a = L.a
    check(tuple(a.shape) == (2, 36, 36), "A 는 (2, 36, 36)")

    print("[5] 증강")
    combo = np.arange(9)
    nv, sh = combo_to_aug(combo)
    check(sorted({(float(a_), round(float(b_), 2)) for a_, b_ in zip(nv, sh)}).__len__() == 9, "증강 9 가지가 서로 다르다")
    it9 = np.repeat(items[k: k + 1], 9, axis=0)
    base = make_stacks([L], it9)
    aug = make_stacks([L], it9, noise_var=nv, shift=sh, generator=torch.Generator().manual_seed(1))
    db = (aug["b"] - base["b"])
    check(torch.allclose(db.flatten(1).mean(1), sh * 255.0, atol=1e-3) and db.flatten(1).std(1).max().item() < 1e-3, "B 에는 밝기 이동만 들어간다 (잡음 없음)")
    dc = (aug["c"] - base["c"]) - (sh * 255.0)[:, None, None, None]
    sd = dc.flatten(1).std(1)
    want_sd = nv.sqrt() * 255.0
    check(torch.allclose(sd, want_sd, rtol=0.05, atol=1e-3), f"C 의 잡음 σ = {[round(float(v), 2) for v in want_sd.unique()]} 단계")
    da = (aug["a"] - base["a"]) - (sh * 255.0)[:, None, None, None]
    check(torch.allclose(da.flatten(1).std(1), want_sd, rtol=0.15, atol=1e-3), "A 에도 같은 잡음이 들어간다")

    print("[6] 레이어 추론 · 타일 목록 · EMA")
    m = DSCNN(n_classes=10).eval()
    pred = predict_layer(m, L, pixel_budget=3 * t * t)
    check(tuple(pred.shape) == (h, w) and int(pred.min()) >= 0 and int(pred.max()) < 10, "예측이 (H, W) 이고 0–9 로 찼다")
    ann2 = ann.copy()
    ann2[:t, :t] = -1                              # 한 타일은 주석이 없다
    L2 = make_layer(i0, i1, ann2, key="t/1", group="g", t=t, mean=100.0, device=dev)
    bank = TileBank([L, L2])
    check(len(bank) == bank.n_grid - 1, "주석이 10% 에 못 미치는 타일이 빠진다")
    batches = bank.epoch_batches(rng)
    seen = sorted((int(a_), int(b_)) for tid, cb in batches for a_, b_ in zip(tid, cb))
    check(seen == sorted((i, c) for i in range(len(bank)) for c in range(9)), "한 epoch 가 (타일 × 증강 9 가지) 를 한 번씩 돈다")
    big = make_layer(rng.integers(0, 256, (900, 900), dtype=np.uint8), rng.integers(0, 256, (900, 900), dtype=np.uint8),
                     np.zeros((900, 900), dtype=np.int8), key="t/2", group="big", t=450, mean=100.0, device=dev)
    bank2 = TileBank([L, big])
    b2 = bank2.epoch_batches(rng)
    sizes = [int((bank2.tile_t[tid] ** 2).sum()) for tid, _ in b2]
    check(max(len(tid) for tid, _ in b2) <= cfg.BATCH_MAX_TILES and max(sizes) <= cfg.BATCH_PIXEL_BUDGET,
          f"배치가 {cfg.BATCH_MAX_TILES} 장 · 화소 예산을 넘지 않는다")
    check(any(len(set(bank2.tile_t[tid].tolist())) > 1 for tid, _ in b2), "한 배치에 타일 크기가 다른 소스가 섞인다")
    check(len(b2) == bank2.steps_per_epoch() or abs(len(b2) - bank2.steps_per_epoch()) <= 2, "steps_per_epoch 가 실제 배치 수와 맞는다")
    wts = median_frequency_weights(np.array([800, 100, 0, 50, 50, 0, 0, 0, 0, 0]))
    check(abs(wts[0] - 0.09375) < 1e-6 and wts[2] == 0 and abs(wts[3] - 1.5) < 1e-6, "클래스 가중치 = 중앙값 빈도 / 빈도 (없는 클래스는 0)")
    m.train()
    ema = Ema(m, 0.9)
    tgt = copy.deepcopy(m)
    with torch.no_grad():
        for p in m.parameters():
            p.add_(1.0)
        m.regional.d1.bn.running_mean.add_(3.0)
    ema.update(m)
    ema.copy_to(tgt, m)
    p0 = next(iter(m.parameters()))
    q0 = next(iter(tgt.parameters()))
    check(torch.allclose(p0 - q0, torch.full_like(p0, 2.0 / 11.0), atol=1e-6), "EMA 가 처음에는 빠르게 따라간다 (decay = (1 + n) / (10 + n) = 2/11)")
    check(torch.equal(tgt.regional.d1.bn.running_mean, m.regional.d1.bn.running_mean), "EMA 사본이 BatchNorm 통계를 그대로 받는다")
    print("selftest: 모두 통과")


if __name__ == "__main__":
    main()
