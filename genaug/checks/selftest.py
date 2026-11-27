"""GenAug 자체 점검 — 배선이 맞는지 CPU 로 1–2분 안에 본다 (학습 없음).

보는 것:
  1. holdout 목록 — 생성 모델이 읽는 사람 GT 에 두 판정 fold 의 검증 레이어가 하나도 없다
  2. 학습 창 — 값의 범위, 그릴 자리 밖의 라벨이 전부 "모름", 정상 라벨을 적는 빌드와 아닌 빌드
  3. 모델 — 입력 · 출력의 모양, 가중치 수
  4. 확산 과정 — 잡음을 섞었다 되돌리는 식이 맞는다
  5. 그리기 — 그릴 자리 밖의 화소가 한 값도 바뀌지 않는다 (강도 0 · 0.5 · 1)
  6. 합성 세트 — 창의 자리와 순서가 안쪽을 다 덮는다, 라벨 맵의 값과 자리 규칙, 같은 seed 는 같은 라벨 맵

실행 (CWD = 3DP_VPPM/, DefSeg_AM/venv):
    python -m DefSeg_AM.genaug.checks.selftest
"""
from __future__ import annotations

import time

import numpy as np
import torch

from ...common.utils.log import setup_logger
from .. import config_genaug as cfg
from ..data.canvas_pool import CanvasPool
from ..data.gt_pool import GTPool, load_holdout
from ..data.windows import KIND_GT_DEFECT, FillWindows, crop_ornl
from ..models.diffusion import Diffusion, sample_fill, sample_steps, training_loss
from ..models.unet import build_unet, count_params
from ..synth.label_map import DB, RH, RS, SE, Rules, ShapeBank, build_label_map, to_annotation
from ..synth.layer import frame_px, owner_map, window_order, window_positions


def main():
    log = setup_logger(0, "genaug.selftest")
    torch.manual_seed(0)

    # 1. holdout
    both, inl = load_holdout("both"), load_holdout("inlayer")
    assert len(both) == 38 and len(inl) == 22 and set(inl) <= set(both)
    t = time.time()
    gt = GTPool("both", log=log)
    assert len(gt) == cfg.HOLDOUT_EXPECTED_LAYERS["both"] and not (set(gt.keys) & set(both))
    assert not any(k.startswith("v2021_LPBF/") for k in gt.keys)
    log.info(f"1. holdout OK — {len(gt)} 레이어, 뺀 레이어와 겹침 0 ({time.time() - t:.0f}초)")

    # 2. 학습 창
    normal = tuple(b for b in cfg.NORMAL_LABEL_BUILDS if b in cfg.TRAIN_BUILDS)
    canvas = CanvasPool(cfg.TRAIN_BUILDS, log=log)
    ds = FillWindows(gt, canvas, n_virtual=10, seed=1, normal_builds=normal)
    kinds, cls_cnt, t = {0: 0, 1: 0, 2: 0}, {}, time.time()
    for i in range(300):
        s = ds[i]
        x, lab, reg = s["x"], s["label"], s["region"]
        assert x.shape == (2, cfg.WIN, cfg.WIN) and -1.0001 <= float(x.min()) and float(x.max()) <= 1.0001
        assert lab.dtype == torch.uint8 and int(lab.max()) <= cfg.UNKNOWN
        assert set(np.unique(reg.numpy()).tolist()) <= {0.0, 1.0} and float(reg.sum()) > 0
        assert bool((lab[reg[0] == 0] == cfg.UNKNOWN).all())
        kinds[s["kind"]] += 1
        if s["kind"] == KIND_GT_DEFECT:
            cls_cnt[cfg.CLASS_ABBRS[s["cls"]]] = cls_cnt.get(cfg.CLASS_ABBRS[s["cls"]], 0) + 1
    log.info(f"2. 학습 창 OK — 초당 {300 / (time.time() - t):.0f} 창 (프로세스 1개), 종류 (결함 창 · GT 창 · ORNL 창) {list(kinds.values())}, 결함 창의 클래스 {cls_cnt}")
    if normal:
        vals = set()
        for i in range(40):
            b, row = normal[0], int(np.random.default_rng(i).integers(canvas.n_layers[normal[0]]))
            vals |= set(np.unique(crop_ornl(canvas, b, row, 300, 300, cfg.WIN, True)[2]).tolist())
        assert vals <= {cfg.POWDER, cfg.PRINTED, cfg.UNKNOWN} and cfg.POWDER in vals
    other = [b for b in cfg.TRAIN_BUILDS if b not in normal]
    if other:
        assert set(np.unique(crop_ornl(canvas, other[0], 0, 300, 300, cfg.WIN, False)[2]).tolist()) == {cfg.UNKNOWN}
    log.info(f"   ORNL 라벨 맵 OK — 정상 라벨을 적는 빌드 {list(normal) or '없음'}, 나머지는 전부 '모름'")

    # 3. 모델
    mc = dict(in_ch=cfg.MODEL_IN_CH, out_ch=cfg.IMG_CH, base=cfg.BASE_CH, mult=cfg.CH_MULT, n_res=cfg.N_RES_BLOCKS,
              attn_res=cfg.ATTN_RESOLUTIONS, heads=cfg.ATTN_HEADS, dropout=cfg.DROPOUT, win=cfg.WIN)
    log.info(f"3. 모델 — 기본 구성 {count_params(build_unet(mc)) / 1e6:.1f} M")
    small = dict(mc, base=32, mult=(1, 2, 2, 4), n_res=1, attn_res=(8,), win=64)
    net = build_unet(small)
    y = net(torch.randn(2, cfg.MODEL_IN_CH, 64, 64), torch.tensor([10, 900]))
    assert y.shape == (2, cfg.IMG_CH, 64, 64) and float(y.detach().abs().max()) == 0.0

    # 4. 확산 과정
    d = Diffusion(cfg.T_STEPS)
    assert d.sqrt_ac[0] > 0.99 and d.sqrt_ac[-1] < 0.01 and bool(torch.all(d.sqrt_ac[1:] < d.sqrt_ac[:-1]))
    x0 = torch.rand(3, 2, 32, 32) * 2 - 1
    n = torch.randn_like(x0)
    tt = torch.tensor([0, 500, 999])
    r0, re = d.x0_eps_from_v(d.q_sample(x0, tt, n), tt, d.v_target(x0, tt, n))
    assert torch.allclose(r0, x0, atol=1e-4) and torch.allclose(re, n, atol=1e-4)
    log.info("4. 확산 과정 OK")

    # 5. 그리기
    dsq = FillWindows(gt, canvas, n_virtual=10, seed=2, normal_builds=normal, win=64)
    b = [dsq[i] for i in range(4)]
    x = torch.stack([s["x"] for s in b])
    reg = torch.stack([s["region"] for s in b])
    lab = torch.stack([s["label"] for s in b]).long()
    loss, _, _ = training_loss(net, d, x, reg, lab)
    loss.backward()
    g = torch.Generator().manual_seed(0)
    out = sample_fill(net, d, x, reg, lab, steps=6, guidance=2.0, generator=g)
    assert torch.equal(out * (1 - reg), x * (1 - reg))
    out = sample_fill(net, d, x, reg, lab, steps=6, guidance=1.0, strength=0.5, init=x.flip(-1), generator=g)
    assert torch.equal(out * (1 - reg), x * (1 - reg))
    assert torch.equal(sample_fill(net, d, x, reg, lab, strength=0.0, init=x.flip(-1)), reg * x.flip(-1) + (1 - reg) * x)
    log.info(f"5. 그리기 OK — 그릴 자리 밖의 화소가 그대로다 (손실 {loss.item():.3f})")

    # 6. 합성 세트 — 창의 자리 · 순서, 라벨 맵
    for sp in ("uniform", "quad"):
        ts = sample_steps(cfg.T_STEPS - 1, 50, sp)
        assert int(ts[0]) == cfg.T_STEPS - 1 and int(ts[-1]) == 0 and bool(torch.all(ts[1:] < ts[:-1]))
    S = canvas.img_size
    E = frame_px(S)
    pos = window_positions(S, E)
    assert pos[0] == E - cfg.BORDER and pos[-1] + cfg.WIN == S - E + cfg.BORDER and max(np.diff(pos)) <= cfg.SET_STRIDE
    for order in ("spiral", "raster"):
        seq = window_order(len(pos), order)
        assert sorted(seq) == sorted((i, j) for i in range(len(pos)) for j in range(len(pos)))
        own = owner_map(S, cfg.SET_STRIDE, order)
        assert int(own.min()) == -1 and bool((own[E:S - E, E:S - E] >= 0).all()) and bool((own[:E] == -1).all())
    rules = Rules()
    bank = ShapeBank(gt, rules)
    n_def, px = 0, np.zeros(cfg.N_CLASSES, dtype=np.int64)
    for i in range(24):
        b = cfg.SET_BUILDS[i % len(cfg.SET_BUILDS)]
        _, _, part = canvas.crop(b, int(np.random.default_rng(i).integers(canvas.n_layers[b])), 0, 0, S, S)
        lab, placed = build_label_map(part, np.random.default_rng([7, i]), bank, rules, E)
        lab2, _ = build_label_map(part, np.random.default_rng([7, i]), bank, rules, E)
        assert np.array_equal(lab, lab2)                                        # 같은 seed 는 같은 라벨 맵
        assert set(np.unique(lab).tolist()) <= {cfg.POWDER, cfg.PRINTED, RS, RH, SE, DB, cfg.UNKNOWN}
        ann = to_annotation(lab, E)
        assert bool((ann[:E] == -1).all() and (ann[:, -E:] == -1).all()) and int(ann.min()) == -1
        inner_def = (ann >= 2)
        assert not inner_def[:E + rules.margin].any() and not inner_def[:, S - E - rules.margin:].any()   # 결함은 안쪽에만
        for c in (RS, RH):                                                      # 줄은 분말 위에만 — CAD 의 파트 안에는 없다
            assert not (part & (lab == c)).any()
        n_def += bool(placed)
        px += np.bincount(ann[ann >= 0].astype(np.int64), minlength=cfg.N_CLASSES)
    assert 8 <= n_def <= 22 and all(px[c] > 0 for c in (RS, RH, SE, DB))
    log.info(f"6. 합성 세트 OK — 창 {len(pos)}×{len(pos)} (자리 {pos}), 라벨 맵 24 장 가운데 결함이 든 것 {n_def} 장, "
             + ", ".join(f"{cfg.CLASS_ABBRS[c]} {int(px[c]):,d} px" for c in (RS, RH, SE, DB)))
    log.info("자체 점검 통과")


if __name__ == "__main__":
    main()
