"""학습된 생성 모델로 표본 시트를 다시 만든다 (학습 뒤 — seed · 클래스당 줄 수 · 단계 수 · guidance 를 바꿔 가며).

실행 (CWD = 3DP_VPPM/, DefSeg_AM/venv):
    python -m DefSeg_AM.genaug.sampling.sample_sheet                          # checkpoints/genaug/fill_w256/last.pt
    python -m DefSeg_AM.genaug.sampling.sample_sheet --per-class 4 --seed 7 --guidance 3.0
    python -m DefSeg_AM.genaug.sampling.sample_sheet --ckpt checkpoints/genaug/fill_w256/ema_step_050000.pt
산출물: figures/genaug/<run>/samples/sheet_step<step>_seed<seed>_g<guidance>.png
"""
from __future__ import annotations

import argparse
from pathlib import Path

import torch

from ...common.utils.log import setup_logger
from .. import config_genaug as cfg
from ..data.canvas_pool import CanvasPool
from ..data.gt_pool import GTPool
from .fill import load_generator
from .preview import build_preview, write_preview


def main():
    ap = argparse.ArgumentParser(description="GenAug — 표본 시트")
    ap.add_argument("--run-name", default=cfg.RUN_NAME)
    ap.add_argument("--ckpt", type=Path, default=None, help="기본: checkpoints/genaug/<run>/last.pt")
    ap.add_argument("--raw", action="store_true", help="EMA 사본 대신 학습용 가중치로 그린다 (last.pt 만)")
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    ap.add_argument("--per-class", type=int, default=2)
    ap.add_argument("--steps", type=int, default=cfg.SAMPLE_STEPS)
    ap.add_argument("--guidance", type=float, default=cfg.SAMPLE_CFG)
    ap.add_argument("--no-amp", dest="amp", action="store_false", default=cfg.USE_AMP)
    ap.add_argument("--out", type=Path, default=None)
    args = ap.parse_args()
    log = setup_logger(0, "genaug.sample")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ckpt = args.ckpt or (cfg.ckpt_dir(args.run_name) / "last.pt")
    if not ckpt.is_file():
        raise SystemExit(f"FATAL: {ckpt} 없음 — 생성 모델을 먼저 학습한다 (compose 의 train phase)")
    model, diff, ck = load_generator(ckpt, device, use_ema=not args.raw)
    c = ck["config"]
    log.info(f"{ckpt}  step {ck['step']}  holdout={c['holdout']}  창 {c['win']}²  배경 {c['train_builds']}  정상 라벨 {c['normal_builds']}")
    gt = GTPool(holdout=c["holdout"], log=log, limit=16 if c.get("quick") else 0)
    canvas = CanvasPool(tuple(c["train_builds"]), log=log) if c["train_builds"] else None
    items = build_preview(gt, canvas, tuple(c["normal_builds"]), win=c["win"], seed=args.seed, per_class=args.per_class)
    out = args.out or (cfg.fig_dir(c["run_name"]) / "samples" / f"sheet_step{ck['step']:06d}_seed{args.seed}_g{args.guidance:g}.png")
    write_preview(model, diff, items, out,
                  header=f"{c['run_name']}  step {ck['step']}  {'raw' if args.raw else 'EMA'}  steps {args.steps}  guidance {args.guidance}  seed {args.seed}",
                  steps=args.steps, guidance=args.guidance, seed=args.seed, amp=args.amp)
    log.info(f"{len(items)} 줄 → {out}")


if __name__ == "__main__":
    main()
