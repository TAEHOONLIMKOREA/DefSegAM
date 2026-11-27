"""결함 인페인팅 확산 모델의 학습 (PLAN_GENAUG §3.2).

실행 (CWD = 3DP_VPPM/, DefSeg_AM/venv) — 풀런은 compose 로 돌린다 (docker/README.md):
    python -m DefSeg_AM.genaug.training.train_fill --quick                         # 스모크 (CPU 도 된다)
    torchrun --nproc_per_node=2 -m DefSeg_AM.genaug.training.train_fill            # GPU 2대

산출물:
    checkpoints/genaug/<run>/last.pt            이어 돌리기용 (학습용 가중치 · EMA 사본 · optimizer · step)
    checkpoints/genaug/<run>/ema_step_NNNNNN.pt EMA 사본만 (KEEP_EVERY 마다, 지우지 않는다)
    figures/genaug/<run>/samples/step_NNNNNN.png 표본 시트 (sampling/preview.py)
    figures/genaug/<run>/{run.json, curve.csv}

용어: step = 창 한 묶음 (GPU 수 × batch) 을 보고 가중치를 한 번 고치는 것.
      EMA = 학습용 가중치와 따로 들고 가는 평균 사본. 그림을 그릴 때는 이 사본을 쓴다.
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import os
import subprocess
import time
from pathlib import Path

import numpy as np
import torch
from torch.utils.data import DataLoader

from ...common.training.dist_utils import init_distributed, is_main, unwrap
from ...common.utils.log import setup_logger
from .. import config_genaug as cfg
from ..data.canvas_pool import CanvasPool
from ..data.gt_pool import GTPool
from ..data.windows import KIND_GT_DEFECT, KIND_GT_RANDOM, KIND_ORNL, FillWindows
from ..models.diffusion import Diffusion, training_loss
from ..models.unet import build_unet, count_params
from ..sampling.preview import build_preview, write_preview

KIND_NAMES = {KIND_GT_DEFECT: "gt_defect", KIND_GT_RANDOM: "gt_random", KIND_ORNL: "ornl"}


def builds_arg(v: list[str]) -> tuple[str, ...]:
    return () if (len(v) == 1 and v[0].lower() == "none") else tuple(v)


def parse_args():
    ap = argparse.ArgumentParser(description="GenAug — 결함 인페인팅 확산 모델 학습")
    ap.add_argument("--run-name", default=cfg.RUN_NAME)
    ap.add_argument("--steps", type=int, default=cfg.TOTAL_STEPS)
    ap.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE, help="GPU 당")
    ap.add_argument("--lr", type=float, default=cfg.LR)
    ap.add_argument("--warmup", type=int, default=cfg.WARMUP_STEPS)
    ap.add_argument("--ema-decay", type=float, default=cfg.EMA_DECAY)
    ap.add_argument("--win", type=int, default=cfg.WIN)
    ap.add_argument("--base-ch", type=int, default=cfg.BASE_CH)
    ap.add_argument("--ch-mult", type=int, nargs="+", default=list(cfg.CH_MULT))
    ap.add_argument("--n-res", type=int, default=cfg.N_RES_BLOCKS)
    ap.add_argument("--attn-res", type=int, nargs="+", default=list(cfg.ATTN_RESOLUTIONS))
    ap.add_argument("--holdout", choices=list(cfg.HOLDOUT_EXPECTED_LAYERS), default=cfg.DEFAULT_HOLDOUT,
                    help="보지 않을 사람 GT. both = 두 판정 fold 의 검증 레이어 (87 레이어로 학습), inlayer = inlayer 의 검증만 (103)")
    ap.add_argument("--train-builds", nargs="+", default=list(cfg.TRAIN_BUILDS), help="주석 없는 배경으로 쓸 빌드. none = 쓰지 않는다")
    ap.add_argument("--normal-builds", nargs="+", default=list(cfg.NORMAL_LABEL_BUILDS),
                    help="라벨 맵에 Powder / Printed 를 적어 주는 빌드 (Q0 통과). none = 적지 않는다")
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--no-amp", dest="amp", action="store_false", default=cfg.USE_AMP, help="bfloat16 autocast 를 끈다 (fp32. 메모리가 더 든다)")
    ap.add_argument("--seed", type=int, default=cfg.SEED)
    ap.add_argument("--log-every", type=int, default=cfg.LOG_EVERY)
    ap.add_argument("--sample-every", type=int, default=cfg.SAMPLE_EVERY)
    ap.add_argument("--save-every", type=int, default=cfg.SAVE_EVERY)
    ap.add_argument("--keep-every", type=int, default=cfg.KEEP_EVERY)
    ap.add_argument("--sample-steps", type=int, default=cfg.SAMPLE_STEPS)
    ap.add_argument("--guidance", type=float, default=cfg.SAMPLE_CFG)
    ap.add_argument("--resume", action="store_true", help="checkpoints/genaug/<run>/last.pt 에서 이어 돌린다")
    ap.add_argument("--quick", action="store_true", help="스모크 — 작은 모델 · 64 창 · 30 step. run-name 에 _quick 이 붙는다")
    args = ap.parse_args()
    args.train_builds = builds_arg(args.train_builds)
    args.normal_builds = builds_arg(args.normal_builds) if args.train_builds else ()   # 배경을 안 쓰면 정상 라벨을 적을 빌드도 없다
    if args.quick:
        args.run_name += "_quick"
        args.win, args.base_ch, args.ch_mult, args.n_res, args.attn_res = 64, 32, [1, 2, 2, 4], 1, [8]
        args.steps, args.batch_size, args.warmup = min(args.steps, 30), min(args.batch_size, 4), 5
        args.log_every, args.sample_every, args.save_every, args.keep_every = 5, 15, 15, 30
        args.sample_steps, args.num_workers = 8, min(args.num_workers, 2)
    return args


def git_rev() -> str:
    try:
        root = Path(__file__).resolve().parents[2]
        rev = subprocess.run(["git", "-C", str(root), "rev-parse", "--short", "HEAD"], capture_output=True, text=True, timeout=10).stdout.strip()
        dirty = subprocess.run(["git", "-C", str(root), "status", "--porcelain", "--", "genaug", "common"],
                               capture_output=True, text=True, timeout=10).stdout.strip()
        return rev + ("-dirty" if dirty else "")
    except Exception:
        return "unknown"


def atomic_save(obj, path: Path) -> None:
    tmp = path.with_suffix(path.suffix + ".tmp")
    torch.save(obj, tmp)
    os.replace(tmp, path)


def main():
    args = parse_args()
    rank, world, local_rank = init_distributed()
    log = setup_logger(rank=rank, name="genaug")
    main_proc = is_main(rank)
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")
    torch.manual_seed(args.seed + rank)
    if device.type == "cuda":
        torch.backends.cudnn.benchmark = True
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True

    ckpt_dir, fig_dir = cfg.ckpt_dir(args.run_name), cfg.fig_dir(args.run_name)
    if main_proc:
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        (fig_dir / "samples").mkdir(parents=True, exist_ok=True)

    # ── 데이터 ────────────────────────────────────────────────────────────
    gt = GTPool(holdout=args.holdout, log=log if main_proc else None, limit=16 if args.quick else 0)
    canvas = CanvasPool(args.train_builds, log=log if main_proc else None) if args.train_builds else None
    bad = [b for b in args.normal_builds if b not in args.train_builds]
    if bad:
        raise SystemExit(f"FATAL: --normal-builds 의 {bad} 가 --train-builds 에 없다")
    if main_proc and canvas is None:
        log.warning("주석 없는 배경 없이 학습한다 (--train-builds none) — 사람 GT 창만 쓴다")
    if main_proc and canvas is not None and not args.normal_builds:
        log.warning("정상 라벨을 적는 빌드가 없다 (--normal-builds none) — ORNL 배경의 '창 전체 그리기' 는 쓸 수 없다 (PLAN §3.9)")

    # ── 모델 ──────────────────────────────────────────────────────────────
    model_cfg = dict(in_ch=cfg.MODEL_IN_CH, out_ch=cfg.IMG_CH, base=args.base_ch, mult=tuple(args.ch_mult), n_res=args.n_res,
                     attn_res=tuple(args.attn_res), heads=cfg.ATTN_HEADS, dropout=cfg.DROPOUT, win=args.win)
    model = build_unet(model_cfg).to(device)
    if device.type == "cuda":
        model = model.to(memory_format=torch.channels_last)
    ema = copy.deepcopy(model).eval() if main_proc else None
    if ema is not None:
        for p in ema.parameters():
            p.requires_grad_(False)
    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, betas=(0.9, 0.999), weight_decay=cfg.WEIGHT_DECAY)
    diff = Diffusion(cfg.T_STEPS, device)

    step = 0
    last_path = ckpt_dir / "last.pt"
    if args.resume:
        if not last_path.is_file():
            raise SystemExit(f"FATAL: --resume 인데 {last_path} 가 없다")
        ck = torch.load(last_path, map_location="cpu", weights_only=False)
        if ck["model_cfg"] != model_cfg:
            raise SystemExit(f"FATAL: 이어 돌릴 ckpt 의 모델 구성이 다르다: {ck['model_cfg']} != {model_cfg}")
        model.load_state_dict(ck["model"])
        opt.load_state_dict(ck["opt"])
        if ema is not None:
            ema.load_state_dict(ck["ema"])
        step = int(ck["step"])
        if main_proc:
            log.info(f"이어 돌리기: step {step} 부터 ({last_path})")
    if world > 1:
        model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[local_rank])
    net = unwrap(model)

    remaining = max(args.steps - step, 0)
    ds = FillWindows(gt, canvas, n_virtual=max(remaining, 1) * args.batch_size, seed=args.seed, rank=rank,
                     normal_builds=args.normal_builds, win=args.win, stream=step)
    dl = DataLoader(ds, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers, drop_last=True,
                    pin_memory=device.type == "cuda", persistent_workers=args.num_workers > 0)

    config = dict(vars(args), T=cfg.T_STEPS, world_size=world, effective_batch=world * args.batch_size,
                  n_classes=cfg.N_CLASSES, class_abbrs=cfg.CLASS_ABBRS, gen_classes=list(cfg.GEN_CLASSES),
                  schema=cfg.SCHEMA_VERSION, region_mix_defect=cfg.REGION_MIX_DEFECT, region_mix_other=cfg.REGION_MIX_OTHER,
                  p_labeled=ds.p_labeled, p_defect_centered=cfg.P_DEFECT_CENTERED, cond_drop=cfg.COND_DROP,
                  gt_keys=gt.keys, git=git_rev())
    config["train_builds"], config["normal_builds"] = list(args.train_builds), list(args.normal_builds)

    preview = None
    if main_proc:
        n_par = count_params(net)
        log.info(f"run={args.run_name}  device={device}  GPU {world}대 × batch {args.batch_size} = {world * args.batch_size}  "
                 f"창 {args.win}²  모델 {n_par / 1e6:.1f} M (base {args.base_ch}, mult {tuple(args.ch_mult)})  amp={args.amp}")
        log.info(f"step {step} → {args.steps}  lr {args.lr:g} (warmup {args.warmup})  EMA {args.ema_decay}  "
                 f"결함을 가운데 둔 창의 클래스: {[cfg.CLASS_ABBRS[c] for c in ds.def_classes]}")
        if canvas is not None:
            log.info(f"주석 없는 배경: {list(args.train_builds)}  정상 라벨을 적는 빌드: {list(args.normal_builds) or '없음'}")
        preview = build_preview(gt, canvas, args.normal_builds, win=args.win, seed=args.seed)
        log.info(f"표본 시트: {len(preview)} 줄 → {fig_dir / 'samples'}")
        (fig_dir / "run.json").write_text(json.dumps(dict(config, params=n_par, gt_stats=gt.stats(),
                                                          canvas_layers=canvas.n_layers if canvas else {},
                                                          started=time.strftime("%Y-%m-%dT%H:%M:%S%z")),
                                                     indent=1, ensure_ascii=False), encoding="utf-8")
        curve_path = fig_dir / "curve.csv"
        if not (args.resume and curve_path.is_file()):
            with curve_path.open("w", newline="") as fh:
                csv.writer(fh).writerow(["step", "loss", "loss_gt_defect", "loss_gt_random", "loss_ornl", "lr", "steps_per_s", "gpu_gb"])

    def save(tag_keep: bool) -> None:
        state = {"model": net.state_dict(), "ema": ema.state_dict(), "opt": opt.state_dict(), "step": step,
                 "model_cfg": model_cfg, "config": config}
        atomic_save(state, last_path)
        if tag_keep:
            atomic_save({"ema": ema.state_dict(), "step": step, "model_cfg": model_cfg, "config": config},
                        ckpt_dir / f"ema_step_{step:06d}.pt")

    def sample_sheet() -> None:
        t = time.time()
        path = fig_dir / "samples" / f"step_{step:06d}.png"
        write_preview(ema, diff, preview, path, header=f"{args.run_name}  step {step}  EMA  steps {args.sample_steps}  guidance {args.guidance}",
                      steps=args.sample_steps, guidance=args.guidance, seed=args.seed, amp=args.amp)
        log.info(f"표본 시트 {path.name} ({time.time() - t:.0f}초)")

    # ── 학습 ──────────────────────────────────────────────────────────────
    model.train()
    acc = {k: [0.0, 0] for k in KIND_NAMES}
    loss_sum, loss_n = 0.0, 0
    t_log = time.time()
    t0, step0 = time.time(), step
    for batch in dl:
        if step >= args.steps:
            break
        if device.type == "cuda" and step == step0 + 20:       # 처음 몇 step 의 cudnn 탐색이 쓰는 메모리를 빼고 잰다
            torch.cuda.reset_peak_memory_stats(device)
        x = batch["x"].to(device, non_blocking=True)
        region = batch["region"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True).long()
        if device.type == "cuda":
            x, region = x.contiguous(memory_format=torch.channels_last), region.contiguous(memory_format=torch.channels_last)
        lr = args.lr * min(1.0, (step + 1) / max(args.warmup, 1))
        for g in opt.param_groups:
            g["lr"] = lr
        with torch.autocast(device_type=device.type, dtype=torch.bfloat16, enabled=args.amp and device.type == "cuda"):
            loss, per, _ = training_loss(model, diff, x, region, label)
        if not torch.isfinite(loss):
            raise RuntimeError(f"FATAL: step {step} 에서 손실이 유한하지 않다 ({loss.item()})")
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.GRAD_CLIP_NORM)
        opt.step()
        step += 1

        if main_proc:
            d = min(args.ema_decay, (1 + step) / (10 + step))
            with torch.no_grad():
                for pe, pm in zip(ema.parameters(), net.parameters()):
                    pe.mul_(d).add_(pm.detach(), alpha=1 - d)
            loss_sum += loss.item()
            loss_n += 1
            kinds = batch["kind"].numpy()
            pv = per.float().cpu().numpy()
            for k in KIND_NAMES:
                m = kinds == k
                if m.any():
                    acc[k][0] += float(pv[m].sum())
                    acc[k][1] += int(m.sum())

            if step % args.log_every == 0 or step == args.steps:
                dt = time.time() - t_log
                sps = (args.log_every if step % args.log_every == 0 else loss_n) / max(dt, 1e-6)
                eta = (args.steps - step) / max((step - step0) / max(time.time() - t0, 1e-6), 1e-9)
                mem = torch.cuda.max_memory_allocated(device) / 1e9 if device.type == "cuda" else 0.0
                kl = {KIND_NAMES[k]: (acc[k][0] / acc[k][1] if acc[k][1] else float("nan")) for k in KIND_NAMES}
                log.info(f"step [{step:6d}/{args.steps}] loss={loss_sum / max(loss_n, 1):.4f}  "
                         f"(v 오차 — 결함 창 {kl['gt_defect']:.3f} · GT 창 {kl['gt_random']:.3f} · ORNL 창 {kl['ornl']:.3f})  "
                         f"lr={lr:.2e}  {sps:.2f} step/s  GPU {mem:.1f} GB  남은 시간 {eta / 60:.0f}분")
                with (fig_dir / "curve.csv").open("a", newline="") as fh:
                    csv.writer(fh).writerow([step, f"{loss_sum / max(loss_n, 1):.6f}", f"{kl['gt_defect']:.6f}", f"{kl['gt_random']:.6f}",
                                             f"{kl['ornl']:.6f}", f"{lr:.3e}", f"{sps:.3f}", f"{mem:.2f}"])
                acc = {k: [0.0, 0] for k in KIND_NAMES}
                loss_sum, loss_n = 0.0, 0
                t_log = time.time()
            if step % args.save_every == 0 or step == args.steps:
                save(tag_keep=(step % args.keep_every == 0 or step == args.steps))
            if step % args.sample_every == 0 or step == args.steps:
                sample_sheet()
                t_log = time.time()

    if main_proc:
        log.info(f"끝: step {step}, {(time.time() - t0) / 60:.1f}분 — {last_path}")


if __name__ == "__main__":
    main()
