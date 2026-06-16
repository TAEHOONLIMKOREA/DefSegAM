"""v3 Stage 1 학습 — ORNL HDF5 (DSCNN pred) KD pretrain, 10-class.

v1 train_stage1 의 helper (init_distributed, update_counts, reduce_counts) 와
모델 (DefSegModel), 공통 sampler 등 그대로 재사용.

추가 사항 (PLAN_v4 §4.1, §6 + DSCNN_Summary §5.6, §12):
  - 10-class 출력 (config_v4.N_CLASSES_V4 = 10)
  - v4 dataset (image v1 cache + label v4 cache) + augmentation 통합
  - EMA weight saving (decay 0.9999) — 학습 후반 weight 안정화
  - `--use-hard-bootstrap` 옵션 (default off) — noisy teacher pred 대응
  - DDP, warmup+cosine, grad clip, AMP off 그대로

사전조건:
    python -m DefSeg_AM.v4.data.build_cache_v4     # label 만 v4 11-class 재빌드

사용:
    python -m DefSeg_AM.v4.training.train_stage1 [--use-hard-bootstrap] [--quick]
"""
from __future__ import annotations

import argparse
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# 공통 모듈 재사용 (common)
from ...common.data.samplers import DistributedWeightedSampler
from ...common.models.losses import focal_loss, sqrt_inv_class_weight
from ...common.models.model import DefSegModel, round_to_patch
from ...common.training.dist_utils import (
    init_distributed, is_main, unwrap, reduce_counts,
)
from ...common.utils.log import setup_logger

# v4 모듈
from .. import config_v4 as cfg
from ..data.data_ornl_v4 import (
    DefSegORNLCachedDatasetV4,
    estimate_class_counts_v4,
)
from ..models.losses_v4 import hard_bootstrap_loss, median_inv_class_weight


# ---------------------------------------------------------------------------
# v4 의 update_counts (N_CLASSES_V4=11 사용 — v1 의 update_counts 는 config.N_CLASSES=12)
# ---------------------------------------------------------------------------

@torch.no_grad()
def update_counts_v4(
    logits: torch.Tensor, label: torch.Tensor,
    correct: torch.Tensor, total: torch.Tensor,
    inter: torch.Tensor, union: torch.Tensor,
) -> None:
    pred = logits.argmax(dim=1)
    valid = label != cfg.IGNORE_INDEX
    correct += (pred[valid] == label[valid]).sum()
    total += valid.sum()
    for c in range(cfg.N_CLASSES_V4):
        p = (pred == c) & valid
        g = (label == c) & valid
        inter[c] += (p & g).sum()
        union[c] += (p | g).sum()


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=cfg.S1_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=cfg.S1_BATCH_SIZE)
    ap.add_argument("--img-size", type=int, default=cfg.IMG_SIZE)
    ap.add_argument("--lr", type=float, default=cfg.S1_LR)
    ap.add_argument("--backbone", type=str, default=cfg.DINO_BACKBONE)
    ap.add_argument("--num-workers", type=int, default=8)
    ap.add_argument("--gamma", type=float, default=cfg.S1_FOCAL_GAMMA)
    ap.add_argument("--class-aware-alpha", type=float, default=cfg.CLASS_AWARE_ALPHA,
                    help="class-aware sampler 의 inverse-frequency power. 1.0 = strict inv, 0.5 = sqrt")
    ap.add_argument("--n-weight-sample", type=int, default=500)
    ap.add_argument("--val-log-every", type=int, default=50)
    ap.add_argument(
        "--use-hard-bootstrap", action="store_true",
        help="Stage 1 의 KD teacher pred 가 noisy 라 hard-bootstrap loss 적용 "
             f"(λ={cfg.HARD_BOOTSTRAP_LAMBDA})",
    )
    ap.add_argument(
        "--class-weight-mode", choices=["sqrt_inv", "median_inv"], default="sqrt_inv",
        help="DSCNN 원본 (median/freq) 도 옵션 (PLAN_v4 §6.E)",
    )
    ap.add_argument("--ema-decay", type=float, default=cfg.EMA_DECAY)
    ap.add_argument("--no-ema", action="store_true", help="EMA off (debug)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--run-name", type=str, default=None)
    args = ap.parse_args()

    if args.quick:
        args.epochs = 2
        args.batch_size = 1
        args.img_size = 224
        args.num_workers = 0

    if args.run_name is None:
        args.run_name = f"{args.backbone}_dpt_dual_sz{args.img_size}_11cls_v4"

    rank, world_size, local_rank = init_distributed()
    log = setup_logger(rank=rank, name="stage1_v4")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main(rank):
        log.info(f"=== v4 stage1 KD pretrain ===  world_size={world_size}  run_name={args.run_name}")
        log.info(f"args: {vars(args)}")
        log.info(f"classes: {cfg.ORNL_CLASS_NAMES_V4}")

    # ----- Checkpoint paths -----
    run_ckpt_dir = cfg.CHECKPOINT_DIR / args.run_name
    if is_main(rank):
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    ckpt_path = run_ckpt_dir / "stage1_best.pt"

    # ----- Model (n_classes=8 자동) -----
    model = DefSegModel(backbone_name=args.backbone, n_classes=cfg.N_CLASSES_V4).to(device)
    new_size = round_to_patch(args.img_size, model.patch_size)
    if new_size != args.img_size:
        if is_main(rank):
            log.warning(f"img_size {args.img_size} → {new_size}")
        args.img_size = new_size

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    n_train_params = sum(p.numel() for p in unwrap(model).trainable_parameters())
    if is_main(rank):
        log.info(f"device={device}  img_size={args.img_size}  bs_per_gpu={args.batch_size}  "
                 f"trainable={n_train_params/1e6:.2f}M  n_classes={cfg.N_CLASSES_V4}")

    # ----- EMA -----
    use_ema = not args.no_ema
    ema_model = None
    if use_ema:
        ema_model = AveragedModel(
            unwrap(model),
            multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay),
        )
        if is_main(rank):
            log.info(f"EMA enabled (decay={args.ema_decay})")

    # ----- v4 Dataset (image v1 cache + label v4 cache + Copy-Paste S1 augmentation) -----
    # Copy-Paste S1 supplier
    from ..data.copy_paste import CopyPasteSampler, load_supplier_index
    supplier_path = cfg.v4_cache_dir(args.img_size) / "rare_class_supplier.json"
    cp_sampler_s1 = None
    if cfg.CP_ENABLE and supplier_path.exists():
        items_all = load_supplier_index(supplier_path)
        # Stage 1 supplier 만 추출
        items_s1 = {c: items_all[c] for c in cfg.CP_RARE_CLASSES_S1 if c in items_all}
        # 일단 dataset 만들고 fetch_fn 등록
        train_ds_tmp = DefSegORNLCachedDatasetV4("train", args.img_size, training=False)
        def _fetch_fn(key, _ds=train_ds_tmp):
            build_id, row = key
            return _ds.fetch_by_build_row(build_id, row)
        cp_sampler_s1 = CopyPasteSampler(
            items_per_class=items_s1, fetch_fn=_fetch_fn,
            rare_classes=cfg.CP_RARE_CLASSES_S1, prob=cfg.CP_PROB,
        )
        if is_main(rank):
            log.info(f"Copy-Paste S1 supplier: {[(c, len(items_s1.get(c, []))) for c in cfg.CP_RARE_CLASSES_S1]}")
    elif is_main(rank):
        log.warning(f"rare_class_supplier.json 없음 ({supplier_path}). Copy-Paste 비활성.")

    train_ds = DefSegORNLCachedDatasetV4("train", args.img_size, training=True, cp_sampler=cp_sampler_s1)
    val_ds = DefSegORNLCachedDatasetV4("val", args.img_size, training=False)
    if is_main(rank):
        log.info(f"train={len(train_ds)} layers, val={len(val_ds)} layers (v4 cache)")

    # ----- Train sampler: class-aware (PLAN_v4 §4) -----
    from ..data.sampler_helpers import compute_layer_weights
    sampler_weights = compute_layer_weights(
        train_ds.pix_per_layer, alpha=args.class_aware_alpha, eps=cfg.CLASS_AWARE_EPS,
    )
    if is_main(rank):
        log.info(
            f"class-aware sampler: α={args.class_aware_alpha}, "
            f"layer_w stats min={sampler_weights.min():.2e} max={sampler_weights.max():.2e} "
            f"mean={sampler_weights.mean():.2e}"
        )
    train_sampler = DistributedWeightedSampler(
        weights=sampler_weights,
        num_samples_total=len(train_ds) if not args.quick else 8 * world_size,
        num_replicas=world_size, rank=rank, replacement=True,
    )
    val_sampler = DistributedSampler(
        val_ds, num_replicas=world_size, rank=rank, shuffle=False,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    # ----- Class weights (v4 11-class 기반) -----
    if is_main(rank):
        log.info(f"estimating class weights from {args.n_weight_sample} sampled layers …")
    counts = estimate_class_counts_v4(train_ds, n_sample=args.n_weight_sample)
    if args.class_weight_mode == "median_inv":
        class_weights = median_inv_class_weight(
            counts, clip=cfg.S1_CLASS_WEIGHT_CLIP,
        ).to(device)
    else:
        class_weights = sqrt_inv_class_weight(
            counts, clip=cfg.S1_CLASS_WEIGHT_CLIP,
        ).to(device)
    if is_main(rank):
        log.info(f"class counts: {counts.tolist()}")
        log.info(
            f"class weights (mode={args.class_weight_mode}, clip={cfg.S1_CLASS_WEIGHT_CLIP}): "
            f"{class_weights.cpu().numpy().round(3).tolist()}"
        )

    # ----- Optim + Warmup + Cosine -----
    optim = torch.optim.AdamW(
        unwrap(model).trainable_parameters(),
        lr=args.lr, weight_decay=cfg.S1_WEIGHT_DECAY,
    )
    total_steps = args.epochs * max(len(train_loader), 1)
    warmup_steps = min(cfg.S1_WARMUP_STEPS, total_steps // 4)
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    if is_main(rank):
        log.info(
            f"AMP=OFF (FP32)  warmup={warmup_steps} steps  "
            f"grad_clip={cfg.S1_GRAD_CLIP_NORM}  total_steps={total_steps}  "
            f"loss={'hard_bootstrap' if args.use_hard_bootstrap else 'focal'}"
        )

    best_val = -1.0
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        if isinstance(val_sampler, DistributedSampler):
            val_sampler.set_epoch(epoch)

        # ----- train -----
        model.train(); unwrap(model).backbone.eval()
        t_ep = time.time()
        train_loss_sum = torch.zeros(1, device=device)
        train_n = torch.zeros(1, device=device)
        train_correct = torch.zeros(1, device=device)
        train_total = torch.zeros(1, device=device)

        for step, batch in enumerate(train_loader):
            img0 = batch["img0"].to(device, non_blocking=True)
            img1 = batch["img1"].to(device, non_blocking=True)
            label = batch["label"].to(device, non_blocking=True)

            optim.zero_grad(set_to_none=True)
            logits = model(img0, img1)
            if args.use_hard_bootstrap:
                loss = hard_bootstrap_loss(
                    logits, label,
                    lambda_trust=cfg.HARD_BOOTSTRAP_LAMBDA,
                    alpha_weight=class_weights,
                    ignore_index=cfg.IGNORE_INDEX,
                )
            else:
                loss = focal_loss(
                    logits, label,
                    gamma=args.gamma, alpha_weight=class_weights,
                    ignore_index=cfg.IGNORE_INDEX,
                )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                unwrap(model).trainable_parameters(),
                max_norm=cfg.S1_GRAD_CLIP_NORM,
            )
            optim.step()
            sched.step()
            if use_ema:
                ema_model.update_parameters(unwrap(model))

            train_loss_sum += loss.detach()
            train_n += 1
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                valid = label != cfg.IGNORE_INDEX
                train_correct += (pred[valid] == label[valid]).sum()
                train_total += valid.sum()

            if is_main(rank) and step % 20 == 0:
                lr_now = sched.get_last_lr()[0]
                log.info(
                    f"e{epoch:02d} train [{step:4d}/{len(train_loader)}] "
                    f"loss={loss.item():.4f}  lr={lr_now:.2e}  gnorm={grad_norm.item():.3f}"
                )

        reduce_counts(world_size, train_loss_sum, train_n, train_correct, train_total)
        train_loss_avg = (train_loss_sum / train_n.clamp(min=1)).item()
        train_acc = (train_correct / train_total.clamp(min=1)).item()

        # ----- val (EMA model 사용) -----
        eval_model = ema_model if use_ema else unwrap(model)
        eval_model.eval()
        val_correct = torch.zeros(1, device=device)
        val_total = torch.zeros(1, device=device)
        val_inter = torch.zeros(cfg.N_CLASSES_V4, device=device)
        val_union = torch.zeros(cfg.N_CLASSES_V4, device=device)
        t_val = time.time()

        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                img0 = batch["img0"].to(device, non_blocking=True)
                img1 = batch["img1"].to(device, non_blocking=True)
                label = batch["label"].to(device, non_blocking=True)
                if use_ema:
                    # AveragedModel forward 는 input 1개씩 받는 단일 forward
                    logits = ema_model.module(img0, img1)
                else:
                    logits = eval_model(img0, img1)
                update_counts_v4(logits, label, val_correct, val_total, val_inter, val_union)
                if is_main(rank) and step % args.val_log_every == 0:
                    log.info(f"e{epoch:02d} val   [{step:4d}/{len(val_loader)}] running")

        reduce_counts(world_size, val_correct, val_total, val_inter, val_union)
        val_acc = (val_correct / val_total.clamp(min=1)).item()
        per_cls_iou = []
        for c in range(cfg.N_CLASSES_V4):
            u = val_union[c].item()
            per_cls_iou.append(val_inter[c].item() / u if u > 0 else float("nan"))
        valid_ious = [x for x in per_cls_iou if x == x]
        miou = sum(valid_ious) / max(len(valid_ious), 1)

        if is_main(rank):
            log.info(
                f"e{epoch:02d} SUMMARY  train_loss={train_loss_avg:.4f}  "
                f"train_acc={train_acc:.4f}  val_acc={val_acc:.4f}  mIoU={miou:.4f}  "
                f"(epoch {(time.time()-t_ep)/60:.1f}m, val {(time.time()-t_val)/60:.1f}m)"
            )
            for c, iou in enumerate(per_cls_iou):
                tag = f"{c} {cfg.ORNL_CLASS_NAMES_V4[c]:<22s}"
                line = f"  iou {tag} = {iou:.4f}" if iou == iou else f"  iou {tag} = n/a"
                log.info(line)

            if val_acc > best_val:
                best_val = val_acc
                save_obj = unwrap(model) if not use_ema else ema_model.module
                torch.save({
                    "model_state": save_obj.trainable_state_dict(),
                    "config": {
                        "version": "v3",
                        "backbone": args.backbone,
                        "n_classes": cfg.N_CLASSES_V4,
                        "img_size": args.img_size,
                        "intermediate_layers": list(cfg.INTERMEDIATE_LAYERS),
                        "decoder_channels": cfg.DECODER_CHANNELS,
                        "use_hard_bootstrap": args.use_hard_bootstrap,
                        "class_weight_mode": args.class_weight_mode,
                        "ema_decay": args.ema_decay if use_ema else None,
                    },
                    "epoch": epoch, "val_acc": val_acc, "miou": miou,
                }, ckpt_path)
                log.info(f"↑ saved best (EMA={use_ema}) to {ckpt_path}")

        if world_size > 1:
            dist.barrier()

    if is_main(rank):
        log.info(f"done. best val_acc={best_val:.4f}")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
