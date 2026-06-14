"""v2 Stage 2 학습 — DSCNN_Dataset finetune, 8-class, **Cross-Validation**.

PLAN_v2 §5.2 참조. Stage 1 best 로 init, 표준 CE + EMA.

8-fold leave-one-source-out:
  fold k → val=sources[k], train=나머지 7 source

사용 (single GPU):
    python -m DefSeg_AM.v2.training.train_stage2 --fold 0
    python -m DefSeg_AM.v2.training.train_stage2 --fold 0 --quick
    # 모든 fold 순회 (loop) — scripts/run_stage2_v2_all.sh

DDP 도 지원하지만, Stage 2 는 데이터가 작아서 1 GPU 권장.
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

# 공통 모듈 재사용 (common)
from ...common.models.losses import sqrt_inv_class_weight
from ...common.models.model import DefSegModel, round_to_patch
from ...common.training.dist_utils import (
    init_distributed, is_main, unwrap, reduce_counts,
)
from ...common.utils.log import setup_logger

# v2 모듈
from .. import config_v2 as cfg
from ..data.data_dscnn_v2 import (
    DefSegDSCNNDatasetV2,
    compute_class_counts_v2,
    split_train_val_by_fold,
)
from ..models.losses_v2 import median_inv_class_weight
from .train_stage1 import update_counts_v2


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--fold", type=int, required=True,
        help=f"Cross-val fold id ∈ [0, {cfg.N_FOLDS}). "
             f"Available sources: {cfg.DSCNN_CV_SOURCE_NAMES}",
    )
    ap.add_argument("--epochs", type=int, default=cfg.S2_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=cfg.S2_BATCH_SIZE)
    ap.add_argument("--img-size", type=int, default=cfg.IMG_SIZE)
    ap.add_argument("--lr", type=float, default=cfg.S2_LR)
    ap.add_argument("--backbone", type=str, default=cfg.DINO_BACKBONE)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--init-from", type=str, default=None,
                    help="명시적 ckpt 경로 (default: <run_ckpt_dir>/stage1_best.pt)")
    ap.add_argument("--no-init", action="store_true",
                    help="Stage 1 load 없이 random init (ablation: S2-only)")
    ap.add_argument(
        "--class-weight-mode", choices=["sqrt_inv", "median_inv"], default="sqrt_inv",
    )
    ap.add_argument("--ema-decay", type=float, default=cfg.EMA_DECAY)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--val-log-every", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if not (0 <= args.fold < cfg.N_FOLDS):
        raise ValueError(f"--fold must be in [0, {cfg.N_FOLDS})")

    if args.quick:
        args.epochs = 2
        args.batch_size = 1
        args.img_size = 224
        args.num_workers = 0

    if args.run_name is None:
        args.run_name = f"{args.backbone}_dpt_dual_sz{args.img_size}_8cls_v2"

    rank, world_size, local_rank = init_distributed()
    log = setup_logger(rank=rank, name=f"stage2_v2_f{args.fold}")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    val_source_name = cfg.DSCNN_CV_SOURCE_NAMES[args.fold]
    if is_main(rank):
        log.info(
            f"=== v2 stage2 CV ===  fold={args.fold}/{cfg.N_FOLDS}  "
            f"val_source={val_source_name}  run_name={args.run_name}"
        )

    # ----- ckpt paths -----
    run_ckpt_dir = cfg.CHECKPOINT_DIR / args.run_name
    if is_main(rank):
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    ckpt_path = run_ckpt_dir / f"stage2_best_fold{args.fold}_{val_source_name}.pt"
    init_path = args.init_from or str(run_ckpt_dir / "stage1_best.pt")

    # ----- Model + Stage 1 load -----
    model = DefSegModel(backbone_name=args.backbone, n_classes=cfg.N_CLASSES_V2).to(device)
    new_size = round_to_patch(args.img_size, model.patch_size)
    if new_size != args.img_size:
        if is_main(rank):
            log.warning(f"img_size {args.img_size} → {new_size}")
        args.img_size = new_size

    if not args.no_init:
        if not Path(init_path).exists():
            raise FileNotFoundError(
                f"Stage 1 ckpt not found: {init_path}\n"
                "Run `python -m DefSeg_AM.v2.training.train_stage1` first or pass --no-init."
            )
        sd = torch.load(init_path, map_location=device, weights_only=False)
        # config sanity (n_classes 일치 확인)
        ckpt_cfg = sd.get("config", {})
        if ckpt_cfg.get("n_classes") not in (None, cfg.N_CLASSES_V2):
            log.warning(
                f"Stage 1 ckpt n_classes={ckpt_cfg.get('n_classes')} ≠ v2 {cfg.N_CLASSES_V2}"
            )
        model.load_trainable_state_dict(sd["model_state"])
        if is_main(rank):
            log.info(f"loaded stage1 ckpt: {init_path}  val_acc={sd.get('val_acc', '?')}")
    else:
        if is_main(rank):
            log.warning("training from scratch (--no-init)")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    n_train_params = sum(p.numel() for p in unwrap(model).trainable_parameters())
    if is_main(rank):
        log.info(
            f"device={device}  img_size={args.img_size}  bs_per_gpu={args.batch_size}  "
            f"trainable={n_train_params/1e6:.2f}M  n_classes={cfg.N_CLASSES_V2}"
        )

    # ----- EMA -----
    use_ema = not args.no_ema
    ema_model = None
    if use_ema:
        ema_model = AveragedModel(
            unwrap(model),
            multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay),
        )

    # ----- Cross-val data split -----
    train_specs, val_specs, _ = split_train_val_by_fold(args.fold)
    if is_main(rank):
        log.info(
            f"train={len(train_specs)} layers (7 sources), "
            f"val={len(val_specs)} layers ({val_source_name})"
        )

    train_ds = DefSegDSCNNDatasetV2(train_specs, img_size=args.img_size, training=True)
    val_ds = DefSegDSCNNDatasetV2(val_specs, img_size=args.img_size, training=False)

    if world_size > 1:
        train_sampler = DistributedSampler(train_ds, num_replicas=world_size, rank=rank, shuffle=True)
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        train_sampler = None
        val_sampler = None

    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size,
        sampler=train_sampler, shuffle=(train_sampler is None),
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    # ----- Class weights -----
    if is_main(rank):
        log.info(f"computing class weights from {len(train_specs)} train layers …")
    counts = compute_class_counts_v2(train_specs)
    if args.class_weight_mode == "median_inv":
        class_weights = median_inv_class_weight(counts, clip=cfg.S2_CLASS_WEIGHT_CLIP).to(device)
    else:
        class_weights = sqrt_inv_class_weight(counts, clip=cfg.S2_CLASS_WEIGHT_CLIP).to(device)
    if is_main(rank):
        log.info(f"class counts: {counts.tolist()}")
        log.info(
            f"class weights (mode={args.class_weight_mode}, clip={cfg.S2_CLASS_WEIGHT_CLIP}): "
            f"{class_weights.cpu().numpy().round(3).tolist()}"
        )

    # ----- Optim + Warmup + Cosine -----
    optim = torch.optim.AdamW(
        unwrap(model).trainable_parameters(),
        lr=args.lr, weight_decay=cfg.S2_WEIGHT_DECAY,
    )
    total_steps = args.epochs * max(len(train_loader), 1)
    warmup_steps = min(cfg.S2_WARMUP_STEPS, total_steps // 4)
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    if is_main(rank):
        log.info(
            f"AMP=OFF  warmup={warmup_steps}  grad_clip={cfg.S2_GRAD_CLIP_NORM}  "
            f"total_steps={total_steps}"
        )

    best_val = -1.0
    best_per_class = None
    best_epoch = -1
    for epoch in range(args.epochs):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        if val_sampler is not None and hasattr(val_sampler, "set_epoch"):
            val_sampler.set_epoch(epoch)

        # train
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
            loss = F.cross_entropy(
                logits, label,
                weight=class_weights, ignore_index=cfg.IGNORE_INDEX,
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                unwrap(model).trainable_parameters(),
                max_norm=cfg.S2_GRAD_CLIP_NORM,
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

            if is_main(rank) and step % 5 == 0:
                lr_now = sched.get_last_lr()[0]
                log.info(
                    f"e{epoch:02d} train [{step:3d}/{len(train_loader)}] "
                    f"loss={loss.item():.4f}  lr={lr_now:.2e}  gnorm={grad_norm.item():.3f}"
                )

        reduce_counts(world_size, train_loss_sum, train_n, train_correct, train_total)
        train_loss_avg = (train_loss_sum / train_n.clamp(min=1)).item()
        train_acc = (train_correct / train_total.clamp(min=1)).item()

        # val
        eval_model = ema_model if use_ema else unwrap(model)
        eval_model.eval()
        val_correct = torch.zeros(1, device=device)
        val_total = torch.zeros(1, device=device)
        val_inter = torch.zeros(cfg.N_CLASSES_V2, device=device)
        val_union = torch.zeros(cfg.N_CLASSES_V2, device=device)
        t_val = time.time()

        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                img0 = batch["img0"].to(device, non_blocking=True)
                img1 = batch["img1"].to(device, non_blocking=True)
                label = batch["label"].to(device, non_blocking=True)
                if use_ema:
                    logits = ema_model.module(img0, img1)
                else:
                    logits = eval_model(img0, img1)
                update_counts_v2(logits, label, val_correct, val_total, val_inter, val_union)
                if is_main(rank) and step % args.val_log_every == 0:
                    log.info(f"e{epoch:02d} val   [{step:3d}/{len(val_loader)}] running")

        reduce_counts(world_size, val_correct, val_total, val_inter, val_union)
        val_acc = (val_correct / val_total.clamp(min=1)).item()
        per_cls_iou = []
        for c in range(cfg.N_CLASSES_V2):
            u = val_union[c].item()
            per_cls_iou.append(val_inter[c].item() / u if u > 0 else float("nan"))
        valid_ious = [x for x in per_cls_iou if x == x]
        miou = sum(valid_ious) / max(len(valid_ious), 1)

        if is_main(rank):
            log.info(
                f"e{epoch:02d} SUMMARY  fold={args.fold}  val_src={val_source_name}  "
                f"train_loss={train_loss_avg:.4f}  train_acc={train_acc:.4f}  "
                f"val_acc={val_acc:.4f}  mIoU={miou:.4f}  "
                f"(epoch {(time.time()-t_ep)/60:.1f}m, val {(time.time()-t_val)/60:.1f}m)"
            )
            for c, iou in enumerate(per_cls_iou):
                tag = f"{c} {cfg.ORNL_CLASS_NAMES_V2[c]:<22s}"
                line = f"  iou {tag} = {iou:.4f}" if iou == iou else f"  iou {tag} = n/a"
                log.info(line)

            if val_acc > best_val:
                best_val = val_acc
                best_per_class = per_cls_iou
                best_epoch = epoch
                save_obj = unwrap(model) if not use_ema else ema_model.module
                torch.save({
                    "model_state": save_obj.trainable_state_dict(),
                    "config": {
                        "version": "v2",
                        "fold": args.fold,
                        "val_source": val_source_name,
                        "backbone": args.backbone,
                        "n_classes": cfg.N_CLASSES_V2,
                        "img_size": args.img_size,
                        "intermediate_layers": list(cfg.INTERMEDIATE_LAYERS),
                        "decoder_channels": cfg.DECODER_CHANNELS,
                        "class_weight_mode": args.class_weight_mode,
                        "ema_decay": args.ema_decay if use_ema else None,
                        "init_from": init_path if not args.no_init else None,
                    },
                    "epoch": epoch, "val_acc": val_acc, "miou": miou,
                    "per_class_iou": per_cls_iou,
                }, ckpt_path)
                log.info(f"↑ saved best (EMA={use_ema}) to {ckpt_path}")

        if world_size > 1:
            dist.barrier()

    # cv_summary 갱신 (rank 0 만)
    if is_main(rank):
        summary_path = run_ckpt_dir / cfg.CV_SUMMARY_FILE
        summary = {}
        if summary_path.exists():
            with open(summary_path) as f:
                summary = json.load(f)
        summary[f"fold_{args.fold}"] = {
            "val_source": val_source_name,
            "best_val_acc": best_val,
            "best_epoch": best_epoch,
            "best_per_class_iou": best_per_class,
        }
        with open(summary_path, "w") as f:
            json.dump(summary, f, indent=2)
        log.info(f"cv_summary updated: {summary_path}")
        log.info(f"fold {args.fold} done. best val_acc={best_val:.4f}")

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
