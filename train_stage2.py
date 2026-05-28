"""Stage 2 학습 — DSCNN_Dataset human GT 로 finetune. DDP 멀티 GPU.

PLAN.md §5.2 참조. Stage 1 best 로 init, 표준 CE.

사용 (4 GPU torchrun):
    torchrun --standalone --nproc-per-node=4 -m DefSeg_AM.train_stage2 \\
        --run-name vits14_dpt_dual_sz1036 [--quick] [--no-init]
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from . import config
from .data_dscnn import (
    DefSegDSCNNDataset,
    compute_class_counts,
    split_train_val,
)
from .log import setup_logger
from .losses import sqrt_inv_class_weight
from .model import DefSegModel, round_to_patch
from .train_stage1 import (
    init_distributed, is_main, unwrap,
    update_counts, reduce_counts,
)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=config.S2_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.S2_BATCH_SIZE)
    ap.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    ap.add_argument("--lr", type=float, default=config.S2_LR)
    ap.add_argument("--backbone", type=str, default=config.DINO_BACKBONE)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--run-name", type=str, default=None,
                    help="Stage 1 과 동일 run_name (stage1_best.pt 자동 로드)")
    ap.add_argument("--init-from", type=str, default=None,
                    help="명시적 ckpt 경로")
    ap.add_argument("--no-init", action="store_true",
                    help="Stage 1 load 없이 random init (ablation: S2-only)")
    ap.add_argument("--val-log-every", type=int, default=5)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.epochs = 2
        args.batch_size = 1
        args.img_size = 224
        args.num_workers = 0

    if args.run_name is None:
        args.run_name = f"{args.backbone}_dpt_dual_sz{args.img_size}"

    rank, world_size, local_rank = init_distributed()
    log = setup_logger(rank=rank, name="stage2")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main(rank):
        log.info(f"=== stage2 GT finetune ===  world_size={world_size}  run_name={args.run_name}")
        log.info(f"args: {vars(args)}")

    run_ckpt_dir = config.CHECKPOINT_DIR / args.run_name
    if is_main(rank):
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    ckpt_path = run_ckpt_dir / "stage2_best.pt"
    init_path = args.init_from or str(run_ckpt_dir / "stage1_best.pt")

    # ----- Model + Stage 1 load -----
    model = DefSegModel(backbone_name=args.backbone).to(device)
    new_size = round_to_patch(args.img_size, model.patch_size)
    if new_size != args.img_size:
        if is_main(rank):
            log.warning(f"img_size {args.img_size} → {new_size}")
        args.img_size = new_size

    if not args.no_init:
        if not Path(init_path).exists():
            raise FileNotFoundError(
                f"Stage 1 ckpt not found: {init_path}\n"
                "Run stage1 first or pass --no-init."
            )
        sd = torch.load(init_path, map_location=device, weights_only=False)
        model.load_trainable_state_dict(sd["model_state"])
        if is_main(rank):
            log.info(f"loaded stage1 ckpt: {init_path}  val_acc={sd.get('val_acc', '?')}")
    else:
        if is_main(rank):
            log.warning("training from scratch (--no-init)")

    if world_size > 1:
        # fusion_blocks[3].res_skip 미사용 → find_unused 필요 (train_stage1 과 동일)
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    n_train_params = sum(p.numel() for p in unwrap(model).trainable_parameters())
    if is_main(rank):
        log.info(f"device={device}  img_size={args.img_size}  bs_per_gpu={args.batch_size}  "
                 f"trainable={n_train_params/1e6:.2f}M")

    # ----- DSCNN data -----
    train_specs, val_specs = split_train_val()
    if is_main(rank):
        log.info(f"train={len(train_specs)} layers, val={len(val_specs)} layers (DSCNN_Dataset)")

    train_ds = DefSegDSCNNDataset(train_specs, img_size=args.img_size, training=True)
    val_ds = DefSegDSCNNDataset(val_specs, img_size=args.img_size, training=False)

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
    counts = compute_class_counts(train_specs)
    class_weights = sqrt_inv_class_weight(counts, clip=config.S2_CLASS_WEIGHT_CLIP).to(device)
    if is_main(rank):
        log.info(f"class counts: {counts.tolist()}")
        log.info(f"class weights (clip={config.S2_CLASS_WEIGHT_CLIP}): "
                 f"{class_weights.cpu().numpy().round(3).tolist()}")

    # ----- Optim + Warmup + Cosine scheduler -----
    optim = torch.optim.AdamW(
        unwrap(model).trainable_parameters(),
        lr=args.lr, weight_decay=config.S2_WEIGHT_DECAY,
    )
    total_steps = args.epochs * max(len(train_loader), 1)
    warmup_steps = min(config.S2_WARMUP_STEPS, total_steps // 4)
    import math
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    scaler = torch.amp.GradScaler("cuda", enabled=False)  # AMP off (stability)
    if is_main(rank):
        log.info(f"AMP=OFF (FP32)  warmup={warmup_steps} steps  "
                 f"grad_clip={config.S2_GRAD_CLIP_NORM}  total_steps={total_steps}")

    best_val = -1.0
    for epoch in range(args.epochs):
        if train_sampler is not None and hasattr(train_sampler, "set_epoch"):
            train_sampler.set_epoch(epoch)
        if val_sampler is not None and hasattr(val_sampler, "set_epoch"):
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
            loss = F.cross_entropy(
                logits, label,
                weight=class_weights, ignore_index=config.IGNORE_INDEX,
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                unwrap(model).trainable_parameters(),
                max_norm=config.S2_GRAD_CLIP_NORM,
            )
            optim.step()
            sched.step()

            train_loss_sum += loss.detach()
            train_n += 1
            with torch.no_grad():
                pred = logits.argmax(dim=1)
                valid = label != config.IGNORE_INDEX
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

        # ----- val -----
        model.eval()
        val_correct = torch.zeros(1, device=device)
        val_total = torch.zeros(1, device=device)
        val_inter = torch.zeros(config.N_CLASSES, device=device)
        val_union = torch.zeros(config.N_CLASSES, device=device)
        t_val = time.time()

        with torch.no_grad():
            for step, batch in enumerate(val_loader):
                img0 = batch["img0"].to(device, non_blocking=True)
                img1 = batch["img1"].to(device, non_blocking=True)
                label = batch["label"].to(device, non_blocking=True)
                logits = model(img0, img1)
                update_counts(logits, label, val_correct, val_total, val_inter, val_union)
                if is_main(rank) and step % args.val_log_every == 0:
                    log.info(f"e{epoch:02d} val   [{step:3d}/{len(val_loader)}] running")

        reduce_counts(world_size, val_correct, val_total, val_inter, val_union)
        val_acc = (val_correct / val_total.clamp(min=1)).item()
        per_cls_iou = []
        for c in range(config.N_CLASSES):
            u = val_union[c].item()
            per_cls_iou.append(val_inter[c].item() / u if u > 0 else float("nan"))
        valid_ious = [x for x in per_cls_iou if x == x]
        miou = sum(valid_ious) / max(len(valid_ious), 1)

        if is_main(rank):
            ep_time = time.time() - t_ep
            val_time = time.time() - t_val
            log.info(
                f"e{epoch:02d} SUMMARY  train_loss={train_loss_avg:.4f}  train_acc={train_acc:.4f}  "
                f"val_acc={val_acc:.4f}  mIoU={miou:.4f}  "
                f"(epoch {ep_time/60:.1f}m, val {val_time/60:.1f}m)"
            )
            for c, iou in enumerate(per_cls_iou):
                tag = f"{c:2d} {config.ORNL_CLASS_NAMES[c]:<22s}"
                line = f"  iou {tag} = {iou:.4f}" if iou == iou else f"  iou {tag} = n/a"
                log.info(line)

            if val_acc > best_val:
                best_val = val_acc
                torch.save({
                    "model_state": unwrap(model).trainable_state_dict(),
                    "config": {
                        "backbone": args.backbone,
                        "n_classes": config.N_CLASSES,
                        "img_size": args.img_size,
                        "intermediate_layers": list(config.INTERMEDIATE_LAYERS),
                        "decoder_channels": config.DECODER_CHANNELS,
                        "init_from": init_path if not args.no_init else None,
                    },
                    "epoch": epoch, "val_acc": val_acc, "miou": miou,
                }, ckpt_path)
                log.info(f"↑ saved best to {ckpt_path}")

        if world_size > 1:
            dist.barrier()

    if is_main(rank):
        log.info(f"done. best val_acc={best_val:.4f}")
    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
