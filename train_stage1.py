"""Stage 1 학습 — ORNL HDF5 (DSCNN pred) 으로 KD pretrain. DDP 멀티 GPU.

PLAN.md §5.1 참조.

사전조건:
    python -m DefSeg_AM.build_cache_stage1     # 사전 resize+uint8 cache 빌드 (1회)

사용 (4 GPU torchrun):
    torchrun --standalone --nproc-per-node=4 -m DefSeg_AM.train_stage1 \\
        --epochs 30 --batch-size 2 --img-size 1036 \\
        --run-name vits14_dpt_dual_sz1036 [--quick]
"""
from __future__ import annotations

import argparse
import os
import time
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils.data import DataLoader
from torch.utils.data.distributed import DistributedSampler

from . import config
from .build_cache_stage1 import cache_dir_for
from .data_ornl import DefSegORNLCachedDataset, estimate_class_counts_cached
from .log import setup_logger
from .losses import focal_loss, sqrt_inv_class_weight
from .model import DefSegModel, round_to_patch
from .samplers import DistributedWeightedSampler


# ---------------------------------------------------------------------------
# DDP helpers
# ---------------------------------------------------------------------------

def init_distributed() -> tuple[int, int, int]:
    """torchrun 환경이면 DDP init, 아니면 single-process. Returns (rank, world_size, local_rank)."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        dist.init_process_group(backend="nccl")
        rank = dist.get_rank()
        world_size = dist.get_world_size()
        local_rank = int(os.environ.get("LOCAL_RANK", 0))
        torch.cuda.set_device(local_rank)
        return rank, world_size, local_rank
    return 0, 1, 0


def is_main(rank: int) -> bool:
    return rank == 0


def unwrap(m):
    return m.module if hasattr(m, "module") else m


# ---------------------------------------------------------------------------
# Metric accumulators (count-based for IoU correctness; DDP-friendly all_reduce)
# ---------------------------------------------------------------------------

@torch.no_grad()
def update_counts(
    logits: torch.Tensor, label: torch.Tensor,
    correct: torch.Tensor, total: torch.Tensor,
    inter: torch.Tensor, union: torch.Tensor,
) -> None:
    """모든 텐서는 device 상의 1-d 누적 tensor. label/pred 는 device 상."""
    pred = logits.argmax(dim=1)
    valid = label != config.IGNORE_INDEX
    correct += (pred[valid] == label[valid]).sum()
    total += valid.sum()
    for c in range(config.N_CLASSES):
        p = (pred == c) & valid
        g = (label == c) & valid
        inter[c] += (p & g).sum()
        union[c] += (p | g).sum()


def reduce_counts(world_size: int, *tensors: torch.Tensor) -> None:
    if world_size > 1:
        for t in tensors:
            dist.all_reduce(t, op=dist.ReduceOp.SUM)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--epochs", type=int, default=config.S1_EPOCHS)
    ap.add_argument("--batch-size", type=int, default=config.S1_BATCH_SIZE,
                    help="per-GPU batch size (effective = batch_size × world_size)")
    ap.add_argument("--img-size", type=int, default=config.IMG_SIZE)
    ap.add_argument("--lr", type=float, default=config.S1_LR)
    ap.add_argument("--backbone", type=str, default=config.DINO_BACKBONE)
    ap.add_argument("--num-workers", type=int, default=8,
                    help="DataLoader workers per rank (cache memmap I/O 빠름 → 늘려도 OK)")
    ap.add_argument("--gamma", type=float, default=config.S1_FOCAL_GAMMA)
    ap.add_argument("--oversample-power", type=float, default=config.S1_OVERSAMPLE_POWER)
    ap.add_argument("--n-weight-sample", type=int, default=500,
                    help="class weight 추정용 sub-sample layer 수 (cache 라 빠름)")
    ap.add_argument("--val-log-every", type=int, default=50,
                    help="val 진행 print 간격 (batch 단위)")
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--run-name", type=str, default=None)
    args = ap.parse_args()

    if args.quick:
        args.epochs = 2
        args.batch_size = 1
        args.img_size = 224
        args.num_workers = 0

    if args.run_name is None:
        args.run_name = f"{args.backbone}_dpt_dual_sz{args.img_size}"

    # ----- DDP init -----
    rank, world_size, local_rank = init_distributed()
    log = setup_logger(rank=rank, name="stage1")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main(rank):
        log.info(f"=== stage1 KD pretrain ===  world_size={world_size}  run_name={args.run_name}")
        log.info(f"args: {vars(args)}")

    # ----- Checkpoint paths -----
    run_ckpt_dir = config.CHECKPOINT_DIR / args.run_name
    if is_main(rank):
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    ckpt_path = run_ckpt_dir / "stage1_best.pt"

    # ----- Model -----
    model = DefSegModel(backbone_name=args.backbone).to(device)
    new_size = round_to_patch(args.img_size, model.patch_size)
    if new_size != args.img_size:
        if is_main(rank):
            log.warning(f"img_size {args.img_size} → {new_size} (patch {model.patch_size} 의 배수)")
        args.img_size = new_size

    if world_size > 1:
        # backbone 은 requires_grad=False → DDP 자동 skip.
        # fusion_blocks[3] 의 res_skip 은 forward 에서 skip 인자 없이 호출 → 미사용 → True 필요.
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    n_train_params = sum(p.numel() for p in unwrap(model).trainable_parameters())
    if is_main(rank):
        log.info(f"device={device}  img_size={args.img_size}  bs_per_gpu={args.batch_size}  "
                 f"trainable={n_train_params/1e6:.2f}M")

    # ----- Cache load -----
    cache_root = cache_dir_for(args.img_size)
    train_ds = DefSegORNLCachedDataset(cache_root, "train", args.img_size, training=True)
    val_ds = DefSegORNLCachedDataset(cache_root, "val", args.img_size, training=False)
    if is_main(rank):
        log.info(f"train={len(train_ds)} layers, val={len(val_ds)} layers (from cache)")

    if args.quick:
        # quick mode: dataset 자체는 풀 사이즈, sampler 가 N 만 뽑게 함 (cache 안 줄어들도록)
        pass

    # ----- Train sampler: DistributedWeightedSampler (defect-ratio oversampling) -----
    sampler_weights = train_ds.defect_ratios ** args.oversample_power + config.S1_OVERSAMPLE_EPS
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

    # ----- Class weights (cache 기반, 빠름) -----
    if is_main(rank):
        log.info(f"estimating class weights from {args.n_weight_sample} sampled layers …")
    counts = estimate_class_counts_cached(train_ds, n_sample=args.n_weight_sample)
    class_weights = sqrt_inv_class_weight(counts, clip=config.S1_CLASS_WEIGHT_CLIP).to(device)
    if is_main(rank):
        log.info(f"class counts: {counts.tolist()}")
        log.info(f"class weights (clip={config.S1_CLASS_WEIGHT_CLIP}): "
                 f"{class_weights.cpu().numpy().round(3).tolist()}")

    # ----- Optim + Warmup + Cosine scheduler -----
    optim = torch.optim.AdamW(
        unwrap(model).trainable_parameters(),
        lr=args.lr, weight_decay=config.S1_WEIGHT_DECAY,
    )
    total_steps = args.epochs * max(len(train_loader), 1)
    warmup_steps = min(config.S1_WARMUP_STEPS, total_steps // 4)
    import math
    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)         # linear 0 → 1
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))     # cosine 1 → 0
    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)
    # AMP off (NaN fix): FP32 forward+backward. PCIe Gen1 환경에서 어차피 속도 cap
    scaler = torch.amp.GradScaler("cuda", enabled=False)
    if is_main(rank):
        log.info(f"AMP=OFF (FP32)  warmup={warmup_steps} steps  "
                 f"grad_clip={config.S1_GRAD_CLIP_NORM}  total_steps={total_steps}")

    # ----- Training loop -----
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
            # AMP off → autocast 도 enabled=False 로 noop. FP32 안정.
            logits = model(img0, img1)
            loss = focal_loss(
                logits, label,
                gamma=args.gamma, alpha_weight=class_weights,
                ignore_index=config.IGNORE_INDEX,
            )
            loss.backward()
            grad_norm = torch.nn.utils.clip_grad_norm_(
                unwrap(model).trainable_parameters(),
                max_norm=config.S1_GRAD_CLIP_NORM,
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

            if is_main(rank) and step % 20 == 0:
                lr_now = sched.get_last_lr()[0]
                log.info(
                    f"e{epoch:02d} train [{step:4d}/{len(train_loader)}] "
                    f"loss={loss.item():.4f}  lr={lr_now:.2e}  gnorm={grad_norm.item():.3f}"
                )

        # aggregate train metrics across ranks
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
                    log.info(f"e{epoch:02d} val   [{step:4d}/{len(val_loader)}] running")

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
