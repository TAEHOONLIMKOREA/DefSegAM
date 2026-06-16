"""v3 Stage 2 학습 — DSCNN_Dataset 8 source 전수 train, ORNL Build 1 로 평가.

PLAN_v3 §4.2 참조.

- 8 source 의 모든 GT 를 train 에 활용 (CV 없음)
- Source 별 균등 추출 (DistributedWeightedSampler)
- Replicate factor K=4 → epoch 당 effective sample 수 ↑
- ORNL Build 1 (v3 label cache) 의 mIoU 로 best ckpt 선정
- 단일 ckpt: `stage2_best.pt`

사용:
    torchrun --nproc_per_node=4 -m DefSeg_AM.v3.training.train_stage2
    torchrun --nproc_per_node=1 -m DefSeg_AM.v3.training.train_stage2 --quick
"""
from __future__ import annotations

import argparse
import math
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

from ...common.data.samplers import DistributedWeightedSampler
from ...common.models.losses import sqrt_inv_class_weight
from ...common.models.model import DefSegModel, round_to_patch
from ...common.training.dist_utils import (
    init_distributed, is_main, unwrap, reduce_counts,
)
from ...common.utils.log import setup_logger

from .. import config_v3 as cfg
from ..data.data_dscnn_v3 import (
    DefSegDSCNNDatasetV3,
    compute_class_counts_v3,
    compute_source_balanced_weights,
    enumerate_samples_v3,
)
from ..data.data_ornl_v3 import DefSegORNLCachedDatasetV3
from ..models.losses_v3 import median_inv_class_weight
from .train_stage1 import update_counts_v3


def main():
    ap = argparse.ArgumentParser()
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
                    help="Stage 1 load 없이 random init (ablation)")
    ap.add_argument(
        "--class-weight-mode", choices=["sqrt_inv", "median_inv"], default="sqrt_inv",
    )
    ap.add_argument("--ema-decay", type=float, default=cfg.EMA_DECAY)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--replicate-factor", type=int, default=cfg.S2_REPLICATE_FACTOR)
    ap.add_argument("--eval-n-layers", type=int, default=cfg.S2_EVAL_N_LAYERS)
    ap.add_argument("--seed", type=int, default=cfg.S2_RANDOM_SEED)
    ap.add_argument("--quick", action="store_true")
    args = ap.parse_args()

    if args.quick:
        args.epochs = 2
        args.batch_size = 1
        args.img_size = 224
        args.num_workers = 0
        args.replicate_factor = 1
        args.eval_n_layers = 8

    if args.run_name is None:
        args.run_name = f"{args.backbone}_dpt_dual_sz{args.img_size}_10cls_v3"

    rank, world_size, local_rank = init_distributed()
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)

    log = setup_logger(rank=rank, name="stage2_v3")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    if is_main(rank):
        log.info(f"=== v3 stage2 (전수 학습 + ORNL Build 1 eval) ===  "
                 f"world_size={world_size}  run_name={args.run_name}")
        log.info(f"args: {vars(args)}")

    # ----- ckpt paths -----
    run_ckpt_dir = cfg.CHECKPOINT_DIR / args.run_name
    if is_main(rank):
        run_ckpt_dir.mkdir(parents=True, exist_ok=True)
    if world_size > 1:
        dist.barrier()
    ckpt_path = run_ckpt_dir / "stage2_best.pt"
    init_path = args.init_from or str(run_ckpt_dir / "stage1_best.pt")

    # ----- Model + Stage 1 load -----
    model = DefSegModel(backbone_name=args.backbone, n_classes=cfg.N_CLASSES_V3).to(device)
    new_size = round_to_patch(args.img_size, model.patch_size)
    if new_size != args.img_size:
        if is_main(rank):
            log.warning(f"img_size {args.img_size} → {new_size}")
        args.img_size = new_size

    if not args.no_init:
        if not Path(init_path).exists():
            raise FileNotFoundError(
                f"Stage 1 ckpt not found: {init_path}\n"
                "Run `python -m DefSeg_AM.v3.training.train_stage1` first or pass --no-init."
            )
        sd = torch.load(init_path, map_location=device, weights_only=False)
        ckpt_cfg = sd.get("config", {})
        if ckpt_cfg.get("n_classes") not in (None, cfg.N_CLASSES_V3):
            log.warning(
                f"Stage 1 ckpt n_classes={ckpt_cfg.get('n_classes')} ≠ v3 {cfg.N_CLASSES_V3}"
            )
        model.load_trainable_state_dict(sd["model_state"])
        if is_main(rank):
            log.info(f"loaded stage1 ckpt: {init_path}  val_acc={sd.get('val_acc', '?')}")
    elif is_main(rank):
        log.warning("training from scratch (--no-init)")

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    n_train_params = sum(p.numel() for p in unwrap(model).trainable_parameters())
    if is_main(rank):
        log.info(
            f"device={device}  img_size={args.img_size}  bs_per_gpu={args.batch_size}  "
            f"trainable={n_train_params/1e6:.2f}M  n_classes={cfg.N_CLASSES_V3}"
        )

    # ----- EMA -----
    use_ema = not args.no_ema
    ema_model = None
    if use_ema:
        ema_model = AveragedModel(
            unwrap(model),
            multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay),
        )

    # ----- Train: DSCNN_Dataset 8 source 전수 -----
    train_specs = enumerate_samples_v3()
    train_ds = DefSegDSCNNDatasetV3(
        train_specs, img_size=args.img_size, training=True,
        replicate_factor=args.replicate_factor,
    )
    if is_main(rank):
        log.info(
            f"train={len(train_specs)} unique samples × K={args.replicate_factor} "
            f"= {len(train_ds)} effective"
        )
        from collections import Counter
        src_counter = Counter(s.source_name for s in train_specs)
        log.info(f"source distribution: {dict(src_counter)}")

    # source 균등 추출 weights (replicate factor 까지 broadcast)
    src_w = compute_source_balanced_weights(train_specs)
    full_weights = np.tile(src_w, args.replicate_factor)
    train_sampler = DistributedWeightedSampler(
        weights=full_weights,
        num_samples_total=len(train_ds) if not args.quick else 8 * world_size,
        num_replicas=world_size, rank=rank, replacement=True,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
        persistent_workers=(args.num_workers > 0),
    )

    # ----- Val: ORNL Build 1 의 v3 cache 균등 샘플링 -----
    full_val_ds = DefSegORNLCachedDatasetV3("val", args.img_size, training=False)
    val_n = min(args.eval_n_layers, len(full_val_ds))
    val_idx = np.linspace(0, len(full_val_ds) - 1, val_n, dtype=int).tolist()
    val_ds = torch.utils.data.Subset(full_val_ds, val_idx)
    if is_main(rank):
        log.info(
            f"eval = ORNL Build 1 {val_n}/{len(full_val_ds)} layers "
            f"(균등 샘플링, 매 epoch mIoU 측정)"
        )
    if world_size > 1:
        val_sampler = DistributedSampler(val_ds, num_replicas=world_size, rank=rank, shuffle=False)
    else:
        val_sampler = None
    val_loader = DataLoader(
        val_ds, batch_size=args.batch_size, sampler=val_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=False,
        persistent_workers=(args.num_workers > 0),
    )

    # ----- Class weights -----
    if is_main(rank):
        log.info(f"computing class weights from {len(train_specs)} unique samples …")
    counts = compute_class_counts_v3(train_specs)
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

    best_miou = -1.0
    best_per_class = None
    best_epoch = -1
    for epoch in range(args.epochs):
        train_sampler.set_epoch(epoch)
        if val_sampler is not None:
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

            if is_main(rank) and step % 20 == 0:
                lr_now = sched.get_last_lr()[0]
                log.info(
                    f"e{epoch:02d} train [{step:4d}/{len(train_loader)}] "
                    f"loss={loss.item():.4f}  lr={lr_now:.2e}  gnorm={grad_norm.item():.3f}"
                )

        reduce_counts(world_size, train_loss_sum, train_n, train_correct, train_total)
        train_loss_avg = (train_loss_sum / train_n.clamp(min=1)).item()
        train_acc = (train_correct / train_total.clamp(min=1)).item()

        # ----- eval: ORNL Build 1 -----
        eval_model = ema_model if use_ema else unwrap(model)
        eval_model.eval()
        val_correct = torch.zeros(1, device=device)
        val_total = torch.zeros(1, device=device)
        val_inter = torch.zeros(cfg.N_CLASSES_V3, device=device)
        val_union = torch.zeros(cfg.N_CLASSES_V3, device=device)
        t_val = time.time()

        with torch.no_grad():
            for batch in val_loader:
                img0 = batch["img0"].to(device, non_blocking=True)
                img1 = batch["img1"].to(device, non_blocking=True)
                label = batch["label"].to(device, non_blocking=True)
                if use_ema:
                    logits = ema_model.module(img0, img1)
                else:
                    logits = eval_model(img0, img1)
                update_counts_v3(logits, label, val_correct, val_total, val_inter, val_union)

        reduce_counts(world_size, val_correct, val_total, val_inter, val_union)
        val_acc = (val_correct / val_total.clamp(min=1)).item()
        per_cls_iou = []
        for c in range(cfg.N_CLASSES_V3):
            u = val_union[c].item()
            per_cls_iou.append(val_inter[c].item() / u if u > 0 else float("nan"))
        valid_ious = [x for x in per_cls_iou if x == x]
        miou = sum(valid_ious) / max(len(valid_ious), 1)

        if is_main(rank):
            log.info(
                f"e{epoch:02d} SUMMARY  train_loss={train_loss_avg:.4f}  "
                f"train_acc={train_acc:.4f}  build1_val_acc={val_acc:.4f}  "
                f"build1_mIoU={miou:.4f}  "
                f"(epoch {(time.time()-t_ep)/60:.1f}m, val {(time.time()-t_val)/60:.1f}m)"
            )
            for c, iou in enumerate(per_cls_iou):
                tag = f"{c} {cfg.ORNL_CLASS_NAMES_V3[c]:<22s}"
                line = f"  iou {tag} = {iou:.4f}" if iou == iou else f"  iou {tag} = n/a"
                log.info(line)

            if miou > best_miou:
                best_miou = miou
                best_per_class = per_cls_iou
                best_epoch = epoch
                save_obj = unwrap(model) if not use_ema else ema_model.module
                torch.save({
                    "model_state": save_obj.trainable_state_dict(),
                    "config": {
                        "version": "v3",
                        "backbone": args.backbone,
                        "n_classes": cfg.N_CLASSES_V3,
                        "img_size": args.img_size,
                        "intermediate_layers": list(cfg.INTERMEDIATE_LAYERS),
                        "decoder_channels": cfg.DECODER_CHANNELS,
                        "class_weight_mode": args.class_weight_mode,
                        "ema_decay": args.ema_decay if use_ema else None,
                        "init_from": init_path if not args.no_init else None,
                        "replicate_factor": args.replicate_factor,
                        "eval_n_layers": val_n,
                        "seed": args.seed,
                    },
                    "epoch": epoch, "val_acc": val_acc, "miou": miou,
                    "per_class_iou": per_cls_iou,
                }, ckpt_path)
                log.info(f"↑ saved best (EMA={use_ema}, mIoU={miou:.4f}) to {ckpt_path}")

        if world_size > 1:
            dist.barrier()

    if is_main(rank):
        log.info(
            f"done. best mIoU={best_miou:.4f} at epoch {best_epoch}"
        )
        if best_per_class:
            for c, iou in enumerate(best_per_class):
                tag = f"{c} {cfg.ORNL_CLASS_NAMES_V3[c]:<22s}"
                line = f"  best iou {tag} = {iou:.4f}" if iou == iou else f"  best iou {tag} = n/a"
                log.info(line)

    if world_size > 1:
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
