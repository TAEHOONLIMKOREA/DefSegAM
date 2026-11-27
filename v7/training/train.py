"""v7 학습 — v6 의 학습에 합성 도메인 하나를 더한 것 (PLAN_v7 §3).

v6 의 `training/train.py` 를 복사해 고친 것이다. **v7 에서 v6 의 사본은 이 파일 하나뿐이다** — v6 의 학습은 한 함수 (`main`) 라서
밖에서 끼워 넣을 자리가 없다. 데이터 읽기 · fold 분할 · 증강 · Copy-Paste · 추출 식 · 손실 · 채점은 v6 의 모듈을 그대로 부른다.
v6 와 달라지는 곳에는 `[v7]` 표시를 했다.

- **합성 세트 (`--synth-set`) 를 주지 않으면 v6 의 학습과 같은 계산이다** (같은 seed → 같은 표본열, 같은 loss). 기준선 (N) 이 이것이다.
- 합성 세트를 주면 그 표본을 학습 쪽에만 붙인다. 추출에서 합성의 몫은 `--synth-share` (기본 0.3) 이고 합성 안에서는 고르게 뽑는다.
- fold 분할 · class weight · 채점 클래스 목록 · 배선 검사 · Copy-Paste 의 조각 목록은 **사람 GT 표본으로만** 계산한다.
  합성 표본에는 Copy-Paste 를 걸지 않는다. 검증은 떼어 둔 사람 GT 로만 한다 (합성은 채점하지 않는다).
- `run.json` 의 `train_samples` 에는 사람 GT 표본만 적고 합성 표본은 `synth_samples` 에 따로 적는다 — v6 의 채점 · 요약 스크립트가 그대로 읽게.

산출물 (폴더는 v6 의 함수로 만든다 — 그림 쪽 경로에 `v6` 가 들어가는 것은 그 때문이다):
    checkpoints/<run>/cv/<fold>/{best.pt, last.pt}   또는   checkpoints/<run>/full/{last.pt, last_raw.pt}
    figures/<run>/v6/<full | cv/<fold>>/{run.json, curve.csv, result.json, domain_curve.csv}
    run 이름: `<backbone>_dpt_dual_sz<img>_10cls_v7-<꼬리표>-s<seed>`

사용 (CWD = 3DP_VPPM/) — 풀런은 compose 로 돌린다 (docker/README.md):
    torchrun --nproc_per_node=4 -m DefSeg_AM.v7.training.train --fold inlayer --seed 43                      # 기준선 IN-N
    torchrun --nproc_per_node=4 -m DefSeg_AM.v7.training.train --fold inlayer --variant W --synth-set w600_v1
    torchrun --nproc_per_node=1 -m DefSeg_AM.v7.training.train --fold inlayer --quick --variant W --synth-set smoke24_v0
"""
from __future__ import annotations

import argparse
import csv
import json
import math
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.optim.swa_utils import AveragedModel, get_ema_multi_avg_fn
from torch.utils.data import DataLoader

from ...common.data.samplers import DistributedWeightedSampler
from ...common.models.losses import sqrt_inv_class_weight
from ...common.models.model import DefSegModel, round_to_patch
from ...common.training.dist_utils import init_distributed, is_main, unwrap
from ...common.utils.log import setup_logger

# [v7] v6 의 모듈을 그대로 부른다 (복사하지 않는다)
from ...v6 import config_v6 as v6cfg
from ...v6.data.copy_paste import CopyPasteSampler
from ...v6.data.data_gt_v6 import (
    DefSegGTDatasetV6,
    GTPool,
    build_cp_supplier_items,
    enumerate_samples_v6,
    quick_subset,
    split_by_fold,
)
from ...v6.data.sampler_helpers import effective_sample_size
from ...v6.inference.metrics import accumulate_confusion, summarize
from ...v6.models.losses_v6 import median_inv_class_weight
from ...v6.training.train import _dataset_info, _fmt_iou_line, _summary_to_json, _wiring_check

from .. import config_v7 as cfg
from ..data.synth_domain import DefSegDatasetV7, compute_weights_v7, load_synth_samples, synth_set_info


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=str, default=None, choices=list(v6cfg.CV_FOLDS),
                    help="검증 fold 이름. v7 의 판정은 inlayer 와 loso_L21 로 한다 (PLAN_v7 §7 의 V12). 생략하면 전수 학습 (검증 없음)")
    ap.add_argument("--total-steps", type=int, default=v6cfg.TOTAL_STEPS)
    ap.add_argument("--eval-every", type=int, default=v6cfg.EVAL_EVERY)
    ap.add_argument("--warmup-steps", type=int, default=v6cfg.WARMUP_STEPS)
    ap.add_argument("--batch-size", type=int, default=v6cfg.BATCH_SIZE, help="GPU 당")
    ap.add_argument("--img-size", type=int, default=v6cfg.IMG_SIZE)
    ap.add_argument("--lr", type=float, default=v6cfg.LR)
    ap.add_argument("--backbone", type=str, default=v6cfg.DINO_BACKBONE)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--run-name", type=str, default=None,
                    help="기본: <backbone>_dpt_dual_sz<img>_10cls_v7-<꼬리표>-s<seed>")
    ap.add_argument(
        "--class-weight-mode", choices=["sqrt_inv", "median_inv"], default="sqrt_inv",
    )
    ap.add_argument("--ema-decay", type=float, default=v6cfg.EMA_DECAY)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--no-copy-paste", action="store_true")
    ap.add_argument("--seed", type=int, default=v6cfg.RANDOM_SEED)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--quick", action="store_true",
                    help="스모크: 6 step, 224², 소스당 3 레이어 (+ 합성 4 장), EMA decay 0.5")
    # [v7]
    ap.add_argument("--variant", type=str, default=cfg.DEFAULT_VARIANT,
                    help="실행의 꼬리표 — run 이름에 붙는다 (N = 합성 없음, W = 라벨 맵으로 그린 레이어 …)")
    ap.add_argument("--synth-set", type=str, default=None,
                    help="합성 세트의 이름 (cache/genaug/sets/ 아래) 또는 폴더 경로. 생략하면 v6 와 같은 학습이다")
    ap.add_argument("--synth-share", type=float, default=cfg.SYNTH_SHARE,
                    help="표본 추출에서 합성 도메인의 몫 (합성 세트를 줬을 때만 쓴다)")
    return ap.parse_args()


def main():
    args = parse_args()

    if args.quick:
        args.total_steps = 6
        args.eval_every = 3
        args.warmup_steps = 2
        args.batch_size = 1
        args.img_size = 224
        args.num_workers = 0
        args.log_every = 1
        args.ema_decay = 0.5          # 6 step 안에 EMA 사본이 움직이는 것이 보이도록

    rank, world_size, local_rank = init_distributed()

    log = setup_logger(rank=rank, name="train_v7")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    fold_spec = v6cfg.CV_FOLDS[args.fold] if args.fold is not None else None

    # ----- Model (decoder 무작위 초기값 — ckpt 를 읽지 않는다) -----
    # decoder 초기값은 모든 rank 가 같은 seed 로 만든다. 그 뒤의 난수 (dropout 등) 만 rank 마다 다르게 한다.
    torch.manual_seed(args.seed)
    model = DefSegModel(backbone_name=args.backbone, n_classes=v6cfg.N_CLASSES_V6).to(device)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    new_size = round_to_patch(args.img_size, model.patch_size)
    if new_size != args.img_size:
        if is_main(rank):
            log.warning(f"img_size {args.img_size} → {new_size}")
        args.img_size = new_size
    if args.run_name is None:
        args.run_name = cfg.default_run_name(args.backbone, args.img_size, args.variant, args.seed)     # [v7]

    ckpt_dir = v6cfg.run_ckpt_dir(args.run_name, args.fold)
    fig_dir = v6cfg.run_fig_dir(args.run_name, args.fold)
    if is_main(rank):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            f"=== v7 train (v6 + 합성 도메인. {v6cfg.SCHEMA_VERSION} 10-class, 단일 단계) ===  world_size={world_size}  "
            f"run_name={args.run_name}  fold={args.fold or '(전수 학습)'}  합성 세트={args.synth_set or '(없음 — v6 와 같은 학습)'}"
        )
        log.info(f"args: {vars(args)}")
    if world_size > 1:
        dist.barrier()

    if world_size > 1:
        model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
    net = unwrap(model)
    n_train_params = sum(p.numel() for p in net.trainable_parameters())

    # 모든 rank 의 decoder 가 같은 값에서 출발하는지 확인한다. 다르면 EMA 사본 (rank 0) 과 나머지 rank 가
    # 조용히 어긋난 채 학습이 진행되므로, 시작할 때 한 번 잡는다.
    if world_size > 1:
        with torch.no_grad():
            checksum = torch.stack(
                [p.detach().double().abs().sum() for p in net.trainable_parameters()]
            ).sum().reshape(1)
        lo, hi = checksum.clone(), checksum.clone()
        dist.all_reduce(lo, op=dist.ReduceOp.MIN)
        dist.all_reduce(hi, op=dist.ReduceOp.MAX)
        if (hi - lo).abs().item() > 1e-9 * max(hi.abs().item(), 1.0):
            raise RuntimeError(
                f"[r{rank}] rank 마다 decoder 초기값이 다르다 (checksum {lo.item():.6f} … {hi.item():.6f})"
            )
        if is_main(rank):
            log.info(f"decoder 초기값: {world_size} rank 가 같다 (checksum {hi.item():.4f})")

    # ----- Data: 사람 GT (+ 합성 세트) 를 한 번 읽어 RAM 에 둔다 -----
    data_root = v6cfg.LEGACY_PBF_ROOT
    legacy = enumerate_samples_v6(data_root)
    n_dataset = len(legacy)
    legacy_keys = {s.key for s in legacy}
    if args.quick:
        legacy = quick_subset(legacy, fold_spec, per_group=3)

    # [v7] 합성 세트 — 합성 도메인의 표본. 사람 GT 뒤에 붙인다 (사람 GT 의 위치 번호가 v6 와 같게 남는다)
    synth: list = []
    synth_info: dict | None = None
    if args.synth_set:
        synth_root = cfg.resolve_synth_root(args.synth_set)
        synth = load_synth_samples(synth_root, legacy_keys)
        synth_info = synth_set_info(synth_root, len(synth))
        # [v7] 누출 막기 — 세트의 생성 모델이 이 fold 의 검증 레이어를 봤으면 멈춘다 (PLAN_v7 §2.4). 전수 학습 (fold 없음) 은 막지 않는다
        vf = synth_info.get("valid_folds")
        if args.fold is not None and vf is not None and args.fold not in vf:
            raise SystemExit(
                f"FATAL: 합성 세트 {synth_info['name']} 는 fold '{args.fold}' 에 쓸 수 없다 — 이 세트를 그린 생성 모델이 그 fold 의 "
                f"검증 레이어를 봤다 (쓸 수 있는 fold: {vf})"
            )
        if args.fold is not None and vf is None and is_main(rank):
            log.warning(f"합성 세트 {synth_info['name']} 의 manifest 에 valid_folds 가 없다 — 누출 여부를 확인하지 못했다")
        if args.quick:
            synth = synth[:4]
    samples = legacy + synth
    n_legacy = len(legacy)

    if is_main(rank):
        log.info(f"데이터셋: {data_root}  ({n_dataset} 레이어, {v6cfg.SCHEMA_VERSION})")
        if synth_info is not None:
            log.info(f"합성 세트: {synth_info['root']}  ({synth_info['n_layers']} 장, 몫 {args.synth_share})"
                     + (f"  — {synth_info['purpose']}" if synth_info.get("purpose") else ""))
        log.info(f"GT pool: {len(samples)} 표본 (사람 GT {n_legacy} + 합성 {len(synth)}) 을 {args.img_size}² 로 줄여 메모리에 올린다 …")
    t_pool = time.time()
    pool = GTPool(samples, args.img_size, log=log if is_main(rank) else None)

    # [v7] fold 분할은 사람 GT 에만 건다. 합성 표본은 언제나 학습 쪽이다 (검증에 들어가지 않는다)
    train_idx_real, val_idx = split_by_fold(legacy, pool.pix[:n_legacy], fold_spec, allow_missing=args.quick)
    synth_idx = list(range(n_legacy, len(samples)))
    train_idx = list(train_idx_real) + synth_idx
    is_synth_local = np.array([False] * len(train_idx_real) + [True] * len(synth_idx), dtype=bool)   # 학습 Dataset 위치 순
    has_val = len(val_idx) > 0

    train_pix = pool.pix[train_idx]                     # (n_train, C) — 학습 Dataset 위치 순 (사람 GT, 그 뒤 합성)
    # [v7] 통계는 사람 GT 표본으로만 — 합성을 넣어도 class weight · 채점 클래스 · 배선 검사가 v6 와 같게 남는다
    train_px_total = pool.pix[train_idx_real].sum(axis=0)
    synth_px_total = pool.pix[synth_idx].sum(axis=0) if synth_idx else np.zeros(v6cfg.N_CLASSES_V6, dtype=np.int64)
    val_px_total = pool.pix[val_idx].sum(axis=0) if has_val else np.zeros(v6cfg.N_CLASSES_V6, dtype=np.int64)
    # 결함 mIoU 에 넣는 클래스: 학습 화소도 있고 검증 화소도 있는 결함 클래스
    eval_defect = [
        c for c in v6cfg.DEFECT_IDS_V6
        if train_px_total[c] > 0 and val_px_total[c] > 0
    ]
    px_per_layer = args.img_size * args.img_size
    unlabeled_train = float(pool.unlabeled[train_idx_real].sum()) / (len(train_idx_real) * px_per_layer)
    unlabeled_synth = float(pool.unlabeled[synth_idx].sum()) / (len(synth_idx) * px_per_layer) if synth_idx else None
    unlabeled_val = float(pool.unlabeled[val_idx].sum()) / (len(val_idx) * px_per_layer) if has_val else None
    wiring = None if args.quick else _wiring_check(
        fold_spec, args.img_size, train_px_total, val_px_total, len(train_idx_real), len(val_idx),
    )

    # ----- Copy-Paste supplier (학습 분할의 사람 GT 만) -----
    train_ds = DefSegDatasetV7(                         # [v7] 합성 표본에는 Copy-Paste 를 걸지 않는 Dataset
        pool, train_idx, training=True, cp_sampler=None, seed=args.seed, rank=rank,
    )
    use_cp = v6cfg.CP_ENABLE and not args.no_copy_paste
    cp_items: dict = {}
    if use_cp:
        cp_pix = train_pix.copy()
        cp_pix[is_synth_local] = 0                      # [v7] 합성 레이어는 조각을 대지 않는다
        cp_items = build_cp_supplier_items(
            cp_pix, classes=v6cfg.CP_CLASSES,
            min_component_px=v6cfg.CP_MIN_COMPONENT_PX,
        )

        def _fetch_fn(key, _ds=train_ds):
            return _ds.fetch_local(int(key))

        train_ds.cp_sampler = CopyPasteSampler(
            items_per_class=cp_items, fetch_fn=_fetch_fn,
            classes=v6cfg.CP_CLASSES, prob=v6cfg.CP_PROB,
        )

    # ----- Sampler: 도메인 비중 × [그룹 균등 + 클래스 균등 혼합, 표본당 상한] -----
    train_sources = [samples[i].group for i in train_idx]
    train_domains = [samples[i].domain for i in train_idx]
    source_names = sorted(set(train_sources))
    src_to_idx = {n: i for i, n in enumerate(source_names)}
    source_idx = np.array([src_to_idx[n] for n in train_sources], dtype=np.int64)
    weights, domain_share = compute_weights_v7(         # [v7] 합성이 없으면 v6 의 함수 그대로
        train_pix, train_sources, train_domains, synth_share=args.synth_share,
        mix=v6cfg.SAMPLER_SOURCE_MIX, max_repeat=v6cfg.SAMPLER_MAX_REPEAT, diva_max=v6cfg.DOMAIN_SHARE_DIVA_MAX,
    )
    train_sampler = DistributedWeightedSampler(
        weights=weights,
        num_samples_total=args.total_steps * args.batch_size * world_size,
        num_replicas=world_size, rank=rank, replacement=True, seed=args.seed,
    )
    train_loader = DataLoader(
        train_ds, batch_size=args.batch_size, sampler=train_sampler,
        num_workers=args.num_workers, pin_memory=True, drop_last=True,
    )

    # ----- Val (rank 0 혼자 전체를 한 번씩 — 표본이 적어 DistributedSampler 의 패딩 중복이 지표를 흔든다) -----
    val_loader = None
    if has_val and is_main(rank):
        val_ds = DefSegGTDatasetV6(pool, val_idx, training=False)
        val_loader = DataLoader(
            val_ds, batch_size=args.batch_size, shuffle=False,
            num_workers=0, pin_memory=True, drop_last=False,
        )

    # ----- Class weights (학습 분할의 사람 GT, 학습 해상도 기준 화소 수) -----
    if args.class_weight_mode == "median_inv":
        class_weights = median_inv_class_weight(train_px_total, clip=v6cfg.CLASS_WEIGHT_CLIP).to(device)
    else:
        class_weights = sqrt_inv_class_weight(train_px_total, clip=v6cfg.CLASS_WEIGHT_CLIP).to(device)

    # ----- EMA (rank 0 만 — DDP 의 모든 rank 는 같은 가중치를 갖는다) -----
    use_ema = not args.no_ema
    ema_model = None
    if use_ema and is_main(rank):
        ema_model = AveragedModel(net, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay))
        ema_model.eval()

    # ----- Optim + Warmup + Cosine -----
    optim = torch.optim.AdamW(
        net.trainable_parameters(), lr=args.lr, weight_decay=v6cfg.WEIGHT_DECAY,
    )
    total_steps = args.total_steps
    warmup_steps = min(args.warmup_steps, max(total_steps // 4, 1))

    def lr_lambda(step: int) -> float:
        if step < warmup_steps:
            return float(step) / max(1, warmup_steps)
        progress = (step - warmup_steps) / max(1, total_steps - warmup_steps)
        return 0.5 * (1.0 + math.cos(math.pi * progress))

    sched = torch.optim.lr_scheduler.LambdaLR(optim, lr_lambda)

    # ----- 기록 (rank 0) -----
    p_w = weights / weights.sum()
    share_by_source = {n: float(p_w[source_idx == i].sum()) for n, i in src_to_idx.items()}
    top5_share = float(np.sort(p_w)[::-1][:5].sum())
    synth_block = None                                  # [v7]
    if synth_info is not None:
        synth_block = dict(
            synth_info, share=float(args.synth_share), n_used=len(synth_idx),
            class_px=[int(v) for v in synth_px_total],
            class_layers=[int(v) for v in (pool.pix[synth_idx] > 0).sum(axis=0)],
            unlabeled_frac=unlabeled_synth,
        )
    base_config = {
        "version": cfg.VERSION,
        "variant": args.variant,
        "schema": v6cfg.SCHEMA_VERSION,
        "backbone": args.backbone,
        "n_classes": v6cfg.N_CLASSES_V6,
        "class_names": list(v6cfg.CLASS_NAMES_V6),
        "class_keys": list(v6cfg.CLASS_KEYS_V6),
        "orientation": f"recoat axis = image {v6cfg.CANONICAL_RECOAT_AXIS}",
        "img_size": args.img_size,
        "intermediate_layers": list(v6cfg.INTERMEDIATE_LAYERS),
        "decoder_channels": v6cfg.DECODER_CHANNELS,
        "class_weight_mode": args.class_weight_mode,
        "ema_decay": args.ema_decay if use_ema else None,
        "init_from": None,
        "label_source": "Peregrine human annotations converted to pbf-2026.11 (no DSCNN predictions)"
                        + (" + synthetic layers drawn from label maps by the GenAug generator" if synth_idx else ""),
        "dataset": _dataset_info(data_root, n_dataset),
        "synth": synth_block,
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "batch_size": args.batch_size,
        "world_size": world_size,
        "effective_batch": args.batch_size * world_size,
        "lr": args.lr,
        "weight_decay": v6cfg.WEIGHT_DECAY,
        "seed": args.seed,
        "fold": args.fold,
        "fold_spec": list(fold_spec) if fold_spec is not None else None,
        "n_train": len(train_idx_real),                 # 사람 GT 의 학습 레이어 수 (v6 와 같은 뜻)
        "n_synth": len(synth_idx),
        "n_val": len(val_idx),
        "copy_paste": bool(use_cp),
        "copy_paste_rule": {
            "classes": list(v6cfg.CP_CLASSES),
            "required_under": {str(k): v for k, v in v6cfg.CP_REQUIRED_UNDER.items()},
            "region_min_frac": v6cfg.CP_REGION_MIN_FRAC, "position_tries": v6cfg.CP_POSITION_TRIES,
            "anchor_on_required": v6cfg.CP_ANCHOR_ON_REQUIRED,
        } if use_cp else None,
        "sampler": {
            "source_mix": v6cfg.SAMPLER_SOURCE_MIX, "max_repeat": v6cfg.SAMPLER_MAX_REPEAT,
            "domain_share": domain_share,
        },
        "class_train_px": [int(v) for v in train_px_total],
    }
    curve_path = fig_dir / "curve.csv"
    curve_fields = ["step", "lr", "train_loss"]
    if has_val:
        for tag in ("raw", "ema"):
            curve_fields += [f"{tag}_acc", f"{tag}_miou", f"{tag}_defect_miou"]
            curve_fields += [f"{tag}_iou_{c}" for c in range(v6cfg.N_CLASSES_V6)]
    # [v7] 도메인별 loss 는 curve.csv 를 건드리지 않고 따로 쓴다 (v6 의 요약 스크립트가 curve.csv 를 읽는다)
    track_domains = bool(synth_idx) and is_main(rank)
    domain_curve_path = fig_dir / "domain_curve.csv"

    if is_main(rank):
        log.info(
            f"device={device}  img_size={args.img_size}  bs_per_gpu={args.batch_size}  "
            f"effective_batch={args.batch_size * world_size}  trainable={n_train_params/1e6:.2f}M  "
            f"n_classes={v6cfg.N_CLASSES_V6}  (pool {time.time()-t_pool:.0f}s)"
        )
        log.info(
            f"train={len(train_idx_real)} 레이어 (사람 GT) + 합성 {len(synth_idx)} 장  val={len(val_idx)} 레이어  "
            f"source 분포(train)={dict(Counter(train_sources))}"
        )
        if has_val:
            log.info(f"val 소스 분포={dict(Counter(samples[i].group for i in val_idx))}")
            log.info(
                "결함 mIoU 에 넣는 클래스 (학습 · 검증 화소가 모두 있는 결함): "
                f"{[v6cfg.CLASS_ABBRS_V6[c] for c in eval_defect]}"
            )
            skipped = [v6cfg.CLASS_ABBRS_V6[c] for c in v6cfg.DEFECT_IDS_V6
                       if val_px_total[c] > 0 and train_px_total[c] == 0]
            if skipped:
                log.warning(f"검증에는 있지만 학습 화소가 0 인 클래스 (채점 제외): {skipped}")
        log.info(f"class px (train · 사람 GT, {args.img_size}²): {[int(v) for v in train_px_total]}")
        if synth_idx:
            log.info(f"class px (합성, {args.img_size}²):          {[int(v) for v in synth_px_total]}"
                     f"  (class weight 에는 들어가지 않는다)")
        if has_val:
            log.info(f"class px (val,   {args.img_size}²): {[int(v) for v in val_px_total]}")
        log.info(
            f"미레이블 화소: train {100 * unlabeled_train:.1f}%"
            + (f"  합성 {100 * unlabeled_synth:.1f}%" if unlabeled_synth is not None else "")
            + (f"  val {100 * unlabeled_val:.1f}%" if unlabeled_val is not None else "")
        )
        if wiring is not None and wiring["match"]:
            log.info(f"배선 확인: 클래스별 화소 수 · 레이어 수가 계획의 표 (splits/{wiring['split_file']}) 와 같다")
        elif wiring is not None:
            log.warning(
                f"배선 확인 실패: 클래스별 화소 수가 계획의 표 (splits/{wiring['split_file']}) 와 다르다 — "
                f"train 차이 {wiring['diff_train_px']}  val 차이 {wiring['diff_val_px']}. "
                "데이터셋을 다른 옵션으로 다시 변환했다면 정상이고, 아니면 로더가 계획과 다른 것이다 (PLAN_v6 §7.1-1)."
            )
        log.info(
            f"class weights (mode={args.class_weight_mode}, clip={v6cfg.CLASS_WEIGHT_CLIP}): "
            f"{[round(float(v), 3) for v in class_weights.cpu()]}"
        )
        log.info(
            f"sampler: ESS={effective_sample_size(p_w):.1f}/{len(p_w)}  max={100*p_w.max():.2f}%  "
            f"top5={100*top5_share:.1f}%  by source="
            f"{ {v6cfg.SOURCE_SHORT.get(n, n): round(100*v, 1) for n, v in share_by_source.items()} }"
            f"  domain={ {d: round(v, 3) for d, v in domain_share.items()} }"
        )
        if use_cp:
            log.info(
                "Copy-Paste supplier (class, n 레이어 — 사람 GT 만): "
                f"{[(v6cfg.CLASS_ABBRS_V6[c], len(cp_items.get(c, []))) for c in v6cfg.CP_CLASSES]}  "
                f"위치 규칙 (밑에 있어야 하는 클래스): "
                f"{ {v6cfg.CLASS_ABBRS_V6[c]: v6cfg.CLASS_ABBRS_V6[r] for c, r in v6cfg.CP_REQUIRED_UNDER.items() if c in v6cfg.CP_CLASSES} }"
            )
        if use_ema and not args.quick and 1.0 / (1.0 - args.ema_decay) > total_steps / 10:
            log.warning(
                f"EMA 시상수 {1.0/(1.0-args.ema_decay):.0f} step 이 총 step 의 1/10 ({total_steps/10:.0f}) 보다 크다 — "
                f"저장되는 EMA 사본에 초기값이 {100*args.ema_decay**total_steps:.1f}% 남는다. --ema-decay 를 낮출 것."
            )
        log.info(
            f"AMP=OFF  total_steps={total_steps}  warmup={warmup_steps}  eval_every={args.eval_every}  "
            f"grad_clip={v6cfg.GRAD_CLIP_NORM}  ema={'%.4g' % args.ema_decay if use_ema else 'off'}"
        )
        with (fig_dir / "run.json").open("w") as fh:
            json.dump({
                "config": base_config,
                "args": vars(args),
                # [v7] train_samples 는 사람 GT 만 — v6 의 evaluate · baselines 가 이 목록을 사람 GT 데이터셋에서 찾는다
                "train_samples": [samples[i].key for i in train_idx_real],
                "synth_samples": [samples[i].key for i in synth_idx],
                "val_samples": [samples[i].key for i in val_idx],
                "class_names": list(v6cfg.CLASS_NAMES_V6),
                "class_abbrs": list(v6cfg.CLASS_ABBRS_V6),
                "class_train_px": [int(v) for v in train_px_total],
                "class_synth_px": [int(v) for v in synth_px_total],
                "class_val_px": [int(v) for v in val_px_total],
                "class_train_layers": [int(v) for v in (pool.pix[train_idx_real] > 0).sum(axis=0)],
                "class_val_layers": [int(v) for v in (pool.pix[val_idx] > 0).sum(axis=0)] if has_val else None,
                "unlabeled_frac": {"train": unlabeled_train, "synth": unlabeled_synth, "val": unlabeled_val},
                "wiring_check": wiring,
                "defect_eval_classes": [int(c) for c in eval_defect],
                "class_weights": [round(float(v), 4) for v in class_weights.cpu()],
                "sampler": {
                    "ess": effective_sample_size(p_w), "n": int(len(p_w)),
                    "max_share": float(p_w.max()), "top5_share": top5_share,
                    "share_by_source": share_by_source,
                    "domain_share": domain_share,
                },
                "copy_paste_suppliers": {str(c): len(v) for c, v in cp_items.items()},
            }, fh, ensure_ascii=False, indent=1)
        with curve_path.open("w", newline="") as fh:
            csv.writer(fh).writerow(curve_fields)
        if track_domains:
            with domain_curve_path.open("w", newline="") as fh:
                csv.writer(fh).writerow(["step", "loss_legacy", "loss_synth", "n_legacy", "n_synth"])

    def save_ckpt(path: Path, module, weights_kind: str, step: int, val: dict | None) -> None:
        torch.save({
            "model_state": {k: v.detach().cpu().clone() for k, v in module.trainable_state_dict().items()},
            "config": base_config,
            "weights": weights_kind,            # "ema" | "raw"
            "step": step,
            "val": val,                         # 검증 요약 (fold 학습) 또는 None
            # v4 · v5 형식을 읽는 코드와의 호환용 키
            "epoch": None,
            "val_acc": val["acc"] if val else None,
            "miou": val["miou"] if val else None,
        }, path)

    def run_validation(step: int) -> tuple[dict, dict]:
        """rank 0: 학습용 가중치와 EMA 사본을 둘 다 채점."""
        cm_raw = accumulate_confusion(net, val_loader, device, v6cfg.N_CLASSES_V6)
        s_raw = summarize(cm_raw, eval_defect)
        if ema_model is not None:
            cm_ema = accumulate_confusion(ema_model.module, val_loader, device, v6cfg.N_CLASSES_V6)
            s_ema = summarize(cm_ema, eval_defect)
        else:
            s_ema = s_raw
        return s_raw, s_ema

    # ----- Train loop -----
    best_score = -1.0
    best_step = -1
    last_val: dict | None = None
    step = 0
    win_loss, win_n = 0.0, 0          # 로그 창
    seg_loss, seg_n = 0.0, 0          # 검증 사이 구간 (curve.csv)
    # [v7] 도메인별 loss (rank 0 이 본 배치에서). num = Σ 가중 loss, den = Σ 가중, n = 표본 수. "win" 은 로그 창, "seg" 는 검증 사이 구간
    dom = {k: {d: [0.0, 0.0, 0] for d in ("legacy", "synth")} for k in ("win", "seg")}
    t_start = time.time()
    t_val_total = 0.0

    def _dom_text(acc: dict) -> str:
        parts = []
        for d in ("legacy", "synth"):
            num, den, n = acc[d]
            parts.append(f"{d} {num / den:.4f} ({n})" if den > 0 else f"{d} – (0)")
        return "  ".join(parts)

    model.train()
    net.backbone.eval()
    for batch in train_loader:
        img0 = batch["img0"].to(device, non_blocking=True)
        img1 = batch["img1"].to(device, non_blocking=True)
        label = batch["label"].to(device, non_blocking=True)

        lr_now = sched.get_last_lr()[0]
        optim.zero_grad(set_to_none=True)
        logits = model(img0, img1)
        loss = F.cross_entropy(
            logits, label, weight=class_weights, ignore_index=v6cfg.IGNORE_INDEX,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"[r{rank}] loss 가 유한하지 않다 (step {step + 1}): {loss.item()}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            net.trainable_parameters(), max_norm=v6cfg.GRAD_CLIP_NORM,
        )
        optim.step()
        sched.step()
        step += 1
        if ema_model is not None:
            ema_model.update_parameters(net)

        loss_val = loss.item()
        win_loss += loss_val; win_n += 1
        seg_loss += loss_val; seg_n += 1

        if track_domains:
            # [v7] 기록용 — 최적화하는 loss 는 위의 것 그대로다. 같은 식 (가중 CE) 을 도메인별로 나눠 센다
            with torch.no_grad():
                is_syn = batch["is_synth"].to(device)
                per_pix = F.cross_entropy(
                    logits.detach(), label, weight=class_weights, ignore_index=v6cfg.IGNORE_INDEX, reduction="none",
                )
                valid = label >= 0
                w_map = class_weights[label.clamp(min=0)] * valid
                for d, m in (("legacy", ~is_syn), ("synth", is_syn)):
                    if bool(m.any()):
                        num, den, n = float(per_pix[m].sum()), float(w_map[m].sum()), int(m.sum())
                        for k in ("win", "seg"):
                            dom[k][d][0] += num; dom[k][d][1] += den; dom[k][d][2] += n

        if is_main(rank) and (step % args.log_every == 0 or step == 1):
            elapsed = time.time() - t_start - t_val_total
            rate = step / max(elapsed, 1e-6)
            eta_min = (total_steps - step) / max(rate, 1e-6) / 60.0
            log.info(
                f"step [{step:6d}/{total_steps}] loss={win_loss / max(win_n, 1):.4f}  "
                f"lr={lr_now:.2e}  gnorm={grad_norm.item():.3f}  {rate:.2f} it/s  ETA {eta_min:.0f}m"
                + (f"  |  {_dom_text(dom['win'])}" if track_domains else "")
            )
            win_loss, win_n = 0.0, 0
            dom["win"] = {d: [0.0, 0.0, 0] for d in ("legacy", "synth")}

        at_eval = (step % args.eval_every == 0) or (step == total_steps)
        if at_eval and track_domains:
            a = dom["seg"]
            with domain_curve_path.open("a", newline="") as fh:
                csv.writer(fh).writerow([
                    step,
                    f"{a['legacy'][0] / a['legacy'][1]:.6f}" if a["legacy"][1] > 0 else "",
                    f"{a['synth'][0] / a['synth'][1]:.6f}" if a["synth"][1] > 0 else "",
                    a["legacy"][2], a["synth"][2],
                ])
            dom["seg"] = {d: [0.0, 0.0, 0] for d in ("legacy", "synth")}
        if at_eval and has_val:
            if is_main(rank):
                t_v = time.time()
                s_raw, s_ema = run_validation(step)
                t_val_total += time.time() - t_v
                last_val = _summary_to_json(s_ema)
                sel = s_ema["defect_miou"]
                if sel != sel:                    # 채점할 결함 클래스가 없으면 mIoU 로
                    sel = s_ema["miou"]
                log.info(
                    f"step {step:6d} VAL  train_loss={seg_loss / max(seg_n, 1):.4f}  "
                    f"raw: acc={s_raw['acc']:.4f} mIoU={s_raw['miou']:.4f} defect={s_raw['defect_miou']:.4f}  |  "
                    f"ema: acc={s_ema['acc']:.4f} mIoU={s_ema['miou']:.4f} defect={s_ema['defect_miou']:.4f}  "
                    f"(val {time.time() - t_v:.0f}s)"
                )
                log.info(f"  iou(ema): {_fmt_iou_line(s_ema)}")
                row = [step, f"{lr_now:.6e}", f"{seg_loss / max(seg_n, 1):.6f}"]
                for s in (s_raw, s_ema):
                    row += [s["acc"], s["miou"], s["defect_miou"]]
                    row += [s["iou"][c] for c in range(v6cfg.N_CLASSES_V6)]
                row = ["" if (isinstance(v, float) and v != v) else v for v in row]
                with curve_path.open("a", newline="") as fh:
                    csv.writer(fh).writerow(row)
                if sel == sel and sel > best_score:
                    best_score, best_step = float(sel), step
                    save_obj = ema_model.module if ema_model is not None else net
                    save_ckpt(ckpt_dir / "best.pt", save_obj,
                              "ema" if ema_model is not None else "raw", step, last_val)
                    log.info(f"  ↑ saved best ({'EMA' if ema_model is not None else 'raw'}, "
                             f"defect mIoU={sel:.4f}) to {ckpt_dir / 'best.pt'}")
            if world_size > 1:
                dist.barrier()
            model.train()
            net.backbone.eval()
            seg_loss, seg_n = 0.0, 0
        elif at_eval and is_main(rank):
            with curve_path.open("a", newline="") as fh:
                csv.writer(fh).writerow([step, f"{lr_now:.6e}", f"{seg_loss / max(seg_n, 1):.6f}"])
            seg_loss, seg_n = 0.0, 0

    # ----- 마무리 -----
    if is_main(rank):
        if step != total_steps:
            log.warning(f"예정한 {total_steps} step 중 {step} step 만 돌았다")
        wall = time.time() - t_start
        final_obj = ema_model.module if ema_model is not None else net
        final_kind = "ema" if ema_model is not None else "raw"
        save_ckpt(ckpt_dir / "last.pt", final_obj, final_kind, step, last_val)
        log.info(f"saved last ({final_kind}, step {step}) to {ckpt_dir / 'last.pt'}")
        if not has_val and ema_model is not None:
            # 전수 학습: 학습용 가중치도 남긴다 (EMA 사본이 제대로 따라왔는지 나중에 확인할 수 있게)
            save_ckpt(ckpt_dir / "last_raw.pt", net, "raw", step, None)
            log.info(f"saved last_raw (step {step}) to {ckpt_dir / 'last_raw.pt'}")
        with (fig_dir / "result.json").open("w") as fh:
            json.dump({
                "run_name": args.run_name, "fold": args.fold, "steps": step,
                "variant": args.variant, "seed": args.seed,
                "synth_set": synth_info["name"] if synth_info else None,
                "wall_minutes": wall / 60.0,
                "train_it_per_s": step / max(wall - t_val_total, 1e-6),
                "best_step": best_step if has_val else None,
                "best_defect_miou_ema": best_score if has_val and best_step > 0 else None,
                "final_val_ema": last_val,
            }, fh, ensure_ascii=False, indent=1)
        if has_val:
            log.info(
                f"done. {step} step, {wall/60:.1f}m.  best defect mIoU(ema)={best_score:.4f} at step {best_step}  "
                f"final={last_val['defect_miou'] if last_val else None}"
            )
        else:
            log.info(f"done. {step} step, {wall/60:.1f}m. (전수 학습 — 검증 없음)")

    if world_size > 1:
        dist.barrier()
    if dist.is_initialized():
        dist.destroy_process_group()


if __name__ == "__main__":
    main()
