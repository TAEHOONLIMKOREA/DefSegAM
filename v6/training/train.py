"""v6 학습 — `pbf-2026.11` 10-class, 사람 GT 만으로 decoder 를 무작위 초기값에서, 단일 단계 · step 기준.

PLAN_v6 §5 참조. v5 의 training/train.py 를 복사해 고친 것이다. 학습 방식 (길이 · 스케줄 · EMA · sampler 식 · 증강 · 손실) 은 v5 와 같고,
달라진 것은 둘이다 — 클래스 체계 (변환해 둔 데이터셋 `ORNL_Data/DSCNN_Dataset_pbf-2026.11` 의 10-class 라벨을 그대로 읽는다) 와
Copy-Paste 의 위치 규칙 (data/copy_paste.py).

- 학습 데이터: Peregrine 주석 8 소스 125 레이어를 새 체계로 옮긴 것 (--fold 를 주면 그 fold 가 떼어 둔 표본을 뺀다)
- Stage 1 없음 · ckpt 초기화 없음. ORNL HDF5 (DSCNN 예측) 는 읽지 않는다
- 길이는 optimizer step (--total-steps). warmup → cosine
- EMA 사본 (decay 0.999) 을 저장 · 배포한다. 검증에서는 EMA 와 학습용 가중치를 둘 다 채점해 곡선에 남긴다
- 검증 (fold 지정 시): --eval-every step 마다 rank 0 이 떼어 둔 사람 GT 전체를 한 번씩 채점 (학습 해상도)
- 고정 목록 fold (inlayer) 는 시작할 때 클래스별 화소 수를 계획의 표 (splits/legacy_inlayer_v1.json 의 expected) 와 대조한다

산출물:
    checkpoints/<run>/full/{last.pt, last_raw.pt}                 전수 학습 (Peregrine 주석 125 레이어)
    checkpoints/<run>/cv/<fold>/{best.pt, last.pt}                검증 fold
    figures/<run>/v6/<full | cv/<fold>>/{run.json, curve.csv, result.json}

사용 (CWD = 3DP_VPPM/):
    torchrun --nproc_per_node=4 -m DefSeg_AM.v6.training.train --fold inlayer     # P0 (PLAN_v6 §7.1)
    torchrun --nproc_per_node=4 -m DefSeg_AM.v6.training.train                    # 전수 학습
    torchrun --nproc_per_node=1 -m DefSeg_AM.v6.training.train --fold inlayer --quick
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

from .. import config_v6 as cfg
from ..data.copy_paste import CopyPasteSampler
from ..data.data_gt_v6 import (
    DefSegGTDatasetV6,
    GTPool,
    build_cp_supplier_items,
    enumerate_samples_v6,
    load_split_list,
    quick_subset,
    split_by_fold,
)
from ..data.sampler_helpers import compute_domain_sample_weights, effective_sample_size
from ..inference.metrics import accumulate_confusion, summarize
from ..models.losses_v6 import median_inv_class_weight


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=str, default=None, choices=list(cfg.CV_FOLDS),
                    help="검증 fold 이름 (P0 는 inlayer). 생략하면 8 소스 전수 학습 (검증 없음)")
    ap.add_argument("--total-steps", type=int, default=cfg.TOTAL_STEPS)
    ap.add_argument("--eval-every", type=int, default=cfg.EVAL_EVERY)
    ap.add_argument("--warmup-steps", type=int, default=cfg.WARMUP_STEPS)
    ap.add_argument("--batch-size", type=int, default=cfg.BATCH_SIZE, help="GPU 당")
    ap.add_argument("--img-size", type=int, default=cfg.IMG_SIZE)
    ap.add_argument("--lr", type=float, default=cfg.LR)
    ap.add_argument("--backbone", type=str, default=cfg.DINO_BACKBONE)
    ap.add_argument("--num-workers", type=int, default=4)
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument(
        "--class-weight-mode", choices=["sqrt_inv", "median_inv"], default="sqrt_inv",
    )
    ap.add_argument("--ema-decay", type=float, default=cfg.EMA_DECAY)
    ap.add_argument("--no-ema", action="store_true")
    ap.add_argument("--no-copy-paste", action="store_true")
    ap.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    ap.add_argument("--log-every", type=int, default=50)
    ap.add_argument("--quick", action="store_true",
                    help="스모크: 6 step, 224², 소스당 3 레이어, EMA decay 0.5")
    return ap.parse_args()


def _nan_to_none(x) -> float | None:
    x = float(x)
    return None if x != x else x


def _summary_to_json(s: dict) -> dict:
    """metrics.summarize 결과 → json/pickle 로 그대로 쓸 수 있는 dict."""
    return {
        "acc": _nan_to_none(s["acc"]),
        "miou": _nan_to_none(s["miou"]),
        "defect_miou": _nan_to_none(s["defect_miou"]),
        "iou": [_nan_to_none(v) for v in s["iou"]],
        "defect_classes": [int(c) for c in s["defect_classes"]],
    }


def _fmt_iou_line(s: dict) -> str:
    parts = []
    for c in range(cfg.N_CLASSES_V6):
        v = s["iou"][c]
        if v == v:
            parts.append(f"{cfg.CLASS_ABBRS_V6[c]} {v:.3f}")
    return "  ".join(parts)


def _dataset_info(root: Path, n_layers: int) -> dict:
    """어느 데이터셋으로 학습했는지 — 같은 폴더가 다른 옵션으로 다시 변환될 수 있으므로 변환 기록을 함께 남긴다."""
    info = {"root": str(root), "schema": cfg.SCHEMA_VERSION, "n_layers": n_layers, "conversion": None}
    conv = root / "conversion.json"
    if conv.is_file():
        c = json.loads(conv.read_text(encoding="utf-8"))
        info["conversion"] = {k: c.get(k) for k in ("created_at", "code", "options")}
    return info


def _wiring_check(fold_spec, img_size: int, train_px, val_px, n_train: int, n_val: int) -> dict | None:
    """고정 목록 fold: 클래스별 화소 수가 계획의 표 (분할 파일의 expected) 와 같은지. 대조할 표가 없으면 None."""
    if fold_spec is None or fold_spec[0] != "list":
        return None
    exp = load_split_list(fold_spec[1]).get("expected")
    if not exp or exp.get("img_size") != img_size:
        return None
    got_t, got_v = [int(v) for v in train_px], [int(v) for v in val_px]
    match = (
        got_t == list(exp["class_train_px"]) and got_v == list(exp["class_val_px"])
        and n_train == exp["n_train"] and n_val == exp["n_val"]
    )
    out = {"split_file": fold_spec[1], "match": bool(match), "expected_dataset": exp.get("dataset")}
    if not match:
        out["diff_train_px"] = {cfg.CLASS_ABBRS_V6[c]: got_t[c] - int(exp["class_train_px"][c])
                                for c in range(cfg.N_CLASSES_V6) if got_t[c] != int(exp["class_train_px"][c])}
        out["diff_val_px"] = {cfg.CLASS_ABBRS_V6[c]: got_v[c] - int(exp["class_val_px"][c])
                              for c in range(cfg.N_CLASSES_V6) if got_v[c] != int(exp["class_val_px"][c])}
        out["n_train_val"] = [n_train, n_val, exp["n_train"], exp["n_val"]]
    return out


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

    log = setup_logger(rank=rank, name="train_v6")
    device = torch.device(f"cuda:{local_rank}" if torch.cuda.is_available() else "cpu")

    fold_spec = cfg.CV_FOLDS[args.fold] if args.fold is not None else None

    # ----- Model (decoder 무작위 초기값 — ckpt 를 읽지 않는다) -----
    # decoder 초기값은 모든 rank 가 같은 seed 로 만든다. 그 뒤의 난수 (dropout 등) 만 rank 마다 다르게 한다.
    torch.manual_seed(args.seed)
    model = DefSegModel(backbone_name=args.backbone, n_classes=cfg.N_CLASSES_V6).to(device)
    torch.manual_seed(args.seed + rank)
    np.random.seed(args.seed + rank)
    new_size = round_to_patch(args.img_size, model.patch_size)
    if new_size != args.img_size:
        if is_main(rank):
            log.warning(f"img_size {args.img_size} → {new_size}")
        args.img_size = new_size
    if args.run_name is None:
        args.run_name = cfg.default_run_name(args.backbone, args.img_size)

    ckpt_dir = cfg.run_ckpt_dir(args.run_name, args.fold)
    fig_dir = cfg.run_fig_dir(args.run_name, args.fold)
    if is_main(rank):
        ckpt_dir.mkdir(parents=True, exist_ok=True)
        fig_dir.mkdir(parents=True, exist_ok=True)
        log.info(
            f"=== v6 train ({cfg.SCHEMA_VERSION} 10-class, 사람 GT 전용, 단일 단계) ===  world_size={world_size}  "
            f"run_name={args.run_name}  fold={args.fold or '(전수 학습)'}"
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

    # ----- Data: 사람 GT 를 한 번 읽어 RAM 에 둔다 -----
    data_root = cfg.LEGACY_PBF_ROOT
    samples = enumerate_samples_v6(data_root)
    n_dataset = len(samples)
    if args.quick:
        samples = quick_subset(samples, fold_spec, per_group=3)
    if is_main(rank):
        log.info(f"데이터셋: {data_root}  ({n_dataset} 레이어, {cfg.SCHEMA_VERSION})")
        log.info(f"GT pool: {len(samples)} 표본을 {args.img_size}² 로 줄여 메모리에 올린다 …")
    t_pool = time.time()
    pool = GTPool(samples, args.img_size, log=log if is_main(rank) else None)
    train_idx, val_idx = split_by_fold(samples, pool.pix, fold_spec, allow_missing=args.quick)
    has_val = len(val_idx) > 0

    train_pix = pool.pix[train_idx]                     # (n_train, C) — 학습 Dataset 위치 순
    train_px_total = train_pix.sum(axis=0)
    val_px_total = pool.pix[val_idx].sum(axis=0) if has_val else np.zeros(cfg.N_CLASSES_V6, dtype=np.int64)
    # 결함 mIoU 에 넣는 클래스: 학습 화소도 있고 검증 화소도 있는 결함 클래스
    eval_defect = [
        c for c in cfg.DEFECT_IDS_V6
        if train_px_total[c] > 0 and val_px_total[c] > 0
    ]
    px_per_layer = args.img_size * args.img_size
    unlabeled_train = float(pool.unlabeled[train_idx].sum()) / (len(train_idx) * px_per_layer)
    unlabeled_val = float(pool.unlabeled[val_idx].sum()) / (len(val_idx) * px_per_layer) if has_val else None
    wiring = None if args.quick else _wiring_check(
        fold_spec, args.img_size, train_px_total, val_px_total, len(train_idx), len(val_idx),
    )

    # ----- Copy-Paste supplier (학습 분할만) -----
    train_ds = DefSegGTDatasetV6(
        pool, train_idx, training=True, cp_sampler=None, seed=args.seed, rank=rank,
    )
    use_cp = cfg.CP_ENABLE and not args.no_copy_paste
    cp_items: dict = {}
    if use_cp:
        cp_items = build_cp_supplier_items(
            train_pix, classes=cfg.CP_CLASSES,
            min_component_px=cfg.CP_MIN_COMPONENT_PX,
        )

        def _fetch_fn(key, _ds=train_ds):
            return _ds.fetch_local(int(key))

        train_ds.cp_sampler = CopyPasteSampler(
            items_per_class=cp_items, fetch_fn=_fetch_fn,
            classes=cfg.CP_CLASSES, prob=cfg.CP_PROB,
        )

    # ----- Sampler: 도메인 비중 × [그룹 균등 + 클래스 균등 혼합, 표본당 상한] -----
    # 그룹 = Peregrine 소스. 도메인이 하나뿐이면 (Peregrine 주석만) v5 의 식 그대로다.
    train_sources = [samples[i].group for i in train_idx]
    train_domains = [samples[i].domain for i in train_idx]
    source_names = sorted(set(train_sources))
    src_to_idx = {n: i for i, n in enumerate(source_names)}
    source_idx = np.array([src_to_idx[n] for n in train_sources], dtype=np.int64)
    weights, domain_share = compute_domain_sample_weights(
        train_pix, train_sources, train_domains,
        mix=cfg.SAMPLER_SOURCE_MIX, max_repeat=cfg.SAMPLER_MAX_REPEAT, diva_max=cfg.DOMAIN_SHARE_DIVA_MAX,
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

    # ----- Class weights (학습 분할, 학습 해상도 기준 화소 수) -----
    if args.class_weight_mode == "median_inv":
        class_weights = median_inv_class_weight(train_px_total, clip=cfg.CLASS_WEIGHT_CLIP).to(device)
    else:
        class_weights = sqrt_inv_class_weight(train_px_total, clip=cfg.CLASS_WEIGHT_CLIP).to(device)

    # ----- EMA (rank 0 만 — DDP 의 모든 rank 는 같은 가중치를 갖는다) -----
    use_ema = not args.no_ema
    ema_model = None
    if use_ema and is_main(rank):
        ema_model = AveragedModel(net, multi_avg_fn=get_ema_multi_avg_fn(args.ema_decay))
        ema_model.eval()

    # ----- Optim + Warmup + Cosine -----
    optim = torch.optim.AdamW(
        net.trainable_parameters(), lr=args.lr, weight_decay=cfg.WEIGHT_DECAY,
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
    base_config = {
        "version": "v6",
        "schema": cfg.SCHEMA_VERSION,
        "backbone": args.backbone,
        "n_classes": cfg.N_CLASSES_V6,
        "class_names": list(cfg.CLASS_NAMES_V6),
        "class_keys": list(cfg.CLASS_KEYS_V6),
        "orientation": f"recoat axis = image {cfg.CANONICAL_RECOAT_AXIS}",
        "img_size": args.img_size,
        "intermediate_layers": list(cfg.INTERMEDIATE_LAYERS),
        "decoder_channels": cfg.DECODER_CHANNELS,
        "class_weight_mode": args.class_weight_mode,
        "ema_decay": args.ema_decay if use_ema else None,
        "init_from": None,
        "label_source": "Peregrine human annotations converted to pbf-2026.11 (no DSCNN predictions)",
        "dataset": _dataset_info(data_root, n_dataset),
        "total_steps": total_steps,
        "warmup_steps": warmup_steps,
        "batch_size": args.batch_size,
        "world_size": world_size,
        "effective_batch": args.batch_size * world_size,
        "lr": args.lr,
        "weight_decay": cfg.WEIGHT_DECAY,
        "seed": args.seed,
        "fold": args.fold,
        "fold_spec": list(fold_spec) if fold_spec is not None else None,
        "n_train": len(train_idx),
        "n_val": len(val_idx),
        "copy_paste": bool(use_cp),
        "copy_paste_rule": {
            "classes": list(cfg.CP_CLASSES),
            "required_under": {str(k): v for k, v in cfg.CP_REQUIRED_UNDER.items()},
            "region_min_frac": cfg.CP_REGION_MIN_FRAC, "position_tries": cfg.CP_POSITION_TRIES,
            "anchor_on_required": cfg.CP_ANCHOR_ON_REQUIRED,
        } if use_cp else None,
        "sampler": {
            "source_mix": cfg.SAMPLER_SOURCE_MIX, "max_repeat": cfg.SAMPLER_MAX_REPEAT,
            "domain_share": domain_share,
        },
        "class_train_px": [int(v) for v in train_px_total],
    }
    curve_path = fig_dir / "curve.csv"
    curve_fields = ["step", "lr", "train_loss"]
    if has_val:
        for tag in ("raw", "ema"):
            curve_fields += [f"{tag}_acc", f"{tag}_miou", f"{tag}_defect_miou"]
            curve_fields += [f"{tag}_iou_{c}" for c in range(cfg.N_CLASSES_V6)]

    if is_main(rank):
        log.info(
            f"device={device}  img_size={args.img_size}  bs_per_gpu={args.batch_size}  "
            f"effective_batch={args.batch_size * world_size}  trainable={n_train_params/1e6:.2f}M  "
            f"n_classes={cfg.N_CLASSES_V6}  (pool {time.time()-t_pool:.0f}s)"
        )
        log.info(
            f"train={len(train_idx)} 레이어  val={len(val_idx)} 레이어  "
            f"source 분포(train)={dict(Counter(train_sources))}"
        )
        if has_val:
            log.info(f"val 소스 분포={dict(Counter(samples[i].group for i in val_idx))}")
            log.info(
                "결함 mIoU 에 넣는 클래스 (학습 · 검증 화소가 모두 있는 결함): "
                f"{[cfg.CLASS_ABBRS_V6[c] for c in eval_defect]}"
            )
            skipped = [cfg.CLASS_ABBRS_V6[c] for c in cfg.DEFECT_IDS_V6
                       if val_px_total[c] > 0 and train_px_total[c] == 0]
            if skipped:
                log.warning(f"검증에는 있지만 학습 화소가 0 인 클래스 (채점 제외): {skipped}")
        log.info(f"class px (train, {args.img_size}²): {[int(v) for v in train_px_total]}")
        if has_val:
            log.info(f"class px (val,   {args.img_size}²): {[int(v) for v in val_px_total]}")
        log.info(
            f"미레이블 화소: train {100 * unlabeled_train:.1f}%"
            + (f"  val {100 * unlabeled_val:.1f}%" if unlabeled_val is not None else "")
        )
        if wiring is not None and wiring["match"]:
            log.info(f"배선 확인: 클래스별 화소 수 · 레이어 수가 계획의 표 (splits/{wiring['split_file']}) 와 같다")
        elif wiring is not None:
            log.warning(
                f"배선 확인 실패: 클래스별 화소 수가 계획의 표 (splits/{wiring['split_file']}) 와 다르다 — "
                f"train 차이 {wiring['diff_train_px']}  val 차이 {wiring['diff_val_px']}. "
                "데이터셋을 다른 옵션 (--swelling, 판정표) 으로 다시 변환했다면 정상이고, 아니면 로더가 계획과 다른 것이다 (PLAN_v6 §7.1-1)."
            )
        log.info(
            f"class weights (mode={args.class_weight_mode}, clip={cfg.CLASS_WEIGHT_CLIP}): "
            f"{[round(float(v), 3) for v in class_weights.cpu()]}"
        )
        log.info(
            f"sampler: ESS={effective_sample_size(p_w):.1f}/{len(p_w)}  max={100*p_w.max():.2f}%  "
            f"top5={100*top5_share:.1f}%  by source="
            f"{ {cfg.SOURCE_SHORT.get(n, n): round(100*v, 1) for n, v in share_by_source.items()} }"
            f"  domain={ {d: round(v, 3) for d, v in domain_share.items()} }"
        )
        if use_cp:
            log.info(
                "Copy-Paste supplier (class, n 레이어): "
                f"{[(cfg.CLASS_ABBRS_V6[c], len(cp_items.get(c, []))) for c in cfg.CP_CLASSES]}  "
                f"위치 규칙 (밑에 있어야 하는 클래스): "
                f"{ {cfg.CLASS_ABBRS_V6[c]: cfg.CLASS_ABBRS_V6[r] for c, r in cfg.CP_REQUIRED_UNDER.items() if c in cfg.CP_CLASSES} }"
            )
        if use_ema and not args.quick and 1.0 / (1.0 - args.ema_decay) > total_steps / 10:
            log.warning(
                f"EMA 시상수 {1.0/(1.0-args.ema_decay):.0f} step 이 총 step 의 1/10 ({total_steps/10:.0f}) 보다 크다 — "
                f"저장되는 EMA 사본에 초기값이 {100*args.ema_decay**total_steps:.1f}% 남는다. --ema-decay 를 낮출 것."
            )
        log.info(
            f"AMP=OFF  total_steps={total_steps}  warmup={warmup_steps}  eval_every={args.eval_every}  "
            f"grad_clip={cfg.GRAD_CLIP_NORM}  ema={'%.4g' % args.ema_decay if use_ema else 'off'}"
        )
        with (fig_dir / "run.json").open("w") as fh:
            json.dump({
                "config": base_config,
                "args": vars(args),
                "train_samples": [samples[i].key for i in train_idx],
                "val_samples": [samples[i].key for i in val_idx],
                "class_names": list(cfg.CLASS_NAMES_V6),
                "class_abbrs": list(cfg.CLASS_ABBRS_V6),
                "class_train_px": [int(v) for v in train_px_total],
                "class_val_px": [int(v) for v in val_px_total],
                "class_train_layers": [int(v) for v in (train_pix > 0).sum(axis=0)],
                "class_val_layers": [int(v) for v in (pool.pix[val_idx] > 0).sum(axis=0)] if has_val else None,
                "unlabeled_frac": {"train": unlabeled_train, "val": unlabeled_val},
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
        cm_raw = accumulate_confusion(net, val_loader, device, cfg.N_CLASSES_V6)
        s_raw = summarize(cm_raw, eval_defect)
        if ema_model is not None:
            cm_ema = accumulate_confusion(ema_model.module, val_loader, device, cfg.N_CLASSES_V6)
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
    t_start = time.time()
    t_val_total = 0.0

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
            logits, label, weight=class_weights, ignore_index=cfg.IGNORE_INDEX,
        )
        if not torch.isfinite(loss):
            raise RuntimeError(f"[r{rank}] loss 가 유한하지 않다 (step {step + 1}): {loss.item()}")
        loss.backward()
        grad_norm = torch.nn.utils.clip_grad_norm_(
            net.trainable_parameters(), max_norm=cfg.GRAD_CLIP_NORM,
        )
        optim.step()
        sched.step()
        step += 1
        if ema_model is not None:
            ema_model.update_parameters(net)

        loss_val = loss.item()
        win_loss += loss_val; win_n += 1
        seg_loss += loss_val; seg_n += 1

        if is_main(rank) and (step % args.log_every == 0 or step == 1):
            elapsed = time.time() - t_start - t_val_total
            rate = step / max(elapsed, 1e-6)
            eta_min = (total_steps - step) / max(rate, 1e-6) / 60.0
            log.info(
                f"step [{step:6d}/{total_steps}] loss={win_loss / max(win_n, 1):.4f}  "
                f"lr={lr_now:.2e}  gnorm={grad_norm.item():.3f}  {rate:.2f} it/s  ETA {eta_min:.0f}m"
            )
            win_loss, win_n = 0.0, 0

        at_eval = (step % args.eval_every == 0) or (step == total_steps)
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
                    row += [s["iou"][c] for c in range(cfg.N_CLASSES_V6)]
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
