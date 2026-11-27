"""DSCNN 재구현을 사람 GT 만으로 학습한다 — v6 과 같은 데이터셋 · 같은 fold (GPU 1대).

학습 방식은 논문 (Scime et al. 2020, 3.7 절) 을 따른다:
  - 원본 해상도 타일 (소스마다 크기가 다르다 — config_dscnn.TILE_SIZE), 주석이 10% 이상인 타일만
  - Adam (step 1e-4, β 0.9 · 0.999, ε 1e-4), 스케줄 없음. 저장하는 가중치는 EMA (decay 0.9999)
  - 손실 = 화소별 cross-entropy × 클래스 가중치 w_k = Median({f}) / f_k, 미레이블 화소는 0
  - 증강 = 잡음 (없음 · 2 단계) × 밝기 이동 (없음 · ±10%) 의 9 가지를 타일마다 모두. 회전 · 반전 · Copy-Paste 없음
  - 초기값 CONV N(0, 0.05²) · FC N(0, 0.004²) · bias 0.1
논문에 없어 정한 것 (config_dscnn 의 "정함"): 여러 소스를 한 모델로 (한 배치에 여러 소스의 타일을 섞는다), 2022 판 소스의 타일 크기.

DSCNN 예측 (ORNL HDF5 의 segmentation_results) 은 읽지 않는다. Stage 1 · 사전 학습 가중치도 없다 (무작위 초기값).

검증 (fold 지정 시): --eval-every step 마다 떼어 둔 레이어 전체를 원본 해상도로 추론해 채점한다 (학습용 가중치 · EMA 둘 다).

산출물:
    checkpoints/<run>/cv/<fold>/{best.pt, last.pt, last_raw.pt}      (<run> = dscnn_native_10cls_gt-s<seed>)
    figures/<run>/cv/<fold>/{run.json, curve.csv, result.json}

사용 (CWD = 3DP_VPPM/. compose 로는 DSCNN_PHASE=cv):
    python -m DefSeg_AM.dscnn.training.train --fold inlayer
    python -m DefSeg_AM.dscnn.training.train --fold inlayer --quick
"""
from __future__ import annotations

import argparse
import copy
import csv
import json
import time
from collections import Counter
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

from ...common.models.losses import sqrt_inv_class_weight
from ...common.utils.log import setup_logger
from ...v6 import config_v6 as v6cfg
from ...v6.data.data_gt_v6 import enumerate_samples_v6, load_sample_native, quick_subset, split_by_fold
from ...v6.inference.evaluate import label_pix
from ...v6.inference.metrics import confusion_from_pred, summarize

from .. import config_dscnn as cfg
from ..data.tiles import (
    Layer, TileBank, combo_to_aug, make_layer, make_stacks, median_frequency_weights,
)
from ..inference.predict import predict_layer
from ..models.dscnn import DSCNN, count_parameters

AB = v6cfg.CLASS_ABBRS_V6
NC = cfg.N_CLASSES


def parse_args() -> argparse.Namespace:
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=str, default=None, choices=list(v6cfg.CV_FOLDS),
                    help="검증 fold (v6 과 같은 이름). 생략하면 125 레이어 전수 학습 (검증 없음)")
    ap.add_argument("--epochs", type=float, default=cfg.EPOCHS, help="한 epoch = 학습 타일 × 증강 9 가지")
    ap.add_argument("--max-steps", type=int, default=None, help="step 수를 직접 준다 (epochs 대신)")
    ap.add_argument("--eval-every", type=int, default=cfg.EVAL_EVERY)
    ap.add_argument("--lr", type=float, default=cfg.LR)
    ap.add_argument("--ema-decay", type=float, default=cfg.EMA_DECAY)
    ap.add_argument("--fc-relu", action="store_true", help="FC 뒤에 ReLU 를 넣는다 (논문의 표에는 없다)")
    ap.add_argument("--class-weight", choices=["median", "sqrt_inv"], default="median",
                    help="클래스 가중치. median = 논문의 중앙값 빈도 / 빈도 (상한 없음), sqrt_inv = v6 의 √역빈도 (상한 10) — 논문 방식이 아니다")
    ap.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--tag", type=str, default="", help="run 이름에 붙는 꼬리표")
    ap.add_argument("--log-every", type=int, default=200)
    ap.add_argument("--quick", action="store_true", help="스모크: 소스당 3 레이어, 30 step")
    return ap.parse_args()


class Ema:
    """파라미터의 지수 이동 평균. 처음에는 TensorFlow 처럼 decay 를 min(decay, (1 + n) / (10 + n)) 로 낮춰 초기값이 남지 않게 한다."""

    def __init__(self, model: torch.nn.Module, decay: float):
        self.decay = float(decay)
        self.n = 0
        self.shadow = {k: p.detach().clone() for k, p in model.named_parameters()}

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.n += 1
        d = min(self.decay, (1.0 + self.n) / (10.0 + self.n))
        for k, p in model.named_parameters():
            self.shadow[k].mul_(d).add_(p.detach(), alpha=1.0 - d)

    @torch.no_grad()
    def copy_to(self, target: torch.nn.Module, source: torch.nn.Module) -> None:
        """target ← EMA 파라미터 + source 의 buffer (BatchNorm 의 이동 평균 · 분산)."""
        state = {k: v.clone() for k, v in source.state_dict().items()}
        state.update({k: v.clone() for k, v in self.shadow.items()})
        target.load_state_dict(state)


@torch.no_grad()
def score_layers(model: DSCNN, layers: list[Layer]) -> np.ndarray:
    """레이어들을 원본 해상도로 추론 → (C, C) confusion (rows = 정답)."""
    model.eval()
    cm = torch.zeros(NC * NC, dtype=torch.long, device=layers[0].img.device)
    for L in layers:
        cm += confusion_from_pred(predict_layer(model, L), L.label.long(), NC)
    return cm.reshape(NC, NC).cpu().numpy()


def source_means(samples, native, train_idx) -> dict[str, float]:
    """소스의 ū (논문 식 1 의 "보정할 때 정한 전체 평균 밝기") — 그 소스의 학습 레이어 두 사진의 화소 평균.

    학습 레이어가 없는 소스 (loso fold 의 검증 소스) 는 그 소스의 사진 전부로 낸다. 라벨을 쓰지 않는 값이다.
    """
    train_set = set(train_idx)
    acc_train: dict[str, list[float]] = {}
    acc_all: dict[str, list[float]] = {}
    for i, s in enumerate(samples):
        tot = float(native[i][0].sum(dtype=np.float64) + native[i][1].sum(dtype=np.float64))
        cnt = 2.0 * native[i][0].size
        for acc in ([acc_all, acc_train] if i in train_set else [acc_all]):
            a = acc.setdefault(s.group, [0.0, 0.0])
            a[0] += tot
            a[1] += cnt
    return {g: (acc_train[g][0] / acc_train[g][1] if g in acc_train else v[0] / v[1]) for g, v in acc_all.items()}


def _nan_to_none(x) -> float | None:
    x = float(x)
    return None if x != x else x


def _summary_json(s: dict) -> dict:
    return {
        "acc": _nan_to_none(s["acc"]), "miou": _nan_to_none(s["miou"]), "defect_miou": _nan_to_none(s["defect_miou"]),
        "iou": [_nan_to_none(v) for v in s["iou"]], "defect_classes": [int(c) for c in s["defect_classes"]],
    }


def _iou_line(s: dict) -> str:
    return "  ".join(f"{AB[c]} {s['iou'][c]:.3f}" for c in range(NC) if s["iou"][c] == s["iou"][c])


def main():
    args = parse_args()
    if args.quick:
        args.max_steps = 30
        args.eval_every = 15
        args.log_every = 5
        args.ema_decay = 0.5
        if not args.tag:
            args.tag = "quick"

    log = setup_logger(rank=0, name="train_dscnn")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    rng = np.random.default_rng(args.seed)
    gen = torch.Generator(device=device)
    gen.manual_seed(args.seed)

    fold_spec = v6cfg.CV_FOLDS[args.fold] if args.fold is not None else None
    run_name = args.run_name or cfg.default_run_name(args.seed, args.tag)
    ckpt_dir = cfg.run_ckpt_dir(run_name, args.fold)
    fig_dir = cfg.run_fig_dir(run_name, args.fold)
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    fig_dir.mkdir(parents=True, exist_ok=True)
    log.info(f"=== DSCNN 재구현 학습 ({v6cfg.SCHEMA_VERSION} 10-class, 사람 GT 전용) ===  run={run_name}  fold={args.fold or '(전수 학습)'}")
    log.info(f"args: {vars(args)}")

    # ----- 데이터: v6 과 같은 데이터셋 · 같은 분할 -----
    data_root = v6cfg.LEGACY_PBF_ROOT
    samples = enumerate_samples_v6(data_root)
    n_dataset = len(samples)
    if args.quick:
        samples = quick_subset(samples, fold_spec, per_group=3)
    pix_ref = label_pix(samples, v6cfg.IMG_SIZE)          # v6 과 같은 기준 (1036²) — fold 와 "결함 mIoU 에 넣는 클래스" 를 v6 과 똑같이 정한다
    train_idx, val_idx = split_by_fold(samples, pix_ref, fold_spec, allow_missing=args.quick)
    has_val = len(val_idx) > 0
    tp, vp = pix_ref[train_idx].sum(axis=0), (pix_ref[val_idx].sum(axis=0) if has_val else np.zeros(NC, dtype=np.int64))
    eval_defect = [c for c in v6cfg.DEFECT_IDS_V6 if tp[c] > 0 and vp[c] > 0]

    log.info(f"데이터셋: {data_root} ({n_dataset} 레이어) — {len(samples)} 레이어를 원본 해상도로 읽는다 …")
    t0 = time.time()
    native = [load_sample_native(s) for s in samples]
    source_mean = source_means(samples, native, train_idx)
    tile_size = {g: cfg.tile_size_for(g) for g in source_mean}

    layers = [
        make_layer(native[i][0], native[i][1], native[i][2], key=s.key, group=s.group,
                   t=tile_size[s.group], mean=source_mean[s.group], device=device)
        for i, s in enumerate(samples)
    ]
    del native
    train_layers = [layers[i] for i in train_idx]
    val_layers = [layers[i] for i in val_idx]
    bank = TileBank(train_layers)
    class_px = bank.class_px()
    if args.class_weight == "median":
        weights_np = median_frequency_weights(class_px)
    else:
        weights_np = sqrt_inv_class_weight(class_px, clip=v6cfg.CLASS_WEIGHT_CLIP).numpy()
    class_weights = torch.from_numpy(weights_np).to(device)
    log.info(f"  읽기 {time.time() - t0:.0f}s  GPU 메모리 {torch.cuda.memory_allocated() / 2**30:.1f} GB" if device.type == "cuda" else f"  읽기 {time.time() - t0:.0f}s")

    # ----- 모델 -----
    model = DSCNN(n_classes=NC, in_ch=cfg.IN_CHANNELS, fc_relu=args.fc_relu).to(device)
    n_par = count_parameters(model)
    optim = torch.optim.Adam(model.parameters(), lr=args.lr, betas=cfg.ADAM_BETAS, eps=cfg.ADAM_EPS)
    ema = Ema(model, args.ema_decay)
    ema_model = copy.deepcopy(model)

    steps_per_epoch = bank.steps_per_epoch()
    total_steps = int(args.max_steps) if args.max_steps is not None else int(round(args.epochs * steps_per_epoch))

    groups = sorted(set(bank.group))
    tile_stats = {}
    for g in groups:
        n_tiles = sum(1 for x in bank.group if x == g)
        tile_stats[g] = {
            "tile_size": tile_size[g], "mean": round(source_mean[g], 2),
            "train_layers": sum(1 for L in train_layers if L.group == g), "train_tiles": n_tiles,
        }
    base_config = {
        "arch": "dscnn", "version": "dscnn-reimpl",
        "paper": "Scime et al., Additive Manufacturing 36 (2020) 101453",
        "schema": v6cfg.SCHEMA_VERSION, "n_classes": NC, "in_channels": cfg.IN_CHANNELS, "fc_relu": bool(args.fc_relu),
        "class_names": list(v6cfg.CLASS_NAMES_V6), "class_keys": list(v6cfg.CLASS_KEYS_V6),
        "tile_size": tile_size, "source_mean": source_mean,
        "label_source": "Peregrine human annotations converted to pbf-2026.11 (no DSCNN predictions, no pretraining)",
        "dataset": {"root": str(data_root), "n_layers": n_dataset},
        "fold": args.fold, "fold_spec": list(fold_spec) if fold_spec is not None else None, "seed": args.seed,
        "n_train": len(train_idx), "n_val": len(val_idx),
        "optimizer": {"name": "Adam", "lr": args.lr, "betas": list(cfg.ADAM_BETAS), "eps": cfg.ADAM_EPS},
        "ema_decay": args.ema_decay, "epochs": args.epochs if args.max_steps is None else None,
        "steps_per_epoch": steps_per_epoch, "total_steps": total_steps,
        "augmentation": {"noise_var": list(cfg.NOISE_VARS), "shift_frac": list(cfg.SHIFT_FRACS), "flip_rotate": False},
        "batch": {"mixed_sources": True, "max_tiles": cfg.BATCH_MAX_TILES, "pixel_budget": cfg.BATCH_PIXEL_BUDGET},
        "class_weight": "median frequency (no clip)" if args.class_weight == "median" else f"sqrt inverse frequency (clip {v6cfg.CLASS_WEIGHT_CLIP})",
        "min_labeled_frac": cfg.MIN_LABELED_FRAC,
        "parameters": n_par,
    }

    log.info(f"device={device}  파라미터 {n_par['learnable']:,} + BN 통계 {n_par['bn_stats']:,} = {n_par['total']:,}")
    log.info(f"train={len(train_idx)} 레이어  val={len(val_idx)} 레이어  소스 (train)={dict(Counter(L.group for L in train_layers))}")
    log.info(f"학습 타일 {len(bank)} / 격자 {bank.n_grid} (주석 {100 * cfg.MIN_LABELED_FRAC:.0f}% 미만 제외)  ·  1 epoch = {steps_per_epoch} step  ·  총 {total_steps} step")
    for g in groups:
        st = tile_stats[g]
        log.info(f"  {v6cfg.SOURCE_SHORT.get(g, g):5s} T={st['tile_size']:4d}  ū={st['mean']:6.1f}  레이어 {st['train_layers']:2d}  타일 {st['train_tiles']:5d}")
    log.info(f"클래스 화소 (학습 타일, 원본 해상도): {[int(v) for v in class_px]}")
    log.info(f"클래스 가중치 ({'중앙값 빈도' if args.class_weight == 'median' else '√역빈도, 상한 10 — 논문 방식 아님'}): {[round(float(v), 3) for v in weights_np]}")
    if has_val:
        log.info(f"결함 mIoU 에 넣는 클래스: {[AB[c] for c in eval_defect]}  (v6 과 같은 규칙)")

    with (fig_dir / "run.json").open("w") as fh:
        json.dump({
            "config": base_config, "args": vars(args),
            "train_samples": [samples[i].key for i in train_idx],
            "val_samples": [samples[i].key for i in val_idx],
            "class_names": list(v6cfg.CLASS_NAMES_V6), "class_abbrs": list(AB),
            "defect_eval_classes": [int(c) for c in eval_defect],
            "class_train_px_tiles": [int(v) for v in class_px],
            "class_weights": [round(float(v), 4) for v in weights_np],
            "tiles": tile_stats, "n_train_tiles": len(bank), "n_grid_tiles": bank.n_grid,
        }, fh, ensure_ascii=False, indent=1)

    curve_path = fig_dir / "curve.csv"
    fields = ["step", "epoch", "train_loss"]
    if has_val:
        for tag in ("raw", "ema"):
            fields += [f"{tag}_acc", f"{tag}_miou", f"{tag}_defect_miou"] + [f"{tag}_iou_{c}" for c in range(NC)]
    with curve_path.open("w", newline="") as fh:
        csv.writer(fh).writerow(fields)

    def save_ckpt(path: Path, module: torch.nn.Module, kind: str, step: int, val: dict | None) -> None:
        torch.save({
            "model_state": {k: v.detach().cpu().clone() for k, v in module.state_dict().items()},
            "config": base_config, "weights": kind, "step": step, "val": val,
        }, path)

    # ----- 학습 -----
    best_score, best_step, last_val = -1.0, -1, None
    step, epoch = 0, 0
    win_loss, win_n, seg_loss, seg_n = 0.0, 0, 0.0, 0
    t_start, t_val = time.time(), 0.0
    model.train()
    done = False
    while not done:
        for tile_ids, combo in bank.epoch_batches(rng):
            # 한 배치에 여러 소스가 섞여 있다. regional · global 은 입력 크기가 늘 같아 한꺼번에 돌리고 (BatchNorm 이 섞인 배치를 본다),
            # localization · 분류는 타일 크기가 같은 것끼리 돈다.
            ts = bank.tile_t[tile_ids]
            parts = []
            for t in np.unique(ts):
                m = ts == t
                noise_var, shift = combo_to_aug(combo[m])
                parts.append(make_stacks(train_layers, bank.items[tile_ids[m]], noise_var=noise_var, shift=shift, generator=gen))
            reg = model.regional(torch.cat([p["c"] for p in parts]))
            glo = model.glob(torch.cat([p["a"] for p in parts]))
            num, den, o = 0.0, 0, 0
            for p in parts:
                n_p = p["b"].shape[0]
                logits = model.head(p["b"], p["d"], reg[o: o + n_p], glo[o: o + n_p])
                o += n_p
                label = p["label"]
                labeled = label >= 0
                ce = F.cross_entropy(logits, label, ignore_index=cfg.IGNORE_INDEX, reduction="none")
                num = num + (ce * class_weights[label.clamp(min=0)] * labeled).sum()
                den = den + labeled.sum()
            loss = num / den.clamp(min=1)
            if not torch.isfinite(loss):
                raise RuntimeError(f"loss 가 유한하지 않다 (step {step + 1}): {loss.item()}")
            optim.zero_grad(set_to_none=True)
            loss.backward()
            optim.step()
            ema.update(model)
            step += 1

            lv = loss.item()
            win_loss += lv; win_n += 1
            seg_loss += lv; seg_n += 1
            if step % args.log_every == 0 or step == 1:
                rate = step / max(time.time() - t_start - t_val, 1e-6)
                mem = f"  mem {torch.cuda.max_memory_allocated() / 2**30:.1f} GB" if device.type == "cuda" else ""
                log.info(f"step [{step:6d}/{total_steps}] epoch {step / steps_per_epoch:.2f}  loss={win_loss / max(win_n, 1):.4f}  "
                         f"{rate:.2f} it/s  ETA {(total_steps - step) / max(rate, 1e-6) / 60:.0f}m{mem}")
                win_loss, win_n = 0.0, 0

            if step % args.eval_every == 0 or step == total_steps:
                row = [step, f"{step / steps_per_epoch:.3f}", f"{seg_loss / max(seg_n, 1):.6f}"]
                if has_val:
                    tv = time.time()
                    ema.copy_to(ema_model, model)
                    s_raw = summarize(score_layers(model, val_layers), eval_defect)
                    s_ema = summarize(score_layers(ema_model, val_layers), eval_defect)
                    model.train()
                    t_val += time.time() - tv
                    last_val = _summary_json(s_ema)
                    sel = s_ema["defect_miou"] if s_ema["defect_miou"] == s_ema["defect_miou"] else s_ema["miou"]
                    log.info(f"step {step:6d} VAL (원본 해상도)  train_loss={seg_loss / max(seg_n, 1):.4f}  "
                             f"raw: acc={s_raw['acc']:.4f} mIoU={s_raw['miou']:.4f} defect={s_raw['defect_miou']:.4f}  |  "
                             f"ema: acc={s_ema['acc']:.4f} mIoU={s_ema['miou']:.4f} defect={s_ema['defect_miou']:.4f}  (val {time.time() - tv:.0f}s)")
                    log.info(f"  iou(ema): {_iou_line(s_ema)}")
                    for s in (s_raw, s_ema):
                        row += [s["acc"], s["miou"], s["defect_miou"]] + [s["iou"][c] for c in range(NC)]
                    row = ["" if (isinstance(v, float) and v != v) else v for v in row]
                    if sel == sel and sel > best_score:
                        best_score, best_step = float(sel), step
                        save_ckpt(ckpt_dir / "best.pt", ema_model, "ema", step, last_val)
                        log.info(f"  ↑ saved best (EMA, defect mIoU={sel:.4f})")
                with curve_path.open("a", newline="") as fh:
                    csv.writer(fh).writerow(row)
                seg_loss, seg_n = 0.0, 0
            if step >= total_steps:
                done = True
                break
        epoch += 1

    wall = time.time() - t_start
    ema.copy_to(ema_model, model)
    save_ckpt(ckpt_dir / "last.pt", ema_model, "ema", step, last_val)
    save_ckpt(ckpt_dir / "last_raw.pt", model, "raw", step, None)
    with (fig_dir / "result.json").open("w") as fh:
        json.dump({
            "run_name": run_name, "fold": args.fold, "steps": step, "epochs": step / steps_per_epoch,
            "wall_minutes": wall / 60.0, "train_it_per_s": step / max(wall - t_val, 1e-6),
            "best_step": best_step if has_val else None,
            "best_defect_miou_ema_native": best_score if has_val and best_step > 0 else None,
            "final_val_ema_native": last_val,
        }, fh, ensure_ascii=False, indent=1)
    log.info(f"saved last (EMA, step {step}) → {ckpt_dir / 'last.pt'}")
    if has_val:
        log.info(f"done. {step} step ({step / steps_per_epoch:.1f} epoch), {wall / 60:.1f}m.  best defect mIoU(ema)={best_score:.4f} at step {best_step}  "
                 f"final={last_val['defect_miou'] if last_val else None}")
    else:
        log.info(f"done. {step} step, {wall / 60:.1f}m. (전수 학습 — 검증 없음)")


if __name__ == "__main__":
    main()
