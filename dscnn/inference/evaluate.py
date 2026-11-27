"""DSCNN 재구현을 사람 GT 로 채점한다 — v6 의 `inference/evaluate.py` 와 같은 표본 · 같은 지표 · 같은 파일 형식.

채점 해상도 (v6 과 같은 둘):
  native    원본 해상도. DSCNN 은 타일을 원본 해상도로 넣어 원본 해상도의 예측을 낸다 (줄였다 키우는 단계가 없다). **주 지표.**
  resized   원본 해상도의 예측과 정답을 둘 다 NEAREST 로 1036² 로 옮겨 견준다 (v6 의 1036² 채점과 같은 정답).

표본: --fold 의 검증 레이어 (학습이 남긴 figures/<run>/cv/<fold>/run.json 의 목록).
출력: figures/<run>/cv/<fold>/eval_{native,resized}_<which>/ — metrics.json, confusion_matrix.csv, per_class_metrics.csv, (옵션) panels/

사용 (CWD = 3DP_VPPM/. compose 로는 DSCNN_PHASE=eval):
    python -m DefSeg_AM.dscnn.inference.evaluate --fold inlayer
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import torch

from ...common.data.image_utils import resize_label
from ...v6 import config_v6 as v6cfg
from ...v6.data.data_gt_v6 import enumerate_samples_v6, load_sample_native
from ...v6.inference.evaluate import dump_scores, select_samples
from ...v6.inference.metrics import confusion_from_pred
from ...v6.inference.viz import save_panel

from .. import config_dscnn as cfg
from ..data.tiles import make_layer
from .predict import load_dscnn, photo_mean, predict_layer

NC = cfg.N_CLASSES


def resolve_mean_and_tile(group: str, ckpt_cfg: dict, photos_of_group) -> tuple[float, int]:
    """소스의 ū 와 타일 크기. 학습 때 본 소스면 ckpt 의 값을, 아니면 (loso 의 검증 소스) 그 소스의 사진에서 낸 평균을 쓴다."""
    means = ckpt_cfg.get("source_mean", {}) or {}
    tiles = ckpt_cfg.get("tile_size", {}) or {}
    mean = float(means[group]) if group in means else photo_mean(photos_of_group())
    t = int(tiles[group]) if group in tiles else cfg.tile_size_for(group)
    return mean, t


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--fold", type=str, default=None, choices=list(v6cfg.CV_FOLDS))
    ap.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    ap.add_argument("--tag", type=str, default="")
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--which", type=str, default="last", choices=["last", "best", "last_raw"])
    ap.add_argument("--checkpoint", type=str, default=None)
    ap.add_argument("--img-size", type=int, default=v6cfg.IMG_SIZE, help="resized 채점의 크기 (v6 과 같은 1036)")
    ap.add_argument("--tag-prefix", type=str, default="eval")
    ap.add_argument("--save-figs", type=int, default=0, help="비교 panel 을 이만큼 저장 (원본 해상도)")
    ap.add_argument("--max-layers", type=int, default=None, help="스모크용: 앞 N 레이어만")
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or cfg.default_run_name(args.seed, args.tag)
    fig_dir = cfg.run_fig_dir(run_name, args.fold)
    ckpt_path = Path(args.checkpoint) if args.checkpoint else cfg.run_ckpt_dir(run_name, args.fold) / f"{args.which}.pt"
    if not ckpt_path.exists():
        ap.error(f"ckpt not found: {ckpt_path}")
    model, ckpt = load_dscnn(ckpt_path, device)
    ccfg = ckpt.get("config", {}) or {}

    samples, defect_eval, _ = select_samples(
        enumerate_samples_v6(), args.fold, None,
        fold_run_json=(fig_dir / "run.json") if args.fold else None, img_size=args.img_size,
    )
    if args.max_layers is not None:
        samples = samples[: args.max_layers]
    if not samples:
        ap.error("채점할 표본이 없다")
    which = Path(ckpt_path).stem if args.checkpoint else args.which
    tags = {r: f"{args.tag_prefix}_{r}_{which}" for r in ("native", "resized")}
    print(f"{len(samples)} 레이어 채점 | fold={args.fold} | out={fig_dir}/{{{', '.join(tags.values())}}}")

    res = {r: {"total": np.zeros((NC, NC), dtype=np.int64), "by_group": {}} for r in tags}
    fig_idx = set(np.linspace(0, len(samples) - 1, min(args.save_figs, len(samples)), dtype=int).tolist()) if args.save_figs > 0 else set()
    cache: dict[str, tuple[float, int]] = {}
    for n, s in enumerate(samples):
        i0, i1, gt = load_sample_native(s)
        if s.group not in cache:
            def photos(_g=s.group):
                out = []
                for x in samples:
                    if x.group == _g:
                        a, b, _ = load_sample_native(x)
                        out += [a, b]
                return out
            cache[s.group] = resolve_mean_and_tile(s.group, ccfg, photos)
        mean, t = cache[s.group]
        layer = make_layer(i0, i1, gt, key=s.key, group=s.group, t=t, mean=mean, device=device)
        pred = predict_layer(model, layer)
        pairs = {
            "native": (pred, layer.label.long()),
            "resized": (
                torch.from_numpy(resize_label(pred.cpu().numpy(), args.img_size).astype(np.int64)).to(device),
                torch.from_numpy(resize_label(gt, args.img_size).astype(np.int64)).to(device),
            ),
        }
        for r, (p, g) in pairs.items():
            cm = confusion_from_pred(p, g, NC).reshape(NC, NC).cpu().numpy()
            res[r]["total"] += cm
            res[r]["by_group"][s.group] = res[r]["by_group"].get(s.group, 0) + cm
        if n in fig_idx:
            save_panel(
                i0, i1, gt, pred.cpu().numpy().astype(np.int16),
                out_path=fig_dir / tags["native"] / "panels" / f"{s.group}_{s.layer:07d}.png",
                title=s.key, ref_title="human GT (pbf-2026.11)", pred_title=f"DSCNN re-implementation ({ckpt_path.parent.name}/{ckpt_path.name})",
            )
        if (n + 1) % 10 == 0 or n + 1 == len(samples):
            print(f"  [{n + 1:3d}/{len(samples)}] 채점 …", flush=True)
        del layer, pred

    for r, tag in tags.items():
        meta = {
            "kind": "dscnn-reimpl", "checkpoint": str(ckpt_path), "ckpt_weights": ckpt.get("weights"), "ckpt_step": ckpt.get("step"),
            "run_name": run_name, "fold": args.fold, "tag": tag, "img_size": args.img_size, "resolution": r,
            "tta": False, "postprocess": False, "n_layers": len(samples), "samples": [s.key for s in samples],
            "tile_size": {g: v[1] for g, v in cache.items()}, "source_mean": {g: v[0] for g, v in cache.items()},
        }
        print(f"\n################ {tag} ({'원본 해상도' if r == 'native' else f'{args.img_size}²'}) ################")
        dump_scores(
            res[r]["total"], res[r]["by_group"], fig_dir / tag, list(v6cfg.CLASS_NAMES_V6), defect_eval, meta,
            title=f"{run_name} {cfg.run_tag(args.fold)} / {tag} (human GT, {r})",
        )


if __name__ == "__main__":
    main()
