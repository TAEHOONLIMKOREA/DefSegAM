"""v8 학습 — 얇은 껍데기 (PLAN_v8 §5). 사람 GT 의 자리를 보정 세트로 바꿔 놓고 **v7 의 학습을 그대로 부른다.**

v8 에서 달라지는 것은 모델에 들어가는 사진뿐이다 (보정 세트 = 밝기를 맞추고 조명을 걷어낸 사진, 라벨은 원본 그대로).
학습 코드는 기준선 (v7 의 `N` 실행) 과 같은 것이다 — 모델 · 손실 · 길이 · 증강 · Copy-Paste · 추출 · fold · 검증.
**생성 데이터는 쓰지 않는다** (`--synth-set` 을 주면 멈춘다).

껍데기가 하는 일:
  1. `--norm-set <이름>` 의 보정 세트를 이 프로세스의 사람 GT 데이터셋으로 삼는다 (v6 설정의 `LEGACY_PBF_ROOT` 를 바꾼다)
  2. run 이름을 `<backbone>_dpt_dual_sz<img>_10cls_v8-<꼬리표>-s<seed>` 로 준다
  3. 그림 폴더에 `norm_set.json` 을 남긴다 (어느 보정 세트로 학습했는가)
  4. 나머지 인자를 그대로 넘겨 `DefSeg_AM.v7.training.train` 의 `main()` 을 부른다

보정 세트 `raw` (화소가 원본과 같다) 로 돌리면 v7 의 `N` 실행과 같은 계산이다 — 껍데기가 다른 것을 바꾸지 않았다는 스모크가 이것이다.

산출물 (v7 과 같은 배치 — 폴더는 v6 의 함수가 만든다):
    checkpoints/<run>/cv/<fold>/{best.pt, last.pt}
    figures/<run>/v6/cv/<fold>/{run.json, curve.csv, result.json, norm_set.json}

사용 (CWD = 3DP_VPPM/) — 풀런은 compose 로 돌린다 (docker/README.md):
    torchrun --nproc_per_node=4 -m DefSeg_AM.v8.training.train --norm-set b115 --variant B --fold inlayer
    torchrun --nproc_per_node=1 -m DefSeg_AM.v8.training.train --norm-set b115 --variant B --fold inlayer --quick
"""
from __future__ import annotations

import argparse
import os
import sys

from ...common.models.model import round_to_patch
from ...v6 import config_v6 as v6cfg
from .. import config_v8 as cfg
from ..data.norm_set import describe, use_norm_set, write_run_record

DINO_PATCH = 14          # DINOv2 의 patch 크기 — v7 의 학습이 img_size 를 이 배수로 맞춘다 (run 이름에 그 값이 들어간다)


def main():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--norm-set", type=str, required=True, help="보정 세트의 이름 (cache/v8_norm/ 아래) 또는 폴더 경로")
    ap.add_argument("--variant", type=str, default=None, help="꼬리표 — run 이름에 붙는다 (기본: 보정 세트의 mode — B · BL · raw)")
    # 아래는 run 이름을 정하려고 읽기만 한다 — 그대로 v7 의 학습에 넘긴다
    ap.add_argument("--fold", type=str, default=None)
    ap.add_argument("--seed", type=int, default=v6cfg.RANDOM_SEED)
    ap.add_argument("--backbone", type=str, default=v6cfg.DINO_BACKBONE)
    ap.add_argument("--img-size", type=int, default=v6cfg.IMG_SIZE)
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--quick", action="store_true")
    ap.add_argument("--synth-set", type=str, default=None)
    known, rest = ap.parse_known_args()
    if known.synth_set:
        raise SystemExit("FATAL: v8 은 생성 데이터를 쓰지 않는다 (--synth-set 을 줄 수 없다) — 전처리만 다른 비교가 깨진다")

    root, doc = use_norm_set(known.norm_set)
    variant = known.variant or str(doc.get("mode"))
    img_size = round_to_patch(224 if known.quick else known.img_size, DINO_PATCH)
    run_name = known.run_name or cfg.default_run_name(known.backbone, img_size, variant, known.seed)

    if int(os.environ.get("RANK", "0")) == 0:
        print(f"[v8] 보정 세트 {doc.get('name')} ({root}) 를 사람 GT 로 읽는다 — {describe(doc)}", flush=True)
        print(f"[v8] run_name={run_name}  (학습 코드는 v7 의 것 그대로, 생성 데이터 없음)", flush=True)
        write_run_record(v6cfg.run_fig_dir(run_name, known.fold), root, doc)

    argv = [sys.argv[0], "--variant", variant, "--run-name", run_name, "--seed", str(known.seed),
            "--backbone", known.backbone, "--img-size", str(known.img_size)]
    if known.fold is not None:
        argv += ["--fold", known.fold]
    if known.quick:
        argv += ["--quick"]
    sys.argv = argv + rest

    from ...v7.training import train as v7_train          # 여기서 불러야 위의 FATAL 이 torch 를 올리기 전에 난다
    v7_train.main()


if __name__ == "__main__":
    main()
