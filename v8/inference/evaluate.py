"""v8 채점 — 얇은 껍데기 (PLAN_v8 §5). 사람 GT 의 자리를 보정 세트로 바꿔 놓고 **v6 의 채점을 그대로 부른다.**

모델이 학습할 때 본 것과 같은 보정을 건 사진으로 채점해야 한다. 그래서 학습이 남긴 `norm_set.json` 의 세트와
`--norm-set` 이 다르면 멈춘다 (일부러 다른 세트로 채점하려면 `--allow-other-set` 과 `--tag-prefix` 를 함께 준다).

출력은 v6 의 채점과 같다: figures/<run>/v6/cv/<fold>/eval_{native,resized}_<which>/{metrics.json, …}

사용 (CWD = 3DP_VPPM/):
    python -m DefSeg_AM.v8.inference.evaluate --norm-set b115 --run-name vitb14_reg_dpt_dual_sz1036_10cls_v8-B-s42 --fold inlayer
"""
from __future__ import annotations

import argparse
import sys

from ...v6 import config_v6 as v6cfg
from ..data.norm_set import describe, read_run_record, use_norm_set


def main():
    ap = argparse.ArgumentParser(add_help=False, allow_abbrev=False)
    ap.add_argument("--norm-set", type=str, required=True, help="보정 세트의 이름 (cache/v8_norm/ 아래) 또는 폴더 경로")
    ap.add_argument("--allow-other-set", action="store_true", help="학습에 쓴 세트와 다른 세트로 채점하는 것을 허락한다 (--tag-prefix 를 함께 줄 것)")
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--fold", type=str, default=None)
    ap.add_argument("--tag-prefix", type=str, default=None)
    known, rest = ap.parse_known_args()
    if known.run_name is None:
        raise SystemExit("FATAL: --run-name 을 줄 것 (v8 의 run 이름은 꼬리표와 seed 로 정해진다)")

    root, doc = use_norm_set(known.norm_set)
    rec = read_run_record(v6cfg.run_fig_dir(known.run_name, known.fold))
    trained = rec.get("norm_set") if rec else None
    if trained != doc.get("name"):
        msg = (f"run {known.run_name} 은 보정 세트 '{trained}' 로 학습했는데 (기록이 없으면 None — 보정하지 않은 실행이다) "
               f"'{doc.get('name')}' 로 채점하려 한다")
        if not known.allow_other_set:
            raise SystemExit(f"FATAL: {msg}. 일부러 하는 것이면 --allow-other-set --tag-prefix <이름> 을 준다")
        if known.tag_prefix in (None, "eval"):
            raise SystemExit("FATAL: 다른 세트로 채점할 때는 --tag-prefix 를 eval 이 아닌 이름으로 준다 (본 채점의 폴더를 덮어쓰지 않게)")
        print(f"[v8] 주의: {msg}", flush=True)
    print(f"[v8] 보정 세트 {doc.get('name')} ({root}) 의 사진으로 채점한다 — {describe(doc)}", flush=True)

    argv = [sys.argv[0], "--run-name", known.run_name]
    if known.fold is not None:
        argv += ["--fold", known.fold]
    if known.tag_prefix is not None:
        argv += ["--tag-prefix", known.tag_prefix]
    sys.argv = argv + rest

    from ...v6.inference import evaluate as v6_eval
    v6_eval.main()


if __name__ == "__main__":
    main()
