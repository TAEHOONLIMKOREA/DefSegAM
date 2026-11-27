"""v7 설정 — v6 에 더하는 값만 둔다 (PLAN_v7 §0, §3).

v7 = v6 + 합성 도메인 하나. 클래스 · 모델 · 길이 · 스케줄 · 증강 · Copy-Paste · 손실 · 채점의 값은 **v6 의 config 를 그대로 쓴다**
(`from ..v6 import config_v6`). 여기에는 합성 도메인의 이름 · 몫 · 세트의 자리 · run 이름만 있다.

이 폴더는 저장소의 규칙 ("버전끼리 임포트하지 않는다") 의 예외다 — v6 와 클래스 · 학습 방식이 같은 덧붙임이라서 v6 를 불러 쓴다.
"""
from __future__ import annotations

from pathlib import Path

from ..v6 import config_v6 as v6cfg

VERSION = "v7"

# 합성 도메인 — v6 의 "legacy" (Peregrine 주석) · "diva" 에 이은 세 번째 도메인
DOMAIN_SYNTH = "synth"
SYNTH_GROUP_PREFIX = "synth_"          # 합성 세트의 index.csv 의 process_code 는 이것으로 시작해야 한다 (사람 GT 의 키와 겹치지 않게)
SYNTH_SHARE = 0.3                      # 표본 추출에서 합성 도메인의 몫 (제안값 — PLAN_v7 §3). 사람 GT 를 뽑는 횟수가 그만큼 준다
SYNTH_SETS_DIR = v6cfg.OUTPUT_DIR / "cache" / "genaug" / "sets"     # 합성 세트가 놓이는 곳 (genaug 가 만든다)

# 실행의 꼬리표 — run 이름에 붙는다. N = 합성 없음 (기준선), W = 라벨 맵으로 그린 레이어 (PLAN_v7 §4.2)
DEFAULT_VARIANT = "N"


def resolve_synth_root(name_or_path: str) -> Path:
    """`--synth-set` 의 값 → 세트 폴더. 폴더 경로를 주거나, `cache/genaug/sets/` 아래의 이름을 준다."""
    p = Path(name_or_path)
    if p.is_dir():
        return p.resolve()
    q = SYNTH_SETS_DIR / name_or_path
    if q.is_dir():
        return q
    raise FileNotFoundError(f"FATAL: 합성 세트 '{name_or_path}' 없음 — 폴더 경로도 아니고 {SYNTH_SETS_DIR} 아래에도 없다")


def default_run_name(backbone: str = v6cfg.DINO_BACKBONE, img_size: int = v6cfg.IMG_SIZE,
                     variant: str = DEFAULT_VARIANT, seed: int = v6cfg.RANDOM_SEED) -> str:
    """`<backbone>_dpt_dual_sz<img>_10cls_v7-<꼬리표>-s<seed>` (예: vitb14_reg_dpt_dual_sz1036_10cls_v7-W-s42).

    ckpt 와 그림 폴더는 v6 의 함수 (`run_ckpt_dir` · `run_fig_dir`) 로 만든다 — v6 의 채점 · 요약 스크립트가 `--run-name` 만으로 찾게.
    그래서 그림은 `figures/<run>/v6/…` 에 쌓인다 (v6 코드가 그 이름을 쓴다).
    """
    base = v6cfg.default_run_name(backbone, img_size)
    if not base.endswith("_v6"):
        raise RuntimeError(f"v6 의 run 이름 규칙이 바뀌었다: {base}")
    return f"{base[:-3]}_v7-{variant}-s{int(seed)}"
