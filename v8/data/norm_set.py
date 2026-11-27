"""보정 세트를 v6 · v7 의 코드가 읽게 하는 일 (PLAN_v8 §5) — 학습 · 채점 껍데기가 함께 쓴다.

v6 의 로더는 데이터셋의 자리를 `config_v6.LEGACY_PBF_ROOT` 에서 읽는다 (부를 때마다 읽는다). 그 값을 이 프로세스 안에서만
보정 세트의 폴더로 바꿔 놓으면, v6 · v7 의 코드를 고치지 않고 보정한 사진으로 학습 · 채점이 된다. 라벨은 원본과 같다.
"""
from __future__ import annotations

import json
from pathlib import Path

from ...v6 import config_v6 as v6cfg
from .. import config_v8 as cfg


def use_norm_set(name_or_path: str) -> tuple[Path, dict]:
    """보정 세트를 이 프로세스의 사람 GT 데이터셋으로 삼는다 → (세트 폴더, norm.json 의 내용).

    세트가 원본 데이터셋과 맞는지 (레이어 수, 원본을 변환한 때) 확인한다 — 원본을 다시 변환했는데 보정 세트가 옛것이면 멈춘다.
    """
    try:
        root = cfg.resolve_norm_root(name_or_path)
    except FileNotFoundError as e:
        raise SystemExit(str(e)) from None
    doc = json.loads((root / cfg.NORM_JSON).read_text(encoding="utf-8"))
    for f in ("index.csv", "classes.json"):
        if not (root / f).is_file():
            raise SystemExit(f"FATAL: 보정 세트 {root} 에 {f} 가 없다 — 다시 만든다 (DEFSEG_V8_PHASE=norm, DEFSEG_V8_EXTRA=--overwrite)")
    if not (root / "annotations").is_dir():
        raise SystemExit(f"FATAL: 보정 세트 {root} 의 annotations 가 없다 (링크가 깨졌다) — 다시 만든다")
    src = v6cfg.LEGACY_PBF_ROOT
    conv = src / "conversion.json"
    if conv.is_file() and doc.get("source_conversion_created_at") is not None:
        now = json.loads(conv.read_text(encoding="utf-8")).get("created_at")
        if now != doc["source_conversion_created_at"]:
            raise SystemExit(
                f"FATAL: 보정 세트 {root.name} 는 원본 데이터셋의 옛 변환 ({doc['source_conversion_created_at']}) 으로 만든 것이다 — "
                f"지금의 원본은 {now} 에 변환됐다. 보정 세트를 다시 만든다 (DEFSEG_V8_PHASE=norm, DEFSEG_V8_EXTRA=--overwrite)"
            )
    v6cfg.LEGACY_PBF_ROOT = root
    return root, doc


def describe(doc: dict) -> str:
    """norm.json → 한 줄 요약 (로그용)."""
    src = doc.get("sources", {})
    shifts = " ".join(f"{d.get('short', k)}{d['applied']['shift']:+.0f}" for k, d in src.items())
    maps = [d.get("short", k) for k, d in src.items() if d["applied"]["map"]]
    return f"mode={doc.get('mode')}  더한 값: {shifts}  조명 지도로 나눈 소스: {' · '.join(maps) if maps else '없음'}"


def write_run_record(fig_dir: Path, root: Path, doc: dict) -> None:
    """실행의 그림 폴더에 어느 보정 세트로 학습했는지 남긴다 — 채점할 때 같은 세트인지 확인하는 데 쓴다."""
    fig_dir.mkdir(parents=True, exist_ok=True)
    (fig_dir / cfg.RUN_NORM_JSON).write_text(
        json.dumps({"norm_set": doc.get("name"), "root": str(root), "norm": doc}, ensure_ascii=False, indent=1), encoding="utf-8")


def read_run_record(fig_dir: Path) -> dict | None:
    p = fig_dir / cfg.RUN_NORM_JSON
    return json.loads(p.read_text(encoding="utf-8")) if p.is_file() else None
