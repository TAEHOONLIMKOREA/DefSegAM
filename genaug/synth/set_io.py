"""합성 세트 폴더를 읽는 작은 함수들 — 점검 스크립트 (`checks/`) 가 쓴다.

세트의 구성 (사람 GT 데이터셋과 같다 + 내력):
    <세트>/visible/{0,1}/{id}.png   그린 사진 (가장자리 5% 는 그 레이어의 진짜 화소)
    <세트>/annotations/{id}.npy     라벨 (int8, 0–9, 미레이블 -1 — 가장자리와 파트 경계)
    <세트>/meta/{id}.json           레이어 한 장의 내력 (배경, 넣은 결함, seed)
    <세트>/index.csv, classes.json  v6 · v7 의 로더가 읽는 목록
    <세트>/set_config.json          만든 설정 (같은 설정으로만 이어 만든다)
    <세트>/manifest.json            세트 전체의 내력과 통계 (다 만든 뒤에 쓴다)
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from PIL import Image

from .. import config_genaug as cfg


def resolve_set(name_or_path: str) -> Path:
    p = Path(name_or_path)
    if (p / "manifest.json").is_file():
        return p.resolve()
    q = cfg.set_dir(name_or_path)
    if (q / "manifest.json").is_file():
        return q
    raise FileNotFoundError(f"FATAL: 합성 세트 '{name_or_path}' 없음 (또는 만드는 중이다 — manifest.json 이 없다): {q}")


def read_manifest(root: Path) -> dict:
    return json.loads((root / "manifest.json").read_text(encoding="utf-8"))


def read_layer(root: Path, sid: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """→ (visible/0, visible/1, 라벨) — uint8, uint8, int8."""
    i0 = np.array(Image.open(root / "visible" / "0" / f"{sid}.png"))
    i1 = np.array(Image.open(root / "visible" / "1" / f"{sid}.png"))
    return i0, i1, np.load(root / "annotations" / f"{sid}.npy")


def pick_layers(manifest: dict, n: int, seed: int = 0) -> list[dict]:
    """세트에서 n 장을 빌드가 고르게 섞이도록 고른다 (n <= 0 이면 전부)."""
    layers = manifest["layers"]
    if n <= 0 or n >= len(layers):
        return list(layers)
    idx = np.random.default_rng(seed).permutation(len(layers))[:n]
    return [layers[int(i)] for i in sorted(idx)]
