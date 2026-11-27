"""사람 GT 풀 — 변환해 둔 레이블 데이터셋에서 생성 모델이 봐도 되는 레이어만 읽는다 (PLAN_GENAUG §3.2, §5.1).

읽는 곳: `ORNL_Data/DSCNN_Dataset_pbf-2026.11/` (`visible/{0,1}/{id}.png`, `annotations/{id}.npy`, `index.csv`, `classes.json`).
v6 의 `GTPool` 과 같은 방법으로 1036² 로 줄여 RAM 에 둔다 (사진 BILINEAR, 라벨 NEAREST) — 분할 모델이 보는 것과 같은 격자다.

**누출 금지**: `splits/holdout_v1.json` 의 레이어는 읽지 않는다. 뺀 뒤의 레이어 수가 기대값 (87 또는 103) 과 다르면 멈춘다.
"""
from __future__ import annotations

import csv
import json
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image
from scipy import ndimage

from ...common.data.image_utils import resize_image_uint8, resize_label
from .. import config_genaug as cfg

_CC8 = np.ones((3, 3), dtype=bool)
# 객체 표의 열
O_SAMPLE, O_CLS, O_Y0, O_Y1, O_X0, O_X1, O_AREA, O_SY, O_SX = range(9)


def load_holdout(kind: str) -> list[str]:
    """holdout 종류 ("both" / "inlayer") → 읽지 않을 레이어의 key 목록."""
    spec = json.loads(cfg.HOLDOUT_FILE.read_text(encoding="utf-8"))
    if kind not in spec["sets"]:
        raise ValueError(f"FATAL: holdout 종류는 {list(spec['sets'])} 가운데 하나다: {kind!r}")
    return list(spec["sets"][kind]["keys"])


def check_schema(root: Path) -> None:
    path = root / "classes.json"
    if not path.is_file():
        raise FileNotFoundError(f"FATAL: {path} 없음 — 변환한 레이블 데이터셋이 아니다 (v6 의 convert_legacy 가 만든다)")
    meta = json.loads(path.read_text(encoding="utf-8"))
    keys = [c["key"] for c in meta.get("classes", [])]
    ids = [c["id"] for c in meta.get("classes", [])]
    if meta.get("schema") != cfg.SCHEMA_VERSION or keys != cfg.CLASS_KEYS or ids != list(range(cfg.N_CLASSES)):
        raise RuntimeError(f"FATAL: {path} 의 클래스가 config_genaug ({cfg.SCHEMA_VERSION}) 와 다르다: {meta.get('schema')} {keys}")
    if meta.get("unlabeled") != cfg.IGNORE_INDEX:
        raise RuntimeError(f"FATAL: {path} 의 미레이블 값이 {cfg.IGNORE_INDEX} 이 아니다")


def _load_one(args):
    root, sid, size = args
    ann = np.load(root / "annotations" / f"{sid}.npy")
    if ann.ndim != 2 or int(ann.min()) < cfg.IGNORE_INDEX or int(ann.max()) >= cfg.N_CLASSES:
        raise ValueError(f"FATAL: 라벨 값이 0–{cfg.N_CLASSES - 1} · -1 밖이다: {sid}")
    out = []
    for v in ("0", "1"):
        img = Image.open(root / "visible" / v / f"{sid}.png")
        if img.mode != "L":
            img = img.convert("L")
        arr = np.array(img)
        if arr.dtype != np.uint8 or arr.shape != ann.shape:
            raise ValueError(f"FATAL: visible/{v}/{sid}.png 가 (H, W) uint8 이 아니거나 라벨과 크기가 다르다")
        out.append(resize_image_uint8(arr, size))
    return out[0], out[1], resize_label(ann.astype(np.int8), size)


class GTPool:
    """생성 모델이 봐도 되는 사람 GT 레이어.

    Attributes:
        i0, i1   (N, S, S) uint8   visible/0 · visible/1
        ann      (N, S, S) int8    클래스 0–9, 미레이블 -1
        keys     list[str]         `<process_code>/<layer:07d>`
        sources  list[str]         process_code
        objects  (M, 9) int32      결함 객체 표 — 열: 표본, 클래스, y0, y1, x0, x1, 넓이, 객체 안의 한 화소 (sy, sx)
    """

    def __init__(self, holdout: str = cfg.DEFAULT_HOLDOUT, root: Path | None = None,
                 img_size: int = cfg.IMG_SIZE, log=None, limit: int = 0):
        root = Path(root) if root is not None else cfg.GT_ROOT
        if not (root / "index.csv").is_file():
            raise FileNotFoundError(f"FATAL: {root} 없음 — Peregrine 주석을 pbf-2026.11 로 변환한 데이터셋")
        check_schema(root)
        with (root / "index.csv").open(encoding="utf-8-sig", newline="") as fh:
            rows = sorted(csv.DictReader(fh), key=lambda r: r["id"])
        key = lambda r: f"{r['process_code']}/{int(r['layer']):07d}"
        all_keys = [key(r) for r in rows]
        held = set(load_holdout(holdout))
        missing = held - set(all_keys)
        if missing:
            raise RuntimeError(f"FATAL: holdout 목록의 레이어가 데이터셋에 없다: {sorted(missing)[:3]} …")
        keep = [r for r in rows if key(r) not in held]
        if len(keep) != cfg.HOLDOUT_EXPECTED_LAYERS[holdout]:
            raise RuntimeError(f"FATAL: holdout={holdout} 을 뺀 레이어가 {len(keep)} 개다 (기대값 {cfg.HOLDOUT_EXPECTED_LAYERS[holdout]})")
        for r in keep:
            if (r.get("recoat_axis") or "").strip() != "horizontal" or (r.get("swap_sd") or "0").strip() not in ("", "0"):
                raise RuntimeError(f"FATAL: 표본 {r['id']} 의 방향이 리코팅 축 = 가로가 아니거나 두 사진이 바뀌어 있다")
        if limit > 0:                                              # 스모크용 — 소스가 고르게 남도록 건너뛰며 고른다
            keep = keep[:: max(1, len(keep) // limit)][:limit]

        self.holdout = holdout
        self.img_size = img_size
        self.keys = [key(r) for r in keep]
        self.sources = [r["process_code"] for r in keep]
        self.sample_ids = [r["id"] for r in keep]
        n = len(keep)
        self.i0 = np.zeros((n, img_size, img_size), dtype=np.uint8)
        self.i1 = np.zeros((n, img_size, img_size), dtype=np.uint8)
        self.ann = np.zeros((n, img_size, img_size), dtype=np.int8)
        with ThreadPoolExecutor(max_workers=8) as ex:
            for j, (a0, a1, lab) in enumerate(ex.map(_load_one, [(root, r["id"], img_size) for r in keep])):
                self.i0[j], self.i1[j], self.ann[j] = a0, a1, lab
        assert not (set(self.keys) & held)
        self.objects = self._find_objects()
        if log is not None:
            src = {}
            for s in self.sources:
                src[s] = src.get(s, 0) + 1
            log.info(f"사람 GT: {n} 레이어 (holdout={holdout}, 뺀 레이어 {len(held)}) — " + ", ".join(f"{k} {v}" for k, v in sorted(src.items())))
            log.info("결함 객체 (30 px 이상): " + ", ".join(
                f"{cfg.CLASS_ABBRS[c]} {int((self.objects[:, O_CLS] == c).sum())}" for c in range(2, cfg.N_CLASSES)))

    def __len__(self) -> int:
        return len(self.keys)

    def _find_objects(self) -> np.ndarray:
        rows = []
        for j in range(len(self.keys)):
            ann = self.ann[j]
            for c in range(2, cfg.N_CLASSES):
                m = ann == c
                if not m.any():
                    continue
                lab, k = ndimage.label(m, structure=_CC8)
                areas = ndimage.sum(m, lab, index=np.arange(1, k + 1)).astype(np.int64)
                for i, sl in enumerate(ndimage.find_objects(lab)):
                    if areas[i] < cfg.MIN_OBJECT_PX:
                        continue
                    ys, xs = np.nonzero(lab[sl] == i + 1)
                    rows.append((j, c, sl[0].start, sl[0].stop, sl[1].start, sl[1].stop, int(areas[i]),
                                 sl[0].start + int(ys[0]), sl[1].start + int(xs[0])))
        return np.array(rows, dtype=np.int32).reshape(-1, 9)

    def objects_fitting(self, cls: int, max_side: int) -> np.ndarray:
        """클래스 cls 의 객체 가운데 bbox 의 긴 변이 max_side 이하인 것 (객체 표의 행 번호)."""
        o = self.objects
        side = np.maximum(o[:, O_Y1] - o[:, O_Y0], o[:, O_X1] - o[:, O_X0])
        return np.nonzero((o[:, O_CLS] == cls) & (side <= max_side))[0]

    def stats(self) -> dict:
        o = self.objects
        return {
            "layers": len(self),
            "holdout": self.holdout,
            "objects_ge30": {cfg.CLASS_ABBRS[c]: int((o[:, O_CLS] == c).sum()) for c in range(2, cfg.N_CLASSES)},
            "objects_fit_window": {cfg.CLASS_ABBRS[c]: int(len(self.objects_fitting(c, cfg.WIN))) for c in range(2, cfg.N_CLASSES)},
            "class_px": [int((self.ann == c).sum()) for c in range(cfg.N_CLASSES)],
            "unlabeled_px": int((self.ann == cfg.IGNORE_INDEX).sum()),
        }
