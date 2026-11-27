"""Peregrine 주석 (`ORNL_Data/DSCNN_Dataset`) → `pbf-2026.11` 레이블 데이터셋 (PLAN_v6 §2.3, §3.1).

산출물은 DIVA 의 내보내기 zip 을 푼 것과 같은 구성이다 (docs/DEFSEGAM_CLASS_UPDATE.md §6).
v6 의 로더가 DIVA 내보내기와 Peregrine 주석을 같은 방법으로 읽게 하려는 것이다.

    visible/0/{id}.png      용융 (Binder Jet 은 binder 도포) 직후 — 원본 tif 와 화소 값이 같은 무손실 PNG
    visible/1/{id}.png      분말 도포 직후
    annotations/{id}.npy    int16 (H, W) — 클래스 id 0–9, 미레이블 -1
    index.csv               {id} → 소스 · 원본 파일 · 크기 · 미레이블 화소 수 (DIVA 의 13 열 + 추가 열)
    classes.json            클래스 표 (DIVA 가 내보내는 것과 같은 내용)
    README.txt              사람이 읽는 설명
    conversion.json         변환 옵션 · 매핑 표 · 레이어별 화소 수 (재현 · 대조용)
    layer_stats.csv         레이어별 클래스 화소 수 (표로 보기 좋게)

DIVA 내보내기와 다른 점은 README.txt 에 적는다. 핵심은 둘이다 —
이미지 방향이 리코팅 축 = 가로이고 (`recoat_axis` 열), 미레이블 화소 (-1) 가 섞여 있다.

실행 (CWD = 3DP_VPPM/):
    python -m DefSeg_AM.v6.data.convert_legacy                        # 기본값으로 변환. v6/legacy_review.csv 가 있으면 반영한다 (D4)
    python -m DefSeg_AM.v6.data.convert_legacy --swelling ignore      # D3 대안: Swelling 을 미레이블로
    python -m DefSeg_AM.v6.data.convert_legacy --is-review <csv>      # D4: 다른 판정표로
    python -m DefSeg_AM.v6.data.convert_legacy --no-is-review         # D4: 판정표 없이 — 기존 IS 를 전부 미레이블로
    python -m DefSeg_AM.v6.data.convert_legacy --out <dir> --limit 1  # 스모크 (소스마다 1 레이어)
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import re
import shutil
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
from PIL import Image

from ...common.utils.log import setup_logger
from .. import config_v6 as cfg
from .legacy_map import count_classes, remap_legacy, target_of

FORMAT_TAG = "defseg-legacy-conversion/1"     # conversion.json 의 표지 — 덮어쓸 때 우리 산출물인지 확인하는 데 쓴다
MARKER = ".defseg_legacy_conversion"          # 만드는 도중의 폴더에 두는 표지

# DIVA 내보내기의 index.csv 열 (같은 이름 · 같은 순서) + 이 데이터셋에만 있는 열
DIVA_COLS = ["id", "agency", "mc_id", "pp_id", "process_code", "layer", "status", "labeler", "updated_at",
             "width", "height", "unlabeled_px", "swap_sd"]
EXTRA_COLS = ["recoat_axis", "machine", "material", "process_type", "orig_ann"]


@dataclass(frozen=True)
class LegacySample:
    source: dict
    stem: str
    img0: Path
    img1: Path
    ann: Path
    part: Path | None

    @property
    def key(self) -> str:
        """`<소스>/<원본 파일 번호>` — v5 의 표본 이름과 같다 (fold 목록을 그대로 쓸 수 있다)."""
        return f"{self.source['name']}/{self.stem}"


def enumerate_legacy_samples(limit_per_source: int | None = None) -> tuple[list[LegacySample], list[str]]:
    """8 소스의 (visible/0, visible/1, 주석) 을 나열한다. 순서는 v5 와 같다 — 소스는 config 순, 소스 안은 파일 이름순.

    Returns: (표본 목록, 건너뛴 것의 설명)
    """
    samples: list[LegacySample] = []
    skipped: list[str] = []
    for src in cfg.LEGACY_SOURCES:
        root = src["root"]
        ann_dir = root / "annotations"
        if not ann_dir.is_dir():
            raise FileNotFoundError(f"FATAL: {src['name']} 의 주석 폴더가 없다: {ann_dir}")
        n = 0
        for ann in sorted(ann_dir.glob("*.npy")):
            v0 = root / "data" / "visible" / "0" / f"{ann.stem}.tif"
            v1 = root / "data" / "visible" / "1" / f"{ann.stem}.tif"
            if not (v0.is_file() and v1.is_file()):
                skipped.append(f"{src['name']}/{ann.stem}: visible/0 또는 visible/1 이 없다")
                continue
            part = root / "parts" / f"{ann.stem}.png"
            samples.append(LegacySample(src, ann.stem, v0, v1, ann, part if part.is_file() else None))
            n += 1
            if limit_per_source is not None and n >= limit_per_source:
                break
    return samples, skipped


def parse_info_txt(path: Path) -> dict:
    """v2022 판의 info.txt → {"classes": {id: 이름}, "meta": {항목: 값}}."""
    classes: dict[int, str] = {}
    meta: dict[str, str] = {}
    for line in path.read_text(errors="replace").splitlines():
        m = re.match(r"^\s*(-?\d+)\s*=\s*(.+?)\s*$", line)
        if m:
            classes[int(m.group(1))] = m.group(2)
            continue
        if ":" in line:
            k, v = line.split(":", 1)
            if v.strip():
                meta[k.strip()] = v.strip()
    return {"classes": classes, "meta": meta}


def check_native_tables(log) -> dict[str, dict]:
    """config 의 native 이름 표를 v2022 의 info.txt 와 대조한다. 다르면 FATAL. 소스별 info.txt 의 메타를 돌려준다."""
    out: dict[str, dict] = {}
    for src in cfg.LEGACY_SOURCES:
        info = src["root"].parent / "info.txt"
        if not info.is_file():
            out[src["name"]] = {}              # v2021 판 — 표의 출처는 readme.pdf
            continue
        parsed = parse_info_txt(info)
        table = {k: v for k, v in parsed["classes"].items() if k != cfg.NATIVE_UNLABELED}
        if table != cfg.LEGACY_NATIVE_NAMES[src["name"]]:
            raise RuntimeError(
                f"FATAL: {src['name']} 의 native 이름 표가 {info} 와 다르다\n"
                f"  info.txt : {table}\n  config_v6: {cfg.LEGACY_NATIVE_NAMES[src['name']]}")
        if parsed["classes"].get(cfg.NATIVE_UNLABELED, "unlabeled") != "unlabeled":
            raise RuntimeError(f"FATAL: {info} 의 -1 이 unlabeled 가 아니다")
        out[src["name"]] = parsed["meta"]
    log.info(f"native 이름 표: v2022 소스 {sum(bool(v) for v in out.values())}개를 info.txt 와 대조했다 (일치)")
    return out


def load_is_review(path: Path, samples: list[LegacySample], strict: bool = True) -> tuple[dict[str, str], set[str]]:
    """D4 판정표 → ({표본 key: "is" | "ps" | "ignore"}, 영역을 채울 표본 key 의 집합).

    CSV 열: source, layer, decision, fill (+ 그 밖의 열은 무시). source 는 소스 이름 (v2021_LPBF) 또는 약어 (L21),
    layer 는 원본 파일 번호 (4 또는 0000004), decision 은 IS / PS / ignore (2 / 3 / -1 도 받는다).
    fill 이 yes 인 레이어는 원본이 경계선만 칠한 것이다 — 선이 가두는 쪽의 Powder 를 decision 의 클래스로 채운다.
    표에 없는 레이어의 기존 IS 는 미레이블로 남는다.
    """
    by_name = {s["name"]: s["name"] for s in cfg.LEGACY_SOURCES} | {s["short"]: s["name"] for s in cfg.LEGACY_SOURCES}
    keys = {s.key for s in samples}
    alias = {"is": "is", "2": "is", "ps": "ps", "3": "ps", "ignore": "ignore", "-1": "ignore", "": "ignore"}
    out: dict[str, str] = {}
    fills: set[str] = set()
    with open(path, encoding="utf-8-sig", newline="") as fp:
        for ln, row in enumerate(csv.DictReader(fp), start=2):
            src = by_name.get((row.get("source") or "").strip())
            dec = alias.get((row.get("decision") or "").strip().lower())
            try:
                layer = int((row.get("layer") or "").strip())
            except ValueError:
                layer = None
            if src is None or dec is None or layer is None:
                raise ValueError(f"FATAL: {path} {ln}행을 읽을 수 없다: {row}")
            key = f"{src}/{layer:07d}"
            if key not in keys:
                if strict:
                    raise ValueError(f"FATAL: {path} {ln}행의 레이어가 데이터셋에 없다: {key}")
                continue                   # --limit 스모크: 변환하지 않는 레이어의 행은 건너뛴다
            if key in out and out[key] != dec:
                raise ValueError(f"FATAL: {path} 에 {key} 가 서로 다른 결정으로 두 번 나온다")
            out[key] = dec
            if (row.get("fill") or "").strip().lower() in ("yes", "y", "1", "true"):
                if dec == "ignore":
                    raise ValueError(f"FATAL: {path} {ln}행 — fill 은 IS / PS 판정과 함께만 쓸 수 있다: {key}")
                fills.add(key)
    return out, fills


def _rel(path: Path) -> str:
    """PROJECT_ROOT (3DP_VPPM/) 기준 상대 경로. 밖에 있으면 그대로."""
    try:
        return Path(path).resolve().relative_to(cfg.PROJECT_ROOT).as_posix()
    except ValueError:
        return str(path)


def _review_flag(is_review: Path | None) -> str:
    """같은 결과를 다시 만드는 데 필요한 판정표 옵션 (README 에 적는다)."""
    default = cfg.LEGACY_REVIEW_CSV
    if is_review is None:
        return " --no-is-review" if default.is_file() else ""
    if Path(is_review).resolve() == default.resolve():
        return ""
    return f" --is-review {_rel(is_review)}"


def _git_describe() -> str:
    try:
        repo = Path(__file__).resolve().parents[2]
        r = subprocess.run(["git", "-C", str(repo), "describe", "--always", "--dirty"],
                           capture_output=True, text=True, timeout=10)
        return r.stdout.strip() or "unknown"
    except Exception:
        return "unknown"


def _save_png_checked(arr: np.ndarray, dst: Path) -> None:
    """(H, W) uint8 → 무손실 PNG. 다시 읽어 화소 값이 같은지 확인한다."""
    Image.fromarray(arr).save(dst, format="PNG", compress_level=6)
    back = np.array(Image.open(dst))
    if back.shape != arr.shape or back.dtype != arr.dtype or not np.array_equal(back, arr):
        raise RuntimeError(f"FATAL: PNG 로 저장한 것이 원본과 다르다: {dst}")


def _load_tif(path: Path) -> np.ndarray:
    arr = np.array(Image.open(path))
    if arr.ndim != 2 or arr.dtype != np.uint8:
        raise ValueError(f"FATAL: (H, W) uint8 이미지가 아니다 (shape={arr.shape}, dtype={arr.dtype}): {path}")
    return arr


def _classes_json() -> dict:
    """DIVA 가 내보내기에 넣는 classes.json 과 같은 구성."""
    keys = ("id", "key", "name", "abbr", "group", "source", "color", "criteria_en", "criteria_ko")
    return {
        "schema": cfg.SCHEMA_VERSION,
        "unlabeled": cfg.SCHEMA["unlabeled"]["export_value"],
        "classes": [{k: c.get(k) for k in keys} for c in cfg.SCHEMA["classes"]],
        "image_types": cfg.SCHEMA["image_types"],
    }


def _mapping_table(swelling: str) -> dict[str, dict[str, list]]:
    """소스별 {native id: [이름, 새 id]} — conversion.json 에 남긴다 (기존 IS 는 검토 전 기본값으로 적는다)."""
    return {
        src: {str(nid): [name, int(target_of(name, swelling, "ignore"))] for nid, name in sorted(names.items())}
        for src, names in cfg.LEGACY_NATIVE_NAMES.items()
    }


def _readme(meta: dict) -> str:
    ab = cfg.CLASS_ABBRS_V6
    tot = meta["totals"]
    rv = {k: sum(1 for r in meta["layers"] if r["is_decision"] == k) for k in ("is", "ps", "ignore")}
    lines = [
        "Peregrine 주석 (ORNL DSCNN_Dataset) 을 DIVA 레이블 체계로 옮긴 레이블 데이터셋",
        f"스키마: {cfg.SCHEMA_VERSION} (classes.json) · 만든 시각: {meta['created_at']} · 코드: DefSeg_AM {meta['code']}",
        f"층 수: {tot['layers']} (소스 {len(meta['per_source'])}개) · 상태: {cfg.LEGACY_STATUS}",
        "",
        "구성은 DIVA 의 내보내기 zip 과 같다.",
        "visible/0/{id}.png   Scanning (레이저 용융 직후. Binder Jet 은 binder 도포 직후) — 원본 tif 와 화소 값이 같은 무손실 PNG",
        "visible/1/{id}.png   Deposition (분말 도포 직후)",
        "annotations/{id}.npy   int16 (H, W) — 클래스 id 0~9, 미레이블 -1",
        "index.csv          id → 소스 · 원본 파일 · 크기 · 미레이블 수. 앞의 13 열은 DIVA 내보내기와 같은 이름 · 같은 순서다",
        "classes.json       클래스 표 (DIVA 가 내보내는 것과 같은 내용)",
        "conversion.json    변환 옵션 · 소스별 매핑 표 · 레이어별 화소 수",
        "layer_stats.csv    레이어별 클래스 화소 수",
        "",
        "DIVA 내보내기와 다른 점",
        "- 레이블은 DIVA 에서 사람이 새 기준으로 칠한 것이 아니다. ORNL 의 전문가 주석 (Peregrine v2021-03 · v2022-10.1) 을",
        "  이름 기준 규칙으로 옮긴 것이다. 그래서 status 가 labeled / reviewed 가 아니라 converted 다.",
        "- 이미지 방향: 리코팅 축이 가로다 (index.csv 의 recoat_axis = horizontal). DIVA 내보내기는 리코팅 축이 세로인",
        "  DIVA 화면 방향이다. 원본 주석의 방향 그대로이고 (DefSegAM v1–v5 가 학습한 방향), 이 변환에서 돌리지 않았다.",
        "  DIVA 내보내기와 함께 쓸 때는 recoat_axis 로 방향을 맞춘다.",
        "- 미레이블 (-1) 화소가 있다. 원본 주석의 미레이블, 새 체계에 대응이 없는 클래스, "
        + ("판정표에서 미레이블로 둔 기존 Incomplete Spreading 이다." if meta["options"]["is_review"] else "기존 Incomplete Spreading 전부다."),
        "  손실에서 -1 을 빼야 한다. 레이어별 수는 index.csv 의 unlabeled_px.",
        "- agency · mc_id · pp_id 는 DIVA 의 번호가 아니라 이름이다 (agency = ORNL, mc_id = 장비, pp_id = process_code = 소스).",
        "  한 소스는 빌드 하나가 아니라 같은 장비 · 재료의 여러 빌드에서 고른 층들이다.",
        "- layer 는 원본 파일 번호다 (실제 층 번호가 아니다). `{process_code}/{layer:07d}` 가 DefSeg_AM v5 의 표본 이름과 같다.",
        "",
        "옮긴 규칙 (DefSeg_AM/v6/PLAN_v6.md §2.3 — 번호가 아니라 원본 클래스 이름으로 옮긴다)",
        "- Powder / Un-Fused Powder → 0 Powder,  Printed / Printed Material → 1 Printed",
        "- Recoater Streaking / Roller Streaking → 4 RS,  Recoater Hopping → 5 RH",
        "- Super-Elevation → 6 SE,  Spatter on Powder → 7 SP,  Excessive Melting → 8 OM,  Debris → 9 DB",
        f"- Swelling / Edge Swelling → {'6 SE' if meta['options']['swelling'] == 'se' else '미레이블 (-1)'}   [D3 — 옵션 --swelling {meta['options']['swelling']}]",
        "- Soot → 7 SP. CAD 파트 마스크가 있는 레이어에서 마스크 안쪽의 Soot 만 8 OM   [D10]",
        "- Incomplete Spreading / Powder Short Feed / Short Feed → "
        + ("미레이블 (-1)   [D4 — 판정표 없이 변환했다. 새 기준의 IS 와 PS 가 섞여 있어 번호만 옮기지 않는다]" if not meta["options"]["is_review"]
           else f"판정표에 따라 레이어 단위로 2 IS ({rv['is']} 층) / 3 PS ({rv['ps']} 층) / 미레이블 ({rv['ignore']} 층)   [D4 — {meta['options']['is_review']}]"),
    ] + ([
        f"  그 가운데 {tot['is_fill_layers']} 층은 원본이 덮이지 않은 영역 대신 경계선만 칠한 것이다. 선이 이미지 가장자리와 함께 가두는 쪽의 Powder 화소",
        f"  {tot['is_fill_px']:,d} 개를 그 층의 클래스 (IS 또는 PS) 로 채웠다. 그 안의 Printed · SE · SP 와 미레이블 화소는 그대로다 (layer_stats.csv 의 is_fill_px).",
    ] if tot.get("is_fill_px") else []) + [
        "- Exposed Part, Stripe Boundary, Localized Bright Spot, Localized Dark Regions, Cornrows, Misprint, Crashing, Mounding Powder → 미레이블 (-1)",
        "",
        "클래스별 화소 수 (원본 해상도, 전체)",
    ]
    for i in range(cfg.N_CLASSES_V6):
        lines.append(f"  {i} {ab[i]:<3s} {cfg.CLASS_NAMES_V6[i]:<21s} {tot['class_px'][i]:>13,d} px   {tot['class_layers'][i]:>3d} 층")
    lines += [
        f"  미레이블 (-1)               {tot['unlabeled_px']:>13,d} px   (전체의 {tot['unlabeled_px'] / tot['total_px'] * 100:.1f}%)",
        "",
        "다시 만들기 (CWD = 3DP_VPPM/):  python -m DefSeg_AM.v6.data.convert_legacy --overwrite"
        + ("" if meta["options"]["swelling"] == cfg.LEGACY_SWELLING_DEFAULT else f" --swelling {meta['options']['swelling']}")
        + meta["options"]["is_review_flag"],
        "원본: ORNL Peregrine Dataset v2021-03 · v2022-10.1 (각 판의 readme.pdf 참조)",
        "",
    ]
    return "\n".join(lines)


def _prepare_dirs(out_dir: Path, overwrite: bool) -> Path:
    """만드는 동안 쓸 폴더 (<out>.partial) 를 준비한다. 기존 산출물은 다 만든 뒤에 바꿔 끼운다."""
    if out_dir.exists():
        if not overwrite:
            raise FileExistsError(f"FATAL: 이미 있다: {out_dir} — 다시 만들려면 --overwrite")
        conv = out_dir / "conversion.json"
        ok = False
        if conv.is_file():
            try:
                ok = json.loads(conv.read_text(encoding="utf-8")).get("format") == FORMAT_TAG
            except Exception:
                ok = False
        if not ok:
            raise RuntimeError(f"FATAL: {out_dir} 는 이 스크립트가 만든 폴더가 아니다 (conversion.json 없음) — 지우지 않는다")
    tmp = out_dir.with_name(out_dir.name + ".partial")
    if tmp.exists():
        if not (tmp / MARKER).is_file():
            raise RuntimeError(f"FATAL: {tmp} 가 있는데 이 스크립트의 표지가 없다 — 직접 확인하고 지운 뒤 다시 실행한다")
        shutil.rmtree(tmp)
    for sub in ("visible/0", "visible/1", "annotations"):
        (tmp / sub).mkdir(parents=True)
    (tmp / MARKER).write_text("convert_legacy 가 만드는 중인 폴더\n", encoding="utf-8")
    return tmp


def convert(out_dir: Path, swelling: str, is_review: Path | None, overwrite: bool,
            limit_per_source: int | None, log) -> dict:
    t0 = time.time()
    if not cfg.DSCNN_ROOT.is_dir():
        raise FileNotFoundError(f"FATAL: 원본이 없다: {cfg.DSCNN_ROOT}")
    info_meta = check_native_tables(log)
    samples, skipped = enumerate_legacy_samples(limit_per_source)
    if not samples:
        raise RuntimeError("FATAL: 변환할 레이어가 없다")
    decisions, fills = load_is_review(is_review, samples, strict=limit_per_source is None) if is_review else ({}, set())
    log.info(f"원본: {cfg.DSCNN_ROOT}")
    log.info(f"레이어 {len(samples)}개 (소스 {len(cfg.LEGACY_SOURCES)}개), 건너뜀 {len(skipped)}개 · "
             f"Swelling → {swelling} (D3) · 기존 IS → " + (f"판정표 {_rel(is_review)} ({len(decisions)} 레이어, 그 가운데 영역 채움 {len(fills)})" if is_review else "전부 미레이블 (판정표 없음)"))

    tmp = _prepare_dirs(out_dir, overwrite)
    created_at = datetime.now(timezone.utc).astimezone().isoformat(timespec="seconds")
    rel_root = cfg.DSCNN_ROOT.parent                      # index.csv 의 orig_ann 은 ORNL_Data/ 기준 상대 경로
    n_cls = cfg.N_CLASSES_V6
    index_rows: list[list] = []
    layers: list[dict] = []

    for i, s in enumerate(samples):
        src = s.source
        sid = f"{i:07d}"
        ann = np.load(s.ann)
        img0 = _load_tif(s.img0)
        img1 = _load_tif(s.img1)
        if not (ann.shape == img0.shape == img1.shape):
            raise RuntimeError(f"FATAL: {s.key} 의 크기가 서로 다르다 — 주석 {ann.shape}, visible/0 {img0.shape}, visible/1 {img1.shape}")
        part = np.array(Image.open(s.part)) if s.part is not None else None
        dec = decisions.get(s.key, "ignore")
        label, st = remap_legacy(ann, src["name"], part, swelling=swelling, is_decision=dec, is_fill=s.key in fills)
        class_px, unlabeled = count_classes(label)
        h, w = label.shape
        if int(class_px.sum()) + unlabeled != h * w:
            raise RuntimeError(f"FATAL: {s.key} 의 화소 수가 맞지 않는다")

        _save_png_checked(img0, tmp / "visible" / "0" / f"{sid}.png")
        _save_png_checked(img1, tmp / "visible" / "1" / f"{sid}.png")
        np.save(tmp / "annotations" / f"{sid}.npy", label.astype(np.int16))

        native = st["native_px"]
        old_is = sum(native.get(n, 0) for n in cfg.LEGACY_IS_NAMES)
        swell = sum(native.get(n, 0) for n in cfg.LEGACY_SWELLING_NAMES)
        dropped = sum(native.get(n, 0) for n in cfg.LEGACY_NAME_IGNORED)
        index_rows.append([
            sid, cfg.LEGACY_AGENCY, src["mc_id"], src["name"], src["name"], int(s.stem), cfg.LEGACY_STATUS,
            f"ORNL ({src['release']})", created_at, w, h, unlabeled, 0,
            cfg.LEGACY_RECOAT_AXIS, src["machine"], src["material"], src["process"],
            s.ann.relative_to(rel_root).as_posix(),
        ])
        layers.append({
            "id": sid, "key": s.key, "source": src["name"], "layer": int(s.stem), "height": h, "width": w,
            "class_px": [int(v) for v in class_px], "unlabeled_px": unlabeled,
            "native_px": native, "native_unlabeled_px": st["native_unlabeled_px"],
            "old_is_px": int(old_is), "is_decision": dec if old_is else "",
            "old_is_unlabeled_px": int(old_is) if dec == "ignore" else 0,
            "is_fill_px": st["is_fill_px"], "is_fill_region_px": st["is_fill_region_px"],
            "swelling_px": int(swell), "soot_px": int(native.get(cfg.LEGACY_SOOT_NAME, 0)),
            "soot_to_om_px": st["soot_to_om_px"], "dropped_other_px": int(dropped), "part_mask": st["part_mask"],
        })
        if (i + 1) % 25 == 0 or i + 1 == len(samples):
            log.info(f"  {i + 1}/{len(samples)} 레이어 ({time.time() - t0:.0f}s)")

    # ---- 요약 ----
    def summarize(rows: list[dict]) -> dict:
        px = np.sum([r["class_px"] for r in rows], axis=0)
        return {
            "layers": len(rows),
            "class_px": [int(v) for v in px],
            "class_layers": [int(sum(1 for r in rows if r["class_px"][c] > 0)) for c in range(n_cls)],
            "unlabeled_px": int(sum(r["unlabeled_px"] for r in rows)),
            "total_px": int(sum(r["height"] * r["width"] for r in rows)),
            "old_is_px": int(sum(r["old_is_px"] for r in rows)),
            "old_is_layers": int(sum(1 for r in rows if r["old_is_px"] > 0)),
            "old_is_unlabeled_px": int(sum(r["old_is_unlabeled_px"] for r in rows)),
            "is_fill_px": int(sum(r["is_fill_px"] for r in rows)),
            "is_fill_layers": int(sum(1 for r in rows if r["is_fill_px"] > 0)),
            "swelling_px": int(sum(r["swelling_px"] for r in rows)),
            "soot_px": int(sum(r["soot_px"] for r in rows)),
            "soot_to_om_px": int(sum(r["soot_to_om_px"] for r in rows)),
            "dropped_other_px": int(sum(r["dropped_other_px"] for r in rows)),
            "native_unlabeled_px": int(sum(r["native_unlabeled_px"] for r in rows)),
        }

    per_source = {}
    for src in cfg.LEGACY_SOURCES:
        rows = [r for r in layers if r["source"] == src["name"]]
        if rows:
            per_source[src["name"]] = summarize(rows) | {
                "short": src["short"], "machine": src["machine"], "material": src["material"],
                "process": src["process"], "release": src["release"], "info_txt": info_meta.get(src["name"], {}),
            }
    meta = {
        "format": FORMAT_TAG,
        "schema": cfg.SCHEMA_VERSION,
        "created_at": created_at,
        "code": _git_describe(),
        "source_root": cfg.DSCNN_ROOT.relative_to(cfg.PROJECT_ROOT).as_posix(),
        "options": {"swelling": swelling, "is_review": _rel(is_review) if is_review else None,
                    "is_review_flag": _review_flag(is_review), "limit_per_source": limit_per_source},
        "decisions": {
            "D3_swelling": swelling,
            "D4_old_is": "review table" if is_review else "ignore (-1)",
            "D4_review_rows": decisions,
            "D4_fill_layers": sorted(fills),
            "D4_fill_rule": "boundary-line layers: Powder pixels on the side the line encloses with the image border get the layer's class",
            "D10_soot": "Spatter; Over Melting inside a non-empty CAD part mask",
            "D11_exposed_part": "ignore (-1)",
        },
        "recoat_axis": cfg.LEGACY_RECOAT_AXIS,
        "image_format": "PNG (lossless, verified equal to the source tif)",
        "mapping": _mapping_table(swelling),
        "totals": summarize(layers),
        "per_source": per_source,
        "skipped": skipped,
        "layers": layers,
    }

    # ---- 파일 쓰기 ----
    buf = io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(DIVA_COLS + EXTRA_COLS)
    wr.writerows(index_rows)
    (tmp / "index.csv").write_bytes(buf.getvalue().encode("utf-8-sig"))
    (tmp / "classes.json").write_text(json.dumps(_classes_json(), ensure_ascii=False, indent=2), encoding="utf-8")
    (tmp / "conversion.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1), encoding="utf-8")
    (tmp / "README.txt").write_text(_readme(meta), encoding="utf-8")

    buf = io.StringIO()
    wr = csv.writer(buf)
    wr.writerow(["id", "process_code", "layer", "width", "height"] + [f"px_{a}" for a in cfg.CLASS_ABBRS_V6]
                + ["unlabeled_px", "native_unlabeled_px", "old_is_px", "is_decision", "is_fill_px", "swelling_px", "soot_px",
                   "soot_to_om_px", "dropped_other_px", "part_mask"])
    for r in layers:
        wr.writerow([r["id"], r["source"], r["layer"], r["width"], r["height"]] + r["class_px"]
                    + [r["unlabeled_px"], r["native_unlabeled_px"], r["old_is_px"], r["is_decision"], r["is_fill_px"], r["swelling_px"],
                       r["soot_px"], r["soot_to_om_px"], r["dropped_other_px"], r["part_mask"]])
    (tmp / "layer_stats.csv").write_bytes(buf.getvalue().encode("utf-8-sig"))

    # ---- 바꿔 끼우기 ----
    (tmp / MARKER).unlink()
    old = None
    if out_dir.exists():
        old = out_dir.with_name(out_dir.name + f".old-{int(time.time())}")
        out_dir.rename(old)
    tmp.rename(out_dir)
    if old is not None:
        shutil.rmtree(old)

    # ---- 로그 ----
    ab = cfg.CLASS_ABBRS_V6
    log.info("소스별 화소 수 (원본 해상도)")
    log.info("  " + f"{'소스':<6s} {'층':>3s} " + " ".join(f"{a:>11s}" for a in ab) + f" {'미레이블':>11s}")
    for name, ps in per_source.items():
        log.info("  " + f"{ps['short']:<6s} {ps['layers']:>3d} " + " ".join(f"{v:>11,d}" for v in ps["class_px"]) + f" {ps['unlabeled_px']:>11,d}")
    tot = meta["totals"]
    log.info("  " + f"{'합계':<6s} {tot['layers']:>3d} " + " ".join(f"{v:>11,d}" for v in tot["class_px"]) + f" {tot['unlabeled_px']:>11,d}")
    log.info("  " + f"{'층 수':<6s} {'':>3s} " + " ".join(f"{v:>11d}" for v in tot["class_layers"]))
    log.info(f"미레이블 {tot['unlabeled_px'] / tot['total_px'] * 100:.1f}% = 원본 미레이블 {tot['native_unlabeled_px']:,d} + 기존 IS "
             f"{tot['old_is_unlabeled_px']:,d} (전체 {tot['old_is_px']:,d} / {tot['old_is_layers']} 층) + 그 밖의 클래스 {tot['dropped_other_px']:,d}"
             + (f" + Swelling {tot['swelling_px']:,d}" if swelling == "ignore" else ""))
    if tot["is_fill_px"]:
        log.info(f"영역 채움: {tot['is_fill_layers']} 층에서 경계선이 가두는 쪽의 Powder {tot['is_fill_px']:,d} px → 그 층의 IS / PS")
    log.info(f"Swelling {tot['swelling_px']:,d} px → {'SE' if swelling == 'se' else '미레이블'} · "
             f"Soot {tot['soot_px']:,d} px 가운데 파트 마스크 안쪽 {tot['soot_to_om_px']:,d} px → OM, 나머지 → SP")
    log.info(f"완료: {out_dir}  ({len(samples)} 레이어, {time.time() - t0:.0f}s)")
    return meta


def main() -> None:
    ap = argparse.ArgumentParser(description="Peregrine 주석 → pbf-2026.11 레이블 데이터셋 (DIVA 내보내기와 같은 구성)")
    ap.add_argument("--out", type=Path, default=cfg.LEGACY_PBF_ROOT, help=f"산출 폴더 (기본: {cfg.LEGACY_PBF_ROOT})")
    ap.add_argument("--swelling", choices=sorted(cfg.LEGACY_SWELLING_CHOICES), default=cfg.LEGACY_SWELLING_DEFAULT,
                    help="D3 — Swelling / Edge Swelling 을 Super Elevation 으로 (se) 또는 미레이블로 (ignore)")
    ap.add_argument("--is-review", type=Path, default=None,
                    help=f"D4 — 기존 IS 의 레이어 단위 판정표 (CSV: source, layer, decision = IS / PS / ignore). "
                         f"주지 않으면 {cfg.LEGACY_REVIEW_CSV.name} 가 있을 때 그것을 쓴다")
    ap.add_argument("--no-is-review", action="store_true", help="D4 — 판정표를 쓰지 않는다 (기존 IS 를 전부 미레이블로)")
    ap.add_argument("--overwrite", action="store_true", help="이미 있는 산출 폴더를 새로 만든 것으로 바꾼다")
    ap.add_argument("--limit", type=int, default=None, help="스모크용 — 소스마다 앞에서 이만큼만")
    args = ap.parse_args()
    log = setup_logger()
    if args.no_is_review and args.is_review:
        ap.error("--is-review 와 --no-is-review 를 함께 줄 수 없다")
    review = None if args.no_is_review else (args.is_review or (cfg.LEGACY_REVIEW_CSV if cfg.LEGACY_REVIEW_CSV.is_file() else None))
    if review is not None and not Path(review).is_file():
        ap.error(f"판정표가 없다: {review}")
    convert(args.out.resolve(), args.swelling, review, args.overwrite, args.limit, log)


if __name__ == "__main__":
    main()
