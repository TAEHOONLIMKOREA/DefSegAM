"""ORNL 5 빌드에서 무작위로 뽑은 레이어에 v6 모델과 옛 모델을 돌려 나란히 본다 (참고용 — PLAN_v6 §6.5).

ORNL 빌드에는 사람 GT 가 없다. 그래서 여기서 나오는 것은 정확도가 아니라 **모델끼리의 비교**다.
    열:  visible/0 | visible/1 | DSCNN 예측 (HDF5) | v4 | v5 | v6
옛 예측 (DSCNN 12-class, v4 · v5 11-class) 은 모두 새 id (pbf-2026.11) 로 옮겨 같은 색으로 그린다
(Swelling → SE, 옛 IS → IS, Misprint · Under Melting → 대응 없음). 여기서 나온 일치도 수치는 어떤 결정에도 쓰지 않는다.

입력 밝기 — 모델마다 학습 때 본 대로 넣는다 (`infer.py` 의 설명 참조):
    v6 · v5 (사람 GT 만으로 학습)    원본 그대로 (0–255 로 자르기만)
    v4 (ORNL 로 Stage 1 을 학습)      레이어마다 1–99 백분위를 0–255 로 늘린 사진 (v4 의 infer 와 같은 방식)
TTA · 후처리는 어느 모델에도 걸지 않는다 (--tta 를 주면 세 모델 모두 8-way 평균).

레이어 뽑기: 빌드마다 레이어 번호를 seed 로 섞어 앞에서부터, (a) CAD 파트 마스크 (`slices/part_ids`) 가 비어 있지 않고
(b) 두 사진이 비어 있지 않은 (가운데 가로 띠의 평균 밝기 ≥ 20 — 빌드의 첫 · 마지막 레이어는 한쪽 사진이 비어 있다) 레이어를 n 개 고른다.
뽑은 목록은 layers.json 에 남고, --layers-json 으로 같은 목록을 다시 쓸 수 있다.

CAD 파트 마스크와의 비교: 사람 GT 가 없는 빌드에서 모델 예측이 아닌 기준은 이것뿐이다. "파트 쪽 클래스" (Printed · SE · OM, 옛 모델의
대응 없음 = Under Melting · Misprint 포함) 로 낸 화소와 CAD 파트 화소가 얼마나 겹치는지를 모델마다 낸다.

출력 (figures/<run>/v6/<full | cv/<fold>>/ornl_random<N>_<which>[_tta]/):
    layers.json                     뽑은 레이어 (seed · 규칙 · 빌드별 목록)
    overview.png                    빌드마다 한 레이어 — 뽑힌 레이어 가운데 v6 의 결함 화소 비율이 중앙값인 것 (먼저 볼 한 장)
    pages/page_XX_<build>.png       비교 그림 — 뽑힌 레이어 전부, 한 장에 5 레이어 × 6 열. 사진 타일은 원본의 3×3 평균,
                                    예측 타일은 3×3 칸마다 클래스 하나 (분말이 아닌 것 가운데 가장 많은 클래스 — `tile_label`)
    summary.png                     클래스별 예측 화소 비율 · v6 와의 일치 · CAD 파트 마스크와의 겹침
    summary.md · summary.json       같은 수치의 표 + RS 선의 가로 길이, 고정 위치의 Debris (카메라 쪽의 점)
    layer_stats.csv · stats.npz     레이어별 화소 수 (모델 × 클래스), v6 대 다른 모델의 교차표, CAD 안팎의 예측, RS 행 통계, 화소별 Debris 횟수
    preds/<build>_<layer>.npz       원본 해상도 예측 (int8, 10 = 대응 없음) 과 CAD 파트 마스크 — --no-save-preds 로 끈다

사용 (CWD = 3DP_VPPM/. compose 로는 DEFSEG_V6_PHASE=ornl_compare):
    python -m DefSeg_AM.v6.inference.compare_ornl                                  # 전수 학습 모델, 빌드마다 20 레이어
    python -m DefSeg_AM.v6.inference.compare_ornl --from-preds                     # 모델을 다시 돌리지 않고 그림 · 수치만 (CPU)
    python -m DefSeg_AM.v6.inference.compare_ornl --fold inlayer --builds B1 --n-per-build 2
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from pathlib import Path

import h5py
import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image, ImageDraw, ImageFont
from torch.utils.data import DataLoader, Dataset

from ...common.data.image_utils import ornl_image_to_uint8
from ...common.models.model import round_to_patch

from .. import config_v6 as cfg
from .baselines import NONE_ID, old_to_new_lut
from .evaluate import load_model, predict_pair
from .infer import ornl_raw_uint8, ornl_segmentation_argmax_v6

N_BINS = cfg.N_CLASSES_V6 + 1                 # 새 id 10개 + 대응 없음
PART_LIKE = (1, 6, 8, NONE_ID)                # 파트 쪽 클래스: Printed · SE · OM · (옛 모델의) 대응 없음
BLANK_MEAN = 20.0                             # 사진이 비어 있다고 보는 평균 밝기
RS_ID = cfg.CLASS_ABBRS_V6.index("RS")
DB_ID = cfg.CLASS_ABBRS_V6.index("DB")
RS_ROW_MIN = 20                               # 한 행에 RS 화소가 이만큼 넘어야 "RS 가 있는 행" 으로 센다
FIXED_MIN_LAYERS = 20                         # 고정 위치의 Debris 는 레이어가 이만큼은 있어야 센다
LAYER_MIN_PX = 1000                           # "그 클래스를 낸 레이어" 로 세는 화소 수 (1842² 의 0.03%)
TILE_TARGET = 614                             # 그림 타일 한 변 (1842 를 3×3 평균으로 줄인 크기)

MODEL_INFO = {   # 이름 → (열 제목, 설명, summary 의 표시 이름)
    "dscnn": ("DSCNN", "ORNL HDF5 segmentation_results → new ids", "DSCNN (HDF5)"),
    "v4": ("DefSeg-AM v4", "serving model, DSCNN-pretrained → new ids", "v4 (served)"),
    "v5": ("DefSeg-AM v5", "human GT only, 11 classes → new ids", "v5 (GT only, 11-class)"),
    "v6": ("DefSeg-AM v6", "human GT only, pbf-2026.11", "v6 (GT only, pbf-2026.11)"),
}
# summary.png 의 계열 색 · 표식. dataviz 기본 팔레트의 앞 세 칸 (모든 쌍 검증 통과) + 기준 (DSCNN) 은 무채색
MODEL_STYLE = {
    "v6": ("#2a78d6", "o"), "v5": ("#eb6834", "s"), "v4": ("#1baf7a", "^"), "dscnn": ("#7a7974", "D"),
}
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"


# ---------------------------------------------------------------------------
# 레이어 뽑기
# ---------------------------------------------------------------------------

def _hdf5_path(build: str) -> Path:
    return cfg.ORNL_HDF5_DIR / cfg.ORNL_BUILD_FILES[build]


def _photo_blank(dset: h5py.Dataset, li: int) -> bool:
    """가운데 가로 띠 (chunk 한 줄) 만 읽어 사진이 비어 있는지 본다 — 한 레이어를 다 읽으면 2초 가까이 걸린다."""
    h = dset.shape[1]
    ch = dset.chunks[1] if dset.chunks else 32
    r0 = (h // 2 // ch) * ch
    return float(np.asarray(dset[li, r0:r0 + ch, :]).mean()) < BLANK_MEAN


def pick_layers(build: str, n: int, seed: int) -> dict:
    """빌드 하나에서 n 레이어. 섞은 순서로 훑어 조건에 맞는 것을 앞에서부터 고른다 (조건에 맞는 레이어 가운데 고르게 뽑는 것과 같다)."""
    path = _hdf5_path(build)
    with h5py.File(path, "r") as f:
        v0 = f["slices/camera_data/visible/0"]
        v1 = f["slices/camera_data/visible/1"]
        pid = f["slices/part_ids"]
        n_total = v0.shape[0]
        order = np.random.default_rng([seed, int(build[1:])]).permutation(n_total)
        picked: list[int] = []
        skipped: list[dict] = []
        for li in order:
            li = int(li)
            if _photo_blank(v0, li) or _photo_blank(v1, li):
                skipped.append({"layer": li, "why": "blank photo"})
                continue
            if not (pid[li] > 0).any():
                skipped.append({"layer": li, "why": "empty part mask"})
                continue
            picked.append(li)
            if len(picked) == n:
                break
    return {"file": path.name, "n_layers": int(n_total), "layers": sorted(picked), "skipped": skipped}


# ---------------------------------------------------------------------------
# 읽기 — worker 프로세스가 HDF5 를 따로 연다 (chunk 가 레이어 56개씩 묶여 있어 한 레이어 읽기가 느리다)
# ---------------------------------------------------------------------------

class OrnlLayers(Dataset):
    def __init__(self, items: list[tuple[str, int]], photos_only: bool = False):
        self.items = items
        self.photos_only = photos_only        # --from-preds: 그림에 넣을 사진만 읽는다 (DSCNN 마스크 12장을 읽지 않아 빠르다)
        self._files: dict[str, h5py.File] = {}

    def __len__(self) -> int:
        return len(self.items)

    def __getitem__(self, i: int) -> dict:
        build, li = self.items[i]
        if build not in self._files:
            self._files[build] = h5py.File(_hdf5_path(build), "r")
        f = self._files[build]
        r0 = f["slices/camera_data/visible/0"][li]
        r1 = f["slices/camera_data/visible/1"][li]
        if self.photos_only:
            return {"build": build, "layer": li, "raw0": ornl_raw_uint8(r0), "raw1": ornl_raw_uint8(r1)}
        return {
            "build": build, "layer": li,
            "raw0": ornl_raw_uint8(r0), "raw1": ornl_raw_uint8(r1),          # v5 · v6 의 입력
            "str0": ornl_image_to_uint8(r0), "str1": ornl_image_to_uint8(r1),  # v4 의 입력
            "dscnn": ornl_segmentation_argmax_v6(f["slices/segmentation_results"], li),
            "part": np.asarray(f["slices/part_ids"][li] > 0),
        }


def _identity(x):
    return x


def _rel(p: Path) -> str:
    """ckpt 경로를 DefSeg_AM/ 기준으로 적는다 (호스트와 컨테이너에서 같은 글자가 되게)."""
    try:
        return str(Path(p).resolve().relative_to(cfg.OUTPUT_DIR.resolve()))
    except ValueError:
        return str(p)


# ---------------------------------------------------------------------------
# 그림 — 타일과 쪽
# ---------------------------------------------------------------------------

def display_u8(raw_u8: np.ndarray) -> np.ndarray:
    """보기 좋게 대비만 늘린 사진. 그림에만 쓴다 — 모델 입력이 아니다.

    1–99 백분위를 0–255 로 늘리되 위쪽 끝은 중앙값 + 9σ (σ = 1.4826 × MAD) 를 넘지 않게 한다.
    용융 직후 사진은 파트가 아주 밝아서 백분위만 쓰면 분말이 까맣게 눌려 분말 위의 자국이 보이지 않는다.
    """
    a = raw_u8.astype(np.float32)
    lo, p99 = np.percentile(a, [1, 99])
    med = float(np.median(a))
    sig = max(1.4826 * float(np.median(np.abs(a - med))), 1.0)
    hi = max(min(float(p99), med + 9.0 * sig), float(lo) + 1.0)
    return (np.clip((a - lo) / (hi - lo), 0, 1) * 255).astype(np.uint8)


def _pool_factor(h: int) -> int:
    return max(1, int(round(h / TILE_TARGET)))


def tile_gray(u8: np.ndarray, device: torch.device) -> np.ndarray:
    k = _pool_factor(u8.shape[0])
    t = torch.from_numpy(u8).to(device).float()[None, None]
    return F.avg_pool2d(t, k)[0, 0].round().clamp(0, 255).byte().cpu().numpy()


def tile_label(lab: torch.Tensor, palette: torch.Tensor) -> np.ndarray:
    """(H, W) long (0..10) → k×k 칸마다 클래스 하나를 골라 색을 입힌 (h, w, 3) uint8.

    칸 안에서 분말이 아닌 화소가 1/3 이상이면 (3×3 이면 3개 이상) 그 가운데 가장 많은 클래스를, 아니면 분말을 보인다.
    같은 수면 결함 쪽을 보인다. 색을 평균하지 않는 까닭: 폭 1–2 화소인 Printed (파랑) 와 Spatter (자홍) 가 섞이면
    PS 의 범례 색과 같은 보라가 되어, 아무도 내지 않은 PS 가 그림에 있는 것처럼 보인다. 이 규칙이면 범례의 색만 나오고,
    폭 1 화소인 RS 선도 (칸의 3/9) 그대로 남는다.
    """
    k = _pool_factor(lab.shape[0])
    if k == 1:
        return palette[lab].round().clamp(0, 255).byte().cpu().numpy()
    onehot = F.one_hot(lab, N_BINS).permute(2, 0, 1)[None].float()
    cnt = F.avg_pool2d(onehot, k)[0] * (k * k)            # (11, h, w) — 칸 안의 클래스별 화소 수
    fg = cnt[1:].clone()
    fg[1:] += 0.5                                         # Printed 와 같은 수면 결함 쪽
    top = fg.argmax(dim=0) + 1
    show = cnt[1:].sum(dim=0) >= float(np.ceil(k * k / 3))
    out = torch.where(show, top, torch.zeros_like(top))
    return palette[out].round().clamp(0, 255).byte().cpu().numpy()


def _font(size: int, bold: bool = False) -> ImageFont.FreeTypeFont:
    import matplotlib
    name = "DejaVuSans-Bold.ttf" if bold else "DejaVuSans.ttf"
    return ImageFont.truetype(str(Path(matplotlib.get_data_path()) / "fonts" / "ttf" / name), size)


PAGE_NOTE = (
    "No human ground truth exists for these builds — columns 3–6 are all model predictions. "
    "Old classes are shown with pbf-2026.11 ids: Swelling → SE, old Incomplete Spreading → IS, Misprint / Under Melting → no class.",
    "Inputs: v5 and v6 get raw brightness, v4 the per-layer stretched photos it was trained on. No TTA, no post-processing. "
    "Photos are 3×3 averages, contrast-stretched for display only; each label pixel shows the most common non-powder class "
    "of its 3×3 block (Powder if under 3 of 9 pixels).",
)


def render_page(rows: list[dict], columns: list[tuple[str, str]], title: str, out_path: Path) -> None:
    """rows = [{"label": str, "tiles": [uint8 (T, T) | (T, T, 3), …]}] → PNG 한 장."""
    t = rows[0]["tiles"][0].shape[0]
    n_col = len(columns)
    gap, left, right = 8, 62, 16
    title_h, head_h, legend_h, note_h = 62, 74, 100, 74
    w = left + n_col * t + (n_col - 1) * gap + right
    h = title_h + head_h + len(rows) * t + (len(rows) - 1) * gap + legend_h + note_h
    page = Image.new("RGB", (w, h), (255, 255, 255))
    d = ImageDraw.Draw(page)
    ink, ink2 = (11, 11, 11), (82, 81, 78)

    d.text((left, 12), title, font=_font(32, True), fill=ink)
    f_main, f_sub = _font(28, True), _font(20)
    for c, (main, sub) in enumerate(columns):
        cx = left + c * (t + gap) + t // 2
        d.text((cx, title_h + 20), main, font=f_main, fill=ink, anchor="mm")
        d.text((cx, title_h + 52), sub, font=f_sub, fill=ink2, anchor="mm")

    f_row = _font(28, True)
    for r, row in enumerate(rows):
        y = title_h + head_h + r * (t + gap)
        for c, tile in enumerate(row["tiles"]):
            page.paste(Image.fromarray(tile).convert("RGB"), (left + c * (t + gap), y))
        tmp = Image.new("RGB", (t, 44), (255, 255, 255))
        ImageDraw.Draw(tmp).text((t // 2, 22), row["label"], font=f_row, fill=ink, anchor="mm")
        page.paste(tmp.rotate(90, expand=True), (8, y))

    # 범례 — 두 줄 (클래스 10개 + 대응 없음)
    entries = [(cfg.DIVA_COLORS[c], f"{cfg.CLASS_ABBRS_V6[c]}  {cfg.CLASS_NAMES_V6[c]}") for c in range(cfg.N_CLASSES_V6)]
    entries.append((cfg.UNLABELED_COLOR, "no class (old-scheme class without a new id)"))
    f_leg = _font(22)
    y0 = title_h + head_h + len(rows) * t + (len(rows) - 1) * gap + 16
    per_row = 6
    cell = (w - left - right) // per_row
    for i, (color, name) in enumerate(entries):
        x = left + (i % per_row) * cell
        y = y0 + (i // per_row) * 38
        d.rectangle([x, y, x + 26, y + 26], fill=tuple(color), outline=(120, 120, 120))
        d.text((x + 36, y + 13), name, font=f_leg, fill=ink, anchor="lm")

    f_note = _font(19)
    for i, line in enumerate(PAGE_NOTE):
        d.text((left, y0 + 84 + i * 26), line, font=f_note, fill=ink2)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    page.save(out_path)


# ---------------------------------------------------------------------------
# 수치
# ---------------------------------------------------------------------------

def _bincount2(a: torch.Tensor, b: torch.Tensor, na: int, nb: int) -> np.ndarray:
    return torch.bincount((a * nb + b).reshape(-1), minlength=na * nb).reshape(na, nb).cpu().numpy()


def agreement(cm: np.ndarray) -> dict:
    """cm[a, b] = v6 가 a, 상대 모델이 b 로 낸 화소 수 (10 × 11). 클래스별 IoU 는 두 예측을 견준 것이다 (정답과의 IoU 가 아니다)."""
    row, col = cm.sum(axis=1), cm.sum(axis=0)[: cfg.N_CLASSES_V6]
    iou = []
    for c in range(cfg.N_CLASSES_V6):
        union = row[c] + col[c] - cm[c, c]
        iou.append(float(cm[c, c] / union) if union > 0 else None)
    inter = cm[2:4, 2:4].sum()
    union = row[2:4].sum() + col[2:4].sum() - inter
    total = cm.sum()
    diag = float(np.trace(cm[:, : cfg.N_CLASSES_V6]))
    return {
        "iou": iou,
        "is_ps_union_iou": float(inter / union) if union > 0 else None,
        "pixel_agreement": diag / total,
        "pixel_agreement_is_ps_merged": (diag + float(cm[2, 3] + cm[3, 2])) / total,
    }


def part_overlap(pc: np.ndarray) -> dict:
    """pc[0 | 1, b] = CAD 파트 밖 | 안의 화소 가운데 b 로 예측한 수 (2 × 11)."""
    like = list(PART_LIKE)
    tp = float(pc[1, like].sum())
    pred = float(pc[:, like].sum())
    cad = float(pc[1].sum())
    off = float(pc[0].sum())
    return {
        "part_iou": tp / (pred + cad - tp) if (pred + cad - tp) > 0 else None,
        "cad_recall": tp / cad if cad > 0 else None,
        "pred_precision": tp / pred if pred > 0 else None,
        "off_part_printed_rate": float(pc[0, 1]) / off if off > 0 else None,
        "off_part_part_like_rate": float(pc[0, like].sum()) / off if off > 0 else None,
        "cad_pixels_by_class": [float(v) / cad if cad > 0 else None for v in pc[1]],
        "off_part_pixels_by_class": [float(v) / off if off > 0 else None for v in pc[0]],
    }


def rs_row_stats(pred: torch.Tensor) -> tuple[float, int, int]:
    """RS 선이 가로로 얼마나 이어지는가 — (RS 가 있는 행에서 RS 가 덮은 폭 비율의 평균, 그런 행 수, 폭의 90% 넘게 이은 행 수).

    RS 는 리코터가 지나가는 방향으로 난 줄이고, 그 방향이 사진의 가로다 (`cfg.CANONICAL_RECOAT_AXIS`).
    "RS 가 있는 행" = RS 화소가 RS_ROW_MIN 개 넘는 행 (낱개 화소는 세지 않는다).
    """
    per_row = (pred == RS_ID).sum(dim=1)
    rows = per_row >= RS_ROW_MIN
    n_rows = int(rows.sum())
    w = pred.shape[1]
    cover = float(per_row[rows].float().mean()) / w if n_rows else float("nan")
    return cover, n_rows, int((per_row >= 0.9 * w).sum())


def rs_summary(a: np.ndarray) -> dict:
    """a = (레이어, 3) — rs_row_stats 를 쌓은 것."""
    has = ~np.isnan(a[:, 0])
    return {
        "row_cover_median": float(np.median(a[has, 0])) if has.any() else None,
        "rows_median": float(np.median(a[:, 1])),
        "full_width_rows": int(a[:, 2].sum()),
        "layers_with_rs_rows": int(has.sum()),
    }


def fixed_debris(hits: np.ndarray, n_layers: int) -> dict | None:
    """hits[y, x] = 그 화소를 Debris 로 낸 레이어 수. 뽑힌 레이어의 절반 넘게 Debris 인 곳을 센다.

    레이어도 빌드도 바뀌는데 같은 화소에 계속 있는 점은 분말 위의 이물이 아니라 카메라 쪽 (렌즈 · 창 · 센서) 에 붙은 것이다.
    레이어가 FIXED_MIN_LAYERS 보다 적으면 (스모크) 뜻이 없어 내지 않는다.
    """
    if n_layers < FIXED_MIN_LAYERS:
        return None
    from scipy import ndimage as ndi

    th = int(np.ceil(0.5 * n_layers))
    mask = hits >= th
    lab, k = ndi.label(mask)
    total = int(hits.sum())
    spots = []
    for j in range(1, k + 1):
        ys, xs = np.nonzero(lab == j)
        spots.append({"x": int(round(float(xs.mean()))), "y": int(round(float(ys.mean()))), "px": int(len(xs)),
                      "max_layers": int(hits[ys, xs].max())})
    return {
        "min_layers": th, "n_spots": int(k), "px": int(mask.sum()),
        "share_of_debris_px": float(hits[mask].sum()) / total if total else None,
        "spots": spots,
    }


def _pct(v: float | None) -> str:
    if v is None:
        return "–"
    p = 100.0 * v
    if p == 0:
        return "0"
    if p >= 10:
        return f"{p:.1f}"
    if p >= 0.1:
        return f"{p:.2f}"
    if p >= 0.0001:
        return f"{p:.4f}"
    return "<0.0001"


def _f3(v: float | None) -> str:
    return "–" if v is None else f"{v:.3f}"


def write_summary_md(path: Path, s: dict) -> None:
    ab = list(cfg.CLASS_ABBRS_V6)
    models = s["models"]
    lines = [
        f"# ORNL {len(s['builds'])} 빌드 무작위 {s['n_layers']} 레이어 — 모델끼리의 예측 비교 (참고용)",
        "",
        "ORNL 빌드에는 사람 GT 가 없다. 아래 수치는 모델끼리 견준 것이지 정확도가 아니고, 어떤 결정 (길이 · ckpt 선정 · 합격 판정) 에도 쓰지 않는다 (PLAN_v6 §6.5).",
        "옛 예측 (DSCNN 12-class, v4 · v5 11-class) 은 새 id 로 옮겼다: Swelling → SE, 옛 IS → IS, Misprint · Under Melting → 대응 없음.",
        "입력 밝기: v5 · v6 는 원본 그대로, v4 는 레이어마다 늘린 사진 (학습 때 본 대로). "
        + ("TTA 8-way 평균, 후처리 없음." if s["tta"] else "TTA · 후처리 없음."),
        "",
        "| 모델 | ckpt |",
        "|---|---|",
    ]
    for m in models:
        lines.append(f"| {MODEL_INFO[m][2]} | {s['checkpoints'].get(m, 'HDF5 `slices/segmentation_results`')} |")
    lines += [
        "",
        "## 뽑은 레이어",
        "",
        f"seed {s['seed']}. {s['rule']}",
        "",
        "| 빌드 | 파일 | 전체 레이어 | 뽑은 레이어 |",
        "|---|---|---:|---|",
    ]
    for b, info in s["builds"].items():
        lines.append(f"| {b} | {info['file']} | {info['n_layers']:,} | {', '.join(str(v) for v in info['layers'])} |")

    lines += [
        "",
        "## 클래스별 예측 화소 비율 (%) — 전체",
        "",
        "| 모델 | " + " | ".join(ab) + " | 대응 없음 |",
        "|---|" + "---:|" * (len(ab) + 1),
    ]
    for m in models:
        lines.append(f"| {MODEL_INFO[m][2]} | " + " | ".join(_pct(v) for v in s["share"][m]) + " |")

    lw, ds = s.get("layers_with_class"), s.get("defect_share_per_layer")
    if lw and ds:
        lines += [
            "",
            f"레이어마다 본 것 — 결함 (IS–DB) 으로 낸 화소 비율 (%) 과, 그 클래스를 {lw['min_px']:,} 화소 넘게 낸 레이어 수 ({s['n_layers']} 레이어 가운데)",
            "",
            "| 모델 | 결함 비율 중앙값 | 최대 | " + " | ".join(ab[2:]) + " | 대응 없음 |",
            "|---|---:|---:|" + "---:|" * (len(ab) - 1),
        ]
        for m in models:
            lines.append(
                f"| {MODEL_INFO[m][2]} | {100 * ds[m]['median']:.2f} | {100 * ds[m]['max']:.1f} | "
                + " | ".join(str(v) for v in lw["counts"][m][2:]) + " |"
            )

    lines += [
        "",
        "## 빌드별 예측 화소 비율 (%)",
        "",
        "| 빌드 | 모델 | " + " | ".join(ab[1:]) + " | 대응 없음 |",
        "|---|---|" + "---:|" * len(ab),
    ]
    for b in s["builds"]:
        for m in models:
            lines.append(f"| {b} | {MODEL_INFO[m][2]} | " + " | ".join(_pct(v) for v in s["share_by_build"][b][m][1:]) + " |")

    lines += [
        "",
        "## v6 와의 일치 — 두 예측을 견준 클래스별 IoU",
        "",
        "v6 의 예측과 상대 모델의 예측 (새 id 로 옮긴 것) 이 겹치는 정도다. 옛 모델은 PS 를 낼 수 없으므로 IS 와 PS 는 합쳐서 본다 (IS ∪ PS).",
        "",
        "| 상대 | Pow | Prt | IS ∪ PS | RS | RH | SE | SP | OM | DB | 화소 일치율 | 〃 (IS · PS 합침) |",
        "|---|" + "---:|" * 11,
    ]
    for m in models:
        if m == "v6":
            continue
        a = s["agreement"][m]
        cells = [_f3(a["iou"][0]), _f3(a["iou"][1]), _f3(a["is_ps_union_iou"])] + [_f3(a["iou"][c]) for c in range(4, 10)]
        lines.append(
            f"| {MODEL_INFO[m][2]} | " + " | ".join(cells)
            + f" | {100 * a['pixel_agreement']:.2f}% | {100 * a['pixel_agreement_is_ps_merged']:.2f}% |"
        )

    lines += [
        "",
        "## CAD 파트 마스크와의 겹침",
        "",
        "`slices/part_ids > 0` (설계상 파트 단면) 과 견준 것이다 — 이 빌드에서 모델 예측이 아닌 기준은 이것뿐이다.",
        "파트 쪽 클래스 = Printed · SE · OM, 옛 모델의 대응 없음 (Under Melting · Misprint) 포함.",
        "",
        "| 모델 | 파트 영역 IoU | CAD 파트 화소 → 파트 쪽 클래스 | 파트 쪽 예측 → CAD 파트 안 | 파트 밖 화소 → Printed | "
        + " | ".join(f"{b} IoU" for b in s["builds"]) + " |",
        "|---|---:|---:|---:|---:|" + "---:|" * len(s["builds"]),
    ]
    pb = s.get("part_iou_by_build", {})
    for m in models:
        p = s["part"][m]
        lines.append(
            f"| {MODEL_INFO[m][2]} | {_f3(p['part_iou'])} | {_pct(p['cad_recall'])}% | {_pct(p['pred_precision'])}% | {_pct(p['off_part_printed_rate'])}% | "
            + " | ".join(_f3(pb.get(b, {}).get(m)) for b in s["builds"]) + " |"
        )
    lines += [
        "",
        "CAD 파트 화소를 무엇으로 냈는가 (%)",
        "",
        "| 모델 | " + " | ".join(ab) + " | 대응 없음 |",
        "|---|" + "---:|" * (len(ab) + 1),
    ]
    for m in models:
        lines.append(f"| {MODEL_INFO[m][2]} | " + " | ".join(_pct(v) for v in s["part"][m]["cad_pixels_by_class"]) + " |")
    lines += [
        "",
        "CAD 파트 밖의 화소 (분말 영역) 를 무엇으로 냈는가 (%)",
        "",
        "| 모델 | " + " | ".join(ab) + " | 대응 없음 |",
        "|---|" + "---:|" * (len(ab) + 1),
    ]
    for m in models:
        lines.append(f"| {MODEL_INFO[m][2]} | " + " | ".join(_pct(v) for v in s["part"][m]["off_part_pixels_by_class"]) + " |")

    rs = s.get("rs_rows")
    if rs:
        lines += [
            "",
            "## RS 선의 가로 길이",
            "",
            f"RS 는 리코터가 지나가는 방향 (사진의 가로) 으로 난 줄이다. RS 화소가 {rs['min_px_per_row']} 개 넘게 있는 행만 골라 "
            "그 행 폭의 얼마를 RS 로 냈는지 본다 (1.0 = 판의 끝에서 끝까지 이은 선). 폭 비율은 레이어별 평균의 중앙값이다.",
            "",
            "| 모델 | 행 폭 가운데 RS | RS 가 있는 행 수 (레이어 중앙값) | 폭의 90% 넘게 이은 행 (전체 합) | "
            + " | ".join(f"{b} 폭 비율" for b in s["builds"]) + " |",
            "|---|---:|---:|---:|" + "---:|" * len(s["builds"]),
        ]
        for m in models:
            a = rs["all"][m]
            lines.append(
                f"| {MODEL_INFO[m][2]} | {_f3(a['row_cover_median'])} | {a['rows_median']:.0f} | {a['full_width_rows']:,} | "
                + " | ".join(_f3(rs["by_build"][b][m]["row_cover_median"]) for b in s["builds"]) + " |"
            )

    fd = s.get("fixed_debris")
    if fd and any(v for v in fd.values()):
        th = next(v["min_layers"] for v in fd.values() if v)
        lines += [
            "",
            "## 고정 위치의 Debris",
            "",
            f"같은 화소를 뽑힌 레이어의 절반 넘게 ({th} 레이어 이상) Debris 로 낸 곳이다. 레이어도 빌드도 바뀌는데 같은 자리에 있으므로 "
            "분말 위의 이물이 아니라 카메라 쪽 (렌즈 · 창 · 센서) 에 붙은 점으로 본다.",
            "",
            "| 모델 | 그런 곳 | 화소 | 그 모델의 Debris 화소 가운데 | 위치 (x, y) · 화소 수 · 가장 많이 겹친 레이어 수 |",
            "|---|---:|---:|---:|---|",
        ]
        for m in models:
            v = fd.get(m)
            if not v:
                continue
            spots = "; ".join(f"({p['x']}, {p['y']}) · {p['px']} px · {p['max_layers']}" for p in v["spots"][:8])
            lines.append(f"| {MODEL_INFO[m][2]} | {v['n_spots']} | {v['px']:,} | {_pct(v['share_of_debris_px'])}% | {spots or '–'} |")

    ov = ", ".join(f"{b} {li}" for b, li in s.get("overview_layers", {}).items())
    lines += [
        "",
        "그림: `overview.png` (빌드마다 한 레이어 — 뽑힌 레이어 가운데 v6 의 결함 화소 비율이 중앙값인 것: " + ov + "), "
        "`pages/page_XX_<빌드>.png` (뽑힌 레이어 전부, 5개씩), `summary.png`.",
        "예측 타일은 원본의 3×3 칸마다 클래스 하나를 보인다 — 분말이 아닌 화소가 3개 이상이면 그 가운데 가장 많은 클래스, 아니면 분말 "
        "(색을 평균하지 않으므로 범례에 있는 색만 나온다).",
    ]
    ps = ab.index("PS")
    if all(s["share"][m][ps] == 0 for m in models):
        lines.append(
            "PS 를 낸 모델은 없다. 그림을 줄여 보면 얇은 고리가 PS 의 범례 색 (보라) 처럼 보일 수 있는데, "
            "Printed (파랑) 와 Spatter (자홍) 가 한 화소씩 붙어 있어서 그렇다 — 원래 크기로 보면 두 색이 따로 보인다."
        )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def plot_summary(path: Path, s: dict) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    models = s["models"]
    order = [m for m in ("v6", "v5", "v4", "dscnn") if m in models]      # 계열 순서 (색 칸 1 · 2 · 3 · 무채색)
    short = {"v6": "v6", "v5": "v5", "v4": "v4", "dscnn": "DSCNN"}
    ab = list(cfg.CLASS_ABBRS_V6)
    surface = "#fcfcfb"
    plt.rcParams.update({"font.size": 11, "axes.edgecolor": GRID, "axes.labelcolor": INK2,
                         "xtick.color": INK2, "ytick.color": INK, "text.color": INK})
    fig = plt.figure(figsize=(17.0, 8.2), facecolor=surface)
    gs = fig.add_gridspec(
        2, 2, width_ratios=[1.3, 1.0], height_ratios=[2.7, 1.0], hspace=0.46, wspace=0.26,
        left=0.125, right=0.985, top=0.845, bottom=0.085,
    )
    ax_a = fig.add_subplot(gs[:, 0])
    ax_b = fig.add_subplot(gs[0, 1])
    ax_c = fig.add_subplot(gs[1, 1])

    def style(ax):
        ax.set_facecolor(surface)
        for sp in ("top", "right", "left"):
            ax.spines[sp].set_visible(False)
        ax.tick_params(length=0)
        ax.grid(axis="x", color=GRID, linewidth=0.8)
        ax.set_axisbelow(True)

    # (A) 클래스별 예측 화소 비율 — log 축 점 그림. 0 은 축 왼쪽의 "0" 눈금에 빈 표식으로 둔다
    rows = list(range(cfg.N_CLASSES_V6)) + [NONE_ID]
    labels = [f"{ab[c]}  {cfg.CLASS_NAMES_V6[c]}" for c in range(cfg.N_CLASSES_V6)] + ["no class"]
    zero_x, lo_x = 1e-5, 1e-4
    offs = np.linspace(-0.27, 0.27, len(order))
    for k, m in enumerate(order):
        color, marker = MODEL_STYLE[m]
        for r, c in enumerate(rows):
            v = 100.0 * s["share"][m][c]
            y = r + offs[k]
            if v <= 0:
                ax_a.plot(zero_x, y, marker=marker, ms=7, mfc="none", mec=color, mew=1.2, ls="none")
            else:
                ax_a.plot(max(v, lo_x), y, marker=marker, ms=7.5, color=color, mec=surface, mew=0.8, ls="none")
    ax_a.set_xscale("log")
    ax_a.set_xlim(zero_x / 2.5, 200)
    ax_a.set_xticks([zero_x, 1e-4, 1e-3, 1e-2, 1e-1, 1, 10, 100], ["0", "≤0.0001", "0.001", "0.01", "0.1", "1", "10", "100"])
    ax_a.minorticks_off()
    ax_a.set_yticks(range(len(rows)), labels)
    ax_a.set_ylim(len(rows) - 0.5, -0.5)
    for r in range(len(rows) - 1):
        ax_a.axhline(r + 0.5, color=GRID, linewidth=0.6)
    ax_a.set_xlabel("share of all pixels in the sampled layers (%, log scale; hollow marker = none predicted)", loc="left")
    ax_a.set_title("Predicted area by class", loc="left", fontsize=13, fontweight="bold")
    style(ax_a)
    tr = ax_a.get_yaxis_transform()                                       # x: 축 비율, y: 데이터
    ax_a.text(1.012, -0.62, "v6, %", transform=tr, va="center", ha="left", fontsize=10, color=INK2, clip_on=False)
    for r, c in enumerate(rows):                                          # v6 의 값만 숫자로 (전부 적지 않는다)
        if c < cfg.N_CLASSES_V6:
            ax_a.text(1.012, r, _pct(s["share"]["v6"][c]), transform=tr, va="center", ha="left",
                      fontsize=10, color=INK2, clip_on=False)

    # (B) v6 와의 일치 — 클래스별 IoU
    b_rows = [("Pow", lambda a: a["iou"][0]), ("Prt", lambda a: a["iou"][1]), ("IS ∪ PS", lambda a: a["is_ps_union_iou"])]
    b_rows += [(ab[c], (lambda a, c=c: a["iou"][c])) for c in range(4, 10)]
    others = [m for m in order if m != "v6"]
    offs_b = np.linspace(-0.22, 0.22, max(len(others), 1))
    for k, m in enumerate(others):
        color, marker = MODEL_STYLE[m]
        for r, (_, fn) in enumerate(b_rows):
            v = fn(s["agreement"][m])
            if v is not None:
                ax_b.plot(v, r + offs_b[k], marker=marker, ms=7.5, color=color, mec=surface, mew=0.8, ls="none")
    ax_b.set_xlim(-0.03, 1.0)
    ax_b.set_yticks(range(len(b_rows)), [n for n, _ in b_rows])
    ax_b.set_ylim(len(b_rows) - 0.5, -0.5)
    for r in range(len(b_rows) - 1):
        ax_b.axhline(r + 0.5, color=GRID, linewidth=0.6)
    ax_b.set_xlabel("IoU of v6's prediction vs the other model's (agreement, not accuracy)", loc="left")
    ax_b.set_title("Agreement with v6, by class", loc="left", fontsize=13, fontweight="bold")
    style(ax_b)

    # (C) CAD 파트 마스크와의 겹침
    for k, m in enumerate(order):
        color, _ = MODEL_STYLE[m]
        v = s["part"][m]["part_iou"] or 0.0
        ax_c.barh(k, v, height=0.58, color=color)
        ax_c.text(v + 0.012, k, f"{v:.3f}", va="center", ha="left", fontsize=10.5, color=INK)
    ax_c.set_xlim(0, 1.0)
    ax_c.set_yticks(range(len(order)), [short[m] for m in order])
    ax_c.set_ylim(len(order) - 0.5, -0.5)
    ax_c.set_xlabel("IoU: predicted part classes (Printed, SE, OM, no class) vs CAD part mask", loc="left")
    ax_c.set_title("Overlap with the CAD part mask", loc="left", fontsize=13, fontweight="bold")
    style(ax_c)

    fig.suptitle(
        f"ORNL builds {', '.join(s['builds'])} — {s['n_layers']} random layers: model-vs-model comparison (no human ground truth)",
        x=0.012, y=0.975, ha="left", fontsize=14.5, fontweight="bold",
    )
    handles = [
        plt.Line2D([], [], marker=MODEL_STYLE[m][1], color=MODEL_STYLE[m][0], ls="none", ms=8, label=MODEL_INFO[m][2])
        for m in order
    ]
    fig.legend(handles=handles, loc="upper left", bbox_to_anchor=(0.006, 0.94), ncol=len(order), frameon=False,
               fontsize=11.5, handletextpad=0.3, columnspacing=2.0)
    fig.savefig(path, dpi=140, facecolor=fig.get_facecolor())
    plt.close(fig)


# ---------------------------------------------------------------------------
# main
# ---------------------------------------------------------------------------

def main():
    ap = argparse.ArgumentParser(description="ORNL 빌드의 무작위 레이어에서 DSCNN · v4 · v5 · v6 예측 비교 (참고용)")
    ap.add_argument("--run-name", type=str, default=None)
    ap.add_argument("--backbone", type=str, default=cfg.DINO_BACKBONE)
    ap.add_argument("--fold", type=str, default=None, choices=list(cfg.CV_FOLDS),
                    help="검증 fold 의 ckpt 로 (기본: 전수 학습 모델)")
    ap.add_argument("--which", type=str, default="last", choices=["last", "best", "last_raw"])
    ap.add_argument("--checkpoint", type=str, default=None, help="v6 (10-class) ckpt 를 직접 지정")
    ap.add_argument("--v5-checkpoint", type=str, default=str(cfg.CHECKPOINT_DIR / cfg.V5_RUN_NAME / "full" / "last.pt"))
    ap.add_argument("--v4-checkpoint", type=str, default=str(cfg.CHECKPOINT_DIR / cfg.V4_RUN_NAME / "stage2_best.pt"))
    ap.add_argument("--builds", type=str, nargs="+", default=list(cfg.ORNL_BUILD_FILES), choices=list(cfg.ORNL_BUILD_FILES))
    ap.add_argument("--n-per-build", type=int, default=20)
    ap.add_argument("--seed", type=int, default=cfg.RANDOM_SEED)
    ap.add_argument("--layers-json", type=str, default=None, help="이미 뽑아 둔 목록 (layers.json) 을 그대로 쓴다")
    ap.add_argument("--img-size", type=int, default=cfg.IMG_SIZE)
    ap.add_argument("--tta", action="store_true", help="세 모델 모두 flip(LR) × flip(UD) × rot180 8-way 평균")
    ap.add_argument("--rows-per-page", type=int, default=5)
    ap.add_argument("--workers", type=int, default=6, help="HDF5 를 읽는 프로세스 수")
    ap.add_argument("--no-save-preds", action="store_true")
    ap.add_argument("--from-preds", action="store_true",
                    help="모델을 돌리지 않고, 앞선 실행이 남긴 preds/ · layers.json · summary.json 으로 그림과 수치만 다시 만든다 (GPU 를 쓰지 않는다)")
    ap.add_argument("--out-dir", type=str, default=None)
    args = ap.parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    run_name = args.run_name or cfg.default_run_name(args.backbone, args.img_size)

    # ----- 앞선 실행의 예측을 다시 쓰는 경우 (--from-preds): 모델도 GPU 도 쓰지 않는다 -----
    prev: dict | None = None
    if args.from_preds:
        prev_dir = Path(args.out_dir) if args.out_dir else (
            cfg.run_fig_dir(run_name, args.fold)
            / f"ornl_random{len(args.builds) * args.n_per_build}_{args.which}{'_tta' if args.tta else ''}"
        )
        for need in ("summary.json", "layers.json", "preds"):
            if not (prev_dir / need).exists():
                ap.error(f"--from-preds: {prev_dir / need} 가 없다 — 모델을 돌린 실행이 먼저 있어야 한다")
        prev = json.loads((prev_dir / "summary.json").read_text(encoding="utf-8"))
        args.layers_json = args.layers_json or str(prev_dir / "layers.json")
        args.out_dir = str(prev_dir)
        device = torch.device("cpu")
        print(f"[compare] --from-preds: {prev_dir} 의 예측으로 그림 · 수치만 다시 만든다 (모델을 돌리지 않는다)")

    # ----- 모델 -----
    old: dict[str, torch.nn.Module] = {}
    if prev is not None:
        m6, c6 = None, {"step": prev.get("ckpt_step"), "weights": prev.get("ckpt_weights")}
        img_size, models, checkpoints = prev["img_size"], list(prev["models"]), dict(prev["checkpoints"])
        lut = None
    else:
        ckpt6 = Path(args.checkpoint) if args.checkpoint else cfg.run_ckpt_dir(run_name, args.fold) / f"{args.which}.pt"
        if not ckpt6.exists():
            ap.error(f"ckpt not found: {ckpt6}")
        m6, c6 = load_model(ckpt6, device)
        img_size = round_to_patch(args.img_size, m6.patch_size)
        checkpoints = {"v6": _rel(ckpt6)}
        for name, p in (("v4", Path(args.v4_checkpoint)), ("v5", Path(args.v5_checkpoint))):
            if p.exists():
                old[name], _ = load_model(p, device, n_classes=cfg.N_CLASSES_V4)
                checkpoints[name] = _rel(p)
            else:
                print(f"[compare] {name} ckpt 없음 ({p}) — 그 열을 뺀다")
        models = ["dscnn"] + [m for m in ("v4", "v5") if m in old] + ["v6"]
        lut = old_to_new_lut("se", device)
    palette = torch.tensor(list(cfg.DIVA_COLORS) + [cfg.UNLABELED_COLOR], dtype=torch.float32, device=device)

    # ----- 레이어 -----
    rule = (
        "빌드마다 레이어 번호를 seed 로 섞어 앞에서부터, CAD 파트 마스크가 비어 있지 않고 두 사진이 비어 있지 않은 "
        f"(가운데 가로 띠의 평균 밝기 ≥ {BLANK_MEAN:g}) 레이어를 고른다."
    )
    if args.layers_json:
        picked = json.loads(Path(args.layers_json).read_text(encoding="utf-8"))
        builds, seed, rule = picked["builds"], picked["seed"], picked.get("rule", rule)
        print(f"[compare] 레이어 목록을 {args.layers_json} 에서 읽었다")
    else:
        seed = args.seed
        builds = {}
        for b in args.builds:
            t0 = time.time()
            builds[b] = pick_layers(b, args.n_per_build, seed)
            print(f"[compare] {b}: {builds[b]['n_layers']} 레이어 가운데 {len(builds[b]['layers'])} 개 "
                  f"(건너뜀 {len(builds[b]['skipped'])})  {time.time() - t0:.0f}s  → {builds[b]['layers']}")
    items = [(b, li) for b, info in builds.items() for li in info["layers"]]
    n_layers = len(items)

    out_dir = Path(args.out_dir) if args.out_dir else (
        cfg.run_fig_dir(run_name, args.fold) / f"ornl_random{n_layers}_{args.which}{'_tta' if args.tta else ''}"
    )
    (out_dir / "pages").mkdir(parents=True, exist_ok=True)
    stale = sorted((out_dir / "pages").glob("page_*.png"))
    for p in stale:          # 앞선 실행의 쪽 (쪽 수나 이름이 다르면 섞여 남는다) — 이 스크립트가 만든 그림만 지운다
        p.unlink()
    if stale:
        print(f"[compare] 앞선 실행의 쪽 그림 {len(stale)} 장을 지우고 다시 만든다")
    pred_dir = out_dir / "preds"
    save_preds = not args.no_save_preds and prev is None      # --from-preds 면 이미 있는 것을 읽기만 한다
    if save_preds:
        pred_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "layers.json").write_text(
        json.dumps({"seed": seed, "rule": rule, "builds": builds}, ensure_ascii=False, indent=1), encoding="utf-8",
    )
    print(f"[compare] {n_layers} 레이어 | 모델 {models} | img_size={img_size} | tta={args.tta} | device={device} | out={out_dir}")

    # ----- 추론 + 수치 + 그림 -----
    loader = DataLoader(
        OrnlLayers(items, photos_only=prev is not None), batch_size=None, shuffle=False, num_workers=args.workers,
        collate_fn=_identity, prefetch_factor=2 if args.workers > 0 else None,
    )
    counts = np.zeros((n_layers, len(models), N_BINS), dtype=np.int64)                 # 모델 × 클래스
    cross = np.zeros((n_layers, len(models), cfg.N_CLASSES_V6, N_BINS), dtype=np.int64)   # v6 × 모델
    part = np.zeros((n_layers, len(models), 2, N_BINS), dtype=np.int64)                # CAD 밖 · 안 × 클래스
    rs_rows = np.zeros((n_layers, len(models), 3), dtype=np.float64)                   # RS 행: 덮은 폭 비율 · 행 수 · 끝까지 이은 행 수
    db_hits: torch.Tensor | None = None                                                # (모델, H, W) — 그 화소를 Debris 로 낸 레이어 수
    db_hits_ok = True
    photo_mean = np.zeros((n_layers, 2), dtype=np.float32)
    columns = [("visible/0", "after melting"), ("visible/1", "after spreading")] + [MODEL_INFO[m][:2] for m in models]
    # 쪽은 빌드가 바뀔 때도 넘긴다 (한 쪽에 한 빌드)
    n_pages = sum(int(np.ceil(len(info["layers"]) / args.rows_per_page)) for info in builds.values())
    page_rows: list[dict] = []
    build_rows: list[dict] = []          # 지금 빌드의 줄 전부 — 빌드가 끝나면 개요 그림에 넣을 한 줄을 고른다
    overview_rows: list[dict] = []
    page_no = 0
    t_start = time.time()

    def flush_page():
        nonlocal page_rows, page_no
        if not page_rows:
            return
        page_no += 1
        bs = sorted({r["build"] for r in page_rows})
        title = (
            f"ORNL {' / '.join(bs)} — random layers, DSCNN vs DefSeg-AM v4 · v5 · v6 "
            f"(v6 = {cfg.run_tag(args.fold)} {args.which}{', TTA' if args.tta else ''})    page {page_no} / {n_pages}"
        )
        render_page(page_rows, columns, title, out_dir / "pages" / f"page_{page_no:02d}_{'-'.join(bs)}.png")
        page_rows = []

    for i, item in enumerate(loader):
        b, li = item["build"], int(item["layer"])
        raw0, raw1 = item["raw0"], item["raw1"]
        pred: dict[str, torch.Tensor] = {}
        if prev is not None:
            z = np.load(pred_dir / f"{b}_{li:04d}.npz")
            pred = {m: torch.from_numpy(z[m].astype(np.int64)) for m in models}
            on_part = torch.from_numpy(z["part"]).long()
        else:
            d = torch.from_numpy(item["dscnn"].astype(np.int64)).to(device)
            pred["dscnn"] = torch.where(d < 0, torch.full_like(d, NONE_ID), d)
            if "v4" in old:
                pred["v4"] = lut[predict_pair(old["v4"], item["str0"], item["str1"], img_size, device, ("native",), tta=args.tta)["native"]]
            if "v5" in old:
                pred["v5"] = lut[predict_pair(old["v5"], raw0, raw1, img_size, device, ("native",), tta=args.tta)["native"]]
            pred["v6"] = predict_pair(m6, raw0, raw1, img_size, device, ("native",), tta=args.tta)["native"]
            on_part = torch.from_numpy(item["part"]).to(device).long()

        for k, m in enumerate(models):
            counts[i, k] = torch.bincount(pred[m].reshape(-1), minlength=N_BINS).cpu().numpy()
            cross[i, k] = _bincount2(pred["v6"], pred[m], cfg.N_CLASSES_V6, N_BINS)
            part[i, k] = _bincount2(on_part, pred[m], 2, N_BINS)
            rs_rows[i, k] = rs_row_stats(pred[m])
        if db_hits is None and db_hits_ok:
            db_hits = torch.zeros((len(models), *pred["v6"].shape), dtype=torch.int16, device=device)
        if db_hits is not None:
            if tuple(db_hits.shape[1:]) == tuple(pred["v6"].shape):
                for k, m in enumerate(models):
                    db_hits[k] += (pred[m] == DB_ID)
            else:                                   # 빌드마다 사진 크기가 다르면 화소 위치를 견줄 수 없다
                db_hits, db_hits_ok = None, False
        photo_mean[i] = (raw0.mean(), raw1.mean())

        tiles = [tile_gray(display_u8(raw0), device), tile_gray(display_u8(raw1), device)]
        tiles += [tile_label(pred[m], palette) for m in models]
        v6_defect = counts[i, models.index("v6"), 2: cfg.N_CLASSES_V6].sum() / counts[i, models.index("v6")].sum()
        row = {"build": b, "layer": li, "label": f"{b} · layer {li}", "tiles": tiles, "v6_defect": float(v6_defect)}
        page_rows.append(row)
        build_rows.append(row)
        last_of_build = (i + 1 == n_layers) or (items[i + 1][0] != b)
        if len(page_rows) == args.rows_per_page or last_of_build:
            flush_page()
        if last_of_build:
            # 개요 그림: 그 빌드에서 뽑힌 레이어 가운데 v6 의 결함 화소 비율이 중앙값인 레이어 (골라 낸 예가 아니라 보통의 레이어)
            overview_rows.append(sorted(build_rows, key=lambda r: r["v6_defect"])[len(build_rows) // 2])
            build_rows = []
        if save_preds:
            np.savez_compressed(
                pred_dir / f"{b}_{li:04d}.npz", part=item["part"],
                **{m: pred[m].cpu().numpy().astype(np.int8) for m in models},
            )
        el = time.time() - t_start
        print(f"[{i + 1:3d}/{n_layers}] {b} layer {li:4d}  v6 결함 화소 {100 * v6_defect:5.2f}%  "
              f"({el:.0f}s, 남은 시간 {el / (i + 1) * (n_layers - i - 1):.0f}s)", flush=True)
    flush_page()
    render_page(
        overview_rows, columns,
        f"ORNL {' / '.join(builds)} — one layer per build: the sampled layer with the median v6 defect area "
        f"(v6 = {cfg.run_tag(args.fold)} {args.which}{', TTA' if args.tta else ''})",
        out_dir / "overview.png",
    )

    # ----- 요약 -----
    build_of = np.array([b for b, _ in items])
    total_px = counts[:, 0].sum()
    summary = {
        "kind": "ornl_compare",
        "run_name": run_name, "fold": args.fold, "which": args.which, "tta": args.tta, "postprocess": False,
        "img_size": img_size, "seed": seed, "rule": rule,
        "n_layers": n_layers, "builds": {b: {k: v for k, v in info.items() if k != "skipped"} for b, info in builds.items()},
        "skipped": {b: info.get("skipped", []) for b, info in builds.items()},
        "overview_layers": {r["build"]: r["layer"] for r in overview_rows},
        "models": models, "checkpoints": checkpoints,
        "ckpt_step": c6.get("step"), "ckpt_weights": c6.get("weights"),
        "class_names": list(cfg.CLASS_NAMES_V6), "class_abbrs": list(cfg.CLASS_ABBRS_V6), "none_id": NONE_ID,
        "input_brightness": {"v6": "raw", "v5": "raw", "v4": "per-layer 1–99 percentile stretch", "dscnn": "n/a (stored prediction)"},
        "share": {m: [float(v) / float(total_px) for v in counts[:, k].sum(axis=0)] for k, m in enumerate(models)},
        "share_by_build": {
            b: {m: [float(v) / float(counts[build_of == b, k].sum()) for v in counts[build_of == b, k].sum(axis=0)]
                for k, m in enumerate(models)}
            for b in builds
        },
        "agreement": {m: agreement(cross[:, k].sum(axis=0)) for k, m in enumerate(models) if m != "v6"},
        "part": {m: part_overlap(part[:, k].sum(axis=0)) for k, m in enumerate(models)},
        "part_iou_by_build": {
            b: {m: part_overlap(part[build_of == b, k].sum(axis=0))["part_iou"] for k, m in enumerate(models)} for b in builds
        },
        "defect_share_per_layer": {      # 레이어마다 결함 (IS–DB) 으로 낸 화소 비율
            m: {"median": float(np.median(ds)), "mean": float(ds.mean()), "max": float(ds.max())}
            for k, m in enumerate(models)
            for ds in [counts[:, k, 2: cfg.N_CLASSES_V6].sum(axis=1) / counts[:, k].sum(axis=1)]
        },
        "layers_with_class": {           # 그 클래스를 LAYER_MIN_PX 넘게 낸 레이어 수
            "min_px": LAYER_MIN_PX,
            "counts": {m: [int((counts[:, k, c] > LAYER_MIN_PX).sum()) for c in range(N_BINS)] for k, m in enumerate(models)},
        },
        "rs_rows": {
            "min_px_per_row": RS_ROW_MIN,
            "all": {m: rs_summary(rs_rows[:, k]) for k, m in enumerate(models)},
            "by_build": {b: {m: rs_summary(rs_rows[build_of == b, k]) for k, m in enumerate(models)} for b in builds},
        },
        "fixed_debris": (
            {m: fixed_debris(db_hits[k].cpu().numpy(), n_layers) for k, m in enumerate(models)} if db_hits is not None else None
        ),
        "from_preds": prev is not None,     # True = 모델을 다시 돌리지 않고 저장해 둔 예측으로 그림 · 수치만 다시 만든 것
        "note": "model-vs-model comparison on builds without human ground truth; not accuracy, not used for any decision",
    }
    (out_dir / "summary.json").write_text(json.dumps(summary, ensure_ascii=False, indent=1), encoding="utf-8")
    np.savez_compressed(
        out_dir / "stats.npz", builds=build_of, layers=np.array([li for _, li in items]), models=np.array(models),
        counts=counts, cross_v6=cross, part=part, rs_rows=rs_rows, photo_mean=photo_mean,
        **({"debris_hits": db_hits.cpu().numpy()} if db_hits is not None else {}),
    )
    with (out_dir / "layer_stats.csv").open("w", newline="") as fh:
        w = csv.writer(fh)
        w.writerow(["build", "layer", "model"] + list(cfg.CLASS_ABBRS_V6) + ["none", "cad_part_px"])
        for i, (b, li) in enumerate(items):
            for k, m in enumerate(models):
                w.writerow([b, li, m] + [int(v) for v in counts[i, k]] + [int(part[i, k, 1].sum())])
    write_summary_md(out_dir / "summary.md", summary)
    plot_summary(out_dir / "summary.png", summary)
    print(f"[compare] done. {n_layers} 레이어, {time.time() - t_start:.0f}s → {out_dir}")


if __name__ == "__main__":
    main()
