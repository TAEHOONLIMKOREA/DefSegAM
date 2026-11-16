"""외형 기준 결함 분류 체계(docs/DEFECT_TAXONOMY.md §3) 의 클래스별 예시 사진 figure.

입력은 docs/Defect Example.pptx (클래스별 슬라이드, 사진 30장 + 장비·빌드·레이어 메모).
슬라이드에 그려진 도형(타원·화살표·자유곡선)은 사용하지 않고, 결함 위치를 이 스크립트가
직접 표시한다:
  * 붉은 표시  — 그 figure 가 다루는 클래스의 결함 (둥근 상자 / 회전 상자 / 점선 경계)
  * 흰 표시+태그 — 같은 사진에 함께 나타난 다른 클래스 (약어 태그: IS, PC, RS, RH, SE, SP, OM, DB)
  * 확대 인셋  — 작은 결함 (왼쪽 아래에 배율)

사진은 자르기(crop) 와, 사진 전체에 균일하게 적용하는 레벨·감마 조정만 한다
(어두운 사진이 많아 인쇄에서 결함이 보이도록; 증폭은 MAX_GAIN 으로 제한; --raw 로 끔).

figure 는 클래스별 8장, 각각 2열 × 2행 (패널 폭 = figure 폭의 절반):
  IS_RH   Incomplete Spreading ×3 + Recoater Hopping ×1
  PC      Powder Collapse ×4
  RS1/RS2 Recoater Streaking ×8
  SE1     Super Elevation ×4
  SE2_SP  Super Elevation ×2 + Super Elevation/Spatter ×1 + Spatter ×1
  OM      Over Melting ×3 + 확대 1
  DB      Debris ×3 + 확대 1

사용 (CWD = 3DP_VPPM/):
    DefSeg_AM/venv/bin/python -m DefSeg_AM.scripts.figure_defect_examples
    DefSeg_AM/venv/bin/python -m DefSeg_AM.scripts.figure_defect_examples --only SE1
    DefSeg_AM/venv/bin/python -m DefSeg_AM.scripts.figure_defect_examples --raw      # 밝기 조정 없이
    DefSeg_AM/venv/bin/python -m DefSeg_AM.scripts.figure_defect_examples --dump     # 슬라이드 구성 출력

출력: paper/DefectSegAM_Ko/Figs/Defect_Examples_<name>.{pdf,png}
"""
from __future__ import annotations

import argparse
import io
import math
import re
import zipfile
from pathlib import Path
from xml.dom import minidom

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib import patches
from matplotlib import patheffects as pe
from matplotlib.transforms import Affine2D
from PIL import Image

HERE = Path(__file__).resolve().parent          # DefSeg_AM/scripts
REPO = HERE.parent                              # DefSeg_AM
PPTX = REPO / "docs" / "Defect Example.pptx"
OUT_DIR = REPO / "paper" / "DefectSegAM_Ko" / "Figs"

EMU = 914400  # EMU per inch

# ----------------------------------------------------------------------------
# 스타일
# ----------------------------------------------------------------------------
ACCENT = "#FF3B1F"            # figure 의 주 클래스 표시
SECOND = "#FFFFFF"            # 함께 나타난 다른 클래스 표시
DARK = "#1A1A1A"
LW = 1.8
TITLE_FS = 11.0               # 패널 라벨 1행 (클래스명)
META_FS = 8.5                 # 패널 라벨 2행 (장비·빌드·레이어)
TAG_FS = 8.6                  # preprint 본문 폭(390 pt)으로 축소돼도 7 pt 이상

# 밝기 조정 (사진 전체에 균일)
TONE_TARGET = 0.36            # 분말 베드 중앙값이 놓일 밝기 (0-1)
MAX_GAIN = 2.6                # 선형 증폭 한도
GAMMA_MIN = 0.5

CLASS_NAMES = {
    "IS": "Incomplete Spreading", "PC": "Powder Collapse", "RS": "Recoater Streaking",
    "RH": "Recoater Hopping", "SE": "Super Elevation", "SP": "Spatter",
    "OM": "Over Melting", "DB": "Debris",
}

# ----------------------------------------------------------------------------
# 표시(mark) 만들기 — 좌표는 모두 사진 전체 기준 (0-1)
#   cls   : 클래스 약어. figure 의 primary 에 있으면 붉은색, 아니면 흰색
#   tag   : True 면 약어 태그를 붙임 (tag_at=(x, y) 로 위치 지정 가능)
#   inset : dict(anchor=(x, y), w=폭, src=(x0,y0,x1,y1)) — 패널 기준 위치·폭에 확대 인셋
# ----------------------------------------------------------------------------
def box(cls, x0, y0, x1, y1, tag=False, **kw):
    return dict(kind="box", cls=cls, box=(x0, y0, x1, y1), tag=tag, **kw)


def rbox(cls, cx, cy, length, thick, angle, tag=False, **kw):
    """회전 상자. length/thick 는 사진 폭 대비, angle 은 화면에서 시계 방향 각도(도)."""
    return dict(kind="rbox", cls=cls, c=(cx, cy), length=length, thick=thick, angle=angle, tag=tag, **kw)


def curve(cls, pts, tag=False, **kw):
    """점선 안내선 (도포 경계 등). pts 는 경계 위의 점 [(x, y), ...].
    offset(패널 폭 대비, 부호 = 진행 방향의 왼쪽/오른쪽) 만큼 경계 옆으로 비켜 그린다."""
    return dict(kind="curve", cls=cls, pts=pts, tag=tag, **kw)


def panel(media, cls, meta, crop=(0, 0, 1, 1), marks=(), title=None):
    return dict(media=media, cls=cls, meta=meta, crop=crop, marks=list(marks), title=title)


# 사진별 표시 (다른 패널에서 확대해 다시 쓰는 것은 이름을 붙여 둠)
M_IMG26 = [box("OM", 0.615, 0.04, 0.955, 0.395)]
M_IMG27 = [box("DB", 0.85, 0.355, 0.955, 0.425),
           box("SE", 0.745, 0.262, 0.965, 0.348, tag=True, tag_at=(0.745, 0.250))]

FIGURES = {
    # ------------------------------------------------------------------ 리코팅: 도포 부족 / 호핑
    # aspect 1.06: 표시 규약을 적는 첫 그림이라 캡션이 길다 — 본문 높이 548.5 pt 안에 들도록 조금 낮춤
    "IS_RH": dict(primary=("IS", "RH"), aspect=1.06, panels=[
        panel("image1.png", "IS", "M280_20260911_0143 · layer 10", crop=(0.0, 0.01, 0.915, 0.99), marks=[
            curve("IS", [(0.00, 0.11), (0.04, 0.21), (0.09, 0.29), (0.15, 0.30), (0.19, 0.34), (0.24, 0.46),
                         (0.29, 0.58), (0.345, 0.64), (0.40, 0.70), (0.44, 0.64), (0.48, 0.645), (0.52, 0.655),
                         (0.56, 0.69), (0.595, 0.75), (0.60, 0.83), (0.635, 0.875), (0.68, 0.745),
                         (0.725, 0.61), (0.775, 0.40), (0.825, 0.19), (0.865, 0.05)],
                  tag=True, tag_at=(0.33, 0.46), offset=-0.022),
        ]),
        panel("image3.jpeg", "IS", "M160_20260108_1843 · layer 176", crop=(0.0, 0.0, 1.0, 0.986), marks=[
            curve("IS", [(0.01, 0.10), (0.03, 0.25), (0.07, 0.38), (0.12, 0.48), (0.18, 0.57), (0.25, 0.64),
                         (0.32, 0.69), (0.38, 0.72)], tag=True, tag_at=(0.17, 0.40), offset=-0.022),
            curve("IS", [(0.915, 0.30), (0.91, 0.45), (0.905, 0.60), (0.895, 0.75), (0.885, 0.85),
                         (0.86, 0.92), (0.82, 0.965), (0.77, 0.985)], offset=0.022),
        ]),
        panel("image2.png", "IS", "M160_20260108_1657 · layer 107", crop=(0.0, 0.0, 1.0, 0.978), marks=[
            curve("IS", [(0.04, 0.02), (0.07, 0.12), (0.10, 0.22), (0.125, 0.32), (0.15, 0.42), (0.175, 0.50),
                         (0.20, 0.58), (0.235, 0.67), (0.27, 0.76), (0.30, 0.84), (0.33, 0.93), (0.345, 0.975)],
                  tag=True, tag_at=(0.22, 0.34), offset=-0.022),
            box("OM", 0.02, 0.66, 0.215, 0.905, tag=True),
            box("OM", 0.62, 0.69, 0.815, 0.925, tag=True),
        ]),
        panel("image15.png", "RH", "M280_20260826 · layer 76", crop=(0.05, 0.0, 0.95, 1.0), marks=[
            box("RH", 0.07, 0.295, 0.93, 0.505),
        ]),
    ]),
    # ------------------------------------------------------------------ 분말 무너짐
    "PC": dict(primary=("PC",), aspect=1.12, panels=[
        panel("image4.png", "PC", "M160_20260521_1137 · layer 1323", crop=(0.0, 0.02, 1.0, 0.98), marks=[
            box("PC", 0.36, 0.82, 0.58, 0.975, inset=dict(anchor=(0.03, 0.03), w=0.42)),
        ]),
        panel("image5.png", "PC", "M160_20260521_1137 · layer 1455", crop=(0.07, 0.0, 1.0, 1.0), marks=[
            box("PC", 0.795, 0.045, 0.905, 0.495),
        ]),
        panel("image6.png", "PC", "M160_20260521_1137 · layer 1531", marks=[
            box("PC", 0.785, 0.24, 0.89, 0.61),
            box("PC", 0.29, 0.41, 0.50, 0.61),
        ]),
        panel("image30.png", "PC", "M160_20260521_1137", crop=(0.22, 0.0, 0.96, 1.0), marks=[
            box("PC", 0.41, 0.29, 0.63, 0.69),
            box("PC", 0.80, 0.24, 0.89, 0.57),
        ]),
    ]),
    # ------------------------------------------------------------------ 리코터 줄무늬
    "RS1": dict(primary=("RS",), aspect=1.05, panels=[
        panel("image7.png", "RS", "M160_20260810_1717 · layer 166", crop=(0.29, 0.145, 0.89, 0.912), marks=[
            box("RS", 0.485, 0.155, 0.545, 0.895),
            box("RS", 0.69, 0.155, 0.75, 0.895),
            box("RS", 0.365, 0.50, 0.405, 0.895),
        ]),
        panel("image8.png", "RS", "M160_20260804_1039 · layers 148–152", crop=(0.0, 0.02, 1.0, 0.965), marks=[
            # 줄무늬가 매우 희미함 — 분말 픽셀의 열 방향 평균 밝기가 주변보다 어두운 두 위치
            # (x ≈ 0.183, 0.274–0.283; 서로 다른 세 높이 구간에서 2–4σ) 만 표시
            box("RS", 0.16, 0.30, 0.205, 0.93),
            box("RS", 0.252, 0.05, 0.30, 0.93),
        ]),
        panel("image11.jpeg", "RS", "M160_20260806_1054 · layer 83", marks=[
            box("RS", 0.655, 0.02, 0.725, 0.64),
        ]),
        panel("image12.jpeg", "RS", "M160_20260806_1703 · layer 88", crop=(0.0, 0.02, 1.0, 0.975), marks=[
            rbox("RS", 0.405, 0.50, 0.94, 0.15, 83.5),
        ]),
    ]),
    "RS2": dict(primary=("RS",), aspect=1.0, panels=[
        panel("image9.png", "RS", "M160_20260427_1701 · layer 182", crop=(0.01, 0.0, 0.99, 1.0), marks=[
            box("RS", 0.24, 0.10, 0.345, 0.985),
        ]),
        panel("image10.png", "RS", "M160_20260420_1048 · layer 172+", marks=[
            box("RS", 0.575, 0.015, 0.635, 0.985),
            box("RS", 0.165, 0.33, 0.215, 0.80),
        ]),
        panel("image13.png", "RS", "M160_20250508 · layer 335", marks=[
            box("RS", 0.195, 0.03, 0.255, 0.97),
            box("RS", 0.52, 0.03, 0.57, 0.97),
            box("RS", 0.695, 0.03, 0.755, 0.97),
        ]),
        panel("image14.png", "RS", "M160_20241203 · layer 282", crop=(0.0, 0.04, 1.0, 0.95), marks=[
            box("RS", 0.075, 0.045, 0.165, 0.945),
            box("RS", 0.19, 0.045, 0.33, 0.945),
            box("RS", 0.43, 0.40, 0.50, 0.945),
        ]),
    ]),
    # ------------------------------------------------------------------ 파트 돌출
    "SE1": dict(primary=("SE",), aspect=1.0, panels=[
        panel("image17.jpeg", "SE", "M160_20260108_1843 · layer 158", crop=(0.0, 0.005, 1.0, 0.99), marks=[
            rbox("SE", 0.6625, 0.335, 0.56, 0.085, 68.9),
            box("OM", 0.745, 0.59, 0.835, 0.705, tag=True, tag_at=(0.845, 0.64)),
        ]),
        panel("image18.png", "SE", "M160_20260810 · layer 158", marks=[
            rbox("SE", 0.07, 0.238, 0.10, 0.045, -38),
            rbox("SE", 0.46, 0.255, 0.105, 0.045, -38),
            rbox("SE", 0.54, 0.178, 0.10, 0.045, -38),
            rbox("SE", 0.372, 0.338, 0.105, 0.045, -38),
            rbox("SE", 0.455, 0.63, 0.115, 0.075, -38, inset=dict(anchor=(0.60, 0.60), w=0.37,
                                                                  src=(0.39, 0.565, 0.52, 0.695))),
            rbox("SE", 0.378, 0.755, 0.105, 0.05, -38),
            rbox("SE", 0.288, 0.82, 0.115, 0.06, -38),
            box("RS", 0.16, 0.015, 0.215, 0.30, tag=True, tag_at=(0.225, 0.07)),
            box("RS", 0.35, 0.015, 0.40, 0.205),
        ]),
        panel("image19.png", "SE", "M160_20251127 · layer 81", crop=(0.02, 0.0, 0.98, 1.0), marks=[
            box("SE", 0.83, 0.285, 0.975, 0.40, inset=dict(anchor=(0.60, 0.60), w=0.37)),
            curve("IS", [(0.03, 0.205), (0.10, 0.225), (0.17, 0.25), (0.25, 0.295), (0.32, 0.355), (0.37, 0.43),
                         (0.41, 0.50), (0.44, 0.57), (0.47, 0.655), (0.50, 0.75), (0.53, 0.87), (0.555, 0.985)],
                  tag=True, tag_at=(0.30, 0.25), offset=-0.022),
        ]),
        panel("image20.png", "SE", "M160_20250704 · layer 627", crop=(0.03, 0.0, 0.97, 1.0), marks=[
            rbox("SE", 0.41, 0.20, 0.80, 0.085, -27.5),
            box("RS", 0.705, 0.33, 0.82, 0.985, tag=True, tag_at=(0.83, 0.45)),
            box("RS", 0.085, 0.44, 0.165, 0.985),
        ]),
    ]),
    "SE2_SP": dict(primary=("SE", "SP"), aspect=1.05, panels=[
        panel("image21.png", "SE", "M280_20260810_0827 · layers 450–500", crop=(0.035, 0.03, 0.965, 0.975), marks=[
            *[rbox("SE", cx, cy, 0.115, 0.08, -31) for cx, cy in
              ((0.824, 0.548), (0.71, 0.643), (0.509, 0.751), (0.386, 0.828), (0.263, 0.923),
               (0.539, 0.295), (0.421, 0.377), (0.298, 0.46))],
            rbox("SE", 0.925, 0.468, 0.115, 0.08, -31, inset=dict(anchor=(0.645, 0.02), w=0.335,
                                                                   src=(0.85, 0.375, 0.99, 0.53))),
        ]),
        panel("image22.png", "SE", "M160_20260409_1619 · layer 165+", crop=(0.0, 0.03, 1.0, 0.98), marks=[
            rbox("SE", 0.538, 0.86, 0.37, 0.09, -23),
            rbox("SE", 0.877, 0.815, 0.27, 0.09, -23),
            box("SE", 0.925, 0.452, 0.998, 0.522),
        ]),
        panel("image16.jpeg", "SE / Spatter", "M280_20260810_0827 · layer 535", crop=(0.03, 0.0, 0.975, 1.0),
              marks=[
            rbox("SE", 0.415, 0.365, 0.335, 0.15, -33.5, tag=True, tag_at=(0.54, 0.195)),
            rbox("SE", 0.82, 0.53, 0.335, 0.15, -33.5),
            rbox("SE", 0.385, 0.845, 0.33, 0.15, -33.5),
            box("SP", 0.035, 0.425, 0.205, 0.535, tag=True),
            box("SP", 0.04, 0.875, 0.195, 0.975, tag=True),
            box("SP", 0.455, 0.59, 0.625, 0.69, tag=True),
        ]),
        panel("image23.png", "SP", "M280_20260903_0825 · layer 225", crop=(0.0, 0.02, 1.0, 0.975), marks=[
            box("SP", 0.575, 0.155, 0.71, 0.25),
            box("SP", 0.755, 0.175, 0.89, 0.27),
            box("SP", 0.755, 0.36, 0.89, 0.455),
        ]),
    ]),
    # ------------------------------------------------------------------ 과용융
    "OM": dict(primary=("OM",), aspect=1.1, panels=[
        panel("image24.png", "OM", "M280_20260313_1125 · layers 108–123", crop=(0.02, 0.02, 0.98, 0.98), marks=[
            box("OM", 0.195, 0.235, 0.385, 0.435),
            box("OM", 0.655, 0.615, 0.855, 0.835),
        ]),
        panel("image25.png", "OM", "M160_20251127 · layer 83", crop=(0.0, 0.0, 1.0, 0.95), marks=[
            rbox("OM", 0.42, 0.085, 0.26, 0.065, 27),
            box("SE", 0.73, 0.235, 0.975, 0.395, tag=True),
        ]),
        panel("image26.png", "OM", "M280_20260821 · layer 134", crop=(0.03, 0.0, 0.97, 1.0), marks=M_IMG26),
        panel("image26.png", "OM", "M280_20260821 · layer 134", crop=(0.60, 0.02, 0.98, 0.42), marks=[],
              title="Detail of (c)"),
    ]),
    # ------------------------------------------------------------------ 파편
    "DB": dict(primary=("DB",), aspect=1.1, panels=[
        panel("image27.png", "DB", "M160_20251127_1744 · layer 83", crop=(0.0, 0.03, 1.0, 0.97), marks=M_IMG27),
        panel("image28.jpeg", "DB", "M160_20260720_1217 · layer 86", marks=[
            box("DB", 0.15, 0.31, 0.33, 0.455),
            box("DB", 0.395, 0.17, 0.575, 0.275),
        ]),
        panel("image29.jpeg", "DB", "M160_20260720_1217 · layer 88", marks=[
            box("DB", 0.135, 0.565, 0.285, 0.70),
            box("DB", 0.485, 0.515, 0.59, 0.635),
        ]),
        panel("image27.png", "DB", "M160_20251127_1744 · layer 83", crop=(0.70, 0.22, 1.0, 0.50), marks=M_IMG27,
              title="Detail of (a)"),
    ]),
}


# ----------------------------------------------------------------------------
# pptx 읽기
# ----------------------------------------------------------------------------
IMG_EXT = {".png", ".jpg", ".jpeg", ".bmp", ".gif", ".tif", ".tiff"}


def read_media(path: Path) -> dict:
    z = zipfile.ZipFile(path)
    out = {}
    for n in z.namelist():
        if n.startswith("ppt/media/") and Path(n).suffix.lower() in IMG_EXT:
            out[Path(n).name] = Image.open(io.BytesIO(z.read(n))).convert("RGB")
    return out


def dump(path: Path):
    """슬라이드 순서대로 제목·사진·메모를 출력 (pptx 가 바뀌었을 때 FIGURES 갱신용)."""
    z = zipfile.ZipFile(path)
    pres = z.read("ppt/presentation.xml").decode("utf-8")
    rels = z.read("ppt/_rels/presentation.xml.rels").decode("utf-8")
    pmap = {re.search(r'Id="([^"]+)"', s).group(1): re.search(r'Target="([^"]+)"', s).group(1)
            for s in re.findall(r"<Relationship [^>]+>", rels)}
    for page, rid in enumerate(re.findall(r'<p:sldId [^>]*r:id="([^"]+)"', pres), 1):
        t = pmap[rid]
        idx = re.search(r"slide(\d+)\.xml", t).group(1)
        srel = z.read(f"ppt/slides/_rels/slide{idx}.xml.rels").decode("utf-8")
        rmap = {re.search(r'Id="([^"]+)"', s).group(1): re.search(r'Target="([^"]+)"', s).group(1)
                for s in re.findall(r"<Relationship [^>]+>", srel)}
        doc = minidom.parseString(z.read("ppt/" + t))
        print(f"== page {page}")
        for n in doc.getElementsByTagName("p:pic"):
            rid2 = n.getElementsByTagName("a:blip")[0].getAttribute("r:embed")
            print("   PIC ", Path(rmap[rid2]).name)
        for n in doc.getElementsByTagName("p:sp"):
            txt = "".join(t_.firstChild.nodeValue for t_ in n.getElementsByTagName("a:t") if t_.firstChild)
            if txt.strip():
                print("   TEXT", repr(txt.strip()))


# ----------------------------------------------------------------------------
# 밝기 조정 — 사진 전체에 같은 식 (레벨 + 감마)
# ----------------------------------------------------------------------------
def tone(arr: np.ndarray):
    a = arr.astype(np.float32)
    h, w = a.shape[:2]
    by, bx = int(h * 0.06), int(w * 0.06)                       # 화면 캡처 테두리 제외
    y = 0.299 * a[..., 0] + 0.587 * a[..., 1] + 0.114 * a[..., 2]
    yc = y[by:h - by, bx:w - bx]
    p1, hi, med = (float(v) for v in np.percentile(yc, [1.0, 99.9, 50.0]))
    lo = max(0.0, p1 - 0.6 * (med - p1))                         # 어두운 모서리를 덜 자르도록 여유
    hi = max(hi, lo + 255.0 / MAX_GAIN)                          # 증폭 한도
    m = min(max((med - lo) / (hi - lo), 1e-3), 0.999)
    g = min(max(math.log(TONE_TARGET) / math.log(m), GAMMA_MIN), 1.0)
    y_out = 255.0 * np.clip((y - lo) / (hi - lo), 0, 1) ** g
    # 휘도만 바꾸고 채널 비율(색상·채도)은 그대로 둔다 — 어두운 사진의 색 틀어짐 방지
    out = np.clip(a * (y_out / np.maximum(y, 1.0))[..., None], 0, 255)
    return (out + 0.5).astype(np.uint8), dict(lo=lo, hi=hi, gamma=g)


# ----------------------------------------------------------------------------
# 그리기
# ----------------------------------------------------------------------------
def fit_crop(crop, w, h, aspect):
    """요청 crop 을 가로세로비 aspect 에 맞게 (중심 유지, 안쪽으로) 조정 → 픽셀 박스."""
    x0, y0, x1, y1 = crop[0] * w, crop[1] * h, crop[2] * w, crop[3] * h
    cw, ch = x1 - x0, y1 - y0
    if cw / ch > aspect:
        nw = ch * aspect
        x0 += (cw - nw) / 2
        x1 = x0 + nw
    else:
        nh = cw / aspect
        y0 += (ch - nh) / 2
        y1 = y0 + nh
    return int(round(x0)), int(round(y0)), int(round(x1)), int(round(y1))


def chaikin(pts: np.ndarray, n=3) -> np.ndarray:
    """꺾은선을 부드럽게 (corner cutting)."""
    for _ in range(n):
        q = 0.75 * pts[:-1] + 0.25 * pts[1:]
        r = 0.25 * pts[:-1] + 0.75 * pts[1:]
        pts = np.vstack([pts[:1], np.column_stack([q, r]).reshape(-1, 2), pts[-1:]])
    return pts


def curve_path(pts, off: float = 0.0) -> np.ndarray:
    """안내선 좌표: 꺾은선을 부드럽게 한 뒤(chaikin) 법선 방향으로 off 만큼 비켜 놓는다.
    경계를 가리지 않도록 옆에 그리기 위한 것 (pptx 내보내기와 공용)."""
    pts = chaikin(np.asarray(pts, dtype=float))
    if off:
        t = np.gradient(pts, axis=0)
        nrm = np.column_stack([-t[:, 1], t[:, 0]])
        nrm /= np.maximum(np.linalg.norm(nrm, axis=1, keepdims=True), 1e-9)
        k = np.ones(21) / 21                            # 급한 굽이에서 선이 꺾이지 않게 법선을 평활
        pad = np.pad(nrm, ((10, 10), (0, 0)), mode="edge")
        nrm = np.column_stack([np.convolve(pad[:, i], k, mode="valid") for i in (0, 1)])
        pts = pts + off * nrm
    return pts


def style(cls, primary):
    """→ (선 색, 테두리(halo) 색, 태그 배경, 태그 글자색)"""
    if cls in primary:
        return ACCENT, "white", ACCENT, "white"
    return SECOND, DARK, "white", DARK


def draw_tag(ax, x, y, text, bg, fg, ha="left", va="bottom"):
    ax.text(x, y, text, ha=ha, va=va, fontsize=TAG_FS, fontweight="bold", color=fg, zorder=9,
            bbox=dict(boxstyle="round,pad=0.22,rounding_size=0.6", fc=bg, ec=DARK if bg == "white" else "none",
                      lw=0.6))


def draw_panel(ax, img: Image.Image, spec: dict, primary, aspect: float):
    w, h = img.size
    x0, y0, x1, y1 = fit_crop(spec["crop"], w, h, aspect)
    full = np.asarray(img)
    pw, ph = x1 - x0, y1 - y0
    ax.imshow(full[y0:y1, x0:x1], interpolation="lanczos", zorder=1)
    ax.set_xlim(0, pw)
    ax.set_ylim(ph, 0)
    ax.set_xticks([])
    ax.set_yticks([])
    for s in ax.spines.values():
        s.set_linewidth(0.6)
        s.set_color("#333333")
    r = 0.014 * pw                                              # 모서리 반경

    def P(x, y):                                                # 사진 전체 기준 → 패널 픽셀
        return x * w - x0, y * h - y0

    for mk in spec["marks"]:
        col, halo, tbg, tfg = style(mk["cls"], primary)
        tag_xy = None
        if mk["kind"] == "box":
            bx0, by0, bx1, by1 = mk["box"]
            (px0, py0), (px1, py1) = P(bx0, by0), P(bx1, by1)
            for lw, c, z in ((LW + 1.8, halo, 5), (LW, col, 6)):
                ax.add_patch(patches.FancyBboxPatch((px0, py0), px1 - px0, py1 - py0,
                                                    boxstyle=f"round,pad=0,rounding_size={r}",
                                                    fill=False, edgecolor=c, linewidth=lw, zorder=z))
            tag_xy = (px0 + 0.01 * pw, py0 - 0.012 * ph)
        elif mk["kind"] == "rbox":
            cx, cy = P(*mk["c"])
            L, T = mk["length"] * w, mk["thick"] * w
            tf = Affine2D().rotate_deg(mk["angle"]).translate(cx, cy) + ax.transData
            for lw, c, z in ((LW + 1.8, halo, 5), (LW, col, 6)):
                ax.add_patch(patches.FancyBboxPatch((-L / 2, -T / 2), L, T,
                                                    boxstyle=f"round,pad=0,rounding_size={min(r, T / 2)}",
                                                    fill=False, edgecolor=c, linewidth=lw, zorder=z,
                                                    transform=tf))
            tag_xy = (cx, cy - T / 2 - 0.02 * ph)
        elif mk["kind"] == "curve":
            pts = curve_path([P(x, y) for x, y in mk["pts"]], mk.get("offset", 0.0) * pw)
            ax.plot(pts[:, 0], pts[:, 1], color=halo, lw=LW + 1.2, alpha=0.6, solid_capstyle="round", zorder=5)
            ax.plot(pts[:, 0], pts[:, 1], color=col, lw=LW - 0.2, dashes=(4, 3), dash_capstyle="round", zorder=6)
            tag_xy = tuple(pts[len(pts) // 2])

        if mk.get("tag"):
            if mk.get("tag_at"):
                tag_xy = P(*mk["tag_at"])
            draw_tag(ax, tag_xy[0], tag_xy[1], mk["cls"], tbg, tfg)

        ins = mk.get("inset")
        if ins:
            if ins.get("src"):
                sx0, sy0, sx1, sy1 = ins["src"]
            elif mk["kind"] == "box":
                sx0, sy0, sx1, sy1 = mk["box"]
            else:
                raise ValueError("inset on a non-box mark needs src")
            sub = full[int(sy0 * h):int(sy1 * h), int(sx0 * w):int(sx1 * w)]
            sh, sw = sub.shape[:2]
            iw = ins["w"]
            ih = iw * (sh / sw) * (pw / ph)
            ax_in = ax.inset_axes([ins["anchor"][0], 1 - ins["anchor"][1] - ih, iw, ih], zorder=12)
            ax_in.imshow(sub, interpolation="lanczos")
            ax_in.set_xticks([])
            ax_in.set_yticks([])
            for s in ax_in.spines.values():
                s.set_linewidth(LW)
                s.set_color(col)
                s.set_path_effects([pe.Stroke(linewidth=LW + 1.8, foreground=halo), pe.Normal()])
            ax_in.text(0.04, 0.05, f"×{(iw * pw) / sw:.1f}", transform=ax_in.transAxes, ha="left", va="bottom",
                       fontsize=TAG_FS, fontweight="bold", color="white",
                       path_effects=[pe.Stroke(linewidth=2.0, foreground="black"), pe.Normal()])


def build_figure(name: str, fig_spec: dict, media: dict, out_dir: Path, width_in: float):
    panels, aspect, primary = fig_spec["panels"], fig_spec["aspect"], fig_spec["primary"]
    ncol = 2
    nrow = (len(panels) + ncol - 1) // ncol
    gap_x = 0.12 / width_in
    label_in, row_gap_in, top_in0 = 0.40, 0.06, 0.02
    pw = (1 - gap_x * (ncol - 1)) / ncol
    ph_in = pw * width_in / aspect
    row_in = ph_in + label_in + row_gap_in
    height_in = top_in0 + nrow * row_in - row_gap_in + 0.02
    fig = plt.figure(figsize=(width_in, height_in), facecolor="white")

    for k, spec in enumerate(panels):
        row, col = divmod(k, ncol)
        left = col * (pw + gap_x)
        bottom = 1 - (top_in0 + row * row_in + ph_in) / height_in
        ax = fig.add_axes([left, bottom, pw, ph_in / height_in])
        draw_panel(ax, media[spec["media"]], spec, primary, aspect)
        title = spec["title"] or " / ".join(CLASS_NAMES.get(c.strip(), c.strip()) for c in spec["cls"].split("/"))
        cx = left + pw / 2
        fig.text(cx, bottom - 0.05 / height_in, f"({chr(97 + k)}) {title}", ha="center", va="top",
                 fontsize=TITLE_FS)
        fig.text(cx, bottom - 0.225 / height_in, spec["meta"], ha="center", va="top",
                 fontsize=META_FS, color="#444444")

    out_dir.mkdir(parents=True, exist_ok=True)
    pdf = out_dir / f"Defect_Examples_{name}.pdf"
    png = out_dir / f"Defect_Examples_{name}.png"
    fig.savefig(pdf, dpi=300, facecolor="white")
    fig.savefig(png, dpi=300, facecolor="white")
    plt.close(fig)
    return pdf, (width_in, height_in)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pptx", type=Path, default=PPTX)
    ap.add_argument("--out-dir", type=Path, default=OUT_DIR)
    ap.add_argument("--width", type=float, default=6.5, help="figure 폭 (인치)")
    ap.add_argument("--only", choices=sorted(FIGURES), help="하나의 figure 만 생성")
    ap.add_argument("--raw", action="store_true", help="밝기 조정을 하지 않음")
    ap.add_argument("--dump", action="store_true", help="슬라이드 구성만 출력하고 종료")
    args = ap.parse_args()

    if args.dump:
        dump(args.pptx)
        return
    media = read_media(args.pptx)
    if not args.raw:
        for k in list(media):
            arr, p = tone(np.asarray(media[k]))
            media[k] = Image.fromarray(arr)
    used = set()
    for name, spec in FIGURES.items():
        for p in spec["panels"]:
            if p["media"] not in media:
                raise SystemExit(f"FATAL: {p['media']} not in {args.pptx}")
            used.add(p["media"])
        if args.only and name != args.only:
            continue
        pdf, (wi, hi) = build_figure(name, spec, media, args.out_dir, args.width)
        print(f"wrote {pdf}  ({wi:.2f} x {hi:.2f} in, {len(spec['panels'])} panels)")
    print("unused media:", sorted(set(media) - used) or "none")


if __name__ == "__main__":
    main()
