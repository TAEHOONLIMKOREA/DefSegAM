"""논문 figure (figure_defect_examples.py) 와 같은 내용을 편집 가능한 PowerPoint 덱으로 내보낸다.

* 사진·자르기·밝기 조정·표시 위치는 모두 figure_defect_examples.FIGURES 를 그대로 쓴다
  (FIGURES 를 고치면 논문 figure 와 이 덱이 함께 바뀐다).
* 표시(상자·기울어진 상자·점선 안내선·약어 태그)는 그림이 아니라 PowerPoint 도형이라
  이동·크기·회전을 PowerPoint 에서 바로 고칠 수 있다. 사진은 원본 전체를 넣고 자르기(crop)만
  걸어 두었으므로 자르기 범위도 PowerPoint 에서 조정할 수 있다.
* 패널을 크게 보이도록 논문 figure 한 장(4패널)을 슬라이드 두 장(2패널씩)으로 나눈다.
* 테마·레이아웃은 원자료 덱(docs/Defect Example.pptx)의 것을 그대로 쓴다. 외부 패키지
  없이(zipfile) OOXML 을 직접 쓴다.

사용 (CWD = 3DP_VPPM/):
    DefSeg_AM/venv/bin/python -m DefSeg_AM.scripts.export_defect_examples_pptx
    DefSeg_AM/venv/bin/python -m DefSeg_AM.scripts.export_defect_examples_pptx --raw   # 밝기 조정 없이

출력: docs/Defect_Examples_Annotated.pptx
"""
from __future__ import annotations

import argparse
import io
import math
import re
import zipfile
from pathlib import Path
from xml.sax.saxutils import escape, quoteattr

import numpy as np
from PIL import Image

from DefSeg_AM.scripts.figure_defect_examples import (
    ACCENT, CLASS_NAMES, FIGURES, MAX_GAIN, GAMMA_MIN, PPTX, REPO,
    curve_path, fit_crop, read_media, style, tone,
)

OUT = REPO / "docs" / "Defect_Examples_Annotated.pptx"

EMU = 914400
SLIDE_W, SLIDE_H = 13.333, 7.5            # 원자료 덱과 같은 16:9 (12192000 × 6858000 EMU)

# 레이아웃 (원자료 덱의 slideLayoutN.xml)
LAYOUT_TITLE, LAYOUT_TITLE_ONLY = 1, 6

# 색
RED = ACCENT.lstrip("#").upper()          # FF3B1F
INK, MUTED, DARK_BG, SWATCH = "1F2328", "6B6F76", "1F2328", "5A5B5F"

# 내용 슬라이드 배치 (인치)
PY, PH, GAP = 1.15, 4.70, 0.60
LINE_PT, CURVE_PT, GLOW_PT = 2.25, 2.0, 2.2

# 패널 설명 (논문 3.2절 본문과 같은 내용)
DESC = {
    "IS_RH": ["베드 위쪽에서 불규칙하게 내려오는 도포 경계(점선)",
              "왼쪽과 오른쪽의 호 모양 도포 경계",
              "왼쪽의 사선 도포 경계. 흰 상자(OM): 표면이 거뭇하게 변색된 파트",
              "시편 배열의 둘째·셋째 행을 가로지르는 반복 가로 띠"],
    "PC": ["베드 아래쪽 가장자리의 분말 함몰 (확대 인셋)",
           "오른쪽 가장자리의 길쭉한 함몰 두 곳",
           "오른쪽 가장자리의 함몰과 베드 중앙의 원형 함몰",
           "중앙 함몰이 더 커진 모습. 파트 이탈 뒤 분말만 조사된 흔적의 클래스는 미정"],
    "RS1": ["세로 줄무늬 세 줄",
            "매우 희미한 줄무늬. 열 방향 평균 밝기가 주변보다 낮은 두 위치만 표시",
            "분말 위로 드러난 파트를 지나는 굵은 세로 줄무늬",
            "넓은 띠로 발달한 줄무늬"],
    "RS2": ["희미하고 가는 세로 줄무늬 묶음",
            "뚜렷한 세로 줄무늬 한 줄과 희미한 한 줄",
            "시편을 지나는 세로 줄무늬 세 줄",
            "베드 왼쪽에 조밀하게 반복되는 세로 줄무늬"],
    "SE1": ["분말 위로 드러난 긴 막대형 파트. 흰 상자(OM): 막대 끝의 밝은 부분",
            "분말 위로 드러난 파트 가장자리 일곱 곳. 흰 상자(RS): 세로 줄무늬",
            "분말 위로 드러난 파트 끝. 흰 점선(IS): 도포 경계",
            "긴 파트의 가장자리가 솟아 어두운 선으로 보임. 흰 상자(RS): 세로 줄무늬"],
    "SE2_SP": ["시편 아홉 개의 앞쪽 가장자리가 분말 위로 드러남",
               "빗살 모양 파트의 살 끝이 솟아 변색",
               "SE: 분말 위로 드러난 시편 세 줄. SP: 시편 왼쪽 분말 영역의 검은 입자",
               "오른쪽 위 시편 세 개의 왼쪽 분말 영역이 어둡게 보임"],
    "OM": ["고리 모양 파트의 안쪽 표면 두 곳이 거뭇하게 변색",
           "막대형 파트 표면이 검게 변색. 흰 상자(SE): 분말 위로 드러난 파트",
           "5×5 시편 배열의 오른쪽 위 네 시편이 거뭇하게 변색",
           "(c)의 붉은 상자를 확대"],
    "DB": ["분말 면의 검은 덩어리. 흰 상자(SE): 드러난 파트 (둘의 구별이 어려운 사례)",
           "제자리에서 밀려나 분말 면에 놓인 파트 조각 두 곳",
           "밀려난 조각 두 곳 (위쪽에 리코터). 끝내 리코터에 걸림",
           "(a)의 표시 영역을 확대"],
}

# 표지의 클래스별 미리보기: (약어, 사진, 중심 x, 중심 y, 한 변 = 사진 폭 대비)
THUMBS = [
    ("IS", "image3.jpeg", 0.27, 0.52, 0.52), ("PC", "image6.png", 0.40, 0.52, 0.36),
    ("RS", "image14.png", 0.30, 0.50, 0.56), ("RH", "image15.png", 0.50, 0.42, 0.62),
    ("SE", "image21.png", 0.42, 0.40, 0.46), ("SP", "image16.jpeg", 0.21, 0.50, 0.36),
    ("OM", "image26.png", 0.79, 0.21, 0.34), ("DB", "image28.jpeg", 0.33, 0.33, 0.38),
]


def emu(v: float) -> int:
    return int(round(v * EMU))


def pt(v: float) -> int:                  # pt → EMU (선 굵기)
    return int(round(v * 12700))


# ----------------------------------------------------------------------------
# 기하
# ----------------------------------------------------------------------------
def rounded_rect_outline(cx, cy, w, h, r, angle_deg=0.0, n=6) -> np.ndarray:
    """둥근 사각형 외곽선 점열(닫힘: 첫 점 반복). angle 은 화면 기준 시계 방향."""
    r = max(min(r, w / 2, h / 2), 0.0)
    pts = []
    for (ox, oy, a0) in ((w / 2 - r, -h / 2 + r, -90), (w / 2 - r, h / 2 - r, 0),
                         (-w / 2 + r, h / 2 - r, 90), (-w / 2 + r, -h / 2 + r, 180)):
        for k in range(n + 1):
            a = math.radians(a0 + 90 * k / n)
            pts.append((ox + r * math.cos(a), oy + r * math.sin(a)))
    pts.append(pts[0])
    p = np.array(pts)
    t = math.radians(angle_deg)
    rot = np.array([[math.cos(t), -math.sin(t)], [math.sin(t), math.cos(t)]])   # y 아래 방향 좌표 → 시계 방향
    return p @ rot.T + (cx, cy)


def _clip_seg(a, b, rect):
    """Liang–Barsky. → (p, q, 시작이 잘림, 끝이 잘림) 또는 None."""
    x0, y0, x1, y1 = rect
    dx, dy = b[0] - a[0], b[1] - a[1]
    t0, t1 = 0.0, 1.0
    for p_, q_ in ((-dx, a[0] - x0), (dx, x1 - a[0]), (-dy, a[1] - y0), (dy, y1 - a[1])):
        if abs(p_) < 1e-12:
            if q_ < 0:
                return None
            continue
        t = q_ / p_
        if p_ < 0:
            if t > t1:
                return None
            t0 = max(t0, t)
        else:
            if t < t0:
                return None
            t1 = min(t1, t)
    if t1 - t0 < 1e-9:
        return None
    return ((a[0] + t0 * dx, a[1] + t0 * dy), (a[0] + t1 * dx, a[1] + t1 * dy), t0 > 1e-9, t1 < 1 - 1e-9)


def clip_runs(pts, rect, closed=False):
    """꺾은선을 사각형 안쪽 부분만 남겨 여러 개의 연속 구간으로 나눈다."""
    runs, cur = [], []
    for a, b in zip(pts[:-1], pts[1:]):
        seg = _clip_seg(a, b, rect)
        if seg is None:
            if cur:
                runs.append(cur)
                cur = []
            continue
        p, q, cut0, cut1 = seg
        if cur and cut0:
            runs.append(cur)
            cur = []
        if not cur:
            cur = [p]
        cur.append(q)
        if cut1:
            runs.append(cur)
            cur = []
    if cur:
        runs.append(cur)
    runs = [r for r in runs if len(r) >= 2]
    if closed and len(runs) >= 2:                       # 시작점이 안쪽이면 마지막 구간과 첫 구간을 잇는다
        f, l = runs[0], runs[-1]
        if math.dist(f[0], l[-1]) < 1e-6:
            runs = [l + f[1:]] + runs[1:-1]
    return runs


# ----------------------------------------------------------------------------
# 슬라이드 XML
# ----------------------------------------------------------------------------
NS = ('xmlns:a="http://schemas.openxmlformats.org/drawingml/2006/main" '
      'xmlns:r="http://schemas.openxmlformats.org/officeDocument/2006/relationships" '
      'xmlns:p="http://schemas.openxmlformats.org/presentationml/2006/main"')
REL_IMG = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/image"
REL_LAYOUT = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slideLayout"
REL_SLIDE = "http://schemas.openxmlformats.org/officeDocument/2006/relationships/slide"


def _fill(color, alpha=None):
    a = f'<a:alpha val="{int(alpha * 1000)}"/>' if alpha is not None else ""
    return f'<a:solidFill><a:srgbClr val="{color}">{a}</a:srgbClr></a:solidFill>'


def _ln(color, width_pt, dash=None):
    d = f'<a:prstDash val="{dash}"/>' if dash else ""
    return f'<a:ln w="{pt(width_pt)}" cap="rnd">{_fill(color)}{d}<a:round/></a:ln>'


def _glow(color, alpha=80):
    return (f'<a:effectLst><a:glow rad="{pt(GLOW_PT)}"><a:srgbClr val="{color}">'
            f'<a:alpha val="{alpha * 1000}"/></a:srgbClr></a:glow></a:effectLst>')


def _xfrm(x, y, w, h, rot=0.0):
    r = f' rot="{int(round((rot % 360) * 60000))}"' if abs(rot % 360) > 1e-6 else ""
    return f'<a:xfrm{r}><a:off x="{emu(x)}" y="{emu(y)}"/><a:ext cx="{max(emu(w), 1)}" cy="{max(emu(h), 1)}"/></a:xfrm>'


def _runs(text, size, bold=False, color=INK):
    """줄바꿈(\\n)은 문단으로. 글꼴은 테마(Calibri / 맑은 고딕)를 따른다."""
    out = []
    for line in text.split("\n"):
        b = ' b="1"' if bold else ""
        out.append(f'<a:r><a:rPr lang="ko-KR" altLang="en-US" sz="{int(size * 100)}"{b} dirty="0">'
                   f'{_fill(color)}</a:rPr><a:t>{escape(line)}</a:t></a:r>')
    return out


class Slide:
    def __init__(self, layout: int, bg: str | None = None):
        self.layout, self.bg = layout, bg
        self.shapes: list[str] = []
        self.media: dict[str, str] = {}             # media 파일명 → rId
        self._id = 1

    def _nid(self):
        self._id += 1
        return self._id

    def _rid(self, media: str) -> str:
        if media not in self.media:
            self.media[media] = f"rId{len(self.media) + 2}"     # rId1 = layout
        return self.media[media]

    # --- 텍스트 ---------------------------------------------------------
    def title(self, text, x, y, w, h, size=30, color=INK, ph="title", align="l", bold=True):
        i = self._nid()
        paras = "".join(f'<a:p><a:pPr algn="{align}"/>{r}</a:p>' for r in _runs(text, size, bold, color))
        self.shapes.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{i}" name="Title {i}"/><p:cNvSpPr><a:spLocks noGrp="1"/></p:cNvSpPr>'
            f'<p:nvPr><p:ph type="{ph}"/></p:nvPr></p:nvSpPr><p:spPr>{_xfrm(x, y, w, h)}</p:spPr>'
            f'<p:txBody><a:bodyPr lIns="0" tIns="0" rIns="0" bIns="0" anchor="ctr"><a:normAutofit/></a:bodyPr>'
            f'<a:lstStyle/>{paras}</p:txBody></p:sp>')

    def text(self, text, x, y, w, h, size=12, bold=False, color=INK, align="l", anchor="t", name="Text",
             fill=None, fill_alpha=None, line=None, geom="rect", inset=0.0):
        i = self._nid()
        f = _fill(fill, fill_alpha) if fill else "<a:noFill/>"
        ln = _ln(*line) if line else "<a:ln><a:noFill/></a:ln>"
        av = '<a:avLst><a:gd name="adj" fmla="val 30000"/></a:avLst>' if geom == "roundRect" else "<a:avLst/>"
        ins = emu(inset)
        paras = "".join(f'<a:p><a:pPr algn="{align}"/>{r}</a:p>' for r in _runs(text, size, bold, color))
        self.shapes.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{i}" name={quoteattr(f"{name} {i}")}/><p:cNvSpPr txBox="1"/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr>{_xfrm(x, y, w, h)}<a:prstGeom prst="{geom}">{av}</a:prstGeom>{f}{ln}</p:spPr>'
            f'<p:txBody><a:bodyPr wrap="square" lIns="{ins}" tIns="{ins}" rIns="{ins}" bIns="{ins}" anchor="{anchor}">'
            f'<a:noAutofit/></a:bodyPr><a:lstStyle/>{paras}</p:txBody></p:sp>')

    # --- 그림 / 도형 ----------------------------------------------------
    def picture(self, media, x, y, w, h, src=None, line=None, glow=None, name="Picture", descr=""):
        """src = 사진 전체 기준 (x0, y0, x1, y1) 0-1 자르기."""
        i = self._nid()
        sr = ""
        if src:
            l, t, r, b = (int(round(v * 100000)) for v in (src[0], src[1], 1 - src[2], 1 - src[3]))
            sr = f'<a:srcRect l="{l}" t="{t}" r="{r}" b="{b}"/>'
        ln = _ln(*line) if line else ""
        fx = _glow(*glow) if glow else ""
        self.shapes.append(
            f'<p:pic><p:nvPicPr><p:cNvPr id="{i}" name={quoteattr(f"{name} {i}")} descr={quoteattr(descr)}/>'
            f'<p:cNvPicPr><a:picLocks noChangeAspect="1"/></p:cNvPicPr><p:nvPr/></p:nvPicPr>'
            f'<p:blipFill><a:blip r:embed="{self._rid(media)}"/>{sr}<a:stretch><a:fillRect/></a:stretch></p:blipFill>'
            f'<p:spPr>{_xfrm(x, y, w, h)}<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>{ln}{fx}</p:spPr></p:pic>')

    def rect(self, x, y, w, h, fill=None, line=None, glow=None, rot=0.0, radius=None, name="Shape"):
        """radius 가 있으면 둥근 사각형 (radius = 모서리 반경, 인치)."""
        i = self._nid()
        if radius is not None:
            adj = int(round(min(radius / max(min(w, h), 1e-6), 0.5) * 100000))
            geom = f'<a:prstGeom prst="roundRect"><a:avLst><a:gd name="adj" fmla="val {adj}"/></a:avLst></a:prstGeom>'
        else:
            geom = '<a:prstGeom prst="rect"><a:avLst/></a:prstGeom>'
        f = _fill(fill) if fill else "<a:noFill/>"
        ln = _ln(*line) if line else "<a:ln><a:noFill/></a:ln>"
        fx = _glow(*glow) if glow else ""
        self.shapes.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{i}" name={quoteattr(f"{name} {i}")}/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr>{_xfrm(x, y, w, h, rot)}{geom}{f}{ln}{fx}</p:spPr>'
            f'<p:txBody><a:bodyPr rtlCol="0" anchor="ctr"/><a:lstStyle/><a:p><a:endParaRPr lang="ko-KR" altLang="en-US"/></a:p></p:txBody></p:sp>')

    def freeform(self, runs, line, glow=None, name="Freeform"):
        """runs = [[(x, y), ...], ...] (인치). 열린 경로들을 하나의 자유형 도형으로."""
        allp = np.array([p for r in runs for p in r])
        x0, y0 = allp.min(0)
        x1, y1 = allp.max(0)
        w, h = max(x1 - x0, 1e-4), max(y1 - y0, 1e-4)
        i = self._nid()
        path = []
        for r in runs:
            for k, (x, y) in enumerate(r):
                tag = "moveTo" if k == 0 else "lnTo"
                path.append(f'<a:{tag}><a:pt x="{emu(x - x0)}" y="{emu(y - y0)}"/></a:{tag}>')
        fx = _glow(*glow) if glow else ""
        self.shapes.append(
            f'<p:sp><p:nvSpPr><p:cNvPr id="{i}" name={quoteattr(f"{name} {i}")}/><p:cNvSpPr/><p:nvPr/></p:nvSpPr>'
            f'<p:spPr>{_xfrm(x0, y0, w, h)}<a:custGeom><a:avLst/><a:gdLst/><a:ahLst/><a:cxnLst/>'
            f'<a:rect l="0" t="0" r="r" b="b"/><a:pathLst><a:path w="{max(emu(w), 1)}" h="{max(emu(h), 1)}" fill="none">'
            f'{"".join(path)}</a:path></a:pathLst></a:custGeom><a:noFill/>{_ln(*line)}{fx}</p:spPr>'
            f'<p:txBody><a:bodyPr rtlCol="0" anchor="ctr"/><a:lstStyle/><a:p><a:endParaRPr lang="ko-KR" altLang="en-US"/></a:p></p:txBody></p:sp>')

    def tag(self, x, y_bottom, text, bg, fg, bounds=None, size=12, name="Tag"):
        """약어 태그. (x, y_bottom) = 왼쪽 아래. bounds 안에 들어오도록 밀어 넣는다."""
        w, h = 0.20 + 0.115 * len(text), 0.27
        y = y_bottom - h
        if bounds:
            bx0, by0, bx1, by1 = bounds
            x = min(max(x, bx0 + 0.04), bx1 - w - 0.04)
            y = min(max(y, by0 + 0.04), by1 - h - 0.04)
        line = (INK, 0.75) if bg == "FFFFFF" else None
        self.text(text, x, y, w, h, size=size, bold=True, color=fg, align="ctr", anchor="ctr", name=name,
                  fill=bg, line=line, geom="roundRect")

    # --- 직렬화 ---------------------------------------------------------
    def xml(self) -> str:
        bg = f'<p:bg><p:bgPr>{_fill(self.bg)}<a:effectLst/></p:bgPr></p:bg>' if self.bg else ""
        return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                f'<p:sld {NS}><p:cSld>{bg}<p:spTree><p:nvGrpSpPr><p:cNvPr id="1" name=""/><p:cNvGrpSpPr/><p:nvPr/></p:nvGrpSpPr>'
                '<p:grpSpPr><a:xfrm><a:off x="0" y="0"/><a:ext cx="0" cy="0"/><a:chOff x="0" y="0"/><a:chExt cx="0" cy="0"/></a:xfrm></p:grpSpPr>'
                f'{"".join(self.shapes)}</p:spTree></p:cSld><p:clrMapOvr><a:masterClrMapping/></p:clrMapOvr></p:sld>')

    def rels(self) -> str:
        r = [f'<Relationship Id="rId1" Type="{REL_LAYOUT}" Target="../slideLayouts/slideLayout{self.layout}.xml"/>']
        r += [f'<Relationship Id="{rid}" Type="{REL_IMG}" Target="../media/{m}"/>' for m, rid in self.media.items()]
        return ('<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
                f'<Relationships xmlns="http://schemas.openxmlformats.org/package/2006/relationships">{"".join(r)}</Relationships>')


# ----------------------------------------------------------------------------
# 패널 그리기 — figure_defect_examples.draw_panel 과 같은 좌표 계산
# ----------------------------------------------------------------------------
def media_name(src: str) -> str:
    return Path(src).stem + ".jpeg"


def hexcol(c: str) -> str:
    """matplotlib 색 표기('#RRGGBB', 'white') → OOXML 'RRGGBB'."""
    return {"white": "FFFFFF", "black": "000000"}.get(c.lower(), c.lstrip("#").upper())


def draw_panel(s: Slide, img: Image.Image, spec: dict, primary, aspect: float, PX: float, PW: float):
    w, h = img.size
    x0, y0, x1, y1 = fit_crop(spec["crop"], w, h, aspect)
    pw, ph = x1 - x0, y1 - y0
    k = PW / pw                                         # 픽셀 → 인치
    panel = (PX, PY, PX + PW, PY + PH)
    med = media_name(spec["media"])

    def P(x, y):                                        # 사진 전체 기준 (0-1) → 슬라이드 인치
        return PX + (x * w - x0) * k, PY + (y * h - y0) * k

    names = " / ".join(CLASS_NAMES.get(c.strip(), c.strip()) for c in spec["cls"].split("/"))
    s.picture(med, PX, PY, PW, PH, src=(x0 / w, y0 / h, x1 / w, y1 / h), line=("333333", 0.75),
              name="Photo", descr=f"{spec['title'] or names} — {spec['meta']}")
    r = 0.014 * PW
    tags, insets = [], []

    for mk in spec["marks"]:
        col, _halo, tbg, tfg = (hexcol(c) for c in style(mk["cls"], primary))
        glow = ("FFFFFF", 85) if col == RED else (INK, 75)
        nm = f"Mark {mk['cls']}"
        tag_xy = None
        if mk["kind"] in ("box", "rbox"):
            if mk["kind"] == "box":
                (ax0, ay0), (ax1, ay1) = P(*mk["box"][:2]), P(*mk["box"][2:])
                cx, cy, bw, bh, ang = (ax0 + ax1) / 2, (ay0 + ay1) / 2, ax1 - ax0, ay1 - ay0, 0.0
                tag_xy = (ax0 + 0.01 * PW, ay0 - 0.012 * PH)
            else:
                cx, cy = P(*mk["c"])
                bw, bh, ang = mk["length"] * w * k, mk["thick"] * w * k, mk["angle"]
                tag_xy = (cx, cy - bh / 2 - 0.02 * PH)
            outline = rounded_rect_outline(cx, cy, bw, bh, r, ang)
            inside = (outline[:, 0].min() >= panel[0] - 1e-6 and outline[:, 0].max() <= panel[2] + 1e-6 and
                      outline[:, 1].min() >= panel[1] - 1e-6 and outline[:, 1].max() <= panel[3] + 1e-6)
            if inside:                                  # 그대로 편집 가능한 둥근 사각형
                s.rect(cx - bw / 2, cy - bh / 2, bw, bh, line=(col, LINE_PT), glow=glow, rot=ang, radius=r, name=nm)
            else:                                       # 패널 밖으로 나가는 부분은 잘라 낸 자유형
                runs = clip_runs([tuple(p) for p in outline], panel, closed=True)
                if runs:
                    s.freeform(runs, line=(col, LINE_PT), glow=glow, name=nm)
        elif mk["kind"] == "curve":
            pts = curve_path([((x * w - x0), (y * h - y0)) for x, y in mk["pts"]], mk.get("offset", 0.0) * pw)
            pts = [(PX + px * k, PY + py * k) for px, py in pts]
            runs = clip_runs(pts, panel)
            if runs:
                s.freeform(runs, line=(col, CURVE_PT, "dash"), glow=glow, name=f"{nm} boundary")
            tag_xy = pts[len(pts) // 2]

        if mk.get("tag"):
            if mk.get("tag_at"):
                tag_xy = P(*mk["tag_at"])
            tags.append((tag_xy, mk["cls"], tbg, tfg))

        ins = mk.get("inset")
        if ins:
            sx0, sy0, sx1, sy1 = ins.get("src") or mk["box"]
            sw, sh = (sx1 - sx0) * w, (sy1 - sy0) * h
            iw = ins["w"] * PW
            ih = iw * sh / sw
            insets.append((PX + ins["anchor"][0] * PW, PY + ins["anchor"][1] * PH, iw, ih,
                           (sx0, sy0, sx1, sy1), col, glow, iw / (sw * k)))

    for ix, iy, iw, ih, src, col, glow, zoom in insets:
        s.picture(med, ix, iy, iw, ih, src=src, line=(col, LINE_PT), glow=glow, name="Inset",
                  descr=f"확대 ×{zoom:.1f}")
        s.text(f"×{zoom:.1f}", ix + 0.06, iy + ih - 0.33, 0.62, 0.27, size=12, bold=True, color="FFFFFF",
               align="ctr", anchor="ctr", name="Zoom", fill=INK, fill_alpha=70, geom="roundRect")
    for (tx, ty), text, bg, fg in tags:
        s.tag(tx, ty, text, bg, fg, bounds=panel)
    return names


def content_slide(name: str, spec: dict, idxs, media) -> Slide:
    s = Slide(LAYOUT_TITLE_ONLY)
    aspect, primary = spec["aspect"], spec["primary"]
    PW = PH * aspect
    left = (SLIDE_W - (2 * PW + GAP)) / 2
    classes = []
    for col, k in enumerate(idxs):
        p = spec["panels"][k]
        for c in p["cls"].split("/"):
            n = CLASS_NAMES.get(c.strip(), c.strip())
            if n not in classes:
                classes.append(n)
    s.title(" · ".join(classes), 0.5, 0.28, 9.4, 0.72)
    letters = "–".join(f"({chr(97 + k)})" for k in (idxs[0], idxs[-1]))
    s.text(f"논문 그림  Figs/Defect_Examples_{name}  {letters}", 9.2, 0.50, 3.63, 0.30, size=11, color=MUTED,
           align="r", anchor="ctr", name="Figure reference")
    for col, k in enumerate(idxs):
        p = spec["panels"][k]
        PX = left + col * (PW + GAP)
        names = draw_panel(s, media[p["media"]], p, primary, aspect, PX, PW)
        y = PY + PH
        s.text(f"({chr(97 + k)}) {p['title'] or names}", PX, y + 0.10, PW, 0.32, size=16, bold=True,
               align="ctr", anchor="ctr", name="Panel label")
        s.text(p["meta"], PX, y + 0.44, PW, 0.24, size=11, color=MUTED, align="ctr", anchor="ctr", name="Build / layer")
        s.text(DESC[name][k], PX, y + 0.70, PW, 0.46, size=12, align="ctr", anchor="t", name="Description")
    return s


def title_slide(media, n_photos: int) -> Slide:
    s = Slide(LAYOUT_TITLE, bg=DARK_BG)
    s.title("PBF 결함 예시 사진", 0.7, 1.20, 11.9, 1.0, size=44, color="FFFFFF", ph="ctrTitle")
    s.text(f"외형 기준 결함 분류 체계(제안)  ·  결함 클래스 8종  ·  사진 {n_photos}장", 0.7, 2.25, 11.9, 0.45,
           size=18, color="C9CDD3", anchor="ctr", name="Subtitle")
    s.text("원자료: Defect Example.pptx  ·  논문 그림 Figs/Defect_Examples_*.pdf 와 같은 내용  ·  표시는 편집 가능한 도형",
           0.7, 2.73, 11.9, 0.36, size=13, color="9AA0A8", anchor="ctr", name="Source")
    n = len(THUMBS)
    tw, gap = 1.32, 0.19
    x = (SLIDE_W - (n * tw + (n - 1) * gap)) / 2
    for ab, m, cx, cy, size in THUMBS:
        w, h = media[m].size
        half = size * w / 2
        px0, py0 = cx * w - half, cy * h - half
        px0 = min(max(px0, 0), w - 2 * half)
        py0 = min(max(py0, 0), h - 2 * half)
        src = (px0 / w, py0 / h, (px0 + 2 * half) / w, (py0 + 2 * half) / h)
        s.picture(media_name(m), x, 4.05, tw, tw, src=src, line=("8A9099", 0.75), name="Thumbnail",
                  descr=CLASS_NAMES[ab])
        # 두 낱말 이름은 일부러 두 줄로 — 글꼴이 바뀌어도 줄바꿈 위치가 같다
        s.text(CLASS_NAMES[ab].replace(" ", "\n"), x - 0.09, 4.05 + tw + 0.10, tw + 0.18, 0.50, size=12,
               color="E6E8EB", align="ctr", anchor="t", name="Class")
        x += tw + gap
    return s


def legend_slide(media) -> Slide:
    s = Slide(LAYOUT_TITLE_ONLY)
    s.title("표시 방식과 사진 처리", 0.5, 0.28, 12.3, 0.72)
    sw_x, sw_w, sw_h, tx = 0.6, 2.1, 1.02, 3.0
    rows = [
        ("붉은 상자", "그 슬라이드가 다루는 클래스의 결함 위치.\n길쭉하게 기울어진 결함에는 기울어진 상자를 쓴다."),
        ("흰 상자와 약어 태그", "같은 사진에 함께 나타난 다른 클래스.\n태그의 약어로 클래스를 구분한다."),
        ("점선", "도포 경계를 따라 그린 안내선.\n경계를 가리지 않도록 옆으로 살짝 비켜 그렸다."),
        ("확대 인셋", "작은 결함을 확대한 것.\n왼쪽 아래 숫자는 확대 배율이다."),
    ]
    y = 1.30
    for k, (head, body) in enumerate(rows):
        s.rect(sw_x, y, sw_w, sw_h, fill=SWATCH, name="Swatch")
        cx, cy = sw_x + sw_w / 2, y + sw_h / 2
        if k == 0:
            s.rect(cx - 0.85, cy - 0.30, 0.80, 0.60, line=(RED, LINE_PT), glow=("FFFFFF", 85), radius=0.06, name="Sample")
            s.rect(cx + 0.10, cy - 0.11, 0.85, 0.22, line=(RED, LINE_PT), glow=("FFFFFF", 85), radius=0.06, rot=-28,
                   name="Sample")
        elif k == 1:
            s.rect(cx - 0.55, cy - 0.22, 0.90, 0.56, line=("FFFFFF", LINE_PT), glow=(INK, 75), radius=0.06, name="Sample")
            s.tag(cx + 0.42, cy - 0.02, "RS", "FFFFFF", INK)
        elif k == 2:
            xs = np.linspace(-0.85, 0.85, 40)
            pts = [(cx + v, cy + 0.26 * math.sin(v * 2.2)) for v in xs]
            s.freeform([pts], line=(RED, CURVE_PT, "dash"), glow=("FFFFFF", 85), name="Sample")
        else:
            m, src = "image19.png", (0.83, 0.285, 0.975, 0.40)
            w, h = media[m].size
            ih = 0.86
            iw = ih * ((src[2] - src[0]) * w) / ((src[3] - src[1]) * h)
            s.picture(media_name(m), cx - iw / 2, cy - ih / 2, iw, ih, src=src, line=(RED, LINE_PT), glow=("FFFFFF", 85),
                      name="Sample inset", descr="확대 인셋 예")
            s.text("×2.5", cx - iw / 2 + 0.05, cy + ih / 2 - 0.30, 0.56, 0.25, size=11, bold=True, color="FFFFFF",
                   align="ctr", anchor="ctr", name="Zoom", fill=INK, fill_alpha=70, geom="roundRect")
        s.text(head, tx, y + 0.02, 4.3, 0.32, size=16, bold=True, anchor="ctr", name="Legend head")
        s.text(body, tx, y + 0.38, 4.3, 0.62, size=13, color="3A3F45", anchor="t", name="Legend body")
        y += sw_h + 0.28

    # 오른쪽: 약어, 사진 처리
    rx = 7.75
    s.text("약어", rx, 1.30, 4.8, 0.32, size=16, bold=True, anchor="ctr", name="Head")
    abbr = list(CLASS_NAMES.items())
    for k, (ab, full) in enumerate(abbr):
        col, row = divmod(k, 4)
        ax, ay = rx + col * 2.58, 1.76 + row * 0.40
        s.text(ab, ax, ay, 0.46, 0.27, size=12, bold=True, color="FFFFFF", align="ctr", anchor="ctr", name="Tag",
               fill=RED, geom="roundRect")
        s.text(full, ax + 0.56, ay - 0.02, 1.98, 0.31, size=12, anchor="ctr", name="Class name")
    s.text("사진 처리", rx, 3.62, 4.8, 0.32, size=16, bold=True, anchor="ctr", name="Head")
    s.text("자르기와, 사진마다 화면 전체에 균일하게 적용한 레벨·감마 조정만 했다.\n"
           f"휘도 기준, 선형 증폭 {MAX_GAIN:g}배 이하, 감마 지수 {GAMMA_MIN:g}–1.0.\n"
           "국소 보정과 합성은 하지 않았다.",
           rx, 4.00, 5.05, 1.05, size=13, color="3A3F45", anchor="t", name="Body")
    s.text("이 덱에서 고칠 수 있는 것", rx, 5.22, 4.8, 0.32, size=16, bold=True, anchor="ctr", name="Head")
    s.text("상자·점선·태그는 모두 도형이라 옮기고 크기와 각도를 바꿀 수 있다.\n"
           "사진에는 자르기만 걸려 있어 범위를 다시 잡을 수 있다.\n"
           "패널 아래의 빌드·레이어 표기는 원자료 슬라이드의 메모다.",
           rx, 5.60, 5.05, 1.05, size=13, color="3A3F45", anchor="t", name="Body")
    return s


# ----------------------------------------------------------------------------
# 패키지 쓰기 — 원자료 덱의 테마·마스터·레이아웃을 그대로 쓰고 슬라이드만 새로 쓴다
# ----------------------------------------------------------------------------
def write_package(template: Path, out: Path, slides: list[Slide], images: dict[str, bytes]):
    zin = zipfile.ZipFile(template)
    keep = [n for n in zin.namelist()
            if not re.match(r"ppt/(slides|notesSlides|media|comments|changesInfos)/", n)
            and n not in ("docProps/thumbnail.jpeg", "ppt/revisionInfo.xml")]
    data = {n: zin.read(n) for n in keep}

    # presentation.xml — 슬라이드 목록 교체
    pres = data["ppt/presentation.xml"].decode("utf-8")
    prels = data["ppt/_rels/presentation.xml.rels"].decode("utf-8")
    prels = re.sub(rf'<Relationship [^>]*Type="{re.escape(REL_SLIDE)}"[^>]*/>', "", prels)
    used = {int(v) for v in re.findall(r'Id="rId(\d+)"', prels)}
    nxt = max(used) + 1
    ids, rel_new = [], []
    for k in range(len(slides)):
        rid = f"rId{nxt + k}"
        ids.append(f'<p:sldId id="{256 + k}" r:id="{rid}"/>')
        rel_new.append(f'<Relationship Id="{rid}" Type="{REL_SLIDE}" Target="slides/slide{k + 1}.xml"/>')
    pres = re.sub(r"<p:sldIdLst>.*?</p:sldIdLst>", f'<p:sldIdLst>{"".join(ids)}</p:sldIdLst>', pres, flags=re.S)
    prels = prels.replace("</Relationships>", "".join(rel_new) + "</Relationships>")
    data["ppt/presentation.xml"] = pres.encode("utf-8")
    data["ppt/_rels/presentation.xml.rels"] = prels.encode("utf-8")

    # [Content_Types].xml
    ct = data["[Content_Types].xml"].decode("utf-8")
    ct = re.sub(r'<Override PartName="/ppt/(slides|notesSlides|comments)/[^"]*"[^>]*/>', "", ct)
    if 'Extension="jpeg"' not in ct:
        ct = ct.replace("<Default ", '<Default Extension="jpeg" ContentType="image/jpeg"/><Default ', 1)
    ov = "".join(f'<Override PartName="/ppt/slides/slide{k + 1}.xml" '
                 'ContentType="application/vnd.openxmlformats-officedocument.presentationml.slide+xml"/>'
                 for k in range(len(slides)))
    data["[Content_Types].xml"] = ct.replace("</Types>", ov + "</Types>").encode("utf-8")

    # 최상위 rels — 옛 미리보기 그림 관계 제거
    top = data["_rels/.rels"].decode("utf-8")
    data["_rels/.rels"] = re.sub(r'<Relationship [^>]*Target="docProps/thumbnail\.jpeg"[^>]*/>', "", top).encode("utf-8")
    # app.xml — 슬라이드 수만 맞춘 최소 속성
    data["docProps/app.xml"] = (
        '<?xml version="1.0" encoding="UTF-8" standalone="yes"?>\n'
        '<Properties xmlns="http://schemas.openxmlformats.org/officeDocument/2006/extended-properties">'
        f'<Application>Microsoft Office PowerPoint</Application><Slides>{len(slides)}</Slides></Properties>').encode("utf-8")

    for k, s in enumerate(slides):
        data[f"ppt/slides/slide{k + 1}.xml"] = s.xml().encode("utf-8")
        data[f"ppt/slides/_rels/slide{k + 1}.xml.rels"] = s.rels().encode("utf-8")
    used_media = {m for s in slides for m in s.media}
    for m in sorted(used_media):
        data[f"ppt/media/{m}"] = images[m]

    out.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(out, "w", zipfile.ZIP_DEFLATED) as z:
        z.writestr("[Content_Types].xml", data.pop("[Content_Types].xml"))       # 관례상 첫 항목
        for n in sorted(data):
            z.writestr(n, data[n], zipfile.ZIP_STORED if n.endswith(".jpeg") else zipfile.ZIP_DEFLATED)
    return sorted(used_media)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--pptx", type=Path, default=PPTX, help="원자료 덱 (사진과 테마를 가져옴)")
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--raw", action="store_true", help="밝기 조정을 하지 않음")
    args = ap.parse_args()
    if args.out.resolve() == args.pptx.resolve():
        raise SystemExit("FATAL: --out 이 원자료 덱과 같다 (덮어쓰지 않음)")

    media = read_media(args.pptx)
    if not args.raw:
        media = {k: Image.fromarray(tone(np.asarray(v))[0]) for k, v in media.items()}
    images = {}
    for k, im in media.items():
        buf = io.BytesIO()
        im.save(buf, "JPEG", quality=95, subsampling=0)
        images[media_name(k)] = buf.getvalue()

    slides = [title_slide(media, len(media)), legend_slide(media)]
    for name, spec in FIGURES.items():
        n = len(spec["panels"])
        for i in range(0, n, 2):
            slides.append(content_slide(name, spec, list(range(i, min(i + 2, n))), media))
    used = write_package(args.pptx, args.out, slides, images)
    size = args.out.stat().st_size / 1e6
    print(f"wrote {args.out}  ({len(slides)} slides, {len(used)} photos, {size:.1f} MB)")


if __name__ == "__main__":
    main()
