#!/usr/bin/env python3
"""LaTeX 원고 정적 점검기 — paper/DefectSegAM_Ko (elsarticle, Additive Manufacturing).

로컬에 LaTeX 툴체인이 없으므로(README 참조: Overleaf 컴파일) 컴파일 없이 잡을 수 있는
오류/불일치를 stdlib 만으로 점검한다. defseg-paper-author 서브에이전트가 매 편집 후 실행한다.

점검 항목
  ERROR  — 컴파일이 깨지거나 참조가 [?] 로 출력될 것이 확실한 항목
           (중괄호/환경/수식 delimiter 불균형, 미정의 \\ref·\\cite 키, 없는 \\input·그림 파일,
            fragment 안의 preamble, 라벨 중복, tabular 밖의 unescaped '&', TAB 문자)
  WARN   — 저널 규정·원고 규약 위반 가능성 (highlights 수/길이, keyword 수, 끝맺음 절 누락,
           \\TODO 잔존, 미사용 bib 엔트리, orphan 라벨, math 밖의 '_'/'^', 비-ASCII 특수문자)
  INFO   — 문체 일관성 통계 (Section~ vs 절, Fig.~ vs 그림~, $..$ vs \\(..\\), 라틴어 뒤 조사 띄어쓰기, 긴 줄)

사용 (CWD 무관, 절대경로 기본값):
    python DefSeg_AM/scripts/check_paper_tex.py                # 전체 점검
    python DefSeg_AM/scripts/check_paper_tex.py --paper-dir <dir>
    python DefSeg_AM/scripts/check_paper_tex.py --no-info      # ERROR/WARN 만
    python DefSeg_AM/scripts/check_paper_tex.py --todo         # \\TODO 인벤토리만 출력
종료 코드: ERROR 가 하나라도 있으면 1, 아니면 0.
"""
from __future__ import annotations

import argparse
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

DEFAULT_PAPER_DIR = Path(__file__).resolve().parents[1] / "paper" / "DefectSegAM_Ko"

# 표 계열 환경 — 이 안의 '&' 는 정상
TABULAR_ENVS = {"tabular", "tabularx", "tabular*", "array", "longtable", "aligned", "align",
                "align*", "cases", "split", "matrix", "pmatrix", "bmatrix", "eqnarray", "alignat"}
MATH_ENVS = {"equation", "equation*", "align", "align*", "aligned", "gather", "gather*",
             "multline", "multline*", "eqnarray", "split", "alignat", "cases", "matrix",
             "pmatrix", "bmatrix", "displaymath", "math"}
ELSEVIER_ENDMATTER = [  # (표시 이름, main.tex 에서 찾을 정규식, 심각도)
    ("CRediT authorship contribution statement", r"CRediT", "WARN"),
    ("Declaration of competing interest", r"[Cc]ompeting interest|이해상충", "WARN"),
    ("Data availability", r"Data availability|데이터 가용성", "WARN"),
    ("Acknowledgements / Funding", r"Acknowledg|감사의 글|Funding|연구비", "WARN"),
    ("Declaration of generative AI use (Elsevier 정책, 해당 시 필수)", r"[Gg]enerative AI|생성형 AI", "WARN"),
]

HANGUL = re.compile(r"[\u1100-\u11FF\u3130-\u318F\uAC00-\uD7A3]")
PARTICLES = "은|는|이|가|을|를|의|에|에서|과|와|로|으로|도|만|까지|부터|처럼|보다|이다|이며|였다|이고"


class Report:
    def __init__(self) -> None:
        self.items: list[tuple[str, str, str]] = []  # (severity, location, message)

    def add(self, sev: str, loc: str, msg: str) -> None:
        self.items.append((sev, loc, msg))

    def count(self, sev: str) -> int:
        return sum(1 for s, _, _ in self.items if s == sev)


def strip_comments(line: str) -> str:
    """'%' 주석 제거 (escape 된 '\\%' 는 유지)."""
    out = []
    i = 0
    while i < len(line):
        c = line[i]
        if c == "\\" and i + 1 < len(line):
            out.append(line[i:i + 2])
            i += 2
            continue
        if c == "%":
            break
        out.append(c)
        i += 1
    return "".join(out)


def read_lines(path: Path) -> list[str]:
    return path.read_text(encoding="utf-8").splitlines()


def code_lines(path: Path) -> list[tuple[int, str]]:
    return [(i + 1, strip_comments(l)) for i, l in enumerate(read_lines(path))]


def strip_tex_cmds(s: str) -> str:
    """캐릭터 수 세기용 — \\cmd{...} 의 명령 이름만 제거하고 인자 텍스트는 남김."""
    s = re.sub(r"\\[A-Za-z]+\*?", "", s)
    s = re.sub(r"[{}]", "", s)
    s = re.sub(r"\s+", " ", s)
    return s.strip()


# --------------------------------------------------------------------------- per-file checks
def check_braces(path: Path, rep: Report) -> None:
    depth = 0
    first_neg = None
    for ln, l in code_lines(path):
        j = 0
        while j < len(l):
            c = l[j]
            if c == "\\" and j + 1 < len(l):
                j += 2
                continue
            if c == "{":
                depth += 1
            elif c == "}":
                depth -= 1
                if depth < 0 and first_neg is None:
                    first_neg = ln
            j += 1
    if first_neg is not None:
        rep.add("ERROR", f"{path.name}:{first_neg}", "닫는 '}' 가 여는 '{' 보다 먼저 나옴 (중괄호 불균형)")
    elif depth != 0:
        rep.add("ERROR", path.name, f"중괄호 불균형: 파일 끝에서 depth={depth:+d}")


def check_envs(path: Path, rep: Report, is_main: bool) -> tuple[set[str], list[tuple[int, str]]]:
    """\\begin/\\end 스택 검사. 반환: (사용된 env 집합, 환경 시작 로그)."""
    stack: list[tuple[str, int]] = []
    used: set[str] = set()
    pat = re.compile(r"\\(begin|end)\{([A-Za-z*]+)\}")
    for ln, l in code_lines(path):
        for m in pat.finditer(l):
            kind, env = m.group(1), m.group(2)
            used.add(env)
            if kind == "begin":
                stack.append((env, ln))
            else:
                if not stack:
                    rep.add("ERROR", f"{path.name}:{ln}", f"\\end{{{env}}} 에 대응하는 \\begin 없음")
                elif stack[-1][0] != env:
                    top, tln = stack[-1]
                    rep.add("ERROR", f"{path.name}:{ln}", f"\\end{{{env}}} 이지만 열린 환경은 {top} (line {tln})")
                    stack.pop()
                else:
                    stack.pop()
    for env, ln in stack:
        rep.add("ERROR", f"{path.name}:{ln}", f"\\begin{{{env}}} 이 닫히지 않음")
    if not is_main and "document" in used:
        rep.add("ERROR", path.name, "fragment 안에 \\begin{document} 가 있음")
    return used, []


def check_math_delims(path: Path, rep: Report) -> tuple[int, int]:
    """'$' 짝수, \\( \\) 및 \\[ \\] 균형. 반환: (inline $..$ 개수, \\(..\\) 개수)."""
    dollars = 0
    n_paren_open = n_paren_close = n_brk_open = n_brk_close = 0
    for ln, l in code_lines(path):
        j = 0
        while j < len(l):
            c = l[j]
            if c == "\\" and j + 1 < len(l):
                nxt = l[j + 1]
                if nxt == "(":
                    n_paren_open += 1
                elif nxt == ")":
                    n_paren_close += 1
                elif nxt == "[":
                    n_brk_open += 1
                elif nxt == "]":
                    n_brk_close += 1
                j += 2
                continue
            if c == "$":
                dollars += 1
            j += 1
    if dollars % 2:
        rep.add("ERROR", path.name, f"'$' 개수가 홀수({dollars}) — 인라인 수식 delimiter 불균형")
    if n_paren_open != n_paren_close:
        rep.add("ERROR", path.name, f"\\( {n_paren_open} 개 vs \\) {n_paren_close} 개 — 불균형")
    if n_brk_open != n_brk_close:
        rep.add("ERROR", path.name, f"\\[ {n_brk_open} 개 vs \\] {n_brk_close} 개 — 불균형")
    return dollars // 2, n_paren_open


def iter_with_context(path: Path):
    """각 코드 라인에 대해 (ln, line, in_tabular, in_math_env) 를 생성."""
    env_stack: list[str] = []
    pat = re.compile(r"\\(begin|end)\{([A-Za-z*]+)\}")
    for ln, l in code_lines(path):
        in_tab = any(e in TABULAR_ENVS for e in env_stack)
        in_math = any(e in MATH_ENVS for e in env_stack)
        yield ln, l, in_tab, in_math
        for m in pat.finditer(l):
            if m.group(1) == "begin":
                env_stack.append(m.group(2))
            elif env_stack and env_stack[-1] == m.group(2):
                env_stack.pop()


def masked_math_lines(path: Path) -> list[str]:
    """파일 전체를 문자 단위로 훑어 $...$, $$...$$, \\(...\\), \\[...\\] 내부(여러 줄 걸침 포함)를
    공백으로 마스킹한 코드 라인 목록을 반환 (특수문자 검사용)."""
    text = "\n".join(l for _, l in code_lines(path))
    out: list[str] = []
    mode: str | None = None  # None | '$' | '$$' | '\\(' | '\\['
    i = 0
    n = len(text)
    while i < n:
        c = text[i]
        two = text[i:i + 2]
        if mode is None:
            if two == "\\(":
                mode, i = "\\(", i + 2
                out.append("  ")
            elif two == "\\[":
                mode, i = "\\[", i + 2
                out.append("  ")
            elif c == "\\" and i + 1 < n:
                out.append(two)
                i += 2
            elif two == "$$":
                mode, i = "$$", i + 2
                out.append("  ")
            elif c == "$":
                mode, i = "$", i + 1
                out.append(" ")
            else:
                out.append(c)
                i += 1
        else:
            closer = {"$": "$", "$$": "$$", "\\(": "\\)", "\\[": "\\]"}[mode]
            if c == "\n":
                out.append("\n")
                i += 1
            elif text.startswith(closer, i) and not (mode == "$" and two == "$$"):
                out.append(" " * len(closer))
                i += len(closer)
                mode = None
            elif c == "\\" and i + 1 < n:
                out.append("  ")
                i += 2
            else:
                out.append(" ")
                i += 1
    return "".join(out).split("\n")


def check_special_chars(path: Path, rep: Report) -> None:
    masked_lines = masked_math_lines(path)
    for ln, l, in_tab, in_math in iter_with_context(path):
        if "\t" in l:
            rep.add("ERROR", f"{path.name}:{ln}", "TAB 문자 — verbatim 밖에서는 스페이스로 교체")
        if in_math:
            continue
        masked = masked_lines[ln - 1]
        # unescaped '&' outside tabular-like envs
        if not in_tab:
            for m in re.finditer(r"(?<!\\)&", masked):
                rep.add("ERROR", f"{path.name}:{ln}", "tabular 밖의 escape 안 된 '&' (\\& 로)")
                break
        # unescaped '_' / '^' outside math (heuristic; \label/\ref/\cite/\input/\includegraphics 인자는 허용)
        safe = re.sub(r"\\(label|ref|eqref|cite[tp]?|input|includegraphics|url|href|texttt|path|hyperref|bibliography)"
                      r"(\[[^\]]*\])?\{[^}]*\}", "", masked)
        if re.search(r"(?<!\\)_", safe):
            rep.add("WARN", f"{path.name}:{ln}", "math 밖의 '_' — 식별자면 '\\_' 로 escape 필요")
        if re.search(r"(?<!\\)\^", safe):
            rep.add("WARN", f"{path.name}:{ln}", "math 밖의 '^' — '\\textasciicircum' 또는 수식으로")
        if re.search(r"(?<!\\)#", safe) and "newcommand" not in l and "renewcommand" not in l:
            rep.add("WARN", f"{path.name}:{ln}", "escape 안 된 '#' (\\# 로)")


def check_non_ascii(path: Path, rep: Report) -> None:
    """한글 이외의 비-ASCII 문자. Latin-1 보충(·, ×, °, µ, ±)과 대시/따옴표/줄임표는 inputenc utf8 이
    처리하므로 INFO, 그 외(→, ≈, ≥, ≤, 그리스 문자 등)는 pdfLaTeX 에서 미정의일 수 있어 WARN."""
    found: dict[str, list[int]] = defaultdict(list)
    for ln, l in code_lines(path):
        for ch in l:
            if ord(ch) < 128 or HANGUL.match(ch):
                continue
            found[ch].append(ln)
    for ch, lns in sorted(found.items(), key=lambda kv: -len(kv[1])):
        where = ", ".join(map(str, lns[:6])) + (" ..." if len(lns) > 6 else "")
        safe = 0x00A0 <= ord(ch) <= 0x00FF or ch in "–—‘’“”…"
        sev = "INFO" if safe else "WARN"
        hint = "컴파일은 되나 LaTeX 표기 권장" if safe else "pdfLaTeX+inputenc 에서 미정의일 수 있음 — 매크로로 대체"
        rep.add(sev, f"{path.name}", f"비-ASCII 문자 U+{ord(ch):04X} '{ch}' ×{len(lns)} (line {where}) — {hint}")


def check_lines(path: Path, rep: Report, max_len: int = 100) -> None:
    for i, raw in enumerate(read_lines(path), 1):
        if len(raw) > max_len and not raw.lstrip().startswith("%"):
            rep.add("INFO", f"{path.name}:{i}", f"긴 줄 ({len(raw)} chars > {max_len})")
        if raw != raw.rstrip():
            rep.add("INFO", f"{path.name}:{i}", "후행 공백")


# --------------------------------------------------------------------------- cross-file checks
LABEL_RE = re.compile(r"\\label\{([^}]+)\}")
REF_RE = re.compile(r"\\(?:ref|eqref|pageref|autoref|cref|Cref)\{([^}]+)\}")
CITE_RE = re.compile(r"\\cite[tp]?\*?(?:\[[^\]]*\]){0,2}\{([^}]+)\}")
BIBKEY_RE = re.compile(r"^\s*@[A-Za-z]+\s*\{\s*([^,\s]+)\s*,", re.M)
INPUT_RE = re.compile(r"\\(?:input|include)\{([^}]+)\}")
GRAPHICS_RE = re.compile(r"\\includegraphics(?:\[[^\]]*\])?\{([^}]+)\}")
TODO_RE = re.compile(r"\\TODO\{")


def extract_braced(s: str, start: int) -> str:
    """s[start] 가 '{' 일 때 대응하는 '}' 까지의 내용을 반환 (중첩 허용)."""
    depth = 0
    i = start
    buf = []
    while i < len(s):
        c = s[i]
        if c == "\\" and i + 1 < len(s):
            buf.append(s[i:i + 2])
            i += 2
            continue
        if c == "{":
            depth += 1
            if depth == 1:
                i += 1
                continue
        elif c == "}":
            depth -= 1
            if depth == 0:
                break
        buf.append(c)
        i += 1
    return "".join(buf)


def todo_inventory(files: list[Path]) -> list[tuple[str, int, str]]:
    out = []
    for f in files:
        text = f.read_text(encoding="utf-8")
        for m in TODO_RE.finditer(text):
            ln = text.count("\n", 0, m.start()) + 1
            body = extract_braced(text, m.end() - 1)
            out.append((f.name, ln, " ".join(body.split())))
    return out


def check_cross(paper: Path, main: Path, sections: list[Path], rep: Report) -> None:
    all_files = [main] + sections
    texts = {f: "\n".join(l for _, l in code_lines(f)) for f in all_files}

    # \input 대상 존재 + 모든 sections/*.tex 가 main 에서 호출되는지
    inputs = INPUT_RE.findall(texts[main])
    input_targets = set()
    for tgt in inputs:
        p = paper / tgt
        cand = p if p.suffix == ".tex" else p.with_suffix(".tex")
        if not cand.exists():
            rep.add("ERROR", "main.tex", f"\\input{{{tgt}}} 대상 파일 없음")
        else:
            input_targets.add(cand.resolve())
    for s in sections:
        if s.resolve() not in input_targets:
            rep.add("WARN", s.name, "main.tex 에서 \\input 되지 않음 (고아 fragment)")

    # 라벨 / 참조
    labels: dict[str, list[str]] = defaultdict(list)
    refs: dict[str, list[str]] = defaultdict(list)
    for f, t in texts.items():
        for ln, l in code_lines(f):
            for m in LABEL_RE.finditer(l):
                labels[m.group(1)].append(f"{f.name}:{ln}")
            for m in REF_RE.finditer(l):
                for key in m.group(1).split(","):
                    refs[key.strip()].append(f"{f.name}:{ln}")
    for k, locs in labels.items():
        if len(locs) > 1:
            rep.add("ERROR", locs[1], f"라벨 중복 \\label{{{k}}} (first at {locs[0]})")
    for k, locs in refs.items():
        if k not in labels:
            rep.add("ERROR", locs[0], f"\\ref{{{k}}} — 정의된 \\label 없음 ([?] 출력)")
    for k, locs in labels.items():
        if k not in refs:
            # 표/그림은 본문에서 반드시 언급되어야 함 (Elsevier) → WARN; 절/수식 라벨 미참조는 INFO
            sev = "WARN" if k.startswith(("tab:", "fig:")) else "INFO"
            rep.add(sev, locs[0], f"\\label{{{k}}} 이 본문 어디에서도 참조되지 않음 (orphan)")

    # 인용 / bib
    bib = paper / "refs.bib"
    bibkeys = set(BIBKEY_RE.findall(bib.read_text(encoding="utf-8"))) if bib.exists() else set()
    if not bib.exists():
        rep.add("ERROR", "refs.bib", "파일 없음")
    cited: dict[str, list[str]] = defaultdict(list)
    for f in all_files:
        for ln, l in code_lines(f):
            for m in CITE_RE.finditer(l):
                for key in m.group(1).split(","):
                    cited[key.strip()].append(f"{f.name}:{ln}")
    bib_fields: dict[str, set[str]] = {}
    if bib.exists():
        for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,(.*?)(?=^\s*@|\Z)", bib.read_text(encoding="utf-8"), re.S | re.M):
            bib_fields[m.group(2)] = {f.lower() for f in re.findall(r"^\s*([A-Za-z\-]+)\s*=", m.group(3), re.M)}
    for k, locs in cited.items():
        if k not in bibkeys:
            rep.add("ERROR", locs[0], f"\\cite{{{k}}} — refs.bib 에 키 없음 ([?] 출력)")
        elif "TODO" in k.upper():
            fs = bib_fields.get(k, set())
            complete = {"title", "year"} <= fs and ({"doi", "url", "howpublished", "journal"} & fs)
            if complete:
                rep.add("INFO", locs[0], f"인용 키 {k} 는 이름만 placeholder (서지 필드는 갖춰짐) — 투고 전 키 rename + \\cite 일괄 교체 고려")
            else:
                rep.add("WARN", locs[0], f"placeholder 인용 키 {k} — 서지 정보 미완(필드: {sorted(fs)}); 투고 전 확정")
    for k in sorted(bibkeys - set(cited)):
        rep.add("WARN", "refs.bib", f"bib 엔트리 {k} 가 본문에서 인용되지 않음 (elsarticle-num 은 미인용 항목을 출력하지 않음)")
    if bib.exists():
        # elsarticle-num.bst 는 ARTICLE-NUMBER 필드를 읽지 않음 → pages 가 없으면 쪽/논문번호 없이 출력됨
        for m in re.finditer(r"@(\w+)\s*\{\s*([^,\s]+)\s*,(.*?)(?=^\s*@|\Z)", bib.read_text(encoding="utf-8"), re.S | re.M):
            etype, key, body = m.group(1).lower(), m.group(2), m.group(3)
            fields = {f.lower() for f in re.findall(r"^\s*([A-Za-z\-]+)\s*=", body, re.M)}
            if etype == "article" and "article-number" in fields and "pages" not in fields:
                rep.add("INFO", "refs.bib", f"{key}: ARTICLE-NUMBER 만 있고 pages 없음 — elsarticle-num 은 article number 를 출력하지 않음 (pages 또는 note 로 이관)")
            if etype in ("article", "inproceedings") and "doi" not in fields:
                rep.add("INFO", "refs.bib", f"{key}: doi 필드 없음 (Elsevier: DOI 기재 권장)")

    # 그림 파일
    for f, t in texts.items():
        for ln, l in code_lines(f):
            for m in GRAPHICS_RE.finditer(l):
                g = paper / m.group(1)
                if not (g.exists() or any((paper / (m.group(1) + ext)).exists() for ext in (".pdf", ".png", ".jpg", ".eps"))):
                    rep.add("ERROR", f"{f.name}:{ln}", f"\\includegraphics 파일 없음: {m.group(1)}")

    # main.tex 필수 요소 (fragment 에 preamble 금지)
    mt = texts[main]
    if "\\documentclass" not in mt:
        rep.add("ERROR", "main.tex", "\\documentclass 없음")
    elif "elsarticle" not in mt:
        rep.add("WARN", "main.tex", "elsarticle 클래스가 아님 (Additive Manufacturing 은 Elsevier elsarticle)")
    if not re.search(r"\\journal\{Additive Manufacturing\}", mt):
        rep.add("WARN", "main.tex", "\\journal{Additive Manufacturing} 없음")
    if not re.search(r"\\bibliographystyle\{elsarticle-num\}", mt):
        rep.add("WARN", "main.tex", "\\bibliographystyle{elsarticle-num} 아님 (AM 저널은 numbered reference)")
    for s in sections:
        st = texts[s]
        for bad in ("\\documentclass", "\\usepackage", "\\begin{document}", "\\end{document}"):
            if bad in st:
                rep.add("ERROR", s.name, f"fragment 안에 {bad} 가 있음")
    # 영문 투고본: kotex 없이 한글이 남아 있으면 pdfLaTeX 컴파일 실패 + 언어 규정 위반
    has_kotex = re.search(r"\\usepackage(\[[^\]]*\])?\{[^}]*kotex", mt) is not None
    if not has_kotex:
        for f, t in texts.items():
            n = len(HANGUL.findall(t))
            if n:
                rep.add("ERROR", f.name, f"kotex 미로드 상태에서 한글 {n}자 잔존 (영문 투고본이면 번역 누락)")
    m_ead = re.search(r"\\ead\{([^}]*)\}", mt)
    if m_ead and re.search(r"@(gmail|163)\.", m_ead.group(1)):
        rep.add("INFO", "main.tex", f"교신저자 이메일 {m_ead.group(1)} — AM 저널은 기관 이메일 요구(gmail/163 불가, 투고 전 교체)")
    # 매크로 정의 확인
    defined = set(re.findall(r"\\(?:newcommand|renewcommand|providecommand|DeclareMathOperator\*?)\{?\\([A-Za-z]+)", mt))
    for mac in ("TODO", "argmax"):
        used_in = [s.name for s in sections if f"\\{mac}" in texts[s]]
        if used_in and mac not in defined:
            rep.add("ERROR", "main.tex", f"\\{mac} 가 {used_in} 에서 쓰이지만 main.tex 에 정의 없음")

    # Elsevier front/end matter
    for name, rx, sev in ELSEVIER_ENDMATTER:
        if not re.search(rx, mt):
            rep.add(sev, "main.tex", f"끝맺음 절 누락: {name}")
    hl = re.search(r"\\begin\{highlights\}(.*?)\\end\{highlights\}", mt, re.S)
    if hl:
        items = [strip_tex_cmds(x) for x in re.split(r"\\item", hl.group(1))[1:]]
        if not 3 <= len(items) <= 5:
            rep.add("WARN", "main.tex", f"Highlights 개수 {len(items)} (Elsevier: 3–5 개)")
        for i, it in enumerate(items, 1):
            if len(it) > 85:
                rep.add("WARN", "main.tex", f"Highlight {i} 길이 {len(it)} chars > 85 (영문 기준 규정; 한글판은 영문 원문 길이로 판단)")
        # elsarticle >= 3.3 은 highlights/graphicalabstract 환경을 자체 정의하고 \end{frontmatter} 의 \maketitle 에서만
        # 예비 페이지로 출력한다 → frontmatter 밖에 두면 (자체 폴백이 있어도) PDF 에서 조용히 사라짐
        fm_end = mt.find("\\end{frontmatter}")
        if fm_end != -1 and hl.start() > fm_end:
            rep.add("WARN", "main.tex", "\\begin{highlights} 가 \\end{frontmatter} 뒤에 있음 — elsarticle ≥3.3(Overleaf) 에서는 출력되지 않음; "
                                        "\\end{abstract} 와 \\begin{keyword} 사이(frontmatter 안)로 옮기고 자체 폴백 정의는 제거")
    else:
        rep.add("INFO", "main.tex", "highlights 환경 없음 (AM 저널: 선택 사항이나 별도 파일 제출 권장 — 원고에 두면 elsarticle 이 예비 페이지로 출력)")
    kw = re.search(r"\\begin\{keyword\}(.*?)\\end\{keyword\}", mt, re.S)
    if kw:
        n = len([k for k in re.split(r"\\sep", kw.group(1)) if k.strip()])
        if n > 6:
            rep.add("WARN", "main.tex", f"keyword {n} 개 > 6 (Elsevier 최대 6)")
    else:
        rep.add("WARN", "main.tex", "keyword 환경 없음")
    ab = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", mt, re.S)
    if ab:
        body = strip_tex_cmds(ab.group(1))
        words = len(body.split())
        chars = len(body.replace(" ", ""))
        if "\\cite" in ab.group(1):
            rep.add("WARN", "main.tex", "abstract 안에 \\cite 사용 — Elsevier 는 초록 내 인용 지양")
        rep.add("INFO", "main.tex", f"abstract 길이: {words} 어절 / {chars} 자 (영문 투고본 기준 Elsevier 권장 상한 확인 필요)")
    if "\\TODO" in mt or any("\\TODO" in texts[s] for s in sections):
        n = sum(len(TODO_RE.findall(t)) for t in texts.values())
        rep.add("WARN", "paper", f"\\TODO 잔존 {n} 개 (투고 전 0 개여야 함) — `--todo` 로 목록 확인")
    if re.search(r"\\linenumbers", mt) is None and "review" in (re.search(r"\\documentclass\[([^\]]*)\]", mt) or [None, ""])[1]:
        rep.add("INFO", "main.tex", "review 옵션이면 \\linenumbers 권장")


# --------------------------------------------------------------------------- style stats
def style_stats(sections: list[Path], rep: Report) -> None:
    tot = Counter()
    per_file: dict[str, Counter] = {}
    for s in sections:
        c = Counter()
        for _, l in code_lines(s):
            c["Section~\\ref"] += len(re.findall(r"Section~\\ref", l))
            c["\\ref{..}절"] += len(re.findall(r"\\ref\{[^}]+\}절", l))
            c["Fig.~\\ref"] += len(re.findall(r"Fig\.~\\ref", l))
            c["그림~\\ref"] += len(re.findall(r"그림~\\ref", l))
            c["Table~\\ref"] += len(re.findall(r"Table~\\ref", l))
            c["표~\\ref"] += len(re.findall(r"표~\\ref", l))
            c["$..$"] += len(re.findall(r"(?<!\\)\$[^$]+\$", l))
            c["\\(..\\)"] += len(re.findall(r"\\\(", l))
            c["라틴어 조사 띄움 (X 은)"] += len(re.findall(rf"[A-Za-z0-9)\]] +(?:{PARTICLES})(?=[\s,.)])", l))
            c["라틴어 조사 붙임 (X은)"] += len(re.findall(rf"[A-Za-z0-9)\]](?:{PARTICLES})(?=[\s,.)])", l))
            c["\\,단위"] += len(re.findall(r"\d\\,[A-Za-z\\%]", l))
            c["~단위"] += len(re.findall(r"\d~[A-Za-z]", l))
        per_file[s.name] = c
        tot.update(c)
    pairs = [("Section~\\ref", "\\ref{..}절"), ("Fig.~\\ref", "그림~\\ref"), ("Table~\\ref", "표~\\ref"),
             ("$..$", "\\(..\\)"), ("라틴어 조사 띄움 (X 은)", "라틴어 조사 붙임 (X은)"), ("\\,단위", "~단위")]
    for a, b in pairs:
        detail = ", ".join(f"{n}: {c[a]}/{c[b]}" for n, c in per_file.items() if c[a] or c[b])
        rep.add("INFO", "style", f"{a} vs {b} = {tot[a]}/{tot[b]}  [{detail}]")


# --------------------------------------------------------------------------- hook mode
def find_paper_dir(p: Path) -> Path | None:
    """편집된 파일에서 위로 올라가며 main.tex + sections/ 를 가진 원고 루트를 찾는다 (Ko/영문판 공용)."""
    for anc in p.parents:
        if (anc / "main.tex").exists() and (anc / "sections").is_dir():
            return anc
    return None


def run_hook(paper: Path, main_tex: Path) -> int:
    """PostToolUse(Edit|Write) 훅. 편집된 파일이 원고(.tex/.bib) 일 때만 ERROR 항목을 stderr 로 돌려준다."""
    import json
    try:
        payload = json.load(sys.stdin)
    except Exception:
        return 0
    ti = payload.get("tool_input") or {}
    cwd = Path(payload.get("cwd") or ".")
    candidates: list[Path] = []
    fp = ti.get("file_path") or ""          # Edit / Write / MultiEdit
    if fp:
        candidates.append(Path(fp))
    cmd = ti.get("command") or ""           # Bash: sed -i / heredoc / python 으로 원고를 건드린 경우
    for tok in re.findall(r"[^\s'\"<>|;&()]+\.(?:tex|bib)\b", cmd):
        candidates.append(Path(tok))
    papers: dict[Path, None] = {}
    for c in candidates:
        c = (c if c.is_absolute() else cwd / c).resolve()
        if c.suffix not in (".tex", ".bib"):
            continue
        found = find_paper_dir(c)
        if found is not None:
            papers[found] = None
    if not papers:
        return 0
    all_errors: list[tuple[str, str, str]] = []
    for paper in papers:
        main_tex = paper / "main.tex"
        sections = sorted((paper / "sections").glob("*.tex"))
        rep = Report()
        for f in [main_tex] + sections:
            check_braces(f, rep)
            check_envs(f, rep, is_main=(f == main_tex))
            check_math_delims(f, rep)
            check_special_chars(f, rep)
        check_cross(paper, main_tex, sections, rep)
        all_errors += [(paper.name, loc, msg) for sev, loc, msg in rep.items if sev == "ERROR"]
    if not all_errors:
        return 0
    print(f"check_paper_tex --hook: 편집 후 ERROR {len(all_errors)} 개 — 컴파일/참조가 깨질 항목이므로 바로 수정할 것", file=sys.stderr)
    for pname, loc, msg in all_errors[:30]:
        print(f"  [{pname}] {loc}: {msg}", file=sys.stderr)
    return 2


# --------------------------------------------------------------------------- main
def main(argv: list[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--paper-dir", type=Path, default=DEFAULT_PAPER_DIR)
    ap.add_argument("--no-info", action="store_true", help="INFO 항목 숨김")
    ap.add_argument("--todo", action="store_true", help="\\TODO 인벤토리만 출력")
    ap.add_argument("--max-line", type=int, default=100)
    ap.add_argument("--hook", action="store_true",
                    help="Claude Code PostToolUse 훅 모드: stdin 의 JSON(tool_input.file_path) 이 원고 파일이면 점검하고 "
                         "ERROR 가 있을 때만 stderr 로 보고하며 exit 2 (그 외 무출력 exit 0)")
    args = ap.parse_args(argv)

    paper: Path = args.paper_dir.resolve()
    main_tex = paper / "main.tex"
    if args.hook:
        return run_hook(paper, main_tex)
    if not main_tex.exists():
        print(f"main.tex 없음: {main_tex}", file=sys.stderr)
        return 2
    sections = sorted((paper / "sections").glob("*.tex"))

    if args.todo:
        inv = todo_inventory([main_tex] + sections)
        print(f"\\TODO 인벤토리 — {len(inv)} 개")
        for fn, ln, body in inv:
            print(f"  {fn}:{ln}  {body[:110]}{'…' if len(body) > 110 else ''}")
        return 0

    rep = Report()
    for f in [main_tex] + sections:
        check_braces(f, rep)
        check_envs(f, rep, is_main=(f == main_tex))
        check_math_delims(f, rep)
        check_special_chars(f, rep)
        check_non_ascii(f, rep)
        check_lines(f, rep, args.max_line)
    check_cross(paper, main_tex, sections, rep)
    style_stats(sections, rep)

    order = {"ERROR": 0, "WARN": 1, "INFO": 2}
    for sev in ("ERROR", "WARN", "INFO"):
        if sev == "INFO" and args.no_info:
            continue
        items = [it for it in rep.items if it[0] == sev]
        print(f"\n== {sev} ({len(items)}) ==")
        for _, loc, msg in items:
            print(f"  [{sev}] {loc}: {msg}")
    print(f"\n요약: ERROR {rep.count('ERROR')} / WARN {rep.count('WARN')} / INFO {rep.count('INFO')}"
          f"  (paper: {paper})")
    return 1 if rep.count("ERROR") else 0


if __name__ == "__main__":
    sys.exit(main())
