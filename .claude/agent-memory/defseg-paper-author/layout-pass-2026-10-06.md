---
name: layout-pass-2026-10-06
description: Float/overfull layout pass on the Korean manuscript (2026-10-06) — local TeX Live now exists and is the verification path; table/figure conventions adopted (footnotesize + tabularx + makecell, [tp] for page-filling floats, topfraction 0.9, emergencystretch 2em, §3.2 photo figures grouped + \clearpage); what the user may still want to change
metadata:
  type: project
---

2026-10-06 layout pass (request via main session: tables/figures were piling up after the references in the 69-page PDF; fix and verify by local compile, no commit).

Facts that override older notes:
- **Local LaTeX exists**: TeX Live 2026 in `~/.local/bin` (pdflatex, latexmk, bibtex) plus `pdftotext`/`pdftoppm`. Verify with `cd <paper dir> && latexmk -pdf -interaction=nonstopmode -file-line-error main.tex`; the agent definition's "로컬 툴체인 없음" is stale. The shell's `grep` is a ugrep alias — use `/usr/bin/grep -a` on `.log` files.
- Geometry: text block 390 x 548.5 pt (letter, 12pt), captions `\footnotesize`, default `\topfraction` 0.7 => any `[t]` float taller than 384 pt was stranded (with every later float of the same kind) until `\clearpage` at the end. That — one 135 pt-too-tall tab:taxonomy — was the whole "뒤죽박죽" cause.

Decisions applied (keep unless the user overrides):
- Preamble (`main.tex`): `\usepackage{makecell}`; `\setlength{\emergencystretch}{2em}` (kotex Hangul glue has no stretch, so pdfTeX otherwise emits Overfull lines in plain Korean paragraphs); `\renewcommand{\topfraction}{0.9}`, `\renewcommand{\textfraction}{0.1}`.
- Every table is `\footnotesize`; wide ones are `tabularx{\textwidth}` with `>{\raggedright\arraybackslash}X` columns and `\makecell{..\\..}` two-line headings; `\tabcolsep` 3pt where needed. Page-filling floats use `[tp]` (tab:taxonomy, tab:datasets, fig:qualitative-dscnn with `height=0.80\textheight,keepaspectratio`); all other floats stay `[t]`. No `\resizebox`, no rotated tables, glyphs never below 10 pt.
- Table blocks sit directly after the paragraph that first references them (tab:classes, tab:taxonomy, tab:method-hparams, tab:datasets were moved). Rule stays: a float block goes after, never before, its first `\ref`.
- §3.2: the eight `[p]` example-photo figures are grouped after the last body paragraph of the subsection and followed by `\clearpage`, so they print as eight consecutive pages between §3.2 and §3.3 (user-specified). tab:taxonomy's long footnote block became a body paragraph right after the first-reference paragraph (content verbatim; "(본문 참조)" -> "(이 소절의 끝 참조)"); the duplicate sentence "이미지 열은 저자들의 관찰에 따른 것이다." was dropped and its `\TODO` moved to the end of the new paragraph; the caption lost the column explanations (now in that paragraph) and is 4 lines.
- Unbreakable tokens get break points instead of rewording: `\texttt{slices/\allowbreak camera\_data/\allowbreak visible/0}`, `\texttt{slices/\allowbreak segmentation\_\allowbreak results}`, `fusion\slash decoder\slash head`.

**Why:** LaTeX places same-type floats strictly in order, so one oversized table pushed Tables 3–11 (and Figure 10 pushed Figure 11) behind the references; the user wants the Korean review PDF readable without an Overleaf round-trip.

**How to apply:** after any table/figure edit, recompile locally and require: `Float too large` 0, `Overfull \hbox` > 5 pt 0, undefined refs 0, no floats after References (pdftotext page map). New tables: `\footnotesize`, tabularx when wider than 390 pt, `[tp]` only when the float fills most of a page (> ~440 pt); never leave a `[t]` float taller than ~490 pt.

Still the user's call: whether eight figure pages between §3.2 and §3.3 are acceptable (alternative: appendix — explicitly out of scope on 2026-10-06); whether the trimmed tab:taxonomy caption and the table-notes paragraph wording are fine; the `.claude/agent-memory/` directory that appeared inside `paper/DefectSegAM_Ko/` (empty, harness-created) — memory is kept only at `DefSeg_AM/.claude/agent-memory/`.

Related: [[taxonomy-section-2026-09-28]], [[journal-format-pass-2026-09-17]].
