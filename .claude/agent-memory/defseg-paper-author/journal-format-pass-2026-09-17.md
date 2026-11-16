---
name: journal-format-pass-2026-09-17
description: Decisions made in the 2026-09-17 Additive Manufacturing format pass on the Korean manuscript (end-matter headings in English, highlights inside frontmatter, bib conventions) and what still awaits the user
metadata:
  type: project
---

2026-09-17 format pass (user request: "AM 저널 형식으로, 한글 본문 유지, 나중에 영문 번역").

Decisions applied (do not re-litigate unless the user asks):
- Journal-mandated end-matter headings are English-only (`CRediT authorship contribution statement`, `Declaration of competing interest`, `Data availability`, `Acknowledgements`, `Funding`, `Declaration of generative AI and AI-assisted technologies in the manuscript preparation process`); prose under them stays Korean. Body section titles stay Korean (1 서론 … 7 결론) — user said keep Korean body; structure (Intro / Related / Method / Setup / Results / Discussion / Conclusion) kept as-is.
- Highlights live between `\end{abstract}` and `\begin{keyword}`; the old fallback environment is gone. English wording + lengths are kept in a `%` comment above the environment for the future translation.
- Non-standard abbreviations are listed in a `\tnotetext[abbr]` title footnote (Elsevier first-page footnote rule).
- refs.bib: journal names use LTWA abbreviations; article numbers stored in `PAGES`; proper nouns brace-protected. Snow2021Layerwise was corrected to J. Manuf. Syst. 59 (2021) 12–26 / 10.1016/j.jmsy.2021.01.008 (old DOI pointed to an unrelated paper). ORNL_CoRegistered_Dataset now has DOI 10.13139/ORNLNCCS/2001425 and the landing-page author list (6 authors).
- `\biboptions{sort&compress}` is deliberately NOT used: the static checker flags the `&`.

**Why:** the user wants a Korean review copy that already satisfies the journal's structural rules so the later English translation is mechanical.

**How to apply:** keep this layout in future edits; when producing the English version, reuse the commented English highlights and the abbreviation footnote.

Still waiting on the user (left as `\TODO` or unchanged):
- affiliation/city/country, institutional e-mail (gmail must go), co-author list and full CRediT roles, funding sentence, acknowledgements, AI-tool name/version/purpose for the AI declaration, code/checkpoint release statement.
- `Peregrine_v2025_TODO` key rename (bib fields are complete) — not renamed.
- Wording of the pseudo-label teacher ("production DSCNN" vs the HDF5 change_log's "DMSCNN" 2023-08-20 model) — untouched.
- Whether keyword "Laser powder bed fusion" should become the strict ISO/ASTM 52900 wording in the English version.
