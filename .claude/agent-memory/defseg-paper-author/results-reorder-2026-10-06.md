---
name: results-reorder-2026-10-06
description: §5 결과 재편 결정 (2026-10-06) — 완료 결과(일치도·다운스트림·효율) 먼저, 미수행 세 평가는 5.4 sec:results-planned 아래 subsubsection; cross-machine 세트 = Veltz/EOS 만(2025 Peregrine 은 held-out 전용); 4장은 결과 표 대신 결과 절을 참조
metadata:
  type: project
---

2026-10-06 §5 reorder (user decision relayed by the main session; verified by local `latexmk`: 69 pages, LaTeX Error / Float too large / Overfull / undefined all 0, no float after References, linter ERROR 0, `\TODO` stays 38; no commit).

Decisions applied (keep unless the user overrides):
- §5 order is now 5.1 `sec:results-agreement` -> 5.2 `sec:results-downstream` -> 5.3 `sec:results-efficiency` -> 5.4 `sec:results-planned` ("계획된 평가: held-out 사람 GT 비교, cross-machine 일반화, ablation") containing `\subsubsection`s `sec:results-heldout` / `sec:results-crossmachine` / `sec:results-ablation` with their scaffold tables. All old labels kept; only the 5.4 label is new. §5 intro now says "네 부분" (three completed + one planned); 5.4 opens with the protocol/scaffold sentences moved from the old intro.
- Table numbering after the move: 5 agreement-miou, 6 agreement-perclass, 7 downstream, 8 efficiency, 9 heldout-main, 10 crossmachine, 11 ablation. 4장 therefore references the result *sections* (`\ref{sec:results-heldout}절`, `-crossmachine`, `-ablation`) instead of those tables, so first mentions stay 1..11 (same rule as 1장/3장 in [[structure-pass-2026-10-06]]).
- Cross-machine set definition follows §4.2.2 only: Veltz M160/M280 + EOS PBF (machines absent from every training corpus). The 2025 Peregrine builds (Concept Laser M2, ExOne Innovent — both already training-corpus machine types) are held-out only; the two 2025 Peregrine rows were removed from `tab:crossmachine` and the §5 cross-machine prose now says so explicitly. The cross-machine `\TODO` says "세 개 장비 부분집합" (was 다섯 개). `Peregrine_v2025_TODO` is still cited in main.tex, 1장, 4장, 5.4.1.
- Order-dependent wording fixed only in 5장 ("위의 held-out 평가" -> "\ref{sec:results-heldout}절의 held-out 평가"); no literal "5.N절" references exist anywhere.

**Why:** the user wanted completed measurements to lead §5 and the three pending experiments grouped as one clearly-labelled planned block; the 2025 Peregrine machines contradicted the §4.2.2 cross-machine definition.

**How to apply:** new forward references from 1장-4장 to §5 must point at sections, never at Tables 5-11; when the held-out/cross-machine/ablation results arrive, fill the 5.4.x scaffolds in place (keep the labels) and consider promoting them back to `\subsection`s — that is the user's call. Open points for the user: whether the EOS cross-machine machine (`\TODO{모델명}`) is a different model/camera from the Stage-2 EOS M290 (otherwise §4.2.2's "어떠한 학습 코퍼스에도 등장하지 않는" claim weakens); 7장's `\TODO{\ref{sec:results}절에 스캐폴딩된 …}` could point to `sec:results-planned` (left untouched — TODO text is frozen).

Related: [[structure-pass-2026-10-06]], [[layout-pass-2026-10-06]].
