---
name: structure-pass-2026-10-06
description: Ordering/cross-reference pass on the Korean manuscript (2026-10-06, after the layout pass) — §3.2 internal order, table first-mention rule, new section labels (sec:related-gap, sec:setup-heldout-peregrine/-crossmachine), §2/§6 roadmap paragraphs, \paragraph titles without trailing period, §5.1/§5.5 condensation, and what was left for the user
metadata:
  type: project
---

2026-10-06 structure pass (main session asked to apply only the "safe now" ordering and cross-reference fixes from a read-only review; no commit, no `_backup/`, verified by local `latexmk`: 69 pages, LaTeX Error / Float too large / Overfull / undefined all 0, no float after References, linter ERROR 0, `\TODO` stays 38).

Decisions applied (keep unless the user overrides):
- §3.2 body order is now: motivation -> tab:taxonomy notes paragraph + table -> changes (i)-(v) -> discrimination rules -> remap/data-availability paragraph -> the four example-photo paragraphs (the closing "이 체계는 차기 주석과 학습을 위한 제안이며 …" sentence stays glued to the fourth) -> eight `[p]` figures + `\clearpage`. Reason: the photo paragraphs presuppose the rules ("제안 체계의 판별 규칙에서 …"). The notes-paragraph pointer "(이 소절의 끝 참조)" became "(아래 참조)".
- Table first-mention rule: tables must be first `\ref`'d in numeric order (1..11). 1장 therefore points to result *sections* (`Section~\ref{sec:results-heldout}` / `-crossmachine` / `-downstream`), never to result tables; 3장 points to `\S\ref{sec:setup-stage2-corpus}` instead of tab:datasets. Keep this when adding new forward references in 1장-3장. Figures were already monotone.
- New labels: `sec:related-gap` (2.5, promoted from `\paragraph`), `sec:setup-heldout-peregrine` (4.2.1), `sec:setup-heldout-crossmachine` (4.2.2) — their `\label` sits on the next line like the parent `sec:setup-heldout`. §4.3's two `\paragraph`s were deliberately left as run-in heads.
- `\paragraph{…}` titles never end with "." — elsarticle's `\paragraph` appends one itself (elsarticle.cls `{#1.}`), so the 5장 trailing periods were removed.
- §2 and §6 now open with a 2-3 sentence roadmap paragraph (same style as §3-§5 intros); the 1장 roadmap lists the real §3 order (문제 설정·온톨로지 -> 분류 체계 -> 아키텍처 -> 2단계 학습 -> 불균형 -> 증강·추론·후처리); the 4장 roadmap includes §4.6.
- 3.4 no longer promises a full per-source mapping dictionary in §4: it now says representative examples are in §4.1.2 and the full dictionaries are "구현 코드에 정의되어 있다" (no `\TODO` added). If code will not be released, this sentence should become an appendix table — user decision.
- 6.3 "클래스 통계" now points to `\ref{sec:method-problem}절(표~\ref{tab:classes})`.
- §5.1 restatement of the held-out set / baselines replaced by one pointer sentence (nothing lost — all facts are in §4.2.1/§4.3). §5.5: only the supervoxel/aggregation/N/CV-split restatement was replaced by a pointer to §4.6; the MLP spec (8->128->1, L1, targets in [-1,1]), min-over-supervoxel specimen prediction, fold mean +- std reporting, naive floor and the channel-set caveat stayed in §5.5 because §4.6 does not state them (user may move them to §4.6 later).

Still the user's call (explicitly out of scope on 2026-10-06): moving §3.2 to an appendix, merging the duplicated recipe/corpus descriptions (3.4 vs 4.1/4.5), consolidating the scattered taxonomy statements, deleting the disclaimer sentences. Remaining bare `\ref{sec:setup}` pointers (3장 ×4, 5장 ×2, 1장 ×2) are correct but could be made more specific.

Related: [[layout-pass-2026-10-06]], [[taxonomy-section-2026-09-28]], [[journal-format-pass-2026-09-17]], [[results-reorder-2026-10-06]] (§5 reorder done later the same day).
