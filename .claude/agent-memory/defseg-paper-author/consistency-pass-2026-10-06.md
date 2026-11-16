---
name: consistency-pass-2026-10-06
description: Cross-reference/consistency clean-up after the 2026-10-06 reorder+dedup passes — new single-source homes (Debris 90 % in §3.1, "85 annotated layers" in §4.1.2, pseudo-label conversion mechanics only in §4.1.1), pointer targets changed (§6.2 -> §4.4 + Table 4, §3.2 Soot -> §4.1.2, Under Melting TODO -> sec:setup-stage2-corpus), §1/§2 wording now in force, and the two small deviations from the review's literal text
metadata:
  type: project
---

2026-10-06 consistency pass (15 findings of a read-only review relayed by the main session; verified by local `latexmk`: 66 pages (was 67), LaTeX Error / Float too large / Overfull / undefined 0, References on p. 59 with no float after it, Tables and Figures first-mentioned 1..11, linter ERROR 0, `\TODO` stays 38, all 58 bib keys cited; no commit).

Homes / pointers now in force (keep when editing):
- Build concentration: §3.1 now lists all four (Over Melting 99.8 %, Under Melting 99.9 % -> B1.2; Super-Elevation 98 % -> B1.5; Debris 90 % -> B1.3). Debris 90 % = 1,757,892 px with 90.4 % in B3 (`cache/resized_sz1036_v4/B*/pix_per_layer.npy`, class 6). §6.3 keeps its restatement; nowhere else.
- "주석이 있는 85개 레이어" (six v2022-10 sources: 14+9+5+24+26+7) has its home in §4.1.2 ("릴리스별로는 v2021-03의 두 소스에서 40개, v2022-10의 여섯 소스에서 85개"); §3.2 only points.
- Pseudo-label mechanics (12-channel argmax -> hard label -> 11-class remap, Misprint ignore) live only in §4.1.1 + tab:datasets caption. §3.4 Stage 1 keeps the KD framing ("hard 의사 라벨 … hard-label KD, production DSCNN = teacher") and points to `sec:setup-stage1-corpus`. §3.4 Stage 2 no longer lists the ignored native classes (Soot/Crashing/Stripe Boundary — §4.1.2 only); its sampler sentence is a bare pointer ("소스 균형 샘플링을 사용한다") and the weighted-CE sentence ends with "(근거는 §3.5)" — the "no focusing term because small, resampled corpus" reason is stated in §3.5 only.
- §3.2's Soot sentence now points to `sec:setup-stage2-corpus` (not §3.4) because that is the only place naming Soot as ignored.
- §6.2 cites "§4.4 and Table 4" (not §5.1) for "model selection uses DSCNN-derived labels in both stages". §6.1's derived pixel counts carry "(레이어 수는 §4.1, 모델 해상도 1036×1036은 §3.3)"; 6장 keeps 띄움 after math (`$1036 \times 1036$ 은`).
- §4.5: the TTA/post-processing default sentences ("8-way TTA와 두 후처리 규칙은 §3.6에 기술 / 별도 명시 없으면 단일 forward / 개별 기여는 §5.4.3 ablation") are two sentences at the end of the hardware paragraph, before the hyperparameter-range `\TODO`; the one-sentence paragraph after tab:method-hparams is gone, so the table is the last block of §4.5.
- §7 third finding: "계산 효율 프로파일은 학습 가능 파라미터 수와 학습 메모리 측면에서 frozen 인코더 설계의 비용 구조를 확인하지만, … 추론 지연시간은 아직 측정되지 않았다" (no checkpoint-content or 4-GPU-node claims — §5.3 does not state them).
- §2.5: "DefectSegAM은 이 세 가지 결핍을 다룬다. 즉, …평가한다. 아울러 §2.1에서 지적한 … 분류 체계를 §3.2에서 제안한다." (taxonomy clause last, so "즉" follows its antecedent).
- §2.1 ASTM sentence = "ASTM E3353-22 표준 지침의 이상 범주 역시 원인 기준이다(상세 대응은 §3.2)." The §4-vs-§8 detail of the standard lives in §3.2 only.
- §1 ¶4 taxonomy sentence = "아울러 향후 어노테이션을 위한 외형 기준(appearance-based) 결함 분류 체계를 제안한다(§3.2)." — no `\cite{ASTME3353}` there any more (first citation of ASTME3353 is now §2.1; contribution 5 keeps the full wording and the "실험과 별도로" disclaimer). §1 roadmap now names §2.5 ("연구 공백을 종합한다") and the full §4 list (평가 지표, 구현 세부사항, 다운스트림 설정).
- §5.4.1: "평가 세트는 … 2025 Peregrine held-out 세트이며, 베이스라인은 …의 설정을 따른다" (two predicates). §5.4.2: 2025 Peregrine exclusion is "같은 절에서 밝힌 이유로 포함하지 않는다" (pointer only; the reason lives in §4.2.2).
- Under Melting `\TODO` in §3.2 points to `sec:setup-stage2-corpus` (label update only).

Deviations from the review's literal text (small, reported in the handoff): (a) §2.1's closing sentence lost its tail "기존 어휘 및 ASTM 용어와의 상세 대응도 그곳에 정리한다" because the new ASTM sentence already carries "(상세 대응은 §3.2)"; (b) §5.4.2 uses "같은 절에서 밝힌 이유로" instead of a second `\ref{sec:setup-heldout-crossmachine}` in the same sentence.

Left untouched on purpose: the three remaining restatements of 0.8643 / 25픽셀 (user said keep), every `\TODO` wording, all numbers.

**Why:** the review found pointers to content that had moved or been deleted in the same-day dedup passes (table footnote, §5.1 "emphasis", §5.3 claims) and facts without a single home.

**How to apply:** when adding a pointer, grep that the target section actually states the fact; when a number appears in §3/§6, make sure §4 (or §3.1 for class statistics) is its home.

Related: [[dedup-pass-3-2026-10-06]], [[dedup-pass-2-2026-10-06]], [[dedup-pass-1-2026-10-06]], [[results-reorder-2026-10-06]].
