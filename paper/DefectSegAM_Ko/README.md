# DefectSegAM 논문 — 한국어 버전 (내부 검토용)

영문 원고 [`../DefectSegAM/`](../DefectSegAM/)의 **한국어 완역 미러**입니다.
지도교수·공저자 내부 검토용이며, **투고 정본(master)은 영문 버전**입니다 — 내용 수정은 영문에 먼저 반영하고 이 폴더에 동기화하세요.

## 파일 구조 (영문 버전과 1:1 대응)

```
main.tex                     # elsarticle + kotex, 제목/초록/highlights (한국어)
sections/1_introduction.tex  # 서론
sections/2_related_work.tex  # 관련 연구
sections/3_method.tex        # DefectSegAM 방법론
sections/4_experimental_setup.tex  # 데이터셋·베이스라인·지표·구현
sections/5_results.tex       # 결과 (미수행 실험은 \TODO 골격)
sections/6_discussion.tex    # 고찰·한계
sections/7_conclusion.tex    # 결론
refs.bib                     # 영문 버전에서 복사 (참고문헌은 영어 유지)
Figs/                        # 영문 버전과 동일한 임시(placeholder) 그림
```

## 규약

- **구조 동일성**: `\label`/`\cite` 키, 수식, 표 구조, `\TODO` 위치는 영문 버전과 동일. 산문만 한국어.
- **`\TODO{...}`** = 미수행 실험/미확정 정보. 어떤 표가 어떤 예정 실험(2025 Peregrine held-out, Veltz M160/M280 + EOS 라벨링, DSCNN 재구현)을 기다리는지는 영문 [`../DefectSegAM/README.md`](../DefectSegAM/README.md) 참조.
- **수치 정책**: 실수치는 다운스트림 물성 결과(UTS RMSE 70.7→37.5 등)와 DSCNN **일치도** 분석뿐. ORNL 빌드 대상 mIoU 는 human-GT 정확도가 아님.
- **컴파일**: pdfLaTeX + `kotex` (main.tex 에 포함). 로컬 TeX Live 2026(`latexmk -pdf main.tex`)과 Overleaf 둘 다 같은 파일로 컴파일된다.
- Figs/ 는 학위논문 그림 임시 재사용 — 최종본은 새로 그려야 함 (영문 README 동일 항목 참조).
