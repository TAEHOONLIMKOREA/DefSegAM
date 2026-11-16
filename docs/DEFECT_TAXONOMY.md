# DefectSegAM 결함 분류 체계 (제안)

> 출처: [`docs/결함분류 체계.pdf`](결함분류%20체계.pdf) (2026-09-28) 의 두 슬라이드를 정리하고,
> 기존 라벨(ORNL Peregrine 12-class, DSCNN_Dataset 재료별 native class, ASTM E3353) 과의
> 대응과 데이터 가용성을 붙인 문서. 작성일 2026-09-28.
>
> **상태: 제안 단계, 코드 미반영.** 현재 학습에 쓰이는 체계는 v4 11-class (`v4/config_v4.py`,
> `PLAN_v4.md` §2) 이다. 이 문서는 차기 버전의 클래스 공간을 정의하기 위한 정본이며,
> 코드 반영 시 부록 A 의 초안을 `vN/config_vN.py` 로 옮긴다.

## 0. 요약

- **원칙: 결함을 "원인" 이 아니라 "이미지에 나타난 특징" 을 판정 기준으로 분류한다.**
  위치(파트 영역 / 파우더 영역) 판정에는 파트 마스크를 보조로 쓴다 (§4.2).
  기존 데이터셋·논문 용어는 원인 기준과 외형 기준이 섞여 있어 어노테이션 기준이 애매하다 (§1).
- 클래스 수: **결함 8 종 + 배경 2 종 (Powder, Printed) = 10.**
- v4 11-class 대비 변경:
  - `Swelling` 을 `Super Elevation` 에 **흡수** (외형이 비슷해 구별이 어려움).
  - `Spatter` / `Over Melting` 을 **파우더 영역 / 파트 영역** 기준으로 명시적으로 구분.
    `Over Melting` 은 원인 기준(고에너지 밀도) 정의에서 외형 기준(파트 위 거뭇한 영역)으로
    재정의되므로 기존 라벨 재검토가 필요하다 (§7-9).
  - `Powder Collapse` **신규** (기존 라벨 없음 → 어노테이션 필요).
  - `Under Melting` 은 외형 기준이 아니라 공정 파라미터 기준 클래스 → **처리 미결** (§7).
  - `Debris` 유지. PDF 의 "데이터 확인 필요" 는 확인 완료 — 데이터 있음 (§6).

## 1. 배경 — 기존 용어의 문제

ASTM E3353-22 (*Standard Guide for In-Process Monitoring Using Optical and Thermal Methods for
Laser Powder Bed Fusion*) 의 공개된 §4 (Significance and Use; store.astm.org 미리보기, 본문 전체는
미확인) 는 분말층 이상의 원인을 리코팅 오류(skipping, scraping, insufficient powder delivery,
part strike) 와 파트 형성 오류(distortion, humping, balling, superelevation) 로 범주화하고
(§4.7.3.2), 레이어 이미지에서의 상세 기술은 비공개 8절로 미룬다. 슬라이드(PDF 1쪽)는 이 용어들을
논문·Peregrine 용어와 다음과 같이 대응시킨다.

| ASTM E3353 표준 | 논문 / Peregrine 용어 | 의미 | 특징 (슬라이드) |
|---|---|---|---|
| Skipping | Incomplete Spreading | 리코팅이 일부 영역을 제대로 덮지 못함 | 원인이 비슷함 |
| Insufficient powder delivery | Incomplete Spreading | 공급 분말 부족 | (위와 동일) |
| Scraping | Recoater Streaking | 리코터가 파트와 접촉하며 긁음 | — |
| Part strike | Recoater Hopping | 리코터와 돌출된 파트가 충돌 | — |
| Distortion | Swelling / Super-elevation | 파트의 변형 (warping) | 현상이 모두 비슷함 |
| Humping | Swelling | 용융부가 위쪽으로 솟음 | (위와 동일) |
| Superelevation | Super-elevation | 파트가 powder surface 보다 돌출 | (위와 동일) |
| Balling | — | melt track 이 불연속적인 구형으로 형상화 | — (비고: Peregrine 에는 v2021-03 EB-PBF 소스에만 Balling 클래스가 있고 L-PBF 어휘에는 없음) |
| Excessive spatter / ejecta | Spatter | 용융 중 과도한 입자 방출 | — |

슬라이드의 문제 제기는 "결함 분류 기준이 애매함 (이미지에서 나타나는 특징과 원인 기준이 섞여
있음)" 이다. 확인된 사실은 다음과 같다.
- 표준의 공개부는 skipping 과 insufficient powder delivery 를 원인 범주로 나열할 뿐 둘의 레이어
  이미지 외형 차이를 제시하지 않고, ORNL·DSCNN 코퍼스도 둘을 구분해 라벨하지 않는다.
- Incomplete Spreading 의 정의("분말 도포량 부족") 는 원인만 규정하므로, 외형이 크게 다를 수 있는
  얕은 미도포 자국과 무너져 내린 분말이 한 용어에 포함될 수 있다 (→ §3 Powder Collapse 분리).
- 슬라이드에 없는 표준 항목: §4.7.3.1 *Overheating, Overmelting, or Thermal Heterogeneity* 는
  징후로 과열 영역의 변색·scorching 과 humping·elevation 을 들고, *Excessive Spatter/Ejecta* 를
  그 하위 항목으로 기술한다 (→ §5.3 에서 Over Melting·Spatter 에 대응).

## 2. 촬영 시점 용어

각 레이어는 두 장의 가시광 이미지로 기록된다. 이 문서의 "발견 이미지" 열은 결함이 관찰되는
이미지를 뜻한다.

| 이 문서 | 촬영 시점 | ORNL HDF5 | DSCNN_Dataset | DefectSegAM 입력 |
|---|---|---|---|---|
| **Scanning** | 레이저 용융 직후 | `slices/camera_data/visible/0` | `.../0/` 폴더 | 백본 입력 f0 |
| **Deposition** | 분말 도포 직후 | `slices/camera_data/visible/1` | `.../1/` 폴더 | 백본 입력 f1 |
| **Both** | 두 이미지 모두에서 관찰 | — | — | f0, f1, f1−f0 |

## 3. 새 분류 체계

ID 는 PDF 의 행 순서를 따른 **잠정** 번호다. 코드 반영 시 부록 A 참조.

| ID | 클래스 | 발견 이미지 | 형상 (이미지에 나타난 특징) | 비고 |
|---:|---|---|---|---|
| 0 | Powder | — | 결함도 파트도 없는 분말 베드 | 배경 (v4 와 동일) |
| 1 | Printed | — | 결함이 없는 프린트 영역 | 배경 (v4 와 동일) |
| 2 | **Incomplete Spreading** | Deposition | 파우더가 부족해서 일부 영역을 제대로 덮지 못함. 파우더 영역에 덮이다 만 **미세한 자국** 이 남음 | Super Elevation 과 구별하기 어려움 (§4.3) |
| 3 | **Powder Collapse** | Both | 파우더가 부족하거나 빌드플레이트 패드가 마모돼서 파우더가 **산사태처럼 무너져 내린** 형상 | Incomplete Spreading 과 원인(분말 부족)이 일부 겹치지만 이미지 특징이 너무 달라 분리. **기존 라벨 없음** (§6) |
| 4 | **Recoater Streaking** | Both | 손상된 리코터로 리코팅해서 생기는, 리코팅 방향과 **평행한** 줄무늬 | 표준 명칭(Scraping) 과 다르지만 논문에서 통용되는 용어 |
| 5 | **Recoater Hopping** | Both | 리코터가 분말 표면 바로 아래의 파트와 충돌하여 생기는, 리코팅 방향과 **수직인** 반복 줄무늬 | 표준 명칭(Part strike) 과 다르지만 논문에서 통용되는 용어. PDF 원문은 "돌출된 파트" 이나 ORNL·DSCNN 정의("part just below the powder surface") 에 맞춤 |
| 6 | **Super Elevation** | Deposition | 파트가 파우더 위로 **돌출** 된 형상 | Distortion / Swelling / Super-elevation 을 합침 (이미지 양상이 비슷함). Humping 은 레이어 이미지 외형 확인 전까지 잠정 편입. 추후 면적 기준으로 Edge-Swelling 과 구분 예정 |
| 7 | **Spatter** | Scanning | 용융 중 과도한 입자 방출로 **파우더 영역** 에 재가 날린 것처럼 보이는 거뭇한 형상 | Over Melting 과 시각적 양상이 유사하고 **영역** 으로 구분 (§4.2) |
| 8 | **Over Melting** | Scanning | **파트 영역** 위에 거뭇하게 재가 보임 | Spatter 와 시각적 양상이 유사하고 **영역** 으로 구분 (§4.2). 기존 ORNL·DSCNN 라벨은 공정 파라미터(원인) 기준이라 재검토 필요 (§7-9) |
| 9 | **Debris** | Both | 파우더면에 불규칙한 **덩어리나 조각** 이 포착됨 | 데이터 존재 확인 완료 (§6). 기존 Debris 는 "소-중형 교란" 포괄 클래스라 새 정의보다 넓음 → 재검토 (§7-9) |

클래스 이름은 코드·논문 표기에 맞춰 Title Case(Powder Collapse, Over Melting)로 쓴다.
슬라이드 표기는 Powder collapse, Over melting 이다.

예시 사진: `docs/Defect Example.pptx` (2026-10-05 판; 클래스별 슬라이드에 사진 30장과 장비·빌드·레이어
메모; Debris 사진 3장 포함). `scripts/figure_defect_examples.py` 가 30장을 모두 써서 논문 figure 8장
`paper/DefectSegAM_Ko/Figs/Defect_Examples_{IS_RH,PC,RS1,RS2,SE1,SE2_SP,OM,DB}.{pdf,png}` (각 2열 × 2행)
을 만든다. 슬라이드의 도형은 쓰지 않고 스크립트의 `FIGURES` 명세대로 표시한다: 붉은 표시 = 그 figure 의
클래스, 흰 표시 + 약어 태그 = 함께 나타난 다른 클래스, 점선 = 도포 경계, 인셋 = 확대. 사진은 자르기와
사진 전체에 균일한 레벨·감마 조정(휘도만, 선형 증폭 2.6배 이하; `--raw` 로 끔)만 한다.

같은 내용의 편집용 덱: `scripts/export_defect_examples_pptx.py` 가 같은 `FIGURES` 명세로
`docs/Defect_Examples_Annotated.pptx` (18쪽; 표지, 표시 방식, 패널 2개씩 16쪽) 를 만든다. 표시는
PowerPoint 도형이고 사진에는 자르기만 걸려 있어 덱에서 바로 고칠 수 있다.

## 4. 판별 규칙 (헷갈리는 쌍)

### 4.1 Recoater Streaking vs Recoater Hopping — 줄무늬 방향
- 리코팅 방향과 **평행** → Streaking. **수직** → Hopping.
- 리코팅 방향 메타데이터: DSCNN_Dataset v2022 는 각 재료 폴더의 `info.txt` 에
  `powder recoater direction` (+x, −x, ±x) 이 있다. ORNL Co-Registered HDF5 는 README 에는
  없지만 파일 루트 attrs `printer/recoating_direction` 에 기록되어 있으며 다섯 빌드 모두
  `-x` 다 (2026-09-28 h5py 로 확인, `printer/gas_flow_direction`
  도 `-x`). v2021-03 릴리스에는 `info.txt` 가 없어 방향 메타데이터가 없다.
- 90°/270° 회전(또는 transpose) 증강은 줄무늬의 평행·수직 관계를 바꾼다. flip(LR/UD)과 180°
  회전은 방향의 부호(−x↔+x)만 바꾸고 평행·수직 관계는 보존한다. v3/v4 가 rot90 을 빼고
  flip(LR/UD)+rot180 만 쓴 이유가 이것이며, 새 체계에서도 같은 제약이 유지된다.

### 4.2 Spatter vs Over Melting — 파우더 영역 / 파트 영역
- 둘 다 Scanning 이미지에서 거뭇하게 나타난다. **해당 픽셀이 파트 영역 안이면 Over Melting,
  파우더 영역이면 Spatter.**
- 영역 판정은 CAD 파트 마스크로 한다: ORNL `slices/part_ids` (0 = 파트 외부),
  DSCNN_Dataset v2022 의 `parts/` 이진 마스크. 단 v2022 학습 분할의 마스크 88장 중 61장이 빈 마스크
  (소스별 빈 마스크/마스크 수: 17-4 14/14, I2 24/24, MS 15/26, H13 6/7, I1 2/5, GP 0/12. GP 는 마스크
  12장 중 주석 레이어가 9개이며, 주석 레이어 85개 기준으로도 빈 마스크는 61개).
  v2021-03 릴리스에는 파트 마스크가 없다 (2026-09-28 확인).
- 이 규칙으로 기존에 IGNORE 하던 DSCNN native `Soot` (확산된 어두운 영역) 를 파트 마스크로 잘라
  두 클래스로 나눌 수 있지만, 비어 있지 않은 마스크가 있는 레이어에서만 가능하다. 현재 공개
  데이터로는 MS 9개 레이어의 약 0.53 M px(파트 안 65 K → Over Melting, 밖 469 K → Spatter),
  즉 전체 Soot 5.7 M px 의 약 9 % 뿐이다. L21 은 마스크가 없고 17-4 는 전부 빈 마스크다
  (2026-09-29 집계; §6, §7-2).

### 4.3 Incomplete Spreading vs Super Elevation — 둘 다 Deposition 에서 "덮이지 않은" 형태
- **파트 마스크 안**에서 금속 표면이 파우더 위로 솟아 밝게 보이면 Super Elevation.
- **파우더 영역**에 얕게 덮이다 만 자국이면 Incomplete Spreading.
- PDF 도 "구별하기 어려움" 으로 표시했다. 어노테이션 가이드에 예시 이미지를 넣을 것.

### 4.4 Incomplete Spreading vs Powder Collapse — 규모와 형상
- Incomplete Spreading 은 얕고 미세한 자국이며 Deposition 에서만 보인다.
- Powder Collapse 는 파우더가 산사태처럼 무너져 내린 큰 형상이며, 용융 뒤에도 남아 Scanning
  에서도 보인다 (슬라이드: Both). 경계가 뚜렷하다는 점은 가설이며 주석 시 확인한다.

### 4.5 Super Elevation 의 세분
- 현재는 하나의 클래스. 추후 **면적** 기준으로 Edge-Swelling (파트 가장자리의 작은 융기) 을
  분리할 예정. 분리 시에도 어노테이션은 Super Elevation 하나로 하고 후처리에서 나누는 것이
  라벨 일관성에 유리하다.

## 5. 기존 클래스 → 새 체계 매핑

### 5.1 ORNL Peregrine 12-class (Stage 1 의사 라벨, `common/config.py::ORNL_CLASS_NAMES`)

| ORNL ID | ORNL 클래스 | → 새 ID | 새 클래스 | 비고 |
|---:|---|---:|---|---|
| 0 | Powder | 0 | Powder | |
| 1 | Printed | 1 | Printed | |
| 2 | Recoater Hopping | 5 | Recoater Hopping | |
| 3 | Recoater Streaking | 4 | Recoater Streaking | |
| 4 | Incomplete Spreading | 2 | Incomplete Spreading | Powder Collapse 로 볼 픽셀이 섞여 있는지 확인 필요 |
| 5 | Swelling | 6 | Super Elevation | **통합** |
| 6 | Debris | 9 | Debris | 기존 정의는 포괄(catch-all) 클래스 → 새 정의(불규칙한 덩어리·조각) 충족 여부 재검토 (§7-9) |
| 7 | Super-Elevation | 6 | Super Elevation | **통합** |
| 8 | Spatter | 7 | Spatter | ORNL 정의가 "분말 베드에 착지한 비산물" 이므로 파우더 영역 기준과 일치 |
| 9 | Misprint | −1 | IGNORE | 형상 기준이 아니라 CAD 대비 기준 (v4 와 동일) |
| 10 | Over Melting | 8 | Over Melting | ORNL 정의는 고에너지 밀도 공정 파라미터(원인 기준), 픽셀의 99.8 % 가 B2. 새 정의(파트 영역의 거뭇한 외형) 충족 여부를 레이어별로 재검토한 뒤 재매핑 (§7-9) |
| 11 | Under Melting | −1 또는 별도 | (미결) | §7 |

### 5.2 DSCNN_Dataset 재료별 native class (Stage 2 사람 GT)

소스 약어: L21 = v2021_LPBF (ConceptLaser M2, SS316L), 17-4 = 17-4_PH_Stainless_Steel,
GP = GammaPrint-700, I1 / I2 = Inconel_718_1 / _2, MS = Maraging_Steel (AddUp, 롤러형 리코터),
BJ21 = v2021_BJ, H13 = v2022_BJ_H13. native ID 는 각 `info.txt` / readme 참조.

| native 클래스 | 등장 소스 | → 새 클래스 | 비고 |
|---|---|---|---|
| Powder / Un-Fused Powder | 전부 | Powder | |
| Printed / Printed Material | 전부 | Printed | |
| Recoater Hopping | L21, 17-4, GP, I1, I2, MS | Recoater Hopping | MS 정의는 "롤러가 만드는 물결 모양 세로선" 으로 약간 다름 |
| Recoater Streaking / Roller Streaking | L21, 17-4, GP, I1, I2, MS, BJ21, H13 | Recoater Streaking | |
| Incomplete Spreading / Powder Short Feed / Short Feed | L21, 17-4, GP, I1, I2, MS, BJ21, H13 | Incomplete Spreading | Powder Collapse 형상이 섞여 있으면 재라벨 |
| Swelling / Edge Swelling | L21, MS / 17-4, GP, I1, I2 | Super Elevation | **통합** |
| Super-Elevation | L21, 17-4, GP, I1, I2, MS | Super Elevation | **통합** |
| Exposed Part | H13 | Super Elevation (후보) | "파트가 파우더에 덮이지 않음" 정의가 일치. 기존엔 IGNORE → **결정 필요** |
| Debris | 전부 | Debris | 포괄 정의 → 새 정의 충족 여부 재검토 (§7-9) |
| Soot | L21, 17-4, I2, MS | Spatter (파우더 영역) / Over Melting (파트 영역) | 파트 마스크로 분할. 분할 가능한 것은 MS 의 9개 레이어뿐 (L21 마스크 없음, 17-4 전부 빈 마스크, I2 는 Soot 0 px). 나머지는 IGNORE 유지. 기존엔 IGNORE → **결정 필요** |
| Spatter on Powder | GP, I1 | Spatter | |
| Excessive Melting | 17-4, GP, I1, I2 | Over Melting | DSCNN 정의도 고에너지 밀도 공정 파라미터 기준 → 외형 재검토 필요 (§7-9) |
| Crashing | 17-4, I2 | IGNORE | 리코터 충돌 (Part strike 계열). GT 픽셀 0 |
| Mounding Powder | GP, I1 | IGNORE | 형상상 Debris 후보이나 GT 픽셀 0 |
| Stripe Boundary | GP, I1 | IGNORE | 정상 해치 경계, 결함 아님 |
| Localized Bright Spot / Localized Dark Regions | GP, I1 | IGNORE | 광학 단층(OT) 이미지 기반. v4 는 Dark Regions → Under Melting 이었음 (§7) |
| Cornrows | H13 | IGNORE | 바인더젯 노즐 경계, L-PBF 에 없음 |
| Misprint | 17-4, GP, I1, I2, MS, H13 | IGNORE | |

기존 2-step 매핑 (native → ORNL 12 → 버전 클래스) 은 그대로 쓸 수 있지만, `Soot` 분할과
`Exposed Part` 는 ORNL 12-class 에 대응이 없으므로 **native → 새 체계 직접 규칙** 이 추가로
필요하다.

### 5.3 ASTM E3353 → 새 체계

| ASTM E3353 | → 새 클래스 |
|---|---|
| Skipping, Insufficient powder delivery | Incomplete Spreading (형상이 무너짐이면 Powder Collapse) |
| Scraping | Recoater Streaking |
| Part strike | Recoater Hopping |
| Distortion, Superelevation, Humping (잠정) | Super Elevation |
| Overheating, Overmelting (§4.7.3.1; 변색·scorching) | Over Melting (슬라이드에 없는 항목, 2026-09-29 추가) |
| Excessive spatter / ejecta (§4.7.3.1 하위 항목) | Spatter (파우더 영역) / Over Melting (파트 영역, 본 체계의 영역 규칙) |
| Balling | 대응 없음 — L-PBF 어휘에 없음, v2021-03 EB-PBF 소스에만 존재 (§7-6) |

## 6. 데이터 가용성

ORNL 열은 `PLAN_v4.md` §3 의 B1.1–B1.5 합산 (v4 매핑·1036² 재추출 후, DSCNN 예측 라벨).
빌드별 점유율은 2026-09-29 에 `cache/resized_sz1036_v4/B*/pix_per_layer.npy` 로 재계산한 값이다.
DSCNN GT 열은 2026-09-28 에 `DSCNN_Dataset/**/annotations/*.npy` 를 전수 집계한 값
(레이어 수는 해당 클래스 픽셀이 1개 이상인 레이어). 소스 약어는 §5.2 와 같다.

| 새 클래스 | ORNL HDF5 (px, 집중 빌드) | DSCNN 사람 GT (px / 레이어) | 판단 |
|---|---|---|---|
| Incomplete Spreading | 867 K (B1 38 %, B2 37 %, B3 14 %, B4 11 %) | L21 3.72 M/9, MS 29.5 M/20, 17-4 70 K/5, GP 71 K/2, BJ21 7.89 M/5, H13 16.1 M/4 | 충분. 단 Powder Collapse 형상이 섞여 있을 수 있음 |
| Powder Collapse | **없음** | **없음** | **신규 어노테이션 필요.** ORNL B1.5 (분말 공급 부족 실험) 와 L21/MS 의 Incomplete Spreading 레이어를 재검토해 분리 |
| Recoater Streaking | 187 M (B5 45 %, B2 34 %, B3 16 %) | L21 262 K/15, MS 79 K/4, I2 52 K/10, GP 3.6 K/1, BJ21 901 K/6, H13 153 K/1 | 충분 |
| Recoater Hopping | **25 px** (B4) | MS 4.39 M/25 | ORNL 에는 사실상 없음. Stage 2 의 MS 가 유일한 소스 (롤러형 리코터) |
| Super Elevation (통합) | Swelling 10 M (B2 58 %) + Super-Elevation 268 K (B5 98 %) | Swelling/Edge Swelling: L21 143 K/20, MS 27 K/6, 17-4 1.2 K/13, GP 46 K/9, I1 43 K/5, I2 15 K/22. Super-Elevation: L21 963 K/6, GP 234 K/2, I2 2.4 K/2 | 충분. 통합으로 v4 의 두 rare class 가 하나의 중간 규모 클래스가 됨 |
| Spatter | 667 M (B4 63 %) | Spatter on Powder: GP 173 K/8, I1 1.21 M/3. Soot 분할 시 추가: MS 469 K/9 (L21 3.87 M·17-4 1.12 M 은 마스크 없음/빈 마스크로 분할 불가) | 충분 |
| Over Melting | 394 K (B2 99.8 %) | Excessive Melting: I2 13.5 K/11. Soot 분할 시 MS 65 K/9 추가 | ORNL 편향 심함. Soot 분할로 늘어나는 양은 작음. 기존 라벨의 외형 재검토 필요 (§7-9) |
| Debris | 1.8 M (B3 90 %) | L21 492 K/8, I2 72 K/12, MS 41 K/15, BJ21 209 K/5, H13 22 K/3 | **데이터 있음** (PDF 의 확인 항목 해소). 규모는 작음 |

참고: DSCNN native 중 `info.txt` / v2021-03 readme 에 이름은 있지만 공개 학습 subset 에 GT 픽셀이
0 인 클래스 — Recoater Hopping (L21, 17-4, GP, I1, I2), Recoater Streaking (17-4, I1),
Incomplete Spreading (I1, I2), Debris (17-4, GP, I1), Super-Elevation (17-4, I1, MS),
Excessive Melting (17-4, GP, I1), Soot (I2), Crashing (17-4, I2), Mounding Powder (GP, I1),
Localized Dark Regions (I1), Misprint (17-4, GP, I1, I2, MS, H13).

## 7. 미결 사항

1. **Under Melting 처리.** ORNL 정의("저에너지 밀도 공정 파라미터로 용융된 영역") 는 원인 기준이라
   이 체계의 원칙에 어긋난다. ORNL 에 28 M px (B2 99.9 %) 가 있고 v4 가 학습 대상으로 쓰고 있으므로
   (a) IGNORE, (b) 외형 기준으로 재정의해 10번 클래스로 유지, 중 선택 필요.
   v4 의 GP `Localized Dark Regions → Under Melting` 매핑도 같이 결정.
2. **Soot 분할 채택 여부.** 전체 Soot 약 5.7 M px 중 현재 공개 마스크로 자동 분할 가능한 것은
   MS 9개 레이어의 약 0.53 M px (Spatter 469 K + Over Melting 65 K, 약 9 %) 뿐이다. L21(마스크 없음)과
   17-4(전부 빈 마스크)는 주석자가 파트 영역을 직접 판정해야 한다. 마스크 경계 오차도 라벨 노이즈가
   될 수 있다.
3. **Exposed Part (H13) → Super Elevation** 채택 여부. 바인더젯 소스라 L-PBF 외형과 같은지 확인.
4. **Powder Collapse 어노테이션.** 어떤 레이어에서 뽑을지 (ORNL B1.5 우선), 누가·어떤 도구로 할지.
5. ~~**ORNL 빌드의 리코팅 방향.**~~ 해소 (2026-09-28): HDF5 attrs `printer/recoating_direction = -x`
   (§4.1). 남는 문제는 방향 메타데이터가 없는 v2021-03 소스 (L21, BJ21) 의 처리.
6. **Balling** 은 대응 클래스가 없다. ORNL·DSCNN L-PBF 에 라벨이 없으므로(v2021-03 EB-PBF 소스에만
   존재) 이번 체계에서 제외.
7. **Super Elevation 의 Edge-Swelling 세분 기준** (면적 임계값) 은 추후.
8. **ID 순서.** 이 문서는 PDF 행 순서를 따랐다. v4 confusion matrix 와 비교하기 쉽도록 기존
   순서(Hopping 2, Streaking 3, …) 를 유지하는 대안도 있음.
9. **기존 라벨의 새 정의 충족 여부 검토.** 이름은 같아도 정의가 바뀐 클래스가 있다. Over Melting 은
   ORNL·DSCNN 에서 고에너지 밀도라는 원인으로 정의되었고, Debris 는 포괄 클래스였으며, Incomplete
   Spreading 에는 Powder Collapse 형상이 섞여 있을 수 있다. 재매핑 전에 레이어별로 외형을 확인한다.
   논문에서 keyhole 전조로 해석한 11-class Over Melting 결과와 새 Over Melting 은 같은 대상이 아니다.

## 부록 A. 코드 반영 초안

`vN/config_vN.py` 로 옮길 때의 출발점. 값은 §3, §5.1 과 동일하다 (클래스 이름은 Title Case).

```python
# 새 체계 — 결함 8 + 배경 2 (docs/DEFECT_TAXONOMY.md §3)
CLASS_NAMES_NEW = [
    "Powder",               # 0
    "Printed",              # 1
    "Incomplete Spreading", # 2  Deposition
    "Powder Collapse",      # 3  Both      (신규 — 라벨 없음, 어노테이션 전까지 support 0)
    "Recoater Streaking",   # 4  Both      (리코팅 방향과 평행)
    "Recoater Hopping",     # 5  Both      (리코팅 방향과 수직)
    "Super Elevation",      # 6  Deposition (= ORNL Swelling + Super-Elevation)
    "Spatter",              # 7  Scanning  (파우더 영역의 거뭇한 영역)
    "Over Melting",         # 8  Scanning  (파트 영역의 거뭇한 영역)
    "Debris",               # 9  Both
]
N_CLASSES_NEW = len(CLASS_NAMES_NEW)  # 10

# ORNL 12-class → 새 체계 (§5.1). -1 = IGNORE
ORNL_12_TO_NEW: dict[int, int] = {
    0: 0,    # Powder
    1: 1,    # Printed
    2: 5,    # Recoater Hopping
    3: 4,    # Recoater Streaking
    4: 2,    # Incomplete Spreading
    5: 6,    # Swelling             → Super Elevation
    6: 9,    # Debris
    7: 6,    # Super-Elevation      → Super Elevation
    8: 7,    # Spatter
    9: -1,   # Misprint
    10: 8,   # Over Melting
    11: -1,  # Under Melting        → §7-1 결정 전까지 IGNORE
}

# DSCNN native 중 ORNL 12-class 를 거치지 않고 직접 처리해야 하는 항목 (§5.2)
#   Soot         → 비어 있지 않은 파트 마스크가 있는 레이어에서만:
#                  마스크 안이면 8 (Over Melting), 밖이면 7 (Spatter).
#                  마스크가 없거나 빈 레이어에서는 -1 (IGNORE) 유지          [§7-2]
#   Exposed Part → 6 (Super Elevation)                                       [§7-3]
# 나머지는 MATERIAL_TO_ORNL_V4 (v4/config_v4.py; common/config.py 의 LPBF 6 소스 + BJ 2 소스)
# → ORNL_12_TO_NEW 의 2-step 으로 충분.
```

## 참고

- ASTM E3353-22 — <https://store.astm.org/e3353-22.html>
- ORNL Co-Registered HDF5 README — `ORNL_Data/Co-Registered In-Situ and Ex-Situ Dataset/[baseline] (Peregrine v2023-11)/README.md` (12-class 정의, `visible/0`·`visible/1` 촬영 시점)
- DSCNN_Dataset readme — `ORNL_Data/DSCNN_Dataset/Peregrine Dataset v2021-03/readme.pdf`, `.../v2022-10.1/readme.pdf` 및 각 재료 폴더의 `info.txt` (native class ID·리코팅 방향)
- 현행 클래스 공간 — `v4/PLAN_v4.md` §2–3, `v4/config_v4.py`, 논문 `paper/DefectSegAM_Ko/sections/3_method.tex` 표 `tab:classes`
