# DSCNN 분류 클래스 정리

원 논문: Scime, L.; Siddel, D.; Baird, S.; Paquit, V.
*"Layer-wise anomaly detection and classification for powder bed additive manufacturing processes:
A machine-agnostic algorithm for real-time pixel-wise semantic segmentation"*,
**Additive Manufacturing 36 (2020) 101453**, doi:10.1016/j.addma.2020.101453.

> 비고: 폴더 내 PDF 파일 `(DSCNN) Layer-wise anomaly detection ....pdf` 는
> 임베디드 폰트/콘텐츠 stream 의 zlib FCHECK 에러로 텍스트 추출과 페이지 렌더링이 모두 실패합니다
> (`pdftotext`, `pymupdf` 모두 빈 페이지 반환). 본 정리는 ORNL Peregrine 시스템(DSCNN 의 production 구현체) 이
> 출력하는 라벨 정의서인 `ORNL_Data/.../[baseline] (Peregrine v2023-11)/README.md` 를 1차 출처로 작성했습니다.

---

## 1. DSCNN 의 12 클래스 (Peregrine v2023-11 출력 라벨)

데이터셋 README 의 정식 라벨. `slices/segmentation_results/{ID}` 에서 0/1 마스크로 제공.

| ID | 클래스 | 설명 |
|----|--------|------|
| 0  | Powder | 이상 또는 프린트된 파트가 없는 분말 베드 영역 |
| 1  | Printed | 이상이 감지되지 않은 프린트 영역 |
| 2  | Recoater Hopping | 리코터가 표면 아래 파트에 충돌할 때 발생하는 물결무늬 |
| 3  | Recoater Streaking | 리코터 손상 또는 큰 입자 끌림으로 인한 줄무늬 |
| 4  | Incomplete Spreading | 분말 베드에 불충분한 분말 도포 |
| 5  | Swelling | 분말 위로 돌출된 프린트 재료의 변형/뒤틀림 |
| 6  | Debris | 분말 베드의 소-중형 교란 (포괄적 클래스) |
| 7  | Super-Elevation | 프린트 영역 위의 분말 커버리지 부족 |
| 8  | Spatter | 용접 풀에서 튀어나와 분말 베드에 착지한 비산물 |
| 9  | Misprint | 의도된 파트 형상 외부에서 감지된 프린트 재료 |
| 10 | Over Melting | 고에너지 밀도 공정 파라미터로 용융된 영역 |
| 11 | Under Melting | 저에너지 밀도 공정 파라미터로 용융된 영역 |

클래스 이름 목록은 HDF5 의 `slices/segmentation_results/class_names` 키로 조회 가능.

---

## 2. 본 프로젝트가 사용하는 8 클래스 (config.py)

[Sources/vppm/common/config.py:53-63](../../Sources/vppm/common/config.py#L53-L63) 의 `DSCNN_FEATURE_MAP` —
DSCNN 12 클래스 중 8개를 골라 21-피처 베이스라인의 G1 그룹 (#4–11) 으로 사용.

| feat # | feat idx | 이름 | HDF5 cls | 의미 |
|:--:|:--:|:----|:--:|:----|
| 4  | 3 | `seg_powder`             | 0  | 분말 (정상) |
| 5  | 4 | `seg_printed`            | 1  | 프린트 (정상) |
| 6  | 5 | `seg_recoater_streaking` | 3  | 리코터 줄무늬 |
| 7  | 6 | `seg_edge_swelling`      | 5  | 엣지 융기 |
| 8  | 7 | `seg_debris`             | 6  | 잔해 |
| 9  | 8 | `seg_super_elevation`    | 7  | 과돌출 |
| 10 | 9 | `seg_soot`               | 8  | (코드 라벨: Soot, **HDF5 cls 8 = Spatter** — 명칭 불일치) |
| 11 | 10 | `seg_excessive_melting` | 10 | 과용융 (= HDF5 Over Melting) |

> 주의: `seg_soot` 는 코드 별칭일 뿐 실제 HDF5 cls 8 의 데이터셋 라벨은 **Spatter** 이다.
> 학습 입력은 동일 채널이지만 보고서/그래프에 표기할 때 헷갈릴 수 있어 두 이름을 병기 권장.

채택하지 않은 4 클래스: `Recoater Hopping (2)`, `Incomplete Spreading (4)`, `Misprint (9)`, `Under Melting (11)`.

---

## 3. Powder 이미지 vs Melt 이미지 — 1차 검출 분류

**전제**: DSCNN 은 두 카메라 이미지를 채널 방향으로 스택해 한 번에 입력받고 단일 segmentation 을 출력한다.
즉 모델 자체는 둘을 동시에 본다. 아래 분류는 **"각 이상이 물리적으로 어느 이미지에서 식별 단서가 나오는가"** 기준.

데이터셋 정의 ([README.md:103-104](../../ORNL_Data/Co-Registered%20In-Situ%20and%20Ex-Situ%20Dataset/[baseline]%20(Peregrine%20v2023-11)/README.md)):
- `camera_data/visible/0` = **용융 직후** (Melt 이미지)
- `camera_data/visible/1` = **분말 도포 직후** (Powder 이미지)

### 3.1 프로젝트 8 클래스 기준

#### Powder 이미지 (visible/1) — 1차 검출
| HDF5 cls | 코드 이름 | 단서 |
|:--:|---|---|
| 0 | `seg_powder`             | 정상 분말 매트릭스 |
| 3 | `seg_recoater_streaking` | 신선 분말의 줄무늬 |
| 6 | `seg_debris`             | 분말 표면 위 이물질 |

#### Melt 이미지 (visible/0) — 1차 검출
| HDF5 cls | 코드 이름 | 단서 |
|:--:|---|---|
| 1  | `seg_printed`            | 정상 용융 표면 매트릭스 |
| 8  | `seg_soot` (= Spatter)   | 용융 풀 비산물 |
| 10 | `seg_excessive_melting`  | 고에너지 밀도 용융 영역 morphology |

#### 양쪽 이미지 모두 (cross-modal)
| HDF5 cls | 코드 이름 | 비고 |
|:--:|---|---|
| 5 | `seg_edge_swelling`   | 분말 위 돌출 / 멜트 표면 융기 둘 다 단서 |
| 7 | `seg_super_elevation` | 분말 커버리지 부족 / 멜트 엣지 솟음 둘 다 단서 |

### 3.2 DSCNN 12 클래스 전체 기준

#### Powder 이미지 (visible/1) — 1차 검출
| HDF5 cls | 클래스 | 단서 |
|:--:|---|---|
| 0 | Powder              | 정상 분말 매트릭스 |
| 2 | Recoater Hopping    | 신선 분말 위 물결무늬 (리코터-부품 충돌 자국) |
| 3 | Recoater Streaking  | 신선 분말의 줄무늬 |
| 4 | Incomplete Spreading | 분말이 덜 깔린 빈 영역 |
| 6 | Debris              | 분말 표면 위 이물질 |

#### Melt 이미지 (visible/0) — 1차 검출
| HDF5 cls | 클래스 | 단서 |
|:--:|---|---|
| 1  | Printed         | 정상 용융 표면 매트릭스 |
| 8  | Spatter         | 용융 풀 비산물 |
| 9  | Misprint        | CAD 영역 밖에서 검출된 프린트 재료 (멜트 표면에서만 식별) |
| 10 | Over Melting    | 고에너지 밀도 용융 morphology |
| 11 | Under Melting   | 저에너지 밀도 미용융 morphology |

#### 양쪽 이미지 모두 (cross-modal)
| HDF5 cls | 클래스 | 비고 |
|:--:|---|---|
| 5 | Swelling         | 분말 위 돌출 / 멜트 표면 융기 둘 다 단서 |
| 7 | Super-Elevation  | 분말 커버리지 부족 / 멜트 엣지 솟음 둘 다 단서 |

---

## 4. 참고 링크
- [Scime 2020 — ScienceDirect](https://www.sciencedirect.com/science/article/pii/S2214860420308253)
- [ORNL publication record](https://www.ornl.gov/publication/layer-wise-anomaly-detection-and-classification-powder-bed-additive-manufacturing)
- 본 프로젝트 코드: [Sources/vppm/common/config.py:53-63](../../Sources/vppm/common/config.py#L53-L63), [Sources/vppm/FEATURES.md](../../Sources/vppm/FEATURES.md)
- 데이터셋 라벨 정의: [ORNL_Data/.../[baseline] (Peregrine v2023-11)/README.md](../../ORNL_Data/Co-Registered%20In-Situ%20and%20Ex-Situ%20Dataset/[baseline]%20(Peregrine%20v2023-11)/README.md)
