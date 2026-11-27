# DefSegAM 클래스 변경 안내 — 새 레이블 체계 `pbf-2026.11`

DefSegAM 을 새 클래스 체계로 다시 학습하고 서빙할 때 필요한 사실을 DIVA 쪽에서 정리했다. DIVA 저장소를 볼 수 없어도 읽을 수 있게 썼다.

- 기준일: 2026-11-16
- 단일 출처: DIVA 의 `backend/modules/Labeling/schema_pbf_2026_11.json`. 이 문서와 다르면 그 파일이 맞다. 데이터셋을 내보낼 때 같은 내용이 `classes.json` 으로 함께 나간다.
- 근거: 레이블링 가이드 `[레이블링] Defect_Examples_Annotated_rev02.pptx`(35쪽), DIVA 코드, DIVA 가 갖고 있는 DefSegAM v4 API 명세 사본.
- 클래스 10개는 확정이다(2026-11-08). 번호는 DIVA 가 지금 저장과 내보내기에 쓰는 값이다. 확정되지 않은 것은 §8 에 모았다.

---

## 1. 한눈에

- 클래스가 v4 의 11개에서 **10개(정상 2 + 결함 8)** 로 바뀐다.
- **번호를 새로 매겼다.** Powder 0 · Printed 1 만 그대로이고, 이름이 같은 결함도 번호가 다르다(Incomplete Spreading 은 4 → 2).
- 없어진 클래스는 Swelling · Under Melting, 새로 생긴 클래스는 Powder Slumping 이다.
- 결함은 id 2~9 다(v4 는 2~10).
- 미레이블 값이 있다. 학습용 `annotations/*.npy` 에서 `-1` 이고, 클래스가 아니다.
- **v4 서비스를 새 모델로 그 자리에서 바꿔 끼우면 안 된다.** DIVA 가 출력을 v4 번호 · v4 색으로 읽고 있어서 조용히 틀린 결과가 나온다(§7).

---

## 2. 배경

DIVA 는 금속 PBF 3D 프린팅 공정 데이터를 모아 보는 웹 플랫폼이다. 층마다 카메라 사진 두 장(레이저 용융 직후 · 분말 도포 직후)이 쌓이고, DIVA 는 DefSegAM v4 를 HTTP 로 불러 결함 분포를 보여 주거나 레이블링의 초안으로 쓴다.

DIVA 에 사람이 직접 레이블 맵을 만드는 기능이 생기면서, 클래스를 레이블링 가이드대로 새로 정했다. 이 체계의 이름이 `pbf-2026.11` 이다. 이번 DefSegAM 업데이트는 모델의 클래스를 이 체계에 맞추는 일이다. 그때까지 DIVA 는 v4 출력을 §4 의 대응표로 바꿔서 쓴다.

---

## 3. 새 클래스

### 3.1 목록

| id | key | 이름 | 약어 | 구분 | 기준 이미지 | 형태 |
|---:|---|---|---|---|---|---|
| 0 | `powder` | Powder | — | 정상 | 둘 다 | 배경 |
| 1 | `printed` | Printed | — | 정상 | Scanning | 영역 |
| 2 | `incomplete_spreading` | Incomplete Spreading | IS | 결함 | Deposition | 영역 |
| 3 | `powder_slumping` | Powder Slumping | PS | 결함 | 둘 다 | 영역 |
| 4 | `recoater_streaking` | Recoater Streaking | RS | 결함 | 둘 다 | 리코팅 방향과 평행한 선 |
| 5 | `recoater_hopping` | Recoater Hopping | RH | 결함 | 둘 다 | 리코팅 방향과 수직인 선 |
| 6 | `super_elevation` | Super Elevation | SE | 결함 | Deposition | 영역 |
| 7 | `spatter` | Spatter | SP | 결함 | Scanning | 작은 점 여럿 |
| 8 | `over_melting` | Over Melting | OM | 결함 | Scanning | 영역 |
| 9 | `debris` | Debris | DB | 결함 | 둘 다 | 덩어리 |

- 결함 2~9 의 순서는 가이드의 결함 명세 표 순서다.
- 「기준 이미지」는 그 클래스를 주로 어느 사진에서 판단하는지다. Scanning 은 레이저 용융 직후, Deposition 은 분말 도포 직후 사진이다(§5).
- 「형태」는 가이드에 없고 DIVA 가 붙인 힌트다.
- 미레이블은 DIVA 가 저장하는 1채널 PNG 에서 255, 학습용 npy 에서 −1 이다.
- 가이드 안에서 이름 표기가 섞여 있다. 아래는 같은 클래스다. 코드에서는 `key` 로 구분한다.
  - Powder Slumping = Powder Collapse
  - Super Elevation = Superelevation = Super-Elevation(v4 표기)
  - Over Melting = Overmelting

### 3.2 레이블링 기준

가이드의 명세 표를 그대로 옮겼다.

| id | 클래스 | 레이블링 기준 | 설명 |
|---:|---|---|---|
| 0 | Powder | Printed 영역과 결함 영역을 제외한 나머지 모든 영역 | — |
| 1 | Printed | visible/0 에서 레이저가 조사되어 하얗게 나타나는 영역 | — |
| 2 | Incomplete Spreading | 파우더 영역에 덮이다 만 미세한 자국(라인)이 남음 | 파우더가 부족해서 일부 영역을 제대로 덮지 못함. |
| 3 | Powder Slumping | Incomplete Spreading 와 다르게 그림자가 지고 경사면이 있음 | 파우더가 부족하거나 빌드플레이트 패드가 마모돼서 파우더가 산사태처럼 무너져 내린 형상 |
| 4 | Recoater Streaking | 리코팅 방향 줄무늬. 완전한 직선 (본 데이터에서는 세로 줄무늬) | 손상된 리코터로 리코팅해서 생기는, 리코팅 방향과 평행한 줄무늬 |
| 5 | Recoater Hopping | 리코팅 방향과 수직인 줄무늬. 살짝 꼬불꼬불함. (본 데이터에서는 가로 줄무늬) | 리코터가 분말 표면 바로 아래의 파트와 충돌하여 생기는, 리코팅 방향과 수직인 줄무늬 |
| 6 | Super Elevation | Deposition 에서 파우더에 안덮힌 모든 출력물 부분을 표시하면 됨 | 파트가 파우더 위로 돌출 된 형상 |
| 7 | Spatter | 파우더 영역 에 재가 날린 것처럼 보이는 거뭇한 형상 | 용융 중 과도한 입자 방출로 실제로 재가 날린 거라 볼 수 있음 |
| 8 | Over Melting | 파트 영역 위에 거뭇하게 탄 자국이 보임 | 과도한 용융으로 타버림 |
| 9 | Debris | 파우더면에 불규칙한 덩어리나 조각 이 포착됨 (레이블 어려울 수 있음. Super Elevation과 혼동 가능) | 출력물이 리코터랑 강하게 부딪혀서 떨어져나간 것. |

영어 문장은 `classes.json` 의 `criteria_en` 에 있다.

### 3.3 레이블 맵을 만드는 규칙

- 한 층의 두 사진을 함께 보고 레이블 맵 **한 장**을 만든다. 사진마다 따로 만들지 않는다.
- 화소마다 클래스는 하나다. 모든 화소에 클래스가 있어야 완성이다.
- 결함이 정상보다 먼저다. 결함으로 확인한 곳에 결함 클래스를 주고, 남은 곳만 Printed 나 Powder 로 둔다. 예를 들어 이번 층에 녹인 자리라도 도포 뒤에 분말 위로 드러나 있으면 Printed 가 아니라 Super Elevation 이다.
- 기준 이미지는 배타 조건이 아니다. 명세 표에서 Deposition 인 Incomplete Spreading 도 가이드의 예시 쪽에는 "양쪽 사진에서 전부 나타남" 이라고 적혀 있다. Super Elevation 은 "대부분 Deposition 사진에서 나타남(너무 심해지면 Scanning에서도 나타남)" 이다.
- 헷갈리기 쉬운 쌍은 넷이다.
  - IS ↔ PS: 그림자가 지고 경사면이 있으면 PS 다.
  - RS ↔ RH: 리코팅 방향과 평행하고 완전한 직선이면 RS, 수직이고 살짝 꼬불꼬불하면 RH 다.
  - SE ↔ DB: 제자리의 파트가 드러난 것이 SE, 떨어져 나온 조각이 DB 다. 가이드도 구별이 어렵다고 적었다.
  - SP ↔ OM: 거뭇한 자국이 분말 위에 있으면 SP, 파트 위에 있으면 OM 이다.
- RS 와 RH 는 모양이 아니라 **리코팅 방향과의 관계**로 정의된다. "세로 = RS, 가로 = RH" 는 가이드가 다룬 데이터에서만 맞고, 가이드 슬라이드 34 에는 "이장비에선 방향이 반대" 라고 적힌 예시가 있다(§5.3).

---

## 4. v4 → 새 체계

DIVA 가 v4 출력을 레이블링 초안으로 들여올 때 실제로 쓰는 대응표다.

| v4 id | v4 이름 | 새 id | 새 이름 | 비고 |
|---:|---|---:|---|---|
| 0 | Powder | 0 | Powder | |
| 1 | Printed | 1 | Printed | |
| 2 | Recoater Hopping | 5 | Recoater Hopping | |
| 3 | Recoater Streaking | 4 | Recoater Streaking | |
| 4 | Incomplete Spreading | 2 | Incomplete Spreading | 일부는 Powder Slumping 일 수 있다(§4.1) |
| 5 | Swelling | — | 없음 | DIVA 는 미레이블로 들여온다(§4.1) |
| 6 | Debris | 9 | Debris | |
| 7 | Super-Elevation | 6 | Super Elevation | 이름에서 하이픈이 빠졌다 |
| 8 | Spatter | 7 | Spatter | |
| 9 | Over Melting | 8 | Over Melting | |
| 10 | Under Melting | — | 없음 | DIVA 는 미레이블로 들여온다 |
| — | 없음 | 3 | Powder Slumping | v4 에 대응하는 클래스가 없다 |

### 4.1 기존 주석을 새 체계로 옮길 때

가이드의 레이블링 예시(슬라이드 28~35)는 `visible/0` · `visible/1` · `annotations/{id}.npy` 로 된 기존 주석을 그린 그림이다. **범례의 번호는 그 데이터의 "원본 ID" 이고 새 번호가 아니다.** 보이는 범위에서는 v4 번호와 같다(예: 4 Incomplete Spreading, 7 Super-Elevation).

가이드 작성자는 범례의 이름 위에 새 이름을 덮어 적어 두었다. 기존 주석을 새 기준으로 어떻게 읽는지가 여기서 드러난다.

| 범례의 원래 이름 (원본 ID) | 덮어 적은 이름 | 슬라이드 |
|---|---|---|
| Un-Fused Powder (0) · Printed Material (1) | Powder · Printed | 28 · 29 |
| Soot (8) | Spatter | 28 · 29 · 34 |
| Excessive Melting (9) | Over Melting | 30~32 |
| Edge Swelling (5) | Super Elevation | 30~32 |
| Incomplete Spreading (4) | Powder Slumping | 34 |
| Swelling (5) | 행을 가려 놓음 (화소가 0.01% 미만) | 28 · 29 |

여기서 두 가지를 조심해야 한다.

- **기존 주석의 Incomplete Spreading 은 새 기준의 IS 와 PS 가 섞여 있다.** 슬라이드 34 는 분말이 크게 무너진 예시인데, 원본 주석은 그 영역을 Incomplete Spreading 으로 칠했고 가이드는 Powder Slumping 으로 고쳐 불렀다. 번호만 바꾸는 변환으로는 PS 가 만들어지지 않는다.
- **Swelling 을 어디로 보낼지가 두 군데에서 다르다.** DIVA 의 대응표는 미레이블로 두는데, 가이드 예시(슬라이드 30~32)는 Edge Swelling 을 Super Elevation 으로 고쳐 적었다. 어느 쪽으로 할지 정해야 한다(§8).

가이드의 결함 예시 사진(슬라이드 9~26)은 이와 달리 대부분 M160 · M280 장비의 공정 사진이다. 사진마다 장비 · 날짜 · 층이 적혀 있다.

---

## 5. 입력 이미지

### 5.1 두 장의 이름

| DIVA 이름 | 가이드 · 데이터셋 | v4 API 필드 | 뜻 |
|---|---|---|---|
| Scanning | `visible/0` | `img0` (after melt) | 레이저 용융 직후 |
| Deposition | `visible/1` | `img1` (after spread) | 분말 도포 직후 |

- DIVA 가 쓰는 사진은 빌드 플레이트 영역에 맞춰 잘라 낸 전처리본이다. 크기는 공정마다 다르고 정사각형이라는 보장도 없다(예: 1762×1812).
- 레이블링 대상인 PBF 비전 데이터는 주로 M160 · M280 두 기종의 것이다.
- 일부 공정은 저장소의 Scanning · Deposition 폴더가 뒤바뀌어 있다(DIVA 조사 메모 기준으로 M280 의 2026-08 이후 공정). DIVA 가 공정 설정 `swap_sd` 로 바로잡아 보내므로, 모델 쪽에서는 `visible/0` 을 용융 직후로 믿으면 된다. 다만 `swap_sd` 는 사람이 공정마다 켜는 값이다.

### 5.2 방향

v4 명세 §1.1 에 따라 DIVA 는 두 사진을 **시계 방향으로 90° 돌려** 보내고, 받은 결과를 반시계 방향으로 90° 되돌려 쓴다. 백엔드에서 보낼 때는 회색조 PNG 로 바꾼다.

| 방향 | 회전 | 코드 |
|---|---|---|
| DIVA → 모델 | 시계 90° | `PIL.Image.transpose(ROTATE_270)` — numpy 로는 `np.rot90(a, -1)` |
| 모델 → DIVA | 반시계 90° | `np.rot90(a, 1)` |

**내보내기 데이터(§6)는 돌리지 않은 DIVA 방향이다.** 새 모델을 어느 방향으로 학습할지 정해야 한다(§8).

### 5.3 리코팅 방향

DIVA 방향에서 리코팅 방향의 기본값은 세로다. 이때 RS 는 세로 줄, RH 는 가로 줄이다. DIVA 는 공정마다 `recoat_axis`(`vertical` 또는 `horizontal`, 기본 `vertical`)를 갖고 있지만, 지금 내보내기의 `index.csv` 에는 이 값이 없다(§8).

---

## 6. 학습 데이터 — DIVA 내보내기 zip

DIVA 의 레이블링 메뉴 「데이터셋」 탭에서 만든다.

```
visible/0/{id}.{jpg|png}    Scanning — DIVA 에 저장된 전처리본 bytes 그대로
visible/1/{id}.{jpg|png}    Deposition
annotations/{id}.npy        int16 (H, W) — 클래스 id 0~9, 미레이블 -1
index.csv                   {id} 가 어느 기관 · 장비 · 공정 · 층인지
classes.json                §3 의 클래스 표 (기계가 읽는 판)
README.txt
```

- **`{id}` 는 그 zip 안에서 `0000000` 부터 매긴 일련번호다.** zip 마다 달라지므로, 같은 층인지는 `index.csv` 의 `agency` · `mc_id` · `pp_id` · `layer` 로 판단한다.
- `index.csv` 의 열은 `id, agency, mc_id, pp_id, process_code, layer, status, labeler, updated_at, width, height, unlabeled_px, swap_sd` 다(UTF-8 BOM). `mc_id` 는 장비, `pp_id` 는 출력 공정(빌드 한 번), `process_code` 는 사람이 읽는 공정 이름이다(예: `PR-2-1365`).
- `classes.json` 에는 `schema`(`"pbf-2026.11"`), `unlabeled`(`-1`), `classes`(클래스마다 `id, key, name, abbr, group, source, color, criteria_en, criteria_ko`), `image_types` 가 있다.
- 사람이 레이블링해서 상태가 완료(`labeled`)나 검수(`reviewed`)인 층만 담긴다. 모델 출력만 있는 층은 담기지 않는다.
- 기본으로는 미레이블 화소가 0 인 층만 담긴다. 내보낼 때 허용하면 −1 이 섞여 들어오므로, 손실에서 −1 을 빼야 한다.
- 두 사진이 모두 있는 층만 담긴다.
- 레이블 맵의 크기는 `index.csv` 의 `width` · `height` 다. 보통 두 사진과 같지만 DIVA 는 크기가 다를 때 경고만 하므로, 로더에서 한 번 확인하는 편이 안전하다. 사진의 채널 수도 보장하지 않는다.
- 층은 골라서 레이블링한다. 공정 하나가 1,000층을 넘어서, 모델이 결함을 본 층 · 앞 층과 달라진 층 · 사람이 고른 층을 대상으로 삼는다. 그래서 클래스 분포가 결함 쪽으로 치우칠 수 있다.
- 내보내기는 train · val 을 나누지 않는다. 이웃한 층은 거의 같은 그림이므로 공정(`process_code`) 단위로 나눠야 검증 점수가 부풀지 않는다.

---

## 7. DIVA 가 DefSegAM 에서 쓰는 것

DIVA 는 세 엔드포인트만 부른다. `GET /` · `/predict` · `/predict/visualize` 는 쓰지 않는다.

| 엔드포인트 | DIVA 가 보내는 것 | DIVA 가 읽는 것 | 클래스에 묶인 가정 |
|---|---|---|---|
| `POST /predict/mask` | `img0` · `img1` | 팔레트로 칠한 RGB PNG | v4 팔레트 11색을 DIVA 코드에 적어 두고 색을 id 로 되돌린다. 같은 색이 없으면 가장 가까운 색으로 읽는다 |
| `POST /predict/batch` | `files` 에 `{층}_0.png` · `{층}_1.png`, 한 요청에 64쌍, `batch_size=16` | `results.zip` 안의 `labels.npz` (키 = 층 번호, 값 = uint8 (H, W)) | 값을 v4 id 0~10 으로 읽는다. `summary.csv` 는 쓰지 않는다 |
| `GET /health` | — | 응답이 오는지 | 없다 |

DIVA 는 클래스 이름 · 클래스 수 · 결함 범위(2~10)를 모델에 묻지 않고 자기 코드에 적어 두었다. 그래서 다음 두 가지가 따라온다.

- **v4 서비스(포트 55304)를 새 모델로 그 자리에서 바꾸면 DIVA 가 조용히 틀린다.** 새 id 2(IS)를 v4 의 2(Recoater Hopping)로 읽고, 새 팔레트 색을 v4 팔레트에서 가장 가까운 색으로 읽는다. 새 모델은 v1~v4 처럼 별도 포트 · 버전으로 올리고, DIVA 쪽 수정과 함께 전환한다.
- 새 모델이 §3 의 id 0~9 를 그대로 내면 번호를 바꾸는 단계가 필요 없다. DIVA 계획이 이것을 전제로 한다. 다만 지금 DIVA 코드는 모델 버전을 `defseg-v4` 하나만 알고 있어서, 새 버전을 붙이려면 DIVA 쪽도 고쳐야 한다.

출력 크기는 지금처럼 올린 사진의 크기여야 하고, batch 의 파일명 규약과 `labels.npz` 구조도 지금과 같아야 한다.

---

## 8. 정해야 할 것

1. **방향.** 새 모델도 90° 돌린 입력을 기준으로 할지, DIVA 방향 그대로 학습할지. 앞쪽이면 학습 전에 내보내기 데이터를 `np.rot90(a, -1)` 로 돌려야 하고 DIVA 는 그대로다. 뒤쪽이면 DIVA 가 돌리는 코드를 빼야 한다.
2. **`/predict/mask` 의 색.** 새 10색을 무엇으로 할지. DIVA 의 표시 색(부록)은 초안이라 바뀔 수 있다. 그래서 팔레트를 그 색에 묶기보다 id 를 그대로 주는 출력(예: 1채널 PNG)을 두자는 것이 DIVA 쪽 제안이다(미확정).
3. **기존 주석의 Swelling · Under Melting.** 미레이블(무시)로 둘지, Swelling 을 Super Elevation 으로 볼지(§4.1).
4. **기존 주석의 IS 가운데 PS 인 영역.** 다시 주석을 달지, DIVA 에서 새로 레이블링한 데이터로만 PS 를 배울지(§4.1).
5. **IS 를 선만 칠할지, 덮이지 않은 영역 전체를 칠할지.** DIVA 팀 결정을 기다리고 있다. 가이드 예시는 영역 전체를 칠했다. 이 결정에 따라 사람이 만든 IS 의 모양이 달라진다.
6. **리코팅 방향이 다른 장비.** RS 와 RH 를 가르려면 방향을 알아야 하는데 내보내기에 `recoat_axis` 가 없다. 필요하면 DIVA 내보내기의 `index.csv` 에 열을 추가할 수 있다.
7. **결함끼리 겹치는 화소.** 가이드에 우선순위가 없다. DIVA 가이드 탭이 권하는 칠하는 순서는 Printed → Deposition 결함(IS · PS · SE) → Scanning 결함(SP · OM) → 줄무늬(RS · RH) → 남은 곳 Powder 이고, 나중에 칠한 것이 앞의 것을 덮는다(잠근 클래스는 덮이지 않는다).
8. **새 버전의 이름과 포트.** 정해지면 DIVA 쪽 설정에도 그 이름을 넣어야 한다.

---

## 9. v4 명세에서 바뀌는 값

DIVA 가 갖고 있는 v4 API 명세에서 클래스에 묶인 값이다.

| 항목 | v4 | 새 체계 |
|---|---|---|
| `n_classes` (`GET /` · `/health`) | 11 | 10 |
| `class_names` (`GET /`) | v4 이름 11개 | §3.1 의 이름 10개, id 순서 |
| `/predict` 의 `classes` 길이 | 11 | 10 |
| 결함 군 (`dominant_defect`, `summary.csv` 의 `defect_fraction`) | id 2~10 | id 2~9 |
| `labels.npz` 의 값 범위 | 0~10 | 0~9 |
| `/predict/mask` 의 팔레트 | 11색 | 미정 (§8) |

---

## 부록 — 코드에 붙여 쓸 값

```python
SCHEMA_VERSION = "pbf-2026.11"

# index = class id
CLASS_NAMES = [
    "Powder", "Printed", "Incomplete Spreading", "Powder Slumping", "Recoater Streaking",
    "Recoater Hopping", "Super Elevation", "Spatter", "Over Melting", "Debris",
]
CLASS_KEYS = [
    "powder", "printed", "incomplete_spreading", "powder_slumping", "recoater_streaking",
    "recoater_hopping", "super_elevation", "spatter", "over_melting", "debris",
]
CLASS_ABBRS = ["", "", "IS", "PS", "RS", "RH", "SE", "SP", "OM", "DB"]
NORMAL_IDS = [0, 1]
DEFECT_IDS = [2, 3, 4, 5, 6, 7, 8, 9]

UNLABELED_NPY = -1     # annotations/*.npy
UNLABELED_PNG = 255    # DIVA 가 저장하는 1채널 PNG (내보내기 zip 에는 없다)

# v4 id -> 새 id. None = 새 체계에 없음 (DIVA 는 미레이블로 둔다)
V4_TO_NEW = {0: 0, 1: 1, 2: 5, 3: 4, 4: 2, 5: None, 6: 9, 7: 6, 8: 7, 9: 8, 10: None}

# DIVA 화면의 표시 색 (RGB, index = class id). 초안이라 바뀔 수 있다
DIVA_COLORS = [
    (200, 200, 200), (70, 120, 220), (236, 112, 170), (150, 82, 222), (232, 52, 52),
    (246, 206, 32), (0, 196, 222), (255, 0, 255), (24, 168, 92), (255, 136, 0),
]
```
