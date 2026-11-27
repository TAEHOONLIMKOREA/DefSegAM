# 합성 데이터로 학습 데이터를 늘리는 문제 — 논문용 정리

> 작성일 2026-11-17. 원고 (`paper/DefectSegAM_Ko/`) 에 옮겨 쓸 **근거 · 수치 · 문헌**을 모은 문서다. 본문 문장이 아니다.
> 표시: **[측정]** 이 repo 에서 잰 값 (출처는 부록 A) · **[문헌]** 논문에서 확인한 내용 (부록 B) · **[추정]** 재지 않은 판단.
> **여기의 실험은 모두 학습 없이 한 사전 점검이다.** 합성 데이터로 분할 모델을 학습시킨 결과는 아직 없다.
> 그 실험의 계획은 [rulesynth/PLAN_RULESYNTH.md](../rulesynth/PLAN_RULESYNTH.md) (규칙 합성) 와 [genaug/PLAN_GENAUG.md](../genaug/PLAN_GENAUG.md) (생성 모델) 에 있다.

**한눈에**

1. 사람 GT 125 레이어로 학습한 v5 는 같은 빌드의 다른 레이어에서는 결함의 9% 를 놓치고, 처음 보는 장비에서는 75% 를 놓친다.
   모자란 것은 사진의 수보다 **장비의 가짓수**다 (§1). [측정]
2. 같은 125 레이어로만 학습한 생성 모델은 그 데이터에 없던 정보를 만들지 못한다.
   정보가 새로 들어오는 길은 셋이다 — 큰 사전학습 모델, 주석 없는 사진, 사람이 주는 규칙 (§3). [문헌 + 논증]
3. 이 데이터의 Super-Elevation (SE) 은 "도포한 뒤에도 부품이 드러나 보이는 것" 이라 두 사진의 관계만으로 만들 수 있다.
   그렇게 만든 SE 를, 실제 사진만으로 학습한 모델이 실제 SE 와 같은 비율로 SE 라고 한다 (합성 96% · 실제 91%, 고치기 전의 같은 화소는 0.00%) (§5). [측정]
4. 그러나 그것은 모델이 이미 아는 것이다. 모델이 알아보지 못하는 합성 SE 는 **SE 예시가 없는 장비**에 있다 (§5). [측정]
5. 주석 없는 사진은 ORNL 5개 빌드 15,825 레이어가 있다. 사람 GT 의 L21 과 기종 · 재료 · 밝기 눈금이 같다.
   다만 **기존 캐시는 레이어마다 밝기를 늘려 저장한 것**이어서, 사람 GT 로만 학습한 모델에는 원본 밝기로 넣어야 한다 (§6.2). [측정]
6. 그 ORNL 사진에 합성한 SE 는 L21 을 학습한 모델도 59–71% 만 알아본다 (같은 규칙이 L21 사진에서는 96%). 주석 없는 사진에는 아직 배우지 않은 것이 있다 (§6.3). [측정]
7. 주석 없는 사진의 "정상" 은 부품 쪽만 규칙으로 가려낼 수 있다. 분말 쪽은 규칙으로 가려낼 수 없다 (§7). [측정]

**용어**

| 말 | 뜻 |
|---|---|
| visible/0 · visible/1 | 한 레이어의 두 사진. visible/0 = 용융 직후, visible/1 = 분말 도포 직후 |
| 부품 자리 | 그 레이어에서 부품이 있는 화소. 사람 GT 에서는 정답이 Printed 인 곳, ORNL 사진에서는 CAD 마스크 (`slices/part_ids > 0`) |
| 합성 | 실제 사진을 규칙대로 고쳐 결함을 만들어 넣는 것. 넣은 자리가 곧 정답이 된다 |
| 인식률 | 합성한 결함 화소 가운데 모델이 그 클래스로 예측한 비율. **높을수록 좋은 점수가 아니다** — 이미 다 알아보는 사진은 가르칠 것이 적다 |
| `inlayer` 모델 / `loso_L21` 모델 | v5 의 두 검증 fold 모델. 앞의 것은 L21 을 포함한 8 소스의 레이어로, 뒤의 것은 L21 을 통째로 빼고 학습했다 |
| 늘린 밝기 / 원본 밝기 | ORNL 사진을 레이어마다 1–99 백분위로 0–255 에 맞춰 늘린 것 (v1 캐시) / HDF5 의 값 그대로 |
| 덮임 판정 | 도포 직후 사진에서 부품 자리가 주변 분말과 구별되지 않으면 "분말에 덮였다 = 정상 (Printed)" 으로 보는 규칙 (§7.1) |

---

## 1. 출발점 — v5 에서 본 것 [측정]

v5 는 사람 GT (Peregrine 데이터셋의 8 소스 125 레이어) 만으로 학습한 11-class 모델이다. 검증 fold 두 개의 결과다 (native 해상도, 마지막 EMA 사본).

| fold | 검증 대상 | 결함 mIoU | 결함 화소 중 놓친 비율 | 정상 화소 중 결함으로 짚은 비율 |
|---|---|---:|---:|---:|
| `inlayer` | 8 소스에서 5개 중 1개씩 22 레이어 (같은 빌드의 다른 레이어는 학습에 있다) | 0.407 (9종) | 9.4% | 2.6% |
| `loso_L21` | 학습에서 통째로 뺀 장비 (L21, ConceptLaser M2) 20 레이어 | 0.178 (5종) | 74.5% | 0.3% |

SE 한 클래스만 보면 같은 양상이다. L21 을 본 적 없는 모델의 SE 재현율은 0.15 (정밀도 0.92, IoU 0.146) 이고,
L21 의 다른 레이어를 본 모델은 L21 4 레이어에서 IoU 0.753, 전체 22 레이어에서 재현율 0.95 (정밀도 0.88, IoU 0.842) 다.

예시가 적은 클래스는 다음과 같다 (1036² 기준 화소 수 / 레이어 수 / 소스 수).

| 클래스 | 사람 GT | 비고 |
|---|---|---|
| Super-Elevation | 259 K / 10 / 3 (L21 6, GP 2, I2 2) | 레이어가 10개뿐 |
| Spatter | 372 K / 11 / 2 | |
| Recoater Hopping | 1.40 M / 25 / 1 (MS) | 한 소스뿐 |
| Over Melting | 24.8 K / 11 / 1 (I2) | 한 소스뿐 |
| Under Melting | 1.2 K / 2 / 1 (GP) | 사실상 학습되지 않는다 |

Peregrine 데이터셋에는 주석 없는 가시광 사진이 더 없다 (소스마다 사진 수 = 주석 수 × 2).

---

## 2. 생성 모델로 학습 데이터를 늘리는 방식 [문헌]

공통점은 **정답이 저절로 따라오도록** 생성한다는 것이다.

| 방식 | 하는 일 | 정답이 나오는 곳 | 대표 문헌 |
|---|---|---|---|
| ① 정답 → 사진 | 정답 지도를 조건으로 사진을 그리는 모델을 학습하고, 정답 지도를 변형해 넣어 새 사진을 뽑는다 | 넣어 준 정답 지도 | pix2pix [1], SPADE [2], ControlNet [3], FreeMask [5] |
| ② 결함 그려 넣기 | 정상 사진의 한 구역에 결함을 그려 넣는다 | 그려 넣은 구역 | DFMGAN [15], AnomalyDiffusion [16], DefectFill [17] |
| ③ 사진과 정답을 함께 | 사진을 그리는 모델의 내부 정보에서 정답 지도를 함께 뽑는다 | 모델의 출력 | DatasetGAN [6], DiffuMask [7] |
| ④ 화질 변환 · 섞기 | 한 도메인의 사진을 다른 도메인처럼 바꾸거나, 정답 있는 조각을 주석 없는 사진에 얹는다 | 원래의 정답 | CycleGAN [8a], CyCADA [8b], DACS [11] |

생성 모델 없이 **규칙으로 만든 가짜 결함**을 학습에 쓰는 흐름도 따로 있다. 정상 사진만으로 이상 탐지기를 학습시키는 연구들이다.

- CutPaste [12]: 사진의 조각을 오려 다른 자리에 붙인 것을 "이상" 으로 삼는다.
- DRAEM [13]: Perlin 잡음을 이진화한 마스크 안에 바깥 질감 데이터셋 (DTD) 의 질감을 불투명도 0.1–1.0 으로 섞는다.
  질감의 출처를 바꾸거나 단색으로 칠해도 성능이 거의 같았고, 저자들이 겨냥한 것은 사실감이 아니라 "분포를 막 벗어난" 모습이다.
- NSA [14]: 다른 정상 사진의 조각을 Poisson 편집으로 매끄럽게 섞는다.
- Copy-Paste [9]: 객체를 무작위 자리에 그대로 붙이는 것만으로 충분하다 (COCO 의 10% 로 학습할 때 box AP +6.9). v4 · v5 의 학습에 이미 들어 있는 증강이다.
- Cut, Paste and Learn [10]: 붙인 경계의 화소 흔적이 검출기를 해치므로 그것을 무시하게 만드는 방법을 함께 제시했다.

어느 방식이든 절차는 비슷하다: (1) 생성기 준비 — 데이터가 적으면 큰 사전학습 모델을 미세 조정 (LoRA [19a], DreamBooth [19b]) →
(2) 대량 생성 → (3) 걸러 내기 → (4) 실제 데이터와 섞어 학습 → (5) 실제 데이터로만 평가.

**걸러 내기가 결과를 가른다.** FreeMask [5] 는 실제 사진으로 학습한 분할 모델의 화소별 손실이 그 클래스 평균의 1.25배를 넘는 합성 화소를 버린다.
이 걸러 내기와 재추출 없이 합성 사진만으로 학습하면 ADE20K mIoU 가 43.3–43.7 에 머물고 (걸러 내면 48.3, 실제 사진은 48.5) 사진을 늘려도 오르지 않았다.
DatasetGAN [6] 도 생성한 10,000 쌍 가운데 불확실한 10% 를 버린다.

**데이터가 적은 분할 문제에서의 효과는 엇갈린다.**

| 연구 | 조건 | 결과 |
|---|---|---|
| Yu 등 [B4] | 세포핵 분할. 생성기를 라벨 있는 부분집합만으로 학습 (MoNuSeg 30장, Kumar 16장) | MoNuSeg 라벨 10%: Dice 0.797 → 0.829. 전부 사용: 0.821 → 0.834. Kumar 50%: 0.8175 → 0.8158 (내려감). 기준선이 일반 증강을 썼는지는 확인하지 못했다 |
| GenSeg [B5] | 의료 영상 9–50장. 생성기를 분할 모델의 검증 손실로 함께 최적화 | 일반 증강 대비 Dice 0.66 → 0.74 (발 궤양), 0.69 → 0.76 (용종). 생성기를 따로 학습하면 뚜렷이 나빠진다 (0.52 → 0.42) |
| Vu 등 [B6] | 의료 영상 18–500명. StyleGAN2 로 마스크와 사진 생성 | "분할 모델은 생성한 사진에서 이득을 얻지 못했다" (arXiv, 학술지 게재는 확인하지 못함) |

**분말 베드 레이어 사진의 결함 분할에 합성 · 생성 사진을 쓴 연구는 찾지 못했다** (웹 검색 10여 회. Scopus · Web of Science 는 쓰지 못했으므로 없다는 증명은 아니다).
가까운 것은 다음 셋이다: GAN 으로 증강한 in-situ 결함 검출 틀 [B1] (초록만 확인. 분할인지 분류인지, 이득의 크기는 확인하지 못함),
L-PBF 열화상의 레이어 단위 **분류**에서 결함 화소 붙이기 · 단순 반복 추출 · GAN 을 견준 연구 [B2] (세 방법의 정확도가 비슷했다. 예: HR-1 에서 92.0 → 98.5–99.1),
확산 모델로 레이어 사진의 해상도를 올린 연구 [B3] (학습 데이터를 만든 것은 아니다).

DSCNN 원 논문 [26] 이 쓴 증강은 Gaussian 잡음 (동적 범위의 0.01% · 0.1%) 과 평균 밝기 이동 (±10%) 둘뿐이다. 회전은 정답 클래스를 바꿀 수 있다는 이유로 쓰지 않았다.

---

## 3. 같은 데이터로 학습한 생성 모델의 한계 — 정보가 들어오는 세 통로

**논증.** 생성 모델은 학습 때 본 분포를 흉내 낸다. 분할 모델이 이미 가진 125 레이어로만 생성기를 학습시키면, 생성한 사진이 담은 것은 그 125 레이어의 정보를 다시 섞은 것이다.
그래도 효과가 조금 나는 경우는 정보가 늘어서가 아니라 섞임 덕분이다 — 결함과 특정 배경이 한 덩어리로 외워지는 것을 깨고, 드문 클래스의 비중을 올린다.
좌우 뒤집기가 새 정보 없이 도움이 되는 것과 같은 성격이다.

**문헌이 보여 주는 것.**

- 같은 데이터로 학습한 생성기의 표본만으로 학습하면 크게 떨어진다. ImageNet 으로 학습한 BigGAN-deep 의 표본만으로 ResNet-50 을 학습하면 top-1 이 73.09% → 42.65% 다 (VQ-VAE-2 는 54.83%).
  실제 데이터에 섞어도 이득은 한 설정에서 top-5 0.2% 뿐이었다 [20].
- 이득이 보고된 경우에는 생성기가 **바깥 데이터**를 봤다. Imagen (대규모 사전학습 후 ImageNet 미세 조정) 의 사진 120만 장을 더하면 ResNet-50 top-1 이 76.39 → 78.17 로 오르지만,
  합성이 실제의 4배쯤을 넘으면 오히려 내려간다 [22]. FreeMask 의 +3.3 mIoU (ADE20K, Mask2Former Swin-T 48.7 → 52.0) 도 Stable Diffusion 에서 출발한 생성기의 결과이고, 큰 모델 (Swin-L, 합성으로 먼저 학습하는 방식) 에서는 56.0 → 56.4 에 그친다 [5].
- 생성기의 사전학습 데이터에서 실제 사진을 **검색**해 넣는 것이 생성보다 낫거나 같다. ImageNet 10% 에서 증강 없음 57.2, 가장 좋은 확산 방법 61.2, LAION-5B 검색 62.6 [23].
  저자들은 "검색이 더 많은 정보에 닿기 때문일 수 있다" 고만 적었다 — 원인을 증명한 것은 아니다.
- 생성기가 가진 사전 지식을 지우면 이득이 준다. DA-Fusion 의 이득 (few-shot 분류, 최대 +10%p) 은 모델 가중치에서 그 클래스의 지식을 지우면 최대 +5%p 로 줄었다 [18].
  위 주장에 대한 가장 직접적인 근거다.
- 합성 데이터는 실제 데이터가 적을 때와 분포 밖에서 가장 도움이 되지만, 지도학습 분류기에서는 실제 사진보다 뚜렷이 못하다 [21, 24].
- 생성 → 학습 → 생성을 되풀이하면 품질이나 다양성이 무너진다 (이미지 생성 모델에 대해서는 [25b], 언어 모델 · VAE · 가우시안 혼합에 대해서는 [25a]).

**세 통로.** 생성이 의미가 있으려면 바깥에서 정보가 들어와야 한다.

| 통로 | 들어오는 것 | 문헌의 예 | 이 데이터에서 |
|---|---|---|---|
| 큰 사전학습 모델 | 사진 일반의 질감 · 조명 | FreeMask [5], DA-Fusion [18], AnomalyDiffusion [16], DefectFill [17] | 분말 베드 사진과의 거리가 멀다. 쓸 수 있는지는 확인이 필요하다 ([PLAN_GENAUG.md](../genaug/PLAN_GENAUG.md) §3.7) |
| 주석 없는 사진 | 그 장비 · 그 공정의 정상 모습과 부품 형상 | DatasetGAN [6] (StyleGAN 을 주석 없는 대량의 사진으로 학습), DACS [11], CyCADA [8b] | ORNL 5개 빌드 15,825 레이어 (§6) |
| 사람이 주는 규칙 | 결함의 정의 — 무엇이 어디에 어떻게 보이는가 | CutPaste [12], DRAEM [13], NSA [14] | SE 의 정의 (§4) |

논문에서 이 절을 쓸 때의 주의: "정보가 늘지 않는다" 는 것은 논증이고, 인용한 실험들은 그것과 **맞는 방향의 관찰**이다. 분류 문제의 결과가 대부분이고,
데이터가 적은 분할 문제에서는 이득을 보고한 연구도 있다 (§2 의 표). "항상 소용없다" 로 쓰면 틀린다.

---

## 4. 이 데이터에서 규칙으로 만들 수 있는 결함

### 4.1 데이터셋의 정의

Peregrine v2021-03 데이터셋의 설명서 (`ORNL_Data/DSCNN_Dataset/Peregrine Dataset v2021-03/readme.pdf`, 2021-04-23) 가 적은 정의다.

- **Super-Elevation** — "is evident when a part extends above the powder bed as the result of poor powder coverage or warping from residual thermal stresses."
- **Swelling** — "is evident when a part warps up above the powder bed as the result of excessive local energy input."
- **Incomplete Spreading** — "is present when an insufficient amount of powder is spread across the powder bed in one or more layers."
- **Recoater Streaking** — "… characterized by a line parallel to the recoater travel direction."
- **Recoater Hopping** — "… characterized by repeated lines perpendicular to the recoater travel direction."

SE 와 Swelling 은 둘 다 "부품이 분말 위로 올라와 있다" 이고 원인만 다르다. 사진에서는 **도포 직후 사진 (visible/1) 에 부품이 보인다**는 한 가지로 나타난다.

### 4.2 실제 예시를 확대해서 본 것 [측정 — 눈으로 본 관찰]

| 클래스 | 두 사진에서의 모습 | 규칙으로 만들 수 있는가 |
|---|---|---|
| Super-Elevation | visible/1 에 부품이 드러난다. GP 는 밝은 얼룩이 촘촘한 정도로, L21 은 분말이 닿지 않은 띠 안의 부품이 통째로, I2 는 포화된 밝은 덩어리로 | **된다** — visible/0 의 부품 모습을 visible/1 에 비치게 한다 |
| Swelling | visible/1 에서 부품 가장자리의 작은 점 · 짧은 호 | 되지만 SE 와의 경계를 규칙으로 맞추지 못했다 (§5) |
| Incomplete Spreading | MS · H13 은 거친 질감의 넓은 영역, L21 은 판 아래쪽 띠 | 안 된다 — 질감이 다르고 소스마다 칠한 범위가 다르다 |
| Recoater Streaking | H13 · MS 는 확대해도 거의 보이지 않는다. BJ21 은 어두운 세로 직사각형 | 안 된다 |
| Recoater Hopping | MS 한 소스. 두 사진 모두에 가는 세로 물결선 | 안 된다 |
| Debris | L21 은 visible/0 의 부품 윤곽을 따라, H13 은 visible/1 의 어두운 초승달 | 안 된다 — 소스마다 다른 것을 가리킨다 |
| Spatter | 부품 사이의 넓고 흐린 영역 | 안 된다 |
| Over · Under Melting | visible/0 에서 부품 표면의 질감. UM 의 주석은 GP 에서 분말 위의 가는 사선 세 줄 | 안 된다 |

그림: [real_SE_Sw_IS.png](../figures/synth_prototype/real_SE_Sw_IS.png), [real_RS_RH_DB.png](../figures/synth_prototype/real_RS_RH_DB.png),
[real_SP_OM_UM.png](../figures/synth_prototype/real_SP_OM_UM.png), [real_RS_zoom.png](../figures/synth_prototype/real_RS_zoom.png)
(열: visible/0 · visible/1 · 정답 윤곽).

### 4.3 합성 규칙 (시제품)

S 는 사진 한 변 (1036 화소), P 는 부품 자리다. visible/0 은 건드리지 않는다.

1. **노출 확률 장** cov(x) ∈ [0, 1] 을 P 안에 만든다.
   - 덩어리형 (blob): 부품 위의 한 점을 중심으로 한 타원. 반지름 0.05–0.22 S, 축비 0.6–1.7, 경계에 저주파 잡음.
     cov = p · clip(1.25 − d, 0, 1)^γ, 최고값 p 는 0.45–1.0, γ 는 0.6–2.2.
   - 띠형 (band): 판의 위쪽 또는 아래쪽 가장자리에서 폭 0.05–0.30 S. 경계는 사인파 셋의 합. cov = 0.85–1.0. 분말이 닿지 않은 구역 안의 부품에 해당한다.
2. **얼룩** α(x): 화소마다 균등 난수 u 를 뽑아 u < cov 인 곳을 1 로 한다. 알갱이 크기 (0–0.9 화소) 만큼 난수를 흐려 분말 알갱이의 굵기를 흉내 낸다.
3. **사진**: visible/1′ = (1 − α) · visible/1 + α · g · visible/0, g 는 0.90–1.08.
4. **정답**: α 를 σ 1.5 로 흐린 밀도가 0.45 를 넘는 곳이 SE (12 화소 미만의 조각은 버린다). 밀도 0.08–0.45 의 듬성듬성한 띠는 **미레이블** (−1) —
   사람이 어디까지 칠했을지 알 수 없는 곳은 어느 쪽으로 예측해도 틀렸다고 하지 않는다.

Swelling 은 부품 가장자리 1–3 화소의 테에 반경 0.008–0.05 S 의 점 2–8 개를 같은 방법으로 넣는다.

속도는 표본 하나에 0.22초 (SE) · 0.16초 (Swelling), 한 표본이 만드는 SE 화소의 중앙값은 6,844 개다 (CPU 1코어, 8 소스 16 레이어에서 64회 시도). [측정]

**밝혀 둘 것.** 덩어리형과 띠형이라는 두 모양은 GP · L21 (2 레이어) · I2 의 실제 SE 예시를 본 뒤에 정했다. 규칙이 L21 의 주석을 전혀 모르고 만들어진 것은 아니다.
정의 자체 ("poor powder coverage" · "warping") 는 공개 설명서의 것이다.

---

## 5. 사전 점검 1 — 사람 GT 레이어 위의 합성 [측정]

**물은 것**: 실제 사진만으로 학습한 모델이 합성 결함을 실제 결함처럼 보는가.

**방법**: `inlayer` fold 의 검증 22 레이어 (그 모델이 본 적 없는 레이어) 에 SE 31개 · Swelling 24개를 합성했다 (부품 화소가 모자란 H13 레이어에서는 만들어지지 않았다).
그 가운데 L21 4 레이어는 두 모델 모두 본 적이 없다. 1036², TTA · 후처리 없이 CPU 로 추론했다.
"대조" 는 합성한 화소를 **고치기 전 사진**에서 무엇으로 예측했는가다.

| 모델 · 사진 | 실제 SE → SE | 합성 SE → SE | 합성 SE 의 나머지 |
|---|---:|---:|---|
| `inlayer`, L21 4 레이어 | 90.8% (1 레이어, 13.5 K px) | **95.7%** (15개, 354 K px) | Swelling 2.9%, Printed 1.0% |
| `inlayer`, EOS 4 소스 (17-4 · GP · I1 · I2) | 95.6% (GP 1 레이어, 55.8 K px) | **99.1%** (7개, 48 K px) | |
| `inlayer`, Maraging (MS) | 실제 예시 없음 | 1.0% (5개, 27 K px) | IS 56.6%, Printed 33.4% |
| `inlayer`, Binder Jet (BJ21) | 실제 예시 없음 | 10.4% (4개, 29 K px) | IS 47.5%, Printed 40.0% |
| `loso_L21`, L21 4 레이어 | 30.4% (같은 1 레이어) | 31.0% (덩어리형 67.5%, 띠형 15.9%) | 띠형: IS 36.6%, Powder 22.2%, Printed 16.1% |

대조 (`inlayer`, L21 + EOS 의 합성 화소 402 K 개를 고치기 전 사진에서): SE 0.00%, Swelling 0.5%, Printed 99.1%.

| Swelling (`inlayer`, L21 + EOS) | → Swelling | → SE | → Printed |
|---|---:|---:|---:|
| 실제 (12 레이어, 16.4 K px) | 80.9% | 0.3% | 14.5% |
| 합성 (15개, 6.9 K px) | 58.0% | 30.0% | 9.3% |

**읽는 법**

- L21 과 EOS 소스에서는 합성 SE 가 실제 SE 와 같은 비율로 SE 로 예측되고, 고치기 전에는 0% 다. **규칙이 만든 것이 모델이 실제 예시에서 배운 SE 와 같은 종류**라는 뜻이다.
- 같은 사실의 다른 면: 이 소스들에서는 **모델이 이미 아는 것**이다. 같은 소스에 합성을 더 넣어도 새로 배울 것이 적다 [추정 — 학습으로 확인하지 않았다].
- 모델이 알아보지 못하는 것은 SE 예시가 없는 장비다. L21 을 본 적 없는 모델은 L21 의 실제 SE 도 합성 SE 도 30% 만 알아본다. SE 예시가 EOS 소스 (GP · I2) 에만 있을 때 그 지식은 같은 기종 (17-4 · I1) 까지만 넘어간다.
- **소스마다 정답 규칙이 다르다.** MS 와 BJ21 에서 모델은 합성 SE 의 절반을 Incomplete Spreading 으로 냈다.
  그 소스들의 주석이 분말이 덮이지 않은 구역을 부품까지 IS 로 칠한 것으로 보인다 [추정 — 주석을 전수 확인하지 않았다]. 이런 소스에 SE 를 합성해 넣으면 기존 정답과 충돌한다.
  Binder Jet 의 클래스 목록에는 SE 가 없다.
- Swelling 은 SE 와의 경계가 규칙과 주석 사이에서 어긋난다 (합성의 30% 가 SE 로 예측됨). 두 클래스를 합치는 체계 (v6 의 `pbf-2026.11`) 에서는 문제가 되지 않는다.

그림: [fig_se_sources.png](../figures/synth_prototype/fig_se_sources.png) (실제 SE 와 합성 SE),
[fig_new_sources.png](../figures/synth_prototype/fig_new_sources.png) (SE 주석이 없는 소스),
[fig_swelling.png](../figures/synth_prototype/fig_swelling.png), [fig_loso.png](../figures/synth_prototype/fig_loso.png) (L21 을 본 적 없는 모델).
열: visible/0 · visible/1 원본 · visible/1 합성 · 정답 (빨강 = 결함, 노랑 = 미레이블) · 모델의 예측.

---

## 6. 사전 점검 2 — ORNL 빌드의 주석 없는 사진 [측정]

### 6.1 재료

ORNL Co-Registered 데이터셋의 5개 빌드다. 장비는 Concept Laser M2, 재료는 SS 316L, 카메라는 5 MP 가시광 — 사람 GT 의 L21 (Peregrine v2021-03 의 Laser Powder Bed Fusion) 과 같다.

| 빌드 | 캐시된 레이어 (1036²) | 특징 (데이터셋 README) |
|---|---:|---|
| B1 | 3,218 | 기준 공정 조건 |
| B2 | 3,204 | 공정 파라미터를 여러 가지로 바꿈 |
| B3 | 2,968 | 오버행 형상 |
| B4 | 3,217 | 스패터 · 아르곤 유량 변화 |
| B5 | 3,218 | 리코터 손상 · 분말 공급 부족 |
| 합 | 15,825 | 사람 GT 125 레이어의 127배 |

사람이 칠한 정답은 없다. 있는 것은 사진 두 장과 CAD 마스크 (`slices/part_ids`, 사진과 같은 1842² 격자) 다.
같은 파일의 `slices/segmentation_results` 는 DSCNN 의 예측이라 v5 · v6 의 원칙대로 쓰지 않는다.

### 6.2 캐시의 밝기 — 사람 GT 와 다르게 저장되어 있다

HDF5 의 사진 값은 0–255 의 정수이고 (B1 · B2 · B5 의 6 레이어에서 확인), 사람 GT 의 L21 사진과 눈금이 같다.

| | visible/1 의 분말: 평균 / 표준편차 | visible/0 의 부품 평균 / 분말 평균 |
|---|---|---|
| ORNL HDF5 원본 (B1 · B2 · B5, 6 레이어) | 92–103 / 3.6–5.0 | 130–164 / 91–100 |
| 사람 GT L21 (3 레이어) | 93–105 / 3.0–4.8 | 133–174 / 89–106 |

그런데 v1 이 만든 이미지 캐시 (`cache/resized_sz1036/`) 와 추론 코드 (`ornl_image_to_uint8`) 는 **레이어마다 1–99 백분위를 0–255 로 늘린다.**
도포 직후 사진은 값의 폭이 좁아 (예: B2 레이어 390 의 1–99 백분위가 86–111) 대비가 약 10배로 커진다. 사람 GT 의 사진에는 이 처리가 없다.

같은 20 레이어 (B2–B5 에서 5개씩) 를 두 밝기로 v5 모델에 넣어, CAD 부품 자리와 모델의 예측을 견준 결과다.

| `inlayer` 모델 | 늘린 밝기 (캐시) | 원본 밝기 |
|---|---:|---:|
| CAD 부품 자리와 "부품 계열" 예측의 IoU (평균 / 최저) | 0.475 / 0.143 | **0.816** / 0.667 |
| 부품 밖 화소를 Printed 로 예측한 비율 | 11.6% | 1.2% |
| 부품 자리 안: Printed / Powder / Swelling / SE | 90.1 / 4.0 / 3.3 / 0.9% | 90.3 / 5.5 / 2.2 / 0.3% |

`loso_L21` 모델도 같다 (IoU 0.465 → 0.767, 부품 밖의 Printed 11.5% → 2.2%).

Build 1 에서도 같다. v5 의 전수 학습 모델 (`full/last.pt`, 2026-11-17) 을 그림에 쓰인 12 레이어에 두 밝기로 넣은 결과다 (native 채점, TTA · 후처리 없음).

| v5 전수 학습 모델, Build 1 12 레이어 | 늘린 밝기 | 원본 밝기 |
|---|---:|---:|
| DSCNN 의 Printed 예측과의 IoU (정밀도 / 재현율) | 0.562 (0.569 / 0.978) | **0.945** (0.958 / 0.986) |
| CAD 부품 자리와 "부품 계열" 예측의 IoU | 0.547 | 0.866 |
| 부품 밖 화소를 Printed 로 예측한 비율 | 4.33% | 0.50% |

같은 날 `confusion` phase (늘린 밝기, Build 1 전체 3,218 레이어) 가 낸 Printed IoU 는 0.514 (정밀도 0.522, 재현율 0.972) 였다. v4 의 같은 표는 0.914 다.
DSCNN 의 예측은 정답이 아니므로 0.945 도 성능이 아니라 일치도다. 여기서 읽을 것은 밝기 하나로 0.56 이 0.95 가 된다는 점이다.

- **사람 GT 로만 학습한 모델 (v5 · v6) 에는 ORNL 사진을 원본 밝기로 넣어야 한다.** 늘린 밝기에서는 분말 영역의 넓은 부분을 Printed 로 낸다.
- v1–v4 는 이 차이를 겪지 않았다. Stage 1 을 늘린 밝기의 ORNL 사진으로 학습했기 때문이다 (Stage 2 의 사람 GT 는 늘리지 않은 사진이었다).
- v5 의 Build 1 참고 평가 (`confusion` · `infer`) 는 지금 늘린 밝기를 쓴다. 그 수치와 그림을 v5 의 성능으로 읽으면 안 되고, 원본 밝기로 다시 내야 한다 (아래 표).
- 원본 밝기에서 CAD 마스크와 모델의 Printed 예측이 0.82 로 겹친다는 것은 CAD 와 사진의 정합이 맞다는 뜻이기도 하다 ([fig_ornl_raw_blob.png](../figures/synth_prototype/fig_ornl_raw_blob.png) 의 초록 윤곽).

### 6.3 ORNL 사진 위의 합성 SE — 원본 밝기

같은 20 레이어의 CAD 부품 자리에 SE 를 덩어리형 · 띠형 하나씩 합성했다 (각 20개, 150 K · 412 K px).

| 모델 | 덩어리형 → SE | 띠형 → SE | 나머지 (덩어리형) | 고치기 전의 같은 화소 → SE |
|---|---:|---:|---|---:|
| `inlayer` (L21 을 학습함) | **59.3%** | **70.8%** | IS 18.1%, Printed 8.4% | 0.01% · 0.34% |
| `loso_L21` (L21 을 본 적 없음) | 29.0% | 31.4% | Printed 45.5% | 0.00% |

| `inlayer` 모델, 빌드별 | B2 | B3 | B4 | B5 |
|---|---:|---:|---:|---:|
| 덩어리형 → SE | 27.2% | 32.7% | 75.4% | 89.8% |
| 띠형 → SE | 49.3% | 83.4% | 79.4% | 80.9% |

합성 화소를 visible/0 에서의 부품 밝기로 나누면 차이의 한 원인이 보인다 (주변 분말보다 3σ 이상 밝음 / 그 사이 / 3σ 이상 어두움).

| `inlayer` 모델 | 밝은 부품 | 그 사이 | 어두운 부품 |
|---|---:|---:|---:|
| 덩어리형 → SE (화소의 몫) | 70.8% (67%) | 39.3% (28%) | 16.9% (5%) |
| 띠형 → SE (화소의 몫) | 80.7% (73%) | 46.5% (25%) | 25.1% (2%) |

- 같은 규칙이 사람 GT 의 L21 사진에서는 96% 였다. **기종이 같아도 ORNL 빌드의 사진에는 L21 20 레이어에 없던 것이 있다** —
  용융 직후 사진에서 부품이 분말보다 어둡게 찍힌 경우가 그 하나다 (부품 화소의 0.5–8.5%, B3 이 가장 많다).
- 사람 GT 의 SE 는 모두 밝은 부품이었다. 어두운 부품이 드러난 모습이 실제로 어떻게 찍히는지는 정답이 없어 알 수 없다 [추정: 드러난 부품은 visible/0 과 같은 모습일 것이다].
  밝은 부품만으로 좁혀도 B2 (공정 파라미터를 바꾼 빌드) 의 덩어리형은 26.9% 다.
- L21 을 본 적 없는 모델은 29–31% 다. 주석 없는 사진에 합성한 SE 가 이 모델에게는 대부분 새롭다.
- 늘린 밝기의 캐시에서는 인식률이 더 낮았다 (`inlayer` 32.7% · 45.5%, `loso_L21` 2.5% · 7.8%). 입력이 학습 때와 달라서다 (§6.2).

그림: [fig_ornl_raw_blob.png](../figures/synth_prototype/fig_ornl_raw_blob.png), [fig_ornl_raw_band.png](../figures/synth_prototype/fig_ornl_raw_band.png),
[fig_ornl_stretched_blob.png](../figures/synth_prototype/fig_ornl_stretched_blob.png) (늘린 밝기 — 견주어 보기용).
왼쪽 세 칸은 전체 프레임 (visible/0, CAD 윤곽, visible/1 위의 예측), 오른쪽 네 칸은 합성 구역의 확대다.

---

## 7. 주석 없는 사진의 "정상" 을 규칙으로 가려낼 수 있는가 [측정]

합성 표본에서 결함만 정답으로 주면, 모델이 그 장비의 사진에서는 무엇이든 결함으로 내도 손실이 0 이다. 정상 화소의 정답이 함께 있어야 한다.
그런데 주석 없는 사진에는 진짜 결함이 섞여 있고 그 자리를 모른다. 그래서 "확실히 정상인 곳" 만 규칙으로 고를 수 있는지를 사람 GT 125 레이어로 쟀다.

### 7.1 부품 쪽 — 된다 (덮임 판정)

규칙: 도포 직후 사진에서, 주변 분말 (부품 자리에서 3 화소 넘게 떨어진 화소, 61×61 창) 의 평균에서 3σ 안에 있는 부품 화소는 덮인 것으로 본다.
벗어난 화소와 그 둘레 3 화소, 부품 가장자리 2 화소는 판정하지 않는다.

| 소스 | Printed 중 "덮임" | SE 중 "덮임" (오판) | Swelling 중 "덮임" (오판) | "덮임" 화소 중 SE · Sw · OM · UM |
|---|---:|---:|---:|---:|
| L21 | 58.3% | 0.3% | 0.5% | 0.04% |
| 17-4 | 48.0% | – | 0.2% | 0.00% |
| GP | 56.9% | 0.0% | 0.0% | 0.16% |
| I1 | 51.9% | – | 0.1% | 0.00% |
| I2 | 38.7% | 0.0% | 0.2% | 1.27% |
| MS | 56.1% | – | 15.4% | 0.22% |
| 전체 (BJ 포함 122 레이어) | 48.9% | **0.2%** | 1.7% | 0.24% |

- Printed 화소의 절반에 정상 라벨을 줄 수 있고, 그 가운데 실제로 결함인 것은 0.24% 다 (문턱을 2σ · 4σ 로 바꾸면 28.9% · 55.1% 와 0.31% · 0.24%).
- Over · Under Melting 은 54% 가 "덮임" 으로 나온다. 이 둘은 용융 직후 사진의 질감으로 정해지는 결함이라 이 규칙으로 걸러지지 않는다 (I2 의 1.27% 가 이것이다).
- 여기서 부품 자리는 사람 GT 에서 만들었다. ORNL 사진에서는 CAD 마스크를 쓰므로 같은 정확도가 나오는지는 따로 확인하지 못했다 (정답이 없다).

### 7.2 분말 쪽 — 안 된다

같은 방식 (두 사진 모두에서 주변 분말의 3σ 안) 으로 "확실히 정상인 분말" 을 고르면, 분말 쪽 결함의 대부분이 정상으로 들어온다.

| GT 가 이 클래스인 화소 중 "정상 분말" 로 판정된 비율 | Powder | RH | RS | IS | Debris | Spatter |
|---|---:|---:|---:|---:|---:|---:|
| 전체 125 레이어 | 89.4% | 87.5% | 82.7% | 96.0% | 50.3% | 81.3% |

"정상 분말" 로 고른 화소의 13.0% 가 실제로는 결함이다 (MS 40.7%, H13 18.4%, L21 4.8%).
줄무늬와 스패터는 대비가 약하고, Incomplete Spreading 은 넓어서 주변 통계 자체가 결함이다.

→ 주석 없는 사진의 부품 밖 화소에 대해 CAD 가 말해 주는 것은 **"부품 계열 클래스 (Printed · Swelling · SE · Over · Under Melting) 가 아니다"** 까지다.
그 화소가 Powder 인지 분말 쪽 결함인지는 알 수 없다. 학습에서는 "이 다섯 클래스가 아니다" 라는 형태의 정답 (후보 집합 라벨 [27a], 보완 라벨 [27b]) 으로 쓸 수 있다.

---

## 8. 논문에 쓸 수 있는 것 · 아직 쓸 수 없는 것

| 주장 | 상태 | 근거 | 쓸 때의 주의 |
|---|---|---|---|
| 처음 보는 장비에서 결함의 75% 를 놓치고, 같은 빌드에서는 9% 를 놓친다 | 쓸 수 있다 | §1 | fold 하나씩, seed 하나. `inlayer` 는 낙관적인 값이다 |
| 같은 데이터로만 학습한 생성기는 새 정보를 주지 못한다 | 논증으로 쓸 수 있다 | §3 | 실험 근거는 주로 분류 문제. 소량 데이터 분할에서는 이득을 보고한 연구도 있다 |
| SE 는 두 사진의 관계로 정의되어 규칙으로 합성할 수 있다 | 쓸 수 있다 | §4.1, §5 | Swelling 과의 경계는 맞추지 못했다 |
| 규칙으로 만든 SE 를 실제 사진으로 학습한 모델이 실제와 같은 비율로 인식한다 (96% 대 91%, 대조 0%) | 쓸 수 있다 | §5 | 22 레이어 · 합성 31개. 실제 SE 의 기준은 레이어 1–2개다. 인식률은 "쓸모" 의 척도가 아니다 |
| ORNL 캐시의 밝기 처리가 사람 GT 와 다르고, 원본 밝기로 넣어야 한다 | 쓸 수 있다 | §6.2 | 20 레이어. v1–v4 의 기존 수치는 늘린 밝기에서 낸 것이다 |
| 주석 없는 사진의 정상 라벨은 부품 쪽만 규칙으로 줄 수 있다 | 쓸 수 있다 | §7 | 부품 자리를 사람 GT 에서 만든 경우의 값이다 |
| 주석 없는 사진에 규칙으로 결함을 넣으면 그 장비에서의 성능이 오른다 | **아직 쓸 수 없다** | — | 학습 실험이 없다. [PLAN_RULESYNTH.md](../rulesynth/PLAN_RULESYNTH.md) §6.2 의 RS-BR 이 재는 것이다 |
| 주석 없는 사진과 규칙 가운데 어느 쪽이 얼마나 기여하는가 | **아직 쓸 수 없다** | — | 같은 계획의 RS-B · RS-R |
| 분말 베드 레이어 사진에서 합성 데이터로 분할 모델을 학습한 선행 연구가 없다 | 조심해서 | §2 | 웹 검색만 했다. "찾지 못했다" 로 쓰고 Scopus 등에서 한 번 더 확인할 것 |

## 9. 한계

- **학습 실험이 없다.** 이 문서의 수치는 모두 이미 학습된 두 모델에게 합성 사진을 보여 준 결과다.
- **인식률은 사실감의 증거로도 약하다.** 모델이 SE 라고 했다는 것은 모델이 배운 SE 와 닮았다는 뜻이지 실제 SE 와 같다는 뜻이 아니다. 모델이 못 알아본 합성 (MS, 어두운 부품) 이 틀린 그림인지 새로운 정보인지는 가릴 수 없다.
- **표본이 작다.** 사람 GT 쪽은 22 레이어 · 합성 55개, ORNL 쪽은 20 레이어 · 합성 40개다. 실제 SE 의 기준값은 L21 1 레이어 · GP 1 레이어에서 나왔다. seed 는 하나다.
- **규칙은 L21 의 예시를 보고 만들었다** (§4.3). `loso_L21` 로 효과를 잴 때 이 점을 함께 적어야 한다.
- **소스마다 정답 규칙이 다르다** (MS · BJ). 규칙 합성은 SE 를 따로 칠한 소스 (L21 · EOS) 와 그 기종에만 맞는다.
- **ORNL 사진에는 사람 정답이 없다.** 그 위의 합성이 맞는지, CAD 로 만든 정상 라벨이 얼마나 맞는지 직접 잴 수 없다.
- 추론은 CPU 에서 1036², TTA · 후처리 없이 했다 (GPU 는 학습이 쓰고 있었다).

---

## 부록 A. 수치가 나온 곳

모든 스크립트는 `scripts/synth_prototype/` 에 있고 `3DP_VPPM/` 에서 `python -m DefSeg_AM.scripts.synth_prototype.<이름>` 으로 실행한다 (`check_b1_brightness` 말고는 GPU 를 쓰지 않는다).
그림과 수치 기록은 `figures/synth_prototype/`, 중간 배열은 `cache/synth_prototype/` (git 이 무시하는 폴더) 에 있다.

| 절 | 스크립트 | 결과 파일 |
|---|---|---|
| §1 | — | `figures/vitb14_reg_dpt_dual_sz1036_11cls_v5/v5/cv/` 의 `summary.md`, fold 별 `eval_native_last/{per_class_metrics.csv, confusion_matrix.csv}`. 놓친 비율 · 짚은 비율과 클래스별 예시 수는 [PLAN_v6.md](../v6/PLAN_v6.md) §1.1 · [PLAN_v5.md](../v5/PLAN_v5.md) §2 에 정리된 값 |
| §4.2 | `look_real` | `real_*.png` |
| §4.3 | `synth_rules` (규칙), 속도는 2026-11-17 에 따로 잰 값 | — |
| §5 | `eval_on_gt` → `fig_on_gt` | `records_eval_gt.json`, `fig_{se_sources, new_sources, swelling, loso}.png` |
| §6.2 · §6.3 | `preflight_ornl` (인자 `raw` = 원본 밝기, 없으면 캐시) → `fig_ornl` | `records_ornl_raw.json`, `records_ornl_stretched.json`, `fig_ornl_*.png`. 밝기 통계와 밝은 · 어두운 부품의 구분은 2026-11-17 에 같은 배열에서 따로 계산 |
| §6.2 의 Build 1 표 | `check_b1_brightness` (GPU 1대) | 표준 출력. `confusion` phase 의 값은 `figures/vitb14_reg_dpt_dual_sz1036_11cls_v5/v5/full/build1_vs_dscnn/per_class_metrics.csv` |
| §7.1 | `covered_rule` (인자: 문턱 · 창 · 여유) | 표준 출력 |
| §7.2 | `normal_powder_rule` | 표준 출력 |

- 모델: `checkpoints/vitb14_reg_dpt_dual_sz1036_11cls_v5/cv/{inlayer, loso_L21}/last.pt` (EMA 사본, 10,000 step, seed 42).
- 합성의 난수는 (20261117, 레이어 번호, 변형 번호) 로 고정했다. 같은 명령이 같은 수치를 낸다.
- ORNL 20 레이어: B2 – B5 의 캐시 행에서 seed 20261117 로 15개씩 뽑아, 부품 자리가 1% 이상인 앞의 5개.
- 장비 · 재료: ORNL 쪽은 `ORNL_Data/Co-Registered …/[baseline] (Peregrine v2023-11)/README.md`, L21 쪽은 §4.1 의 설명서.

## 부록 B. 참고문헌

서지 정보는 2026-11-17 에 arXiv 초록 페이지 · Crossref 기록 · 학회 공개 페이지에서 확인했다. "확인" 은 저자 · 제목 · 학회 · 연도 · DOI 가 맞았다는 뜻이다.
본문에 옮긴 수치는 논문의 표나 초록에서 읽은 값이다. 쪽 번호는 IEEE (DOI 기준) 와 CVF 공개판이 다른 경우가 많아 적지 않았다 — bib 를 만들 때 DOI 로 받을 것.

| # | 문헌 | 확인 |
|---|---|---|
| 1 | P. Isola, J.-Y. Zhu, T. Zhou, A. A. Efros. Image-to-Image Translation with Conditional Adversarial Networks. CVPR 2017. DOI 10.1109/CVPR.2017.632. arXiv:1611.07004 | 확인 |
| 2 | T. Park, M.-Y. Liu, T.-C. Wang, J.-Y. Zhu. Semantic Image Synthesis with Spatially-Adaptive Normalization. CVPR 2019. DOI 10.1109/CVPR.2019.00244. arXiv:1903.07291 | 확인 |
| 3 | L. Zhang, A. Rao, M. Agrawala. Adding Conditional Control to Text-to-Image Diffusion Models. ICCV 2023. DOI 10.1109/ICCV51070.2023.00355. arXiv:2302.05543 | 확인 |
| 4 | R. Rombach, A. Blattmann, D. Lorenz, P. Esser, B. Ommer. High-Resolution Image Synthesis with Latent Diffusion Models. CVPR 2022. DOI 10.1109/CVPR52688.2022.01042. arXiv:2112.10752 | 확인. **이 논문은 "Stable Diffusion" 이라는 이름을 쓰지 않는다.** 축소 배율 1–32 를 견주고 자기 text-to-image 모델에 8 을 썼다. 공개된 Stable Diffusion v1 이 배율 8 이라는 것은 CompVis/stable-diffusion README 나 [3] ("512×512 → 64×64") 을 인용할 것. SD 2.x · SDXL 은 1차 자료로 확인하지 못했다 |
| 5 | L. Yang, X. Xu, B. Kang, Y. Shi, H. Zhao. FreeMask: Synthetic Images with Dense Annotations Make Stronger Segmentation Models. NeurIPS 2023. arXiv:2310.15160 | 확인 |
| 6 | Y. Zhang, H. Ling, J. Gao 외. DatasetGAN: Efficient Labeled Data Factory with Minimal Human Effort. CVPR 2021. DOI 10.1109/CVPR46437.2021.01001. arXiv:2104.06490 | 확인. 주석 16–40장 |
| 7 | W. Wu, Y. Zhao, M. Z. Shou, H. Zhou, C. Shen. DiffuMask: Synthesizing Images with Pixel-level Annotations for Semantic Segmentation Using Diffusion Models. ICCV 2023. DOI 10.1109/ICCV51070.2023.00117. arXiv:2303.11681 | 확인. 합성 6만 장만으로는 VOC mIoU 70.6 (실제 1.06만 장 84.3), 합성 + 실제 5천 장은 84.9 (실제 5천 장 83.4) |
| 8a | J.-Y. Zhu, T. Park, P. Isola, A. A. Efros. Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks. ICCV 2017. DOI 10.1109/ICCV.2017.244. arXiv:1703.10593 | 확인 |
| 8b | J. Hoffman, E. Tzeng, T. Park 외. CyCADA: Cycle-Consistent Adversarial Domain Adaptation. ICML 2018, PMLR 80:1989–1998. arXiv:1711.03213 | 확인 |
| 9 | G. Ghiasi, Y. Cui, A. Srinivas 외. Simple Copy-Paste is a Strong Data Augmentation Method for Instance Segmentation. CVPR 2021. DOI 10.1109/CVPR46437.2021.00294. arXiv:2012.07177 | 확인 |
| 10 | D. Dwibedi, I. Misra, M. Hebert. Cut, Paste and Learn: Surprisingly Easy Synthesis for Instance Detection. ICCV 2017. DOI 10.1109/ICCV.2017.146. arXiv:1708.01642 | 확인 (초록) |
| 11 | W. Tranheden, V. Olsson, J. Pinto, L. Svensson. DACS: Domain Adaptation via Cross-domain Mixed Sampling. WACV 2021. DOI 10.1109/WACV48630.2021.00142. arXiv:2007.08702 | 확인. GTA5 → Cityscapes mIoU 32.85 → 52.14. 주석 없는 목표 도메인 사진을 쓴다 |
| 12 | C.-L. Li, K. Sohn, J. Yoon, T. Pfister. CutPaste: Self-Supervised Learning for Anomaly Detection and Localization. CVPR 2021. DOI 10.1109/CVPR46437.2021.00954. arXiv:2104.04015 | 확인 (초록) |
| 13 | V. Zavrtanik, M. Kristan, D. Skočaj. DRAEM – A Discriminatively Trained Reconstruction Embedding for Surface Anomaly Detection. ICCV 2021. DOI 10.1109/ICCV48922.2021.00822. arXiv:2108.07610 | 확인 |
| 14 | H. M. Schlüter, J. Tan, B. Hou, B. Kainz. Natural Synthetic Anomalies for Self-Supervised Anomaly Detection and Localization. ECCV 2022. DOI 10.1007/978-3-031-19821-2_27. arXiv:2109.15222 | 확인 (LNCS 권 번호는 미확인) |
| 15 | Y. Duan, Y. Hong, L. Niu, L. Zhang. Few-Shot Defect Image Generation via Defect-Aware Feature Manipulation. AAAI 2023, 37(1):571–578. DOI 10.1609/aaai.v37i1.25132. arXiv:2303.02389 | 확인. **하류 평가는 결함 분류뿐이다** (분할이 아니다) |
| 16 | T. Hu, J. Zhang, R. Yi 외. AnomalyDiffusion: Few-Shot Anomaly Image Generation with Diffusion Model. AAAI 2024, 38(8):8526–8534. DOI 10.1609/aaai.v38i8.28696. arXiv:2312.05767 | 확인. 사전학습된 잠재 확산 모델 위에서, MVTec 결함 사진의 1/3 로 학습 |
| 17 | J. Song, D. Park, K. Baek, S. Lee, J. Choi, E. Kim, S. Yoon. DefectFill: Realistic Defect Generation with Inpainting Diffusion Model for Visual Inspection. CVPR 2025. DOI 10.1109/CVPR52734.2025.01744. arXiv:2503.13985 | 확인. Stable Diffusion 2 인페인팅 + LoRA, 범주당 결함 사진 약 20장. 저자들이 적은 한계: 전체 구조에 걸친 결함에 약하다 |
| 18 | B. Trabucco, K. Doherty, M. Gurinas, R. Salakhutdinov. Effective Data Augmentation With Diffusion Models. ICLR 2024. arXiv:2302.07944 | 확인 |
| 19a | E. J. Hu, Y. Shen, P. Wallis 외. LoRA: Low-Rank Adaptation of Large Language Models. ICLR 2022. arXiv:2106.09685 | 확인 |
| 19b | N. Ruiz, Y. Li, V. Jampani, Y. Pritch, M. Rubinstein, K. Aberman. DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation. CVPR 2023. DOI 10.1109/CVPR52729.2023.02155. arXiv:2208.12242 | 확인 |
| 20 | S. Ravuri, O. Vinyals. Classification Accuracy Score for Conditional Generative Models. NeurIPS 2019. arXiv:1905.10887 | 확인. **초록의 "27.9% · 41.6% 감소" 는 그대로 인용하지 말 것** — 표 1 의 값과 견주면 순서가 바뀐 상대 감소율로 보인다. 표의 값 (73.09 → 42.65) 을 쓴다 |
| 21 | R. He, S. Sun, X. Yu 외. Is Synthetic Data from Generative Models Ready for Image Recognition? ICLR 2023. arXiv:2210.07574 | 확인 |
| 22 | S. Azizi, S. Kornblith, C. Saharia, M. Norouzi, D. J. Fleet. Synthetic Data from Diffusion Models Improves ImageNet Classification. Transactions on Machine Learning Research, 2023. arXiv:2304.08466 | 확인. 합성이 많아지면 이득이 뒤집힌다 |
| 23 | M. F. Burg, F. Wenzel, D. Zietlow 외. Image Retrieval Outperforms Diffusion Models on Data Augmentation. Transactions on Machine Learning Research, 2023. arXiv:2304.10253 | 확인. **"이득이 바깥 데이터에서 온다" 는 인과는 이 논문이 보인 것이 아니다.** 전체 ImageNet 과 Caltech256 에서는 차이가 0.2–0.3 (표준오차 수준) 이다 |
| 24 | L. Fan, K. Chen, D. Krishnan, D. Katabi, P. Isola, Y. Tian. Scaling Laws of Synthetic Images for Model Training … for Now. CVPR 2024. DOI 10.1109/CVPR52733.2024.00705. arXiv:2312.04567 | 확인 (초록) |
| 25a | I. Shumailov, Z. Shumaylov, Y. Zhao, N. Papernot, R. Anderson, Y. Gal. AI Models Collapse When Trained on Recursively Generated Data. Nature 631:755–759, 2024. DOI 10.1038/s41586-024-07566-y | 확인. 실험은 언어 모델 · VAE · 가우시안 혼합 — 이미지 생성 모델은 [25b] 를 인용. 저자 정정 (Nature 640, E6, 2025) 이 있다 |
| 25b | S. Alemohammad, J. Casco-Rodriguez, L. Luzi 외. Self-Consuming Generative Models Go MAD. ICLR 2024. arXiv:2307.01850 | 확인 (초록) |
| 26 | L. Scime, D. Siddel, S. Baird, V. Paquit. Layer-wise anomaly detection and classification for powder bed additive manufacturing processes: A machine-agnostic algorithm for real-time pixel-wise semantic segmentation. Additive Manufacturing 36 (2020) 101453. DOI 10.1016/j.addma.2020.101453 | 서지 확인. 증강 내용은 `paper/DSCNN/` 의 PDF (3.7절) 에서 읽었다 — 잡음과 밝기 이동뿐이고 **뒤집기는 쓰지 않았다** |
| 27a | T. Cour, B. Sapp, B. Taskar. Learning from Partial Labels. Journal of Machine Learning Research 12:1501–1536, 2011 | 확인 |
| 27b | T. Ishida, G. Niu, W. Hu, M. Sugiyama. Learning from Complementary Labels. Advances in Neural Information Processing Systems 30 (NIPS 2017). arXiv:1705.07541 | 확인 |
| B1 | D. Cannizzaro, A. G. Varrella, S. Paradiso 외. In-Situ Defect Detection of Metal Additive Manufacturing: An Integrated Framework. IEEE Transactions on Emerging Topics in Computing 10(1):74–86, 2022. DOI 10.1109/TETC.2021.3108844 | 초록만 확인 |
| B2 | D. N. Phan, S. Jha, J. P. Mavo 외. Scalable artificial intelligence framework for defect detection in metal additive manufacturing. The International Journal of Advanced Manufacturing Technology 142:1619–1633, 2026 (온라인 2025-12). DOI 10.1007/s00170-025-17114-1. arXiv:2411.00960 | 서지 확인. 수치는 arXiv 판에서 읽었다 |
| B3 | F. Ogoke, S. K. Suresh, J. Adamczyk 외. Deep learning based optical image super-resolution via generative diffusion models for layerwise in-situ LPBF monitoring. Additive Manufacturing 107 (2025) 104790. DOI 10.1016/j.addma.2025.104790. arXiv:2409.13171 | 확인 (초록) |
| B4 | X. Yu, G. Li, W. Lou 외. Diffusion-based Data Augmentation for Nuclei Image Segmentation. MICCAI 2023. DOI 10.1007/978-3-031-43993-3_57. arXiv:2310.14197 | 확인 (LNCS 권 번호는 미확인) |
| B5 | L. Zhang, B. Jindal, A. Alaa 외. Generative AI enables medical image segmentation in ultra low-data regimes. Nature Communications 16, 6486 (2025). DOI 10.1038/s41467-025-61754-6. arXiv:2408.17421 | 확인. 수치는 arXiv 판 |
| B6 | M. H. Vu, L. Tronchin, T. Nyholm, T. Löfstedt. Using Synthetic Images to Augment Small Medical Image Datasets. arXiv:2503.00962, 2025 | arXiv 만 확인. 학술지 · 학회 기록은 찾지 못했다 |
