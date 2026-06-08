# DefSeg-AM

ORNL L-PBF(Laser Powder Bed Fusion) 의 layer-wise powder bed 이미지에 대한
**12-class 결함 semantic segmentation** 모델.

**DINOv2 (frozen) backbone + DPT-style multi-scale decoder** 를
**2-stage(KD pretrain → GT finetune)** 로 학습한다.

> 설계 근거 및 상세 의사결정은 [PLAN.md](PLAN.md) 를, 학습 중 만난 이슈/해결 기록은
> [DEBUG_HISTORY.md](DEBUG_HISTORY.md) 를, 클래스 정의는 [docs/DSCNN_Classes.md](docs/DSCNN_Classes.md) 를 참고.

---

## 1. 핵심 아이디어

| 구분 | 내용 |
|---|---|
| **입력** | dual visible — `visible/0`(after melt) + `visible/1`(after spread), 각 1036×1036 |
| **Backbone** | DINOv2 `ViT-S/14` (frozen, `.eval()`) — block `[2,5,8,11]` 의 intermediate token 4-stage 추출 |
| **Decoder** | DPT-style Reassemble + top-down Fusion blocks + classifier head (외부 의존 없는 자체 구현) |
| **Fusion** | 각 stage 에서 `(f0, f1, f1−f0)` concat → 1×1 conv 로 256ch 압축 |
| **출력** | ORNL 12-class logits, `(B, 12, 1036, 1036)` |

### 2-Stage 학습

| Stage | 데이터 | 라벨 | 목적 |
|---|---|---|---|
| **S1 — KD pretrain** | ORNL Co-Registered HDF5 (5 빌드) | `slices/segmentation_results` (DSCNN 예측 mask) → argmax hard label | 도메인 대규모 사전학습 |
| **S2 — GT finetune** | DSCNN_Dataset annotations (6 source) | 사람 GT(native ID) → ORNL 12-class 재매핑 | 소규모 진짜 GT 로 마무리 |

> S1 → S2 전환 시 backbone 은 frozen 유지, decoder/head weight 만 이어받고 optimizer 는 reset.

---

## 2. 12 클래스 (ORNL 공간)

```
0 Powder              1 Printed             2 Recoater Hopping    3 Recoater Streaking
4 Incomplete Spreading 5 Swelling           6 Debris              7 Super-Elevation
8 Spatter             9 Misprint            10 Over Melting        11 Under Melting
```

- `0 Powder`, `1 Printed` 는 정상, `2~11` 은 결함 10종.
- 재매핑 불가능한 native class 및 미할당 pixel = `-1` (IGNORE, loss 계산에서 제외).
- 재료별 native → ORNL 매핑은 [config.py](config.py) 의 `MATERIAL_TO_ORNL` 참조.

---

## 3. 저장소 구조

```
DefSeg_AM/
├── config.py                 # 경로·하이퍼파라미터·클래스 정의·MATERIAL_TO_ORNL
│
├── data/                     # 데이터 파이프라인
│   ├── data_ornl.py          # Stage 1 dataset (DefSegORNLDataset, ORNL HDF5 layer-wise)
│   ├── data_dscnn.py         # Stage 2 dataset (DefSegDSCNNDataset, native → ORNL remap)
│   ├── samplers.py           # DDP-호환 WeightedRandomSampler (defect-ratio oversampling)
│   └── build_cache_stage1.py # ORNL raw HDF5 → resize+uint8+argmax 캐시 사전 생성
│
├── models/
│   ├── model.py              # DefSegModel (DINOv2 + DPT decoder + head)
│   └── losses.py             # Focal loss + sqrt-inverse-frequency class weight
│
├── training/
│   ├── train_stage1.py       # Stage 1 KD pretrain 진입점
│   └── train_stage2.py       # Stage 2 GT finetune 진입점 (--init-from stage1_best.pt)
│
├── inference/
│   └── infer.py              # 학습 모델로 ORNL 4-panel 비교 PNG 생성
│
├── utils/
│   └── log.py                # 로깅 유틸
│
├── scripts/                  # nohup 백그라운드 실행 래퍼
│   ├── run_build_cache.sh
│   ├── run_stage1.sh
│   └── run_stage2.sh
│
├── docs/                     # 참고 문서 + DSCNN 원 논문 PDF
│   └── DSCNN_Classes.md
│
├── PLAN.md / DEBUG_HISTORY.md
├── requirements.txt
│
├── cache/                    # (gitignore) Stage 1 resize/argmax 캐시
├── checkpoints/              # (gitignore) <run_name>/stage{1,2}_best.pt
├── figures/                  # (gitignore) <run_name>/stage{1,2}/inference/*.png
└── logs/                     # (gitignore) 학습 로그
```

---

## 4. 데이터 위치

`config.py` 의 `PROJECT_ROOT` 는 상위 디렉터리(`3DP_VPPM/`)를 가리키며, 데이터는 그 아래에 둔다.

- **Stage 1** — `ORNL_Data/Co-Registered In-Situ and Ex-Situ Dataset/[baseline] (Peregrine v2023-11)/*.hdf5` (5 빌드)
  - Train: Build 2·3·4·5 / Val: **Build 1** (정성 비교용)
- **Stage 2** — `ORNL_Data/DSCNN_Dataset/...` (LPBF 6 source, EBPBF/BJ 제외)
  - Val source: `v2022_Maraging` / 나머지 5개 train

---

## 5. 설치

```bash
# 저장소 루트(3DP_VPPM) 기준, DefSeg_AM/venv 사용
python -m venv DefSeg_AM/venv
./DefSeg_AM/venv/bin/pip install -r DefSeg_AM/requirements.txt
```

DINOv2 backbone 은 최초 실행 시 `torch.hub` 로 자동 다운로드된다.

---

## 6. 실행

모든 스크립트는 저장소 루트(`3DP_VPPM/`)에서 호출하며, 내부적으로 `python -m DefSeg_AM.<module>` 로 실행한다.
GPU 선택은 스크립트 안의 `CUDA_VISIBLE_DEVICES` 로 조정한다.

### (0) 캐시 사전 생성 — 학습 전 1회 필수

```bash
bash DefSeg_AM/scripts/run_build_cache.sh
```

ORNL raw HDF5 를 resize + uint8 + argmax 하여 `cache/resized_sz1036/` 에 저장 (예상 30~60분, 디스크 ~45GB).

### (1) Stage 1 — KD pretrain

```bash
bash DefSeg_AM/scripts/run_stage1.sh
# → checkpoints/<run_name>/stage1_best.pt
```

### (2) Stage 2 — GT finetune

```bash
bash DefSeg_AM/scripts/run_stage2.sh   # 동일 run-name 의 stage1_best.pt 자동 load
# → checkpoints/<run_name>/stage2_best.pt
```

### (3) 추론 / 정성 비교

```bash
./DefSeg_AM/venv/bin/python -m DefSeg_AM.inference.infer \
    --run-name vits14_dpt_dual_sz1036_1gpu_nanfix --stage 2
# → figures/<run_name>/stage2/inference/<build>/layerXXXX.png
#   (visible/0, visible/1, DSCNN pred, our prediction 4-panel)
```

> 직접 모듈을 호출할 때도 작업 디렉터리는 항상 저장소 루트(`3DP_VPPM/`)여야 한다
> (`config.py` 의 상대 경로 기준).

---

## 7. 주요 하이퍼파라미터 ([config.py](config.py))

| | Stage 1 | Stage 2 |
|---|---|---|
| Loss | Focal (γ=2) + class weight | CrossEntropy + class weight |
| LR | 1e-4 | 1e-4 |
| Epochs | 30 | 50 |
| Batch | 2 | 2 |
| Warmup steps | 200 | 50 |
| Sampling | defect-ratio oversampling (`^0.5`) | uniform |
| 공통 | grad-clip 1.0 · class-weight clip 10 · AMP off · `ignore_index=-1` | |

> 위 값들은 초기 run 의 NaN 발생을 잡기 위한 안정화 설정이다 (lr↓, warmup, grad-clip,
> class-weight clip↓, AMP off). 배경은 [DEBUG_HISTORY.md](DEBUG_HISTORY.md) 참고.

---

## 8. 주요 CLI 인자

- **train_stage1**: `--epochs --batch-size --img-size --lr --backbone --num-workers --gamma --oversample-power --val-log-every --run-name --quick`
- **train_stage2**: `--epochs --batch-size --img-size --lr --backbone --num-workers --val-log-every --run-name --init-from --no-init --quick`
- **infer**: `--run-name(필수) --stage{1,2} --checkpoint --build --layers --n-layers --img-size --out-dir`
- **build_cache_stage1**: `--img-size --chunk-size --split{both,train,val} --rebuild`

`--quick` 는 소규모 스모크(짧은 epoch) 실행용이다.
