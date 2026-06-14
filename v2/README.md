# DefSeg-AM v2

> v1 의 모델 구조 그대로 + **8-class + DSCNN-style augmentation + 8-fold cross-validation**.
> 자세한 설계는 [../PLAN_v2.md](../PLAN_v2.md), 원논문 분석은
> [../paper/DSCNN/DSCNN_Summary.md](../paper/DSCNN/DSCNN_Summary.md).

## 폴더 구조

```
v2/
├── README.md                 # 본 문서
├── config_v2.py              # 8-class 상수, BJ 2 source 매핑, 8-fold CV
├── data/
│   ├── augmentation.py       # D4 + cyclic shift + DSCNN Gaussian noise + intensity shift
│   ├── data_ornl_v2.py       # ORNL Stage 1 dataset (8-class)
│   ├── data_dscnn_v2.py      # DSCNN_Dataset Stage 2 dataset (8 source, fold split)
│   └── build_cache_v2.py     # v1 label cache → v2 8-class 재매핑 (image 는 v1 재사용)
├── models/
│   └── losses_v2.py          # hard_bootstrap_loss (Eq.4), median_inv_class_weight (Eq.6)
├── training/
│   ├── train_stage1.py       # KD pretrain + EMA + (옵션) hard-bootstrap
│   └── train_stage2.py       # 8-fold CV finetune + EMA
├── inference/
│   └── infer.py              # single / ensemble / TTA (D4 group 8)
├── scripts/
│   ├── run_build_cache_v2.sh
│   ├── run_stage1_v2.sh
│   ├── run_stage2_v2_fold.sh
│   ├── run_stage2_v2_all.sh
│   └── run_all_v2.sh
└── docker/
    ├── Dockerfile
    ├── docker-compose.yml
    └── README.md             # docker 사용법
```

## v1 대비 변경 요약

| 항목 | v1 | v2 |
|---|---|---|
| Class | 12 (ORNL Peregrine) | **8** (Recoater 통합 + 3개 제거) |
| Stage 2 데이터 | LPBF 5 source (1 val 고정) | **LPBF 6 + BJ 2 = 8 source, 8-fold CV** |
| Augmentation | brightness jitter ±15% | + **D4 rotation/flip** + **Cyclic shift** + **DSCNN Gaussian noise** + **intensity shift** |
| EMA weight | 미적용 | **decay 0.9999** (Stage 1, 2 모두) |
| Hard-bootstrap loss | 미적용 | Stage 1 옵션 (λ=0.8) |
| Class weight | sqrt-inv (default) | sqrt-inv (default) / median-inv 옵션 |
| Inference | 단일 ckpt | + **ensemble (8 fold) + TTA (D4)** |

## v2 호출 경로 (v1 과 namespace 분리)

```bash
# v1 (변경 없음)
python -m DefSeg_AM.v2.training.train_stage1
python -m DefSeg_AM.v2.training.train_stage2
python -m DefSeg_AM.v2.inference.infer

# v2
python -m DefSeg_AM.v2.data.build_cache_v2     # label cache 재빌드
python -m DefSeg_AM.v2.training.train_stage1   # KD pretrain (8-class)
python -m DefSeg_AM.v2.training.train_stage2 --fold 0  # CV finetune
python -m DefSeg_AM.v2.inference.infer --run-name <run> --stage 2 --ensemble --tta
```

## 호스트 직접 실행 (shell scripts)

```bash
# 1) 캐시 (label 8-class 재매핑, ~5분)
bash DefSeg_AM/v2/scripts/run_build_cache_v2.sh

# 2) Stage 1 (~12h)
bash DefSeg_AM/v2/scripts/run_stage1_v2.sh
# 또는 hard-bootstrap 적용:
V2_STAGE1_EXTRA="--use-hard-bootstrap" bash DefSeg_AM/v2/scripts/run_stage1_v2.sh

# 3) Stage 2 — 단일 fold
V2_FOLD=0 bash DefSeg_AM/v2/scripts/run_stage2_v2_fold.sh

# 3') Stage 2 — 8 fold 순회 (~3h)
bash DefSeg_AM/v2/scripts/run_stage2_v2_all.sh

# 4) 전체 파이프라인
bash DefSeg_AM/v2/scripts/run_all_v2.sh
```

## Docker 실행

[docker/README.md](docker/README.md) 참조.

```bash
cd DefSeg_AM/v2/docker
DEFSEG_PHASE=build_cache docker compose up -d --build
DEFSEG_PHASE=stage1 docker compose up -d --build
DEFSEG_PHASE=stage2 DEFSEG_FOLD=all docker compose up -d --build
DEFSEG_PHASE=all docker compose up -d --build   # 전부
```

## 산출물

```
DefSeg_AM/
├── cache/resized_sz1036_v2/                  # v2 label cache (label_v2.npy)
└── checkpoints/vits14_dpt_dual_sz1036_8cls_v2/
    ├── stage1_best.pt                        # Stage 1 best (EMA weight)
    ├── stage2_best_fold0_v2021_LPBF.pt
    ├── stage2_best_fold1_v2022_17-4PH.pt
    ├── stage2_best_fold2_v2022_GammaPrint.pt
    ├── stage2_best_fold3_v2022_Inc718_1.pt
    ├── stage2_best_fold4_v2022_Inc718_2.pt
    ├── stage2_best_fold5_v2022_Maraging.pt
    ├── stage2_best_fold6_v2021_BJ.pt
    ├── stage2_best_fold7_v2022_BJ_H13.pt
    └── cv_summary.json                       # 8-fold 의 val_acc, mIoU, per-class IoU
```

## 8-class 정의 (config_v2.ORNL_CLASS_NAMES_V2)

| ID | 이름 | v1 (12) 출처 |
|---|---|---|
| 0 | Powder | 0 그대로 |
| 1 | Printed | 1 그대로 |
| 2 | **Recoater Disturbance** | 2 (Hopping) + 3 (Streaking) **통합** |
| 3 | Swelling | 5 |
| 4 | Spatter | 8 |
| 5 | Super-Elevation | 7 |
| 6 | Over Melting | 10 |
| 7 | Debris | 6 |

제거: 4 Incomplete Spreading, 9 Misprint, 11 Under Melting → IGNORE (학습/평가 모두 제외)

## v1 과의 공존

- v1 코드는 그대로 동작 (변경 없음, [../config.py](../config.py) 등 v1 파일 미수정)
- v1 의 image cache (`cache/resized_sz1036/<build>/visible_*.npy`) 는 v2 가 그대로 재사용
- v2 의 label cache 만 별도 (`cache/resized_sz1036_v2/<build>/label_v2.npy`)
- v2 의 checkpoints / figures 는 별도 run_name (`*_8cls_v2`) 으로 v1 과 분리
