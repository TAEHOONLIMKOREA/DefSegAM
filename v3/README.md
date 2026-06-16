# DefSeg-AM v3

> v2 의 모델 구조 그대로 + **10-class 복원** + **flip(LR/UD)+rot180+cyclic augmentation**
> + **DSCNN_Dataset 전수 학습 (CV 없음)** + **확장 휴리스틱 후처리** + **4-GPU DDP**.
>
> 자세한 설계는 [PLAN_v3.md](PLAN_v3.md), v2 와의 비교는 PLAN §1 참조.

## 폴더 구조

```
v3/
├── README.md                 # 본 문서
├── PLAN_v3.md                # 설계서
├── config_v3.py              # 10-class 상수, augmentation, PP_*, lr 2e-4
├── data/
│   ├── augmentation.py       # flip LR/UD + 180° rot + cyclic shift + DSCNN noise/intensity + brightness
│   ├── data_ornl_v3.py       # ORNL Stage 1 dataset (10-class)
│   ├── data_dscnn_v3.py      # DSCNN_Dataset Stage 2 dataset (8 source 전수, replicate K)
│   └── build_cache_v3.py     # v1 label cache → v3 10-class 재매핑 (image 는 v1 재사용)
├── models/
│   └── losses_v3.py          # hard_bootstrap_loss + median_inv_class_weight (v2 와 동일)
├── training/
│   ├── train_stage1.py       # KD pretrain + EMA + (옵션) hard-bootstrap — 4-GPU DDP
│   └── train_stage2.py       # 8 source 전수 train + ORNL Build 1 평가 — 4-GPU DDP
├── inference/
│   ├── infer.py              # single + TTA(8-way) + --postprocess
│   ├── postprocess.py        # SE/Swelling 이 부품에서 멀면 Debris 로 재분류
│   └── confusion.py          # stage 1/2 confusion matrix (CV 없음)
├── scripts/
│   ├── run_build_cache_v3.sh
│   ├── run_stage1_v3.sh
│   ├── run_stage2_v3.sh
│   └── run_all_v3.sh
└── docker/
    ├── Dockerfile
    ├── docker-compose.yml
    └── README.md             # docker 사용법
```

## v2 대비 변경 요약

| 항목 | v2 | v3 |
|---|---|---|
| Class | 8 (Recoater 통합) | **10** (Recoater 분리 복원 + Incomplete Spreading 복원) |
| Augmentation | D4 group (4 rot × 2 flip) + cyclic | **flip LR/UD + 180° rot + cyclic** (90/270 제외) |
| Stage 2 CV | leave-one-source-out 8-fold | **CV 없음** — DSCNN_Dataset 8 source 전부 train |
| Stage 2 평가 | fold 별 held-out source | **ORNL Build 1** 의 mIoU |
| Stage 2 ckpt | 8 fold ckpt + cv_summary | **단일 `stage2_best.pt`** |
| Stage 2 augmentation 강화 | 기본 1회 | **replicate factor K=4** (epoch 당 step 4×) |
| 휴리스틱 후처리 | SE/Swelling far → Debris (target=7) | 동일 (target=6, v3 인덱스) |
| GPU 활용 | 1 GPU | **4-GPU DDP** (torchrun) |
| Learning rate | 1e-4 | **2e-4** (sqrt scaling: 1e-4 × √4) |

## v3 호출 경로 (v1/v2 와 namespace 분리)

```bash
python -m DefSeg_AM.v3.data.build_cache_v3
torchrun --nproc_per_node=4 -m DefSeg_AM.v3.training.train_stage1
torchrun --nproc_per_node=4 -m DefSeg_AM.v3.training.train_stage2
python -m DefSeg_AM.v3.inference.infer --run-name <run> --stage 2 --tta --postprocess
python -m DefSeg_AM.v3.inference.confusion --run-name <run> --stage 2
```

(또는 [`scripts/run_*.sh`](scripts/) / [`docker/docker-compose.yml`](docker/docker-compose.yml))

## 산출물

- **v3 label cache**: `cache/resized_sz1036_v3/<build>/label_v3.npy` (image 는 v1 의 `resized_sz1036/<build>/visible_{0,1}.npy` 재사용)
- **ckpt**: `checkpoints/vits14_dpt_dual_sz1036_10cls_v3/{stage1_best, stage2_best}.pt`
- **figure**: `figures/vits14_dpt_dual_sz1036_10cls_v3/v3/stage{1,2}{,_tta,_pp}/inference/...`

## 다음 단계 (PLAN_v3 §8 평가)

학습 완료 후:
1. `confusion` — Stage 1 vs Stage 2 의 ORNL Build 1 per-class IoU 비교
2. `infer --postprocess` — 휴리스틱 전·후 figure 비교
3. v1/v2/v3 통합 결과 비교는 별도 `RESULTS_v1_vs_v2_vs_v3.md`
