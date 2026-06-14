# DefSeg-AM

L-PBF(Laser Powder Bed Fusion) layer-wise powder bed 이미지에 대한
**결함 semantic segmentation** 모델 — **DINOv2 (frozen) backbone + DPT-style decoder**,
2-stage(KD pretrain → GT finetune) 학습.

코드는 **공유(common) · v1 · v2** 세 영역으로 분리되어 있다.

```
DefSeg_AM/
├── common/    # ★ v1·v2 공유 코드
│   ├── config.py              # 경로·ORNL 라벨공간·DSCNN 매핑·backbone·출력경로 (공유 substrate)
│   ├── data/{image_utils.py, samplers.py}   # 이미지/라벨 유틸 + DDP WeightedSampler
│   ├── models/{model.py, losses.py}         # DefSegModel(DINOv2+DPT) + Focal/CE class-weight
│   ├── training/dist_utils.py               # DDP init / rank / all-reduce helper
│   ├── utils/log.py
│   └── MODEL_ARCHITECTURE.md                 # 공유 모델 구조 문서
│
├── v1/        # 12-class (ORNL Peregrine 표준) — 원본 모델
│   ├── config.py  data/  training/  inference/  scripts/  docs/
│   └── README.md / PLAN.md / DEBUG_HISTORY.md
│
├── v2/        # 8-class (Recoater 통합 + 3종 제거) + 8-fold CV + DSCNN aug
│   ├── config_v2.py  data/  training/  inference/  scripts/  docker/
│   └── README.md / PLAN_v2.md
│
├── cache/  checkpoints/  figures/  logs/   # (gitignore) v1·v2 공유 출력 (run-name 으로 구분)
├── paper/         # DSCNN 원 논문 + 클래스 정의
├── requirements.txt
└── venv/          # (gitignore) 호스트·컨테이너 공용
```

## 의존 방향
- **v1 → common**, **v2 → common**. v1 과 v2 는 서로를 임포트하지 않는다.
- `v2/config_v2.py` 는 `common/config.py` 의 공유 상수(경로·`MATERIAL_TO_ORNL`·`DSCNN_TRAIN_SOURCES` 등)를
  재사용하면서 8-class 재매핑·CV·aug 설정을 덧붙인다.
- 공유 모델/손실/샘플러/로깅/이미지 유틸/DDP helper 는 전부 `common/` 에 있다.

## v1 vs v2

| 항목 | v1 | v2 |
|---|---|---|
| Class | 12 (ORNL Peregrine) | 8 (Recoater 통합 + 3종 제거) |
| Stage 2 데이터 | LPBF 6 source (val 1개 고정) | LPBF + BJ 8 source, 8-fold CV |
| Augmentation | brightness jitter | D4 + cyclic shift + DSCNN noise/intensity |
| 모델 구조 | 공유 `common/models/model.py` (동일) | 〃 |

## 실행
- v1: [v1/README.md](v1/README.md) — `python -m DefSeg_AM.v1.<module>`, `bash DefSeg_AM/v1/scripts/run_*.sh`
- v2: [v2/README.md](v2/README.md) — `python -m DefSeg_AM.v2.<module>`, `v2/docker/` compose

> 직접 모듈 호출 시 작업 디렉터리는 항상 저장소 루트(`3DP_VPPM/`) — `common/config.py` 의 상대 경로 기준.

## 설치
```bash
# 저장소 루트(3DP_VPPM) 기준
python -m venv DefSeg_AM/venv
./DefSeg_AM/venv/bin/pip install -r DefSeg_AM/requirements.txt
```
DINOv2 backbone 은 최초 실행 시 `torch.hub` 로 자동 다운로드된다.
