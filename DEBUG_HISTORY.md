# DefSeg-AM 학습 디버깅 히스토리

> 2026-05-26 ~ 2026-05-27 동안 1차 Stage 1 학습 시도 중 발생한 문제와 해결 과정 기록.
> 향후 같은 종류의 이슈가 재발하거나 환경/하드웨어 변경 시 reference 용도.

---

## 환경

- 호스트: `oem-MS73-HB1-000` (Dual-socket Intel Xeon, 256 GB RAM, 32-core)
- 초기 GPU: 4× NVIDIA RTX 5090 (Blackwell, sm_120, 32GB each) — 일부 단계에서 hardware 이슈로 변동
- OS: Ubuntu 22.04, kernel 6.8
- Driver: NVIDIA 575.51.02, CUDA runtime 12.9
- 학습 입력: 1036×1036 dual visible-light camera images (ORNL L-PBF), DINOv2 ViT-S/14 backbone + DPT decoder, 12-class segmentation

---

## 이슈 1 — Stdout buffering: 학습 중인데 로그가 안 보임

### 증상
첫 학습 시도 시 10 시간이 지나도 stage1.log 에 4개 줄 (xFormers warning) 만 보이고 진행상황 print 가 전무. 사용자가 "stuck 인가" 의심.

### 원인
- `nohup python -m ... > stage1.log 2>&1` 형식으로 redirect 시 Python stdout 이 **fully buffered** (4~8 KB 채워야 flush)
- stderr 만 line-buffered 라 warning 은 즉시 보이고 진짜 print 는 buffer 에 갇힘

### 해결
[run_stage1.sh](run_stage1.sh) / [run_stage2.sh](run_stage2.sh) 에 두 곳 적용:
```bash
PYTHONUNBUFFERED=1 ./venv/bin/python -u -m ...
                                    ^^
```
- `PYTHONUNBUFFERED=1` 환경변수 + `-u` 옵션 둘 다 강제 unbuffered.

### 교훈
**Long-running 백그라운드 학습 스크립트는 무조건 `PYTHONUNBUFFERED=1 python -u` 조합 사용**. 그렇지 않으면 진행 상황 보이지 않아 불필요한 kill 위험.

---

## 이슈 2 — Layer-index 빌드가 9 시간 (data prep 단계 너무 느림)

### 증상
첫 학습 시도 시 `build_layer_index` 함수가 5 builds × 약 3000 layer × 12 mask 의 HDF5 read 를 1개씩 순차로 호출. 총 ~169M syscall, 9 시간 소요.

### 원인
- HDF5 chunked storage 에서 `seg[str(c)][li]` 1개씩 access 하면 chunk 마다 decompression 비용 발생
- Python+h5py overhead 도 layer 당 10+ ms

### 해결
[build_cache_stage1.py](build_cache_stage1.py) 신규 작성: **사전 resize + uint8 + argmax cache** 빌드.
- 입력 ORNL float32 1842² → percentile uint8 → PIL BILINEAR resize 1036² → memmap `.npy`
- 12 mask boolean → argmax int8 → NEAREST resize → memmap `.npy`
- **HDF5 chunked read 50 layer 단위** 로 amortize → 1 layer 당 460 ms (vs 이전 2000+ ms)
- 전체 캐시 빌드: **~110 분 (1회만)**, ~48 GB 디스크
- 빌드 후 학습 시 `DefSegORNLCachedDataset` 가 memmap slice 1줄로 batch 데이터 획득 → batch 당 PIL/percentile/argmax 모두 사라짐

### 학습 측 변화
- 기존 `DefSegORNLDataset` (HDF5 직접 read) 는 디버깅용으로 보존
- 학습은 [train_stage1.py](train_stage1.py) 에서 무조건 `DefSegORNLCachedDataset` 사용
- 첫 학습 진입 전에 반드시 `bash DefSeg_AM/run_build_cache.sh` 1회 실행

### 교훈
**Heavy preprocessing (resize, percentile normalize 등) 은 학습 epoch 마다가 아니라 1회 캐시 빌드**. memmap `.npy` 면 worker 마다 다른 process 에서 동시 안전 read.

---

## 이슈 3 — GPU 0% util (DataLoader 가 GPU 못 따라옴)

### 증상
학습 시작 후 GPU memory 5.7 GB 점유했는데 utilization 0%. CPU worker 4개는 99% busy. Throughput 3초/batch.

### 원인
1 GPU + 단일 프로세스 학습 + num_workers=4 + 매 batch 마다 PIL+percentile+argmax 처리 → CPU bound. GPU 가 데이터 기다리는 시간 90%.

### 해결 (병행 3가지)
1. **사전 resize 캐시** (이슈 2 해결책) — 가장 큰 효과 (preprocessing 사라짐)
2. **DDP 멀티 GPU** — `torchrun --nproc-per-node=N` + `DistributedDataParallel` 으로 4 GPU 활용
3. **persistent_workers=True + num_workers=8/rank** — worker spawn overhead 제거 + I/O 병렬화

신규 파일:
- [samplers.py](samplers.py) — `DistributedWeightedSampler` (PyTorch 의 `DistributedSampler` 는 weight 미지원)
- [log.py](log.py) — timestamp + rank-prefix logger (DDP 다중 rank 출력 식별)

### 교훈
- Multi-GPU 학습은 **DDP > DataParallel** (DataParallel deprecated, GIL 경쟁).
- DataLoader 병렬화 와 GPU 병렬화는 별개 — 둘 다 챙겨야 진짜 throughput 나옴.

---

## 이슈 4 — `torchrun` shebang 깨짐 (venv 이전 흔적)

### 증상
```
nohup: failed to run command './venv/bin/torchrun': No such file or directory
```
파일은 존재. 그러나 첫 줄 shebang:
```
#!/home/taehoon/3DP_TensileProp_Prediction/venv/bin/python3
```
이 경로는 더 이상 존재 안 함. 그 venv 가 `3DP_VPPM/venv/` 로 이동/복사되면서 console script 들이 깨진 상태.

### 원인
Python venv 의 `bin/` 안 console script (pip, torchrun, ipython 등) 들은 venv 만들 시점의 절대경로가 shebang 에 박힘. venv 폴더가 이동/복사되면 shebang 이 stale 됨.

### 해결
[run_stage1.sh](run_stage1.sh) / [run_stage2.sh](run_stage2.sh) 에서 console script 직접 호출 회피, **`python -m <module>` 형태로 우회**:
```bash
./venv/bin/python -u -m torch.distributed.run --standalone --nproc-per-node=4 ...
                       ^^^^^^^^^^^^^^^^^^^^^^
                       (= torchrun 의 본체 모듈)
```
- venv 의 python 만 직접 호출하면 shebang 무관

### 교훈
**venv 가 다른 경로로 옮겨질 가능성 있는 환경에서는 console script 보다 `python -m` 사용 권장**. 또는 `pip install --force-reinstall` 으로 shebang 재생성.

---

## 이슈 5 — DDP unused parameters 에러

### 증상
```
RuntimeError: Expected to have finished reduction in the prior iteration ...
Parameter indices which did not receive grad for rank 1: 44 45 46 47
```
rank 1,2,3 모두 동일 메시지. rank 0 은 cuDNN error (부수 효과).

### 원인
[model.py](model.py) 의 `FeatureFusionBlock` 안에는 `res_skip` 모듈이 있는데, **가장 깊은 `fusion_blocks[3]` 은 forward 에서 skip 인자 없이 호출됨** → `res_skip` 사용 안 됨 → 그 안의 4 conv param (인덱스 44~47) 이 backward 에서 grad 못 받음.

DDP 는 매 backward 마다 모든 trainable param 에 grad 가 동기화되길 기대 → 이런 unused param 이 있으면 에러.

### 해결
DDP wrap 시 옵션 변경:
```python
model = DDP(model, device_ids=[local_rank], find_unused_parameters=True)
                                            ^^^^^^^^^^^^^^^^^^^^^^^^^
```
- DDP 가 forward 끝나면 graph 를 한 번 더 traverse 해서 "어떤 param 이 graph 에 포함됐는지" 확인
- 미포함 param 은 reduction sync 대상에서 제외
- Overhead 미미 (4 / 13M params)

### 더 깔끔한 대안 (보류)
`fusion_blocks[3]` 전용으로 `res_skip` 없는 별도 클래스를 만드는 것. 시간 부담 없을 때 refactor 가능.

### 교훈
**DDP 환경에서 모델의 어떤 모듈이 forward path 에서 조건부로 빠질 수 있다면 `find_unused_parameters=True` 가 가장 안전**. Single-GPU 에서는 unused param 이 그냥 0-grad 로 통과되어 에러 안 남.

---

## 이슈 6 — RTX 5090 (Blackwell) + PyTorch 2.11 의 다중 cuDNN/CUDA 버그

> 이번 디버깅에서 가장 시간을 많이 잡아먹은 단계. **PyTorch 2.11.0+cu128 (cuDNN 9.19) 가 RTX 5090 (sm_120) 에서 큰 입력 backward 시 다양한 깊은 버그 노출**.

### 6-A. `upsample_bilinear2d` size-인자 path 깨짐
**증상**: 1036×1036 forward 자체가 illegal memory access.

**진단 과정**:
```python
F.interpolate(x, size=(1036, 1036), mode='bilinear')    # FAIL
F.interpolate(x, scale_factor=2.0,  mode='bilinear')    # OK
F.interpolate(x, size=(1036, 1036), mode='nearest')     # OK
```
size-기반 bilinear path 만 specific bug.

**임시 해결 (시도)**: 마지막 upsample 을 `mode='nearest'` 로 변경.

### 6-B. `nll_loss_2d_backward` 깨짐
**증상**: forward 가 통과돼도 `cross_entropy` 의 backward 에서 device-side assert `Assertion 't >= 0 && t < n_classes'`. label 이 [0, 12) 범위인데도 fail (= 진짜 label 문제 아니라 다른 op 의 memory corruption 으로 label 이 오염된 듯).

**진단 과정**:
```python
loss = F.cross_entropy(logits_4d, label_3d)             # FAIL
loss = F.cross_entropy(logits_flat, label_flat)         # OK
loss = F.mse_loss(logits, target)                       # OK (NLL 만 깨짐)
```

**임시 해결 (시도)**: focal_loss 와 stage2 CE 호출을 모두 `.permute().reshape(-1, C)` flatten 으로 우회 + ignore_index pre-filter.

### 6-C. 더 깊은 conv backward 버그
**증상**: 위 우회 다 적용해도 단순한 `out.sum().backward()` 도 1036 input 에서 illegal memory access.

**진단**: 학습 진입 가능한 사이즈 한계가 224 ~ 504 사이. cuDNN disable 해도 비-cuDNN conv 가 같은 사이즈에서 forward 자체 깨짐 → **PyTorch 2.11 의 cu128 빌드 자체가 RTX 5090 큰 입력에서 신뢰 불가**.

### 6-D. 최종 해결 — PyTorch 다운그레이드

**모든 6-A/B/C 우회를 원복** 한 뒤 venv 새로 만들고 torch 2.9.1+cu128 (cuDNN 9.10.2) 로 다운그레이드.

**검증**:
```
torch=2.9.1+cu128  cudnn=91002
forward 1036:  0.37s
backward 1036: 0.31s  loss=...
=== BLACKWELL BACKWARD AT 1036 ON 2.9.1: WORKS ===
```
모든 우회 코드 제거 가능. 모델/loss 코드 원복 (`mode='bilinear'`, 4D `F.cross_entropy` 직접 호출, `ignore_index=-1` 직접 사용).

### 교훈
- **최신 stable 이 항상 안정적 인 건 아니다.** 새 GPU 아키텍처 (Blackwell sm_120) 는 정식 출시 직후 도입된 torch 버전에 regression 있을 수 있음.
- **다운그레이드 ≠ 후퇴.** PyTorch 2.9.1 (cuDNN 9.10) 이 같은 GPU 에서 2.11.0 (cuDNN 9.19) 보다 안정적. cuDNN version 이 더 중요할 수 있음.
- **하드웨어 + 소프트웨어 호환성 검증 단계 (smoke test "backward at full input size") 는 학습 진입 전 필수**.

---

## 이슈 7 — torch 2.9.1 의 4번째 GPU detection bug

### 증상
```
RuntimeError: device >= 0 && device < num_gpus INTERNAL ASSERT FAILED
device=3, num_gpus=3
```
4 RTX 5090 시스템인데 torch 2.9.1 이 3 개만 인식. CUDA_VISIBLE_DEVICES=3 단독 시도 → "No CUDA GPUs are available".

### 진단
- nvidia-smi: 4 GPU 모두 정상 보임 (당시)
- torch GPU 0,1,2 모두 단독 사용 OK + small backward OK
- torch GPU 3 만 lazy_init 단계에서 fail

### 해결 (당시)
[run_stage1.sh](run_stage1.sh) / [run_stage2.sh](run_stage2.sh) 의 DDP 설정을 4 GPU → 3 GPU 로 축소:
```bash
export CUDA_VISIBLE_DEVICES=0,1,2
torchrun --nproc-per-node=3 ...
```

### 교훈
**torch 와 nvidia driver 가 동일 GPU 를 다르게 인식할 수 있다.** 특히 새 GPU + 새 torch 조합에서. 학습 진입 전에 각 GPU 단독으로 `torch.cuda.get_device_name` + 간단 op 확인 권장.

---

## 이슈 8 — 시스템 재부팅 후 PCIe 레벨에서 GPU 2개 사라짐

### 증상 (재부팅 직후)
```bash
nvidia-smi
# → 2 GPU 만 보임 (GPU 0: ERR! 상태 + GPU 1)

sudo lspci -tv
# → NVIDIA Device 가 2 개만 enumeration 됨 (이전 4 개)

sudo lspci -vv -s 16:00.0 | grep LnkSta
# → LnkSta: Speed 2.5GT/s (downgraded), Width x16 (ok)
#   → PCIe Gen5 (32GT/s) 능력인데 실제 Gen1 (2.5GT/s) 로 13× 다운그레이드
```

### 진단 (소프트웨어 레벨 한계 내)
- 운영체제가 PCIe 레벨에서 4 → 2 GPU 만 enumerate
- 살아남은 2 GPU 도 PCIe link 가 Gen1 으로 자동 다운그레이드 (`(downgraded)` 키워드)
- = **하드웨어/전원 문제 가능성 강력**

### 가능한 원인 (확률 순)
1. **전원 공급 부족** — 4× RTX 5090 = 4×575W = 2300W 만 GPU. 일반 1500-1600W PSU 로는 부족
2. **PCIe 케이블/라이저 손상 또는 접점 불량** — 발열·진동 영향
3. **이전 cuDNN crash 가 하드웨어 일부 손상**
4. BIOS PCIe lane bifurcation 설정 변경

### 해결 (시도)
- AC 전원 완전 차단 + 5분 대기 + 재기동 → 효과 없음 (여전히 2 GPU)
- IT/하드웨어 점검 사용자에게 위임

### 학습 측 대응 — 1 GPU 모드로 전환
[run_stage1.sh](run_stage1.sh) / [run_stage2.sh](run_stage2.sh) 를 DDP 제거, 단일 프로세스로 변경:
```bash
export CUDA_VISIBLE_DEVICES=0
./DefSeg_AM/venv/bin/python -u -m DefSeg_AM.train_stage1 ...   # torchrun 안 씀
```
- [train_stage1.py](train_stage1.py) 의 `init_distributed()` 가 RANK/WORLD_SIZE env 없으면 자동으로 (rank=0, world_size=1) 반환 → DDP wrap, barrier 모두 skip → 단일 GPU 학습으로 자동 fallback
- run_name 에 `_1gpu` suffix 추가 (이전 ckpt 와 분리)

### 교훈
- **PCIe link speed 는 학습 시작 전 한 번 확인 가치 있음.** `(downgraded)` 키워드가 있으면 신호 무결성 문제 의심.
- **다중 GPU 학습은 항상 single-GPU fallback path 가 코드에 있어야 함.** init_distributed 가 env 없을 때 자동 fallback 하도록 설계해두면 hardware 이슈 시 즉시 우회 가능.

---

## 이슈 9 — Stage 1 학습 완료, 그러나 NaN 발산 + 모델 collapse

### 증상
1 GPU 로 30 epoch 모두 완주. 그러나:
- `train_loss=nan` (모든 epoch summary)
- `val_acc=0.9427` 30 epoch 내내 동일 — 정확히 데이터의 Powder pixel 비율
- per-class IoU: Powder=0.9427, **나머지 11 class 모두 0** → **모델이 모든 pixel 을 Powder 라고 예측하는 trivial collapse**

### 정확한 발산 시점 추적
```
e00 step 2980-3100: loss 0.08~0.18      ← 학습 잘 진행 중
e00 step 3120:      loss 1.7            ← 급증 시작
e00 step 3140-3260: loss 0.18~1.84      ← 출렁임 (보통 발산 직전 신호)
e00 step 3280:      loss=nan            ← 영구 발산
e00~e29 (30 epoch): 모두 동일 val_acc   ← NaN gradient 가 weight 망가뜨림
                                        → forward 도 NaN logits 생성
                                        → argmax(NaN tensor)=0 (PyTorch default)
                                        → 모든 pixel 을 class 0 (Powder) 로 예측
```

### 원인 (복합)

| 요인 | 기여도 |
|---|---|
| **Focal Loss × class_weight=50 × rare class concentrated batch** | 큼 — `(1-pt)^γ × ce` 의 `ce` 항이 weight 곱해져 explosion |
| **AMP (FP16) overflow** | 큼 — FP16 max=65536, focal+α=50 이 이를 넘으면 inf |
| **Gradient clipping 부재** | 큼 — backward grad 크기 무제한 |
| **lr=5e-4 가 13M param decoder 에 과함** | 중 — DINOv2 pretrained head 결합 시 보통 1e-4 이하 |

### 해결 — 5가지 fix 일괄 적용

[config.py](config.py) 상수 변경:
```python
S1_LR = 1e-4                    # 5e-4 → 1e-4
S1_WARMUP_STEPS = 200           # 신규 — 0 → lr linear ramp
S1_GRAD_CLIP_NORM = 1.0         # 신규 — gradient L2-norm clip
S1_CLASS_WEIGHT_CLIP = 10.0     # 50 → 10
# (S2 도 동일 구조: S2_WARMUP_STEPS=50, S2_GRAD_CLIP_NORM=1.0, S2_CLASS_WEIGHT_CLIP=10.0)
```

[train_stage1.py](train_stage1.py) / [train_stage2.py](train_stage2.py):
1. **AMP off** (FP32) — `autocast` context 제거 + `GradScaler(enabled=False)`
2. **Gradient clipping** — `torch.nn.utils.clip_grad_norm_(params, max_norm=1.0)` 를 backward 와 optim.step 사이에 추가
3. **Warmup + Cosine LR scheduler** — `LambdaLR` 로 warmup_steps 동안 linear ramp → 이후 cosine annealing
4. **Class weight clip** — `sqrt_inv_class_weight(counts, clip=10.0)` 로 호출 (기존 default 50 무시)
5. **Loss/grad 모니터링 강화** — batch print 에 `gnorm=...` 추가하여 grad explosion 조기 감지

[run_stage1.sh](run_stage1.sh) / [run_stage2.sh](run_stage2.sh):
- `run_name` 에 `_nanfix` suffix 추가 → 이전 망가진 ckpt 와 분리
- 망가진 ckpt 디렉터리 (`checkpoints/vits14_dpt_dual_sz1036_1gpu/`) 는 `..._NaN_failed_20260527` 로 rename 보존
- (2026-05-28 cleanup) NaN-collapsed ckpt 와 빈 DDP 시도 폴더는 이 문서로 기록 대체하고 삭제. 학습 끝난 유효 ckpt 는 `checkpoints/vits14_dpt_dual_sz1036_1gpu_nanfix/stage1_best.pt` 한 개만 남음.

### 검증
새 코드로 mini smoke test (224 입력 1 batch):
```
class_weights (clip=10): [0.324, 4.763, 0.0, 9.35, 10.0, 10.0, ..., 10.0]   ← clip 작동
forward+backward+grad_clip OK: loss=18.5087  grad_norm=9.3352
                                            ↑ clip 없으면 9.33, 있으면 1.0 으로 잘림
```

### 교훈
- **Focal Loss + 큰 class weight + AMP 조합은 수치 폭주 위험 높음.** 특히 class weight clip 을 50 같은 큰 값으로 두면 batch 에 rare class 가 몰릴 때 loss 가 폭발.
- **Gradient clipping 은 train script 의 default 옵션으로 두는 게 안전.** Overhead 미미.
- **Warmup 은 학습 초반 (~수백 step) 의 큰 lr 으로 인한 발산을 막는 표준 도구.** Transformer/ViT 계열 학습에서 거의 필수.
- **NaN 한 번 나면 30 epoch 다 의미 없음.** 학습 시작 직후 (~수백 step) loss / grad_norm 모니터링하고 발산 시 즉시 중단 → 하이퍼파라미터 재조정 사이클이 빠름.

---

## 누적된 코드 변경 요약

이슈 해결 과정에서 새로 만들거나 크게 수정된 파일:

| 파일 | 역할 | 도입 시점 |
|---|---|---|
| [build_cache_stage1.py](build_cache_stage1.py) | 사전 resize/uint8/argmax 캐시 빌드 | 이슈 2 |
| [data_ornl.py](data_ornl.py) `DefSegORNLCachedDataset` | memmap 기반 batch dataset | 이슈 2 |
| [samplers.py](samplers.py) `DistributedWeightedSampler` | DDP 호환 weighted sampling | 이슈 3 |
| [log.py](log.py) | timestamp + rank-prefix logger | 이슈 3 |
| [run_build_cache.sh](run_build_cache.sh) | 캐시 빌드 진입점 | 이슈 2 |
| [run_stage1.sh](run_stage1.sh) / [run_stage2.sh](run_stage2.sh) | DDP/SingleGPU 설정 + env 변수 | 이슈 1,3,4,7,8,9 누적 |
| [model.py](model.py) | DPT decoder + DINOv2 backbone | (원래 있음, 이슈 5/6 디버깅 대상) |
| [losses.py](losses.py) | focal_loss + sqrt_inv weight | (원래 있음, 이슈 9 에서 clip param 사용 강화) |
| [train_stage1.py](train_stage1.py) | DDP + AMP + warmup + grad_clip 통합 | 이슈 3,5,9 |
| [train_stage2.py](train_stage2.py) | 위와 동일 + Stage 1 ckpt load | 이슈 3,5,9 |

---

## 최종 학습 설정 (현재 상태)

| 항목 | 값 |
|---|---|
| GPU | 1× RTX 5090 (CUDA_VISIBLE_DEVICES=0) |
| 입력 해상도 | 1036×1036 |
| Batch size | 2 per GPU (effective=2) |
| Backbone | DINOv2 ViT-S/14 frozen |
| Trainable params | 13.38 M (decoder + head 만) |
| Precision | FP32 (AMP off) |
| Optimizer | AdamW lr=1e-4, weight_decay=1e-4 |
| LR scheduler | Linear warmup 200 step → Cosine annealing |
| Gradient clip | L2-norm max=1.0 |
| Loss (Stage 1) | Focal Loss γ=2.0 + α=sqrt-inv weight (clip=10) |
| Loss (Stage 2) | Standard CE + α=sqrt-inv weight (clip=10) |
| Sampler (Stage 1) | DistributedWeightedSampler (defect_ratio^0.5 oversampling) |
| Epochs | Stage 1: 30 / Stage 2: 50 |
| 1 epoch 예상 시간 | Train ~84분 + Val ~13분 = **~97분/epoch** (1 GPU, PCIe Gen1) |
| 30 epoch 총 | **~48 시간** |
| 캐시 | `cache/resized_sz1036/` (48 GB, 1회 빌드) |

---

## 향후 개선 가능 항목 (current → ideal)

1. **PCIe / PSU 하드웨어 복구** → 4 GPU DDP 로 복귀 시 학습 시간 1/4 (현재 48h → 12h)
2. **PyTorch 2.12+ 안정화 후 재시도** → AMP (FP16) 다시 켜서 속도 1.5~2x ↑
3. **`fusion_blocks[3]` 의 `res_skip` 제거** (별도 클래스) → `find_unused_parameters=False` 로 DDP overhead 미세 ↓
4. **Loss 모니터링 자동화** — `loss.item()` 이 NaN 이면 즉시 stop + dump batch 저장 → 디버깅 용이
5. **Tensorboard / wandb 통합** — text log 가 아닌 시각 그래프로 grad_norm/loss 추이 추적

---

> 작성: 2026-05-27. 디버깅 세션 끝나는 시점.
> 다음 사람 (또는 미래의 본인) 이 비슷한 이슈를 만나면 이 문서의 해당 섹션부터 보면 됩니다.
