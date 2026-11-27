"""합성 세트를 만든다 — 라벨 맵으로 그린 레이어 N 장을 사람 GT 데이터셋과 같은 구성으로 쓴다 (v7/PLAN_v7.md §2).

한 장 = ① 배경 레이어 (ORNL 빌드의 CAD 파트 배치 + 그 사진의 가장자리 5%) 를 고른다 → ② 라벨 맵을 만든다 (`label_map.py`)
       → ③ 생성 모델이 안쪽을 그린다 (`layer.py`) → ④ 사진 두 장 · 라벨 · 내력을 쓴다.

실행 (CWD = 3DP_VPPM/, DefSeg_AM/venv — compose 의 synth phase):
    python -m DefSeg_AM.genaug.synth.write_set                                   # w600_v1, 600 장, 보이는 GPU 전부
    python -m DefSeg_AM.genaug.synth.write_set --name w2000_v1 --n-layers 2000
    python -m DefSeg_AM.genaug.synth.write_set --quick                           # 스모크 — 6 장, 8 단계 (그림의 질에는 뜻이 없다)

- **같은 명령은 같은 세트를 만든다.** 레이어 i 의 배경 · 라벨 맵 · 잡음은 (seed, i) 로만 정해진다. 장 수를 늘려 다시 돌리면 없는 장만 더 만든다.
- 끊겼으면 같은 명령을 다시 준다 — 다 만든 묶음은 건너뛴다. 설정이 다르면 멈춘다 (`--overwrite` 로 지우고 다시 만든다).
- GPU 가 여럿이면 묶음 (기본 8 장) 을 GPU 에 번갈아 나눠 준다. 묶음의 구성은 GPU 수와 무관하다.
- 결함의 모양은 생성 모델이 본 사람 GT 에서만 가져온다 (ckpt 에 적힌 holdout 을 그대로 쓴다) — 세트를 쓸 수 있는 판정 fold 가 그것으로 정해진다.
- DSCNN 예측은 읽지 않는다. 분할 모델의 예측으로 장을 고르지도 않는다.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import shutil
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

from ...common.utils.log import setup_logger
from .. import config_genaug as cfg
from ..data.canvas_pool import CanvasPool
from ..data.gt_pool import GTPool
from ..sampling.fill import load_generator
from .label_map import Rules, ShapeBank, build_label_map, to_annotation
from .layer import ORDERS, draw_layers, frame_px, window_order, window_positions

INDEX_HEADER = ("id,agency,mc_id,pp_id,process_code,layer,status,labeler,updated_at,width,height,unlabeled_px,"
                "swap_sd,recoat_axis,machine,material,process_type,orig_ann").split(",")
# holdout 종류 → 이 세트를 학습에 넣어도 되는 판정 fold (그 fold 의 검증 레이어를 생성 모델이 보지 않았다)
VALID_FOLDS = {"both": ["inlayer", "loso_L21"], "inlayer": ["inlayer"]}


def layer_specs(n_layers: int, builds: tuple[str, ...], canvas: CanvasPool, seed: int) -> list[dict]:
    """레이어 i → 배경. 빌드를 번갈아 쓰고, 빌드 안에서는 seed 로 섞은 순서대로 겹치지 않게 고른다."""
    perms = {b: np.random.default_rng([seed, 1000 + k]).permutation(canvas.n_layers[b]) for k, b in enumerate(builds)}
    specs = []
    for i in range(n_layers):
        b = builds[i % len(builds)]
        k = i // len(builds)
        if k >= len(perms[b]):
            raise SystemExit(f"FATAL: 빌드 {b} 의 배경 레이어 ({len(perms[b])}) 보다 많이 만들 수 없다 — 장 수를 줄이거나 빌드를 더한다")
        row = int(perms[b][k])
        specs.append({"index": i, "id": f"{i:07d}", "build": b, "row": row, "orig_layer": int(canvas.orig_layers[b][row])})
    return specs


def torch_seed(seed: int, index: int) -> int:
    return (int(seed) * 1_000_003 + int(index)) % (2 ** 62)


def _done(root: Path, sid: str) -> bool:
    return (root / "meta" / f"{sid}.json").is_file()


def _save_layer(root: Path, spec: dict, img: np.ndarray, ann: np.ndarray, placed: list[dict], seed: int) -> None:
    sid = spec["id"]
    for v in (0, 1):
        tmp = root / "visible" / str(v) / f".{sid}.tmp.png"
        Image.fromarray(img[v]).save(tmp)
        tmp.replace(root / "visible" / str(v) / f"{sid}.png")
    tmp = root / "annotations" / f".{sid}.tmp.npy"
    np.save(tmp, ann)
    tmp.replace(root / "annotations" / f"{sid}.npy")
    meta = dict(spec, seed=int(seed), defects=placed,
                class_px=[int(v) for v in np.bincount(ann[ann >= 0].astype(np.int64), minlength=cfg.N_CLASSES)],
                unlabeled_px=int((ann < 0).sum()))
    tmp = root / "meta" / f".{sid}.tmp"
    tmp.write_text(json.dumps(meta, ensure_ascii=False), encoding="utf-8")
    tmp.replace(root / "meta" / f"{sid}.json")              # 이 파일이 "이 장은 끝났다" 는 표시다


def work(rank: int, n_workers: int, a: dict) -> None:
    """GPU 하나가 맡은 묶음을 만든다."""
    log = setup_logger(0, f"genaug.synth.{rank}")
    tag = f"[gpu {rank}] " if n_workers > 1 else ""
    dev = torch.device(f"cuda:{rank}" if torch.cuda.is_available() else "cpu")
    root = Path(a["root"])
    model, diff, ck = load_generator(a["ckpt"], dev)
    gt = GTPool(holdout=a["holdout"], log=log if rank == 0 else None)
    builds = tuple(a["builds"])
    canvas = CanvasPool(builds, log=log if rank == 0 else None)
    if canvas.img_size != gt.img_size:
        raise SystemExit(f"FATAL: 배경 캐시의 크기 {canvas.img_size} 가 사람 GT 의 {gt.img_size} 와 다르다")
    rules = Rules(classes=tuple(a["classes"]))
    bank = ShapeBank(gt, rules)
    S = canvas.img_size
    E = frame_px(S)
    specs = layer_specs(a["n_layers"], builds, canvas, a["seed"])
    B = a["batch"]
    chunks = [specs[k:k + B] for k in range(0, len(specs), B)]
    mine = [c for k, c in enumerate(chunks) if k % n_workers == rank and not all(_done(root, s["id"]) for s in c)]
    log.info(f"{tag}맡은 묶음 {len(mine)} 개 ({sum(len(c) for c in mine)} 장)")
    t0, n_made = time.time(), 0
    for c in mine:
        r0 = np.empty((len(c), S, S), dtype=np.uint8)
        r1 = np.empty_like(r0)
        labs = np.empty_like(r0)
        placed = []
        for k, s in enumerate(c):
            r0[k], r1[k], part = canvas.crop(s["build"], s["row"], 0, 0, S, S)
            labs[k], p = build_label_map(part, np.random.default_rng([a["seed"], s["index"]]), bank, rules, E)
            placed.append(p)
        seeds = [torch_seed(a["seed"], s["index"]) for s in c]
        img = draw_layers(model, diff, r0, r1, labs, seeds, steps=a["steps"], guidance=a["guidance"],
                          spacing=a["spacing"], stride=a["stride"], order=a["order"], amp=a["amp"])
        for k, s in enumerate(c):
            _save_layer(root, s, img[k], to_annotation(labs[k], E), placed[k], seeds[k])
        n_made += len(c)
        el = time.time() - t0
        log.info(f"{tag}{n_made}/{sum(len(x) for x in mine)} 장  ({el:.0f}초, 장당 {el / n_made:.1f}초)")


def file_sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for blk in iter(lambda: fh.read(1 << 22), b""):
            h.update(blk)
    return h.hexdigest()


def finalize(root: Path, a: dict, identity: dict, created: str, log) -> dict:
    """index.csv · classes.json · manifest.json — 모든 장이 있는지 확인한 뒤에 쓴다."""
    canvas = CanvasPool(tuple(a["builds"]))
    specs = layer_specs(a["n_layers"], tuple(a["builds"]), canvas, a["seed"])
    missing = [s["id"] for s in specs if not _done(root, s["id"])]
    if missing:
        raise SystemExit(f"FATAL: {len(missing)} 장이 없다 (예: {missing[:3]}) — 같은 명령을 다시 돌린다")
    S = canvas.img_size
    layers, rows = [], []
    class_px = np.zeros(cfg.N_CLASSES, dtype=np.int64)
    for s in specs:
        m = json.loads((root / "meta" / f"{s['id']}.json").read_text(encoding="utf-8"))
        if (m["build"], m["row"]) != (s["build"], s["row"]):
            raise SystemExit(f"FATAL: {s['id']} 의 내력이 지금 설정과 다르다 — 다른 설정으로 만든 장이 섞였다")
        for f in (root / "visible" / "0" / f"{s['id']}.png", root / "visible" / "1" / f"{s['id']}.png",
                  root / "annotations" / f"{s['id']}.npy"):
            if not f.is_file():
                raise SystemExit(f"FATAL: {f} 없음")
        layers.append(m)
        class_px += np.array(m["class_px"], dtype=np.int64)
        b = m["build"]
        rows.append(dict(
            id=m["id"], agency="synthetic", mc_id="ConceptLaser_M2", pp_id=f"{cfg.SET_GROUP_PREFIX}{b}",
            process_code=f"{cfg.SET_GROUP_PREFIX}{b}", layer=m["index"], status="generated",
            labeler=f"genaug {identity['generator']['run_name']} (label map)", updated_at=created, width=S, height=S,
            unlabeled_px=m["unlabeled_px"], swap_sd=0, recoat_axis="horizontal", machine="ConceptLaser M2 (generated)",
            material="Stainless Steel 316L", process_type="LPBF",
            orig_ann=f"label map: CAD of {b} layer {m['orig_layer']} + rule-placed defects",
        ))
    with (root / "index.csv").open("w", encoding="utf-8-sig", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=INDEX_HEADER)
        w.writeheader()
        w.writerows(rows)
    shutil.copy(cfg.GT_ROOT / "classes.json", root / "classes.json")
    AB = cfg.CLASS_ABBRS
    names = [AB[c] for c in a["classes"]]
    doc = dict(
        name=root.name, created=created,
        purpose="v7 synthetic layers — whole layers drawn from label maps (outer 5% of each photo is real)",
        n_layers=len(layers), valid_folds=VALID_FOLDS.get(a["holdout"], []), **identity,
        class_px={AB[c]: int(class_px[c]) for c in range(cfg.N_CLASSES)},
        class_px_per_layer={AB[c]: round(float(class_px[c]) / len(layers), 1) for c in range(cfg.N_CLASSES)},
        layers_with={n: sum(1 for m in layers if any(d["cls"] == n for d in m["defects"])) for n in names},
        objects={n: sum(sum(1 for d in m["defects"] if d["cls"] == n) for m in layers) for n in names},
        layers_without_defect=sum(1 for m in layers if not m["defects"]),
        layers_per_build={b: sum(1 for m in layers if m["build"] == b) for b in a["builds"]},
        layers=layers,
    )
    (root / "manifest.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    log.info(f"세트 {root.name}: {len(layers)} 장 — 결함이 든 장 {doc['layers_with']}, 결함 없는 장 {doc['layers_without_defect']}")
    log.info(f"  클래스별 화소 {doc['class_px']}")
    log.info(f"  객체 수 {doc['objects']}   쓸 수 있는 판정 fold {doc['valid_folds']}")
    return doc


def main():
    ap = argparse.ArgumentParser(description="GenAug — 합성 세트 (라벨 맵으로 그린 레이어)")
    ap.add_argument("--name", default=None, help=f"세트 이름 (기본 {cfg.SET_NAME}) — cache/genaug/sets/<이름>/")
    ap.add_argument("--n-layers", type=int, default=cfg.SET_LAYERS)
    ap.add_argument("--builds", nargs="+", default=list(cfg.SET_BUILDS), help="배경으로 쓸 빌드 (생성 모델이 본 빌드)")
    ap.add_argument("--classes", nargs="+", default=[cfg.CLASS_ABBRS[c] for c in cfg.SET_CLASSES],
                    help="넣을 결함 (RS RH SE DB 가운데)")
    ap.add_argument("--run-name", default=cfg.RUN_NAME, help="생성 모델의 run")
    ap.add_argument("--ckpt", type=Path, default=None, help="기본: checkpoints/genaug/<run>/last.pt (EMA 사본으로 그린다)")
    ap.add_argument("--seed", type=int, default=cfg.SET_SEED)
    ap.add_argument("--steps", type=int, default=cfg.SAMPLE_STEPS)
    ap.add_argument("--spacing", default=cfg.SET_SPACING, choices=("uniform", "quad"))
    ap.add_argument("--guidance", type=float, default=cfg.SAMPLE_CFG)
    ap.add_argument("--stride", type=int, default=cfg.SET_STRIDE, help="창 간격의 상한 (작을수록 더 겹치고 오래 걸린다)")
    ap.add_argument("--order", default=cfg.SET_ORDER, choices=ORDERS)
    ap.add_argument("--batch", type=int, default=cfg.SET_BATCH, help="한 번에 그리는 레이어 수 (GPU 당)")
    ap.add_argument("--gpus", type=int, default=0, help="쓸 GPU 수 (0 = 보이는 것 전부)")
    ap.add_argument("--no-amp", dest="amp", action="store_false", default=cfg.USE_AMP)
    ap.add_argument("--quick", action="store_true", help="스모크 — 6 장, 8 단계, 이름에 _quick")
    ap.add_argument("--overwrite", action="store_true", help="같은 이름의 세트를 지우고 다시 만든다")
    args = ap.parse_args()
    log = setup_logger(0, "genaug.synth")

    name = args.name or (cfg.SET_NAME + ("_quick" if args.quick else ""))
    if args.quick:
        args.n_layers, args.steps = min(args.n_layers, 6), min(args.steps, 8)
    root = cfg.set_dir(name)
    ckpt = args.ckpt or (cfg.ckpt_dir(args.run_name) / "last.pt")
    if not ckpt.is_file():
        raise SystemExit(f"FATAL: {ckpt} 없음 — 생성 모델을 먼저 학습한다 (compose 의 train phase)")
    bad = [c for c in args.classes if c not in ("RS", "RH", "SE", "DB")]
    if bad:
        raise SystemExit(f"FATAL: 넣을 수 있는 결함은 RS RH SE DB 다: {bad}")
    classes = [cfg.CLASS_ABBRS.index(c) for c in args.classes]

    ck = torch.load(ckpt, map_location="cpu", weights_only=False)
    gc, gen_step = ck["config"], int(ck["step"])
    del ck
    holdout = gc["holdout"]
    not_seen = [b for b in args.builds if b not in gc["train_builds"]]
    if not_seen:
        raise SystemExit(f"FATAL: 생성 모델이 보지 않은 빌드 {not_seen} 는 배경으로 쓰지 않는다 (본 빌드: {gc['train_builds']})")
    S = cfg.IMG_SIZE
    E = frame_px(S)
    pos = window_positions(S, E, cfg.WIN, args.stride)
    identity = dict(
        generator=dict(run_name=gc["run_name"], ckpt=str(ckpt.relative_to(cfg.CHECKPOINT_DIR.parent)) if ckpt.is_relative_to(cfg.CHECKPOINT_DIR.parent) else str(ckpt),
                       step=gen_step, sha256=file_sha256(ckpt), weights="ema",
                       train_builds=list(gc["train_builds"]), normal_builds=list(gc["normal_builds"])),
        gt_holdout=holdout, seed=int(args.seed), builds=list(args.builds), classes=list(args.classes),
        rules=Rules(classes=tuple(classes)).to_dict(),
        sampler=dict(steps=int(args.steps), spacing=args.spacing, guidance=float(args.guidance), amp=bool(args.amp),
                     win=cfg.WIN, stride=int(args.stride), order=args.order, window_positions=pos,
                     windows_per_layer=len(pos) ** 2, frame_px=E, batch=int(args.batch)),
        img_size=S, quick=bool(args.quick),
    )
    identity = json.loads(json.dumps(identity))               # 파일에 썼다 읽은 것과 같은 꼴로 (tuple → list) — 아래의 비교가 이것으로 한다

    if root.exists() and args.overwrite:
        if not ((root / "set_config.json").is_file() or (root / "manifest.json").is_file()):
            raise SystemExit(f"FATAL: {root} 는 합성 세트 폴더로 보이지 않는다 — 지우지 않는다")
        log.info(f"--overwrite: {root} 를 지운다")
        shutil.rmtree(root)
    conf = root / "set_config.json"
    if conf.is_file():
        old = json.loads(conf.read_text(encoding="utf-8"))
        if old["identity"] != identity:
            diff_keys = [k for k in identity if old["identity"].get(k) != identity[k]]
            raise SystemExit(f"FATAL: {root} 는 다른 설정으로 만든 세트다 (다른 항목: {diff_keys}) — 이름을 바꾸거나 --overwrite")
        created = old["created"]
    else:
        if root.exists() and any(root.iterdir()):
            raise SystemExit(f"FATAL: {root} 가 이미 있는데 set_config.json 이 없다 — 이 스크립트가 만든 폴더가 아니다")
        created = time.strftime("%Y-%m-%dT%H:%M:%S+09:00")
        for sub in ("visible/0", "visible/1", "annotations", "meta"):
            (root / sub).mkdir(parents=True, exist_ok=True)
        conf.write_text(json.dumps({"created": created, "identity": identity}, indent=1, ensure_ascii=False), encoding="utf-8")

    n_gpu = torch.cuda.device_count()
    n_workers = max(1, min(n_gpu, args.gpus) if args.gpus > 0 else n_gpu)
    a = dict(root=str(root), ckpt=str(ckpt), holdout=holdout, builds=list(args.builds), classes=classes, n_layers=int(args.n_layers),
             seed=int(args.seed), batch=int(args.batch), steps=int(args.steps), guidance=float(args.guidance),
             spacing=args.spacing, stride=int(args.stride), order=args.order, amp=bool(args.amp))
    log.info(f"세트 {name}: {args.n_layers} 장 ({', '.join(args.builds)}), 결함 {args.classes}, seed {args.seed}")
    log.info(f"  생성 모델 {identity['generator']['ckpt']} (holdout={holdout}) — {args.steps} 단계 ({args.spacing}), guidance {args.guidance}, "
             f"창 {len(pos)}×{len(pos)} ({args.order}, 자리 {pos}), 가장자리 {E} px, 묶음 {args.batch} 장, GPU {n_gpu} 대 가운데 {n_workers} 대")
    if n_gpu == 0:
        log.info("  주의: GPU 가 없다 — CPU 로 그린다 (아주 느리다)")
    t0 = time.time()
    if n_workers == 1:
        work(0, 1, a)
    else:
        ctx = torch.multiprocessing.get_context("spawn")
        procs = [ctx.Process(target=work, args=(r, n_workers, a)) for r in range(n_workers)]
        for p in procs:
            p.start()
        for p in procs:
            p.join()
        if any(p.exitcode != 0 for p in procs):
            raise SystemExit(f"FATAL: 작업 프로세스가 실패했다 (exit code {[p.exitcode for p in procs]}) — 같은 명령을 다시 돌리면 남은 묶음부터 만든다")
    finalize(root, a, identity, created, log)
    log.info(f"끝 — {root}  ({(time.time() - t0) / 60:.1f}분)")


if __name__ == "__main__":
    main()
