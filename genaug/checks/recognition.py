"""K2 · K3 · K6 — 분할 모델이 합성 레이어를 어떻게 읽는가 (v7/PLAN_v7.md §2.6).

합성 레이어를 이미 학습된 분할 모델에 넣고, 그리라고 한 라벨과 모델의 예측을 화소 단위로 견준다.

  K2  결함을 그 클래스로 읽는가            기준 (제안값): 그 모델의 진짜 재현율의 절반 이상
  K3  정상으로 그린 곳을 정상으로 읽는가    기준 (제안값): Powder · Printed 의 90% 이상이 그대로. 결함으로 읽는 비율이 같은 레이어의 진짜 사진에서의 2배 이하
  K6  그 장비 (L21) 를 본 적 없는 모델도 읽는가 — 가르칠 것이 남아 있는지의 참고값 (기록만)

**이 표로 합성 레이어를 한 장씩 고르지 않는다** — 클래스를 통째로 넣을지 뺄지만 정한다 (PLAN_v7 §2.4).
분할 모델은 `common` 의 `DefSegModel` 로 읽는다 (버전 폴더를 임포트하지 않는다). ckpt 가 `checkpoints/<run>/cv/<fold>/last.pt` 꼴이면
그 모델의 진짜 재현율을 `figures/<run>/v6/cv/<fold>/eval_native_last/metrics.json` 에서 찾아 함께 적는다 (없으면 생략).

실행 (CWD = 3DP_VPPM/, DefSeg_AM/venv — compose 의 check phase. GPU 1대):
    python -m DefSeg_AM.genaug.checks.recognition --set w600_v1 --models p0_inlayer=<ckpt> [lo_n_s42=<ckpt>]
산출물: figures/genaug/sets/<세트>/recognition_<이름>.json
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import torch

from ...common.data.image_utils import normalize_image
from ...common.models.model import DefSegModel
from ...common.utils.log import setup_logger
from .. import config_genaug as cfg
from ..data.canvas_pool import CanvasPool
from ..sampling.preview import cad_label
from ..synth.layer import frame_px
from ..synth.set_io import pick_layers, read_layer, read_manifest, resolve_set

AB = cfg.CLASS_ABBRS


def load_seg(path: Path, device) -> tuple[DefSegModel, dict]:
    ck = torch.load(path, map_location="cpu", weights_only=False)
    c = ck.get("config", {}) or {}
    if c.get("n_classes") != cfg.N_CLASSES or c.get("schema") != cfg.SCHEMA_VERSION:
        raise SystemExit(f"FATAL: {path} 는 {cfg.SCHEMA_VERSION} 의 {cfg.N_CLASSES}-class 모델이 아니다 "
                         f"(n_classes={c.get('n_classes')}, schema={c.get('schema')})")
    model = DefSegModel(backbone_name=c["backbone"], n_classes=cfg.N_CLASSES).to(device)
    model.load_trainable_state_dict(ck["model_state"])
    model.eval()
    return model, {"path": str(path), "version": c.get("version"), "fold": c.get("fold"), "weights": ck.get("weights"),
                   "step": ck.get("step"), "img_size": c.get("img_size"), "backbone": c["backbone"]}


@torch.no_grad()
def predict(model: DefSegModel, i0: np.ndarray, i1: np.ndarray, device) -> np.ndarray:
    t0 = torch.from_numpy(normalize_image(i0)).unsqueeze(0).to(device)
    t1 = torch.from_numpy(normalize_image(i1)).unsqueeze(0).to(device)
    return model(t0, t1)[0].argmax(0).cpu().numpy()


def real_recall(ckpt: Path) -> dict | None:
    """ckpt 의 자리로 그 모델의 진짜 재현율 (원본 해상도 채점) 을 찾는다. 없으면 None."""
    p = ckpt.resolve().parts
    try:
        k = p.index("checkpoints")
        run, fold = p[k + 1], p[k + 3]
        f = Path(*p[:k]) / "figures" / run / "v6" / "cv" / fold / "eval_native_last" / "metrics.json"
        m = json.loads(f.read_text(encoding="utf-8"))["overall"]
        return {AB[c]: m["recall"][c] for c in range(cfg.N_CLASSES)}
    except Exception:
        return None


def table(conf: np.ndarray, names: list[str]) -> dict:
    out = {}
    for i, n in enumerate(names):
        tot = int(conf[i].sum())
        if tot:
            row = {AB[c]: round(100.0 * conf[i, c] / tot, 2) for c in range(cfg.N_CLASSES) if conf[i, c] / tot >= 0.0005}
            out[n] = {"px": tot, "as": row}
    return out


def defect_frac(conf_rows: np.ndarray) -> float:
    tot = conf_rows.sum()
    return float(conf_rows[:, 2:].sum() / tot) if tot else float("nan")


def main():
    ap = argparse.ArgumentParser(description="GenAug K2 · K3 · K6 — 분할 모델의 인식률")
    ap.add_argument("--set", default=cfg.SET_NAME)
    ap.add_argument("--models", nargs="+", required=True, help="이름=ckpt 경로 (여럿)")
    ap.add_argument("--max-layers", type=int, default=0, help="재는 장 수 (0 = 전부)")
    args = ap.parse_args()
    log = setup_logger(0, "genaug.recog")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    root = resolve_set(args.set)
    man = read_manifest(root)
    S = man["img_size"]
    E = frame_px(S)
    canvas = CanvasPool(tuple(man["builds"]))
    layers = pick_layers(man, args.max_layers)
    inner = np.zeros((S, S), dtype=bool)
    inner[E:S - E, E:S - E] = True
    out = cfg.set_fig_dir(man["name"])
    out.mkdir(parents=True, exist_ok=True)
    targets = list(man["classes"])

    for spec in args.models:
        name, _, path = spec.partition("=")
        model, info = load_seg(Path(path), device)
        if info["img_size"] != S:
            raise SystemExit(f"FATAL: 모델의 입력 크기 {info['img_size']} 가 세트의 {S} 와 다르다")
        gen = np.zeros((cfg.N_CLASSES, cfg.N_CLASSES), dtype=np.int64)        # 그리라고 한 라벨 × 예측 (합성 레이어의 안쪽)
        real = np.zeros((2, cfg.N_CLASSES), dtype=np.int64)                   # CAD 의 (분말, 파트) × 예측 (같은 레이어의 진짜 사진)
        per_build = {b: [np.zeros_like(gen), np.zeros_like(real)] for b in man["builds"]}
        for k, m in enumerate(layers):
            g0, g1, ann = read_layer(root, m["id"])
            r0, r1, part = canvas.crop(m["build"], m["row"], 0, 0, S, S)
            pg, pr = predict(model, g0, g1, device), predict(model, r0, r1, device)
            ok = ann >= 0
            cg = np.bincount(ann[ok].astype(np.int64) * cfg.N_CLASSES + pg[ok].astype(np.int64),
                             minlength=cfg.N_CLASSES ** 2).reshape(cfg.N_CLASSES, cfg.N_CLASSES)
            base = cad_label(part)
            cr = np.stack([np.bincount(pr[inner & (base == c)].astype(np.int64), minlength=cfg.N_CLASSES) for c in (cfg.POWDER, cfg.PRINTED)])
            gen += cg
            real += cr
            per_build[m["build"]][0] += cg
            per_build[m["build"]][1] += cr
            if (k + 1) % 100 == 0:
                log.info(f"  [{name}] {k + 1}/{len(layers)} 장")
        rr = real_recall(Path(path))
        rate = {AB[c]: (float(gen[c, c] / gen[c].sum()) if gen[c].sum() else None) for c in range(cfg.N_CLASSES)}
        k2 = {}
        for c in targets:
            ref = rr.get(c) if rr else None
            k2[c] = {"recognized": rate[c], "real_recall": ref,
                     "pass": (rate[c] >= 0.5 * ref) if (rate[c] is not None and ref) else None}
        gen_def, real_def = defect_frac(gen[:2]), defect_frac(real)
        k3 = {"powder_as_powder": rate["Pow"], "printed_as_printed": rate["Prt"],
              "normal_read_as_defect_generated": gen_def, "normal_read_as_defect_real_photo": real_def,
              "pass": bool(rate["Pow"] >= 0.9 and rate["Prt"] >= 0.9 and gen_def <= 2.0 * real_def)}
        doc = {"set": man["name"], "n_layers": len(layers), "model": dict(info, name=name),
               "note": "generated = 그리라고 한 라벨 → 예측 (%), real = 같은 ORNL 레이어의 진짜 사진에서 CAD 의 분말 · 파트 자리 → 예측 (%). "
                       "진짜 사진의 결함 예측에는 진짜 결함이 들어 있다 (사람 GT 가 없는 사진이다)",
               "K2": k2, "K3": k3, "generated": table(gen, AB), "real": table(real, ["CAD powder", "CAD part"]),
               "per_build": {b: {"generated": table(v[0], AB), "real": table(v[1], ["CAD powder", "CAD part"]),
                                 "normal_read_as_defect_generated": defect_frac(v[0][:2]),
                                 "normal_read_as_defect_real_photo": defect_frac(v[1])} for b, v in per_build.items()}}
        (out / f"recognition_{name}.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
        log.info(f"=== 인식률 — 모델 {name} ({info['version']} / fold {info['fold']} / step {info['step']}), 세트 {man['name']} 의 {len(layers)} 장")
        for c in targets:
            row = doc["generated"].get(c)
            if row is None:
                continue
            ref = f", 그 모델의 진짜 재현율 {k2[c]['real_recall']:.2f} → {'통과' if k2[c]['pass'] else '탈락'}" if k2[c]["real_recall"] else ""
            log.info(f"  K2 {c}: {row['px']:>9,d} px 가운데 {100 * rate[c]:.1f}% 를 {c} 로 읽는다{ref}   (나머지: "
                     + ", ".join(f"{a} {v}%" for a, v in sorted(row["as"].items(), key=lambda t: -t[1]) if a != c)[:90] + ")")
        log.info(f"  K3 Powder → Powder {100 * rate['Pow']:.1f}%, Printed → Printed {100 * rate['Prt']:.1f}%. 정상으로 그린 곳을 결함으로 읽은 비율 "
                 f"{100 * gen_def:.2f}% (같은 레이어의 진짜 사진 {100 * real_def:.2f}%) → {'통과' if k3['pass'] else '탈락'}")
        log.info(f"  → {out / f'recognition_{name}.json'}")
        del model
        if device.type == "cuda":
            torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
