"""K4 — 창을 이은 자리가 보이는가 (v7/PLAN_v7.md §2.6).

레이어는 256 창 25개를 이어 그린 것이다. 분말로 그리라고 한 화소에서, 사진 (visible/0 · visible/1) 과 이음매의 방향 (세로 · 가로) 을 갈라 잰다.

  ① 토막의 계단     이음매 양쪽 w px 띠의 평균 밝기 차 — 이음매를 따라 32 px 토막마다 (w = 8, 32)
  ② 한 변의 계단    같은 값을 창의 한 변 (이어진 토막 3개 이상) 에 걸쳐 평균한 것 (w = 8). 눈에 보이는 "네모" 는 이것이다 —
                    토막마다의 차이는 분말의 무늬에 묻히지만 한 변 전체가 같은 쪽으로 어긋나면 선으로 보인다
  ③ 창 단위의 얼룩  32 px 칸의 평균 밝기를 "그 칸을 그린 창" 으로 묶었을 때 묶음 사이의 표준편차

견주는 값은 둘이다: **같은 레이어의 진짜 사진의 같은 자리** (진짜 사진에는 이음매가 없으므로 그 자리의 자연스러운 밝기 변화다 — 가로 방향은 리코터 줄 때문에 크다),
그리고 **그린 사진 안에서 이음매가 아닌 자리** (대조군). 기준 (제안값): ② 가 진짜의 2배 이하. 대조군과의 비는 함께 적는다.

실행 (CWD = 3DP_VPPM/, DefSeg_AM/venv — compose 의 check phase. GPU 를 쓰지 않는다):
    python -m DefSeg_AM.genaug.checks.seam --set w600_v1
산출물: figures/genaug/sets/<세트>/seam.json
"""
from __future__ import annotations

import argparse
import json

import numpy as np

from ...common.utils.log import setup_logger
from .. import config_genaug as cfg
from ..data.canvas_pool import CanvasPool
from ..synth.layer import frame_px, owner_map
from ..synth.set_io import pick_layers, read_layer, read_manifest, resolve_set

BLOCK = 32            # 이음매를 따라 자르는 토막의 길이, 얼룩을 재는 칸의 크기
WIDTHS = (8, 32)      # 이음매 양쪽 띠의 폭
EDGE_W = 8            # "한 변의 계단" 을 재는 띠의 폭
CONTROL_SHIFT = 60    # 한 변의 대조선 — 이음매에서 이만큼 떨어진 같은 길이의 선
MIN_VALID = 0.6       # 띠 · 칸에서 분말 화소가 이만큼은 있어야 잰다
AXES = {"v": "세로 이음매 (좌우 창 사이)", "h": "가로 이음매 (위아래 창 사이)"}


def _steps(img: np.ndarray, ok: np.ndarray, w: int) -> np.ndarray:
    """세로 이음매용 — (토막, x) 마다 [x, x+w) 의 평균 − [x−w, x) 의 평균. 못 재는 곳은 NaN."""
    S = img.shape[0]
    nb = S // BLOCK
    a = (img * ok)[:nb * BLOCK].reshape(nb, BLOCK, S).sum(1)
    c = ok[:nb * BLOCK].reshape(nb, BLOCK, S).sum(1).astype(np.float64)
    ca = np.concatenate([np.zeros((nb, 1)), np.cumsum(a, 1)], 1)
    cc = np.concatenate([np.zeros((nb, 1)), np.cumsum(c, 1)], 1)
    out = np.full((nb, S), np.nan)
    x = np.arange(w, S - w)
    ls, lc = ca[:, x] - ca[:, x - w], cc[:, x] - cc[:, x - w]
    rs, rc = ca[:, x + w] - ca[:, x], cc[:, x + w] - cc[:, x]
    good = (lc >= MIN_VALID * w * BLOCK) & (rc >= MIN_VALID * w * BLOCK)
    with np.errstate(invalid="ignore", divide="ignore"):
        out[:, w:S - w] = np.where(good, rs / rc - ls / lc, np.nan)
    return out


def _seam_cells(own: np.ndarray, w: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """(토막, x) 마다: 창 사이의 이음매인가 / 진짜 가장자리와의 이음매인가 / 대조군 (가까이에 이음매가 없는 자리) 인가."""
    S = own.shape[0]
    nb = S // BLOCK
    ch = np.zeros((S, S), dtype=bool)
    ch[:, 1:] = own[:, 1:] != own[:, :-1]                       # x 와 x−1 을 다른 창이 그렸다
    fr = np.zeros((S, S), dtype=bool)
    fr[:, 1:] = ch[:, 1:] & ((own[:, 1:] < 0) | (own[:, :-1] < 0))
    frac = lambda m: m[:nb * BLOCK].reshape(nb, BLOCK, S).mean(1)
    f_all, f_fr = frac(ch), frac(fr)
    win_seam = (f_all >= 0.9) & (f_fr < 0.1)
    frame_seam = f_fr >= 0.9
    near = np.zeros((nb, S), dtype=bool)                        # 이음매에서 2w 안
    any_seam = f_all > 0
    for d in range(-2 * w, 2 * w + 1):
        near |= np.roll(any_seam, d, axis=1)
    inner = np.zeros((nb, S), dtype=bool)
    E = int((own[S // 2] < 0).argmin())                         # 가장자리 폭
    inner[:, E + w:S - E - w] = True
    rows_in = (own[:nb * BLOCK].reshape(nb, BLOCK, S)[:, :, S // 2] >= 0).all(1)
    control = ~near & inner & rows_in[:, None]
    return win_seam, frame_seam, control


def _edges(win_seam: np.ndarray, control: np.ndarray) -> tuple[list, list]:
    """창의 한 변들 — (x, 이어진 토막들) 의 목록과, 거기서 CONTROL_SHIFT 만큼 떨어진 대조선의 목록."""
    S = win_seam.shape[1]
    edges, ctrl = [], []
    for x in np.unique(np.nonzero(win_seam)[1]):
        rows = np.nonzero(win_seam[:, x])[0]
        for seg in np.split(rows, np.nonzero(np.diff(rows) > 1)[0] + 1):
            if len(seg) < 3:
                continue
            edges.append((int(x), seg))
            for dx in (CONTROL_SHIFT, -CONTROL_SHIFT):
                if 0 <= x + dx < S and bool(control[seg, x + dx].all()):
                    ctrl.append((int(x + dx), seg))
    return edges, ctrl


def _edge_means(st: np.ndarray, lines: list) -> np.ndarray:
    out = []
    for x, seg in lines:
        v = st[seg, x]
        v = v[np.isfinite(v)]
        if len(v) >= 3:
            out.append(float(v.mean()))
    return np.array(out, dtype=np.float64)


def _block_means(img: np.ndarray, ok: np.ndarray, own: np.ndarray):
    """BLOCK² 칸의 분말 평균 밝기와 그 칸을 그린 창 (칸 전체를 한 창이 그린 칸만)."""
    S = img.shape[0]
    nb = S // BLOCK
    r = lambda a: a[:nb * BLOCK, :nb * BLOCK].reshape(nb, BLOCK, nb, BLOCK)
    cnt = r(ok.astype(np.float64)).sum((1, 3))
    with np.errstate(invalid="ignore", divide="ignore"):
        mean = r(img * ok).sum((1, 3)) / cnt
    o = r(own)
    whole = (o.min((1, 3)) == o.max((1, 3))) & (o.min((1, 3)) >= 0)
    good = whole & (cnt >= MIN_VALID * BLOCK * BLOCK)
    return mean[good], o[:, 0, :, 0][good]


def _between(mean: np.ndarray, grp: np.ndarray) -> tuple[float, float]:
    """(칸 사이의 표준편차, 창 평균 사이의 표준편차) — 칸이 3개 이상인 창만."""
    if len(mean) < 6:
        return float("nan"), float("nan")
    gm = [mean[grp == g].mean() for g in np.unique(grp) if (grp == g).sum() >= 3]
    return float(mean.std()), (float(np.std(gm)) if len(gm) >= 3 else float("nan"))


class Geometry:
    """창의 배치에서 나오는 것 — 레이어마다 같으므로 한 번만 만든다."""

    def __init__(self, own: np.ndarray):
        self.own = own
        self.cells = {(ax, w): _seam_cells(own if ax == "v" else own.T, w) for ax in AXES for w in WIDTHS}
        self.edges = {ax: _edges(self.cells[(ax, EDGE_W)][0], self.cells[(ax, EDGE_W)][2]) for ax in AXES}


def measure_layer(gen: np.ndarray, real: np.ndarray, powder: np.ndarray, geo: Geometry) -> dict:
    """한 채널 (S, S) 의 측정값. powder = 분말로 그리라고 한 화소 (가장자리에서는 CAD 의 분말 자리)."""
    out: dict = {}
    g, r, ok = gen.astype(np.float64), real.astype(np.float64), powder.astype(np.float64)
    for ax in AXES:
        tr = (lambda a: a) if ax == "v" else (lambda a: a.T)
        for w in WIDTHS:
            win_seam, frame_seam, control = geo.cells[(ax, w)]
            for name, img in (("gen", g), ("real", r)):
                st = _steps(tr(img), tr(ok), w)
                for kind, cells in (("seam", win_seam), ("frame", frame_seam), ("control", control)):
                    v = st[cells]
                    out[(ax, f"seg_w{w}", name, kind)] = v[np.isfinite(v)]
                if w == EDGE_W:
                    edges, ctrl = geo.edges[ax]
                    out[(ax, "edge", name, "seam")] = _edge_means(st, edges)
                    out[(ax, "edge", name, "control")] = _edge_means(st, ctrl)
    bm_g, grp = _block_means(g, ok, geo.own)
    bm_r, _ = _block_means(r, ok, geo.own)
    out["blocks"] = {"gen": _between(bm_g, grp), "real": _between(bm_r, grp)}
    return out


def _rms(parts: list) -> tuple[float, int]:
    v = np.concatenate(parts) if parts else np.array([])
    return (float(np.sqrt(np.mean(np.square(v)))), int(len(v))) if len(v) else (float("nan"), 0)


def _ratio(a: float, b: float) -> float:
    return a / b if (b and np.isfinite(a) and np.isfinite(b)) else float("nan")


def summarize(per_layer: list[dict]) -> dict:
    """레이어마다의 측정값 (measure_layer 의 결과 목록, 한 채널) → 요약."""
    doc: dict = {}
    for ax in AXES:
        d: dict = {}
        for stat in [f"seg_w{w}" for w in WIDTHS] + ["edge"]:
            e: dict = {}
            for name in ("gen", "real"):
                for kind in ("seam", "control") + (("frame",) if stat != "edge" else ()):
                    e[f"{name}_{kind}"], e[f"n_{name}_{kind}"] = _rms([m[(ax, stat, name, kind)] for m in per_layer])
            e["seam_gen_over_real"] = _ratio(e["gen_seam"], e["real_seam"])
            e["seam_gen_over_control"] = _ratio(e["gen_seam"], e["gen_control"])
            if stat != "edge":
                e["frame_gen_over_real"] = _ratio(e["gen_frame"], e["real_frame"])
            d[stat] = e
        doc[ax] = d
    b = {}
    for name in ("gen", "real"):
        a = np.array([m["blocks"][name] for m in per_layer], dtype=np.float64).reshape(-1, 2)
        b[name] = {"cell_std": float(np.nanmean(a[:, 0])) if len(a) else float("nan"),
                   "between_window_std": float(np.nanmean(a[:, 1])) if len(a) else float("nan")}
    b["between_window_gen_over_real"] = _ratio(b["gen"]["between_window_std"], b["real"]["between_window_std"])
    doc["blocks"] = b
    return doc


def main():
    ap = argparse.ArgumentParser(description="GenAug K4 — 창 이음매")
    ap.add_argument("--set", default=cfg.SET_NAME)
    ap.add_argument("--max-layers", type=int, default=0, help="재는 장 수 (0 = 전부)")
    args = ap.parse_args()
    log = setup_logger(0, "genaug.seam")
    root = resolve_set(args.set)
    man = read_manifest(root)
    sp = man["sampler"]
    S = man["img_size"]
    E = frame_px(S)
    geo = Geometry(owner_map(S, sp["stride"], sp["order"], sp["win"]))
    canvas = CanvasPool(tuple(man["builds"]))
    layers = pick_layers(man, args.max_layers)
    per = {0: [], 1: []}
    inner = np.zeros((S, S), dtype=bool)
    inner[E:S - E, E:S - E] = True
    for m in layers:
        g0, g1, ann = read_layer(root, m["id"])
        r0, r1, part = canvas.crop(m["build"], m["row"], 0, 0, S, S)
        # 가장자리의 진짜 화소는 라벨이 -1 이다 — 가장자리 이음매를 재려면 그쪽의 분말 자리가 필요하다: CAD 로 채운다
        powder = np.where(inner, ann == cfg.POWDER, ~part)
        for ch, (g, r) in enumerate(((g0, r0), (g1, r1))):
            per[ch].append(measure_layer(g, r, powder, geo))
    doc = {"set": man["name"], "n_layers": len(layers), "block_px": BLOCK, "order": sp["order"], "stride": sp["stride"],
           "unit": "gray levels (0-255), RMS",
           "note": "v = 세로 이음매 (좌우 창 사이), h = 가로 이음매. seg_w* = 32 px 토막마다의 계단, edge = 창의 한 변에 걸친 평균 계단 (w 8). "
                   "seam = 창 사이, frame = 진짜 가장자리와 그린 안쪽 사이, control = 이음매가 아닌 자리. real = 같은 레이어의 진짜 사진의 같은 자리",
           "visible0": summarize(per[0]), "visible1": summarize(per[1])}
    worst = max((doc[ch][ax]["edge"]["seam_gen_over_real"], ch, ax) for ch in ("visible0", "visible1") for ax in AXES)
    doc["K4"] = {"criterion": "edge step (generated) <= 2 x the real photo at the same place — worst of 2 photos x 2 directions",
                 "worst_ratio": worst[0], "worst_at": f"{worst[1]} {worst[2]}", "pass": bool(worst[0] <= 2.0)}
    out = cfg.set_fig_dir(man["name"])
    out.mkdir(parents=True, exist_ok=True)
    (out / "seam.json").write_text(json.dumps(doc, indent=1, ensure_ascii=False), encoding="utf-8")
    log.info(f"K4 창 이음매 — 세트 {man['name']} 의 {len(layers)} 장 ({sp['order']}, 간격 {sp['stride']}). 단위는 밝기 단계 (0–255)")
    for ch in ("visible0", "visible1"):
        for ax, title in AXES.items():
            e, s8 = doc[ch][ax]["edge"], doc[ch][ax]["seg_w8"]
            log.info(f"  {ch} {title}: 한 변의 계단 {e['gen_seam']:.2f} — 진짜 사진의 같은 자리 {e['real_seam']:.2f} ({e['seam_gen_over_real']:.2f}배), "
                     f"그림 안의 대조선 {e['gen_control']:.2f} ({e['seam_gen_over_control']:.2f}배).  토막의 계단 {s8['gen_seam']:.2f} (진짜 {s8['real_seam']:.2f}), "
                     f"가장자리 이음매 {s8['gen_frame']:.2f} (진짜 {s8['real_frame']:.2f})")
        b = doc[ch]["blocks"]
        log.info(f"  {ch} 창 평균 밝기의 흩어짐: 그린 것 {b['gen']['between_window_std']:.2f}, 진짜 {b['real']['between_window_std']:.2f} "
                 f"→ {b['between_window_gen_over_real']:.2f}배")
    k = doc["K4"]
    log.info(f"  K4: 가장 나쁜 곳 {k['worst_at']} 에서 진짜의 {k['worst_ratio']:.2f}배 → {'통과' if k['pass'] else '탈락'} (기준 2배 이하, 제안값)")
    log.info(f"→ {out / 'seam.json'}")


if __name__ == "__main__":
    main()
