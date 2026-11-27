"""사진 묶음 (소스) 의 대표 밝기와 조명 지도를 구하고, 사진 한 장에 건다 (PLAN_v8 §1, §3).

**라벨을 받지 않는다.** 들어오는 것은 사진 (uint8) 뿐이다 — 사람 GT 가 아닌 사진 묶음 (새 장비 · 주석 없는 빌드) 에도 그대로 쓴다.
numpy 와 PIL 만 쓴다 (torch 없음).

    바꾼 사진 = 사진 ÷ 조명 지도(x, y)  +  (기준 밝기 − 소스의 대표 밝기)

쓰는 법:
    acc = SourceAccumulator("L21")
    for img0, img1 in 그_소스의_레이어들:      # (H, W) uint8 — 용융 직후, 도포 직후
        acc.add(img0, img1)
    st = acc.finish()                          # SourceNorm — 대표 밝기, 지도의 계수, 안전 장치의 결과
    out0 = apply_norm(img0, shift=st.shift, light=st.map_for(img0.shape) if st.use_map else None)

구하는 법 (PLAN_v8 §3):
  대표 밝기  소스의 사진 전부 (두 사진 모두) 의 화소 중앙값
  조명 지도  1) 도포 직후 사진을 GRID × GRID 칸으로 줄이고 (칸 안의 평균), 레이어마다 그 레이어의 칸들의 중앙값으로 나눈다
             2) 칸마다 레이어에 걸친 중앙값
             3) 2차식 면을 맞춘다. 면에서 많이 벗어난 칸을 빼고 다시 맞추기를 TRIM_ROUNDS 번
             4) 면을 평균 1 로 맞춘 것이 지도다. 사진의 화소 자리에서 같은 2차식을 계산해 나눈다 (칸 지도를 늘리지 않는다)
  안전 장치  S1 레이어 수 · S2 얼룩의 크기 · S3 면이 칸 지도를 설명하는 정도 — 셋을 모두 지나야 `use_map` 이 참이다
"""
from __future__ import annotations

from dataclasses import asdict, dataclass, field

import numpy as np
from PIL import Image

from .. import config_v8 as cfg

N_COEF = 6           # 2차식의 항: 1, y, y², x, xy, x²


# ---------------------------------------------------------------------------
# 칸 · 면
# ---------------------------------------------------------------------------

def cell_grid(img: np.ndarray, grid: int = cfg.GRID) -> np.ndarray:
    """(H, W) → (grid, grid) float32. 칸 안의 평균 (PIL 의 BOX)."""
    return np.asarray(Image.fromarray(np.asarray(img, dtype=np.float32), mode="F").resize((grid, grid), Image.BOX),
                      dtype=np.float32)


def _design(grid: int) -> np.ndarray:
    """(grid², 6) — 칸 번호 0 … grid−1 을 −1 … +1 로 옮긴 좌표에서의 2차식 항 (1, y, y², x, xy, x²)."""
    t = np.arange(grid, dtype=np.float64) / (grid - 1) * 2.0 - 1.0
    y, x = np.meshgrid(t, t, indexing="ij")
    return np.stack([c.ravel() for c in (np.ones_like(x), y, y * y, x, x * y, x * x)], axis=1)


def fit_surface(cell_map: np.ndarray, trim: bool = True) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """(G, G) 칸 지도 (NaN 가능) → (계수 (6,), 면 (G, G), 맞춤에 쓴 칸 (G, G) bool). 쓸 칸이 200 개 아래면 None.

    trim=True 면 잔차가 큰 칸을 빼며 TRIM_ROUNDS 번 다시 맞춘다 — 레이어마다 같은 자리에 있는 파트 같은 것이 면을 끌고 가지 않게.
    """
    g = cell_map.shape[0]
    a = _design(g)
    v = cell_map.astype(np.float64).ravel()
    finite = np.isfinite(v)
    used = finite
    coef = fit = None
    for k in range(cfg.TRIM_ROUNDS + 1 if trim else 1):
        if used.sum() < 200:
            return None
        coef, *_ = np.linalg.lstsq(a[used], v[used], rcond=None)
        fit = a @ coef
        if not trim or k == cfg.TRIM_ROUNDS:
            break
        res = v - fit
        s = 1.4826 * np.nanmedian(np.abs(res[used] - np.nanmedian(res[used])))
        used = finite & (np.abs(res) < cfg.TRIM_K * max(float(s), 1e-6))
    return coef, fit.reshape(g, g), used.reshape(g, g)


def r_squared(cell_map: np.ndarray, surface: np.ndarray) -> float:
    """면이 칸 지도의 변화를 설명하는 정도 — 1 − (남은 변화) ÷ (전체 변화). **모든 칸**에서 잰다 (맞춤에서 뺀 칸도 넣는다).

    매끄러운 조명이면 1 에 가깝고, 구역 · 줄무늬 · 파트의 무늬처럼 면이 그리지 못하는 것이 많으면 낮다.
    """
    m = np.isfinite(cell_map)
    v, f = cell_map[m].astype(np.float64), surface[m].astype(np.float64)
    tot = float(((v - v.mean()) ** 2).sum())
    return float("nan") if tot <= 0 else 1.0 - float(((v - f) ** 2).sum()) / tot


def surface_at(coef, hw: tuple[int, int], grid: int = cfg.GRID) -> np.ndarray:
    """2차식을 (H, W) 사진의 화소 자리에서 계산한다 → (H, W) float32. 칸 지도와 같은 좌표계다 (칸의 가운데 = 그 칸의 좌표)."""
    h, w = int(hw[0]), int(hw[1])
    c = np.asarray(coef, dtype=np.float64)
    # 화소 p 의 칸 좌표 = (p + 0.5) × grid ÷ 크기 − 0.5, 그것을 −1 … +1 로
    x = (((np.arange(w, dtype=np.float64) + 0.5) * grid / w - 0.5) / (grid - 1) * 2.0 - 1.0)[None, :]
    y = (((np.arange(h, dtype=np.float64) + 0.5) * grid / h - 0.5) / (grid - 1) * 2.0 - 1.0)[:, None]
    out = (c[0] + c[1] * y + c[2] * y * y) + (c[3] + c[4] * y) * x + c[5] * (x * x)
    return out.astype(np.float32)


# ---------------------------------------------------------------------------
# 소스 하나의 통계
# ---------------------------------------------------------------------------

@dataclass
class SourceNorm:
    """소스 하나에서 구한 값. `coef` 는 평균 1 로 맞춘 지도의 2차식 계수다 (지도를 못 구했으면 None)."""
    name: str
    n_layers: int
    brightness: float                    # 대표 밝기 — 두 사진 전체 화소의 중앙값
    shift: float                         # 기준 밝기 − 대표 밝기 (더하는 값)
    coef: list[float] | None = None
    map_sd_pct: float | None = None      # 얼룩의 크기 — 지도의 표준편차 (%)
    map_p2p_pct: float | None = None     # 지도의 최대 − 최소 (%)
    map_r2: float | None = None          # 면이 칸 지도의 변화를 설명하는 정도
    trimmed_pct: float | None = None     # 맞춤에서 뺀 칸의 비율 (%)
    gates: dict = field(default_factory=dict)       # {"S1_layers": bool, "S2_size": bool, "S3_smooth": bool}
    use_map: bool = False                # 안전 장치를 모두 지났는가
    grid: int = cfg.GRID

    def map_for(self, hw: tuple[int, int]) -> np.ndarray:
        """(H, W) 사진에 쓸 조명 지도 (평균 1). 안전 장치의 결과와 상관없이 계산한다 — 쓸지는 부르는 쪽이 `use_map` 을 보고 정한다."""
        if self.coef is None:
            raise ValueError(f"{self.name}: 조명 지도가 없다")
        m = surface_at(self.coef, hw, self.grid)
        lo, hi = float(m.min()), float(m.max())
        if lo < cfg.MAP_CLIP[0] or hi > cfg.MAP_CLIP[1]:
            raise ValueError(f"{self.name}: 조명 지도의 값이 {cfg.MAP_CLIP} 밖이다 (min={lo:.3f}, max={hi:.3f}) — 맞춤이 틀렸다")
        return m

    def to_json(self) -> dict:
        return asdict(self)

    @classmethod
    def from_json(cls, d: dict) -> "SourceNorm":
        return cls(**{k: d[k] for k in cls.__dataclass_fields__ if k in d})


def layer_cells(img1: np.ndarray, grid: int = cfg.GRID) -> np.ndarray:
    """도포 직후 사진 한 장 → 칸 지도 ÷ 그 레이어의 칸들의 중앙값 (G, G). 조명 지도의 재료다."""
    g = cell_grid(img1, grid)
    return g / max(float(np.median(g)), 1.0)


def pooled_cells(cells: list[np.ndarray]) -> np.ndarray:
    """레이어마다의 칸 지도 → 칸마다 레이어에 걸친 중앙값 (G, G)."""
    return np.median(np.stack(cells), axis=0)


def median_from_hist(hist: np.ndarray) -> float:
    """256 칸 히스토그램 → 중앙값 (정수 단계)."""
    cdf = np.cumsum(hist) / max(int(hist.sum()), 1)
    return float(np.searchsorted(cdf, 0.5))


def summarize_map(cell_map: np.ndarray, n_layers: int, trim: bool = True) -> dict:
    """칸 지도 → 지도의 계수와 크기 · 설명하는 정도 · 안전 장치의 결과 (dict). 면을 못 구하면 계수가 None 이다."""
    out = {"coef": None, "map_sd_pct": None, "map_p2p_pct": None, "map_r2": None, "trimmed_pct": None,
           "gates": {"S1_layers": bool(n_layers >= cfg.GATE_MIN_LAYERS), "S2_size": False, "S3_smooth": False},
           "use_map": False}
    res = fit_surface(cell_map, trim=trim)
    if res is None:
        return out
    coef, surf, used = res
    mean = float(surf.mean())
    if not np.isfinite(mean) or mean <= 0 or float(surf.min()) <= 0:
        return out
    f = surf / mean
    out["coef"] = [float(c) / mean for c in coef]
    out["map_sd_pct"] = float(f.std() * 100)
    out["map_p2p_pct"] = float((f.max() - f.min()) * 100)
    out["map_r2"] = r_squared(cell_map, surf)
    out["trimmed_pct"] = float(100.0 * (1.0 - used.sum() / max(int(np.isfinite(cell_map).sum()), 1)))
    out["gates"]["S2_size"] = bool(out["map_sd_pct"] >= cfg.GATE_MIN_SD_PCT)
    out["gates"]["S3_smooth"] = bool(out["map_r2"] == out["map_r2"] and out["map_r2"] >= cfg.GATE_MIN_R2)
    out["use_map"] = bool(all(out["gates"].values()))
    return out


class SourceAccumulator:
    """소스 하나의 사진을 한 장씩 받아 통계를 쌓는다 (사진을 들고 있지 않는다 — 히스토그램과 칸 지도만 남긴다)."""

    def __init__(self, name: str, ref: float = cfg.REF_BRIGHTNESS, grid: int = cfg.GRID):
        self.name, self.ref, self.grid = name, float(ref), int(grid)
        self.hist = np.zeros(256, dtype=np.int64)
        self.cells: list[np.ndarray] = []            # 레이어마다 — 도포 직후 사진의 칸 지도 ÷ 그 레이어의 칸들의 중앙값

    def add(self, img0: np.ndarray, img1: np.ndarray) -> None:
        """img0 = 용융 직후, img1 = 도포 직후. 둘 다 (H, W) uint8."""
        for a in (img0, img1):
            if a.dtype != np.uint8 or a.ndim != 2:
                raise ValueError(f"(H, W) uint8 사진이 아니다 (shape={a.shape}, dtype={a.dtype})")
            self.hist += np.bincount(a.ravel(), minlength=256)
        self.cells.append(layer_cells(img1, self.grid))

    def cell_map(self) -> np.ndarray:
        """칸마다 레이어에 걸친 중앙값 (G, G)."""
        return pooled_cells(self.cells)

    def finish(self) -> SourceNorm:
        if not self.cells:
            raise ValueError(f"{self.name}: 사진이 없다")
        b = median_from_hist(self.hist)
        st = SourceNorm(name=self.name, n_layers=len(self.cells), brightness=b, shift=self.ref - b, grid=self.grid)
        for k, v in summarize_map(self.cell_map(), len(self.cells)).items():
            setattr(st, k, v)
        return st


# ---------------------------------------------------------------------------
# 사진 한 장에 건다
# ---------------------------------------------------------------------------

def apply_norm(img: np.ndarray, shift: float = 0.0, light: np.ndarray | None = None,
               return_clipped: bool = False):
    """사진 ÷ 조명 지도 + shift → (H, W) uint8 (반올림, 0 – 255 로 자른다).

    shift 가 0 이고 지도가 없으면 들어온 화소를 그대로 돌려준다. return_clipped=True 면 (사진, 잘린 화소 수) 를 돌려준다.
    """
    if img.dtype != np.uint8 or img.ndim != 2:
        raise ValueError(f"(H, W) uint8 사진이 아니다 (shape={img.shape}, dtype={img.dtype})")
    if light is None and float(shift) == 0.0:
        return (img.copy(), 0) if return_clipped else img.copy()
    out = img.astype(np.float32)
    if light is not None:
        if light.shape != img.shape:
            raise ValueError(f"조명 지도의 크기가 사진과 다르다: {light.shape} vs {img.shape}")
        out /= light
    out += np.float32(shift)
    np.rint(out, out=out)
    n_clip = int(((out < 0) | (out > 255)).sum()) if return_clipped else 0
    np.clip(out, 0, 255, out=out)
    res = out.astype(np.uint8)
    return (res, n_clip) if return_clipped else res
