"""'확실히 정상인 분말' 을 국소 통계 규칙으로 고를 수 있는가 — 사람 GT 로 본 것 (결론: 넓은 결함에서 실패한다).

배경 사진의 부품 밖 화소에 Powder 라벨을 주려면, 진짜 분말 쪽 결함 (RH · RS · IS · Debris · Spatter) 이 섞이지 않아야 한다.
여기서는 covered_rule 과 같은 방식 (주변 분말의 평균 · 표준편차에서 tau σ 안쪽이면 정상) 을 두 사진 모두에 걸어,
  - GT 가 분말 쪽 결함인 화소 가운데 "정상 분말" 로 판정된 비율 (작아야 한다)
  - GT 가 Powder 인 화소 가운데 "정상 분말" 로 판정된 비율
을 소스별로 센다. 부품 자리는 GT 에서 만든다 (Printed ∪ Swelling ∪ SE ∪ OM ∪ UM).

실행 (CWD = 3DP_VPPM/, DefSeg_AM/venv):
    python -m DefSeg_AM.scripts.synth_prototype.normal_powder_rule [tau=3.0] [win=61] [grow=3]
"""
import sys
import numpy as np
from scipy.ndimage import binary_dilation, uniform_filter

from DefSeg_AM.v5 import config_v5 as cfg
from DefSeg_AM.v5.data.data_gt_v5 import GTPool, enumerate_samples_v5

TAU = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
WIN = int(sys.argv[2]) if len(sys.argv) > 2 else 61
GROW = int(sys.argv[3]) if len(sys.argv) > 3 else 3


def _z(img: np.ndarray, powder: np.ndarray, win: int) -> tuple[np.ndarray, np.ndarray]:
    x = img.astype(np.float32); m = powder.astype(np.float32)
    cnt = uniform_filter(m, win, mode="nearest")
    mu = uniform_filter(x * m, win, mode="nearest") / np.maximum(cnt, 1e-3)
    var = uniform_filter(x * x * m, win, mode="nearest") / np.maximum(cnt, 1e-3) - mu * mu
    return np.abs(x - mu) / np.maximum(np.sqrt(np.maximum(var, 0.0)), 2.0), cnt > 0.05


def normal_powder_mask(i0, i1, foot, tau=TAU, win=WIN, grow=GROW) -> np.ndarray:
    powder = ~binary_dilation(foot, iterations=3)
    z0, ok0 = _z(i0, powder, win); z1, ok1 = _z(i1, powder, win)
    odd = binary_dilation(((z0 > tau) | (z1 > tau)) & powder, iterations=grow)
    return powder & ~odd & ok0 & ok1


if __name__ == "__main__":
    samples = enumerate_samples_v5()
    pool = GTPool(samples, cfg.IMG_SIZE)
    A = cfg.CLASS_ABBRS_V5
    POWDER_SIDE = (2, 3, 4, 6, 8)
    acc = {}
    for j, s in enumerate(samples):
        ann = pool.ann[j]
        foot = np.isin(ann, (1, 5, 7, 9, 10))
        nm = normal_powder_mask(pool.i0[j], pool.i1[j], foot)
        r = acc.setdefault(cfg.SOURCE_SHORT[s.source_name], {"n": 0, "nm": 0, **{c: [0, 0] for c in (0,) + POWDER_SIDE}})
        r["n"] += 1; r["nm"] += int(nm.sum())
        for c in (0,) + POWDER_SIDE:
            g = ann == c
            r[c][0] += int(g.sum()); r[c][1] += int((g & nm).sum())
    print(f"tau={TAU} win={WIN} grow={GROW}   각 칸 = 그 클래스의 GT 화소 가운데 '정상 분말' 로 판정된 비율")
    print(f"{'src':5s} {'layers':>6s} " + " ".join(f"{A[c]:>8s}" for c in (0,) + POWDER_SIDE) + "   판정된 화소 중 결함")
    tot = {"n": 0, "nm": 0, **{c: [0, 0] for c in (0,) + POWDER_SIDE}}
    def line(name, r):
        cells = " ".join((f"{100 * r[c][1] / r[c][0]:7.1f}%" if r[c][0] else "       –") for c in (0,) + POWDER_SIDE)
        bad = sum(r[c][1] for c in POWDER_SIDE)
        print(f"{name:5s} {r['n']:6d} {cells}   {100 * bad / max(r['nm'], 1):.2f}%")
    for name in ("L21", "17-4", "GP", "I1", "I2", "MS", "BJ21", "H13"):
        if name in acc:
            line(name, acc[name])
            tot["n"] += acc[name]["n"]; tot["nm"] += acc[name]["nm"]
            for c in (0,) + POWDER_SIDE:
                tot[c][0] += acc[name][c][0]; tot[c][1] += acc[name][c][1]
    line("all", tot)
