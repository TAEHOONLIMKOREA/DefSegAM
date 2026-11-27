"""'덮임 판정' 규칙 — 도포 직후 사진에서 부품 자리가 주변 분말과 구별되지 않으면 덮인 것으로 본다.

용도: ORNL 캔버스에서 합성하지 않은 부품 화소 가운데 "확실히 덮인" 곳만 Printed (= SE 아님) 로 표시한다.
      드러나 보이는 곳은 실제 SE · Swelling 일 수 있으므로 IGNORE 로 둔다.

검증: 사람 GT 125 레이어에서
  - GT 가 SE / Swelling 인 화소 가운데 "덮임" 으로 잘못 판정된 비율 (작아야 한다)
  - GT 가 Printed 인 화소 가운데 "덮임" 으로 판정된 비율 (크면 negative 를 많이 얻는다)
부품 자리는 GT 에서 만든다: Printed ∪ Swelling ∪ SE ∪ OM ∪ UM (CAD 마스크의 대역).

실행 (CWD = 3DP_VPPM/, DefSeg_AM/venv):
    python -m DefSeg_AM.scripts.synth_prototype.covered_rule [tau=3.0] [win=61] [grow=3]
"""
import sys
import numpy as np
from scipy.ndimage import binary_dilation, binary_erosion, uniform_filter

from DefSeg_AM.v5 import config_v5 as cfg
from DefSeg_AM.v5.data.data_gt_v5 import GTPool, enumerate_samples_v5

TAU = float(sys.argv[1]) if len(sys.argv) > 1 else 3.0
WIN = int(sys.argv[2]) if len(sys.argv) > 2 else 61
GROW = int(sys.argv[3]) if len(sys.argv) > 3 else 3


def covered_mask(i1: np.ndarray, foot: np.ndarray, tau: float = TAU, win: int = WIN, grow: int = GROW) -> np.ndarray:
    """부품 자리 foot 가운데 '확실히 덮인' 화소. i1 = visible/1 (uint8)."""
    x = i1.astype(np.float32)
    powder = ~binary_dilation(foot, iterations=3)                 # 부품 가장자리를 피한 분말 화소
    m = powder.astype(np.float32)
    cnt = uniform_filter(m, win, mode="nearest")
    mu = uniform_filter(x * m, win, mode="nearest") / np.maximum(cnt, 1e-3)
    var = uniform_filter(x * x * m, win, mode="nearest") / np.maximum(cnt, 1e-3) - mu * mu
    sd = np.sqrt(np.maximum(var, 0.0))
    ok = cnt > 0.05                                               # 주변에 분말이 충분히 있는 곳만 판정
    z = np.abs(x - mu) / np.maximum(sd, 2.0)
    exposed = (z > tau) & foot
    exposed = binary_dilation(exposed, iterations=grow)
    return binary_erosion(foot, iterations=2) & ~exposed & ok


if __name__ == "__main__":
    samples = enumerate_samples_v5()
    pool = GTPool(samples, cfg.IMG_SIZE)
    PART = (1, 5, 7, 9, 10)
    rows = {}
    for j, s in enumerate(samples):
        ann = pool.ann[j]
        foot = np.isin(ann, PART)
        if foot.sum() < 500:
            continue
        cov = covered_mask(pool.i1[j], foot)
        r = rows.setdefault(cfg.SOURCE_SHORT[s.source_name], dict(n=0, prt=0, prt_cov=0, se=0, se_cov=0, sw=0, sw_cov=0, om=0, om_cov=0, cov=0))
        r["n"] += 1
        r["prt"] += int((ann == 1).sum()); r["prt_cov"] += int(((ann == 1) & cov).sum())
        r["se"] += int((ann == 7).sum()); r["se_cov"] += int(((ann == 7) & cov).sum())
        r["sw"] += int((ann == 5).sum()); r["sw_cov"] += int(((ann == 5) & cov).sum())
        r["om"] += int(np.isin(ann, (9, 10)).sum()); r["om_cov"] += int((np.isin(ann, (9, 10)) & cov).sum())
        r["cov"] += int(cov.sum())
    print(f"tau={TAU} win={WIN} grow={GROW}")
    print(f"{'src':5s} {'layers':>6s} {'Printed px':>11s} {'→덮임':>7s} | {'SE px':>8s} {'→덮임':>7s} | {'Sw px':>7s} {'→덮임':>7s} | {'OM·UM':>6s} {'→덮임':>7s} | 덮임 화소 중 SE·Sw·OM·UM")
    tot = dict(n=0, prt=0, prt_cov=0, se=0, se_cov=0, sw=0, sw_cov=0, om=0, om_cov=0, cov=0)
    def line(name, r):
        f = lambda a, b: f"{100 * a / b:6.1f}%" if b else "     –"
        bad = r["se_cov"] + r["sw_cov"] + r["om_cov"]
        print(f"{name:5s} {r['n']:6d} {r['prt']:11d} {f(r['prt_cov'], r['prt'])} | {r['se']:8d} {f(r['se_cov'], r['se'])} | {r['sw']:7d} {f(r['sw_cov'], r['sw'])} | "
              f"{r['om']:6d} {f(r['om_cov'], r['om'])} | {100 * bad / max(r['cov'], 1):.2f}%")
    for name in ("L21", "17-4", "GP", "I1", "I2", "MS", "BJ21", "H13"):
        if name in rows:
            line(name, rows[name])
            for k in tot:
                tot[k] += rows[name][k]
    line("all", tot)
