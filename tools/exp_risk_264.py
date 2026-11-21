"""第264轮: 单吊策略的仓位风险预算 —— 该拿多少钱跑

样本: output/backtest_long.json (3000 日, 2014-07 ~ 2026-11, 满仓单吊净额日度曲线)
问题: 保持单吊不变, 把总资金的比例 f 投入策略, 其余 1-f 留现金, 收益与回撤如何权衡。
方法: 用真实日度收益路径(不 bootstrap), 对 f∈[0.1,1.0] 复算 CAGR / 最大回撤 / 最长水下 /
      最差自然年 / 亏损年数; 并用日度收益算 Kelly 与分数 Kelly 作为理论锚。
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def longest_underwater(nav: np.ndarray) -> int:
    peak = np.maximum.accumulate(nav)
    below = nav < peak
    longest = cur = 0
    for b in below:
        cur = cur + 1 if b else 0
        longest = max(longest, cur)
    return int(longest)


def main() -> None:
    raw = json.load(open(ROOT / "output/backtest_long.json"))
    eq = raw["equity"]
    ds = pd.to_datetime([x["d"] for x in eq])
    nav = np.array([x["nav"] for x in eq], dtype=float)

    r = np.diff(nav) / nav[:-1]
    rd = pd.Series(r, index=ds[1:])
    n = len(r)
    ann = 252

    def stats(f: float, rf: float) -> dict:
        rp = f * r + (1 - f) * rf
        navp = np.concatenate([[1.0], np.cumprod(1 + rp)])
        years = pd.Series(rp, index=rd.index).resample("YE").apply(
            lambda x: float(np.prod(1 + x) - 1))
        years = years.dropna()
        cagr = navp[-1] ** (ann / n) - 1
        dd = float((navp / np.maximum.accumulate(navp) - 1).min())
        return dict(f=round(f, 2), rf=rf, cagr_pct=round(cagr * 100, 2),
                    maxdd_pct=round(dd * 100, 2),
                    underwater_days=longest_underwater(navp),
                    worst_year_pct=round(float(years.min()) * 100, 2),
                    neg_years=int((years < 0).sum()),
                    total_years=int(len(years)))

    out = dict(kelly={}, grid_rf0=[], grid_rf2=[])
    for f in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        out["grid_rf0"].append(stats(f, 0.0))
    for f in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1.0]:
        out["grid_rf2"].append(stats(f, 0.02 / ann))

    m, v = float(r.mean()), float(r.var(ddof=1))
    kf = m / v
    out["kelly"] = dict(daily_mean_pct=round(m * 100, 4),
                        daily_std_pct=round(float(r.std(ddof=1)) * 100, 3),
                        full_kelly=round(kf, 2), half_kelly=round(kf / 2, 2),
                        quarter_kelly=round(kf / 4, 2))
    out["full_strategy"] = stats(1.0, 0.0)

    json.dump(out, open(ROOT / "output/exp_risk_264.json", "w"),
              ensure_ascii=False, indent=2)
    print(json.dumps(out, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    main()
