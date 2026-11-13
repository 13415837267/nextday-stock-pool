"""第 220 轮稳健性检验: 622% 里有多少能兑现?

无条件测试的四个追问:
  1. 成本: 0.3% 是理想值。这批票 T 日成交额中位只有 1.0 亿, 实盘滑点远不止千一。
  2. 流动性: 只买成交额 > 2 亿的票, 还剩多少?
  3. 涨停依赖: 剔除 T 日涨停(pct>=9.5)的候选 —— 若是靠"追涨停", 这就是第 213 轮
     已经证伪过的右侧交易, 需要对上账。
  4. 时间衰减: 2026 年单独看。
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"


def sim(p: pd.DataFrame, cost_extra: float = 0.0, amt_min: float = 0.0,
        excl_limit: bool = False, years: tuple | None = None) -> dict:
    d = p
    if years:
        d = d[(d.year >= years[0]) & (d.year <= years[1])]
    if amt_min > 0:
        d = d[d["amt_yi"] >= amt_min]
    if excl_limit:
        d = d[d["pct"] < 9.5]
    if not len(d):
        return {}
    day_dates = sorted(d["trade_date"].unique())
    day_index = {x: i for i, x in enumerate(day_dates)}
    nav, peak, mdd, held, rets = 1.0, 1.0, 0.0, None, []
    for dt, dd in d.groupby("trade_date", sort=True):
        if held is not None and dt <= held:
            continue
        top = dd.sort_values("pred", ascending=False).iloc[0]
        r = (float(top["ret"]) - cost_extra) / 100
        nav *= (1 + r)
        peak = max(peak, nav)
        mdd = min(mdd, nav / peak - 1)
        rets.append(r * 100)
        held = day_dates[min(day_index[dt] + 2, len(day_dates) - 1)]
    d0, d1 = pd.Timestamp(str(int(day_dates[0]))), pd.Timestamp(str(int(day_dates[-1])))
    yrs = max(0.3, (d1 - d0).days / 365.25)
    return {"笔数": len(rets), "年化%": round((nav ** (1 / yrs) - 1) * 100, 1),
            "总倍": round(nav, 1), "回撤%": round(mdd * 100, 1),
            "单笔%": round(float(np.mean(rets)), 3),
            "胜率%": round(float(np.mean([x > 0 for x in rets]) * 100), 1)}


def main():
    p = pd.read_parquet(OUT / "ml_220_pred.parquet")
    f = pd.read_parquet(OUT / "ml_220.parquet",
                        columns=["trade_date", "ts_code", "pct", "amt_yi"])
    p = p.merge(f, on=["trade_date", "ts_code"], how="left")
    p["year"] = p["trade_date"] // 10000

    tests = {
        "基线(成本0.30%)": dict(),
        "成本 1.00%": dict(cost_extra=0.70),
        "成本 1.50%": dict(cost_extra=1.20),
        "成本 2.00%": dict(cost_extra=1.70),
        "只买成交额>2亿": dict(amt_min=2.0),
        "剔除T日涨停": dict(excl_limit=True),
        "只买>2亿且剔涨停": dict(amt_min=2.0, excl_limit=True),
        "仅2026年": dict(years=(2026, 2026)),
        "仅2016-2019": dict(years=(2016, 2019)),
        "仅2020-2022": dict(years=(2020, 2022)),
    }
    out = {}
    print(f"{'检验':<20}{'笔数':>6}{'年化%':>9}{'总倍':>12}{'回撤%':>8}{'单笔%':>8}{'胜率%':>8}")
    for k, kw in tests.items():
        r = sim(p, **kw)
        out[k] = r
        if r:
            print(f"{k:<20}{r['笔数']:>6}{r['年化%']:>9.1f}{r['总倍']:>12.1f}"
                  f"{r['回撤%']:>8.1f}{r['单笔%']:>8.3f}{r['胜率%']:>8.1f}")
    (OUT / "robust_ml_220.json").write_text(json.dumps(out, ensure_ascii=False, indent=1),
                                            encoding="utf-8")


if __name__ == "__main__":
    main()
