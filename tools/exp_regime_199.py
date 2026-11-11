# -*- coding: utf-8 -*-
"""第199轮 · 诊断: 主策略单笔收益 vs 买入日的市场环境

思路
----
主策略已有三道闸门(市场均线 market_ma、宽度 breadth_min、knife_tiers),
但它们都是"做/不做"的离散开关。本轮问的是另一个问题:
**同样的开仓条件, 在不同的市场波动环境里, 单笔期望是否不同?**

如果不同, 说明闸门漏掉了一个连续维度; 如果相同, 说明"环境分层"这条路也平。

口径
----
- 每笔用 **买入日前一交易日**(= 信号日 T 收盘) 的市场指标, 无未来函数。
  实盘 T 日 15:00 就能算出这些数, T+1 开盘前可决策。
- 逐笔 ret 已含双边成本(来自回测引擎), 直接复利即可, 不需要重跑回测。
- 市场指标: 等权日收益(剔除停牌) / 横截面离散度 / 上涨家数占比 / 20 日波动率
  及其 250 日滚动分位。
"""
from __future__ import annotations

import glob
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def market_series() -> pd.DataFrame:
    cache = ROOT / "output" / "exp_mkt_199.csv"
    if cache.exists():
        return pd.read_csv(cache, dtype={"d": str}).set_index("d")
    rows = []
    for f in sorted(glob.glob(str(ROOT / "data" / "panel" / "*.csv.gz"))):
        d = Path(f).name.split(".")[0]
        try:
            x = pd.read_csv(f, usecols=["pct", "is_paused", "is_st", "amount"])
        except Exception:
            continue
        x = x[(x.is_paused == 0) & x.pct.notna()]
        if len(x) < 200:
            continue
        rows.append({"d": d, "ret": float(x.pct.mean()), "xsd": float(x.pct.std()),
                     "breadth": float((x.pct > 0).mean() * 100), "n": int(len(x)),
                     "amt": float(x.amount.sum())})
    m = pd.DataFrame(rows).set_index("d").sort_index()
    m["vol20"] = m.ret.rolling(20).std() * np.sqrt(252)
    m["vol_pct"] = m.vol20.rolling(250, min_periods=120).rank(pct=True)
    m["amt_yoy"] = m.amt / m.amt.rolling(250, min_periods=120).mean() - 1
    m["ret20"] = (1 + m.ret / 100).rolling(20).apply(np.prod, raw=True) - 1
    m.to_csv(cache)
    return m


def load_trades(days: int) -> pd.DataFrame:
    p = ROOT / "output" / f"exp_factor_197_trades_{days}.csv"
    t = pd.read_csv(p, dtype={"buy": str, "sell": str})
    t["buy"] = t["buy"].str.replace("-", "")
    return t


def perf(rets: np.ndarray, holds: int) -> dict:
    """满仓单吊逐笔复利; 笔间空档不计息(与 197 口径一致, 差异 <0.3pp)。"""
    nav = float(np.prod(1 + rets / 100))
    yrs = holds / 252
    cagr = nav ** (1 / yrs) - 1
    return {"倍数": round(nav, 2), "年化%": round(cagr * 100, 1)}


def main():
    m = market_series()
    print(f"市场序列 {m.index[0]} ~ {m.index[-1]}, {len(m)} 日")
    out = {}
    for days in (1500, 3000):
        t = load_trades(days)
        # 信号日 = 买入日的前一交易日
        idx = list(m.index)
        pos = {d: i for i, d in enumerate(idx)}
        sig = []
        for b in t.buy:
            if b in pos and pos[b] > 0:
                sig.append(idx[pos[b] - 1])
            else:
                sig.append(None)
        t["sig"] = sig
        t = t[t.sig.notna()].copy()
        f = m.reindex(t.sig).reset_index(drop=True)
        t["vol20"] = f.vol20.values
        t["vol_pct"] = f.vol_pct.values
        t["xsd"] = f.xsd.values
        t["breadth"] = f.breadth.values
        t["ret20"] = f.ret20.values
        t = t[t.vol_pct.notna()].copy()
        n = len(t)
        base = float(t.ret.mean())
        se = float(t.ret.std() / np.sqrt(n))
        print(f"\n===== {days} 日窗口: {n} 笔, 均值 {base:.3f}%, 标准误 {se:.3f}pp =====")
        blocks = {}
        for col, label in [("vol_pct", "波动率分位"), ("xsd", "横截面离散度"),
                           ("breadth", "上涨家数%"), ("ret20", "市场20日动量")]:
            q = pd.qcut(t[col], 4, labels=["Q1低", "Q2", "Q3", "Q4高"], duplicates="drop")
            g = t.groupby(q, observed=False).ret.agg(["count", "mean", "median",
                                                      lambda s: (s > 0).mean() * 100])
            g.columns = ["笔数", "均值%", "中位%", "胜率%"]
            print(f"\n-- 按{label}四分位 --")
            print(g.round(2).to_string())
            blocks[label] = g.round(2).to_dict("index")
        out[days] = {"n": n, "mean": round(base, 3), "se": round(se, 3), "blocks": blocks}
    (ROOT / "output" / "exp_regime_199.json").write_text(
        json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
