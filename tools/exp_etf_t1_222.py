#!/usr/bin/env python3
"""第222轮(下): 股票ETF 深低开 —— T+1 合规版回测。

上一版诊断暴露两件事:
  1. T+0 那四类 ETF 里, 跨境QDII 是"低开继续跌"(日内动量), 与可转债的反转相反,
     且一个 tick 就要 10~20bp, 无利可图。债券/货币/黄金 振幅 0.01~0.1%, 不够付成本。
     => 用 ETF 替代可转债做 T+0 这条路, 数据上否掉。
  2. 股票ETF(762只/111万行)的深低开反转极强: gap<-3% 时日内 +196bp/胜率70.6%。
     股票ETF 是 T+1, 但这个形状说明"恐慌错杀"在宽基上同样存在, 而且 ETF 无退市、
     无业绩爆雷、流动性远好于同市值个股。

本脚本只测合规口径(不碰 T+0):
  A. 开盘买 -> 当日收盘卖 (仅作形状参考, 股票ETF不可执行)
  B. 开盘买 -> 次日收盘卖 (T+1, 合规, 与现行个股策略同构)
  C. 开盘买 -> 次日开盘卖 (T+1 最短持有)

口径: 成本按 0.06% 单次往返(佣金万0.3双边 + 1个tick滑点), 并对低价ETF如实惩罚。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]


def load() -> pd.DataFrame:
    d = pd.read_parquet(ROOT / "data" / "etf" / "daily.parquet")
    d = d[d["cls"] == "stock"].copy()
    d = d.sort_values(["code", "date"]).reset_index(drop=True)
    g = d.groupby("code", observed=True, sort=False)
    d["pc"] = g["close"].shift(1)
    d["n_open"] = g["open"].shift(-1)
    d["n_close"] = g["close"].shift(-1)
    d["gap"] = d["open"] / d["pc"] - 1
    d["intraday"] = d["close"] / d["open"] - 1          # A: 开->收
    d["to_nd_close"] = d["n_close"] / d["open"] - 1     # B: 开->次收 (T+1)
    d["to_nd_open"] = d["n_open"] / d["open"] - 1       # C: 开->次开
    d["amt"] = d["amount"]
    d = d.dropna(subset=["pc", "gap"])
    d = d[(d["gap"].between(-0.15, 0.10)) & (d["amt"] > 0)]
    return d


def cost_of(price: float) -> float:
    """单次往返成本: 佣金万0.3双边=0.006%, 滑点按 1 tick 且不低于 0.02%。"""
    tick = 0.001 / price
    return 0.00006 + max(tick, 0.0002)


def agg(d: pd.DataFrame, col: str) -> pd.DataFrame:
    rows = []
    d = d.copy()
    d["cost"] = d["close"].map(cost_of)
    d["net"] = d[col] - d["cost"]
    for lo, hi, name in [(-0.15, -0.03, "跌超3%"), (-0.03, -0.02, "-3~-2%"),
                         (-0.02, -0.015, "-2~-1.5%"), (-0.015, -0.010, "-1.5~-1%"),
                         (-0.010, -0.007, "-1~-0.7%"), (-0.007, -0.004, "-0.7~-0.4%")]:
        s = d[(d["gap"] > lo) & (d["gap"] <= hi)]
        if len(s) < 30:
            continue
        rows.append({
            "档": name, "样本": len(s), "只数": s["code"].nunique(),
            "毛bp": round(s[col].mean() * 10000, 0),
            "成本bp": round(s["cost"].mean() * 10000, 1),
            "净bp": round(s["net"].mean() * 10000, 0),
            "胜率%": round((s["net"] > 0).mean() * 100, 1),
            "赔率": round(s.loc[s["net"] > 0, "net"].mean() /
                          abs(s.loc[s["net"] <= 0, "net"].mean()), 2)
            if (s["net"] <= 0).any() else np.nan,
        })
    return pd.DataFrame(rows)


def main() -> None:
    d = load()
    print(f"股票ETF {len(d):,} 行 / {d['code'].nunique()} 只 / {d['date'].min().date()} ~ {d['date'].max().date()}")
    print(f"中位成交额 {d['amt'].median()/1e8:.2f} 亿 / P25 {d['amt'].quantile(.25)/1e8:.2f} 亿")

    for col, title in [("intraday", "A 开盘->当日收盘 (股票ETF不可执行, 仅对比形状)"),
                       ("to_nd_close", "B 开盘->次日收盘 (T+1 合规)"),
                       ("to_nd_open", "C 开盘->次日开盘 (T+1 最短)")]:
        print("\n" + "=" * 92)
        print(title)
        print("=" * 92)
        print(agg(d, col).to_string(index=False))

    # 分层: 只看流动性好的
    print("\n" + "=" * 92)
    print("B 的流动性分层 (T+1 合规, 开->次收)")
    print("=" * 92)
    for q, lo in [("额>1亿", 1e8), ("额>5亿", 5e8), ("额>20亿", 2e9)]:
        s = d[(d["amt"] > lo) & (d["gap"] < -0.015)]
        if len(s) < 50:
            print(f"{q}: 样本不足 ({len(s)})")
            continue
        s = s.copy()
        s["cost"] = s["close"].map(cost_of)
        s["net"] = s["to_nd_close"] - s["cost"]
        print(f"{q}: n={len(s):5d}  净 {s['net'].mean()*10000:+.0f}bp  "
              f"胜率 {(s['net']>0).mean()*100:.1f}%  只数 {s['code'].nunique()}")

    # 年频: 一年有多少次机会
    print("\n" + "=" * 92)
    print("机会频率: 每年出现多少个 gap<-1.5% 的 ETF-日")
    print("=" * 92)
    d["year"] = d["date"].dt.year
    s = d[d["gap"] < -0.015].groupby("year").agg(次数=("code", "size"), 只数=("code", "nunique"))
    print(s.to_string())


if __name__ == "__main__":
    main()
