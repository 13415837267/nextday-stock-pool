# -*- coding: utf-8 -*-
"""第 212 轮 · 左侧"深度超跌反转"策略轻量回测(满仓单吊、T+1买T+2卖)。

用 panel_212 直接模拟(与正式引擎同口径: T+1开盘买、T+2收盘卖、扣0.30%成本):
每天从"深跌股"候选池里按评分挑 1 只, 复利滚动, 无票则空仓。
对比多种评分与条件, 看哪条路能逼近年化200%。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"


def backtest(p: pd.DataFrame, cond, score_col, ascending, name: str) -> dict:
    """满仓单吊复利: 每天按条件选股, 按 score 排序取1只, 当日 ret12_net 计入复利。"""
    sub = p[cond].copy()
    sub = sub.sort_values(["d", score_col], ascending=[True, ascending], kind="mergesort")
    pick = sub.groupby("d").head(1)
    r = pick.groupby("d")["ret12_net"].first()  # 每天1只
    nav = (1 + r / 100).prod()
    days = p["d"].nunique()
    trades = len(r)
    years = days / 242
    cagr = nav ** (1 / years) - 1 if nav > 0 else -1
    # 回撤
    curve = (1 + r / 100).cumprod()
    dd = (curve / curve.cummax() - 1).min()
    return {"name": name, "trades": trades, "days": days,
            "mult": round(float(nav), 1), "cagr%": round(float(cagr) * 100, 1),
            "per_trade%": round(float(r.mean()), 3), "win%": round(float((r > 0).mean()) * 100, 1),
            "maxdd%": round(float(dd) * 100, 1),
            "kama": round(float(cagr) / abs(float(dd)) if dd < 0 else np.inf, 2)}


def main() -> None:
    p = pd.read_parquet(OUT / "panel_212.parquet")
    p["r"] = p["ret12_net"].clip(-21, 21)
    p["year"] = p["d"].str[:4].astype(int)

    results = []
    # 各种"深跌"条件 × 评分
    variants = [
        ("深跌<-55 · 选最深", p["dist_ph250"] < -55, "dist_ph250", True),
        ("深跌<-60 · 选最深", p["dist_ph250"] < -60, "dist_ph250", True),
        ("深跌<-65 · 选最深", p["dist_ph250"] < -65, "dist_ph250", True),
        ("深跌<-55 · 选mom60最低", p["dist_ph250"] < -55, "mom60", True),
        ("深跌<-55 · 选close_pos最低", p["dist_ph250"] < -55, "close_pos", True),
        ("深跌<-55 · 选vol20最低", p["dist_ph250"] < -55, "vol20", True),
        ("大盘暴跌<-3 · 选深跌", p["mkt_ret"] < -3, "dist_ph250", True),
        ("大盘暴跌<-3 · 全市场选深跌", p["mkt_ret"] < -3, "dist_ph250", True),
        ("深跌<-55 且 大盘<-3", (p["dist_ph250"] < -55) & (p["mkt_ret"] < -3), "dist_ph250", True),
        ("深跌<-55 且 大盘<-2", (p["dist_ph250"] < -55) & (p["mkt_ret"] < -2), "dist_ph250", True),
        ("深跌<-55 · 随机", p["dist_ph250"] < -55, "__rand", True),
    ]
    p["__rand"] = np.random.RandomState(42).rand(len(p))
    for name, cond, sc, asc in variants:
        results.append(backtest(p, cond, sc, asc, name))

    df = pd.DataFrame(results)
    print(df.to_string(index=False))
    df.to_csv(OUT / "deep_bt_212.csv", index=False)

    # 分年: 深跌<-55 选最深
    print("\n==== 深跌<-55 · 选最深 · 分年 ====")
    sub = p[p["dist_ph250"] < -55].sort_values(["d", "dist_ph250"])
    pick = sub.groupby("d").head(1)
    pick["year"] = pick["d"].str[:4]
    yr = pick.groupby("year").agg(n=("r", "size"), mean=("r", "mean"),
                                  win=("r", lambda s: (s > 0).mean() * 100)).round(2)
    print(yr.to_string())


if __name__ == "__main__":
    main()
