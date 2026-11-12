# -*- coding: utf-8 -*-
"""第 212 轮 · 深度扫描: 左侧超跌甜点细分 + 二维交叉 + 分年稳定性。

单特征扫描已定位: dist_ph250(腰斩股) / mkt_ret(大盘恐慌) / mom60(深跌) 是仅有的正期望方向。
本脚本把这三个方向做细分箱、两两交叉, 并检验分年稳定性(是否只靠某一年)。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"


def seg(df: pd.DataFrame, r="r") -> dict:
    return {"n": len(df), "mean%": round(df[r].mean(), 3),
            "med%": round(df[r].median(), 3),
            "win%": round((df[r] > 0).mean() * 100, 2)}


def main() -> None:
    p = pd.read_parquet(OUT / "panel_212.parquet")
    p["r"] = p["ret12_net"].clip(-21, 21)
    p["year"] = p["d"].str[:4]
    print(f"样本 {len(p):,}", flush=True)

    # ---- 1) dist_ph250 细分箱(只看负值区间, 越深越细) ----
    print("\n==== dist_ph250 细分(负值) ====")
    edges = [-200, -80, -65, -55, -48, -42, -36, -30, -24, -18, -12, -6, 0]
    sub = p[p["dist_ph250"] < 0]
    sub["bin"] = pd.cut(sub["dist_ph250"], edges)
    g = sub.groupby("bin", observed=True).apply(lambda x: pd.Series(seg(x)), include_groups=False)
    print(g.to_string())

    # ---- 2) mkt_ret 细分 ----
    print("\n==== mkt_ret 细分 ====")
    edges2 = [-12, -5, -3, -2, -1.5, -1, -0.5, 0, 0.5, 1, 2, 12]
    p["bin2"] = pd.cut(p["mkt_ret"], edges2)
    g2 = p.groupby("bin2", observed=True).apply(lambda x: pd.Series(seg(x)), include_groups=False)
    print(g2.to_string())

    # ---- 3) 二维交叉: 超跌 × 大盘 ----
    print("\n==== 交叉: dist_ph250(深跌) × mkt_ret(大盘) ====")
    deep = p[(p["dist_ph250"] < -40)]
    deep["a"] = pd.cut(deep["dist_ph250"], [-200, -55, -45, -40])
    deep["b"] = pd.cut(deep["mkt_ret"], [-12, -2, -1, 0, 2, 12])
    g3 = deep.groupby(["a", "b"], observed=True).apply(
        lambda x: pd.Series(seg(x)), include_groups=False)
    print(g3[g3["n"] >= 500].to_string())

    # ---- 4) 二维交叉: 超跌 × mom60 ----
    print("\n==== 交叉: dist_ph250(深跌) × mom60 ====")
    deep2 = p[(p["dist_ph250"] < -40)].copy()
    deep2["a"] = pd.cut(deep2["dist_ph250"], [-200, -55, -45, -40])
    deep2["c"] = pd.cut(deep2["mom60"], [-200, -60, -40, -20, 0, 200])
    g4 = deep2.groupby(["a", "c"], observed=True).apply(
        lambda x: pd.Series(seg(x)), include_groups=False)
    print(g4[g4["n"] >= 500].to_string())

    # ---- 5) 分年稳定性: 深跌(dist_ph250<-45) 与 大盘恐慌(mkt_ret<-1) ----
    print("\n==== 分年: dist_ph250<-45 深跌股 ====")
    d45 = p[p["dist_ph250"] < -45]
    yr = d45.groupby("year").apply(lambda x: pd.Series(seg(x)), include_groups=False)
    print(yr.to_string())

    print("\n==== 分年: mkt_ret<-1 大盘恐慌日 全部股票 ====")
    mk = p[p["mkt_ret"] < -1]
    yr2 = mk.groupby("year").apply(lambda x: pd.Series(seg(x)), include_groups=False)
    print(yr2.to_string())


if __name__ == "__main__":
    main()
