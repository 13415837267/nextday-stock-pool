# -*- coding: utf-8 -*-
"""regime 信号-环境关系分析: 分桶统计 + 走前纪律(h1选择/h2验证)的规则评估。"""
import sys

import numpy as np
import pandas as pd

df = pd.read_csv("regime_signals.csv", dtype={"date": str})
tr = df[df["tradable"] == 1].dropna(subset=["ret_raw_pct"]).copy()
tr["win"] = (tr["ret_raw_pct"] > 0).astype(int)
print(f"总样本: {len(df)} | 可交易信号: {len(tr)} | 全期均值 {tr['ret_raw_pct'].mean():.3f}% 胜率 {tr['win'].mean()*100:.1f}%")

ENV_COLS = ["ma_ratio_20", "ma_ratio_60", "ma_ratio_120", "ma_ratio_250",
            "breadth", "breadth_ma10", "mkt_ret_20", "mkt_vol_20", "dist_250h"]


def bucket_stats(t: pd.DataFrame, col: str, q=5):
    t = t.dropna(subset=[col]).copy()
    try:
        t["bucket"] = pd.qcut(t[col], q, duplicates="drop")
    except Exception:
        return None
    g = t.groupby("bucket", observed=True)["ret_raw_pct"].agg(["count", "mean", "median"])
    g["win%"] = t.groupby("bucket", observed=True)["win"].mean() * 100
    return g.round(3)


print("\n========== 全期分桶(qcut5) ==========")
for c in ENV_COLS:
    g = bucket_stats(tr, c)
    if g is not None:
        print(f"\n--- {c} ---")
        print(g.to_string())

print("\n========== 分年份稳定性(均值% / 胜率% / 笔数) ==========")
yr = tr.groupby("year").agg(mean=("ret_raw_pct", "mean"), win=("win", "mean"),
                            n=("ret_raw_pct", "count"))
yr["win"] = (yr["win"] * 100).round(1)
print(yr.round(3).to_string())

# 关键交叉: 早段(2012~2024-08) vs 近段(2024-09~) 在各环境变量下的方向是否一致
tr["era"] = np.where(tr["date"] < "202409", "early", "recent")
print("\n========== era × 环境变量方向一致性(qcut3均值%) ==========")
for c in ["ma_ratio_120", "ma_ratio_250", "dist_250h", "mkt_vol_20", "breadth"]:
    t = tr.dropna(subset=[c]).copy()
    try:
        t["bucket"] = pd.qcut(t[c], 3, duplicates="drop")
    except Exception:
        continue
    piv = t.pivot_table(index="bucket", columns="era", values="ret_raw_pct",
                        aggfunc="mean", observed=True).round(3)
    print(f"\n--- {c} ---")
    print(piv.to_string())
