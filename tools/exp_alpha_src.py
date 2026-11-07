# -*- coding: utf-8 -*-
"""第一百六十五轮: alpha 归因 —— "低开"本身是负期望, 只有"低开 × 特定形态"才是正期望。

问题
----
主策略的成交条件是"T+1 开盘缺口 <= 0 才买"(风控日 <= -2.5%)。第 119 轮扫过 buy_gap_below
-0.3 / -0.5 全部崩掉, 但一直没分清两件事:
  · "低开"这个动作本身有没有 alpha?
  · 还是"低开"只在这个特定的超跌形态池里才是好事?
本脚本用全市场 260 万个(日,票)做对照组。

口径
----
同一口径: T 日收盘为信号日, T+1 开盘买, T+2 开盘卖(持有 1 个交易日), 复权价, 无成本。
全市场池: 主板 / 非 ST / 非停牌 / 收盘价 3~150 / 成交额 >= 0.8 亿(与 universe 一致)。

用法: python tools/exp_alpha_src.py
"""
import glob
import os
import pickle

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys_path = ROOT
import sys
sys.path.insert(0, sys_path)
from src.config import board_of

CACHE = os.path.join(ROOT, "data", "cache", "panel_all.pkl")


def load():
    if os.path.exists(CACHE):
        return pickle.load(open(CACHE, "rb"))
    frames = []
    for f in sorted(glob.glob(os.path.join(ROOT, "data", "panel", "*.csv.gz"))):
        x = pd.read_csv(f, dtype={"ts_code": str},
                        usecols=["ts_code", "open", "close", "pre_close", "factor",
                                 "amount", "is_paused", "is_st"])
        x["d"] = os.path.basename(f)[:8]
        frames.append(x)
    df = pd.concat(frames, ignore_index=True).sort_values(["ts_code", "d"]).reset_index(drop=True)
    pickle.dump(df, open(CACHE, "wb"))
    return df


def main():
    df = load()
    df["board"] = df["ts_code"].map(board_of)
    df["open_adj"] = df["open"] * df["factor"]
    g = df.groupby("ts_code", sort=False)
    df["o_next"] = g["open_adj"].shift(-1)
    df["o_next2"] = g["open_adj"].shift(-2)
    df["gap"] = df["open"] / df["pre_close"] - 1
    df["ret"] = df["o_next2"] / df["o_next"] - 1
    base = df[(df.is_paused == 0) & (df.is_st == 0) & (df.board == "主板")
              & df.close.between(3, 150) & (df.amount >= 0.8e8)
              & df.gap.notna() & df.ret.notna() & df.o_next.notna()]
    print(f"全市场可算样本 {len(base):,}")

    def show(label, s):
        n = len(s)
        m, se = s.mean(), s.std(ddof=1) / np.sqrt(n)
        print(f"{label:<22} n={n:>8,}  均 {m*100:+.3f}%  中位 {s.median()*100:+.3f}%  "
              f"胜率 {(s>0).mean()*100:.1f}%  t {m/se:+.2f}")
        return dict(label=label, n=n, mean=round(m * 100, 3), t=round(m / se, 2))

    rows = [show("全市场(不过滤)", base.ret)]
    for lo in [-0.005, -0.01, -0.015, -0.02, -0.025]:
        rows.append(show(f"gap<={lo*100:.1f}%", base.loc[base.gap <= lo, "ret"]))

    print("\n=== 低开幅度分档 ===")
    b = pd.cut(base.gap, [-1, -0.05, -0.03, -0.02, -0.01, 0, 0.01, 0.03, 1],
               labels=["≤-5%", "-5~-3%", "-3~-2%", "-2~-1%", "-1~0%", "0~1%", "1~3%", ">3%"])
    tab = base.groupby(b, observed=True).ret.agg(
        n="count", mean=lambda x: (x.mean() * 100).round(3),
        med=lambda x: (x.median() * 100).round(3), win=lambda x: ((x > 0).mean() * 100).round(1))
    print(tab.to_string())

    print("\n=== 对照: 主策略实际成交(353 笔, 净口径) ===")
    p = os.path.join(ROOT, "output", "backtest.json")
    if os.path.exists(p):
        import json
        m = json.load(open(p))["metrics"]
        print(f"平均单笔 {m['平均单笔收益%']:.3f}%  胜率 {m['胜率%']:.2f}%  n={m['交易次数']}")

    out = dict(market=rows, gap_table=tab.reset_index().to_dict("records"))
    import json
    json.dump(out, open(os.path.join(ROOT, "output", "exp_alpha_src.json"), "w"),
              ensure_ascii=False, indent=1, default=str)
    print("\nsaved output/exp_alpha_src.json")


if __name__ == "__main__":
    main()
