"""第166轮诊断: 把 T+1 竞价缺口拆成 "大盘 beta 缺口" 与 "个股超额缺口"。

背景: buy_gap_below=0 意味着"开盘不挑单", 任何缺口都买。config 里的缺口分桶说明
      "低开才是 edge 来源", 但没有区分 —— 大盘整体跳空低开时, 个股低开是 beta(被动),
      个股独立低开才是"错杀"。若超额缺口才是真信号, 现行口径等于把 beta 噪声也当成了 edge。

数据: output/rank_ret.pkl(逐日池内 1~6 名, 含 d1 开盘/T+2 开盘) + data/cache/panel_all.pkl
口径: ret = T+2 开盘 / T+1 开盘 - 1(毛, 未扣费); gap = T+1 开盘/昨收 - 1
"""
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)


def market_gap():
    cache = os.path.join(ROOT, "data", "cache", "mkt_gap.pkl")
    if os.path.exists(cache):
        return pd.read_pickle(cache)
    p = pd.read_pickle(os.path.join(ROOT, "data", "cache", "panel_all.pkl"))
    p = p[(p["is_paused"] == 0) & (p["pre_close"] > 0)]
    p["g"] = p["open"] / p["pre_close"] - 1
    p = p[p["g"].abs() < 0.11]
    g = p.groupby("d")["g"].agg(med="median", mean="mean", q25=lambda x: x.quantile(0.25),
                                q75=lambda x: x.quantile(0.75), n="count")
    g.to_pickle(cache)
    return g


def main():
    g = market_gap()
    r = pd.read_pickle(os.path.join(ROOT, "output", "rank_ret.pkl"))
    p = pd.read_pickle(os.path.join(ROOT, "data", "cache", "panel_all.pkl"))[
        ["ts_code", "d", "open", "pre_close"]]
    p["gap"] = p["open"] / p["pre_close"] - 1
    p = p.rename(columns={"ts_code": "code", "d": "d1"})
    r = r.merge(p[["code", "d1", "gap"]], on=["code", "d1"], how="left")
    r["mkt"] = r["d1"].map(g["med"])
    r["resid"] = r["gap"] - r["mkt"]
    r = r.dropna(subset=["gap", "mkt", "ret"])
    r["ret"] = r["ret"] * 100

    print(f"样本 {len(r)}, 其中 rank1 {int((r['rank'] == 1).sum())}")
    print(f"大盘缺口: 均值 {g['med'].mean()*100:.3f}% 标准差 {g['med'].std()*100:.3f}% "
          f"每日中位数样本 {g['n'].median():.0f} 只")

    print("\n=== rank1: 按原始缺口分桶 ===")
    b = [-99, -2, -1, -0.3, 0, 0.3, 1, 2, 99]
    r1 = r[r["rank"] == 1].copy()
    r1["gb"] = pd.cut(r1["gap"] * 100, b)
    t = r1.groupby("gb", observed=True)["ret"].agg(n="count", mean="mean", med="median",
                                                  win=lambda x: (x > 0).mean() * 100)
    print(t.round(3).to_string())

    print("\n=== rank1: 按超额缺口(个股 - 大盘中位)分桶 ===")
    r1["rb"] = pd.cut(r1["resid"] * 100, b)
    t2 = r1.groupby("rb", observed=True)["ret"].agg(n="count", mean="mean", med="median",
                                                    win=lambda x: (x > 0).mean() * 100)
    print(t2.round(3).to_string())

    print("\n=== rank1: 大盘缺口(当日)分桶 ===")
    r1["mb"] = pd.cut(r1["mkt"] * 100, [-99, -1.5, -0.8, -0.3, 0, 0.3, 99])
    t3 = r1.groupby("mb", observed=True)["ret"].agg(n="count", mean="mean", med="median",
                                                    win=lambda x: (x > 0).mean() * 100)
    print(t3.round(3).to_string())

    print("\n=== 二维: 大盘缺口 × 超额缺口 (rank1) ===")
    piv = r1.pivot_table(index="mb", columns="gb", values="ret", aggfunc=["mean", "count"],
                         observed=True)
    print(piv.round(2).to_string())

    print("\n=== 全 rank: 超额缺口分桶(看单调性) ===")
    r["rb"] = pd.cut(r["resid"] * 100, b)
    t4 = r.groupby("rb", observed=True)["ret"].agg(n="count", mean="mean",
                                                   win=lambda x: (x > 0).mean() * 100)
    print(t4.round(3).to_string())

    r1.to_pickle(os.path.join(ROOT, "data", "cache", "rank1_gap.pkl"))


if __name__ == "__main__":
    main()
