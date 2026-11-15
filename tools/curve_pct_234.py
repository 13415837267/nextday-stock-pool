"""第二百三十四轮: 主发现 —— daytime 涨跌幅与次日收益是倒 U 型

单因子回测暴露出一个矛盾: 按 pct 从大到小买(买当日涨最多的)年化 +33%,
按 pct 从小到大买(买跌最多的)年化 -42.87%, 差 76pp。但策略本身明明是买跌的,
单笔却有 1.25%。三者只有在"倒 U 型"下才同时成立。

本脚本把这条曲线逐档画出来, 并做分年验证 —— 因为如果倒 U 型是稳定的,
那么"排序选股"这个范式本身就抓不到它(只能在两端取极值), 必须用区间条件。
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
FEAT = ROOT / "output" / "feat234"


def load(scope: str) -> pd.DataFrame:
    paths = sorted(FEAT.glob("f*.parquet"))
    cols = ["trade_date", "is_paused", "is_st", "y", "y_1d", "n1_ok", "pct", "listed"]
    parts = []
    for p in paths:
        d = pd.read_parquet(p, columns=cols)
        d = d[(d["is_paused"] == 0) & (d["is_st"] == 0)]
        d = d[d["pct"].between(-11.5, 11.5)]
        d = d[d["y"].between(-40, 60)]
        d = d[d["listed"] >= 250]
        if scope != "all":
            d = d[d["n1_ok"] == True]          # noqa: E712
        if len(d):
            parts.append(d)
    return pd.concat(parts, ignore_index=True)


def main() -> None:
    scope = sys.argv[1] if len(sys.argv) > 1 else "lowopen"
    d = load(scope)
    d["year"] = (d["trade_date"] // 10000).astype(int)
    print(f"{scope}: {len(d)} 行", flush=True)

    edges = np.arange(-11.5, 11.6, 0.5)
    d["b"] = pd.cut(d["pct"], edges)
    g = d.groupby("b", observed=True).agg(n=("y", "size"), y=("y", "mean"),
                                          win=("y", lambda s: (s > 0).mean() * 100),
                                          y1=("y_1d", "mean")).reset_index()
    g["mid"] = g["b"].apply(lambda x: (x.left + x.right) / 2)
    g = g[g["n"] >= 300]
    g.round(3).to_csv(ROOT / "output" / f"w234_curve_pct_{scope}.csv",
                      index=False, encoding="utf-8-sig")
    print(g[["mid", "n", "y", "win", "y1"]].round(3).to_string(index=False), flush=True)

    # 分年形状: 每年按同样区间聚合, 看"峰值位置"落在哪
    rows = []
    for y, gg in d.groupby("year"):
        if len(gg) < 20000:
            continue
        t = gg.groupby("b", observed=True)["y"].mean()
        t = t[t.notna()]
        if len(t) < 10:
            continue
        peak = t.idxmax()
        rows.append({"year": int(y), "峰值区间": f"{peak.left:+.1f}~{peak.right:+.1f}",
                     "峰值收益": round(float(t.max()), 3),
                     "最低收益": round(float(t.min()), 3),
                     "跌9%档": round(float(t.get(pd.Interval(-9.5, -9.0), np.nan)), 3),
                     "涨3%档": round(float(t.get(pd.Interval(2.5, 3.0), np.nan)), 3)})
    yr = pd.DataFrame(rows)
    print("\n==== 分年峰值位置 ====", flush=True)
    print(yr.to_string(index=False), flush=True)
    yr.to_csv(ROOT / "output" / f"w234_curve_pct_year_{scope}.csv",
              index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
