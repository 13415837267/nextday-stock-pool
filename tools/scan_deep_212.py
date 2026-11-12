# -*- coding: utf-8 -*-
"""第 212 轮 · 深跌股子集内因子扫描: 在 dist_ph250<-55 的池子里, 找"次日反弹股"的特征。

深跌股等权 +1.3%, 但"选最深1只"反而 -0.05%(选到退市风险股)。本脚本在深跌池内
对每个特征分箱, 找出能区分"次日涨"与"次日继续跌"的因子, 解决"选哪一只"。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"

FEATS = ["pct", "close_pos", "amp_pct", "low_tail", "body_pct",
         "mom3", "mom5", "mom20", "mom60", "bias5", "bias20", "ma20_slope",
         "vr", "vr5", "amt_ratio", "turnover_rate", "turnover_cool",
         "up_streak", "vol20", "prev_pct", "prev2_pct", "gap1", "mkt_ret"]


def main() -> None:
    p = pd.read_parquet(OUT / "panel_212.parquet")
    p["r"] = p["ret12_net"].clip(-21, 21)
    deep = p[p["dist_ph250"] < -55].copy()
    print(f"深跌池(dist_ph250<-55) 样本 {len(deep):,}  等权均值 {deep['r'].mean():+.3f}%",
          flush=True)

    rows = []
    for f in FEATS:
        x = deep[f].astype("float64")
        lo, hi = x.quantile(0.02), x.quantile(0.98)
        xc = x.clip(lo, hi)
        try:
            bins = pd.qcut(xc, 8, duplicates="drop")
        except ValueError:
            continue
        g = deep.groupby(bins, observed=True)["r"].agg(
            n="size", mean="mean", win=lambda s: (s > 0).mean() * 100)
        g = g.sort_index()
        best = g.loc[g["mean"].idxmax()]
        worst = g.loc[g["mean"].idxmin()]
        # 单调性: 最优箱与最差箱的差距
        spread = best["mean"] - worst["mean"]
        print(f"[{f}] 最优{best.name}: {best['mean']:+.3f}% win{best['win']:.0f}% n{int(best['n']):,}"
              f" | 最差{worst.name}: {worst['mean']:+.3f}% | 差 {spread:+.3f}%", flush=True)
        for iv, r in g.iterrows():
            rows.append({"feat": f, "bin": str(iv), "n": int(r["n"]),
                         "mean%": round(r["mean"], 3), "win%": round(r["win"], 2)})
    pd.DataFrame(rows).to_csv(OUT / "scan_deep_212.csv", index=False)
    print("saved", OUT / "scan_deep_212.csv")


if __name__ == "__main__":
    main()
