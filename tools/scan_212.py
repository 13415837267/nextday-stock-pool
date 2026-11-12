# -*- coding: utf-8 -*-
"""第 212 轮 · 系统扫描: 在 panel_212 上对每个特征分箱, 找 ret12_net 显著为正的甜点区间。

"先得到能赚钱的股票, 再定策略条件": 本脚本就是第一步的证据扫描。
输出: output/scan1d_212.csv(单特征分箱) + 控制台打印每个特征的最优箱。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"

# 参与扫描的特征(数值型)
FEATS = ["pct", "close_pos", "amp_pct", "low_tail", "body_pct",
         "mom3", "mom5", "mom10", "mom20", "mom60", "bias5", "bias20",
         "dist_low20", "dist_ph20", "dist_ph60", "dist_ph250", "ma20_slope",
         "vr", "vr5", "amt_ratio", "turnover_rate", "turnover_cool",
         "up_streak", "lu5", "vol20", "prev_pct", "prev2_pct", "gap1", "mkt_ret"]

NBIN = 10


def main() -> None:
    p = pd.read_parquet(OUT / "panel_212.parquet")
    # 极端值保护: ret12_net 截断到 ±21%(涨停-跌停理论极值≈20% + 成本)
    p["r"] = p["ret12_net"].clip(-21, 21)
    print(f"样本 {len(p):,}  全局 ret12_net 均值 {p['r'].mean():.3f}%  中位数 {p['r'].median():.3f}%",
          flush=True)

    rows = []
    for f in FEATS:
        x = p[f].astype("float64")
        lo, hi = x.quantile(0.01), x.quantile(0.99)
        xc = x.clip(lo, hi)
        try:
            bins = pd.qcut(xc, NBIN, duplicates="drop")
        except ValueError:
            continue
        g = p.groupby(bins, observed=True)["r"].agg(n="size", mean="mean", med="median",
                                                     win=lambda s: (s > 0).mean() * 100)
        g = g.sort_index()
        best = g.loc[g["mean"].idxmax()]
        worst = g.loc[g["mean"].idxmin()]
        for iv, r in g.iterrows():
            rows.append({"feat": f, "bin": str(iv), "n": int(r["n"]),
                         "mean%": round(r["mean"], 3), "med%": round(r["med"], 3),
                         "win%": round(r["win"], 2)})
        print(f"[{f}] 最优箱 {best.name}: mean {best['mean']:+.3f}% win {best['win']:.1f}% n {int(best['n']):,}"
              f"   |  最差箱 {worst.name}: mean {worst['mean']:+.3f}%", flush=True)

    pd.DataFrame(rows).to_csv(OUT / "scan1d_212.csv", index=False)
    print("saved", OUT / "scan1d_212.csv")


if __name__ == "__main__":
    main()
