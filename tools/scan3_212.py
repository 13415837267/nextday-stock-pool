# -*- coding: utf-8 -*-
"""第 212 轮 · 分时段扫描: 找出"近年仍有效"的甜点, 排除 2015 年记忆。

深跌股/恐慌抄底的全史均值被 2015 严重拉高。本脚本分三段(全史/近8年/近5年)
重扫关键特征, 并复现 dip 主策略条件, 看近年到底什么还在赚钱。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"


def seg(df: pd.DataFrame) -> dict:
    return {"n": len(df), "mean%": round(df["r"].mean(), 3),
            "med%": round(df["r"].median(), 3),
            "win%": round((df["r"] > 0).mean() * 100, 2)}


def main() -> None:
    p = pd.read_parquet(OUT / "panel_212.parquet")
    p["r"] = p["ret12_net"].clip(-21, 21)
    p["year"] = p["d"].str[:4].astype(int)

    for label, mask in [("全史", p["year"] >= 2009), ("近8年(2019+)", p["year"] >= 2019),
                        ("近5年(2022+)", p["year"] >= 2022)]:
        q = p[mask]
        print(f"\n########## {label}  n={len(q):,} ##########")

        print(f"--- dist_ph250 细分 ---")
        edges = [-200, -80, -65, -55, -45, -35, -25, -15, -5, 0]
        q1 = q[q["dist_ph250"] < 0].copy()
        q1["b"] = pd.cut(q1["dist_ph250"], edges)
        g = q1.groupby("b", observed=True).apply(lambda x: pd.Series(seg(x)), include_groups=False)
        print(g.to_string())

        print(f"--- mkt_ret 细分 ---")
        q2 = q.copy()
        q2["b"] = pd.cut(q2["mkt_ret"], [-12, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 12])
        g2 = q2.groupby("b", observed=True).apply(lambda x: pd.Series(seg(x)), include_groups=False)
        print(g2.to_string())

        # dip 主策略条件复现: 阴线 + pct(-6,-3) + 缩量(vr<=4) + mom20(5,25) + close_pos<=0.32
        dip = q[(q["pct"].between(-6, -3)) & (q["body_pct"] < 0)
                & (q["vr"] <= 4) & (q["mom20"].between(5, 25))
                & (q["close_pos"] <= 0.32)]
        print(f"--- dip 条件复现 ---")
        print("  ", seg(dip))
        # dip 叠加 大盘恐慌
        dip_mk = dip[dip["mkt_ret"] < -1]
        print(f"--- dip × 大盘恐慌(mkt<-1) ---")
        print("  ", seg(dip_mk))
        # dip 叠加 深跌(dist_ph250<-40)
        dip_deep = dip[dip["dist_ph250"] < -40]
        print(f"--- dip × 深跌(dist_ph250<-40) ---")
        print("  ", seg(dip_deep))


if __name__ == "__main__":
    main()
