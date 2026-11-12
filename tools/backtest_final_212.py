# -*- coding: utf-8 -*-
"""第 212 轮 · 左侧"加速暴跌反转"最终综合回测: 组合最强信号, 分时段检验近年有效性。

组合依据(深跌池内单因子最强项):
  mom20<-30(加速下跌) + amp_pct>8(巨震) + vol20>5(高波动) + prev_pct<-6(前日也大跌)
排除当日跌停(次日大概率继续跌停卖不出), 按 crash_score 选1只, 复利回测。
分 全史 / 近8年 / 近5年 三段, 检验是否靠 2015。
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"


def run(p: pd.DataFrame, cond, key, asc, name: str) -> dict:
    sub = p[cond].copy().sort_values(["d", key], ascending=[True, asc], kind="mergesort")
    pick = sub.groupby("d").head(1)
    r = pick.groupby("d")["ret12_net"].first()
    nav = (1 + r / 100).prod()
    years = p["d"].nunique() / 242
    cagr = nav ** (1 / years) - 1 if nav > 0 else -1
    curve = (1 + r / 100).cumprod()
    dd = (curve / curve.cummax() - 1).min()
    return {"name": name, "trades": len(r), "mult": round(float(nav), 1),
            "cagr%": round(float(cagr) * 100, 1), "per%": round(float(r.mean()), 3),
            "win%": round(float((r > 0).mean()) * 100, 1),
            "maxdd%": round(float(dd) * 100, 1),
            "kama": round(float(cagr) / abs(float(dd)) if dd < 0 else 99, 2)}


def main() -> None:
    p = pd.read_parquet(OUT / "panel_212.parquet")
    p["r"] = p["ret12_net"].clip(-21, 21)
    p["year"] = p["d"].str[:4].astype(int)
    p["crash"] = -p["mom20"] - p["bias20"] + p["vol20"]

    deep = p["dist_ph250"] < -55
    crash = p["mom20"] < -30
    giant = p["amp_pct"] > 8
    hot = p["vol20"] > 5
    prevdrop = p["prev_pct"] < -6
    not_ld = p["pct"] > -9.7            # 当日非跌停(跌停次日难卖)

    C1 = deep & crash
    C2 = deep & crash & giant
    C3 = deep & crash & hot
    C4 = deep & crash & giant & not_ld
    C5 = deep & crash & prevdrop & not_ld
    C6 = deep & crash & giant & hot & not_ld

    for label, mask in [("全史", p["year"] >= 2009), ("近8年2019+", p["year"] >= 2019),
                        ("近5年2022+", p["year"] >= 2022)]:
        q = p[mask]
        print(f"\n########## {label} ##########")
        rows = [
            run(q, C1, "crash", False, "C1 深跌·加速暴跌"),
            run(q, C2, "crash", False, "C2 +巨震"),
            run(q, C3, "crash", False, "C3 +高波动"),
            run(q, C4, "crash", False, "C4 +巨震·非跌停"),
            run(q, C5, "crash", False, "C5 +前日大跌·非跌停"),
            run(q, C6, "crash", False, "C6 全叠加·非跌停"),
            run(q, C6, "mom20", True, "C6 全叠加·选mom20最低"),
        ]
        print(pd.DataFrame(rows).to_string(index=False))

    # C6 全叠加 分年
    print("\n==== C6 全叠加·非跌停 · 选crash 分年 ====")
    sub = p[C6].sort_values(["d", "crash"], ascending=[True, False])
    pick = sub.groupby("d").head(1)
    pick["year"] = pick["d"].str[:4]
    yr = pick.groupby("year").agg(n=("r", "size"), mean=("r", "mean"),
                                  win=("r", lambda s: (s > 0).mean() * 100)).round(2)
    print(yr.to_string())


if __name__ == "__main__":
    main()
