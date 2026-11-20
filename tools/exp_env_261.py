"""第261轮c: 策略效率是不是"小盘环境"的函数

按年单笔: 2017 +0.11%(35笔) / 2018 +0.79% / 2019 +0.91% / 2015 +2.07% / 2021 +2.33% / 2022 +2.56%
2017 是"漂亮50"独涨、小盘阴跌的一年。策略买的是"小盘深跌反弹", 那一年没有反弹。

若效率能被"小盘环境"解释, 就存在一个真正的仓位开关(不是择方向, 是择环境)。
指标: 信号日之前 60 个交易日, 全市场个股涨跌幅**中位数**的累计值(等权, 不被权重股带偏)。
纯 T 日及之前的信息, 盘后可得, 无未来函数。
"""

from __future__ import annotations

import pickle
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data/panel"


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 3200
    pk = ROOT / f"output/w259_fbgate_{n}.pkl"
    g = pickle.load(open(pk, "rb"))
    A = [x for x in g if x["name"].startswith("A ")][0]["trades"]
    t = A[A.ret_pct.notna()].copy().sort_values("buy_date").reset_index(drop=True)

    days = sorted(p.stem.replace(".csv", "") for p in PANEL.glob("*.csv.gz"))
    med = {}
    for d in days:
        try:
            p = pd.read_csv(PANEL / f"{d}.csv.gz", usecols=["pct"], low_memory=False)
        except Exception:
            continue
        v = pd.to_numeric(p["pct"], errors="coerce").dropna()
        if len(v) > 200:
            med[d] = float(v.median())
    s = pd.Series(med).sort_index()
    # 60 日累计(求和即近似复利, 单日都在 1% 量级)
    cum60 = s.rolling(60).sum()
    print(f"中位数序列 {len(s)} 天, 60日累计 中位 {cum60.median():.1f}%  "
          f"p10 {cum60.quantile(.1):.1f}%  p90 {cum60.quantile(.9):.1f}%")

    t["sd"] = t.signal_date.astype(str).str.replace("-", "")
    t["m60"] = t.sd.map(cum60)
    x = t[t.m60.notna()].copy()
    print(f"匹配 {len(x)} 笔")

    b = pd.qcut(x.m60, 5, labels=["Q1最弱", "Q2", "Q3", "Q4", "Q5最强"])
    out = x.groupby(b, observed=True).agg(n=("ret_pct", "size"), mean=("ret_pct", "mean"),
                                          win=("ret_pct", lambda s: (s > 0).mean() * 100),
                                          m60=("m60", "median"))
    out["mean"] = out["mean"].round(2)
    out["win"] = out["win"].round(0)
    out["m60"] = out["m60"].round(1)
    print(out.to_string())

    print("\n阈值扫描(低于阈值时单笔):")
    for th in (-8, -5, -2, 0, 2, 5):
        lo = x[x.m60 <= th]
        hi = x[x.m60 > th]
        print(f"  m60<={th:>3}%: {len(lo):>3}笔 {lo.ret_pct.mean():>5.2f}%  |  "
              f"其余 {len(hi):>3}笔 {hi.ret_pct.mean():>5.2f}%")

    print("\n分年 × 环境:")
    x["y"] = x.buy_date.str[:4].astype(int)
    yy = x.groupby("y").agg(n=("ret_pct", "size"), mean=("ret_pct", "mean"), m60=("m60", "median"))
    yy["mean"] = yy["mean"].round(2)
    yy["m60"] = yy["m60"].round(1)
    print(yy.to_string())
    x.to_csv(ROOT / f"output/w261c_env_{n}.csv", index=False)


if __name__ == "__main__":
    main()
