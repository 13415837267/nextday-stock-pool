"""379 笔成交的微观结构归因：哪些特征真能区分盈亏？

方法：复用 prepare_context 的日面板，按 (signal_date, code) 回查每笔成交在 T 日的
微观特征，然后按分位分组看收益/胜率，并算 Spearman 秩相关。

目的不是"找参数"，是"找假设"——只有某个特征在分组上呈单调关系，才值得为它写过滤器。
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

os.environ.setdefault("BT_FULL_TRADES", "1")

from src.backtest import prepare_context
from src.config import load_config

FEATS = ["amp_pct", "low_tail", "close_pos", "prev_pct", "down2", "dist_ma5",
         "dist_low20", "below_prev_low", "hold_prev_low", "turnover_cool",
         "vol_ratio", "bias20", "mom20", "mom3", "resid_pct", "ivol20"]


def main():
    cfg = load_config()
    days = int(cfg["backtest"]["trading_days"])
    ctx = prepare_context(cfg, days)
    by_date = ctx["by_date"]

    tr = json.load(open("output/backtest.json", encoding="utf-8")).get("trades_all") or []
    real = [t for t in tr if t.get("ret_pct") is not None]
    print(f"成交笔数: {len(real)}")

    rows = []
    miss = 0
    for t in real:
        d = str(t["signal_date"])
        df = by_date.get(d)
        if df is None:
            miss += 1
            continue
        hit = df[df["ts_code"] == t["code"]]
        if hit.empty:
            miss += 1
            continue
        r = hit.iloc[0]
        rec = {"ret": t["ret_pct"], "pool": t.get("pool"), "score": t.get("score")}
        for f in FEATS:
            rec[f] = r.get(f)
        rows.append(rec)
    print(f"匹配成功 {len(rows)} 笔, 未匹配 {miss} 笔\n")

    df = pd.DataFrame(rows)
    print("=" * 78)
    print("一、特征与单笔收益的秩相关（Spearman）")
    print("=" * 78)
    print(f"{'特征':16s} {'有效数':>6s} {'rho':>8s} {'p值':>8s}  方向")
    out = []
    for f in FEATS:
        s = df[[f, "ret"]].dropna()
        if len(s) < 30:
            continue
        rho = s[f].corr(s["ret"], method="spearman")
        n = len(s)
        # t 统计量的正态近似(n>30 时与 t 分布几乎相同), 避免额外依赖 scipy
        if n > 2 and abs(rho) < 1:
            tstat = rho * math.sqrt((n - 2) / (1 - rho ** 2))
            p = 2 * (1 - 0.5 * (1 + math.erf(abs(tstat) / math.sqrt(2))))
        else:
            p = 1.0
        out.append((abs(rho), f, n, rho, p))
    for _, f, n, rho, p in sorted(out, reverse=True):
        flag = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
        print(f"{f:16s} {n:6d} {rho:8.3f} {p:8.4f}  {flag}")
    print("（*** p<0.01, ** p<0.05, * p<0.10）\n")

    print("=" * 78)
    print("二、分位分组（5 档，按特征值从小到大）")
    print("=" * 78)
    for f in ["amp_pct", "low_tail", "close_pos", "dist_ma5", "vol_ratio",
              "bias20", "mom3", "dist_low20"]:
        s = df[[f, "ret"]].dropna()
        if len(s) < 50:
            continue
        try:
            s["g"] = pd.qcut(s[f], 5, labels=False, duplicates="drop")
        except Exception:
            continue
        print(f"\n--- {f} ---")
        print(f"{'档':>3s} {'区间':>18s} {'笔数':>5s} {'均值%':>8s} {'中位%':>8s} {'胜率%':>7s}")
        for g, sub in s.groupby("g"):
            lo, hi = sub[f].min(), sub[f].max()
            print(f"{int(g):>3d} {lo:7.2f}~{hi:7.2f} {len(sub):5d} "
                  f"{sub['ret'].mean():8.2f} {sub['ret'].median():8.2f} "
                  f"{(sub['ret'] > 0).mean() * 100:7.1f}")

    print("\n" + "=" * 78)
    print("三、二值特征对比")
    print("=" * 78)
    for f in ["down2", "below_prev_low", "hold_prev_low"]:
        s = df[[f, "ret"]].dropna()
        if s.empty:
            continue
        print(f"\n--- {f} ---")
        for v, sub in s.groupby(f):
            print(f"  ={int(v)}: {len(sub):4d} 笔, 均值 {sub['ret'].mean():+7.2f}%, "
                  f"胜率 {(sub['ret'] > 0).mean() * 100:5.1f}%")


if __name__ == "__main__":
    main()
