"""amp_pct 是否只是"市值/流动性"的代理？

如果 amp_pct 与 amount(成交额)/turnover/price 强相关，那"低振幅档表现差"就只是
A 股已知的小市值溢价，不是新的 alpha，加过滤器等于偷偷提高 beta 暴露。

同时补出上一轮被截断的完整 Spearman 表与分年稳定性。
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
         "vol_ratio", "bias20", "mom20", "mom3", "resid_pct", "ivol20",
         "amount", "turnover_rate", "close"]


def _pval(rho, n):
    """t 统计量的正态近似, 避免依赖 scipy。"""
    if n <= 2 or abs(rho) >= 1:
        return 1.0
    t = rho * math.sqrt((n - 2) / (1 - rho ** 2))
    return 2 * (1 - 0.5 * (1 + math.erf(abs(t) / math.sqrt(2))))


def main():
    cfg = load_config()
    ctx = prepare_context(cfg, int(cfg["backtest"]["trading_days"]))
    by_date = ctx["by_date"]
    tr = json.load(open("output/backtest.json", encoding="utf-8")).get("trades_all") or []
    real = [t for t in tr if t.get("ret_pct") is not None]

    rows = []
    for t in real:
        df = by_date.get(str(t["signal_date"]))
        if df is None:
            continue
        hit = df[df["ts_code"] == t["code"]]
        if hit.empty:
            continue
        r = hit.iloc[0]
        rec = {"ret": t["ret_pct"], "date": str(t["signal_date"])}
        for f in FEATS:
            rec[f] = r.get(f)
        rows.append(rec)
    df = pd.DataFrame(rows)
    print(f"样本: {len(df)} 笔\n")

    print("=" * 76)
    print("一、完整 Spearman（特征 vs 单笔收益）")
    print("=" * 76)
    print(f"{'特征':16s} {'n':>5s} {'rho':>8s} {'p':>8s}  ")
    res = []
    for f in FEATS:
        s = df[[f, "ret"]].dropna()
        if len(s) < 30:
            continue
        rho = s[f].corr(s["ret"], method="spearman")
        res.append((abs(rho), f, len(s), rho, _pval(rho, len(s))))
    for _, f, n, rho, p in sorted(res, reverse=True):
        flag = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.1 else ""))
        print(f"{f:16s} {n:5d} {rho:8.3f} {p:8.4f}  {flag}")

    print("\n" + "=" * 76)
    print("二、amp_pct 与 市值/流动性代理 的相关性（判断是否只是 beta）")
    print("=" * 76)
    for f in ["amount", "turnover_rate", "close", "vol_ratio", "bias20"]:
        s = df[["amp_pct", f]].dropna()
        if len(s) < 30:
            continue
        rho = s["amp_pct"].corr(s[f], method="spearman")
        print(f"  amp_pct ~ {f:14s} rho={rho:+.3f}  p={_pval(rho, len(s)):.4f}  n={len(s)}")

    print("\n" + "=" * 76)
    print("三、amp_pct 最低档(<4.0)的时间分布与收益")
    print("=" * 76)
    df["year"] = df["date"].str[:4]
    df["low_amp"] = df["amp_pct"] < 4.0
    print(f"{'年':>6s} {'总数':>5s} {'低振幅':>6s} {'占比':>6s} "
          f"{'低振幅均值%':>11s} {'其余均值%':>10s} {'差':>7s}")
    for y, g in df.groupby("year"):
        a = g[g["low_amp"]]["ret"]
        b = g[~g["low_amp"]]["ret"]
        if len(g) == 0:
            continue
        print(f"{y:>6s} {len(g):5d} {len(a):6d} {len(a)/len(g)*100:5.1f}% "
              f"{(a.mean() if len(a) else float('nan')):11.2f} "
              f"{(b.mean() if len(b) else float('nan')):10.2f} "
              f"{(a.mean()-b.mean() if len(a) and len(b) else float('nan')):7.2f}")

    print("\n" + "=" * 76)
    print("四、若加 dip_amp_min=4.0：被剔除票的画像")
    print("=" * 76)
    out = df[df["low_amp"]]
    keep = df[~df["low_amp"]]
    print(f"剔除 {len(out)} 笔（{len(out)/len(df)*100:.1f}%）")
    print(f"  剔除组: 均值 {out['ret'].mean():+.2f}% / 中位 {out['ret'].median():+.2f}% "
          f"/ 胜率 {(out['ret']>0).mean()*100:.1f}%")
    print(f"  保留组: 均值 {keep['ret'].mean():+.2f}% / 中位 {keep['ret'].median():+.2f}% "
          f"/ 胜率 {(keep['ret']>0).mean()*100:.1f}%")
    print(f"  简单均值口径提升: {keep['ret'].mean()-df['ret'].mean():+.3f}pp/笔")
    for col in ["amount", "turnover_rate", "close"]:
        print(f"  剔除组 {col} 中位={out[col].median():.2f} vs 保留组 {keep[col].median():.2f}")


if __name__ == "__main__":
    main()
