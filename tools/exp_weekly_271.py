# -*- coding: utf-8 -*-
"""第271轮: 周线共振假设生成 —— 日线"缩量回调"是否叠加在"周线趋势仍向上"上更强。

复用 prepare_context 日面板与 output/backtest_long.json 成交记录, 按 (signal_date, code)
回查每笔成交在 T 日的周线状态。周线特征只用 signal_date 所在周**之前已完整收盘的周**,
无未来函数。只输出"是否呈单调关系"的证据, 不改任何参数。
"""
import json
import math
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import pandas as pd

from src.backtest import prepare_context
from src.config import load_config

TRADES_FILE = "output/backtest_long.json"
DAYS = 3000
WEEK_MA = 10
WEEK_MOM_FAST = 4
WEEK_MOM_SLOW = 8
WEEK_VOL_BASE = 5

FEATS = ["w_bias", "w_mom_fast", "w_mom_slow", "w_vol_ratio"]


def weekly_frame(view):
    n = len(view)
    dates = []
    close, vol, turn = [], [], []
    for j in range(n):
        r = view.iloc[j]
        dates.append(view.index[j])
        close.append(r["close_adj"])
        vol.append(r["vol"])
        turn.append(r["turnover_rate"])
    s = pd.DataFrame({
        "date": pd.to_datetime(dates, format="%Y%m%d"),
        "close": np.asarray(close, dtype=float),
        "vol": np.asarray(vol, dtype=float),
        "turn": np.asarray(turn, dtype=float),
    }).set_index("date")
    w = s.resample("W-FRI").agg({"close": "last", "vol": "sum", "turn": "sum"})
    w = w.dropna(subset=["close"])
    w["ma"] = w["close"].rolling(WEEK_MA).mean()
    w["mom_fast"] = w["close"].pct_change(WEEK_MOM_FAST) * 100
    w["mom_slow"] = w["close"].pct_change(WEEK_MOM_SLOW) * 100
    w["vol_base"] = w["vol"].shift(1).rolling(WEEK_VOL_BASE).mean()
    w["vol_ratio"] = w["vol"] / w["vol_base"]
    w["bias"] = (w["close"] / w["ma"] - 1) * 100
    return w


def weekly_asof(w, d):
    dt = pd.Timestamp(d)
    week_start = dt - pd.Timedelta(days=dt.weekday())
    prior = w[w.index < week_start]
    if prior.empty:
        return None
    last = prior.iloc[-1]
    return {
        "w_bias": last["bias"],
        "w_mom_fast": last["mom_fast"],
        "w_mom_slow": last["mom_slow"],
        "w_vol_ratio": last["vol_ratio"],
        "w_above_ma": int(last["close"] > last["ma"]),
    }


def main():
    cfg = load_config()
    ctx = prepare_context(cfg, DAYS)
    per_code = ctx["per_code"]

    tr = json.load(open(TRADES_FILE, encoding="utf-8")).get("trades_all") or []
    real = [t for t in tr if t.get("ret_pct") is not None]
    print(f"成交笔数: {len(real)}")

    wcache = {}
    rows = []
    miss = 0
    for t in real:
        code = t["code"]
        if code not in wcache:
            view = per_code.get(code)
            wcache[code] = weekly_frame(view) if view is not None else None
        w = wcache[code]
        if w is None:
            miss += 1
            continue
        f = weekly_asof(w, t["signal_date"])
        if f is None:
            miss += 1
            continue
        rec = {"ret": t["ret_pct"], "pool": t.get("pool"), "score": t.get("score")}
        rec.update(f)
        rows.append(rec)
    print(f"匹配成功 {len(rows)} 笔, 未匹配 {miss} 笔\n")

    df = pd.DataFrame(rows)
    print("=" * 78)
    print("一、周线特征与单笔收益的秩相关（Spearman）")
    print("=" * 78)
    print(f"{'特征':14s} {'有效数':>6s} {'rho':>8s} {'p值':>8s}  方向")
    out = []
    for f in FEATS:
        s = df[[f, "ret"]].dropna()
        if len(s) < 30:
            continue
        rho = s[f].corr(s["ret"], method="spearman")
        n = len(s)
        if n > 2 and abs(rho) < 1:
            tstat = rho * math.sqrt((n - 2) / (1 - rho ** 2))
            p = 2 * (1 - 0.5 * (1 + math.erf(abs(tstat) / math.sqrt(2))))
        else:
            p = 1.0
        out.append((abs(rho), f, n, rho, p))
    for _, f, n, rho, p in sorted(out, reverse=True):
        flag = "***" if p < 0.01 else ("**" if p < 0.05 else ("*" if p < 0.10 else ""))
        print(f"{f:14s} {n:6d} {rho:8.3f} {p:8.4f}  {flag}")
    print("（*** p<0.01, ** p<0.05, * p<0.10）\n")

    print("=" * 78)
    print("二、分位分组（5 档，按特征值从小到大）")
    print("=" * 78)
    for f in FEATS:
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
    print("三、二值特征: 周线收盘是否站上10周均线")
    print("=" * 78)
    for v, sub in df.dropna(subset=["w_above_ma"]).groupby("w_above_ma"):
        print(f"  ={int(v)}: {len(sub):4d} 笔, 均值 {sub['ret'].mean():+7.2f}%, "
              f"中位 {sub['ret'].median():+7.2f}%, 胜率 {(sub['ret'] > 0).mean() * 100:5.1f}%")


if __name__ == "__main__":
    main()
