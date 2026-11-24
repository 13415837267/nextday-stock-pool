# -*- coding: utf-8 -*-
"""第271轮: 量能背离假设生成 —— OBV 趋势是否独立于价格趋势提供增量。

复用 prepare_context 日面板与 output/backtest_long.json 成交记录, 按 (signal_date, code)
回查每笔成交在 T 日的量能状态。OBV 为累积有向量(仅用截至T日数据, 无未来函数)。
只输出"是否呈单调关系"的证据, 不改任何参数。
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
OBV_WIN = 20

FEATS = ["obv_chg", "div"]


def daily_frame(view):
    n = len(view)
    dates = []
    close, vol = [], []
    for j in range(n):
        r = view.iloc[j]
        dates.append(view.index[j])
        close.append(r["close_adj"])
        vol.append(r["vol"])
    close = np.asarray(close, dtype=float)
    vol = np.asarray(vol, dtype=float)
    ret = pd.Series(close, index=dates).pct_change()
    obv = (np.sign(ret.fillna(0.0)) * vol).cumsum()
    return dates, close, obv


def obv_asof(dates, close, obv, d):
    if d not in dates:
        return None
    i = dates.index(d)
    if i < OBV_WIN:
        return None
    obv_chg = obv[i] / obv[i - OBV_WIN] - 1 if obv[i - OBV_WIN] != 0 else np.nan
    price_chg = close[i] / close[i - OBV_WIN] - 1
    return {"obv_chg": obv_chg * 100, "div": (obv_chg - price_chg) * 100}


def main():
    cfg = load_config()
    ctx = prepare_context(cfg, DAYS)
    per_code = ctx["per_code"]

    tr = json.load(open(TRADES_FILE, encoding="utf-8")).get("trades_all") or []
    real = [t for t in tr if t.get("ret_pct") is not None]
    print(f"成交笔数: {len(real)}")

    dcache = {}
    rows = []
    miss = 0
    for t in real:
        code = t["code"]
        if code not in dcache:
            view = per_code.get(code)
            dcache[code] = daily_frame(view) if view is not None else None
        fdata = dcache[code]
        if fdata is None:
            miss += 1
            continue
        f = obv_asof(*fdata, t["signal_date"])
        if f is None:
            miss += 1
            continue
        rec = {"ret": t["ret_pct"], "pool": t.get("pool"), "score": t.get("score")}
        rec.update(f)
        rows.append(rec)
    print(f"匹配成功 {len(rows)} 笔, 未匹配 {miss} 笔\n")

    df = pd.DataFrame(rows)
    print("=" * 78)
    print("一、量能特征与单笔收益的秩相关（Spearman）")
    print("=" * 78)
    print(f"{'特征':12s} {'有效数':>6s} {'rho':>8s} {'p值':>8s}  方向")
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
        print(f"{f:12s} {n:6d} {rho:8.3f} {p:8.4f}  {flag}")
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
    print("三、二值特征: 20日 OBV 是否为正(量能确认上行趋势)")
    print("=" * 78)
    s = df.dropna(subset=["obv_chg"])
    s["pos"] = (s["obv_chg"] > 0).astype(int)
    for v, sub in s.groupby("pos"):
        print(f"  ={int(v)}: {len(sub):4d} 笔, 均值 {sub['ret'].mean():+7.2f}%, "
              f"中位 {sub['ret'].median():+7.2f}%, 胜率 {(sub['ret'] > 0).mean() * 100:5.1f}%")


if __name__ == "__main__":
    main()
