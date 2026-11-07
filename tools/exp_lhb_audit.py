# -*- coding: utf-8 -*-
"""第一百六十七轮: 复核第 166 轮"龙虎榜收盘口径转正" —— 结论是重复计算了 T+1 日内。

第 166 轮算法(见当轮 CHANGELOG 第一节):
    l["d2"] = l["d"].map(d2map)          # d2map: d -> 下一个交易日 = T+1
    l["intra2"] = key[(code, d2)]        # 取到的是 T+1 的日内涨幅
    l["rc"] = (1 + l["r2"]) * (1 + l["intra2"]) - 1
其中 r2 = T+2开盘/T+1开盘, 本身**已经包含** T+1 日内那一段,
再乘一次 T+1 日内 = 重复计算, 虚增约 +0.5pp/笔。

正确算法: T+1开盘 -> T+2收盘 = (1+r2) * (1+T+2日内), 或直接取复权价 c2/o1。

本脚本用复权价矩阵独立重算, 并以主策略 353 笔做口径对账(毛收益-费用 应等于引擎净收益)。
"""
from __future__ import annotations

import os

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
CACHE = os.path.join(ROOT, "data", "cache")


def matrices():
    p = pd.read_pickle(os.path.join(CACHE, "panel_all.pkl"))
    p = p.loc[p["is_paused"] == 0, ["ts_code", "d", "open", "close", "factor"]]
    p["px_o"] = (p["open"].astype("float64") * p["factor"]).astype("float32")
    p["px_c"] = (p["close"].astype("float64") * p["factor"]).astype("float32")
    mo = p.pivot_table(index="d", columns="ts_code", values="px_o", aggfunc="first")
    mc = p.pivot_table(index="d", columns="ts_code", values="px_c", aggfunc="first")
    return mo, mc


def main() -> None:
    mo, mc = matrices()
    days = list(mo.index)
    pos = {d: i for i, d in enumerate(days)}
    o, c = mo.to_numpy(), mc.to_numpy()
    cols = {x: i for i, x in enumerate(mo.columns)}
    o1_map = {d: days[i + 1] for i, d in enumerate(days) if i + 1 < len(days)}
    o2_map = {d: days[i + 2] for i, d in enumerate(days) if i + 2 < len(days)}

    def ret(code: str, d: str, sell: str):
        """sell='o' -> T+2 开盘卖; sell='c' -> T+2 收盘卖。"""
        i, j = pos.get(d), cols.get(code)
        if i is None or j is None or i + 2 >= len(days):
            return np.nan
        o1 = o[i + 1, j]
        px = o[i + 2, j] if sell == "o" else c[i + 2, j]
        if not (o1 > 0) or not (px > 0):
            return np.nan
        r = px / o1 - 1.0
        return r if abs(r) <= 0.6 else np.nan

    # ---------- 1. 主策略对账 ----------
    tr = pd.DataFrame(pd.read_pickle(os.path.join(ROOT, "output", "main_trades_latest.pkl")))
    tr = tr[tr["ret_pct"].notna()].copy()
    tr["d"] = tr["signal_date"].astype(str).str.replace("-", "").str[:8]
    g = np.array([ret(cd, d, "c") for cd, d in zip(tr["code"], tr["d"])]) * 100
    print("=== 1. 主策略 353 笔对账(口径自检) ===")
    print(f"  复权价重算毛收益 {np.nanmean(g):+.3f}%   引擎净收益 {tr['ret_pct'].mean():+.3f}%   "
          f"差 {np.nanmean(g) - tr['ret_pct'].mean():+.3f}pp (费用)")

    # ---------- 2. 龙虎榜三种算法 ----------
    l = pd.read_pickle(os.path.join(CACHE, "lhb_probe.pkl"))
    l["r_open"] = [ret(cd, d, "o") for cd, d in zip(l["code"], l["d"])]
    l["r_close"] = [ret(cd, d, "c") for cd, d in zip(l["code"], l["d"])]
    # 第 166 轮复现
    p2 = pd.read_pickle(os.path.join(CACHE, "panel_all.pkl"))[["ts_code", "d", "open", "close"]]
    p2["intra"] = p2["close"] / p2["open"] - 1
    key = p2.set_index(["ts_code", "d"])["intra"]
    l["d2"] = l["d"].map(o1_map)
    l["intra_wrong"] = key.reindex(pd.MultiIndex.from_arrays([l["code"], l["d2"]])).values
    l["rc166"] = ((1 + l["r2"]) * (1 + l["intra_wrong"]) - 1) * 100

    m = l.dropna(subset=["r_open", "r_close", "rc166"])
    print(f"\n=== 2. 龙虎榜 {len(m)} 条: 三种算法 ===")
    print(f"  T+2 开盘卖(第165轮)          {m['r_open'].mean()*100:+.3f}%")
    print(f"  第166轮 rc(重复算T+1日内)     {m['rc166'].mean():+.3f}%   <- 虚增 {(m['rc166'].mean()-m['r_close'].mean()*100):+.3f}pp")
    print(f"  T+2 收盘卖(正确)             {m['r_close'].mean()*100:+.3f}%")
    print(f"  收盘腿(正确)                 {(m['r_close']-m['r_open']).mean()*100:+.3f}pp")

    # ---------- 3. 分组重算 ----------
    print("\n=== 3. 分组: 正确口径 vs 第166轮报告值 ===")
    groups = {
        "全样本": np.ones(len(m), bool),
        "换手榜": m["reason"].str.contains("换手率").to_numpy(),
        "北向净买": (m["north"] > 0).to_numpy(),
        "北向&跌5%+": ((m["north"] > 0) & (m["chg"] <= -5)).to_numpy(),
        "机构净买": (m["inst"] > 0).to_numpy(),
        "跌幅榜": m["reason"].str.contains("跌幅").to_numpy(),
        "游资净买": (m["youzi"] > 0).to_numpy(),
    }
    rows = []
    for name, mask in groups.items():
        s = m[mask]
        if len(s) < 50:
            continue
        rows.append(dict(组=name, n=len(s),
                         开盘卖=round(s["r_open"].mean() * 100, 3),
                         第166轮=round(s["rc166"].mean(), 3),
                         正确收盘=round(s["r_close"].mean() * 100, 3),
                         虚增=round(s["rc166"].mean() - s["r_close"].mean() * 100, 3)))
    print(pd.DataFrame(rows).to_string(index=False))

    # 分年
    m["y"] = m["d"].str[:4]
    print("\n=== 4. 正确口径分年 ===")
    print(m.groupby("y").agg(n=("r_close", "size"),
                             开盘=("r_open", lambda x: round(x.mean() * 100, 3)),
                             收盘=("r_close", lambda x: round(x.mean() * 100, 3))).to_string())

    m.to_pickle(os.path.join(CACHE, "lhb_audit.pkl"))
    print(f"\n保存 lhb_audit.pkl {m.shape}")


if __name__ == "__main__":
    main()
