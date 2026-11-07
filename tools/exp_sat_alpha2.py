# -*- coding: utf-8 -*-
"""第一百七十一轮: 副策略换 alpha 源 —— 高密度事件扫描。

问题
----
副策略(低波 4 只 / 持 10 日)年化 19.16%, 单笔 +1.366%; 主策略单笔 +1.435%。
**单笔几乎一样, 差 7 倍的是持有期** —— 副策略的钱一年只周转 26 次, 主策略 100 次。
第 163 轮已证明: 低波因子在 1~3 天尺度上没有 alpha(3 日 Q5-Q1 = -0.47%)。
所以"缩短持有期"不能靠在低波池里换排序键, 必须换入场事件。

本脚本用**同一口径**横向评测 13 个候选事件, 回答一个问题:
  有没有一个事件, 能在 **3 个交易日内**做出 **>=1.0%/笔** 的净收益, 且信号密度够(>=0.2 只/日)?

口径(与主策略/副策略一致, 全部无未来函数)
------------------------------------------
· T 日收盘出信号(只用截至 T 日收盘可知的字段)
· T+1 开盘买, 若 T+1 一字涨停(open >= high_limit)或停牌/无行情则买不到, 剔除
· T+1+hold 收盘卖; 复权价; 往返成本 0.3%(与 satellite.cost_pct 同)
· 超额 = 单笔净收益 - 同期市场等权收益(reg.mkt_idx 同区间), 剔除 beta

基础池: 主板+创业板 / 非ST / 非停牌 / bars>=250 / 成交额>=1亿(次新事件放宽到 40~250 日)

用法: python tools/exp_sat_alpha2.py [天窗口, 默认 1500]
输出: output/exp_sat_alpha2.json
"""
from __future__ import annotations

import json
import os
import sys

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.backtest import prepare_context      # noqa: E402
from src.config import board_of, load_config  # noqa: E402

OUT = os.path.join(ROOT, "output")
COST = 0.003          # 往返成本(与 satellite.cost_pct=0.3 同)
HOLDS = (1, 2, 3, 5)

PANEL_COLS = [
    "ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
    "pct", "vol", "amount", "high_limit", "low_limit", "turnover_rate",
    "is_paused", "is_st", "open_adj", "close_adj",
    "ma5", "ma10", "ma20", "vol_ratio", "close_pos", "amp_pct", "low_tail",
    "dist_ma5", "dist_high20", "dist_low20", "mom3", "mom20", "mom60",
    "bias20", "turnover_cool", "ivol20", "bars", "prev_pct", "down2",
]


# ---------------------------------------------------------------- 事件定义
def base_pool(day, min_bars=250):
    """基础池: 主板+创业板 / 非ST / 非停牌 / 上市满 min_bars / 成交额>=1亿。"""
    return ((day["is_paused"] == 0) & (day["is_st"] == 0)
            & (day["bars"] >= min_bars) & (day["amount"] >= 1e8)
            & (day["ivol20"] > 0))


def events(day):
    """返回 {事件名: 布尔掩码}。全部只用截至 T 日收盘可知的字段。"""
    b = base_pool(day)
    b_new = base_pool(day, min_bars=40) & (day["bars"] < 250)   # 次新: 40~250 日
    pct = day["pct"]
    vr = day["vol_ratio"]
    cp = day["close_pos"]
    bd = day["ts_code"].map(board_of)
    gap = (day["open"] / day["pre_close"].replace(0, np.nan) - 1) * 100
    upper = (day["high"] - day[["open", "close"]].max(axis=1)) \
        / day["pre_close"].replace(0, np.nan) * 100
    to_p95 = float(day["turnover_rate"].quantile(0.95))
    ivol_q10 = float(day["ivol20"].quantile(0.10))
    return {
        # 1) 涨停次日回踩(缩量): 昨日涨停, 今日回调但没崩
        "limitup_pullback": b & (day["prev_pct"] >= 9.4) & pct.between(-6, -1) & (vr < 1.3),
        # 2) 涨停次日强势不破(连板预期)
        "limitup_hold": b & (day["prev_pct"] >= 9.4) & pct.between(0, 8) & (cp > 0.6),
        # 3) 大幅低开后被拉回(日内修复)
        "gap_recover": b & (gap < -3) & (cp > 0.7) & (pct > -3),
        # 4) 三连阴缩量(抛压衰竭)
        "triple_down": b & (pct < 0) & (day["down2"] == 1) & (vr < 0.7),
        # 5) 次新股超跌
        "new_stock_dip": b_new & pct.between(-9, -4),
        # 6) 创业板超跌(主策略同构, 换域)
        "gem_dip": b & (bd == "gem") & pct.between(-6, -3) & (vr < 0.8) & (cp < 0.4),
        # 7) 主板超跌(主策略同构, 口径校准用)
        "main_dip": b & (bd == "main") & pct.between(-6, -3) & (vr < 0.8) & (cp < 0.4),
        # 8) 高换手票回踩(游资票, 换手前 5%)
        "high_turn_pullback": b & (day["turnover_rate"] >= to_p95) & pct.between(-6, -2),
        # 9) 长上影 + 放量(冲高回落, 次日修复)
        "long_upper": b & (upper > 2.5) & (vr > 1.3) & pct.between(-5, 3),
        # 10) 回踩 20 日线企稳(趋势中回调)
        "ma20_support": b & day["bias20"].between(-1, 2) & (pct > 0) & (day["mom20"] > 5),
        # 11) 极度缩量横盘(变盘前夜)
        "shrink_flat": b & (vr < 0.5) & (pct.abs() < 2),
        # 12) 现行副策略因子(低波前 10%), 作 3 日持有的对照
        "lowvol_q10": b & (day["ivol20"] <= ivol_q10),
        # 13) 涨停(首板)直接持有: 验证"追涨停"是否可行
        "limitup_day": b & (pct >= 9.4) & (cp > 0.5),
    }


# ---------------------------------------------------------------- 主流程
def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    cfg = load_config()
    print(f"[ctx] 准备 {days} 日上下文 ...", flush=True)
    ctx = prepare_context(cfg, days)
    main_dates = ctx["main_dates"]
    by_date = ctx["by_date"]
    reg = ctx["reg"]
    if not isinstance(reg.index[0], str):
        reg = reg.copy()
        reg.index = reg.index.astype(str)
    lvl_arr = np.array([float(reg.loc[d, "mkt_idx"]) if d in reg.index else np.nan
                        for d in main_dates])
    pos_of = {d: i for i, d in enumerate(main_dates)}
    N = len(main_dates)

    print(f"[panel] 拼接 {N} 日 ...", flush=True)
    frames = []
    for d in main_dates:
        f = by_date.get(d)
        if f is None:
            continue
        frames.append(f[PANEL_COLS])
    panel = pd.concat(frames, ignore_index=True)
    del frames
    print(f"[panel] {len(panel):,} 行", flush=True)

    panel["p"] = panel["trade_date"].map(pos_of).astype("int32")
    panel = panel[panel["p"].notna()].reset_index(drop=True)
    panel["p"] = panel["p"].astype("int32")

    # 按 (日期位置, 代码) 建索引 -> 用 reindex 取 T+1 / T+1+h 的价格, 停牌缺日自动 NaN
    px = panel.set_index(["p", "ts_code"])[["open_adj", "close_adj", "open",
                                            "high_limit", "pre_close", "vol"]]

    def at(k):
        mi = pd.MultiIndex.from_arrays([panel["p"] + k, panel["ts_code"]])
        return px.reindex(mi).to_numpy()

    b_open, b_close_unused, b_open_raw, b_hilim, b_pre, b_vol = at(1).T
    panel["buy_open"] = b_open
    panel["buy_raw"] = b_open_raw
    panel["buy_hi_lim"] = b_hilim
    panel["buy_gap"] = (b_open_raw / np.where(b_pre.astype(float) > 0, b_pre, np.nan) - 1) * 100
    panel["buyable"] = (np.isfinite(b_open.astype(float)) & (b_vol.astype(float) > 0)
                        & (b_open_raw < b_hilim - 1e-9))
    for h in HOLDS:
        s = at(h + 1)[:, 1]                       # close_adj
        panel[f"sell_{h}"] = s
        i1 = np.clip(panel["p"].to_numpy() + 1, 0, N - 1)
        i2 = np.clip(panel["p"].to_numpy() + 1 + h, 0, N - 1)
        panel[f"mkt_{h}"] = lvl_arr[i2] / lvl_arr[i1] - 1
    del px

    # ---- 逐日算事件掩码, 记录行号 ----
    print("[events] 逐日计算掩码 ...", flush=True)
    ev_names = list(events(by_date[main_dates[0]]).keys())
    hits = {k: [] for k in ev_names}
    rid = np.arange(len(panel))
    # 用 searchsorted 定位每日分块(panel 已按日期主序 concat, p 单调不减)
    bounds = np.searchsorted(panel["p"].to_numpy(), np.arange(N + 1))
    for i, d in enumerate(main_dates):
        day = by_date.get(d)
        if day is None:
            continue
        lo, hi = bounds[i], bounds[i + 1]
        if hi <= lo:
            continue
        blk = panel.iloc[lo:hi]
        ev = events(blk)
        for k, m in ev.items():
            mm = np.asarray(m, dtype=bool)
            if mm.any():
                hits[k].append(rid[lo:hi][mm])

    out = {"window": {"days": N, "start": main_dates[0], "end": main_dates[-1]},
           "events": {}}
    for k in ev_names:
        if not hits[k]:
            continue
        ix = np.concatenate(hits[k])
        df = panel.iloc[ix]
        df = df[df["buyable"].fillna(False)]
        if len(df) < 50:
            continue
        rec = {"n_signals": int(len(df)),
               "n_days": int(df["p"].nunique()),
               "per_day": round(len(df) / N, 2),
               "avg_gap": round(float(df["buy_gap"].mean()), 3)}
        for h in HOLDS:
            r = (df[f"sell_{h}"].astype(float) / df["buy_open"].astype(float)) * (1 - COST) - 1
            mk = df[f"mkt_{h}"].astype(float)
            ok = np.isfinite(r.to_numpy()) & np.isfinite(mk.to_numpy())
            r2, mk2 = r[ok], mk[ok]
            if len(r2) < 30:
                continue
            yrs_arr = np.asarray(main_dates, dtype=object)[df.loc[ok, "p"].to_numpy()]
            yr = pd.Series([str(x)[:4] for x in yrs_arr], index=r2.index)
            rec[f"h{h}"] = {
                "n": int(len(r2)),
                "avg_pct": round(float(r2.mean()) * 100, 3),
                "excess_pct": round(float((r2 - mk2).mean()) * 100, 3),
                "win_pct": round(float((r2 > 0).mean()) * 100, 1),
                "median_pct": round(float(r2.median()) * 100, 3),
                "by_year": {str(y): round(float(v.mean()) * 100, 2)
                            for y, v in r2.groupby(yr)},
            }
            if h == 3:
                rec["ann_equiv_pct"] = round(((1 + float(r2.mean())) ** (252.0 / h) - 1) * 100, 1)
        out["events"][k] = rec

    print("\n%-20s %6s %6s %8s %8s %7s %10s" % (
        "事件", "信号数", "只/日", "h3单笔%", "h3超额%", "h3胜率", "年化等价%"))
    order = sorted(out["events"].items(),
                   key=lambda kv: -kv[1].get("h3", {}).get("excess_pct", -99))
    for k, v in order:
        h = v.get("h3")
        if not h:
            continue
        print("%-20s %6d %6.2f %8.3f %8.3f %6.1f%% %10.1f" % (
            k, v["n_signals"], v["per_day"], h["avg_pct"], h["excess_pct"],
            h["win_pct"], v.get("ann_equiv_pct", float("nan"))))

    # 分年(仅 h3)
    print("\n分年 h3 单笔均值(%):")
    yrs = sorted({y for v in out["events"].values()
                  for y in v.get("h3", {}).get("by_year", {})})
    print("%-20s %s" % ("事件", " ".join("%7s" % y for y in yrs)))
    for k, v in order:
        h = v.get("h3")
        if not h:
            continue
        print("%-20s %s" % (k, " ".join("%7.2f" % h["by_year"].get(y, float("nan"))
                                        for y in yrs)))

    md = out["events"].get("main_dip", {})
    if "h1" in md:
        print("\n[口径自检] main_dip h1 单笔 %.3f%% (主策略官方 1.435%%)" % md["h1"]["avg_pct"])

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "exp_sat_alpha2.json")
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1)
    print("\nsaved", p)


if __name__ == "__main__":
    main()
