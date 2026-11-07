# -*- coding: utf-8 -*-
"""第一百七十一轮: 副策略换 alpha 源 —— 高密度事件扫描(v2)。

问题
----
副策略(低波 4 只 / 持 10 日)年化 19.16%, 单笔 +1.366%; 主策略单笔 +1.435%。
**单笔几乎一样, 差 7 倍的是持有期** —— 副策略的钱一年只周转 26 次, 主策略 100 次。
第 163 轮已证明低波因子在 1~3 天没有 alpha(3 日 Q5-Q1 = -0.47%)。
所以"缩短持有期"必须换入场事件, 不能换排序键。

v1 的教训: 不带市场闸门时, 连 main_dip(已知 +1.435%/笔)都测出 -0.19% ——
闸门不是可选项, 它砍掉的正是"下跌市里所有形态都亏"的那一半样本。v2 全部事件
都在信号日 **叠加 reg.trade_ok**, 并用 select_stocks 的 top1 做官方口径校准。

口径
----
· T 日收盘出信号(只用截至 T 日收盘可知的字段) + 信号日闸门 trade_ok
· T+1 开盘买; T+1 一字涨停/停牌/无行情则买不到, 剔除
· 五种卖出: d0=T+1 收盘(日内) / o1=T+2 开盘(主策略口径) / c1=T+2 收盘 / c2=T+3 收盘 / c3=T+4 收盘
· 复权价; 往返成本 0.3%; 超额 = 单笔 - 同期市场等权(reg.mkt_idx 同区间)

基础池: 主板+创业板 / 非ST / 非停牌 / bars>=250 / 成交额>=1亿(次新放宽到 40~250 日)

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

from src.backtest import prepare_context                        # noqa: E402
from src.config import board_of, load_config                    # noqa: E402
from src.selector import select_stocks                          # noqa: E402

OUT = os.path.join(ROOT, "output")
COST = 0.003          # 往返成本(与 satellite.cost_pct=0.3 同)
# 卖出方式 -> (取价相对信号日的行偏移, open/close, 持有交易日数)
EXITS = {"d0": (1, "close", 0), "o1": (2, "open", 1), "c1": (2, "close", 1),
         "c2": (3, "close", 2), "c3": (4, "close", 3)}

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
        # 1) 涨停次日回踩(缩量)
        "limitup_pullback": b & (day["prev_pct"] >= 9.4) & pct.between(-6, -1) & (vr < 1.3),
        # 2) 涨停次日强势不破
        "limitup_hold": b & (day["prev_pct"] >= 9.4) & pct.between(0, 8) & (cp > 0.6),
        # 3) 大幅低开后被拉回
        "gap_recover": b & (gap < -3) & (cp > 0.7) & (pct > -3),
        # 4) 三连阴缩量
        "triple_down": b & (pct < 0) & (day["down2"] == 1) & (vr < 0.7),
        # 5) 次新股超跌
        "new_stock_dip": b_new & pct.between(-9, -4),
        # 6) 创业板超跌(主策略同构, 换域)
        "gem_dip": b & (bd == "创业板") & pct.between(-6, -3) & (vr < 0.8) & (cp < 0.4),
        # 7) 主板超跌(简化版, 只用于相对比较; 官方口径见 main_official)
        "main_dip": b & (bd == "主板") & pct.between(-6, -3) & (vr < 0.8) & (cp < 0.4),
        # 8) 高换手票回踩
        "high_turn_pullback": b & (day["turnover_rate"] >= to_p95) & pct.between(-6, -2),
        # 9) 长上影 + 放量
        "long_upper": b & (upper > 2.5) & (vr > 1.3) & pct.between(-5, 3),
        # 10) 回踩 20 日线企稳
        "ma20_support": b & day["bias20"].between(-1, 2) & (pct > 0) & (day["mom20"] > 5),
        # 11) 极度缩量横盘
        "shrink_flat": b & (vr < 0.5) & (pct.abs() < 2),
        # 12) 现行副策略因子(低波前 10%)
        "lowvol_q10": b & (day["ivol20"] <= ivol_q10),
        # 13) 首板涨停直接持有
        "limitup_day": b & (pct >= 9.4) & (cp > 0.5),
        # 14) 科创板超跌
        "star_dip": b & (bd == "科创板") & pct.between(-8, -4) & (vr < 0.8) & (cp < 0.4),
    }


def official_picks(day, d, cfg):
    """主策略官方 top1(同一套选股器), 用作口径校准。"""
    try:
        picks, _ = select_stocks(d, day, cfg, None, top_n=1, name_filter=False)
    except Exception:
        return None
    return str(picks[0]["code"]) if picks else None


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
    gate_arr = np.array([bool(reg.loc[d, "trade_ok"]) if d in reg.index else False
                         for d in main_dates])
    pos_of = {d: i for i, d in enumerate(main_dates)}
    N = len(main_dates)
    print(f"[gate] 闸门开启 {int(gate_arr.sum())}/{N} 日 ({gate_arr.mean()*100:.1f}%)", flush=True)

    print(f"[panel] 拼接 {N} 日 ...", flush=True)
    frames = [by_date[d][PANEL_COLS] for d in main_dates if d in by_date]
    panel = pd.concat(frames, ignore_index=True)
    del frames
    print(f"[panel] {len(panel):,} 行", flush=True)

    panel["p"] = panel["trade_date"].map(pos_of).astype("int32")
    panel = panel[panel["p"].notna()].reset_index(drop=True)
    panel["p"] = panel["p"].astype("int32")

    px = panel.set_index(["p", "ts_code"])[["open_adj", "close_adj", "open",
                                            "high_limit", "pre_close", "vol"]]

    def at(k):
        mi = pd.MultiIndex.from_arrays([panel["p"] + k, panel["ts_code"]])
        return px.reindex(mi).to_numpy()

    b_open, b_close_d1, b_open_raw, b_hilim, b_pre, b_vol = at(1).T
    panel["buy_open"] = b_open
    panel["buy_raw"] = b_open_raw
    panel["buy_gap"] = (b_open_raw / np.where(b_pre.astype(float) > 0, b_pre, np.nan) - 1) * 100
    panel["buyable"] = (np.isfinite(b_open.astype(float)) & (b_vol.astype(float) > 0)
                        & (b_open_raw < b_hilim - 1e-9))
    p_arr = panel["p"].to_numpy()
    for k, (off, kind, _h) in EXITS.items():
        arr = at(off)
        panel[f"sell_{k}"] = arr[:, 0] if kind == "open" else arr[:, 1]
        i1 = np.clip(p_arr + 1, 0, N - 1)
        i2 = np.clip(p_arr + off, 0, N - 1)
        panel[f"mkt_{k}"] = lvl_arr[i2] / lvl_arr[i1] - 1
    del px

    # ---- 逐日: 事件掩码 + 官方 top1 ----
    print("[events] 逐日计算掩码 ...", flush=True)
    ev_names = list(events(by_date[main_dates[0]]).keys()) + ["main_official"]
    hits = {k: [] for k in ev_names}
    rid = np.arange(len(panel))
    bounds = np.searchsorted(p_arr, np.arange(N + 1))
    for i, d in enumerate(main_dates):
        day = by_date.get(d)
        if day is None:
            continue
        lo, hi = bounds[i], bounds[i + 1]
        if hi <= lo:
            continue
        if not bool(gate_arr[i]):
            continue                      # 闸门关闭: 当日一律不开仓(所有事件同口径)
        blk = panel.iloc[lo:hi]
        ev = events(blk)
        for k, m in ev.items():
            mm = np.asarray(m, dtype=bool)
            if mm.any():
                hits[k].append(rid[lo:hi][mm])
        code = official_picks(day, d, cfg)
        if code:
            idx = blk.index[blk["ts_code"] == code]
            if len(idx):
                hits["main_official"].append(np.array([idx[0]]))

    out = {"window": {"days": N, "start": main_dates[0], "end": main_dates[-1],
                      "gate_on_pct": round(float(gate_arr.mean()) * 100, 1)},
           "events": {}}
    for k in ev_names:
        if not hits[k]:
            continue
        ix = np.concatenate(hits[k])
        df = panel.iloc[ix]
        df = df[df["buyable"].fillna(False)]
        if len(df) < 30:
            continue
        rec = {"n_signals": int(len(df)),
               "n_days": int(df["p"].nunique()),
               "per_day": round(len(df) / N, 2),
               "avg_gap": round(float(df["buy_gap"].mean()), 3)}
        for key in EXITS:
            r = (df[f"sell_{key}"].astype(float) / df["buy_open"].astype(float)) * (1 - COST) - 1
            mk = df[f"mkt_{key}"].astype(float)
            ok = np.isfinite(r.to_numpy()) & np.isfinite(mk.to_numpy())
            r2, mk2 = r[ok], mk[ok]
            if len(r2) < 30:
                continue
            yrs_arr = np.asarray(main_dates, dtype=object)[df.loc[ok, "p"].to_numpy()]
            yr = pd.Series([str(x)[:4] for x in yrs_arr], index=r2.index)
            rec[key] = {
                "n": int(len(r2)),
                "avg_pct": round(float(r2.mean()) * 100, 3),
                "excess_pct": round(float((r2 - mk2).mean()) * 100, 3),
                "win_pct": round(float((r2 > 0).mean()) * 100, 1),
                "median_pct": round(float(r2.median()) * 100, 3),
                "by_year": {str(y): round(float(v.mean()) * 100, 2)
                            for y, v in r2.groupby(yr)},
            }
        out["events"][k] = rec

    keys = list(EXITS)
    print("\n%-20s %6s | %s" % ("事件", "信号数", "  ".join("%7s" % k for k in keys)))
    order = sorted(out["events"].items(), key=lambda kv: -kv[1].get("o1", {}).get("avg_pct", -99))
    for k, v in order:
        g = lambda key: v.get(key, {}).get("avg_pct", float("nan"))
        print("%-20s %6d | %s" % (
            k, v["n_signals"], "  ".join("%7.3f" % g(x) for x in keys)))

    print("\n超额(减同期等权, %):")
    print("%-20s %s" % ("事件", "  ".join("%7s" % k for k in keys)))
    for k, v in order:
        g = lambda key: v.get(key, {}).get("excess_pct", float("nan"))
        print("%-20s %s" % (k, "  ".join("%7.3f" % g(x) for x in keys)))

    print("\n年化等价(满仓天天做, 按持有交易日数折算):")
    for k, v in order:
        vals = []
        for key in keys:
            x = v.get(key, {}).get("avg_pct")
            h = EXITS[key][2]
            if x is None:
                vals.append("    nan")
                continue
            n_turn = 252.0 if h == 0 else 252.0 / h
            vals.append("%7.1f" % (((1 + x / 100.0) ** n_turn - 1) * 100))
        print("%-20s %s" % (k, "  ".join(vals)))

    print("\n分年 o1 单笔均值(%):")
    yrs = sorted({y for v in out["events"].values() for y in v.get("o1", {}).get("by_year", {})})
    print("%-20s %s" % ("事件", " ".join("%7s" % y for y in yrs)))
    for k, v in order:
        yr = v.get("o1", {}).get("by_year", {})
        print("%-20s %s" % (k, " ".join("%7.2f" % yr.get(y, float("nan")) for y in yrs)))

    mo = out["events"].get("main_official", {})
    if "o1" in mo:
        print("\n[口径自检] main_official o1 单笔 %.3f%% (主策略官方 1.435%%), n=%d"
              % (mo["o1"]["avg_pct"], mo["o1"]["n"]))

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "exp_sat_alpha2.json")
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1)
    print("\nsaved", p)


if __name__ == "__main__":
    main()
