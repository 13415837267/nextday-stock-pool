# -*- coding: utf-8 -*-
"""第一百七十一轮: 超跌域 × 执行过滤 × 持有期 的细化验证。

背景
----
exp_sat_alpha2 的横评里, 三个"超跌"事件在 **T+3 收盘卖(c2, 持有 2 日)** 出现明显正收益:
  star_dip(科创板 -8~-4%) c2 单笔 +0.587% / 年化等价 109%
  gem_dip (创业板  -6~-3%) c2 单笔 +0.419% / 年化等价  69%
  main_dip(主板    -6~-3%) c2 单笔 +0.239% / 年化等价  35%
而所有事件在 o1(T+2 开盘卖)都是负的 —— 与主策略"T+2 开盘卖最优"矛盾。
矛盾点必须查清: 是"这些事件需要更久才反弹", 还是"横评口径漏了什么"。

同时横评暴露了一个更大的问题: 主策略官方 top1 在 o1 只有 -0.137%/笔, 而官方报 1.435%。
差的 1.57pp 只能来自**执行过滤**(buy_gap_below=0 即只在低开时买 + 评分门槛 + 池深降档),
即"不买"本身就是 alpha 的一半。本脚本把这条过滤搬到事件上重测。

测试矩阵
--------
域 × 缺口过滤 × 每日取前 k × 卖出时点
  域: 主板 / 创业板 / 科创板 (全部 pct -6~-3, 缩量, 收盘弱, 非ST非停牌, 成交额>=1亿)
  缺口过滤: 无 / gap<=0(主策略条件)
  k: 全买(看信号质量) / 每日 1 只(看容量)
  卖出: o1(T+2 开盘) / c1(T+2 收盘) / c2(T+3 收盘) / c3(T+4 收盘)

用法: python tools/exp_sat_dip.py [天窗口, 默认 1500]
输出: output/exp_sat_dip.json
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
COST = 0.003
EXITS = {"o1": (2, "open", 1), "c1": (2, "close", 1), "c2": (3, "close", 2), "c3": (4, "close", 3)}
BOARDS = {"主板": (-6, -3), "创业板": (-6, -3), "科创板": (-8, -4)}

COLS = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
        "pct", "vol", "amount", "high_limit", "turnover_rate", "is_paused", "is_st",
        "open_adj", "close_adj", "vol_ratio", "close_pos", "ivol20", "bars", "mom20"]


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    cfg = load_config()
    ctx = prepare_context(cfg, days)
    main_dates, by_date, reg = ctx["main_dates"], ctx["by_date"], ctx["reg"]
    if not isinstance(reg.index[0], str):
        reg = reg.copy()
        reg.index = reg.index.astype(str)
    lvl = np.array([float(reg.loc[d, "mkt_idx"]) if d in reg.index else np.nan for d in main_dates])
    gate = np.array([bool(reg.loc[d, "trade_ok"]) if d in reg.index else False for d in main_dates])
    pos_of = {d: i for i, d in enumerate(main_dates)}
    N = len(main_dates)

    rows = []
    for i, d in enumerate(main_dates):
        if not gate[i]:
            continue
        day = by_date.get(d)
        if day is None:
            continue
        b = ((day["is_paused"] == 0) & (day["is_st"] == 0) & (day["bars"] >= 250)
             & (day["amount"] >= 1e8) & (day["ivol20"] > 0))
        bd = day["ts_code"].map(board_of)
        for name, (lo_, hi_) in BOARDS.items():
            m = b & (bd == name) & day["pct"].between(lo_, hi_) & (day["vol_ratio"] < 0.8) \
                & (day["close_pos"] < 0.4)
            sub = day.loc[m]
            if not len(sub):
                continue
            sub = sub.sort_values("mom20")            # 跌得最深的排第一
            for rank in (0, 1, 2, 3):
                if len(sub) <= rank:
                    continue
                rows.append({"d": d, "p": i, "board": name, "rank": rank,
                             "code": str(sub.iloc[rank]["ts_code"]),
                             "mom20": float(sub.iloc[rank]["mom20"])})
    sig = pd.DataFrame(rows)
    print(f"[sig] 信号 {len(sig):,} 行 (闸门开启且池内有票的日子)")

    # 取未来价格
    panel = pd.concat([by_date[d][COLS] for d in main_dates if d in by_date], ignore_index=True)
    panel["p"] = panel["trade_date"].map(pos_of)
    panel = panel[panel["p"].notna()].reset_index(drop=True)
    key = panel.set_index(["p", "ts_code"])

    nxt = key.reindex(pd.MultiIndex.from_arrays([panel["p"] + 1, panel["ts_code"]]))
    buy = nxt["open_adj"].to_numpy()
    raw_open = nxt["open"].to_numpy()
    pre = nxt["pre_close"].to_numpy()
    hilim = nxt["high_limit"].to_numpy()
    vol1 = nxt["vol"].to_numpy()
    panel["buy_open"] = buy
    panel["buyable"] = (np.isfinite(buy.astype(float)) & (vol1.astype(float) > 0)
                        & (raw_open < hilim - 1e-9))
    panel["gap"] = (raw_open / np.where(pre.astype(float) > 0, pre, np.nan) - 1) * 100
    for k, (off, kind, h) in EXITS.items():
        a = key.reindex(pd.MultiIndex.from_arrays([panel["p"] + off, panel["ts_code"]]))
        panel[f"px_{k}"] = (a["open_adj"] if kind == "open" else a["close_adj"]).to_numpy()
        i1 = np.clip(panel["p"].to_numpy() + 1, 0, N - 1)
        i2 = np.clip(panel["p"].to_numpy() + off, 0, N - 1)
        panel[f"mkt_{k}"] = lvl[i2] / lvl[i1] - 1

    m = sig.merge(panel[["p", "ts_code", "buy_open", "buyable", "gap",
                         *[f"px_{k}" for k in EXITS],
                         *[f"mkt_{k}" for k in EXITS]]].rename(columns={"ts_code": "code"}),
                  on=["p", "code"], how="left")
    m = m[m["buyable"].fillna(False)]
    print(f"[fill] 可成交 {len(m):,} 行, 平均缺口 {m['gap'].mean():.3f}%")

    out = {"window": {"days": N, "start": main_dates[0], "end": main_dates[-1]}, "cells": {}}
    print("\n%-6s %-3s %-6s %6s | %s" % ("域", "k", "gap", "笔数", "  ".join("%8s" % x for x in EXITS)))
    for board in BOARDS:
        for topk in (0, 1):
            for gapf in ("全部", "gap<=0"):
                x = m[m["board"] == board]
                if topk:
                    x = x[x["rank"] < topk]
                if gapf == "gap<=0":
                    x = x[x["gap"] <= 0]
                if len(x) < 40:
                    continue
                rec = {"n": int(len(x))}
                line = []
                for k in EXITS:
                    r = (x[f"px_{k}"].astype(float) / x["buy_open"].astype(float)) * (1 - COST) - 1
                    mk = x[f"mkt_{k}"].astype(float)
                    ok = np.isfinite(r.to_numpy()) & np.isfinite(mk.to_numpy())
                    r2, mk2 = r[ok], mk[ok]
                    if len(r2) < 30:
                        line.append("%8s" % "nan")
                        continue
                    yrs = pd.Series([str(main_dates[p])[:4] for p in x.loc[ok, "p"].to_numpy()],
                                    index=r2.index)
                    rec[k] = {"n": int(len(r2)), "avg": round(float(r2.mean()) * 100, 3),
                              "ex": round(float((r2 - mk2).mean()) * 100, 3),
                              "win": round(float((r2 > 0).mean()) * 100, 1),
                              "by_year": {str(y): round(float(v.mean()) * 100, 2)
                                          for y, v in r2.groupby(yrs)}}
                    line.append("%8.3f" % (r2.mean() * 100))
                out["cells"][f"{board}|k{topk}|{gapf}"] = rec
                print("%-6s %-3d %-6s %6d | %s" % (board, topk, gapf, rec["n"], " ".join(line)))

    print("\n分年(c2, T+3 收盘卖):")
    yrs = sorted({y for v in out["cells"].values() for y in v.get("c2", {}).get("by_year", {})})
    print("%-24s %s" % ("单元", " ".join("%7s" % y for y in yrs)))
    for kk, v in out["cells"].items():
        if "c2" not in v:
            continue
        print("%-24s %s" % (kk, " ".join("%7.2f" % v["c2"]["by_year"].get(y, float("nan"))
                                         for y in yrs)))

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "exp_sat_dip.json")
    json.dump(out, open(p, "w"), ensure_ascii=False, indent=1)
    print("\nsaved", p)


if __name__ == "__main__":
    main()
