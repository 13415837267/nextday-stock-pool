# -*- coding: utf-8 -*-
"""第一百零四轮审计(A): 出场端 —— "该不该多拿一天"?

现行出场: T+2 开盘卖 50% + 收盘卖 50%, 无论 T+1 是涨停还是跌停, 规则完全一样。
但 A 股有明确的"涨停次日惯性"与"跌停次日补跌"结构。若 T+1 收盘的浮盈/浮亏本身
就携带了 T+3 的方向信息, 那么"按 T+1 结果决定是否多拿一天"是一条干净的规则 ——
它只用到 T+2 开盘前已知的信息(买入价、T+1 收盘价), 无未来函数。

本脚本不改参数, 逐笔计算并分组:
  t1_ret  = close(T+1)/buy_px - 1        买入当天到收盘的浮动盈亏(已知)
  r_open  = open(T+2)/buy_px - 1         卖出一刻的浮动盈亏(已知)
  intra   = close(T+2)/open(T+2) - 1     当日日内(现行分批要吃的那一段)
  next_o  = open(T+3)/open(T+2) - 1      再多拿一天的增量(开盘对开盘)
  next_c  = open(T+3)/close(T+2) - 1     收盘接次开
用法: python3 tools/exit_hold_audit.py [--days 1500]
"""
from __future__ import annotations
import os
os.environ.setdefault("BT_FULL_TRADES", "1")

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

OUT = ROOT / "output" / "exit_hold.json"
CSV = ROOT / "output" / "exit_hold_trades.csv"


def enrich(trades, ctx):
    per = ctx["per_code"]
    rows = []
    for t in trades:
        if t.get("ret_pct") is None or t.get("pool") != "main":
            continue
        code, sd = t["code"], t.get("sell_date")
        sub = per.get(code)
        if sub is None or sd not in sub.index:
            continue
        pos = sub.index.get_loc(sd)
        if pos + 1 >= len(sub):
            continue
        r2, r3 = sub.iloc[pos], sub.iloc[pos + 1]
        if float(r2["open"]) <= 0 or float(r2["close"]) <= 0 or float(r3["open"]) <= 0:
            continue
        buy_px = float(t["buy_px"])
        if buy_px <= 0:
            continue
        # T+1 收盘: 买入日那张日线(卖出日的前一根)
        r1 = sub.iloc[pos - 1] if pos >= 1 else None
        rows.append({
            "signal_date": t["signal_date"], "buy_date": t["buy_date"], "sell_date": sd,
            "year": sd[:4], "code": code, "ret": float(t["ret_pct"]),
            "score": t.get("score"), "pool_n": t.get("pool_n"), "gap": t.get("gap_pct"),
            "r_open": round((float(r2["open"]) / buy_px - 1) * 100, 3),
            "t1_ret": (round((float(r1["close"]) / buy_px - 1) * 100, 3) if r1 is not None else np.nan),
            "intra": round((float(r2["close"]) / float(r2["open"]) - 1) * 100, 3),
            "next_o": round((float(r3["open"]) / float(r2["open"]) - 1) * 100, 3),
            "next_c": round((float(r3["open"]) / float(r2["close"]) - 1) * 100, 3),
        })
    return pd.DataFrame(rows)


def st(g, col):
    v = g[col].values
    return pd.Series({"n": len(g), "avg": round(float(v.mean()), 3),
                      "wr": round(float((v > 0).mean() * 100), 1),
                      "med": round(float(np.median(v)), 3)})


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    args = ap.parse_args()
    cfg = load_config()
    t0 = time.time()
    ctx = prepare_context(cfg, args.days)
    res = run_backtest(cfg, args.days, ctx)
    m = res["metrics"]
    print(f"回测 {time.time()-t0:.0f}s: {m['总收益率%']}% / {m['最大回撤%']}% / {m['交易次数']} 笔", flush=True)
    d = enrich(res.get("trades_all") or [], ctx)
    print(f"可审计 {len(d)} 笔; intra {d.intra.mean():+.3f}% | next_o {d.next_o.mean():+.3f}% "
          f"| next_c {d.next_c.mean():+.3f}%", flush=True)
    d.to_csv(CSV, index=False, encoding="utf-8-sig")
    out = {"baseline": m, "n": len(d)}

    print("\n== A. 按【T+2 开盘浮动盈亏 r_open】分组 ==")
    d["b_ro"] = pd.cut(d.r_open, [-99, -5, -3, -2, -1, 0, 1, 3, 5, 99])
    for col in ["intra", "next_o", "next_c"]:
        g = d.groupby("b_ro", observed=True).apply(lambda x: st(x, col))
        print(f"-- {col}"); print(g.to_string()); print()

    print("\n== B. 按【T+1 收盘浮动盈亏 t1_ret】分组 ==")
    d["b_t1"] = pd.cut(d.t1_ret, [-99, -5, -3, -2, -1, 0, 1, 2, 3, 5, 99])
    for col in ["intra", "next_o", "next_c"]:
        g = d.groupby("b_t1", observed=True).apply(lambda x: st(x, col))
        print(f"-- {col}"); print(g.to_string()); print()

    print("\n== C. 高浮盈组的构成: r_open >= 3% ==")
    hi = d[d.r_open >= 3]
    print(f"n={len(hi)} 占比 {len(hi)/len(d)*100:.1f}%; next_o 均值 {hi.next_o.mean():+.3f}% "
          f"胜率 {(hi.next_o>0).mean()*100:.0f}%")
    print("分年:"); print(hi.groupby("year").next_o.agg(["count", "mean"]).round(2).to_string())

    print("\n== D. 低浮亏组: r_open <= -3% ==")
    lo = d[d.r_open <= -3]
    print(f"n={len(lo)}; next_o 均值 {lo.next_o.mean():+.3f}% 胜率 {(lo.next_o>0).mean()*100:.0f}%")

    print("\n== E. 分年: r_open>=3 的 next_o ==")
    yr = d.assign(hi=d.r_open >= 3).pivot_table(index="year", columns="hi", values="next_o",
                                                aggfunc=["count", "mean"])
    print(yr.round(2).to_string())

    print("\n== F. 涨停/接近涨停(T+1 收盘相对买入价 >=9%) ==")
    lu = d[d.t1_ret >= 9]
    if len(lu):
        print(f"n={len(lu)}; intra {lu.intra.mean():+.2f}% next_o {lu.next_o.mean():+.2f}% next_c {lu.next_c.mean():+.2f}%")
        print(lu[["sell_date", "code", "r_open", "t1_ret", "intra", "next_o", "next_c", "ret"]].to_string(index=False))

    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=str), encoding="utf-8")
    print(f"\n写入 {OUT} / {CSV}")


if __name__ == "__main__":
    main()
