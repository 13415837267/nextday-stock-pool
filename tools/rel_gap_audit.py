# -*- coding: utf-8 -*-
"""第一百零三轮审计: 把"缺口/跌幅"拆成市场成分与个股成分。

策略论点: 深低开是"错杀"的直接证据 -> 修复概率高。
但"深低开"里有一部分是**大盘整体低开**带来的(beta), 那不是我选的这只票被错杀,
而是所有票一起被打 —— 两者的修复含义不同。同理, T 日的回调跌幅里也有一部分
是大盘一起跌的成分。

本脚本不改任何参数, 只做逐笔画像:
  1) 市场竞价缺口 = 当日全市场(停牌/无成交剔除) open/pre_close-1 的中位数;
     个股超额缺口 = 个股缺口 - 市场缺口。
  2) 市场 T 日涨跌 = signal_date 全市场 pct 中位数; 个股超额跌幅 = 个股 pct - 市场 pct。
  3) 按超额缺口 / 超额跌幅分桶, 看单笔均值/胜率/复利是否单调;
     并按年拆开看是否 6/7 年同向(与第一百零一轮"缺口"的判据同源)。
用法: python3 tools/rel_gap_audit.py [--days 1500]
输出: 控制台 + output/rel_gap_audit.json
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

OUT = ROOT / "output" / "rel_gap_audit.json"


def market_stats(ctx: dict) -> tuple[pd.Series, pd.Series]:
    """每日市场竞价缺口(中位数)与市场涨跌幅(中位数)。"""
    gaps, rets = {}, {}
    for d, df in ctx["by_date"].items():
        ok = (df["is_paused"] == 0) & (df["vol"] > 0) & (df["pre_close"] > 0)
        sub = df.loc[ok, ["open", "pre_close", "pct"]]
        if sub.empty:
            continue
        gaps[d] = float((sub["open"] / sub["pre_close"] - 1).median() * 100)
        rets[d] = float(sub["pct"].median())
    return pd.Series(gaps), pd.Series(rets)


def enrich(trades: list[dict], mgap: pd.Series, mret: pd.Series, ctx: dict) -> pd.DataFrame:
    rows = []
    for t in trades:
        if t.get("ret_pct") is None:
            continue
        bd, sd, code = t["buy_date"], t["signal_date"], t["code"]
        df = ctx["by_date"].get(sd)
        spct = np.nan
        if df is not None:
            r = df.loc[df["ts_code"] == code, "pct"]
            if len(r):
                spct = float(r.iloc[0])
        rows.append({
            "signal_date": sd, "buy_date": bd, "year": bd[:4], "code": code,
            "pool": t.get("pool"), "ret": float(t["ret_pct"]),
            "gap": t.get("gap_pct"), "score": t.get("score"), "pool_n": t.get("pool_n"),
            "mkt_gap": float(mgap.get(bd, np.nan)), "mkt_ret": float(mret.get(sd, np.nan)),
            "stock_ret": spct,
        })
    df = pd.DataFrame(rows)
    df["res_gap"] = df["gap"] - df["mkt_gap"]
    df["res_ret"] = df["stock_ret"] - df["mkt_ret"]
    return df


def buck(vals: np.ndarray, edges: list[float]) -> list[str]:
    lab = []
    for v in vals:
        s = f"<{edges[0]:g}"
        for i, e in enumerate(edges):
            if v <= e:
                s = f"<{e:g}" if i == 0 else f"{edges[i-1]:g}~{e:g}"
                break
            s = f">={e:g}"
        lab.append(s)
    return lab


def stat(g: pd.DataFrame) -> dict:
    if g.empty:
        return {"n": 0}
    rets = g["ret"].values
    comp = float(np.prod(1 + rets / 100) - 1) * 100
    return {"n": int(len(g)), "avg": round(float(rets.mean()), 3),
            "wr": round(float((rets > 0).mean() * 100), 1), "comp": round(comp, 1),
            "med": round(float(np.median(rets)), 3)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    args = ap.parse_args()
    cfg = load_config()
    t0 = time.time()
    ctx = prepare_context(cfg, args.days)
    res = run_backtest(cfg, args.days, ctx)
    print(f"回测完成 {time.time()-t0:.0f}s, 成交 {res['metrics']['交易次数']} 笔, "
          f"{res['metrics']['总收益率%']}% / {res['metrics']['最大回撤%']}%", flush=True)
    mgap, mret = market_stats(ctx)
    df = enrich(res.get("trades_all") or [], mgap, mret, ctx)
    main_df = df[df["pool"] == "main"].copy()
    print(f"\n主池成交 {len(main_df)} 笔; 相关系数: gap~mkt_gap={df['gap'].corr(df['mkt_gap']):.2f} "
          f"| stock_ret~mkt_ret={df['stock_ret'].corr(df['mkt_ret']):.2f}", flush=True)

    out = {"baseline": res["metrics"], "trades": int(len(main_df))}
    print("\n== A. 按【原始竞价缺口】分桶(现行口径) ==")
    main_df["b_gap"] = buck(main_df["gap"].values, [-2, -1, 0, 0.5, 2])
    rows = []
    for k, g in main_df.groupby("b_gap"):
        s = stat(g)
        rows.append({"bucket": k, **s})
        print(f"  {k:>8s}: n={s['n']:>4d} 均值{s['avg']:+.2f}% 胜率{s['wr']:.0f}% 复利{s['comp']:+.0f}%")
    out["by_gap"] = rows

    print("\n== B. 按【超额缺口 = 个股缺口 - 市场缺口】分桶 ==")
    main_df["b_rgap"] = buck(main_df["res_gap"].values, [-2, -1, 0, 1, 2])
    rows = []
    for k, g in main_df.groupby("b_rgap"):
        s = stat(g)
        rows.append({"bucket": k, **s})
        print(f"  {k:>8s}: n={s['n']:>4d} 均值{s['avg']:+.2f}% 胜率{s['wr']:.0f}% 复利{s['comp']:+.0f}%")
    out["by_res_gap"] = rows

    print("\n== C. 按【超额跌幅 = 个股T日涨跌 - 市场T日涨跌】分桶 ==")
    main_df["b_rret"] = buck(main_df["res_ret"].values, [-4, -2, 0, 2])
    rows = []
    for k, g in main_df.groupby("b_rret"):
        s = stat(g)
        rows.append({"bucket": k, **s})
        print(f"  {k:>8s}: n={s['n']:>4d} 均值{s['avg']:+.2f}% 胜率{s['wr']:.0f}% 复利{s['comp']:+.0f}%")
    out["by_res_ret"] = rows

    print("\n== D. 二维交叉: 超额缺口 x 原始缺口 ==")
    ct = main_df.pivot_table(index="b_rgap", columns="b_gap", values="ret", aggfunc=["count", "mean"])
    print(ct.round(2).to_string())
    out["cross"] = json.loads(ct.round(3).to_json(force_ascii=False))

    print("\n== E. 超额缺口 分年(均值%/笔数) ==")
    yr = main_df.pivot_table(index="year", columns="b_rgap", values="ret", aggfunc=["count", "mean"])
    print(yr.round(2).to_string())
    out["by_year_res_gap"] = json.loads(yr.round(3).to_json(force_ascii=False))

    print("\n== F. 超额跌幅 分年(均值%/笔数) ==")
    yr2 = main_df.pivot_table(index="year", columns="b_rret", values="ret", aggfunc=["count", "mean"])
    print(yr2.round(2).to_string())
    out["by_year_res_ret"] = json.loads(yr2.round(3).to_json(force_ascii=False))

    print("\n== G. 市场缺口本身的分布(买入日) ==")
    print(main_df["mkt_gap"].describe().round(2).to_string())
    out["mkt_gap_desc"] = json.loads(main_df["mkt_gap"].describe().round(3).to_json())

    # 逐笔明细落盘(供二次分析/交叉检验, 不入库)
    df.to_csv(ROOT / "output" / "rel_gap_trades.csv", index=False)
    OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写入 {OUT} 与 output/rel_gap_trades.csv")


if __name__ == "__main__":
    main()
