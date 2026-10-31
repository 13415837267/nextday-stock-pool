# -*- coding: utf-8 -*-
"""第一百零四轮审计: 出场端的"市场语境" —— T+2 开盘该不该等收盘?

现行规则 sell_close_if_open_drop=2.0 是一刀切: T+2 开盘相对昨收跌 >=2% 就延卖到收盘。
但第一百零三轮在**买入端**刚刚证明: 同样的跌幅, "大盘一起跌"(beta) 与 "个股独自崩"
(idio) 的后续走势完全相反。出场端理应对称:
  · 市场也在跌 -> 这是 beta 拖累, 日内大概率随大盘修复 -> 值得等收盘;
  · 市场平稳/上涨而它独跌 -> 真利空 -> 开盘就该走, 等下去是接刀。

本脚本不改任何参数, 只做逐笔画像。边际量 = T+2 日内收益 (close/open - 1):
  > 0 表示"等到收盘"更好, < 0 表示"开盘走"更好。
分组: T+2 市场竞价缺口(中位数)、个股超额缺口 = 个股缺口 - 市场缺口。
用法: python3 tools/exit_context_audit.py [--days 1500]
输出: 控制台 + output/exit_context.json + output/exit_context_trades.csv
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

OUT = ROOT / "output" / "exit_context.json"
CSV = ROOT / "output" / "exit_context_trades.csv"


def market_gaps(ctx: dict) -> pd.Series:
    gaps = {}
    for d, df in ctx["by_date"].items():
        ok = (df["is_paused"] == 0) & (df["vol"] > 0) & (df["pre_close"] > 0)
        sub = df.loc[ok, ["open", "pre_close"]]
        if sub.empty:
            continue
        gaps[d] = float((sub["open"] / sub["pre_close"] - 1).median() * 100)
    return pd.Series(gaps)


def enrich(trades: list[dict], mgap: pd.Series, ctx: dict) -> pd.DataFrame:
    rows = []
    for t in trades:
        if t.get("ret_pct") is None or t.get("pool") != "main":
            continue
        sd = t.get("sell_date")
        df = ctx["by_date"].get(sd)
        if df is None:
            continue
        r = df.loc[df["ts_code"] == t["code"]]
        if not len(r):
            continue
        r = r.iloc[0]
        if float(r["pre_close"]) <= 0 or float(r["open"]) <= 0:
            continue
        sgap = (float(r["open"]) / float(r["pre_close"]) - 1) * 100
        intra = (float(r["close_adj"]) / float(r["open_adj"]) - 1) * 100
        rows.append({
            "signal_date": t["signal_date"], "buy_date": t["buy_date"], "sell_date": sd,
            "year": sd[:4], "code": t["code"], "ret": float(t["ret_pct"]),
            "gap": t.get("gap_pct"), "score": t.get("score"), "pool_n": t.get("pool_n"),
            "s_gap": round(sgap, 3), "m_gap": round(float(mgap.get(sd, np.nan)), 3),
            "intra": round(intra, 3),
            "s_ret": round(float(r["pct"]), 3) if "pct" in r else np.nan,
        })
    d = pd.DataFrame(rows)
    d["r_gap"] = d["s_gap"] - d["m_gap"]
    return d


def stat(g: pd.DataFrame, col: str = "intra") -> dict:
    if g.empty:
        return {"n": 0}
    v = g[col].values
    return {"n": int(len(g)), "avg": round(float(v.mean()), 3),
            "wr": round(float((v > 0).mean() * 100), 1),
            "med": round(float(np.median(v)), 3)}


def buck(vals, edges):
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    args = ap.parse_args()
    cfg = load_config()
    t0 = time.time()
    ctx = prepare_context(cfg, args.days)
    res = run_backtest(cfg, args.days, ctx)
    m = res["metrics"]
    print(f"回测 {time.time()-t0:.0f}s: {m['总收益率%']}% / {m['最大回撤%']}% / {m['交易次数']} 笔",
          flush=True)
    mg = market_gaps(ctx)
    df = enrich(res.get("trades_all") or [], mg, ctx)
    print(f"可审计成交 {len(df)} 笔; 日内均值 {df['intra'].mean():+.3f}% "
          f"胜率 {(df['intra']>0).mean()*100:.1f}%", flush=True)

    out = {"baseline": m, "n": int(len(df)), "intra_avg": round(float(df["intra"].mean()), 3)}
    df.to_csv(CSV, index=False, encoding="utf-8-sig")

    print("\n== A. 按【T+2 市场竞价缺口】分桶: 日内收益(收盘/开盘-1) ==")
    df["b_mg"] = buck(df["m_gap"].values, [-1.0, -0.5, -0.2, 0.2, 0.5])
    rows = []
    for k, g in df.groupby("b_mg", sort=False):
        s = stat(g)
        rows.append({"bucket": k, **s})
        print(f"  {k:>10s}: n={s['n']:>4d} 日内{s['avg']:+.3f}% 阳线率{s['wr']:.0f}% 中位{s['med']:+.3f}%")
    out["by_mkt_gap"] = rows

    print("\n== B. 按【个股超额缺口】分桶 ==")
    df["b_rg"] = buck(df["r_gap"].values, [-2.0, -1.0, -0.5, 0.5, 1.0])
    rows = []
    for k, g in df.groupby("b_rg", sort=False):
        s = stat(g)
        rows.append({"bucket": k, **s})
        print(f"  {k:>10s}: n={s['n']:>4d} 日内{s['avg']:+.3f}% 阳线率{s['wr']:.0f}% 中位{s['med']:+.3f}%")
    out["by_res_gap"] = rows

    print("\n== C. 交叉: 行=市场缺口 列=超额缺口 (日内均值/笔数) ==")
    ct = df.pivot_table(index="b_mg", columns="b_rg", values="intra", aggfunc=["count", "mean"])
    print(ct.round(2).to_string())
    out["cross"] = json.loads(ct.round(3).to_json(force_ascii=False))

    print("\n== D. 现行规则的目标人群: 个股缺口 <= -2% 的成交 ==")
    hit = df[df["s_gap"] <= -2.0].copy()
    hit["b_mg2"] = buck(hit["m_gap"].values, [-0.5, -0.2, 0.2])
    rows = []
    for k, g in hit.groupby("b_mg2", sort=False):
        s = stat(g)
        rows.append({"bucket": k, **s})
        print(f"  {k:>8s}: n={s['n']:>3d} 日内{s['avg']:+.3f}% 阳线率{s['wr']:.0f}%")
    out["scd_hits"] = rows
    print(f"  合计 {len(hit)} 笔, 日内 {hit['intra'].mean():+.3f}%")

    print("\n== E. 分年交叉(市场缺口 x 超额缺口, 日内均值) ==")
    for y, g in df.groupby("year"):
        p = g.pivot_table(index="b_mg", columns="b_rg", values="intra", aggfunc="mean")
        print(f"  -- {y} (n={len(g)}) --")
        print(p.round(2).to_string())

    print("\n== F. 市场缺口分年的稳定性(<= -0.5 vs 其余) ==")
    df["dn"] = df["m_gap"] <= -0.5
    yr = df.pivot_table(index="year", columns="dn", values="intra", aggfunc=["count", "mean"])
    print(yr.round(2).to_string())
    out["year_mkt_dn"] = json.loads(yr.round(3).to_json(force_ascii=False))

    print(f"\n写入 {OUT} / {CSV}")


if __name__ == "__main__":
    main()
