# -*- coding: utf-8 -*-
"""PIT财务因子在dip已执行样本上的分层检验(第四十一轮, 事前注册)。

方法: 直接复用 output/backtest.json 的500日回测已执行交易(signal_date=T,
T+1买入, T+2卖出; 与每日线上回测口径完全一致)逐笔取"T日已公告的最新一期
财报"(公告日<=T), 按因子分层的已结算收益。2500/1500日全市场面板在本沙盒
OOM(exit 137), 故不重跑长窗口; 样本=128笔(当前策略实际全部已执行历史)。

事前注册口径(不调参):
  主检验: 各因子五分位(全样本分位, 仅描述用; 128笔只作苗头级判定)
  固定桶: rev_yoy按(-inf,-20),[-20,-10),[-10,0),[0,20),[20,inf)——对应
         第三十一轮观察级苗头③"营收同比-10~0区间h2较好(n=2)"
  一致性: h1(<20240901)与h2(>=20240901)的分层方向须同号才算苗头存活
判定: 样本级检验功效有限, 结果只用于确认/证伪观察级苗头, 不接入执行层。
"""
from __future__ import annotations

import sys
sys.path.insert(0, "/workspace")

import pandas as pd

from src.finance_data import load_finance, pit_lookup, pit_row

ERA = 20240901


def _bucket_rev(v: float) -> str:
    if v < -20:
        return "<-20"
    if v < -10:
        return "[-20,-10)"
    if v < 0:
        return "[-10,0)"
    if v < 20:
        return "[0,20)"
    return ">=20"


def _layer(df: pd.DataFrame, key: str, order: list[str] | None = None) -> None:
    g = df.groupby(key)["ret"]
    rows = [(k, g.get_group(k).mean(), (g.get_group(k) > 0).mean(), len(g.get_group(k)))
            for k in (order or sorted(g.groups))]
    line = "  ".join(f"{k}:{m:+.2f}%(w{w:.0%},n{n})" for k, m, w, n in rows if n >= 3)
    print(f"  {key}: {line or '样本不足'}")
    for era, lbl in ((False, "h1"), (True, "h2")):
        sub = df[(df["signal_date"].astype(int) >= ERA) == era]
        if len(sub) < 10:
            continue
        gg = sub.groupby(key)["ret"]
        seg = "  ".join(f"{k}:{gg.get_group(k).mean():+.2f}(n{len(gg.get_group(k))})"
                        for k in (order or sorted(gg.groups)) if len(gg.get_group(k)) >= 3)
        print(f"    {lbl}: {seg}")


def main() -> None:
    print("== 1/3 读取500日回测已执行交易(output/backtest.json) ==")
    import json
    from pathlib import Path
    bt = json.loads((Path("/workspace/output/backtest.json")).read_text())
    trades = [t for t in bt["trades_executed_recent"] if t.get("ret_pct") is not None]
    print(f"已执行 {len(trades)} 笔")

    print("== 2/3 逐笔取T日已公告最新财报(PIT) ==")
    pit = pit_lookup(load_finance())
    rows, miss = [], 0
    for t in trades:
        r = pit_row(pit, t["code"], t["signal_date"])
        if r is None:
            miss += 1
            continue
        rows.append({"signal_date": t["signal_date"], "code": t["code"], "ret": t["ret_pct"],
                     "rev_yoy": r.get("rev_yoy"), "np_yoy": r.get("np_yoy"),
                     "roe": r.get("roe"), "gross_margin": r.get("gross_margin"),
                     "rev_bucket": _bucket_rev(r["rev_yoy"]) if pd.notna(r.get("rev_yoy")) else "NA"})
    df = pd.DataFrame(rows)
    print(f"有PIT财务 {len(df)} 笔(缺财务 {miss} 笔, 多为2017Q1前信号)")

    print("== 3/3 因子分层(均值%/胜率/n) ==")
    for f in ["rev_yoy", "np_yoy", "roe", "gross_margin"]:
        sub = df[pd.to_numeric(df[f], errors="coerce").notna()].copy()
        sub[f] = pd.to_numeric(sub[f])
        if len(sub) < 30:
            print(f"  {f}: 样本不足({len(sub)})")
            continue
        sub["q"] = pd.qcut(sub[f], 5, labels=False, duplicates="drop")
        _layer(sub, "q")
    print("营收同比固定桶(苗头③验证):")
    _layer(df, "rev_bucket", order=["<-20", "[-20,-10)", "[-10,0)", "[0,20)", ">=20"])


if __name__ == "__main__":
    main()
