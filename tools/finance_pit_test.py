# -*- coding: utf-8 -*-
"""PIT财务因子在dip已执行样本上的分层检验(第四十一轮注册, 第四十二轮扩窗口)。

方法: 复用回测已执行交易(signal_date=T, T+1买入, T+2卖出; 与每日线上回测
口径完全一致)逐笔取"T日已公告的最新一期财报"(公告日<=T), 按因子分层的
已结算收益。

用法: python3 -m tools.finance_pit_test [trades_json]
  缺省读取 output/backtest.json(500日基准, 128笔);
  传 output/backtest_long.json 则读十年(2500日)全量成交 trades_all(~498笔,
  第四十二轮 float32+懒加载改造后沙盒可跑, 见 tools/backtest_long.py)。

事前注册口径(不调参):
  主检验: 各因子五分位(全样本分位, 仅描述用; 苗头级判定)
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
    src = sys.argv[1] if len(sys.argv) > 1 else "/workspace/output/backtest.json"
    print(f"== 1/3 读取回测已执行交易({src}) ==")
    import json
    from pathlib import Path
    bt = json.loads(Path(src).read_text())
    raw = bt.get("trades_all") or bt.get("trades_executed_recent") or []
    trades = [t for t in raw if t.get("ret_pct") is not None]
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
