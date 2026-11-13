#!/usr/bin/env python3
"""第222轮 探针: ETF 是否是可转债(T+0)的合法替代品。

可转债权限作废后, 用户手上剩下的"能日内反复进出"的合法工具只有 T+0 ETF:
  债券ETF 511xxx / 货币ETF 511xxx / 黄金ETF 518xxx / 跨境ETF 513xxx(沪) 159xxx(深)
股票ETF 51xxxx/159xxx 是 T+1, 不能做日内, 本探针也一并统计以作对照。

要回答三件事:
  1. 能不能拿到全历史日线(含已清盘)? 覆盖多少年?
  2. 各类 ETF 的日内振幅有多大? (振幅是日内策略的原材料)
  3. 单次往返成本量级 (ETF 免印花税, 只有佣金+滑点)
"""
from __future__ import annotations

import sys
import time

import akshare as ak
import pandas as pd

pd.set_option("display.width", 200)


def try_call(name, fn, *a, **kw):
    t0 = time.time()
    try:
        d = fn(*a, **kw)
        print(f"[OK]   {name}: shape={getattr(d, 'shape', None)}  {time.time()-t0:.1f}s")
        return d
    except Exception as e:  # noqa: BLE001
        print(f"[FAIL] {name}: {type(e).__name__}: {str(e)[:120]}")
        return None


print("=" * 70)
print("1) ETF 列表")
print("=" * 70)
spot = try_call("fund_etf_spot_em", ak.fund_etf_spot_em)
if spot is not None:
    print(spot.columns.tolist())
    print(spot.head(3).to_string())

# 分类
def cls(code: str) -> str:
    c = str(code).zfill(6)
    if c.startswith("511"):
        return "债券/货币"
    if c.startswith("518"):
        return "黄金"
    if c.startswith("513") or c.startswith("1599") or c.startswith("1595"):
        return "跨境"
    if c.startswith("159") or c.startswith("51") or c.startswith("56") or c.startswith("58"):
        return "股票ETF"
    return "其他"


print()
print("=" * 70)
print("2) 单只历史日线 (含早期年份), 检验可取性")
print("=" * 70)
samples = {
    "510300": "沪深300(股票ETF,T+1)",
    "511010": "国债ETF(T+0)",
    "518880": "黄金ETF(T+0)",
    "513100": "纳指ETF(跨境,T+0)",
    "159920": "恒生ETF(跨境,T+0)",
    "511990": "华宝添益(货币,T+0)",
}
for code, desc in samples.items():
    d = try_call(f"hist {code} {desc}", ak.fund_etf_hist_em,
                 symbol=code, period="daily",
                 start_date="20040101", end_date="20261113", adjust="")
    if d is not None and len(d):
        print(f"        {code}: {d['日期'].min()} ~ {d['日期'].max()}  n={len(d)}")
        print(f"        首行: {d.iloc[0].to_dict()}")
