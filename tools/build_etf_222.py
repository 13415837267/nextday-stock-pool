#!/usr/bin/env python3
"""第222轮: T+0 ETF 面板构建。

背景: 用户无可转债权限, 第221轮那条"空仓期做转债T+0"的帕累托路径作废。
      A股场内剩下的、无资金/权限门槛的 T+0 品种只有 ETF 的四类:
        跨境ETF(QDII) / 债券ETF / 货币ETF / 黄金ETF
      股票ETF 是 T+1, 拉进来只作对照。

数据: 新浪 fund_etf_category_sina 拿全列表, fund_etf_hist_sina 拿全历史日线。
      新浪给的是不复权价, 对"日内"研究正好(日内不需要复权, 分红只影响隔夜)。
      局限: 只有在市 927 只, 已清盘的不在列表里 —— 这是幸存者偏差, 必须记在结论里。

产物:
  data/etf/daily.parquet   原始 OHLCV
  data/etf/universe.csv    代码/名称/类别/上市日/行数
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import akshare as ak
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "data" / "etf"
OUT.mkdir(parents=True, exist_ok=True)

QDIY_KW = (
    "恒生", "纳指", "纳斯达克", "标普", "日经", "东证", "德国", "法国", "英国",
    "东南亚", "中概", "港股", "香港", "海外", "全球", "美国", "道琼斯", "亚太",
    "沙特", "印度", "越南", "韩国", "新兴", "中华", "H股", "中韩", "中日",
)
MONEY_KW = ("货币", "日利", "添益", "快线", "现金", "保证金", "理财")


def classify(code: str, name: str) -> str:
    """按代码前缀+名称判定品种与 T+0 属性。"""
    c = code[2:] if code[:2] in ("sh", "sz") else code
    n = str(name)
    if c.startswith("511"):
        return "money" if any(k in n for k in MONEY_KW) else "bond"
    if c.startswith("518"):
        return "gold"
    if c.startswith("513"):
        return "qdii"
    if any(k in n for k in QDIY_KW):
        return "qdii"
    if c.startswith(("51", "56", "58", "159", "16")):
        return "stock"
    return "other"


T0 = {"money", "bond", "gold", "qdii"}


def main() -> None:
    lst = ak.fund_etf_category_sina(symbol="ETF基金")
    lst = lst.rename(columns={"代码": "symbol", "名称": "name"})
    lst = lst[["symbol", "name"]].drop_duplicates("symbol")
    lst["code"] = lst["symbol"].str[2:]
    lst["cls"] = [classify(s, n) for s, n in zip(lst["symbol"], lst["name"])]
    lst["t0"] = lst["cls"].isin(T0)
    print(f"列表 {len(lst)} 只, 分布:\n{lst['cls'].value_counts().to_string()}")

    frames, rows = [], []
    t0 = time.time()
    for i, r in enumerate(lst.itertuples(index=False), 1):
        if i % 50 == 0:
            print(f"  {i}/{len(lst)}  {time.time()-t0:.0f}s  rows={sum(len(x) for x in frames)}", flush=True)
        try:
            d = ak.fund_etf_hist_sina(symbol=r.symbol)
        except Exception:  # noqa: BLE001
            continue
        if d is None or len(d) < 30:
            continue
        d = d[["date", "open", "high", "low", "close", "volume", "amount"]].copy()
        d["code"] = r.code
        d["cls"] = r.cls
        frames.append(d)
        rows.append({"code": r.code, "symbol": r.symbol, "name": r.name, "cls": r.cls,
                     "t0": r.t0, "start": str(d["date"].iloc[0]), "end": str(d["date"].iloc[-1]),
                     "n": len(d)})

    if not frames:
        sys.exit("全部抓取失败")
    df = pd.concat(frames, ignore_index=True)
    df["date"] = pd.to_datetime(df["date"])
    for c in ("open", "high", "low", "close", "volume", "amount"):
        df[c] = pd.to_numeric(df[c], errors="coerce")
    df = df.dropna(subset=["open", "close"]).query("open>0 and close>0")
    df = df.sort_values(["code", "date"]).reset_index(drop=True)

    df.to_parquet(OUT / "daily.parquet", index=False)
    uni = pd.DataFrame(rows)
    uni.to_csv(OUT / "universe.csv", index=False, encoding="utf-8-sig")

    print(f"\n落盘 {len(df):,} 行 / {df['code'].nunique()} 只 / {df['date'].min().date()} ~ {df['date'].max().date()}")
    print(uni.groupby("cls").agg(只数=("code", "size"), 中位行数=("n", "median"),
                                 最早=("start", "min"), 总行数=("n", "sum")).to_string())


if __name__ == "__main__":
    main()
