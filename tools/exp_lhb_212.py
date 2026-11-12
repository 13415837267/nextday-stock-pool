# -*- coding: utf-8 -*-
"""第 212 轮 · 赢家普查(右侧新 alpha 源): 龙虎榜上榜股 T+1买T+2卖 次日溢价。

龙虎榜于 T 日盘后公布, T+1 开盘前可得, 用于 T+1 买入决策无未来函数。
口径: T+1 开盘买(复权) -> T+2 收盘卖(复权), 扣双边成本 0.30%。
分层: 净买入方向 / 净买入占比 / 是否涨停上榜 / 龙虎榜成交占比。
输出: output/lhb12_212.parquet + output/lhb12_summary_212.csv
"""
from __future__ import annotations

import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
COST = 0.30


def code2ts(code: str) -> str:
    c = str(code).zfill(6)
    if c.startswith(("600", "601", "603", "605", "688", "689")):
        return f"{c}.SH"
    if c.startswith(("000", "001", "002", "003", "300", "301")):
        return f"{c}.SZ"
    return None


def main() -> None:
    lhb = pd.read_parquet(ROOT / "data" / "lhb_hist.parquet")
    lhb["ts_code"] = lhb["code"].map(code2ts)
    lhb = lhb[lhb["ts_code"].notna()].copy()
    lhb["trade_date"] = lhb["date"].astype("int64")
    print(f"龙虎榜 {len(lhb)} 条, 日期 {lhb['date'].min()} -> {lhb['date'].max()}")

    price = pd.read_parquet(OUT / "price_table_212.parquet")
    price = price.sort_values(["ts_code", "trade_date"], kind="mergesort")
    g = price.groupby("ts_code", sort=False)
    price["n1_adj_open"] = g["adj_open"].shift(-1)
    price["n2_adj_close"] = g["adj_close"].shift(-2)
    price["n1_d"] = g["trade_date"].shift(-1)
    price["n2_d"] = g["trade_date"].shift(-2)
    price["n1_paused"] = g["is_paused"].shift(-1).fillna(1)
    price["n1_open"] = g["open"].shift(-1)
    price["n1_hl"] = g["high_limit"].shift(-1)

    # 龙虎榜特征
    lhb["net_yi"] = lhb["net_buy"] / 1e8          # 净买入(亿)
    lhb["lu"] = (lhb["pct"] >= 9.7).astype("int8")  # 涨停上榜
    lhb["ld"] = (lhb["pct"] <= -9.7).astype("int8")  # 跌停上榜

    m = lhb.merge(price[["ts_code", "trade_date", "n1_adj_open", "n2_adj_close",
                         "n1_d", "n2_d", "n1_paused", "n1_open", "n1_hl"]],
                  on=["ts_code", "trade_date"], how="inner")
    # n1_adj_open / n2_adj_close 已由 groupby.shift(-1)/(-2) 保证是"下一/下二交易日",
    # 非空即代表存在 T+1、T+2 交易日数据(跨周末/节假日自动正确)。
    m = m.dropna(subset=["n1_adj_open", "n2_adj_close"])
    m = m[m["n1_paused"] == 0]                      # T+1 未停牌
    m = m[(m["n1_open"] < m["n1_hl"] - 1e-6)]       # T+1 非一字涨停(买不进)

    m["ret12"] = (m["n2_adj_close"] / m["n1_adj_open"] - 1) * 100
    m["ret12_net"] = m["ret12"] - COST

    print(f"对齐后样本 {len(m)}")
    cols = ["ts_code", "name", "date", "pct", "net_yi", "net_ratio", "amt_ratio",
            "turn", "circ_mv", "lu", "ld", "ret12", "ret12_net"]
    out = m[cols].copy()
    out.to_parquet(OUT / "lhb12_212.parquet", index=False)

    def seg(df: pd.DataFrame) -> dict:
        return {"n": len(df),
                "win%": round(float((df["ret12_net"] > 0).mean()) * 100, 2),
                "mean%": round(float(df["ret12_net"].mean()), 3),
                "med%": round(float(df["ret12_net"].median()), 3)}

    rows = []
    rows.append({"bucket": "全部龙虎榜", **seg(out)})
    rows.append({"bucket": "净买入>0", **seg(out[out["net_yi"] > 0])})
    rows.append({"bucket": "净买入<=0", **seg(out[out["net_yi"] <= 0])})
    rows.append({"bucket": "净买入占比>3%", **seg(out[out["net_ratio"] > 3])})
    rows.append({"bucket": "净买入占比>6%", **seg(out[out["net_ratio"] > 6])})
    rows.append({"bucket": "净买入占比>10%", **seg(out[out["net_ratio"] > 10])})
    rows.append({"bucket": "涨停上榜", **seg(out[out["lu"] == 1])})
    rows.append({"bucket": "涨停上榜·净买入>0", **seg(out[(out["lu"] == 1) & (out["net_yi"] > 0)])})
    rows.append({"bucket": "涨停上榜·净买入占比>6%", **seg(out[(out["lu"] == 1) & (out["net_ratio"] > 6)])})
    rows.append({"bucket": "跌停上榜", **seg(out[out["ld"] == 1])})
    rows.append({"bucket": "非涨停跌停(换手/振幅上榜)", **seg(out[(out["lu"] == 0) & (out["ld"] == 0)])})

    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "lhb12_summary_212.csv", index=False)
    print(summ.to_string(index=False))

    # 涨停上榜·净买入>0 分年
    sub = out[(out["lu"] == 1) & (out["net_yi"] > 0)]
    sub["year"] = sub["date"].str[:4]
    yr = sub.groupby("year").apply(
        lambda x: pd.Series({"n": len(x), "win%": (x["ret12_net"] > 0).mean() * 100,
                             "mean%": x["ret12_net"].mean()}), include_groups=False).round(2)
    print("=== 涨停上榜·净买入>0 分年 ===")
    print(yr.to_string())


if __name__ == "__main__":
    main()
