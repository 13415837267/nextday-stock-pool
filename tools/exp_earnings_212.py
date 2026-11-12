# -*- coding: utf-8 -*-
"""第 212 轮 · 换数据源探查: 财报事件驱动(公告日)的 T+1买T+2卖 收益。

本地唯一尚未系统挖掘的非价量维度是季度财报(含 ann_date 公告日 + np_yoy 净利同比)。
验证: 财报公告后 T+1 开盘买、T+2 收盘卖, 按业绩增速分层, 看"业绩超预期/暴雷"有无事件 alpha。
"""
from __future__ import annotations

import glob
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
COST = 0.30


def main() -> None:
    fs = sorted(glob.glob(str(ROOT / "data/finance/*.csv.gz")))
    fin = pd.concat([pd.read_csv(f, compression="gzip",
                                 usecols=["ts_code", "ann_date", "np_yoy", "rev_yoy", "roe"])
                     for f in fs])
    fin["ann"] = pd.to_datetime(fin["ann_date"]).dt.strftime("%Y%m%d").astype("int64")
    fin["np_yoy_c"] = fin["np_yoy"].clip(-200, 200)
    fin = fin.dropna(subset=["ann"])
    print(f"财报 {len(fin)} 条, 公告日 {fin['ann'].min()} -> {fin['ann'].max()}")

    price = pd.read_parquet(OUT / "price_table_212.parquet")
    price = price.sort_values(["ts_code", "trade_date"], kind="mergesort")
    g = price.groupby("ts_code", sort=False)
    price["n1_o"] = g["adj_open"].shift(-1)
    price["n2_c"] = g["adj_close"].shift(-2)
    price["n1_p"] = g["is_paused"].shift(-1).fillna(1)
    price["n1_open"] = g["open"].shift(-1)
    price["n1_hl"] = g["high_limit"].shift(-1)

    # 财报公告日 -> 对齐到 <=ann 的最近交易日(公告日当天), 用 merge_asof
    fin["ts_code"] = fin["ts_code"].astype("string")
    price["ts_code"] = price["ts_code"].astype("string")
    price = price.sort_values("trade_date")
    m = pd.merge_asof(
        fin.sort_values("ann"),
        price[["ts_code", "trade_date", "n1_o", "n2_c", "n1_p", "n1_open", "n1_hl"]],
        left_on="ann", right_on="trade_date", by="ts_code", direction="backward")
    m = m.dropna(subset=["n1_o", "n2_c"])
    m = m[m["n1_p"] == 0]
    m = m[(m["n1_open"] < m["n1_hl"] - 1e-6)]
    m["ret12"] = (m["n2_c"] / m["n1_o"] - 1) * 100
    m["ret12_net"] = m["ret12"] - COST
    print(f"对齐样本 {len(m)}")

    def seg(df: pd.DataFrame) -> dict:
        return {"n": len(df), "mean%": round(df["ret12_net"].mean(), 3),
                "med%": round(df["ret12_net"].median(), 3),
                "win%": round((df["ret12_net"] > 0).mean() * 100, 2)}

    rows = [
        {"bucket": "全部公告", **seg(m)},
        {"bucket": "净利暴雷(<-100%)", **seg(m[m["np_yoy_c"] < -100])},
        {"bucket": "净利大降(-100~-50)", **seg(m[m["np_yoy_c"].between(-100, -50)])},
        {"bucket": "净利略降(-50~0)", **seg(m[m["np_yoy_c"].between(-50, 0)])},
        {"bucket": "净利稳增(0~50)", **seg(m[m["np_yoy_c"].between(0, 50)])},
        {"bucket": "净利高增(50~100)", **seg(m[m["np_yoy_c"].between(50, 100)])},
        {"bucket": "净利暴增(>100%)", **seg(m[m["np_yoy_c"] > 100])},
        {"bucket": "净利暴增+营收正增", **seg(m[(m["np_yoy_c"] > 100) & (m["rev_yoy"] > 0)])},
        {"bucket": "扭亏(np_yoy>100且rev正)", **seg(m[(m["np_yoy_c"] > 100) & (m["rev_yoy"] > 0)])},
    ]
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
