# -*- coding: utf-8 -*-
"""构建全量"日线价表"缓存: 每只股票每天的复权开/收、涨跌停价、量、换手、停牌/ST 标记。

后续所有赢家普查(龙虎榜/涨停/超跌)都复用这份价表, 避免重复读 4108 个 csv.gz。
输出: output/price_table_212.parquet
"""
from __future__ import annotations

import glob
import os
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "panel"
OUT = ROOT / "output"
COLS = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close", "pct",
        "vol", "amount", "turnover_rate", "factor", "is_paused", "is_st",
        "high_limit", "low_limit"]


def main() -> None:
    files = sorted(glob.glob(str(PANEL / "*.csv.gz")))
    lim = int(os.environ.get("PN_LIMIT", "0"))
    if lim:
        files = files[:lim]
    print(f"{len(files)} 天", flush=True)

    parts = []
    for i, f in enumerate(files):
        d = pd.read_csv(f, usecols=COLS, dtype={"ts_code": "string"})
        for c in ("open", "high", "low", "close", "pre_close", "pct", "vol",
                  "amount", "turnover_rate", "factor", "high_limit", "low_limit"):
            d[c] = pd.to_numeric(d[c], errors="coerce").astype("float32")
        d["is_paused"] = d["is_paused"].fillna(1).astype("int8")
        d["is_st"] = d["is_st"].fillna(1).astype("int8")
        d["adj_open"] = (d["open"] * d["factor"]).astype("float32")
        d["adj_close"] = (d["close"] * d["factor"]).astype("float32")
        parts.append(d[["ts_code", "trade_date", "open", "close", "adj_open",
                        "adj_close", "high_limit", "low_limit", "pct", "vol",
                        "amount", "turnover_rate", "is_paused", "is_st"]])
        if (i + 1) % 500 == 0:
            print(f"  {i+1}/{len(files)}", flush=True)

    df = pd.concat(parts, ignore_index=True)
    df.to_parquet(OUT / "price_table_212.parquet", index=False)
    print("rows:", len(df), " codes:", df["ts_code"].nunique(),
          " days:", df["trade_date"].nunique())
    print(df.dtypes.to_string())


if __name__ == "__main__":
    main()
