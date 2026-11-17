"""第246轮 B: 全交易日候选面板 —— 只为回答「补位能不能不只做破位日」

245 只在 198 个「主策略空仓 + 大盘破位」日交易, 11 年里资金利用率约 30%。
剩下 1740 个空仓日全在空转。要把它们用起来, 先得有一张覆盖全部交易日的
「距 250 日高 + 次日缺口 + 两天后收益」表。

用宽表(日期 x 股票)做 rolling, 比 groupby+rolling 快一个量级。
"""
from __future__ import annotations

import glob
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "panel"
OUT = ROOT / "output"
COLS = ["ts_code", "trade_date", "open", "high", "low", "close", "pre_close",
        "pct", "amount", "factor", "turnover_rate", "is_paused", "is_st",
        "high_limit", "low_limit"]
WINS = (60, 90, 120, 150, 180, 200, 250, 300, 360)


def main() -> None:
    files = sorted(PANEL.glob("*.csv.gz"))
    files = [f for f in files if int(f.stem[:4]) >= 2014]
    print(f"读 {len(files)} 日面板", flush=True)
    t0 = time.time()
    parts = []
    for f in files:
        parts.append(pd.read_csv(f, usecols=COLS,
                                 dtype={"ts_code": "str", "trade_date": "int32"}))
    d = pd.concat(parts, ignore_index=True)
    del parts
    for c in ("open", "high", "low", "close", "pre_close", "pct", "amount",
              "factor", "turnover_rate", "high_limit", "low_limit"):
        d[c] = pd.to_numeric(d[c], errors="coerce").astype("float32")
    print(f"  {len(d):,} 行  {time.time()-t0:.0f}s", flush=True)

    d["adj_c"] = d["close"] * d["factor"]
    d["adj_h"] = d["high"] * d["factor"]
    d["adj_o"] = d["open"] * d["factor"]
    # ★ 先过滤股票池再 merge, 把待 merge 的行数压到最小(否则两次 merge 会 OOM)
    d = d[(d["is_paused"] == 0) & (d["is_st"] == 0)]
    d = d[d["pct"].between(-11.5, 11.5)]
    d = d[d["ts_code"].str[:2].isin(["60", "00"])]
    d = d[["ts_code", "trade_date", "adj_c", "adj_h", "adj_o", "pct",
           "amount", "turnover_rate"]]
    print(f"  过滤后 {len(d):,} 行", flush=True)

    # ---- 宽表 rolling: 前 250 个交易日(不含当日)的最高复权价 ----
    t0 = time.time()
    wide = d.pivot_table(index="trade_date", columns="ts_code", values="adj_h",
                         aggfunc="last").sort_index()
    print(f"  宽表 {wide.shape}  {time.time()-t0:.0f}s", flush=True)
    t0 = time.time()
    ws = wide.shift(1)
    cols = {}
    for W in WINS:
        cols[f"dh{W}"] = ws.rolling(W, min_periods=max(W // 3, 20)).max()
    print(f"  rolling {len(WINS)} 个窗口 {time.time()-t0:.0f}s", flush=True)
    dm = pd.concat([v.stack().rename(k) for k, v in cols.items()], axis=1).reset_index()
    dm["trade_date"] = dm["trade_date"].astype("int32")
    print(f"  高点长表 {len(dm):,}", flush=True)
    del wide, ws, cols
    parts = []
    for _yr, g in d.groupby(d.trade_date // 10000, sort=True):
        parts.append(g.merge(dm, on=["ts_code", "trade_date"], how="left"))
    d = pd.concat(parts, ignore_index=True)
    del parts, dm
    for W in WINS:
        d[f"dh{W}"] = (d["adj_c"] / d[f"dh{W}"] - 1) * 100
    d.sort_values(["ts_code", "trade_date"], inplace=True)
    d["listed"] = d.groupby("ts_code").cumcount() + 1

    # ---- 标签: T+1 开买, T+2 收卖; T+1 缺口 ----
    d = d.sort_values(["ts_code", "trade_date"])
    g = d.groupby("ts_code", sort=False)
    d["n1_o"] = g["adj_o"].shift(-1)
    d["n1_c"] = g["adj_c"].shift(-1)
    d["n2_c"] = g["adj_c"].shift(-2)
    d["y"] = (d["n2_c"] / d["n1_o"] - 1) * 100
    d["y_1d"] = (d["n1_c"] / d["n1_o"] - 1) * 100
    d["n1_gap"] = (d["n1_o"] / d["adj_c"] - 1) * 100

    keep = (["ts_code", "trade_date", "listed", "n1_gap", "n1_c", "y", "y_1d",
             "pct", "amount", "turnover_rate", "adj_c", "n1_o"]
            + [f"dh{W}" for W in WINS])
    r = d[keep].copy()
    r["amt_yi"] = r["amount"] / 1e5
    r["circ_mv"] = r["amount"] / r["turnover_rate"].replace(0, np.nan) / 1e6
    r = r.drop(columns=["amount", "turnover_rate"])
    for c in r.columns:
        if r[c].dtype == "float64":
            r[c] = r[c].astype("float32")
    r.to_parquet(OUT / "w246b_all.parquet", index=False, compression="zstd")
    print(f"完成 {len(r):,} 行 x {len(r.columns)} 列 -> w246b_all.parquet", flush=True)
    print(r.head(3).to_string())


if __name__ == "__main__":
    main()
