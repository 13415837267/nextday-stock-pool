# -*- coding: utf-8 -*-
"""第 212 轮 · 从零重建 · 全样本面板: 每只股票每天一行, 特征齐全, 标签 = T+1买T+2卖。

标签 ret12_net = (T+2 收盘复权 / T+1 开盘复权 - 1)*100 - 0.30(双边成本)。
这是"先得到能赚钱的股票"的数据基础: 后续对所有特征做系统分箱扫描,
找出 ret12_net 显著为正的甜点区间, 再反推策略条件。
输出: output/panel_212.parquet
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
COLS = ["ts_code", "open", "high", "low", "close", "pre_close", "pct", "vol",
        "amount", "turnover_rate", "factor", "is_paused", "is_st",
        "high_limit", "low_limit"]
WARM = 260
CHUNK = 260
COST = 0.30


def board_ok(code: str) -> bool:
    c, s = code.split(".")
    if s == "SH":
        return not c.startswith("688")
    if s == "SZ":
        return not c.startswith("30")
    return False


def load_days(files: list[str], off: int = 0) -> pd.DataFrame:
    fr = []
    for i, f in enumerate(files):
        d = pd.read_csv(f, usecols=COLS, dtype={"ts_code": "string"})
        d["di"] = off + i
        fr.append(d)
    df = pd.concat(fr, ignore_index=True)
    for c in ("open", "high", "low", "close", "pre_close", "vol", "amount",
              "turnover_rate", "factor", "high_limit", "low_limit"):
        df[c] = pd.to_numeric(df[c], errors="coerce").astype("float32")
    df["is_paused"] = df["is_paused"].fillna(1).astype("int8")
    df["is_st"] = df["is_st"].fillna(1).astype("int8")
    return df


def main() -> None:
    files = sorted(glob.glob(str(PANEL / "*.csv.gz")))
    lim = int(os.environ.get("PN_LIMIT", "0"))
    if lim:
        files = files[:lim]
    days = [os.path.basename(f)[:8] for f in files]
    print(f"面板 {len(days)} 天: {days[0]} → {days[-1]}", flush=True)

    first: dict[str, int] = {}
    for i, f in enumerate(files):
        for c in pd.read_csv(f, usecols=["ts_code"], dtype={"ts_code": "string"})["ts_code"].unique():
            if c not in first:
                first[c] = i

    keep: list[pd.DataFrame] = []
    start = 0
    while start < len(files):
        lo = max(0, start - WARM)
        hi = min(len(files), start + CHUNK)
        df = load_days(files[lo:hi], off=lo)
        df = df.sort_values(["ts_code", "di"], kind="mergesort")
        g = df.groupby("ts_code", sort=False)

        adj_c = (df["close"] * df["factor"]).astype("float64")
        adj_o = (df["open"] * df["factor"]).astype("float64")
        adj_h = (df["high"] * df["factor"]).astype("float64")
        adj_l = (df["low"] * df["factor"]).astype("float64")
        df["adj_c"], df["adj_o"], df["adj_h"], df["adj_l"] = adj_c, adj_o, adj_h, adj_l
        df["bars"] = df["di"] - df["ts_code"].map(first).astype("int32") + 1

        # 前瞻: T+1 开盘买, T+2 收盘卖
        df["n1_o"] = g["adj_o"].shift(-1)
        df["n2_c"] = g["adj_c"].shift(-2)
        df["n1_d"] = g["di"].shift(-1)
        df["n2_d"] = g["di"].shift(-2)
        df["n1_p"] = g["is_paused"].shift(-1).fillna(1)
        df["n1_open"] = g["open"].shift(-1)
        df["n1_hl"] = g["high_limit"].shift(-1)
        df["n1_yizi"] = (df["n1_open"] >= df["n1_hl"] - 1e-6).astype("int8")
        df["gap1"] = (df["n1_o"] / adj_c - 1) * 100
        df["ret12"] = (df["n2_c"] / df["n1_o"] - 1) * 100
        df["ret12_net"] = df["ret12"] - COST

        def rsum(col: str, n: int, shift: int = 0) -> pd.Series:
            s = g[col].shift(shift) if shift else df[col]
            acc = s.astype("float64")
            for k in range(1, n):
                acc = acc + g[col].shift(k + shift)
            return acc

        def rmin(col: str, n: int, shift: int = 0) -> pd.Series:
            s = g[col].shift(shift) if shift else df[col]
            acc = s.astype("float64")
            for k in range(1, n):
                acc = np.minimum(acc, g[col].shift(k + shift))
            return pd.Series(acc, index=df.index)

        def rmax(col: str, n: int, shift: int = 0) -> pd.Series:
            s = g[col].shift(shift) if shift else df[col]
            acc = s.astype("float64")
            for k in range(1, n):
                acc = np.maximum(acc, g[col].shift(k + shift))
            return pd.Series(acc, index=df.index)

        ma5, ma20, ma60 = (rsum("adj_c", 5) / 5, rsum("adj_c", 20) / 20,
                           rsum("adj_c", 60) / 60)
        low20 = rmin("adj_l", 20)
        ph19 = rmax("adj_h", 19, shift=1)
        ph60 = rmax("adj_h", 60, shift=1)
        ph250 = rmax("adj_h", 249, shift=1)
        volma20 = rsum("vol", 20, shift=1) / 20
        volma5 = rsum("vol", 5, shift=1) / 5
        amtma20 = rsum("amount", 20, shift=1) / 20
        to3 = rsum("turnover_rate", 3) / 3
        to20 = rsum("turnover_rate", 20) / 20

        df["mom3"] = (adj_c / g["adj_c"].shift(3) - 1) * 100
        df["mom5"] = (adj_c / g["adj_c"].shift(5) - 1) * 100
        df["mom10"] = (adj_c / g["adj_c"].shift(10) - 1) * 100
        df["mom20"] = (adj_c / g["adj_c"].shift(20) - 1) * 100
        df["mom60"] = (adj_c / g["adj_c"].shift(60) - 1) * 100
        df["bias5"] = (adj_c / ma5 - 1) * 100
        df["bias20"] = (adj_c / ma20 - 1) * 100
        df["dist_low20"] = (adj_c / low20 - 1) * 100
        df["dist_ph20"] = (adj_c / ph19 - 1) * 100
        df["dist_ph60"] = (adj_c / ph60 - 1) * 100
        df["dist_ph250"] = (adj_c / ph250 - 1) * 100
        df["ma20_slope"] = (ma20 / ma20.groupby(df["ts_code"], sort=False).shift(5) - 1) * 100
        df["bull"] = ((ma5 > ma20) & (ma20 > ma60)).astype("int8")
        df["close_pos"] = ((df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan))
        df["amp_pct"] = (df["high"] - df["low"]) / df["pre_close"] * 100
        df["low_tail"] = (np.minimum(df["open"], df["close"]) - df["low"]) / df["pre_close"] * 100
        df["body_pct"] = (df["close"] - df["open"]) / df["pre_close"] * 100
        df["vr"] = df["vol"] / volma20.replace(0, np.nan)
        df["vr5"] = df["vol"] / volma5.replace(0, np.nan)
        df["amt_yi"] = df["amount"] / 1e8
        df["amt_ratio"] = df["amount"] / amtma20.replace(0, np.nan)
        df["turnover_cool"] = to3 / to20.replace(0, np.nan)
        df["prev_pct"] = g["pct"].shift(1)
        df["prev2_pct"] = g["pct"].shift(2)

        up = (df["pct"] > 0).astype("int8")
        blk = (~(df["pct"] > 0)).groupby(df["ts_code"], sort=False).cumsum()
        df["up_streak"] = up.groupby([df["ts_code"], blk], sort=False).cumsum()
        lu = (df["pct"] >= 9.7).astype("float64")
        acc = lu.copy()
        for k in range(1, 5):
            acc = acc + g["pct"].shift(k).ge(9.7).astype("float64")
        df["lu5"] = acc
        df["limit_up"] = lu.astype("int8")
        dn = (df["pct"] <= -9.7).astype("float64")
        df["limit_dn"] = dn.astype("int8")

        m1 = rsum("pct", 20) / 20
        acc2 = df["pct"].astype("float64") ** 2
        for k in range(1, 20):
            acc2 = acc2 + g["pct"].shift(k).astype("float64") ** 2
        var = acc2 / 20 - m1 ** 2
        df["vol20"] = np.sqrt(var.clip(lower=0))

        bmask = (df["ts_code"].map(board_ok).values & (df["is_paused"] == 0).values)
        mkt_day = df.loc[bmask].groupby("di")["pct"].mean()
        df["mkt_ret"] = df["di"].map(mkt_day).astype("float32")

        # 可交易样本(与回测口径一致)
        snap = df[df["di"] >= start].copy()
        m = snap["ts_code"].map(board_ok).values
        s = snap[m]
        s = s[(s["is_paused"] == 0) & (s["is_st"] == 0)]
        s = s[(s["close"] >= 3) & (s["close"] <= 150)]
        s = s[(s["amt_yi"] >= 0.8) & s["turnover_rate"].between(2, 12)]
        s = s[s["bars"] >= 150]
        # T+1 可买(非停牌、非一字涨停), T+2 有效
        s = s[(s["n1_d"] == s["di"] + 1) & (s["n1_p"] == 0) & (s["n1_yizi"] == 0)]
        s = s[(s["n2_d"] == s["di"] + 2)]
        if len(s):
            s = s.assign(d=[days[int(i)] for i in s["di"]])
            cols = ["d", "ts_code", "pct", "close_pos", "amp_pct", "low_tail", "body_pct",
                    "mom3", "mom5", "mom10", "mom20", "mom60", "bias5", "bias20",
                    "dist_low20", "dist_ph20", "dist_ph60", "dist_ph250",
                    "ma20_slope", "bull", "vr", "vr5", "amt_yi", "amt_ratio",
                    "turnover_rate", "turnover_cool", "up_streak", "lu5", "limit_up",
                    "limit_dn", "prev_pct", "prev2_pct", "vol20", "mkt_ret",
                    "gap1", "ret12", "ret12_net"]
            out = s[cols].copy()
            for c in out.columns:
                if out[c].dtype == "float64":
                    out[c] = out[c].astype("float32")
            keep.append(out)
        start = hi
        print(f"  {start}/{len(files)} 累计样本 {sum(len(x) for x in keep):,}", flush=True)

    p = pd.concat(keep, ignore_index=True)
    p.to_parquet(OUT / "panel_212.parquet", index=False)
    print("总样本", len(p), " 交易日", p["d"].nunique(),
          " 股票", p["ts_code"].nunique())
    print("ret12_net 描述:", p["ret12_net"].describe().round(3).to_dict())


if __name__ == "__main__":
    main()
