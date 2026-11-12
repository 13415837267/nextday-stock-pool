# -*- coding: utf-8 -*-
"""第 212 轮 · 从零重建 · 赢家普查(右侧): 涨停股 T+1买T+2卖 次日溢价。

目标: 不预设条件, 先用全历史数据回答"什么样的涨停股次日能赚钱"。
口径(与实盘/回测引擎一致):
  - T 日收盘后选股(此处 = 当日收盘涨停的股票)
  - T+1 开盘价买入(复权), T+2 收盘价卖出(复权)
  - 扣双边成本: 佣金万2.5×2 + 印花税万5(卖) + 滑点千1×2 ≈ 0.30%/笔
  - 排除 T+1 停牌 / T+1 一字涨停(买不进) / T+2 无收盘
输出: output/lu12_212.parquet(逐样本) + output/lu12_summary_212.csv(分层汇总)
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
COST = 0.30  # 每笔双边成本(%)

# 涨停判定阈值: 用未复权 close 对照 high_limit(一字板 close==high==low)
EPS = 1e-6


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


def board_ok(code: str) -> bool:
    # 只做主板上限10cm的票(20cm的涨停次日逻辑不同, 先排除), 保留主板
    c, s = code.split(".")
    if s == "SH":
        return not c.startswith("688")
    if s == "SZ":
        return not c.startswith("30")
    return False


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
        df["adj_c"], df["adj_o"] = adj_c, adj_o
        df["bars"] = df["di"] - df["ts_code"].map(first).astype("int32") + 1

        # 涨停判定(未复权)
        df["limit_up"] = (df["close"] >= df["high_limit"] - EPS).astype("int8")
        df["yizi"] = ((df["open"] >= df["high_limit"] - EPS)
                      & (df["close"] >= df["high_limit"] - EPS)).astype("int8")

        # 连板数: 连续涨停天数(含今日)
        lu = df["limit_up"].astype("int8")
        brk = (lu == 0).groupby(df["ts_code"], sort=False).cumsum()
        df["streak"] = lu.groupby([df["ts_code"], brk], sort=False).cumsum()

        # 前瞻标签
        df["n1_o"] = g["adj_o"].shift(-1)
        df["n2_c"] = g["adj_c"].shift(-2)
        df["n1_d"] = g["di"].shift(-1)
        df["n2_d"] = g["di"].shift(-2)
        df["n1_p"] = g["is_paused"].shift(-1).fillna(1)
        df["n1_open"] = g["open"].shift(-1)
        df["n1_hl"] = g["high_limit"].shift(-1)
        df["n1_yizi"] = ((df["n1_open"] >= df["n1_hl"] - EPS)).astype("int8")
        df["gap1"] = (df["n1_o"] / adj_c - 1) * 100          # 次日开盘溢价
        df["ret12"] = (df["n2_c"] / df["n1_o"] - 1) * 100     # T+1开->T+2收
        df["ret12_net"] = df["ret12"] - COST

        # 涨停股特征
        df["volma20"] = df.groupby("ts_code", sort=False)["vol"].transform(
            lambda x: x.shift(1).rolling(20, min_periods=1).mean())
        df["vr"] = df["vol"] / df["volma20"].replace(0, np.nan)
        df["amt_yi"] = df["amount"] / 1e8
        df["body"] = (df["close"] - df["open"]) / df["pre_close"] * 100
        df["price"] = df["close"]

        # 只取"T日收盘涨停"且可交易的样本
        m = df["ts_code"].map(board_ok).values
        s = df[(df["limit_up"] == 1) & m & (df["is_st"] == 0)
               & (df["di"] >= start)].copy()
        # T+1 有效且非停牌、非一字涨停
        s = s[(s["n1_d"] == s["di"] + 1) & (s["n1_p"] == 0)
              & (s["n1_yizi"] == 0)]
        # T+2 有效
        s = s[(s["n2_d"] == s["di"] + 2)]
        if len(s):
            s = s.assign(d=[days[int(i)] for i in s["di"]])
            cols = ["d", "ts_code", "price", "pct", "limit_up", "yizi", "streak",
                    "vr", "turnover_rate", "amt_yi", "body", "bars",
                    "gap1", "ret12", "ret12_net"]
            out = s[cols].copy()
            for c in out.columns:
                if out[c].dtype == "float64":
                    out[c] = out[c].astype("float32")
            keep.append(out)
        start = hi
        print(f"  {start}/{len(files)} 涨停样本累计 {sum(len(x) for x in keep):,}", flush=True)

    p = pd.concat(keep, ignore_index=True)
    p.to_parquet(OUT / "lu12_212.parquet", index=False)
    print("=== 总涨停样本 ===", len(p), " 交易日", p["d"].nunique())

    def seg(df: pd.DataFrame) -> dict:
        return {"n": len(df),
                "win%": round(float((df["ret12_net"] > 0).mean()) * 100, 2),
                "mean%": round(float(df["ret12_net"].mean()), 3),
                "med%": round(float(df["ret12_net"].median()), 3),
                "gap1%": round(float(df["gap1"].mean()), 3)}

    rows = []
    rows.append({"bucket": "全部涨停", **seg(p)})
    # 按连板数
    for k, lab in [(1, "首板"), (2, "二板"), (3, "三板")]:
        rows.append({"bucket": lab, **seg(p[p["streak"] == k])})
    rows.append({"bucket": "四板+", **seg(p[p["streak"] >= 4])})
    # 封板形态
    rows.append({"bucket": "一字板", **seg(p[p["yizi"] == 1])})
    rows.append({"bucket": "非一字(实体/T字)", **seg(p[p["yizi"] == 0])})
    # 量比
    rows.append({"bucket": "量比<2", **seg(p[p["vr"] < 2])})
    rows.append({"bucket": "量比2~4", **seg(p[p["vr"].between(2, 4)])})
    rows.append({"bucket": "量比>4", **seg(p[p["vr"] > 4])})
    # 换手
    rows.append({"bucket": "换手<5%", **seg(p[p["turnover_rate"] < 5])})
    rows.append({"bucket": "换手5~15%", **seg(p[p["turnover_rate"].between(5, 15)])})
    rows.append({"bucket": "换手>15%", **seg(p[p["turnover_rate"] > 15])})
    # 首板 × 量比
    f1 = p[p["streak"] == 1]
    rows.append({"bucket": "首板·量比<2", **seg(f1[f1["vr"] < 2])})
    rows.append({"bucket": "首板·量比2~4", **seg(f1[f1["vr"].between(2, 4)])})
    rows.append({"bucket": "首板·量比>4", **seg(f1[f1["vr"] > 4])})

    summ = pd.DataFrame(rows)
    summ.to_csv(OUT / "lu12_summary_212.csv", index=False)
    print(summ.to_string(index=False))

    # 分年(看稳定性)
    p["year"] = p["d"].str[:4]
    yr = p.groupby("year").apply(
        lambda x: pd.Series({"n": len(x), "win%": (x["ret12_net"] > 0).mean() * 100,
                             "mean%": x["ret12_net"].mean()}), include_groups=False).round(2)
    print("=== 全部涨停 分年 ===")
    print(yr.to_string())


if __name__ == "__main__":
    main()
