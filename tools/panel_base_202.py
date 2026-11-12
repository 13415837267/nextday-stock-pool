"""第 202 轮: 全面板基础率.

问题: 回测 3000 天只产生 717 笔, 而面板里有 1400 万个"股票日"。本轮把 dip 形态
放到全样本上算一遍, 看那 717 笔到底是"信号本身的 edge", 还是"评分从一堆里挑出来的"。

口径尽量与回测一致(见 config.yaml):
  票池: 主板 / 价格 3~150 / 成交额 >=0.8亿 / 换手 2~12% / 上市 >=150 根
  dip:  pct ∈ [-6,-3] / close_pos <= 0.32 / mom20 ∈ [5,25] / dist_low20 >= 6 /
        vol_ratio <= 4
  收益: T+1 开盘买 -> T+2 收盘卖, 价格用 close*factor 复权。
        T+1 停牌或开盘涨停(买不到)的行剔除。

输出:
  output/panel_dip_202.parquet  合格股票日(含特征与前瞻收益)
  output/panel_daily_202.csv    每日市场聚合(全主板等权次日收益)
"""
from __future__ import annotations

import glob
import os
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
PANEL = ROOT / "data" / "panel"
OUT = ROOT / "output"
COLS = ["ts_code", "open", "high", "low", "close", "pre_close", "pct", "vol",
        "amount", "turnover_rate", "factor", "is_paused", "is_st",
        "high_limit", "low_limit"]
WARM = 30          # 指标回看
CHUNK = 260        # 每块天数


def board_ok(code: str) -> bool:
    c, s = code.split(".")
    if s == "SH":
        return not c.startswith("688")
    if s == "SZ":
        return not c.startswith("30")
    return False


def load_days(files: list[str]) -> pd.DataFrame:
    fr = []
    for i, f in enumerate(files):
        d = pd.read_csv(f, usecols=COLS, dtype={"ts_code": "string"})
        d["di"] = i
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
    days = [os.path.basename(f)[:8] for f in files]
    print(f"面板 {len(days)} 天: {days[0]} → {days[-1]}", flush=True)

    keep: list[pd.DataFrame] = []
    daily: list[dict] = []
    start = 0
    while start < len(files):
        lo = max(0, start - WARM)
        hi = min(len(files), start + CHUNK)
        df = load_days(files[lo:hi])
        df = df.sort_values(["ts_code", "di"], kind="mergesort")
        g = df.groupby("ts_code", sort=False)

        adj_c = df["close"] * df["factor"]
        adj_o = df["open"] * df["factor"]
        adj_l = df["low"] * df["factor"]
        df["adj_c"] = adj_c
        df["ma20"] = adj_c.groupby(df["ts_code"], sort=False).transform(
            lambda s: s.rolling(20, min_periods=20).mean())
        df["low20"] = adj_l.groupby(df["ts_code"], sort=False).transform(
            lambda s: s.rolling(20, min_periods=20).min())
        df["volma20"] = df["vol"].groupby(df["ts_code"], sort=False).transform(
            lambda s: s.rolling(20, min_periods=20).mean())
        df["bars"] = g.cumcount() + 1

        # 前瞻: T+1 开盘, T+2 收盘(同一只票的下一行/下两行)
        for name, ser in [
            ("nxt1_o", adj_o), ("nxt1_o_raw", df["open"]),
            ("nxt1_p", df["is_paused"]), ("nxt1_hl", df["high_limit"]),
            ("nxt2_c", adj_c), ("nxt2_d", df["di"]), ("nxt1_d", df["di"]),
        ]:
            df[name] = ser.groupby(df["ts_code"], sort=False).shift(-1 if "1" in name[:5] else -2)
        df["nxt2_d"] = df["di"].groupby(df["ts_code"], sort=False).shift(-2)
        df["nxt1_d"] = df["di"].groupby(df["ts_code"], sort=False).shift(-1)
        df["ret"] = (df["nxt2_c"] / df["nxt1_o"] - 1) * 100
        # T+2 那天必须紧接 T+1(中间没有停牌跳过)
        df["ok_fwd"] = (df["nxt1_d"] == df["di"] + 1) & (df["nxt2_d"] == df["di"] + 2)

        df["mom20"] = (adj_c / adj_c.groupby(df["ts_code"], sort=False).shift(20) - 1) * 100
        df["bias20"] = (adj_c / df["ma20"] - 1) * 100
        df["dist_low20"] = (adj_c / df["low20"] - 1) * 100
        df["close_pos"] = ((df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan))
        df["vr"] = df["vol"] / df["volma20"]
        df["amt_yi"] = df["amount"] / 1e8

        # 当日快照(用于市场基准): 主板、可交易
        snap = df[(df["di"] >= start - lo)]
        mb = snap["ts_code"].map(board_ok).values
        base = snap[mb & (snap["is_paused"] == 0) & snap["ret"].notna()]
        if len(base):
            for d_i, sub in base.groupby("di"):
                day = days[lo + d_i] if lo + d_i < len(days) else None
                if day is None or day < days[start]:
                    continue
                daily.append({"d": day, "n": len(sub),
                              "ret": float(sub["ret"].mean()),
                              "up": float((sub["ret"] > 0).mean() * 100)})

        m = (snap["di"] >= start - lo)
        s = snap[m].copy()
        s = s[s["ts_code"].map(board_ok)]
        s = s[(s["is_paused"] == 0) & (s["is_st"] == 0)]
        s = s[(s["close"] >= 3) & (s["close"] <= 150)]
        s = s[(s["amt_yi"] >= 0.8) & s["turnover_rate"].between(2, 12)]
        s = s[(s["bars"] >= 150)]
        # dip 形态
        s = s[s["pct"].between(-6.0, -3.0)]
        s = s[s["close_pos"] <= 0.32]
        s = s[s["mom20"].between(5.0, 25.0)]
        s = s[s["dist_low20"] >= 6.0]
        s = s[s["vr"] <= 4.0]
        # 可成交 + 有前瞻
        s = s[s["ok_fwd"] & (s["nxt1_p"].fillna(1) == 0)]
        s = s[s["nxt1_o_raw"] < s["nxt1_hl"] - 1e-6]   # T+1 开盘一字涨停买不到
        s = s[s["ret"].notna()]
        if len(s):
            s["d"] = [days[lo + i] for i in s["di"].map(lambda x: int(x))]
            keep.append(s[["d", "ts_code", "pct", "close_pos", "mom20", "bias20",
                           "dist_low20", "vr", "amt_yi", "turnover_rate", "ret"]])
        print(f"  {days[start]} +{hi-start}d 合格 {0 if not len(s) else len(s)}", flush=True)
        start = hi

    allq = pd.concat(keep, ignore_index=True)
    allq.to_parquet(OUT / "panel_dip_202.parquet", index=False)
    pd.DataFrame(daily).drop_duplicates("d").to_csv(OUT / "panel_daily_202.csv", index=False)
    print(f"合格股票日 {len(allq)} 行, 覆盖 {allq.d.nunique()} 个交易日")
    print(allq["ret"].describe().round(3).to_string())


if __name__ == "__main__":
    main()
