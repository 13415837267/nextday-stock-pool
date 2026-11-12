"""第 205 轮: 建一份"右侧(追高)"全史样本, 让数据给出追高条件, 而不是人先设条件.

与第 204 轮 dip 面板的区别:
  1. 样本 = 上涨日(pct >= PN_MIN_PCT), 不预设 close_pos / 形态门槛;
  2. 特征补齐右侧专用维度: 突破(收盘创 20/60 日新高)、量能突变(vr/vr5/amt_ratio)、
     连涨天数、5 日内涨停次数、距前高、均线多头排列、跳空缺口、当日超额;
  3. 标签 = T+1 开盘买、持有 1/3/5/10 个交易日后 T+H 收盘卖(与 satellite_backtest 一致),
     已扣双边成本; 并过滤 T+1 停牌与一字涨停(买不进)。

票池口径与回测一致: 主板 / 价 3~150 / 额 >=0.8 亿 / 换手 2~12% / 上市 >=150 根 /
非停牌非 ST。输出 output/panel_right_205.parquet(不入库, 可重建)。
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
MIN_PCT = float(os.environ.get("PN_MIN_PCT", "1.0"))
MAX_PCT = float(os.environ.get("PN_MAX_PCT", "11.5"))


# 第206轮: 板块集改为可配(默认仍是纯主板, 保持 205 轮结果可复现)
#   PN_BOARDS=main  -> 原口径(排除创业板/科创板/北交所)
#   PN_BOARDS=cyb   -> 主板 + 创业板(20cm)
PN_BOARDS = os.environ.get("PN_BOARDS", "main")


def board_ok(code: str) -> bool:
    c, s = code.split(".")
    if s == "SH":
        return True if PN_BOARDS == "cyb" else not c.startswith("688")
    if s == "SZ":
        return True if PN_BOARDS == "cyb" else not c.startswith("30")
    return False


def load_days(files: list[str], off: int = 0) -> pd.DataFrame:
    fr = []
    for i, f in enumerate(files):
        d = pd.read_csv(f, usecols=COLS, dtype={"ts_code": "string"})
        d["di"] = off + i                 # 全局日序, 不是块内序号
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
    print(f"面板 {len(days)} 天: {days[0]} → {days[-1]}, 样本 {MIN_PCT}% ~ {MAX_PCT}%",
          flush=True)

    # 每只票在面板中的首个全局日序 —— 上市天数必须用全局口径, 否则每块重新计数
    # 会把"每块前 150 天"整段丢掉(201 轮曾因此少了一半样本)
    first: dict[str, int] = {}
    for i, f in enumerate(files):
        for c in pd.read_csv(f, usecols=["ts_code"], dtype={"ts_code": "string"})["ts_code"].unique():
            if c not in first:
                first[c] = i
    print(f"票数 {len(first):,}", flush=True)

    keep: list[pd.DataFrame] = []
    mkt: list[dict] = []
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

        # ---- 前瞻: T+1 开盘买, 持有 h 日 ----
        hold = (1, 3, 5, 10)
        for h in hold:
            df[f"c{h}"] = g["adj_c"].shift(-h)
            d_h = g["di"].shift(-h)
            df[f"ok{h}"] = (d_h == df["di"] + h) & df[f"c{h}"].notna()
        df["n1_o"] = g["adj_o"].shift(-1)
        df["n1_d"] = g["di"].shift(-1)
        df["n1_o_raw"] = g["open"].shift(-1)
        df["n1_hl"] = g["high_limit"].shift(-1)
        df["n1_p"] = g["is_paused"].shift(-1).fillna(1)
        for h in hold:
            df[f"ret{h}"] = (df[f"c{h}"] / df["n1_o"] - 1) * 100
        df["gap"] = (df["n1_o"] / adj_c - 1) * 100

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

        ma5, ma10, ma20, ma60 = (rsum("adj_c", 5) / 5, rsum("adj_c", 10) / 10,
                                 rsum("adj_c", 20) / 20, rsum("adj_c", 60) / 60)
        low20 = rmin("adj_l", 20)
        # 前高: 不含今日
        ph19 = rmax("adj_h", 19, shift=1)          # 前 19 日最高(不含今)
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
        df["dist_ph20"] = (adj_c / ph19 - 1) * 100      # >=0 即创 20 日新高
        df["dist_ph60"] = (adj_c / ph60 - 1) * 100
        df["dist_ph250"] = (adj_c / ph250 - 1) * 100
        df["ma20_slope"] = (ma20 / ma20.groupby(df["ts_code"], sort=False).shift(5) - 1) * 100
        df["ma60_slope"] = (ma60 / ma60.groupby(df["ts_code"], sort=False).shift(10) - 1) * 100
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
        df["price"] = df["close"]

        # 连涨天数 / 5 日内涨停次数
        up = (df["pct"] > 0).astype("int8")
        blk = (~(df["pct"] > 0)).groupby(df["ts_code"], sort=False).cumsum()
        df["up_streak"] = up.groupby([df["ts_code"], blk], sort=False).cumsum()
        lu = (df["pct"] >= 9.7).astype("float64")
        acc = lu.copy()
        for k in range(1, 5):                       # 含今日的近 5 日涨停次数
            acc = acc + g["pct"].shift(k).ge(9.7).astype("float64")
        df["lu5"] = acc
        df["limit_up"] = lu.astype("int8")

        bmask = (df["ts_code"].map(board_ok).values & (df["is_paused"] == 0).values)
        mkt_day = df.loc[bmask].groupby("di")["pct"].mean()
        df["mkt_ret"] = df["di"].map(mkt_day).astype("float32")
        df["excess"] = (df["pct"] - df["mkt_ret"]).astype("float32")
        m1 = rsum("pct", 20) / 20
        acc2 = df["pct"].astype("float64") ** 2
        for k in range(1, 20):
            acc2 = acc2 + g["pct"].shift(k).astype("float64") ** 2
        var = acc2 / 20 - m1 ** 2
        df["vol20"] = np.sqrt(var.clip(lower=0))
        mkt_vol = df.groupby("di")["vol20"].transform("median")
        df["ivol20"] = df["vol20"] - mkt_vol

        snap = df[df["di"] >= start].copy()
        m = snap["ts_code"].map(board_ok).values
        s = snap[m]
        s = s[(s["is_paused"] == 0) & (s["is_st"] == 0)]
        s = s[(s["close"] >= 3) & (s["close"] <= 150)]
        s = s[(s["amt_yi"] >= 0.8) & s["turnover_rate"].between(2, 12)]
        s = s[s["bars"] >= 150]
        s = s[s["pct"].between(MIN_PCT, MAX_PCT)]
        s = s[s["ok1"] & (s["n1_p"] == 0)]
        s = s[s["n1_o_raw"] < s["n1_hl"] - 1e-6]     # T+1 一字涨停买不进
        if len(s):
            s = s.assign(d=[days[int(i)] for i in s["di"]])
            cols = ["d", "ts_code", "pct", "close_pos", "amp_pct", "low_tail", "body_pct",
                    "mom3", "mom5", "mom10", "mom20", "mom60", "bias5", "bias20",
                    "dist_low20", "dist_ph20", "dist_ph60", "dist_ph250",
                    "ma20_slope", "ma60_slope", "bull", "vr", "vr5", "amt_yi", "amt_ratio",
                    "turnover_rate", "turnover_cool", "up_streak", "lu5", "limit_up",
                    "prev_pct", "prev2_pct", "vol20", "ivol20", "price",
                    "mkt_ret", "excess", "gap", "ret1", "ret3", "ret5", "ret10"]
            out = s[cols].copy()
            for c in out.columns:
                if out[c].dtype == "float64":
                    out[c] = out[c].astype("float32")
            keep.append(out)
        dd = df.loc[bmask]
        if len(dd):
            gg = dd.groupby("di").agg(n=("pct", "size"), mkt=("pct", "mean"),
                                      up=("pct", lambda x: float((x > 0).mean())))
            for i, row in gg.iterrows():
                di = int(i)
                if di < start or di >= len(days):
                    continue
                mkt.append({"d": days[di], "n": int(row["n"]),
                            "mkt_pct": round(float(row["mkt"]), 4),
                            "up_share": round(float(row["up"]), 4)})
        start = hi
        print(f"  {start}/{len(files)} 累计样本 {sum(len(x) for x in keep):,}", flush=True)

    p = pd.concat(keep, ignore_index=True)
    tag = os.environ.get("PN_TAG", "right")
    p.to_parquet(OUT / f"panel_{tag}_206.parquet", index=False)
    pd.DataFrame(mkt).to_csv(OUT / f"panel_mkt_{tag}_206.csv", index=False)
    print("样本", len(p), "日数", p["d"].nunique())
    print(p["ret5"].describe().round(3).to_string())


if __name__ == "__main__":
    main()
