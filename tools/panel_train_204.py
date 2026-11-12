"""第 204 轮: 建一份可用于"训练"的全史样本(而不是只用来做统计检验).

第 202 轮的面板只留了 9 列, 是给"算基础率"用的。本轮要真的训练选股条件,
需要:
  1. 候选因子与 src/selector.py 的 dip_weights 可选键**一一对应**(训练出来
     的权重能直接写进 config.yaml, 不改一行代码);
     mom20 / neg_bias20 / neg_amount / neg_close_pos / neg_vol_ratio /
     low_tail / neg_turnover_cool / ivol / neg_dist_ma5 / neg_dist_low20
  2. 形态门槛放宽(dip 硬条件只打标记, 不用于筛选), 这样才能检验门槛本身
     —— pct 的 -6/-3、close_pos 的 0.32、mom20 的 5/25、dist_low20 的 6、
     vr 的 4, 到底是不是最优切点, 必须让数据说话。
  3. 多个持有期标签(1/2/3 日)与 T+1 缺口, 用于检验"买点"是否可改进。

票池口径与回测一致: 主板 / 价 3~150 / 额 >=0.8 亿 / 换手 2~12% / 上市 >=150 根 /
非停牌非 ST / T+1 可买入(非一字涨停) / 前瞻完整。

输出 output/panel_train_204.parquet 与 output/panel_mkt_204.csv
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
WARM = 40          # 指标回看(最长 20 日 + 余量)
CHUNK = 260


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
    lim = int(os.environ.get("PN_LIMIT", "0"))
    if lim:
        files = files[:lim]
    days = [os.path.basename(f)[:8] for f in files]
    print(f"面板 {len(days)} 天: {days[0]} → {days[-1]}", flush=True)

    keep: list[pd.DataFrame] = []
    mkt: list[dict] = []
    start = 0
    while start < len(files):
        lo = max(0, start - WARM)
        hi = min(len(files), start + CHUNK)
        df = load_days(files[lo:hi])
        df = df.sort_values(["ts_code", "di"], kind="mergesort")
        g = df.groupby("ts_code", sort=False)

        adj_c = (df["close"] * df["factor"]).astype("float64")
        adj_o = (df["open"] * df["factor"]).astype("float64")
        adj_h = (df["high"] * df["factor"]).astype("float64")
        adj_l = (df["low"] * df["factor"]).astype("float64")
        df["adj_c"], df["adj_o"], df["adj_h"], df["adj_l"] = adj_c, adj_o, adj_h, adj_l
        df["bars"] = g.cumcount() + 1

        # 前瞻(同一只票的下一行/下两行/下三行)
        for k, tag in ((1, "n1"), (2, "n2"), (3, "n3")):
            df[f"{tag}_o"] = g["adj_o"].shift(-k)
            df[f"{tag}_c"] = g["adj_c"].shift(-k)
            df[f"{tag}_d"] = g["di"].shift(-k)
        df["n1_o_raw"] = g["open"].shift(-1)
        df["n1_hl"] = g["high_limit"].shift(-1)
        df["n1_p"] = g["is_paused"].shift(-1).fillna(1)
        df["ret"] = (df["n2_c"] / df["n1_o"] - 1) * 100
        df["ret1"] = (df["n1_c"] / df["n1_o"] - 1) * 100
        df["ret3"] = (df["n3_c"] / df["n1_o"] - 1) * 100
        df["gap"] = (df["n1_o"] / adj_c - 1) * 100
        ok = (df["n1_d"] == df["di"] + 1) & (df["n2_d"] == df["di"] + 2)
        df["ok_fwd"] = ok & df["ret"].notna()

        # 滚动指标: 用 shift 累加代替 rolling, 14M 行下快一个量级
        def rsum(col: str, n: int) -> pd.Series:
            acc = df[col].astype("float64")
            for k in range(1, n):
                acc = acc + g[col].shift(k)
            return acc

        def rmin(col: str, n: int) -> pd.Series:
            acc = df[col].astype("float64")
            for k in range(1, n):
                acc = np.minimum(acc, g[col].shift(k))
            return pd.Series(acc, index=df.index)

        def rmax(col: str, n: int) -> pd.Series:
            acc = df[col].astype("float64")
            for k in range(1, n):
                acc = np.maximum(acc, g[col].shift(k))
            return pd.Series(acc, index=df.index)

        ma5, ma10, ma20 = rsum("adj_c", 5) / 5, rsum("adj_c", 10) / 10, rsum("adj_c", 20) / 20
        low20, high20 = rmin("adj_l", 20), rmax("adj_h", 20)
        volma20 = rsum("vol", 20) / 20
        volma5 = rsum("vol", 5) / 5
        amtma20 = rsum("amount", 20) / 20
        to3 = rsum("turnover_rate", 3) / 3
        to20 = rsum("turnover_rate", 20) / 20

        df["mom3"] = (adj_c / g["adj_c"].shift(3) - 1) * 100
        df["mom5"] = (adj_c / g["adj_c"].shift(5) - 1) * 100
        df["mom10"] = (adj_c / g["adj_c"].shift(10) - 1) * 100
        df["mom20"] = (adj_c / g["adj_c"].shift(20) - 1) * 100
        df["bias5"] = (adj_c / ma5 - 1) * 100
        df["bias20"] = (adj_c / ma20 - 1) * 100
        df["dist_ma5"] = (adj_c / ma5 - 1) * 100
        df["dist_low20"] = (adj_c / low20 - 1) * 100
        df["dist_high20"] = (adj_c / high20 - 1) * 100
        df["ma20_slope"] = (ma20 / ma20.groupby(df["ts_code"], sort=False).shift(5) - 1) * 100
        df["close_pos"] = ((df["close"] - df["low"]) / (df["high"] - df["low"]).replace(0, np.nan))
        df["amp_pct"] = (df["high"] - df["low"]) / df["pre_close"] * 100
        df["low_tail"] = (np.minimum(df["open"], df["close"]) - df["low"]) / df["pre_close"] * 100
        df["body_pct"] = (df["close"] - df["open"]) / df["pre_close"] * 100
        df["vr"] = df["vol"] / volma20
        df["vr5"] = df["vol"] / volma5
        df["amt_yi"] = df["amount"] / 1e8
        df["amt_ratio"] = df["amount"] / amtma20
        df["turnover_cool"] = to3 / to20.replace(0, np.nan)
        df["prev_pct"] = g["pct"].shift(1)
        df["down2"] = (df["pct"] < 0).astype("int8") + (g["pct"].shift(1) < 0).astype("int8")
        df["price"] = df["close"]

        # 横截面: 当日全主板等权收益(市场基准) + 个股 20 日波动
        bmask = (df["ts_code"].map(board_ok).values & (df["is_paused"] == 0).values)
        mkt_day = df.loc[bmask].groupby("di")["pct"].mean()
        df["mkt_ret"] = df["di"].map(mkt_day).astype("float32")
        df["excess"] = (df["pct"] - df["mkt_ret"]).astype("float32")
        # 20 日波动: E[x^2] - E[x]^2, 用 shift 累加平方
        m1 = rsum("pct", 20) / 20
        acc = df["pct"].astype("float64") ** 2
        for k in range(1, 20):
            acc = acc + g["pct"].shift(k).astype("float64") ** 2
        var = acc / 20 - m1 ** 2
        df["vol20"] = np.sqrt(var.clip(lower=0))
        mkt_vol = df.groupby("di")["vol20"].transform("median")
        df["ivol20"] = df["vol20"] - mkt_vol     # 特质波动(近似): 个股波动 - 同日全市场中位波动

        # 当日快照 + 票池
        snap = df[df["di"] >= start - lo].copy()
        m = snap["ts_code"].map(board_ok).values
        s = snap[m]
        s = s[(s["is_paused"] == 0) & (s["is_st"] == 0)]
        s = s[(s["close"] >= 3) & (s["close"] <= 150)]
        s = s[(s["amt_yi"] >= 0.8) & s["turnover_rate"].between(2, 12)]
        s = s[s["bars"] >= 150]
        # 形态宽池: 只要求"跌 + 收在下半区 + 前期有一定涨幅", 门槛本身留给数据检验
        s = s[s["pct"].between(-9.0, -1.0)]
        s = s[s["close_pos"] <= 0.55]
        s = s[s["mom20"].between(0.0, 32.0)]
        s = s[s["dist_low20"].between(1.0, 70.0)]
        s = s[s["vr"] <= 6.0]
        s = s[s["ok_fwd"] & (s["n1_p"] == 0)]
        s = s[s["n1_o_raw"] < s["n1_hl"] - 1e-6]
        if len(s):
            s = s.assign(d=[days[lo + int(i)] for i in s["di"]])
            s["hard"] = ((s["pct"].between(-6.0, -3.0)) & (s["close_pos"] <= 0.32)
                         & (s["mom20"].between(5.0, 25.0)) & (s["dist_low20"] >= 6.0)
                         & (s["vr"] <= 4.0)).astype("int8")
            cols = ["d", "ts_code", "hard", "pct", "close_pos", "amp_pct", "low_tail",
                    "body_pct", "mom3", "mom5", "mom10", "mom20", "bias5", "bias20",
                    "dist_ma5", "dist_low20", "dist_high20", "ma20_slope", "vr", "vr5",
                    "amt_yi", "amt_ratio", "turnover_rate", "turnover_cool", "vol20",
                    "ivol20", "prev_pct", "down2", "price", "mkt_ret", "excess",
                    "gap", "ret", "ret1", "ret3"]
            out = s[cols].copy()
            for c in out.columns:
                if out[c].dtype == "float64":
                    out[c] = out[c].astype("float32")
            keep.append(out)
        # 市场日聚合(全主板等权, 用于择时/环境分组)
        dd = df.loc[bmask]
        if len(dd):
            gg = dd.groupby("di").agg(n=("pct", "size"), mkt=("pct", "mean"),
                                      up=("pct", lambda x: float((x > 0).mean())))
            for i, row in gg.iterrows():
                di = lo + int(i)
                if di < start or di >= len(days):
                    continue
                mkt.append({"d": days[di], "n": int(row["n"]),
                            "mkt_pct": round(float(row["mkt"]), 4),
                            "up_share": round(float(row["up"]), 4)})
        print(f"  {days[start]} +{hi - start}d 宽池 {0 if not len(s) else len(s)}", flush=True)
        start = hi

    allq = pd.concat(keep, ignore_index=True)
    allq.to_parquet(OUT / "panel_train_204.parquet", index=False)
    pd.DataFrame(mkt).drop_duplicates("d").to_csv(OUT / "panel_mkt_204.csv", index=False)
    print(f"宽池样本 {len(allq):,} 行, {allq.d.nunique()} 个交易日, "
          f"其中现行硬条件 {int(allq.hard.sum()):,} 行")
    print(allq[["ret", "ret1", "ret3", "gap"]].describe().round(3).to_string())


if __name__ == "__main__":
    main()
