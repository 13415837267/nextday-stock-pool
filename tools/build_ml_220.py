"""第 220 轮 · 从全历史行情重建机器学习训练集（严格 PIT，无前视）

与前几轮训练实验的区别（这三点决定了本轮不是重复）：
  1. **去掉 gap 特征**。第 204 轮的 FEATS 里有 `gap`（T+1 开盘相对 T 收盘的缺口），
     而标签是 `close[T+2]/open[T+1]-1` —— gap 直接决定分母。用它当特征等于把
     "买入价"喂给模型, 低开越多收益越高, 是机械关系不是预测能力。本轮全部剔除。
  2. **放宽池子**。第 212 轮的面板预先套了 `turnover 2~12%` 这类策略特征, 训练集被
     策略假设污染。本轮只保留**实盘硬约束**(主板/非ST/非停牌/价格 3~150/成交额≥0.8亿/
     上市≥150日), 换手率交给模型自己判断。
  3. **标签口径加"买不到"过滤**。T+1 开盘一字涨停(open ≥ high_limit*0.999)的样本
     实际买不进, 留着会让模型高估。T+1/T+2 停牌同理。

标签: ret = (adj_close[T+2] / adj_open[T+1] - 1) * 100 - 0.30(双边成本)
输出: output/ml_220.parquet
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "output"
COST = 0.30
MIN_BARS = 150


def board_main(code: str) -> bool:
    c, s = code.split(".")
    if s == "SH":
        return c.startswith("60")
    if s == "SZ":
        return c.startswith("00")
    return False


def build() -> pd.DataFrame:
    print("[1/4] 读价格表 ...", flush=True)
    df = pd.read_parquet(
        OUT / "price_table_212.parquet",
        columns=["ts_code", "trade_date", "open", "close", "adj_open", "adj_close",
                 "high_limit", "low_limit", "pct", "vol", "amount",
                 "turnover_rate", "is_paused", "is_st"],
    )
    print(f"  原始 {len(df):,} 行", flush=True)
    df["ts_code"] = df["ts_code"].astype("category")

    print("[2/4] 池子过滤 ...", flush=True)
    df = df[df["ts_code"].map(board_main).values]
    df = df[(df["is_paused"] == 0) & (df["is_st"] == 0)]
    df["amt_yi"] = (df["amount"] / 1e8).astype("float32")
    df["price"] = df["close"].astype("float32")
    df = df[(df["price"] >= 3) & (df["price"] <= 150) & (df["amt_yi"] >= 0.8)]
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    print(f"  过滤后 {len(df):,} 行", flush=True)

    print("[3/4] 特征 ...", flush=True)
    g = df.groupby("ts_code", observed=True, sort=False)

    df["n"] = g.cumcount()
    for c in ("open", "close", "adj_open", "adj_close", "high_limit", "low_limit",
              "pct", "vol", "turnover_rate"):
        df[c] = df[c].astype("float32")

    ao, ac = df["adj_open"], df["adj_close"]
    pre = ac.shift(1).where(df["ts_code"].eq(df["ts_code"].shift(1)))
    r1 = ac / pre - 1                      # 当日收益(复权)

    # price_table 没有 high/low, close_pos/振幅/影线本轮不可得 —— 这 4 个特征留到补数据后
    # --- 动量 ---
    for k in (3, 5, 10, 20, 60):
        df[f"mom{k}"] = (ac / g["adj_close"].shift(k) - 1) * 100

    # --- 均线与乖离 ---
    ma5 = g["adj_close"].transform(lambda s: s.rolling(5).mean())
    ma20 = g["adj_close"].transform(lambda s: s.rolling(20).mean())
    ma60 = g["adj_close"].transform(lambda s: s.rolling(60).mean())
    df["dist_ma5"] = (ac / ma5 - 1) * 100
    df["dist_ma20"] = (ac / ma20 - 1) * 100
    df["dist_ma60"] = (ac / ma60 - 1) * 100
    df["ma20_slope"] = (ma20 / g["adj_close"].transform(
        lambda s: s.rolling(20).mean().shift(5)) - 1) * 100

    # --- 距区间高低点 ---
    min20 = g["adj_close"].transform(lambda s: s.rolling(20).min())
    max20 = g["adj_close"].transform(lambda s: s.rolling(20).max())
    max60 = g["adj_close"].transform(lambda s: s.rolling(60).max())
    max250 = g["adj_close"].transform(lambda s: s.rolling(250, min_periods=120).max())
    df["dist_low20"] = (ac / min20 - 1) * 100
    df["dist_high20"] = (ac / max20 - 1) * 100
    df["dist_high60"] = (ac / max60 - 1) * 100
    df["dist_high250"] = (ac / max250 - 1) * 100

    # --- 量能 ---
    amt_ma5 = g["amt_yi"].transform(lambda s: s.rolling(5).mean())
    amt_ma20 = g["amt_yi"].transform(lambda s: s.rolling(20).mean())
    df["amt_ratio"] = df["amt_yi"] / amt_ma20
    df["amt_trend"] = amt_ma5 / amt_ma20
    df["vr"] = df["vol"] / g["vol"].transform(lambda s: s.rolling(5).mean())
    df["turnover_cool"] = df["turnover_rate"] / g["turnover_rate"].transform(
        lambda s: s.rolling(5).mean())

    # --- 波动 ---
    df["ret1"] = r1 * 100
    df["vol20"] = g["ret1"].transform(lambda s: s.rolling(20).std())
    df["vol60"] = g["ret1"].transform(lambda s: s.rolling(60).std())
    df["maxdd20"] = ((ac / g["adj_close"].transform(lambda s: s.rolling(20).max()) - 1) * 100)

    # --- 连续形态 ---
    up = (df["ret1"] > 0).astype("int8")
    df["up_days5"] = up.groupby(df["ts_code"], observed=True).transform(
        lambda s: s.rolling(5).sum())
    lim = (df["close"] >= df["high_limit"] * 0.999).astype("int8")
    df["lu5"] = lim.groupby(df["ts_code"], observed=True).transform(
        lambda s: s.rolling(5).sum())
    df["lu20"] = lim.groupby(df["ts_code"], observed=True).transform(
        lambda s: s.rolling(20).sum())

    # --- 前一日 ---
    df["prev_pct"] = df["pct"].shift(1).where(df["ts_code"].eq(df["ts_code"].shift(1)))
    df["prev2_pct"] = df["pct"].shift(2).where(df["ts_code"].eq(df["ts_code"].shift(2)))

    # --- 市场状态(当日, 收盘后可得) ---
    mkt = df.groupby("trade_date")["ret1"].mean().rename("mkt_ret")
    df = df.merge(mkt, on="trade_date", how="left")
    mkt_s = df.groupby("trade_date")["ret1"].mean()
    mkt_nav = (1 + mkt_s / 100).cumprod()
    mkt_dev = (mkt_nav / mkt_nav.rolling(20).mean() - 1) * 100
    mkt_vol = mkt_s.rolling(20).std()
    up_share = df.assign(up=(df["ret1"] > 0).astype("int8")).groupby("trade_date")["up"].mean()
    mk = pd.DataFrame({"mkt_dev": mkt_dev, "mkt_vol": mkt_vol, "up_share": up_share})
    df = df.merge(mk, on="trade_date", how="left")
    # merge 会重排索引, groupby 对象必须重建, 否则 shift 取到的是错误行(实测会删掉 88% 样本)
    df = df.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    g = df.groupby("ts_code", observed=True, sort=False)

    print("[4/4] 标签 ...", flush=True)
    # 买入价 = T+1 开盘(复权), 卖出价 = T+2 收盘(复权)
    buy_px = g["adj_open"].shift(-1)
    sell_px = g["adj_close"].shift(-2)
    df["ret"] = (sell_px / buy_px - 1) * 100 - COST
    # 买不到: T+1 开盘一字涨停。注意 high_limit 是**未复权**价, 必须和未复权 open 比 ——
    # 拿复权价(adj_open ~ 900)去比未复权涨停价(~27)会把 88% 的样本误判成"买不到"。
    buy_raw = g["open"].shift(-1)
    hl1 = g["high_limit"].shift(-1)
    df["blocked"] = (buy_raw >= hl1 * 0.999).astype("int8")
    # 停牌/无效
    df["invalid"] = (buy_px.isna() | sell_px.isna() | (buy_px <= 0)).astype("int8")
    df["year"] = (df["trade_date"] // 10000).astype(int)

    return df


FEATS = [
    "pct", "mom3", "mom5", "mom10", "mom20", "mom60",
    "bias5", "bias20", "dist_ma5", "dist_ma20", "dist_ma60", "ma20_slope",
    "dist_low20", "dist_high20", "dist_high60", "dist_high250",
    "amt_yi", "amt_ratio", "amt_trend", "vr", "turnover_rate", "turnover_cool",
    "vol20", "vol60", "maxdd20", "up_days5", "lu5", "lu20",
    "prev_pct", "prev2_pct",
    "mkt_ret", "mkt_dev", "mkt_vol", "up_share",
    "price", "n",
]


def main() -> None:
    df = build()
    df["bias5"] = (df["adj_close"] / df.groupby("ts_code", observed=True)["adj_close"]
                   .transform(lambda s: s.rolling(5).mean()) - 1) * 100
    df["bias20"] = (df["adj_close"] / df.groupby("ts_code", observed=True)["adj_close"]
                    .transform(lambda s: s.rolling(20).mean()) - 1) * 100

    n0 = len(df)
    print(f"  全部 {n0:,}")
    m = (df["invalid"] == 0) & (df["blocked"] == 0)
    print(f"  去掉买不到/无效 -> {m.sum():,} ({(1-m.mean())*100:.1f}% 被删)")
    m2 = m & df["ret"].between(-25, 25) & (df["n"] >= MIN_BARS) & (df["trade_date"] >= 20120101)
    print(f"  收益/新股/起始年份 -> {m2.sum():,}")
    v = df[m2]
    for c in FEATS + ["ret"]:
        n_na = int(v[c].isna().sum())
        if n_na:
            print(f"    特征 {c:<16} NaN {n_na:,} ({n_na/len(v)*100:.1f}%)")
    v = v.dropna(subset=FEATS + ["ret"])
    print(f"  dropna 后 -> {len(v):,}")

    keep = ["trade_date", "ts_code", "year", "ret"] + FEATS
    v = v[keep].copy()
    for c in FEATS:
        v[c] = v[c].astype("float32")
    v["ret"] = v["ret"].astype("float32")

    print(f"  可训练样本 {len(v):,} 行 / {v['trade_date'].nunique()} 天")
    print("  年份分布:", v.groupby("year").size().to_dict())
    print("  标签: 均值 %.3f%%  中位 %.3f%%  标准差 %.3f%%"
          % (v["ret"].mean(), v["ret"].median(), v["ret"].std()))

    p = OUT / "ml_220.parquet"
    v.to_parquet(p, index=False, compression="zstd")
    meta = {"rows": int(len(v)), "days": int(v["trade_date"].nunique()),
            "start": str(v["trade_date"].min()), "end": str(v["trade_date"].max()),
            "feats": FEATS, "cost": COST,
            "label_mean": round(float(v["ret"].mean()), 4),
            "label_std": round(float(v["ret"].std()), 4),
            "yearly": {str(k): int(x) for k, x in v.groupby("year").size().items()}}
    (OUT / "ml_220_meta.json").write_text(json.dumps(meta, ensure_ascii=False, indent=1))
    print("saved", p, f"{p.stat().st_size/1e6:.0f}MB")


if __name__ == "__main__":
    main()
