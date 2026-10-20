# -*- coding: utf-8 -*-
"""选股条件与评分模块。

策略(尾盘次日开盘隔夜动量):
  T日收盘后, 在全市场(主板+创业板)中筛选当日"温和放量上涨、均线多头"的个股,
  按多因子评分取前N只作为次日计划候选池; T+1开盘竞价买入, T+2开盘竞价卖出。

硬性过滤(可在 config.yaml 调整):
  1. 非ST、非停牌、非北交所/科创板
  2. 收盘价 2~80 元
  3. 当日成交额 >= 2亿
  4. 换手率 2%~20%
  5. 当日涨幅 1%~7%(温和上涨, 不追高位加速)
  6. 收盘未封死涨停(留有溢价空间), 非一字板
  7. 上市满120个交易日
  8. 量比 1.2~5.0(温和放量)
  9. 均线多头: 收盘 > MA5 > MA10 > MA20, 且 MA5 向上
 10. 20日乖离率 <= 15%, 20日累计涨幅 <= 30%

评分 = z(量比)*0.35 + z(3日动量)*0.25 + z(20日动量)*0.20
       + z(成交额)*0.10 + z(-乖离率)*0.10, 取分高者。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import board_of, load_config


# ----------------------------------------------------------------------
def build_indicators(panel: pd.DataFrame) -> pd.DataFrame:
    """在长表上向量化计算前复权价格与全部技术指标(仅用截至当日数据, 无未来函数)。"""
    panel = panel.sort_values(["ts_code", "trade_date"]).reset_index(drop=True)
    g = panel.groupby("ts_code", sort=False)

    flast = g["factor"].transform("last")
    k = panel["factor"] / flast.replace(0, np.nan)
    panel["close_adj"] = panel["close"] * k
    panel["open_adj"] = panel["open"] * k

    ca = g["close_adj"]
    panel["ma5"] = ca.transform(lambda s: s.rolling(5).mean())
    panel["ma10"] = ca.transform(lambda s: s.rolling(10).mean())
    panel["ma20"] = ca.transform(lambda s: s.rolling(20).mean())
    panel["ma5_prev"] = g["ma5"].transform(lambda s: s.shift(1))
    panel["vol_ma5_prev"] = g["vol"].transform(lambda s: s.shift(1).rolling(5).mean())
    panel["vol_ratio"] = panel["vol"] / panel["vol_ma5_prev"]
    panel["mom3"] = ca.transform(lambda s: s.pct_change(3)) * 100
    panel["mom20"] = ca.transform(lambda s: s.pct_change(20)) * 100
    panel["bias20"] = (panel["close_adj"] / panel["ma20"] - 1) * 100
    panel["bars"] = g.cumcount() + 1
    panel["above_ma20"] = panel["close_adj"] > panel["ma20"]
    return panel


def split_by_date(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {str(d): sub for d, sub in panel.groupby("trade_date", sort=True)}


# ----------------------------------------------------------------------
def regime_series(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """大盘风控指标: 市场宽度(站上MA20占比) + 全市场等权指数(相对其MA20)。"""
    uni = panel[(panel["is_paused"] == 0) & (panel["is_st"] == 0) & (panel["bars"] >= 21)]
    uni = uni[uni["ts_code"].map(lambda c: board_of(c) in cfg["universe"]["boards"])]
    reg = uni.groupby("trade_date").agg(
        breadth=("above_ma20", "mean"), mkt_pct=("pct", "mean")
    )
    reg = reg.sort_index()
    reg["mkt_idx"] = (1 + reg["mkt_pct"] / 100).cumprod()
    reg["mkt_ma"] = reg["mkt_idx"].rolling(cfg["regime"].get("market_ma", 20)).mean()
    ok = True
    if cfg["regime"].get("enabled", True):
        ok = (reg["breadth"] >= cfg["regime"].get("breadth_min", 0.35)) & (
            reg["mkt_idx"] >= reg["mkt_ma"]
        )
    reg["trade_ok"] = ok
    return reg


# ----------------------------------------------------------------------
def _zs(s: pd.Series) -> pd.Series:
    std = s.std()
    if not std or np.isnan(std):
        return s * 0
    return ((s - s.mean()) / std).clip(-3, 3)


def select_stocks(
    date: str,
    day_df: pd.DataFrame,
    cfg: dict,
    names: dict[str, str] | None = None,
    top_n: int | None = None,
) -> tuple[list[dict], dict]:
    """对某交易日横截面执行过滤+评分, 返回(候选池列表, 统计信息)。"""
    uni_cfg, sel_cfg = cfg["universe"], cfg["selection"]
    weights = sel_cfg["weights"]
    top_n = top_n or sel_cfg["top_n"]

    df = day_df.copy()
    df["board"] = df["ts_code"].map(board_of)
    stats = {"universe": int(len(df))}
    n0 = len(df)

    # ---- 基础过滤(横截面字段) ----
    df = df[(df["is_paused"] == 0) & (df["is_st"] == 0)]
    df = df[df["board"].isin(uni_cfg["boards"])]
    stats["non_st_board"] = int(len(df))

    df = df[df["close"].between(uni_cfg["price_min"], uni_cfg["price_max"])]
    df = df[df["amount"] >= uni_cfg["amount_min_yi"] * 1e8]
    df = df[df["turnover_rate"].between(uni_cfg["turnover_min"], uni_cfg["turnover_max"])]
    stats["liquidity"] = int(len(df))

    df = df[df["pct"].between(sel_cfg["pct_min"], sel_cfg["pct_max"])]
    df = df[df["close"] < df["high_limit"] - 1e-9]          # 未封死涨停
    df = df[df["low"] < df["high"]]                          # 非一字板
    stats["price_action"] = int(len(df))

    # ---- 指标过滤(历史字段) ----
    df = df[df["bars"] >= uni_cfg["listed_min_bars"]]
    df = df[df["vol_ratio"].between(sel_cfg["vol_ratio_min"], sel_cfg["vol_ratio_max"])]
    stats["volume_ok"] = int(len(df))

    ma_ok = (df["close_adj"] > df["ma5"]) & (df["ma5"] > df["ma10"]) & (df["ma10"] > df["ma20"])
    if sel_cfg.get("ma5_slope_up", True):
        ma_ok &= df["ma5"] > df["ma5_prev"]
    df = df[ma_ok]
    df = df[df["bias20"] <= sel_cfg["bias20_max"]]
    df = df[(df["mom20"] > 0) & (df["mom20"] <= sel_cfg["mom20_max"])]
    stats["trend_ok"] = int(len(df))

    if df.empty:
        return [], stats

    # ---- 多因子评分 ----
    score = (
        weights["vol_ratio"] * _zs(df["vol_ratio"])
        + weights["mom3"] * _zs(df["mom3"])
        + weights["mom20"] * _zs(df["mom20"])
        + weights["amount"] * _zs(np.log1p(df["amount"]))
        + weights["neg_bias20"] * _zs(-df["bias20"])
    )
    df = df.assign(score=score).sort_values("score", ascending=False)
    stats["scored"] = int(len(df))

    picks = []
    for _, r in df.head(top_n).iterrows():
        code = r["ts_code"]
        reasons = [
            f"涨幅{r['pct']:.1f}%",
            f"量比{r['vol_ratio']:.1f}",
            "均线多头",
            f"换手{r['turnover_rate']:.1f}%",
            f"20日动量{r['mom20']:.0f}%",
        ]
        picks.append({
            "code": code,
            "name": (names or {}).get(code, code),
            "close": round(float(r["close"]), 2),
            "pct": round(float(r["pct"]), 2),
            "amount_yi": round(float(r["amount"]) / 1e8, 2),
            "turnover_rate": round(float(r["turnover_rate"]), 2),
            "vol_ratio": round(float(r["vol_ratio"]), 2),
            "mom20": round(float(r["mom20"]), 1),
            "score": round(float(r["score"]), 3),
            "reasons": "、".join(reasons),
        })
    return picks, stats
