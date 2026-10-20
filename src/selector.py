# -*- coding: utf-8 -*-
"""选股条件与评分模块(多信号模式)。

执行口径: T日收盘后出观察池 -> T+1开盘竞价按缺口过滤买入 -> T+2开盘竞价卖出。
通过 config.yaml 的 selection.mode 切换信号模式:

  dip       趋势内缩量回调 + 次日深低开竞价承接(主力模式):
            T日选"中期趋势向上个股的缩量阴线回调"为观察池,
            T+1竞价仅买入缺口 <= buy_gap_below% 的深低开票, 博恐慌盘后的当日反抽。
  momentum  隔夜动量(备用): 温和放量上涨+均线多头, 博次日惯性。

通用硬过滤: 非ST、非停牌、主板+创业板、价格区间、成交额下限、换手区间、上市满N日。
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
    panel["ma20_prev"] = g["ma20"].transform(lambda s: s.shift(1))
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
    top_n = top_n or sel_cfg["top_n"]
    mode = sel_cfg.get("mode", "momentum")

    df = day_df.copy()
    df["board"] = df["ts_code"].map(board_of)
    stats = {"universe": int(len(df)), "mode": mode}

    # ---- 通用基础过滤(横截面字段) ----
    df = df[(df["is_paused"] == 0) & (df["is_st"] == 0)]
    df = df[df["board"].isin(uni_cfg["boards"])]
    stats["non_st_board"] = int(len(df))

    df = df[df["close"].between(uni_cfg["price_min"], uni_cfg["price_max"])]
    df = df[df["amount"] >= uni_cfg["amount_min_yi"] * 1e8]
    df = df[df["turnover_rate"] >= uni_cfg["turnover_min"]]
    stats["liquidity"] = int(len(df))

    df = df[df["bars"] >= uni_cfg["listed_min_bars"]]

    # ---- 各模式量价条件与评分 ----
    if mode == "momentum":
        # 隔夜动量: 温和放量上涨, 均线多头
        df = df[df["turnover_rate"] <= uni_cfg["turnover_max"]]
        df = df[df["pct"].between(sel_cfg["pct_min"], sel_cfg["pct_max"])]
        df = df[df["close"] < df["high_limit"] - 1e-9]      # 未封死涨停
        df = df[df["low"] < df["high"]]                     # 非一字板
        df = df[df["vol_ratio"].between(sel_cfg["vol_ratio_min"], sel_cfg["vol_ratio_max"])]
        stats["price_action"] = int(len(df))

        ma_ok = (df["close_adj"] > df["ma5"]) & (df["ma5"] > df["ma10"]) & (df["ma10"] > df["ma20"])
        if sel_cfg.get("ma5_slope_up", True):
            ma_ok &= df["ma5"] > df["ma5_prev"]
        df = df[ma_ok]
        df = df[df["bias20"] <= sel_cfg["bias20_max"]]
        df = df[(df["mom20"] > 0) & (df["mom20"] <= sel_cfg["mom20_max"])]
        stats["trend_ok"] = int(len(df))
        if df.empty:
            return [], stats

        w = sel_cfg["weights"]
        score = (
            w["vol_ratio"] * _zs(df["vol_ratio"])
            + w["mom3"] * _zs(df["mom3"])
            + w["mom20"] * _zs(df["mom20"])
            + w["amount"] * _zs(np.log1p(df["amount"]))
            + w["neg_bias20"] * _zs(-df["bias20"])
        )

    elif mode == "reversal":
        # 趋势内回调低吸: 中期趋势向上, 当日缩量阴线回调, 博次日反抽
        df = df[df["turnover_rate"] <= uni_cfg["turnover_max"]]
        df = df[df["pct"].between(sel_cfg["rev_pct_min"], sel_cfg["rev_pct_max"])]
        df = df[df["close"] < df["open"]]                   # 阴线
        df = df[df["close"] > df["low_limit"]]              # 收盘未跌停
        df = df[df["low"] < df["high"]]                     # 非一字
        df = df[df["vol_ratio"] <= sel_cfg["rev_vol_ratio_max"]]  # 缩量回调
        stats["price_action"] = int(len(df))

        trend = (
            (df["close_adj"] > df["ma20"])
            & (df["mom20"] > 0) & (df["mom20"] <= sel_cfg["mom20_max"])
            & (df["bias20"] <= sel_cfg["bias20_max"])
        )
        if sel_cfg.get("ma20_slope_up", True):
            trend &= df["ma20"] > df["ma20_prev"]
        df = df[trend]
        stats["trend_ok"] = int(len(df))
        if df.empty:
            return [], stats

        w = sel_cfg["rev_weights"]
        score = (
            w["mom20"] * _zs(df["mom20"])                    # 前期趋势越强越好
            + w["neg_bias20"] * _zs(-df["bias20"])           # 乖离越低越安全
            + w["amount"] * _zs(np.log1p(df["amount"]))      # 流动性
        )

    elif mode == "limitup":
        # 低位首板承接: 低位首板涨停(收盘封死, 非一字), 博次日溢价
        df = df[df["turnover_rate"] <= sel_cfg["lu_turnover_max"]]
        df = df[df["close"] >= df["high_limit"] - 1e-9]     # 收盘封死涨停
        df = df[df["low"] < df["high"]]                     # 非一字板
        stats["price_action"] = int(len(df))

        df = df[df["mom20"] <= sel_cfg["lu_mom20_max"]]     # 低位(前期涨幅不大)
        df = df[df["mom20"] > sel_cfg.get("lu_mom20_min", -20.0)]
        stats["trend_ok"] = int(len(df))
        if df.empty:
            return [], stats

        w = sel_cfg["lu_weights"]
        score = (
            w["neg_turnover"] * _zs(-df["turnover_rate"])    # 换手越低封板越实
            + w["neg_amount"] * _zs(-np.log1p(df["amount"]))  # 体量适中偏小
            + w["neg_mom20"] * _zs(-df["mom20"])             # 位置越低越安全
        )

    else:
        raise ValueError(f"未知选股模式: {mode}")

    df = df.assign(score=score).sort_values("score", ascending=False)
    stats["scored"] = int(len(df))

    picks = []
    for _, r in df.head(top_n).iterrows():
        code = r["ts_code"]
        if mode == "momentum":
            reasons = [
                f"涨幅{r['pct']:.1f}%",
                f"量比{r['vol_ratio']:.1f}",
                "均线多头",
                f"换手{r['turnover_rate']:.1f}%",
                f"20日动量{r['mom20']:.0f}%",
            ]
        elif mode == "dip":
            reasons = [
                f"回调{r['pct']:.1f}%",
                f"量比{r['vol_ratio']:.2f}(缩量)",
                f"20日动量{r['mom20']:.0f}%",
                f"乖离{r['bias20']:.1f}%",
                f"换手{r['turnover_rate']:.1f}%",
                "次日竞价深低开承接",
            ]
        else:
            reasons = [
                f"涨停{r['pct']:.1f}%",
                f"换手{r['turnover_rate']:.1f}%",
                f"成交额{r['amount']/1e8:.1f}亿",
                f"20日动量{r['mom20']:.0f}%",
                "低位首板",
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
