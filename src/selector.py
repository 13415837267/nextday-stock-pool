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
    # T日微观结构因子(均仅用截至T日数据):
    panel["prev_close_adj"] = g["close_adj"].transform(lambda s: s.shift(1))
    panel["amp_pct"] = (panel["high"] - panel["low"]) / panel["prev_close_adj"] * 100   # 当日振幅%
    hl_rng = (panel["high"] - panel["low"]).replace(0, np.nan)
    panel["close_pos"] = ((panel["close"] - panel["low"]) / hl_rng).fillna(0.5)          # 收盘在当日区间位置(0=最低1=最高)
    body_low = panel[["open", "close"]].min(axis=1)
    panel["low_tail"] = (body_low - panel["low"]) / panel["prev_close_adj"] * 100        # 下影线长度%
    # 扩展微观结构特征(均仅用截至T日数据, 供可选条件/因子使用):
    panel["prev_pct"] = g["pct"].transform(lambda s: s.shift(1))                          # T-1日涨幅%
    panel["down2"] = ((panel["pct"] < 0) & (panel["prev_pct"] < 0)).astype("int8")        # 连续两日阴线
    panel["dist_ma5"] = (panel["close_adj"] / panel["ma5"] - 1) * 100                     # 收盘相对5日线距离%
    panel["dist_low20"] = (panel["low"] / g["low"].transform(lambda s: s.rolling(20).min()) - 1) * 100  # 距20日最低%
    panel["prev_low"] = g["low"].transform(lambda s: s.shift(1))                          # T-1日最低价
    panel["below_prev_low"] = (panel["low"] < panel["prev_low"]).astype("int8")           # T日盘中破前日低
    panel["hold_prev_low"] = (panel["close"] > panel["prev_low"]).astype("int8")          # T日收盘守住前日低
    return panel


def split_by_date(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    return {str(d): sub for d, sub in panel.groupby("trade_date", sort=True)}


# ----------------------------------------------------------------------
def regime_base(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """大盘风控基础序列(与风控参数无关, 可复用): 市场宽度 + 全市场等权指数。"""
    uni = panel[(panel["is_paused"] == 0) & (panel["is_st"] == 0) & (panel["bars"] >= 21)]
    uni = uni[uni["ts_code"].map(lambda c: board_of(c) in cfg["universe"]["boards"])]
    reg = uni.groupby("trade_date").agg(
        breadth=("above_ma20", "mean"), mkt_pct=("pct", "mean")
    )
    reg = reg.sort_index()
    reg["mkt_idx"] = (1 + reg["mkt_pct"] / 100).cumprod()
    return reg


def apply_regime(reg_base: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """按风控参数在基础序列上生成 trade_ok(均仅用截至当日数据, 无未来函数)。"""
    reg = reg_base.copy()
    reg["mkt_ma"] = reg["mkt_idx"].rolling(cfg["regime"].get("market_ma", 20)).mean()
    ok = True
    if cfg["regime"].get("enabled", True):
        breadth_ok = reg["breadth"] >= cfg["regime"].get("breadth_min", 0.35)
        trend_ok = reg["mkt_idx"] >= reg["mkt_ma"]
        # mode: and=双条件均通过才交易(默认, 保守); or=任一通过即交易(宽松, 增加交易天数)
        if cfg["regime"].get("mode", "and") == "or":
            ok = breadth_ok | trend_ok
        else:
            ok = breadth_ok & trend_ok
        # 恐慌型波动熔断(2026-10-23定稿, 十年3372日信号扫描+走前验证):
        # 仅当 高波动(等权日收益20日std>vol_max) 且 市场20日动量为负 时空仓 ——
        # "暴跌中带高波动"=恐慌深水区, dip承接是接飞刀; "暴涨型高波动"(如2024-10
        # 924行情初期)是趋势启动, 保留参与。被剔除29笔全部位于2015股灾/2016熔断/
        # 2018/2020-03/2022/2026-03恐慌期, 均值-1.72%/胜率41%;
        # h1(前半)-0.488%→+0.104%, h2(后半)+0.702%→+1.092%/胜率56.5%, 双段一致
        vol_max = cfg["regime"].get("vol_max")
        if vol_max:
            m20 = reg["mkt_idx"] / reg["mkt_idx"].shift(20) - 1
            panic = (reg["mkt_pct"].rolling(20).std() > vol_max) \
                    & (m20 < cfg["regime"].get("vol_mkt_ret_min", 0.0))
            ok = ok & ~panic
    reg["trade_ok"] = ok
    # panic 标记: 供回测引擎区分"恐慌日"(不计入连续风控天数, 防止被超限参与机制绕过)
    if cfg["regime"].get("vol_max"):
        reg["panic"] = panic
    else:
        reg["panic"] = False
    return reg


def regime_series(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """大盘风控指标: 市场宽度(站上MA20占比) + 全市场等权指数(相对其MA)。"""
    return apply_regime(regime_base(panel, cfg), cfg)


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

    # 可选: 仅在指定星期几出池(0=周一...4=周五; 缺省=全部)
    _wd = sel_cfg.get("buy_weekdays")
    if _wd is not None:
        if pd.Timestamp(date).dayofweek not in set(int(x) for x in _wd):
            return [], {"universe": 0, "mode": mode, "weekday_skip": 1}

    df = day_df.copy()
    df["board"] = df["ts_code"].map(board_of)
    stats = {"universe": int(len(df)), "mode": mode}

    # ---- 通用基础过滤(横截面字段) ----
    df = df[(df["is_paused"] == 0) & (df["is_st"] == 0)]
    df = df[df["board"].isin(uni_cfg["boards"])]
    stats["non_st_board"] = int(len(df))

    df = df[df["close"].between(uni_cfg["price_min"], uni_cfg["price_max"])]
    df = df[df["amount"] >= uni_cfg["amount_min_yi"] * 1e8]
    if uni_cfg.get("amount_max_yi"):                 # 可选: 成交额上限(剔除过热巨量票)
        df = df[df["amount"] <= uni_cfg["amount_max_yi"] * 1e8]
    df = df[df["turnover_rate"] >= uni_cfg["turnover_min"]]
    stats["liquidity"] = int(len(df))

    df = df[df["bars"] >= uni_cfg["listed_min_bars"]]

    # ---- 各模式量价条件与评分 ----
    if mode == "dip":
        # 趋势内缩量回调: T日选出观察池; T+1竞价仅对缺口<=buy_gap_below的深低开票执行买入
        df = df[df["turnover_rate"] <= uni_cfg["turnover_max"]]
        df = df[df["pct"].between(sel_cfg["dip_pct_min"], sel_cfg["dip_pct_max"])]
        df = df[df["close"] < df["open"]]                   # 阴线回调
        df = df[df["close"] > df["low_limit"]]              # 收盘未跌停
        df = df[df["low"] < df["high"]]                     # 非一字
        df = df[df["vol_ratio"] <= sel_cfg["dip_vol_ratio_max"]]  # 缩量回调(非放量出货)
        if sel_cfg.get("dip_vol_ratio_min") is not None:  # 可选: 量比下限(过度缩量=无承接动能)
            df = df[df["vol_ratio"] >= sel_cfg["dip_vol_ratio_min"]]
        stats["price_action"] = int(len(df))

        trend = (
            (df["mom20"] >= sel_cfg["dip_mom20_min"])
            & (df["mom20"] <= sel_cfg["dip_mom20_max"])     # 有前期趋势但非高位加速
            & (df["bias20"] <= sel_cfg["bias20_max"])       # 乖离不过热
        )
        # 可选微观结构过滤(均通过 config.selection 配置, 未配置即不启用):
        if sel_cfg.get("dip_bias20_min") is not None:       # 乖离下限: 负乖离过深=趋势已崩坏
            trend &= df["bias20"] >= sel_cfg["dip_bias20_min"]
        if sel_cfg.get("dip_mom3_min") is not None:         # 3日动量下限: 短期跌太狠=崩坏中继
            trend &= df["mom3"] >= sel_cfg["dip_mom3_min"]
        if sel_cfg.get("dip_amp_max") is not None:          # T日振幅上限: 振幅过大=分歧过巨
            trend &= df["amp_pct"] <= sel_cfg["dip_amp_max"]
        if sel_cfg.get("dip_close_pos_max") is not None:    # 收盘位置上限: 阴线收太高=尾盘仍在出货
            trend &= df["close_pos"] <= sel_cfg["dip_close_pos_max"]
        if sel_cfg.get("dip_low_tail_min") is not None:     # 下影线下限: 盘中已有恐慌承接
            trend &= df["low_tail"] >= sel_cfg["dip_low_tail_min"]
        if sel_cfg.get("dip_prev_pct_min") is not None:     # T-1日涨幅下限(排除前日已大跌)
            trend &= df["prev_pct"] >= sel_cfg["dip_prev_pct_min"]
        if sel_cfg.get("dip_prev_pct_max") is not None:     # T-1日涨幅上限(排除前日大涨后回调)
            trend &= df["prev_pct"] <= sel_cfg["dip_prev_pct_max"]
        if sel_cfg.get("dip_down2_max") is not None:        # 连阴天数上限(1=禁连阴票)
            trend &= df["down2"] <= sel_cfg["dip_down2_max"]
        if sel_cfg.get("dip_down2_min") is not None:        # 连阴天数下限(1=只要连阴票)
            trend &= df["down2"] >= sel_cfg["dip_down2_min"]
        if sel_cfg.get("dip_dist_low20_min") is not None:   # 距20日最低涨幅下限(排除已深跌票)
            trend &= df["dist_low20"] >= sel_cfg["dip_dist_low20_min"]
        if sel_cfg.get("dip_below_prev_low_max") is not None:  # 破前日低天数上限(1=允许盘中破前低)
            trend &= df["below_prev_low"] <= sel_cfg["dip_below_prev_low_max"]
        if sel_cfg.get("dip_hold_prev_low_min") is not None:   # 守住前日低下限(1=必须收盘守在前日低上)
            trend &= df["hold_prev_low"] >= sel_cfg["dip_hold_prev_low_min"]
        if sel_cfg.get("dip_dist_ma5_max") is not None:     # 收盘距5日线上限(%): 回调未击穿5日线过深
            trend &= df["dist_ma5"] <= sel_cfg["dip_dist_ma5_max"]
        df = df[trend]
        stats["trend_ok"] = int(len(df))
        if df.empty:
            return [], stats

        w = sel_cfg["dip_weights"]
        score = (
            w["mom20"] * _zs(df["mom20"])                    # 前期趋势越强越好
            + w["neg_bias20"] * _zs(-df["bias20"])           # 乖离越低越安全
            + w["neg_amount"] * _zs(-np.log1p(df["amount"])) # 体量适中偏小弹性大
        )
        # 可选扩展评分因子(权重缺省0=不启用, 兼容旧config):
        if w.get("neg_close_pos"):
            score = score + w["neg_close_pos"] * _zs(-df["close_pos"])   # 收盘位置越低(恐慌释放越充分)越好
        if w.get("neg_vol_ratio"):
            score = score + w["neg_vol_ratio"] * _zs(-df["vol_ratio"])   # 缩量越极致越好
        if w.get("low_tail"):
            score = score + w["low_tail"] * _zs(df["low_tail"])          # 下影线越长(盘中承接越强)越好
        if w.get("neg_dist_ma5"):
            score = score + w["neg_dist_ma5"] * _zs(-df["dist_ma5"])     # 贴5日线越近(跌幅未失控)越好
        if w.get("neg_dist_low20"):
            score = score + w["neg_dist_low20"] * _zs(-df["dist_low20"]) # 距20日低点越近(弹性越大)越好

    elif mode == "momentum":
        # 隔夜动量(备用): 温和放量上涨, 均线多头
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
        df = df[(df["mom20"] > 0) & (df["mom20"] <= sel_cfg.get("mom20_max", 25.0))]
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
        else:  # dip
            reasons = [
                f"回调{r['pct']:.1f}%",
                f"量比{r['vol_ratio']:.2f}(缩量)",
                f"20日动量{r['mom20']:.0f}%",
                f"乖离{r['bias20']:.1f}%",
                f"换手{r['turnover_rate']:.1f}%",
                "次日竞价深低开承接",
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
