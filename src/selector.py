# -*- coding: utf-8 -*-
"""选股条件与评分模块(多信号模式)。

执行口径: T日收盘后出观察池 -> T+1开盘竞价按开盘价直接买入(第九十二轮起不设缺口门槛)
          -> T+2开盘竞价卖出(开盘深跌未跌停则改当日收盘卖出)。
通过 config.yaml 的 selection.mode 切换信号模式:

  dip       趋势内缩量回调 + 次日深低开竞价承接(主力模式):
            T日选"中期趋势向上个股的缩量阴线回调"为观察池,
            T+1集合竞价直接按开盘价买入(不按缺口挑单; 停牌/一字涨停买不到除外)。
  momentum  隔夜动量(备用): 温和放量上涨+均线多头, 博次日惯性。

通用硬过滤: 非ST、非停牌、主板+创业板、价格区间、成交额下限、换手区间、上市满N日。
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from .config import board_of, load_config

# 策略中文名(多策略合并池的展示标签: daily_run/site/回测成交记录共用同一套命名)
STRATEGY_CN = {
    "dip": "主力·趋势缩量回调",
    "momentum": "隔夜动量",
    "reversal": "超跌反转",
    "squeeze": "低波缩量横盘",
    "pullback": "趋势回踩5日线",
    "limitup": "涨停深低开承接",
    "spike": "放量突破承接",
    "oversold": "深跌阳线反转",
}


# ----------------------------------------------------------------------
def build_indicators(panel: pd.DataFrame) -> pd.DataFrame:
    """在长表上向量化计算前复权价格与全部技术指标(仅用截至当日数据, 无未来函数)。

    前置: panel为日期主序(fetch_panel/逐日csv拼接均满足)。groupby保组内相对序,
    每票组内即时间序, 故无需全表sort_values——2026-10-24第四十二轮去掉该排序:
    2500日面板全表排序复制~2GB, 是长窗口OOM主因之一; 逐日内源数据已按代码序,
    去排序后逐日帧行序与原先完全一致, 指标零漂移。
    """
    # 数据卫生(2026-10-24): 上市首日行 pre_close=0, 数据源的pct为脏值(实测+7199%)。
    # 置NaN使其在选股区间过滤与风控等权均值(mean跳过NaN)中被显式排除,
    # 不再依赖 bars>=21 过滤的隐性防护; 次日起pct由close_adj正常衔接。
    if "pre_close" in panel.columns:
        panel.loc[panel["pre_close"] <= 0, "pct"] = np.nan
    # float32面板(长窗口回测)下新增指标列同步float32: pandas3的rolling/shift/
    # pct_change对float32输入仍返回float64, 25个新列在2500日(1477万行)上若保持
    # float64累计~2.1GB, 是长窗口OOM主因之三。float64面板(实盘daily_run)路径
    # 逐字节不变。与prepare_context尾部_downcast_float等价, 只是提前到逐列赋值时。
    f32 = panel["close"].dtype == np.float32

    def put(name: str, s: pd.Series, dt: str | None = None) -> None:
        panel[name] = s.astype(dt or "float32", copy=False) if f32 else s

    g = panel.groupby("ts_code", sort=False)

    # IVOL20(第六十七轮文献因子): 个股日收益对市场等权日收益的20日残差标准差(市场调整模型)。
    # 机制: 陆蓉/王麒《经济学(季刊)》2025——机构主导隔夜(压高低分歧票->低开), 散户主导日内
    # (推高->高走), 高特质波动票的深低开承接修复更强。停牌日残差置NaN(rolling跳过)。
    _mkt = panel.loc[panel["is_paused"] == 0].groupby("trade_date")["pct"].mean()
    put("mkt_pct", panel["trade_date"].map(_mkt))
    put("resid_pct", np.where(panel["is_paused"] == 1, np.nan,
                              panel["pct"] - panel["mkt_pct"]))
    put("ivol20", g["resid_pct"].transform(lambda s: s.rolling(20, min_periods=15).std()))

    flast = g["factor"].transform("last")
    k = panel["factor"] / flast.replace(0, np.nan)
    panel["close_adj"] = panel["close"] * k
    panel["open_adj"] = panel["open"] * k

    ca = g["close_adj"]
    put("ma5", ca.transform(lambda s: s.rolling(5).mean()))
    put("ma10", ca.transform(lambda s: s.rolling(10).mean()))
    put("ma20", ca.transform(lambda s: s.rolling(20).mean()))
    put("ma5_prev", g["ma5"].transform(lambda s: s.shift(1)))
    put("ma20_prev", g["ma20"].transform(lambda s: s.shift(1)))
    put("vol_ma5_prev", g["vol"].transform(lambda s: s.shift(1).rolling(5).mean()))
    panel["vol_ratio"] = panel["vol"] / panel["vol_ma5_prev"]
    put("mom3", ca.transform(lambda s: s.pct_change(3)) * 100)
    put("mom20", ca.transform(lambda s: s.pct_change(20)) * 100)
    panel["bias20"] = (panel["close_adj"] / panel["ma20"] - 1) * 100
    put("bars", g.cumcount() + 1, "int32" if f32 else None)
    panel["above_ma20"] = panel["close_adj"] > panel["ma20"]
    # T日微观结构因子(均仅用截至T日数据):
    put("prev_close_adj", g["close_adj"].transform(lambda s: s.shift(1)))
    panel["amp_pct"] = (panel["high"] - panel["low"]) / panel["prev_close_adj"] * 100   # 当日振幅%
    hl_rng = (panel["high"] - panel["low"]).replace(0, np.nan)
    panel["close_pos"] = ((panel["close"] - panel["low"]) / hl_rng).fillna(0.5)          # 收盘在当日区间位置(0=最低1=最高)
    del hl_rng
    body_low = panel[["open", "close"]].min(axis=1)
    panel["low_tail"] = (body_low - panel["low"]) / panel["prev_close_adj"] * 100        # 下影线长度%
    del body_low
    # 扩展微观结构特征(均仅用截至T日数据, 供可选条件/因子使用):
    put("prev_pct", g["pct"].transform(lambda s: s.shift(1)))                             # T-1日涨幅%
    panel["down2"] = ((panel["pct"] < 0) & (panel["prev_pct"] < 0)).astype("int8")        # 连续两日阴线
    panel["dist_ma5"] = (panel["close_adj"] / panel["ma5"] - 1) * 100                     # 收盘相对5日线距离%
    dl = g["low"].transform(lambda s: s.rolling(20).min())
    panel["dist_low20"] = (panel["low"] / dl - 1) * 100   # 距20日最低%
    del dl
    put("prev_low", g["low"].transform(lambda s: s.shift(1)))                             # T-1日最低价
    panel["below_prev_low"] = (panel["low"] < panel["prev_low"]).astype("int8")           # T日盘中破前日低
    panel["hold_prev_low"] = (panel["close"] > panel["prev_low"]).astype("int8")          # T日收盘守住前日低
    return panel


def split_by_date(panel: pd.DataFrame) -> dict[str, pd.DataFrame]:
    """按trade_date切分为逐日帧。

    日期主序面板(所有调用方均为fetch_panel式逐日拼接)下日期块必然连续,
    用位置切片出CoW视图(近零拷贝, 第四十二轮: 原groupby全量复制2500日~2GB,
    OOM主因之二); 块不连续时回退groupby复制保正确性。
    """
    d = panel["trade_date"].to_numpy()
    if len(d):
        bounds = np.flatnonzero(np.r_[True, d[1:] != d[:-1]])
        if len(bounds) == panel["trade_date"].nunique():   # 日期块连续
            edges = np.r_[bounds, len(d)]
            return {str(d[i]): panel.iloc[i:j] for i, j in zip(edges[:-1], edges[1:])}
    return {str(v): sub for v, sub in panel.groupby("trade_date", sort=True)}


# ----------------------------------------------------------------------
def regime_base(panel: pd.DataFrame, cfg: dict) -> pd.DataFrame:
    """大盘风控基础序列(与风控参数无关, 可复用): 市场宽度 + 全市场等权指数。"""
    # 先裁列再过滤(第四十二轮内存优化): 全列过滤副本2500日~2GB, 裁到7列后~0.6GB
    cols = ["trade_date", "ts_code", "is_paused", "is_st", "bars", "above_ma20", "pct"]
    uni = panel.loc[(panel["is_paused"] == 0) & (panel["is_st"] == 0)
                    & (panel["bars"] >= 21), cols]
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
    # 注意: enabled=False时panic计算块被跳过, 此处必须同时检查enabled(2026-10-25修复:
    # 否则enabled=False且vol_max>0的组合触发UnboundLocalError)
    if cfg["regime"].get("enabled", True) and cfg["regime"].get("vol_max"):
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


def _select_one(
    date: str,
    day_df: pd.DataFrame,
    cfg: dict,
    names: dict[str, str] | None = None,
    top_n: int | None = None,
) -> tuple[list[dict], dict]:
    """单一信号模式的过滤+评分(多策略补位的原子实现)。"""
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

    # 风险股名称双保险(2026-10-24): 名称含"ST"(涵盖*ST/S*ST)或含"退"(退市整理期标识)一律排除。
    # 实测20261023: is_st字段已覆盖全部198只ST与1只退市整理期票, 无漏网; 此兜底防数据源
    # is_st字段将来口径变化(如退市整理期摘牌ST标记)。字段过滤为主, 名称为兜底
    if names:
        _nm = df["ts_code"].map(lambda c: str(names.get(c, "")))
        df = df[~(_nm.str.contains("ST", case=False, na=False) | _nm.str.contains("退", na=False))]

    df = df[df["close"].between(uni_cfg["price_min"], uni_cfg["price_max"])]
    df = df[df["amount"] >= uni_cfg["amount_min_yi"] * 1e8]
    if uni_cfg.get("amount_max_yi"):                 # 可选: 成交额上限(剔除过热巨量票)
        df = df[df["amount"] <= uni_cfg["amount_max_yi"] * 1e8]
    df = df[df["turnover_rate"] >= uni_cfg["turnover_min"]]
    stats["liquidity"] = int(len(df))

    df = df[df["bars"] >= uni_cfg["listed_min_bars"]]

    # ---- 各模式量价条件与评分 ----
    if mode == "dip":
        # 趋势内缩量回调: T日选出观察池; T+1集合竞价按开盘价直接买入(无缺口门槛)
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
        if w.get("ivol"):
            score = score + w["ivol"] * _zs(df["ivol20"])                # 特质波动率越高(分歧越大, 深低开修复越强)越好
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

    elif mode == "reversal":
        # 超跌反转(第七十九轮): T日深跌后盘中出现明显承接(长下影+收盘拉回上沿),
        # 博次日开盘的低开修复。与dip互补: dip要求趋势内浅回调, 本模式要的是趋势外深跌。
        df = df[df["turnover_rate"] <= uni_cfg["turnover_max"]]
        df = df[df["pct"].between(sel_cfg.get("rev_pct_min", -9.5),
                                  sel_cfg.get("rev_pct_max", -4.0))]
        df = df[df["close"] > df["low_limit"]]              # 收盘未跌停
        df = df[df["low"] < df["high"]]                     # 非一字
        df = df[df["vol_ratio"] >= sel_cfg.get("rev_vol_ratio_min", 1.2)]  # 放量换手(恐慌盘出清)
        df = df[df["close_pos"] >= sel_cfg.get("rev_close_pos_min", 0.60)]  # 收盘拉回区间上沿
        df = df[df["low_tail"] >= sel_cfg.get("rev_low_tail_min", 0.8)]     # 长下影承接
        df = df[df["mom20"] <= sel_cfg.get("rev_mom20_max", 5.0)]           # 前期已跌(超跌)
        df = df[df["dist_low20"] <= sel_cfg.get("rev_dist_low20_max", 15.0)]  # 贴近20日低点
        stats["trend_ok"] = int(len(df))
        if df.empty:
            return [], stats
        score = (
            0.40 * _zs(df["close_pos"])
            + 0.20 * _zs(df["low_tail"])
            + 0.20 * _zs(-df["mom20"])
            + 0.20 * _zs(-df["dist_low20"])
        )

    elif mode == "squeeze":
        # 低波缩量横盘(第七十九轮): 波动率与振幅双低的窄幅整理票, 温和放量小幅上攻,
        # 博"低波→启动"的次日惯性。与dip互补: dip是回调阴线, 本模式是低波阳线。
        df = df[df["turnover_rate"] <= uni_cfg["turnover_max"]]
        df = df[df["pct"].between(sel_cfg.get("sq_pct_min", 0.5),
                                  sel_cfg.get("sq_pct_max", 5.0))]
        df = df[df["close"] < df["high_limit"] - 1e-9]
        df = df[df["low"] < df["high"]]
        df = df[df["vol_ratio"].between(sel_cfg.get("sq_vol_ratio_min", 1.0),
                                        sel_cfg.get("sq_vol_ratio_max", 2.5))]
        df = df[df["amp_pct"] <= sel_cfg.get("sq_amp_max", 4.0)]           # 当日低振幅
        ma_ok = (df["close_adj"] > df["ma5"]) & (df["ma5"] > df["ma10"]) & (df["ma10"] > df["ma20"])
        df = df[ma_ok]
        df = df[df["bias20"].abs() <= sel_cfg.get("sq_bias_max", 5.0)]     # 贴合均线(未加速)
        df = df[df["mom20"].between(sel_cfg.get("sq_mom20_min", 0.0),
                                    sel_cfg.get("sq_mom20_max", 15.0))]
        if len(df):
            q = sel_cfg.get("sq_ivol_q", 0.35)
            df = df[df["ivol20"] <= df["ivol20"].quantile(q)]              # 截面低波分位
        stats["trend_ok"] = int(len(df))
        if df.empty:
            return [], stats
        score = (
            0.30 * _zs(-df["ivol20"])
            + 0.20 * _zs(-df["amp_pct"])
            + 0.20 * _zs(df["mom3"])
            + 0.20 * _zs(-df["bias20"])
            + 0.10 * _zs(np.log1p(df["amount"]))
        )

    elif mode == "pullback":
        # 趋势回踩5日线(第七十九轮): 中期强趋势票缩量回踩5日线不破, 博趋势延续。
        # 与dip互补: dip要求阴线深浅回调且20日动量<25%, 本模式允许强趋势(动量到45%)。
        df = df[df["turnover_rate"] <= uni_cfg["turnover_max"]]
        df = df[df["pct"].between(sel_cfg.get("pb_pct_min", -3.0),
                                  sel_cfg.get("pb_pct_max", 1.0))]
        df = df[df["close"] > df["low_limit"]]
        df = df[df["low"] < df["high"]]
        df = df[df["close_adj"] > df["ma20"]]
        df = df[df["ma20"] > df["ma20_prev"]]                              # 20日线向上
        df = df[df["mom20"].between(sel_cfg.get("pb_mom20_min", 12.0),
                                    sel_cfg.get("pb_mom20_max", 45.0))]
        df = df[df["dist_ma5"].between(sel_cfg.get("pb_dist_ma5_min", -3.0),
                                       sel_cfg.get("pb_dist_ma5_max", 2.0))]
        df = df[df["vol_ratio"] <= sel_cfg.get("pb_vol_ratio_max", 1.3)]    # 缩量回踩
        df = df[df["bias20"] <= sel_cfg.get("bias20_max", 8.0)]
        stats["trend_ok"] = int(len(df))
        if df.empty:
            return [], stats
        score = (
            0.35 * _zs(df["mom20"])
            + 0.25 * _zs(-df["dist_ma5"])
            + 0.20 * _zs(-df["vol_ratio"])
            + 0.20 * _zs(-df["bias20"])
        )

    elif mode == "limitup":
        # 涨停封板次日深低开承接(第七十九轮): T日封死涨停、位置不高, T+1恐慌低开时接,
        # 博获利盘出尽后的修复。与dip互补: dip是缩量阴线, 本模式是放量涨停。
        df = df[df["turnover_rate"] <= uni_cfg["turnover_max"]]
        df = df[df["close"] >= df["high_limit"] - 1e-9]        # 收盘封板
        df = df[df["low"] < df["high"]]                        # 非一字板(盘中可成交)
        df = df[df["pct"] >= sel_cfg.get("lu_pct_min", 4.5)]
        df = df[df["mom20"] <= sel_cfg.get("lu_mom20_max", 40.0)]  # 非高位连板
        df = df[df["bias20"] <= sel_cfg.get("lu_bias20_max", 12.0)]
        stats["trend_ok"] = int(len(df))
        if df.empty:
            return [], stats
        score = (
            0.40 * _zs(-df["bias20"])
            + 0.30 * _zs(-np.log1p(df["amount"]))
            + 0.20 * _zs(df["mom20"])
            + 0.10 * _zs(-df["vol_ratio"])
        )

    elif mode == "spike":
        # 放量突破次日深低开承接(第七十九轮): T日放量中阳、收盘靠近当日高位,
        # T+1回抽深低开时接。与dip互补: dip要缩量, 本模式要放量。
        df = df[df["turnover_rate"] <= uni_cfg["turnover_max"]]
        df = df[df["pct"].between(sel_cfg.get("sp_pct_min", 3.0),
                                  sel_cfg.get("sp_pct_max", 7.0))]
        df = df[df["close"] < df["high_limit"] - 1e-9]
        df = df[df["low"] < df["high"]]
        df = df[df["vol_ratio"].between(sel_cfg.get("sp_vol_ratio_min", 1.5),
                                        sel_cfg.get("sp_vol_ratio_max", 3.5))]
        df = df[df["close_pos"] >= sel_cfg.get("sp_close_pos_min", 0.65)]
        df = df[df["mom20"].between(sel_cfg.get("sp_mom20_min", 0.0),
                                    sel_cfg.get("sp_mom20_max", 30.0))]
        df = df[df["bias20"] <= sel_cfg.get("bias20_max", 8.0)]
        stats["trend_ok"] = int(len(df))
        if df.empty:
            return [], stats
        score = (
            0.35 * _zs(df["close_pos"])
            + 0.25 * _zs(df["vol_ratio"])
            + 0.20 * _zs(df["mom20"])
            + 0.20 * _zs(-df["bias20"])
        )

    elif mode == "oversold":
        # 深跌趋势中的阳线反转(第七十九轮): 已深跌(mom20很负)的票T日收阳、盘中承接,
        # T+1再深低开时接。与dip互补: dip只在上升趋势内做回调, 本模式专门做下跌趋势末端。
        df = df[df["turnover_rate"] <= uni_cfg["turnover_max"]]
        df = df[df["pct"].between(sel_cfg.get("os_pct_min", 1.0),
                                  sel_cfg.get("os_pct_max", 6.0))]
        df = df[df["close"] > df["low_limit"]]
        df = df[df["low"] < df["high"]]
        df = df[df["mom20"] <= sel_cfg.get("os_mom20_max", -10.0)]      # 20日深跌
        df = df[df["dist_low20"] <= sel_cfg.get("os_dist_low20_max", 10.0)]  # 贴近20日低点
        df = df[df["close_pos"] >= sel_cfg.get("os_close_pos_min", 0.60)]
        df = df[df["vol_ratio"] >= sel_cfg.get("os_vol_ratio_min", 1.2)]
        stats["trend_ok"] = int(len(df))
        if df.empty:
            return [], stats
        score = (
            0.35 * _zs(-df["mom20"])
            + 0.25 * _zs(df["close_pos"])
            + 0.20 * _zs(-df["dist_low20"])
            + 0.20 * _zs(df["vol_ratio"])
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
        elif mode == "reversal":
            reasons = [
                f"深跌{r['pct']:.1f}%",
                f"收盘位置{r['close_pos']:.2f}(拉回)",
                f"下影{r['low_tail']:.1f}%",
                f"量比{r['vol_ratio']:.1f}(放量)",
                f"距20日低{r['dist_low20']:.1f}%",
            ]
        elif mode == "squeeze":
            reasons = [
                f"涨幅{r['pct']:.1f}%",
                f"振幅{r['amp_pct']:.1f}%(低波)",
                f"IVOL{r['ivol20']:.3f}",
                f"乖离{r['bias20']:.1f}%",
                "均线多头",
            ]
        elif mode == "pullback":
            reasons = [
                f"回踩{r['pct']:.1f}%",
                f"距5日线{r['dist_ma5']:.1f}%",
                f"量比{r['vol_ratio']:.2f}(缩量)",
                f"20日动量{r['mom20']:.0f}%",
                "20日线向上",
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
            "strategy": mode,
            "strategy_cn": STRATEGY_CN.get(mode, mode),
        })
    return picks, stats


def select_stocks(
    date: str,
    day_df: pd.DataFrame,
    cfg: dict,
    names: dict[str, str] | None = None,
    top_n: int | None = None,
) -> tuple[list[dict], dict]:
    """多策略出池: 主力池 + 依序排列的备用池(第七十八轮)。

    每个 pick 带 "pool" 字段: main=主力, fb1/fb2...=第N备用池。
    引擎按 pool 顺序顺延买入 —— 主力池当日全部缺达标票时, 自动落到备用池,
    实现"没有票就切换另一个策略"。备用池按 selection.fallbacks 配置, 缺省为空(单策略)。
    """
    picks, stats = _select_one(date, day_df, cfg, names, top_n)
    for p in picks:
        p["pool"] = "main"
    if stats.get("weekday_skip"):        # 非出池日: 备用池同样不出票
        return picks, stats
    sel_cfg = cfg["selection"]
    fbs = sel_cfg.get("fallbacks") or []
    if not fbs or sel_cfg.get("mode") != "dip":
        return picks, stats
    # 单槽漏斗(第九十一轮): 主力池当日有候选时不再并入备用池 —— 网页仍然只给1只,
    # 备用策略只在"主力策略当日完全没有信号"的交易日补位。与"层层顺延"相比,
    # 这条模式避开的是主力已出票但竞价缺口不达标的日子(逆向选择最严重的一天)。
    if sel_cfg.get("fallback_slot") and picks:
        return picks, stats
    have = {p["code"] for p in picks}
    seen = {"main": 1}
    for i, fb in enumerate(fbs, 1):
        if not fb.get("enabled", True) or not fb.get("mode"):
            continue
        sub = {**cfg, "selection": {**sel_cfg, "mode": fb["mode"], "fallbacks": []}}
        fb_picks, _ = _select_one(date, day_df, sub, names, top_n=int(fb.get("top_n") or 0))
        for p in fb_picks:
            if p["code"] in have:          # 已在主力池/前序备用池中的票不重复计入
                continue
            have.add(p["code"])
            p["pool"] = f"fb{i}"
            p["reasons"] = f"{p.get('reasons', '')}｜备用{i}·{p.get('strategy_cn', '')}".lstrip("｜")
            picks.append(p)
            seen[p["pool"]] = seen.get(p["pool"], 0) + 1
    stats["fallback_counts"] = {k: v for k, v in seen.items() if k != "main"}
    return picks, stats

