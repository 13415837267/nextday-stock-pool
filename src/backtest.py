# -*- coding: utf-8 -*-
"""实盘口径复利回测引擎。

执行规则(与实盘完全一致):
  - T日收盘后按 selector 选出候选池(次日计划)
  - T+1 开盘竞价买入: 先按竞价缺口等条件筛出实际可买名单(保持评分排序),
    每笔买入占组合总资产 alloc_frac_pct%(与达标数量解耦, 防恐慌晨逆向选择),
    按评分排序每日最多买入 max_buy_per_day 只(0=不限)
    * 买入口径(第九十二轮): 计划票 T+1 开盘价直接成交, buy_gap_below=0 即无缺口门槛;
      若配负值则退化为"仅买开盘缺口 <= 阈值% 的深低开票"(历史挑单口径)。
    * T+1 停牌/无成交/开盘一字涨停/开盘涨幅 >= skip_open_gain% -> 放弃该票
    * 可选: 开盘跌幅 <= -skip_open_loss%(低开过深, 反转失败概率高) -> 放弃该票
  - T+2 卖出(腿比例由 execution.sell_open_frac 决定: 1=开盘竞价全额卖, 0=当日收盘
    14:55 全额卖, 0<f<1=开盘卖 f、收盘卖 1-f; 现行 config 为 0, 即全额收盘卖)
    * T+2 停牌 -> 顺延至后续交易日开盘卖出(最多5日, 否则按最后收盘价)
    * T+2 开盘跌停且开启保守模式 -> 按当日收盘价卖出(仅对开盘腿生效)
  - 成本: 佣金万2.5(双边) + 印花税万5(卖出) + 滑点千1(双边)
  - 资金全部复利滚动, 卖出资金当日开盘即可再投入
  - 大盘风控触发时次一交易日空仓(不买入)
  - 严格遵守A股T+1制度: 当日买入的股票当日不可卖出, 无任何"当日买入
    当日卖出"规则(无尾盘止损/尾盘止盈/风控尾盘离场), 亦无未来函数——
    每一步决策仅使用决策时点之前已可得的信息(信号用T日收盘数据,
    买卖用对应时点的开盘/收盘价)。

接口:
  - prepare_context(cfg, days): 构建行情面板/指标/风控上下文(耗时主体, 可复用)
  - run_backtest(cfg, days, ctx): 执行逐日模拟与绩效汇总; 传入 ctx 可跳过重复构建
"""
from __future__ import annotations

import argparse
import json
import math
import os
import time
from datetime import datetime, timedelta, timezone

TZ_CN = timezone(timedelta(hours=8))  # 北京时间: Actions 运行于 UTC, 展示型时间戳统一 +8
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, load_config
from .data_api import ZZClient
from .rule_text import rules_line
from .selector import apply_regime, build_indicators, regime_base, select_stocks, split_by_date


# ----------------------------------------------------------------------
def _downcast_float(df: pd.DataFrame) -> pd.DataFrame:
    """数值列 float64->float32(就地): 价格数据7位有效数字足够, 内存减半。"""
    fc = df.select_dtypes(include="float64").columns
    if len(fc):
        df[fc] = df[fc].astype("float32")
    return df


class _LazyPerCode:
    """逐票历史懒加载索引(2026-10-24第四十二轮, 长窗口内存优化)。

    原实现: prepare_context 对全市场面板 groupby("ts_code")+set_index——
    整张面板被完整复制一份(2500日约1400万行, 内存直接翻倍, 沙盒OOM主因)。
    实际回测仅访问"被交易/被持仓"的票(十年约数千只), 改为按需构建
    轻量单票视图(_PerCodeView): 仅缓存该票存在的日期列表(~20KB/票),
    行数据实时从 by_date(CoW惰性视图, 近零内存)按行号取, 零DataFrame拼装
    (拼装版实测: 500日窗口独特票数千只, 每票concat全窗口行~0.5s, 模拟
    阶段超20分钟; 视图版单行查询~0.15ms)。

    反向索引: 逐日ts_code数组+searchsorted(方案A, ~95MB), 乱序日回退
    groupby.indices字典(方案B)。

    接口兼容 dict: .get(code) 与 [code], 值为 _PerCodeView(模拟
    DataFrame(index=trade_date) 的 .loc/.index/.iloc/len 子集)或 None。
    """

    def __init__(self, by_date: dict[str, pd.DataFrame]):
        self.by_date = by_date
        self._dates = sorted(by_date)   # 键=YYYYMMDD字符串, 字典序即时间序
        self.date_i = {d: i for i, d in enumerate(self._dates)}
        self._cache: dict[str, pd.DataFrame | None] = {}
        # 反向索引方案A(默认): 逐日ts_code的numpy数组+searchsorted定位。
        # 前提: 逐日帧按代码序(源数据如此, 启动时逐日校验, 乱序则回退方案B)。
        # 2660日×4400码 ≈ 95MB。
        # 方案B(回退): groupby.indices字典——每日~4400个小ndarray的对象头
        # 开销~0.5MB, 2660日累计~1.3GB(第四十二轮实测OOM元凶), 仅乱序时用。
        self._codes: list | None = []
        for d in self._dates:
            arr = by_date[d]["ts_code"].to_numpy()
            self._codes.append(arr)
            if arr.size > 1 and not pd.Index(arr).is_monotonic_increasing:
                self._codes = None
                break
        if self._codes is None:
            self.pos = [by_date[d]["ts_code"].groupby(by_date[d]["ts_code"]).indices
                        for d in self._dates]

    def _pos(self, i: int, code: str) -> int:
        """code在第i个日期帧中的行号(每日唯一), 不存在返回-1。"""
        if self._codes is not None:
            arr = self._codes[i]
            lo = int(np.searchsorted(arr, code))
            return lo if lo < arr.size and arr[lo] == code else -1
        pm = self.pos[i].get(code)
        return int(pm[0]) if pm is not None else -1

    def get(self, code: str) -> "_PerCodeView | None":
        if code in self._cache:
            return self._cache[code]
        dates = [d for i, d in enumerate(self._dates) if self._pos(i, code) >= 0]
        out = _PerCodeView(self, code, dates) if dates else None
        self._cache[code] = out
        return out

    def __getitem__(self, code: str) -> "_PerCodeView | None":
        return self.get(code)


class _PerCodeView:
    """单票历史轻量视图: 仅实现回测引擎用到的 DataFrame 子集接口。

    兼容用法(见 run_backtest): d in view.index / view.index.get_loc(d) /
    view.index[i] / view.loc[d](单行Series) / view.iloc[j](含-1) / len(view)。
    """

    def __init__(self, store: _LazyPerCode, code: str, dates: list[str]):
        self._s, self._c, self._dates = store, code, dates
        self._idx = _DateIndex(dates)

    def _row_series(self, d: str) -> pd.Series:
        i = self._s.date_i[d]
        p = self._s._pos(i, self._c)   # 每日代码唯一, p即行号
        return self._s.by_date[d].iloc[p]

    @property
    def loc(self) -> "_Loc":
        return _Loc(self)

    @property
    def iloc(self) -> "_ILoc":
        return _ILoc(self)

    @property
    def index(self) -> "_DateIndex":
        return self._idx

    def __len__(self) -> int:
        return len(self._dates)


class _Loc:
    def __init__(self, view: _PerCodeView):
        self._v = view

    def __getitem__(self, d: str) -> pd.Series:
        return self._v._row_series(d)


class _ILoc:
    def __init__(self, view: _PerCodeView):
        self._v = view

    def __getitem__(self, j: int) -> pd.Series:
        return self._v._row_series(self._v._dates[j])


class _DateIndex:
    """类Index: 支持 in / get_loc / [i] / len。"""

    def __init__(self, dates: list[str]):
        self._d = dates
        self._m = {d: i for i, d in enumerate(dates)}

    def __contains__(self, d) -> bool:
        return d in self._m

    def get_loc(self, d) -> int:
        return self._m[d]

    def __getitem__(self, i: int):
        return self._d[i]

    def __len__(self) -> int:
        return len(self._d)


def fetch_panel(client: ZZClient, dates: list[str], log_every: int = 50,
                float32: bool = False) -> pd.DataFrame:
    """按交易日抓取全市场日线并合并为长表(带缓存)。空数据日(未来日期)自动跳过。

    float32=True(回测专用): 数值列降float32 + 流式列拼装——逐日仅保留各列的
    numpy数组(零DataFrame包装), 最终逐列拼接并即时释放中间列表。峰值内存从
    "全量frames列表+全量concat结果"两份共存(2500日约3GB, 受限环境OOM)降至
    单份+ε(约1GB)。默认False走原路径, 实盘链路(daily_run)保持float64不变。
    """
    if not float32:
        frames = []
        t0 = time.time()
        for i, d in enumerate(dates, 1):
            df = client.market_day(d)
            if len(df):
                frames.append(df)
            if i % log_every == 0:
                print(f"  已抓取 {i}/{len(dates)} 个交易日, 累计 {time.time()-t0:.0f}s", flush=True)
        return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()

    num: dict[str, list] = {}
    codes: list[pd.Series] = []
    day_list: list[str] = []
    counts: list[int] = []
    order: list[str] = []
    _intern: dict[str, str] = {}   # ts_code字符串驻留: 各日csv各持独立字符串对象,
    t0 = time.time()               # 2500日1400万行不驻留则~850MB, 驻留后仅~120MB指针
    for i, d in enumerate(dates, 1):
        df = _downcast_float(client.market_day(d))
        if len(df):
            if not order:
                order = [c for c in df.columns if c not in ("ts_code", "trade_date")]
            for c in order:
                num.setdefault(c, []).append(df[c].to_numpy())
            codes.append(df["ts_code"].map(lambda s: _intern.setdefault(s, s)))
            day_list.append(d)
            counts.append(len(df))
        if i % log_every == 0:
            print(f"  已抓取 {i}/{len(dates)} 个交易日, 累计 {time.time()-t0:.0f}s", flush=True)
    data: dict = {}
    for c, parts in num.items():
        data[c] = np.concatenate(parts)
        num[c] = None   # 即时释放, 峰值=单份数据
    num = None
    data["ts_code"] = (pd.concat(codes, ignore_index=True) if codes
                       else pd.Series([], dtype="str"))
    codes = None
    if day_list:
        data["trade_date"] = pd.Series(day_list, dtype="object").repeat(
            np.asarray(counts)).reset_index(drop=True)
    return pd.DataFrame(data) if data else pd.DataFrame()


def prepare_context(cfg: dict, days: int | None = None) -> dict:
    """构建回测所需上下文(行情面板/指标/风控/逐票索引), 供多次模拟复用。"""
    bt_cfg = cfg["backtest"]
    n_days = days or bt_cfg["trading_days"]
    warmup = bt_cfg["warmup_days"]

    client = ZZClient(cfg)
    print(f"[1/4] 获取交易日历与行情面板(主窗口{n_days}日+预热{warmup}日)...")
    all_days = client.trade_days(days=n_days + warmup + 5)
    if len(all_days) < n_days + warmup:
        raise RuntimeError(f"交易日不足: 需要{n_days + warmup}日, 仅获取{len(all_days)}日")
    panel_dates = all_days[-(n_days + warmup):]
    panel = fetch_panel(client, panel_dates, float32=True)
    print(f"[2/4] 计算指标(共{len(panel):,}行)...")
    panel = build_indicators(panel)
    panel = _downcast_float(panel)   # 指标新增列同步降float32(长窗口内存)
    by_date = split_by_date(panel)
    reg_base = regime_base(panel, cfg)
    reg_base = reg_base.astype("float64")   # 风控序列保持float64(均线/累计比较精度)
    reg = apply_regime(reg_base, cfg)
    del panel   # 数据已被 by_date 惰性视图引用, 释放长表对象本身
    # 主窗口仅保留实际有行情的日期(剔除日历中的未来日/无数据日)
    main_dates = [d for d in panel_dates[warmup:] if d in by_date]

    # 逐票历史懒加载(第四十二轮): 替代原groupby全量复制面板(长窗口OOM主因),
    # 见 _LazyPerCode 文档。接口兼容 dict.get / dict[code]。
    per_code = _LazyPerCode(by_date)

    names = {}
    try:
        sb = client.stock_names()
        names = dict(zip(sb["ts_code"], sb["name"]))
    except Exception:
        pass
    return {"main_dates": main_dates, "by_date": by_date, "reg": reg, "reg_base": reg_base,
            "per_code": per_code, "names": names}


def run_backtest(cfg: dict, days: int | None = None, ctx: dict | None = None) -> dict:
    if ctx is None:
        ctx = prepare_context(cfg, days)
    main_dates, by_date = ctx["main_dates"], ctx["by_date"]
    # 风控参数按当前cfg实时重建(market_ma/breadth_min可扫描; 基础序列复用, 无需重算指标)
    reg = apply_regime(ctx["reg_base"], cfg) if "reg_base" in ctx else ctx["reg"]
    # 最大连续空仓上限(可选): 连续风控天数超过 max_block_days 后, 按 block_reenter_frac
    # 降仓恢复参与(短期风控仍全额避险, 仅在超长空仓期小仓参与); 0=禁用(默认)
    _rm = cfg.get("regime") or {}
    max_block, reenter_frac = int(_rm.get("max_block_days", 0) or 0), float(_rm.get("block_reenter_frac", 0.0) or 0.0)
    rf_ramp = bool(_rm.get("rf_ramp", False))   # 渐近恢复: 超限后每多1日风控, 系数+0.05直至1.0
    day_frac: dict[str, float] = {}
    if max_block > 0 and reenter_frac > 0 and len(reg):
        has_panic = "panic" in reg.columns
        consec = 0
        for d_i in reg.index:
            if bool(reg.loc[d_i, "trade_ok"]):
                consec = 0
                day_frac[d_i] = 1.0
            elif has_panic and bool(reg.loc[d_i, "panic"]):
                day_frac[d_i] = 0.0   # 恐慌型波动日: 全额空仓且不计入连续风控天数
                                      # (防恐慌期被超限参与机制绕过熔断; 2026-10-23定稿)
            else:
                consec += 1
                if consec <= max_block:
                    day_frac[d_i] = 0.0
                elif rf_ramp:
                    day_frac[d_i] = min(1.0, reenter_frac + 0.05 * (consec - max_block))
                else:
                    day_frac[d_i] = reenter_frac
    # 风控解除后渐进建仓(第一百一十五轮新增, 可选; reentry_ramp_days=0 即关闭, 与旧口径逐位等价):
    # 指数刚重新站上 MA20 的头 N 个交易日, 方向尚未被确认 —— 第一百零八轮把 market_ma 30→20
    # 换来"更快抓住反弹初期"的收益, 代价正是 2015 年 -15.6pp 的"接假反弹"。折中: 反弹头几天
    # 照常参与(不必等到 MA30 才敢动手), 但按线性 ramp 从 start 系数逐步加回满仓, 第 N 日恢复。
    # 与 rf_ramp 的区别: rf_ramp 作用于"风控持续期内"的系数, 本项作用于"风控刚结束"的建仓节奏,
    # 两者互不重叠; 只用截至当日已实现的风控状态, 无未来函数。
    _rrd = int(_rm.get("reentry_ramp_days", 0) or 0)
    _rrs = float(_rm.get("reentry_ramp_start", 0.5) or 0.5)
    reentry_map: dict[str, float] = {}
    if _rrd > 0 and len(reg):
        _streak = 0
        for _d in reg.index:
            if bool(reg.loc[_d, "trade_ok"]):
                _streak += 1
                if _streak <= _rrd:
                    _f = _rrs + (1.0 - _rrs) * (_streak - 1) / max(1, _rrd - 1)
                    reentry_map[_d] = min(1.0, _f)
            else:
                _streak = 0
    per_code, names = ctx["per_code"], ctx["names"]
    bt_cfg = cfg["backtest"]

    ex = cfg["execution"]
    comm, stamp, slip = ex["commission_rate"], ex["stamp_tax"], ex["slippage"]
    # 第一百六十二轮: 印花税时变。2023-08-28 起卖出印花税由千 1 降至万 5;
    # config 的 stamp_tax=0.0005 是**降税后**口径, 全程套用等于给 2012-2023 的每一笔
    # 少算 0.05%。开关默认关闭以保持历史数值可复现, 打开后按卖出日切换。
    _stamp_var = bool(ex.get("stamp_tax_time_varying"))

    def _stamp_at(r) -> float:
        if not _stamp_var:
            return stamp
        return 0.001 if str(r.get("trade_date", "")) < "20230828" else stamp
    gap_tiers = ex.get("gap_size_tiers") or []   # 缺口分级下注: [{below: -4.0, mult: 1.0}, ...]
    gap_above_tiers = ex.get("gap_above_tiers") or []   # 高开降仓: [{above: 0, mult: 0.5}, ...]
    gap_floor_tiers = ex.get("gap_floor_tiers") or []   # 深低开加仓下限: [{below: -1, min_pool: 11, floor: 0.8}]
    pool_tiers = ex.get("pool_n_tiers") or []           # 池深分档下注: [{min: 41, mult: 1.0}, ...]
    knife_tiers = ex.get("knife_tiers") or []           # 阴跌刀口保护(第一百零三轮): [{mkt_above, res_below, mult}]
    mkt_floor_tiers = ex.get("mkt_floor_tiers") or []   # 全市场恐慌日回满仓: [{mkt_below, floor}]
    mkt_boost_tiers = ex.get("mkt_boost_tiers") or []   # 第一百七十三轮: 恐慌日超配 [{mkt_below, mult}]
    # 信号日"市场典型个股涨跌"(全市场 pct 中位数): 只用 T 日收盘后的横截面, 无未来函数。
    # 用于把"大盘一起跌"(系统性错杀, 次日修复有共振)与"大盘平稳而个股独跌"(个股有事)分开。
    mkt_ret_map: dict[str, float] = {}
    if knife_tiers or mkt_floor_tiers or mkt_boost_tiers:
        for _d in main_dates:
            _dd = by_date.get(_d)
            if _dd is None:
                continue
            _v = _dd["pct"].to_numpy(dtype="float64")
            _v = _v[~np.isnan(_v)]
            if len(_v):
                mkt_ret_map[_d] = float(np.median(_v))
    # 第一百七十四轮: 市场跌幅的滚动分位。
    # 固定百分点阈值有个结构性缺陷: 高波动年份跌 1% 是日常, 低波动年份跌 1% 是事件,
    # 同一条 -1.0% 在两个年份里筛出的样本密度完全不同。分位把"跌"标准化成
    # "在最近 N 日里跌得比多少比例的日子更狠", 于是阈值自动随波动率迁移。
    # 默认 0 = 不启用, 全部走原绝对阈值路径, 逐位等价。
    mkt_rank_map: dict[str, float] = {}
    _rank_win = int(ex.get("mkt_rank_win", 0) or 0)
    if _rank_win > 0 and mkt_ret_map:
        _seq = pd.Series([mkt_ret_map.get(_d, np.nan) for _d in main_dates],
                         index=list(main_dates), dtype="float64")

        def _rk_last(w: np.ndarray) -> float:
            v = w[-1]
            if not np.isfinite(v):
                return np.nan
            a = w[:-1]
            a = a[np.isfinite(a)]
            if len(a) < 20:      # 样本太薄的分位没有意义, 宁可不出信号
                return np.nan
            return float((a < v).mean())

        _rks = _seq.rolling(_rank_win, min_periods=max(40, _rank_win // 5)).apply(_rk_last, raw=True)
        mkt_rank_map = {str(k): float(v) for k, v in _rks.items() if np.isfinite(v)}
    min_top1_score = ex.get("min_top1_score")     # 可选: 当日top1评分低于该值则放弃买入(质量门槛)
    gate_scale = bool(ex.get("score_gate_pool_scale", False))   # 第一百轮: 门槛按池内z分理论上限缩放
    gate_ref = float(ex.get("score_gate_ref_pool", 40) or 40)    # 缩放基准池深(该池深下门槛=min_top1_score)
    # 第九十六轮: 评分分档下注 [{min: 1.5, mult: 1.0}, {min: 1.1, mult: 0.6}, ...] 降序匹配
    # —— 门槛降低后不再"一刀切空仓", 低分票只给小仓, 高分票才满仓; 兼顾笔数与期望。
    score_tiers = ex.get("score_tiers") or []
    skip_gap_deep = ex.get("skip_gap_deep")       # 可选: 缺口深于该值(更恐慌)也放弃(崩塌保护)
    # 第一百七十轮: 融资杠杆。lev=1.0 时与旧口径逐位等价(cash 不可为负)。
    #   lev>1 时允许透支 (lev-1)*equity 买入, 透支额按 margin_rate 年化逐日计息。
    #   实盘含义: 自有资金 1 份 + 融资 (lev-1) 份; 单票集中度/标的资格由券商约束, 回测不建模。
    lev = float(ex.get("leverage", 1.0) or 1.0)
    lev_rate = float(ex.get("margin_rate", 0.055) or 0.0)
    # 第一百七十三轮: 闲置现金收益。策略年均只有约 24% 的交易日有持仓, 其余时间现金
    # 完全空转(收益 0)。实盘上这部分钱可以买国债逆回购/货币 ETF, 年化约 1.5~2%。
    # 这里按"正现金余额 × 年化 / 252"逐日计提, 不改变任何买卖判定(纯增益)。
    # 缺省 0.0 = 旧口径逐位等价。
    cash_yield = float(ex.get("cash_yield", 0.0) or 0.0)
    cash = float(bt_cfg["initial_capital"])
    interest_total = 0.0
    yield_total = 0.0
    curve: dict[str, float] = {}
    trades: list[dict] = []
    pending_buy: tuple[str, list[dict]] | None = None   # (buy_date, picks)
    holding: list[dict] = []                            # 未平仓持仓
    # 第一百二十五轮: 把原来的单一 empty_days 拆成两个指标 ——
    #   risk_empty_days: 风控/连亏熔断导致"当日不出计划、不买入"的天数;
    #   flat_days:       收盘时账户无任何持仓的交易日数(读者直觉里的"空仓天数")。
    # 原 empty_days 在 blocked_frac=0.40 下恒为 0(风控日仍可四成仓接深跌, 不判 blocked),
    # 且 loss_streak_pause 未配置 -> paused 恒 False; 指标名为"空仓天数"却永远显示 0,
    # 是"指标恒为常量"的典型陷阱。两个新指标各有真实信息量。
    risk_empty_days = 0
    flat_days = 0
    fee_total = 0.0
    pending_cash = 0.0   # T+2收盘卖模式: 当日收盘卖出回笼、次日开盘方可投入的资金
    # 第一百六十六轮诊断(仅 BT_DEBUG_CASH=1 时启用, 不改变任何成交):
    # 记录"有意愿买、但资金未回笼(现金不足目标仓位 min_fill_frac)"而被静默跳过的信号。
    # 这批是"现金锁"成本 —— 卖出在 T+2 收盘、回款次日开盘才可用, 于是隔日连发的
    # 信号会被吞掉; 想知道吞掉的是不是好票, 必须先能看见它。
    cash_blocked: list[dict] = [] if os.environ.get("BT_DEBUG_CASH") else []
    # 连亏熔断(可选): 连续亏损达 streak 笔后暂停买入 pause_days 个交易日(0=禁用)
    _ps = (cfg.get("risk") or {}).get("loss_streak_pause") or {}
    ps_streak, ps_days = int(_ps.get("streak", 0)), int(_ps.get("pause_days", 5))
    loss_streak = 0
    pause_until = -1
    # 月度盈利熔断(可选): 当月已实现收益超过 cap_pct% 后, 后续买入仓位乘以 cut_frac
    # (抑制单月利润滚雪球, 降低盈利月度集中度; 仅用已实现信息, 无未来函数)
    _mc = (cfg.get("risk") or {}).get("month_profit_cap") or {}
    mc_cap, mc_cut = float(_mc.get("cap_pct", 0) or 0), float(_mc.get("cut_frac", 1.0) or 1.0)
    _ml = (cfg.get("risk") or {}).get("month_loss_cap") or {}
    ml_cap, ml_cut = float(_ml.get("loss_pct", 0) or 0), float(_ml.get("cut_frac", 1.0) or 1.0)
    month_key, month_start_nav = None, None

    def do_buy(picks: list[dict], d: str, idx: int = 0):
        nonlocal cash, fee_total, interest_total
        frac_buy_open = min(1.0, max(0.0, float(ex.get("buy_open_frac", 1.0) or 0.0)))
        if not picks:
            return
        _sel = cfg.get("selection") or {}
        _env_k = os.environ.get("BT_TOPK")
        _topk = int(_env_k) if _env_k is not None else int(_sel.get("exec_top_n") or 0)
        # 第八十五轮: 绩效门控的池深(可选, 默认关闭) —— 仅用"已结算历史成交"判断深度,
        # 无未来函数: 近 window 笔平均收益 > min_avg 时扩到 hot_top_n, 否则收缩到 cold_top_n。
        # 目的: era 前(单笔为负)自动保持浅池, era 后(单笔为正)自动扩池吃满顺延空间。
        _dg = ex.get("depth_gate") or {}
        if _dg and _env_k is None:
            _w = int(_dg.get("window", 20))
            _real = [t["ret_pct"] for t in trades if t.get("ret_pct") is not None]
            if len(_real) >= _w:
                _hot = sum(_real[-_w:]) / _w > float(_dg.get("min_avg", 0.0))
                _topk = int(_dg.get("hot_top_n", _topk)) if _hot else int(_dg.get("cold_top_n", _topk))
        # 当日风控状态与承接线: 超限小仓参与日可用专属承接线 block_gap_below
        # (弱市环境参与时要求更深承接, 提高胜率; 缺省沿用常规承接线)
        reg_row = reg.loc[d] if d in reg.index else None
        day_ok = reg_row is None or bool(reg_row["trade_ok"])
        floor = ex.get("buy_gap_below", 0)
        if not day_ok:
            # 第一百一十三轮"分档参与": 软参与日(指数<MA20 但宽度尚可且非恐慌)沿用常规承接线
            # —— 这类日子是权重拖累/温和调整, 不必要求"恐慌深跌"才接, 否则与 blocked_frac 同义。
            if reg_row is not None and bool(reg_row.get("soft_ok", False)):
                pass
            # 第一百零二轮: 风控日的"专属承接线"不再依赖超限复参与机制(max_block_days=0 时
            # day_frac 为空、原逻辑形同虚设)。只要配置了 block_gap_below 就在风控日生效 ——
            # 风控期只接"恐慌深跌"这一种票, 其余日照常空仓。
            elif day_frac and day_frac.get(d, 0) > 0:
                floor = ex.get("block_gap_below", floor)
            elif ex.get("block_gap_below"):
                floor = float(ex.get("block_gap_below"))
        # ---- 多策略顺延(第七十八轮): 主力池 -> 备用池依序切换 ----
        # 主力池截到 exec_top_n; 备用池(fb1/fb2...)按各自 top_n 截断, 并按 segment 分段启用:
        #   all=全时段可用 / trend=仅指数>=MA30 / weak=仅指数<MA30
        # 主力池当日无票达标时, 循环自然落到备用池 —— 即"一个策略没票就切另一个"。
        _fbs = _sel.get("fallbacks") or []
        _main = [p for p in picks if p.get("pool", "main") == "main"]
        _fbp = [p for p in picks if p.get("pool", "main") != "main"]
        if _topk > 0:
            _main = _main[:_topk]
        if _fbs and _fbp:
            _up = (reg_row is not None and not pd.isna(reg_row.get("mkt_ma", np.nan))
                   and float(reg_row["mkt_idx"]) >= float(reg_row["mkt_ma"]))
            _kept = []
            for p in _fbp:
                i = int(str(p.get("pool"))[2:]) - 1
                fb = _fbs[i] if 0 <= i < len(_fbs) else {}
                seg = fb.get("segment", "all")
                if (seg == "trend" and not _up) or (seg == "weak" and _up):
                    continue
                if sum(1 for q in _kept if q.get("pool") == p.get("pool")) >= int(fb.get("top_n") or 0):
                    continue
                p = {**p, "_gap_below": fb.get("gap_below"),
                     "_gap_above": fb.get("gap_above"),
                     # gap_own=true: 本池**不继承**当日风控承接线(block_gap_below), 用自己的
                     # gap_above/gap_below 判定(第一百三十二轮)。用于 trend 这类"低开即破坏"的
                     # 策略: 风控日的 -2.5% 是 dip 的专属承接线, 套到趋势票上会把它全部挡掉。
                     # 注意风控日的**仓位**仍按 blocked_frac 执行 —— 风控管仓位, 不管缺口口径。
                     "_gap_own": bool(fb.get("gap_own", False))}
                _kept.append(p)
            _fbp = _kept
        picks = _main + _fbp
        # 第一遍: 先按竞价条件筛出实际可买名单(保持评分排序, 与实盘一致)
        # _rank=该池内评分名次(1起, 第五十八轮诊断字段): 主力/备用各自编号, 便于审计顺延深度
        _cnt: dict[str, int] = {}
        for p in picks:
            k = p.get("pool", "main")
            _cnt[k] = _cnt.get(k, 0) + 1
            p["_rank"] = _cnt[k]
        qualified = []
        for p in picks:
            sub = per_code.get(p["code"])
            row = sub.loc[d] if sub is not None and d in sub.index else None
            if row is None or row["is_paused"] == 1 or row["vol"] <= 0:
                trades.append(_trade(p, "skip", d, note="停牌/无成交"))
                continue
            if row["open"] >= row["high_limit"] - 1e-9:
                trades.append(_trade(p, "skip", d, note="开盘一字涨停"))
                continue
            sig_row = by_date[p["signal_date"]]
            prev_close = sig_row.loc[sig_row["ts_code"] == p["code"], "close_adj"]
            gap = row["open_adj"] / float(prev_close.iloc[0]) - 1 if len(prev_close) else 0
            # 信号日个股涨跌(用于"刀口保护": 与市场典型个股涨跌相减得超额跌幅)
            _pct_sig = None
            if knife_tiers or mkt_floor_tiers or mkt_boost_tiers:
                _pr = sig_row.loc[sig_row["ts_code"] == p["code"], "pct"]
                if len(_pr):
                    _pct_sig = float(_pr.iloc[0])
            if gap * 100 >= ex["skip_open_gain"]:
                trades.append(_trade(p, "skip", d, note=f"开盘高开{gap*100:.1f}%"))
                continue
            if ex.get("skip_open_loss", 0) and gap * 100 <= -abs(ex["skip_open_loss"]):
                trades.append(_trade(p, "skip", d, note=f"开盘低开{gap*100:.1f}%"))
                continue
            # dip 承接过滤: 仅买缺口 <= floor% 的深低开票(0=不过滤); 备用池可配专属承接线
            # ★第一百三十二轮修复(口径一致性): 备用池 gap_below 与当日 floor 必须取**更严**者,
            #   不能互相覆盖。原写法 `_fl = floor if _fl is None else _fl` 让备用池的
            #   gap_below=0 直接盖掉风控日的 block_gap_below=-2.5; 而 0 在下一行 `if _fl:`
            #   里是假值 -> 整条缺口判定被跳过。后果: 同一只票、同一个缺口, 回测按开盘价成交,
            #   实盘(daily_run 用 floor_now=block_gap_below, 对任何池都生效)却空仓。
            #   实测 1500 日 27 笔全部是备用池、全部缺口 >= -2.5%(即实盘都会空仓),
            #   虚增 +1263.82pp(8612.67% vs 7348.85%)、年化 +5.50pp —— 是回测虚高, 不是策略收益。
            _fl = p.get("_gap_below")
            _fa = p.get("_gap_above")      # 缺口下限(第一百三十二轮): 深低开=趋势破坏
            if p.get("_gap_own"):
                # 显式声明"用自己的缺口口径": 不受当日 floor 约束(见 _gap_own 注释)
                pass
            elif _fl is None:
                _fl = floor
            elif floor is not None:
                _fl = min(float(_fl), float(floor))
            # 第八十轮: 名次梯度承接线(可选) —— 池内名次越靠后要求缺口越深,
            # 用"更激进的折价"换深名次的质量, 让扩池新增的成交只发生在最极端的恐慌日。
            # 配置形如 [{rank_above: 5, gap_below: -3.0}, ...], 取第一个满足条件的档。
            for _t in (ex.get("gap_rank_tiers") or []):
                if int(p.get("_rank") or 0) >= int(_t.get("rank_above", 1)):
                    _tg = float(_t.get("gap_below", _fl))
                    _fl = min(_tg, float(floor)) if floor else _tg
            # 缺口下限(第一百三十二轮, 供 trend 等"低开即破坏"的策略): 低于该值直接放弃。
            # 不走限价承接 —— 趋势票要的是"当天就强", 等它跌到某个价再成交, 形态已经不是原来那只。
            if _fa is not None and gap * 100 < float(_fa):
                trades.append(_trade(p, "skip", d,
                                     note=f"缺口{gap*100:.1f}%低于下限{float(_fa)}%(趋势破坏)"))
                continue
            _lim_px = None   # 第八十轮: 盘中止损限价承接价(adj口径), None=按开盘价成交
            if _fl and gap * 100 > _fl:
                _lmt = float(ex.get("buy_limit_fill") or 0)
                # 限价承接(默认关闭): 开盘缺口不够深时, 在承接线价位挂全天限价买单,
                # 当日最低价触及则按该价成交(比开盘更深, 入场价确定)。仅当开盘未过热
                # (<= buy_limit_open_max%)时启用, 避免追"高开后再跳水"的票。
                if _lmt > 0 and gap * 100 <= float(ex.get("buy_limit_open_max", 0.0)):
                    _pc = float(prev_close.iloc[0])
                    _px = _pc * (1 + _fl / 100.0)
                    _f = (float(row["close_adj"]) / float(row["close"])
                          if float(row["close"]) else 0.0)
                    if _f > 0 and float(row["low"]) * _f <= _px + 1e-9:
                        _lim_px = _px
                        gap = _fl / 100.0
                if _lim_px is None:
                    trades.append(_trade(p, "skip", d, note=f"缺口{gap*100:.1f}%未达承接线{_fl}%"))
                    continue
            if skip_gap_deep is not None and gap * 100 < skip_gap_deep:
                trades.append(_trade(p, "skip", d, note=f"缺口{gap*100:.1f}%深于崩塌线{skip_gap_deep}%"))
                continue
            # 第一百轮(实验): 评分门按池深缩放 —— 评分是"池内 z 分", 而 _zs 的
            # 理论最大值为 (n-1)/sqrt(n)(n<≈10 时受此约束, 之后被 clip(±3) 截断),
            # 因此同一个"1.0 门槛"在 5 只的池里要 56% 的理论最大、在 40 只的池里只要 33%,
            # 池浅日被系统性卡死(池深与评分相关 0.53)。缩放后门槛与池深解耦,
            # 浅池日由"池深分档"降仓而不是一票否决。
            _gate = min_top1_score
            if _gate is not None and gate_scale:
                _n = int(p.get("_pool_n") or 0)
                if _n >= 2:
                    _zmax = min(3.0, (_n - 1) / math.sqrt(_n))
                    _ref = min(3.0, (gate_ref - 1) / math.sqrt(gate_ref))
                    if _ref > 0:
                        _gate = _gate * min(1.0, _zmax / _ref)
            # 第一百一十四轮(可选, 默认关闭): 深低开日放宽评分门 ——
            # 缺口是"错杀"的直接证据(第一百零一轮 92 笔 gap<=-1% 均值+1.85%/胜率75%),
            # 而评分只是池内相对强弱。当直接证据足够强时, 不必再用相对评分一票否决,
            # 允许这类票以更低评分门槛参与(仓位仍由 gap_floor_tiers / 池深分档决定)。
            _relax_gap = ex.get("gate_relax_gap_below")
            _relax_to = ex.get("gate_relax_score")
            if (_gate is not None and _relax_gap is not None and _relax_to is not None
                    and gap * 100 <= float(_relax_gap)):
                _gate = min(_gate, float(_relax_to))
            if _gate is not None and p.get("score", 0) < _gate:
                trades.append(_trade(p, "skip", d, note=f"top1评分{p.get('score',0):.2f}低于门槛{_gate:.2f}"))
                continue
            qualified.append({**p, "_row": row, "_gap": gap, "_lim_px": _lim_px,
                              "_pct_sig": _pct_sig})
        if not qualified:
            return
        # 每笔固定占组合总资产 frac%(与达标数量解耦, 防恐慌晨逆向选择),
        # 按评分排序最多买 max_buy_per_day 只; 风控触发日按 blocked_frac 降仓参与(0=空仓)
        max_n = int(ex.get("max_buy_per_day", 0))
        if max_n > 0:
            qualified = qualified[:max_n]
        frac = ex.get("alloc_frac_pct", 5.0) / 100.0
        if not day_ok:
            if day_frac:
                frac *= day_frac.get(d, 0.0)  # 连续空仓超限后的降仓参与系数
            elif reg_row is not None and bool(reg_row.get("soft_ok", False)):
                # 分档参与(第一百一十三轮): 指数<MA20 但宽度尚可的非恐慌日 → 按 soft_trend_frac 降档
                frac *= float(_rm.get("soft_trend_frac", 0.0) or 0.0)
            else:
                frac *= ex.get("blocked_frac", 0.0)  # 大盘风控期参与系数: 0=空仓(默认), 0.5=半仓
        else:
            # 风控解除后的渐进建仓(第一百一十五轮, 可选): 只削"刚恢复"那几天的仓位上限
            frac *= reentry_map.get(d, 1.0)
        if frac <= 0:
            return   # 第五十二轮修复: 信号日可买但买入日突发风控/恐慌(frac==0)时,
                     # 原实现alloc=0且"0<0"不拦截, 产生units=0的幽灵成交(ret_pct=0.0,
                     # 污染交易次数/胜率/连亏计数; 资金路径本就无影响, 此处不产生任何记录)
        # 第一百一十六轮修复: equity 必须含 pending_cash。
        # 原实现漏掉它 —— 今日开盘按收盘价结算卖出的持仓已从 holding 移除, 其回笼款
        # 只进了 pending_cash, 于是 equity 被系统性低估成"仅剩的可动用现金"。sf=0 时
        # equity=0 被 cash<=0 提前拦住, 无影响; 但只要 sf 微小正值(如 0.005), equity
        # 就缩成那 0.5%, 买入额按 0.5% 计 —— 造出一批"名义满仓、实际 0.5% 仓位"的
        # 注水成交(1500 日 379→455 笔全部来自此处), 使 sf 细扫的笔数曲线失真。
        # 总资产 = 可用现金 + 今日收盘腿未到账款 + 持仓市值, 与收盘估值口径一致。
        equity = cash + pending_cash + sum(
            h["units"] * float(per_code[h["code"]].loc[d]["open_adj"])
            for h in holding if d in per_code[h["code"]].index
        )
        # 月度盈利/亏损熔断: 当月已实现收益超阈值后降仓(仅用已实现信息, 无未来函数)
        if month_start_nav:
            mret = (equity / month_start_nav - 1) * 100
            if mc_cap > 0 and mc_cut < 1.0 and mret > mc_cap:
                frac *= mc_cut
            if ml_cap > 0 and ml_cut < 1.0 and mret < -ml_cap:
                frac *= ml_cut
        # 绩效门控仓位(第九十七轮, 可选): 近 window 笔"已结算"成交的均值低于 cut_below 时,
        # 说明当前市场结构对本策略不利(如 2025 全年胜率 45%/单笔 +0.15%), 主动降仓等待;
        # 均值回到 cut_below 之上自动恢复。纯已实现信息, 无未来函数(与 depth_gate 同源)。
        _pg = ex.get("perf_gate") or {}
        if _pg:
            _w = int(_pg.get("window", 20))
            _real = [t["ret_pct"] for t in trades if t.get("ret_pct") is not None]
            if len(_real) >= _w:
                _m = sum(_real[-_w:]) / _w
                if _m < float(_pg.get("cut_below", 0.0)):
                    frac *= float(_pg.get("cut_mult", 0.5))
        for p in qualified:
            row = p.pop("_row")
            pick_frac = frac
            # 评分分档下注(可选): 评分越高仓位越重(降序匹配, 未命中则用最低档)
            if score_tiers:
                _sc = float(p.get("score") or 0.0)
                for _t in score_tiers:
                    if _sc >= float(_t["min"]):
                        pick_frac *= float(_t.get("mult", 1.0))
                        break
            # 缺口分级下注(可选): 竞价缺口越深=错杀越极端/edge越强, 仓位越高; 缺省关闭
            if gap_tiers:
                g100 = p["_gap"] * 100
                for t in gap_tiers:
                    if g100 <= t["below"]:
                        pick_frac *= t["mult"]
                        break
            # 高开降仓(第九十七轮, 可选): 缺口 >= above 时按 mult 缩仓(不拒单)。
            # 逻辑: 本策略的 edge 来自"深低开后的恐慌修复"; 缺口>=0 意味着开盘就已修复,
            # 承接逻辑不成立 —— 实测 260 笔(48%)均值 +0.03%/胜率 44%, 等于零期望占半仓。
            # 与 buy_gap_below 的"挑单"不同: 这里每天照常按开盘价成交, 只是高开不下重注,
            # 因此不减少交易次数(用户约束)。降序匹配第一条满足的档。
            if gap_above_tiers:
                g100 = p["_gap"] * 100
                _sc = float(p.get("score") or 0.0)
                for t in gap_above_tiers:
                    # 可选 score_below: 该档只对"评分不足"的票生效 —— 高开+高分(强势股抢筹)
                    # 与高开+低分(单纯追高)是两种东西, 前者不必减仓。缺省=不限评分。
                    if "score_below" in t and _sc >= float(t["score_below"]):
                        continue
                    if g100 >= float(t["above"]):
                        pick_frac *= float(t.get("mult", 1.0))
                        break
            # 池深分档下注(第九十九轮, 可选): 当日候选池深度 = 同风格票的"共振强度",
            # T 日盘后即已知(无未来函数)。池子越宽说明越多强势股同步回调、次日修复有板块共振,
            # 池子窄则是孤立回调 —— 实测几何期望从池41+的+1.28%单调掉到池11-20的+0.04%。
            if pool_tiers:
                _pn = int(p.get("_pool_n") or 0)
                for _t in pool_tiers:
                    if _pn >= int(_t["min"]):
                        pick_frac *= float(_t.get("mult", 1.0))
                        break
            # 阴跌刀口保护(第一百零三轮, 可选): 市场 T 日"典型个股"在涨/走平(中位数 > mkt_above),
            # 而本票超额下跌超过 res_below 个百分点 —— 这是"个股自己的事"(利空/资金出逃),
            # 不是大盘一起砸出来的系统性错杀, 次日深低开的修复含义不同, 故缩仓(不弃单)。
            # 放在 gap_floor 之前: 深低开是直接证据, 仍可把仓位补回满仓。
            if knife_tiers and p.get("_pct_sig") is not None:
                _mr = mkt_ret_map.get(p["signal_date"])
                if _mr is not None:
                    _res = float(p["_pct_sig"]) - _mr
                    for t in knife_tiers:
                        if _mr > float(t.get("mkt_above", -0.5)) and _res <= float(t.get("res_below", -4.0)):
                            pick_frac *= float(t.get("mult", 1.0))
                            break
            # 全市场恐慌日回满仓(第一百零三轮, 可选): 信号日市场"典型个股"跌幅达到 mkt_below 时,
            # 这一天的回调是大盘一起砸出来的系统性错杀, 次日深低开的修复有全市场共振,
            # 撤销评分/池深折价回到 floor(仍以当日参与系数 frac 为上限基准, 不越过风控降仓)。
            if mkt_floor_tiers and p.get("_pct_sig") is not None:
                _mr = mkt_ret_map.get(p["signal_date"])
                if _mr is not None:
                    for t in mkt_floor_tiers:
                        if _mr <= float(t.get("mkt_below", -1.0)):
                            pick_frac = max(pick_frac, frac * float(t.get("floor", 1.0)))
                            break
            # 深低开加仓(第一百零一轮, 可选): 竞价缺口 <= below 且当日池深 >= min_pool 时,
            # 仓位不低于 frac*floor(以当日参与系数 frac 为基准, 不覆盖风控降仓)。
            # 依据: 缺口是唯一在 6/7 年同向的调节变量 —— 缺口<=-1% 的 92 笔均值 +1.85%/胜率 75%,
            # 而缺口>=0.5% 的 120 笔均值 -0.15%; 现行规则只惩罚高开, 深低开拿不到额外预算。
            if gap_floor_tiers:
                _g100 = p["_gap"] * 100
                _pn = int(p.get("_pool_n") or 0)
                for _t in gap_floor_tiers:
                    if _g100 <= float(_t["below"]) and _pn >= int(_t.get("min_pool", 0)):
                        pick_frac = max(pick_frac, frac * float(_t.get("floor", 1.0)))
                        break
            # 恐慌日超配(第一百七十三轮, 可选; 缺省空=逐位等价)。
            # mkt_floor_tiers 只能把仓位"补回"当日 frac 基准(风控日 frac=0.4 -> 最多 0.4),
            # 语义是"撤销折价"; 本项语义是"在最终仓位之上再乘一个 >1 的倍数", 允许突破满仓。
            # 依据: 信号日市场典型个股跌幅越深, 次日错杀修复的期望越高(200 笔样本:
            # mkt<=-0.5% 的 78 笔 +2.06% vs 其余 +1.21%); 这是加仓不是降仓, 与第 119 轮
            # "自我降仓必崩"不冲突。
            # 可选 mkt_above: 给出区间下限 —— 只在 [mkt_above, mkt_below] 之间加仓。
            # 必要性: 200 笔样本里 mkt<=-2.5% 的 6 笔均值 -1.61%(接飞刀), 而 -0.5~-1.5
            # 的 60 笔均值 +1.9%。不加下限等于连崩塌日一起放大, 是把"错杀修复"和"趋势崩塌"
            # 混为一谈。缺省 None = 不设下限(仅 mkt_below 生效)。
            # 第一百七十四轮: 同一分支支持分位档(rank_below / rank_above, 0~1)。
            # 分位模式不要求个股 pct 存在(它只用市场横截面), 但绝对阈值模式仍保留
            # 原 `_pct_sig is not None` 的语义, 以保证不配 mkt_rank_win 时逐位等价。
            if mkt_boost_tiers:
                _mr = mkt_ret_map.get(p["signal_date"])
                for _t in mkt_boost_tiers:
                    if "rank_below" in _t or "rank_above" in _t:
                        _rk = mkt_rank_map.get(p["signal_date"])
                        if _rk is None:
                            continue
                        _rb = _t.get("rank_below")
                        if _rb is not None and _rk > float(_rb):
                            continue
                        _ra = _t.get("rank_above")
                        if _ra is not None and _rk < float(_ra):
                            continue
                        pick_frac *= float(_t.get("mult", 1.0))
                        break
                    if p.get("_pct_sig") is None or _mr is None:
                        continue
                    if _mr > float(_t.get("mkt_below", -1.5)):
                        continue
                    _ma = _t.get("mkt_above")
                    if _ma is not None and _mr < float(_ma):
                        continue
                    pick_frac *= float(_t.get("mult", 1.0))
                    break
            # 融资可用额度按"本笔目标仓位"重算: pick_frac>1 时杠杆额度须同步放大,
            # 否则 min(pick_frac*equity*lev, cash+margin_avail) 会被 (lev-1)*equity 卡死,
            # 超配永远不生效。pick_frac<=1 时 _lev_eff==lev, 与旧口径逐位一致。
            _lev_eff = lev * max(1.0, pick_frac)
            margin_avail = max(0.0, (_lev_eff - 1.0) * equity - max(0.0, -cash)) if _lev_eff > 1.0 else 0.0
            if cash + margin_avail <= 0:
                break
            alloc = min(pick_frac * equity * lev, cash + margin_avail)
            # 第一百零三轮修复: 分档乘数为 0(或叠乘后为 0)时会生成 units=0 的"零仓成交" ——
            # 既记进交易次数/胜率(0 收益被算作亏损), 又与实际资金路径矛盾。
            # 与 day 级 frac<=0 的守卫同源, 只是那一层拦不住单笔被乘成 0。
            if pick_frac <= 0:
                continue
            # 剩余现金不足"本笔目标仓位"的 30% 才放弃(第九十七轮修正: 原以满仓 frac 为基准,
            # 在评分档×缺口档叠加把目标仓位压到 30% 以下时会把本可成交的单直接丢弃 ——
            # 高开降仓的实验里因此凭空少掉 159 笔交易; 判据应与本笔实际目标仓位同尺度)。
            # 第一百一十四轮(可选): 阈值改为可配置 min_fill_frac(缺省 0.3, 与旧口径逐位等价)。
            # 该阈值直接决定"资金未回笼日要不要用小仓参与", 是成交笔数的开关之一。
            _min_fill = float(ex.get("min_fill_frac", 0.3) if ex.get("min_fill_frac") is not None else 0.3)
            if alloc < pick_frac * equity * _min_fill:
                if os.environ.get("BT_DEBUG_CASH"):
                    cash_blocked.append({
                        "d": d, "code": p["code"], "name": p.get("name"),
                        "rank": p.get("_rank"), "pool": p.get("pool", "main"),
                        "score": p.get("score"), "gap": round(gap * 100, 2),
                        "target": round(pick_frac, 3),
                        "cash_frac": round(cash / equity, 4) if equity else 0.0,
                        "hit": round(alloc / (pick_frac * equity), 3) if pick_frac * equity else 0.0,
                    })
                break
            _lp = p.get("_lim_px")
            # 分批建仓(第九十九轮, 可选; 缺省 1.0=全开盘买, 与旧口径逐位等价):
            # T+1 开盘买 frac_b, 当日收盘买 1-frac_b。与卖出端 sell_open_frac 对称 ——
            # 都在回答"日内这一段该不该全押某一个时点"。成本价按成交金额加权。
            if _lp:
                buy_adj = float(_lp)
            elif frac_buy_open >= 1.0:
                buy_adj = float(row["open_adj"])
            else:
                buy_adj = (frac_buy_open * float(row["open_adj"])
                           + (1 - frac_buy_open) * float(row["close_adj"]))
            buy_eff = buy_adj * (1 + slip)
            units = alloc / (buy_eff * (1 + comm))
            fee = alloc * comm
            fee_total += fee
            cash -= alloc
            _f = (float(row["close_adj"]) / float(row["close"]) if float(row["close"]) else 0.0)
            holding.append({
                **p, "buy_date": d, "units": units, "buy_eff": buy_eff, "buy_adj": buy_adj,
                "buy_px_raw": round(float(_lp) / _f, 4) if (_lp and _f) else float(row["open"]),
                "fill": "limit" if _lp else "open", "buy_idx": idx,
                "pos_frac": round(pick_frac, 4),   # 第一百零四轮: 记下这笔的下注比例, 供"仓位与右尾是否匹配"审计
            })
        # 未成交部分留在现金中, 复利滚动
        # 注: A股T+1制度, 当日买入的股票当日不可卖出, 因此不存在任何
        #     "买入当日止损/止盈"逻辑; 所有持仓最早于次一交易日开盘卖出。

    def do_sell(d: str, idx: int = 0, has_buy: bool = False):
        hold_extra = int(ex.get("hold_extra_days", 0) or 0)  # T+2开盘后再多持有N个交易日(0=原口径)
        frac_open = min(1.0, max(0.0, float(ex.get("sell_open_frac", 1.0) or 0.0)))  # 开盘卖出占比
        # 第一百零二轮(可选): 今日有买入计划时, 开盘一次性清仓(而非50/50) —— 把资金全部
        # 腾给新仓位。理由: 分批出场让一半资金锁到收盘, 下一笔买入只能拿到 ~50% 的目标仓位,
        # 仓位大小由"昨天卖没卖"决定而非由信号强度决定, 是现金流的路径噪声。
        # 只在"确实有下一笔要买"的日子清仓, 没有信号的日子仍按 50/50 吃日内那一段。
        if has_buy and ex.get("sell_full_on_rebuy"):
            frac_open = 1.0
        nonlocal cash, fee_total, pending_cash, loss_streak
        remaining = []
        for h in holding:
            cur_idx = idx   # 停牌顺延分支会把 idx 覆盖成子表索引, 后续判断统一用 cur_idx
            due_i = int(h.get("due_idx", h.get("buy_idx", 0) + 1 + hold_extra))
            if cur_idx < due_i:
                remaining.append(h)   # 未到卖出时点(持有期延长/跨日分批的尾腿)
                continue
            sub = per_code.get(h["code"])
            row = sub.loc[d] if sub is not None and d in sub.index else None
            sell_px_adj, sell_d, note = None, d, ""
            if row is not None and row["is_paused"] == 1:
                row = None
            if row is not None and row["vol"] <= 0:
                row = None
            if row is None:
                # 停牌顺延: 最多向后找5个交易日
                idx = sub.index
                if sub is not None and d in idx:
                    pos = idx.get_loc(d)
                    for j in range(1, 6):
                        if pos + j >= len(idx):
                            break
                        r2 = sub.iloc[pos + j]
                        if r2["is_paused"] == 0 and r2["vol"] > 0:
                            row = r2
                            sell_d, note = str(idx[pos + j]), "停牌顺延"
                            break
                if row is None:  # 退市等极端情况: 按最后已知收盘估算
                    if sub is not None and len(sub):
                        row = sub.iloc[-1]
                        sell_d, note = str(sub.index[-1]), "退市按末收盘"
                    else:
                        remaining.append(h)
                        continue
            open_adj, close_adj = float(row["open_adj"]), float(row["close_adj"])
            # 卖出时点(事前既定规则, 两种均合规): true=T+2开盘竞价卖(默认);
            # false=T+2收盘卖(持有整个T+2交易日, 按收盘价成交)
            sell_px_adj = open_adj if ex.get("sell_at_open", True) else close_adj
            sold_at_close = sell_px_adj is close_adj
            if ex.get("sell_at_open", True) and ex.get("sell_limit_down_fallback", True) \
                    and float(row["open"]) <= float(row["low_limit"]) + 1e-9:
                sell_px_adj = close_adj
                sold_at_close = True
                note = (note + " 开盘跌停按收盘").strip()
            # 可选: T+2开盘深跌(未跌停)延迟至收盘卖(避免恐慌开盘割肉; 用当日已实现的open/prev_close)
            scd = ex.get("sell_close_if_open_drop")
            if (not sold_at_close) and scd and sell_d == d and float(row["pre_close"]) > 0 \
                    and (float(row["open"]) / float(row["pre_close"]) - 1) * 100 <= -scd:
                sell_px_adj = close_adj
                sold_at_close = True
                note = (note + f" 开盘深跌{sced_lbl(scd)}按收盘").strip()
            # 可选(第九十八轮): T+2 开盘"相对我的买入成本"浮亏 -> 延卖至当日收盘。
            # 与 sell_close_if_open_drop 的本质区别 —— 那个看"相对昨收的跳空"(d2),
            # 这个看"相对我的成本"(r2)。同一个 -2% 跳空, 在 T+1 大涨后是浮盈 +1%,
            # 在 T+1 崩盘后是浮亏 -7%, 两者的日内修复概率完全不同; 用 r2 才是
            # "我现在是赚是亏"的正确刻度。实测(545 笔, 仅盘中信息, 无未来函数):
            #   开盘卖 单笔 +0.712%/胜率 56.5%; 跳空延卖(现行) +0.741%/57.1%;
            #   浮亏延卖 +0.920%/67.0%; 且 r_open 落在 -3~-2% 的 49 笔日内平均修复 +1.11%。
            spb = ex.get("sell_close_if_profit_below")
            if (not sold_at_close) and spb is not None \
                    and open_adj / h["buy_adj"] - 1 < float(spb) / 100:
                sell_px_adj = close_adj
                sold_at_close = True
                note = (note + " 开盘浮亏按收盘").strip()
            # 条件延迟卖出(可选): T+2开盘相对买入成交价涨幅超阈值(强势高开)时,
            # 延迟至当日收盘卖出让利润奔跑; 卖出所得收盘后回笼, 次日开盘方可投入
            if (not sold_at_close) and ex.get("hold_if_open_gain", 0) > 0 \
                    and open_adj / h["buy_eff"] - 1 > ex.get("hold_if_open_gain") / 100:
                sell_px_adj = close_adj
                sold_at_close = True
                note = (note + " 高开强延卖").strip()
            # 分批出场(第九十八轮, 可选; 缺省 frac_open=1.0 时与旧口径逐位等价):
            # 一部分按开盘价卖(资金当日回笼可再投)、其余按收盘价卖(资金次日回笼)。
            # 动机: "日内是正还是负"随市场结构翻转(2021~24 日内正向、2025~26 负向),
            # 押单一时点等于押结构不变; 两条腿各走一半, 两种结构下都只吃一半亏。
            # 第一百零四轮(可选): 高浮盈开盘一次性清仓 —— 与 scd(深跌延卖) 严格对称的另一半。
            # 逐笔证据(575笔): T+2 开盘相对买入价浮盈 >=5% 的 49 笔(8.5%), 当日日内 -0.83%/胜率35%,
            #   再拿一天(开盘对开盘) -1.00%/胜率35%; 而浮盈 <5% 的 526 笔日内 +0.37%/53%。
            #   这 49 笔贡献了全部复利(等权 +3154%, 其余 526 笔合计 -53%) —— 右尾已经把利润
            #   全部前置到开盘这一刻, 再等收盘只是把它交回去。
            hfrac = frac_open
            sfg = ex.get("sell_full_if_open_gain")
            if (not sold_at_close) and sfg and open_adj / h["buy_eff"] - 1 >= float(sfg) / 100:
                hfrac = 1.0
                note = (note + f" 开盘浮盈{sced_lbl(float(sfg))}全卖").strip()
            u_o = 0.0 if sold_at_close else h["units"] * hfrac
            u_c = h["units"] - u_o
            # 跨日分批出场(第一百一十五轮新增, 可选; defer_frac=0 即关闭, 与旧口径逐位等价):
            # 正常卖出日只卖 (1-defer_frac) 的仓位, 余下部分延后 defer_days 个交易日再按同样规则卖。
            # 动机: 收盘腿从 40% 加到 90%(第一百一十一轮)证明"T+2 日内仍向上"; 但那是**日内的斜率**,
            #   不等于"隔夜仍有溢价"。尾仓多拿一天是独立的赌注 —— T+3 隔夜涨则赚、跌则吐回,
            #   必须实测, 不能从 sf 的单调性外推。
            # 资金含义: 延迟部分 T+3 收盘后才回笼(实盘=T+4 开盘可用), 会挤占下一笔的仓位, 是真实成本。
            _df = float(ex.get("defer_frac", 0.0) or 0.0)
            _defer_units = 0.0
            if _df > 0 and not h.get("_deferred") and cur_idx == due_i:
                _defer_units = h["units"] * min(1.0, max(0.0, _df))
                u_o *= (1 - _df)
                u_c *= (1 - _df)
            gross_o = u_o * open_adj * (1 - slip)
            gross_c = u_c * close_adj * (1 - slip)
            _st = _stamp_at(row)
            net_o = gross_o * (1 - _st) * (1 - comm)
            net_c = gross_c * (1 - _st) * (1 - comm)
            proceeds = net_o + net_c
            fee = gross_o * (_st + comm) + gross_c * (_st + comm)
            fee_total += fee
            # 开盘卖出: 资金当日开盘即回笼可再投入; 收盘卖出(整日/跌停回退/延卖/分批尾腿):
            # 资金收盘后回笼, 次一交易日开盘才可用于买入(防止虚假资金复用)
            if u_o:
                cash += net_o
            if u_c:
                pending_cash += net_c
            if 0 < hfrac < 1 and not sold_at_close:
                note = (note + f" 分批{hfrac:.0%}开盘/{1 - hfrac:.0%}收盘").strip()
            if _defer_units > 0:
                note = (note + f" 跨日分批卖{1 - _df:.0%}余{_df:.0%}延后").strip()
            # 收益率分母只计"本次真正卖出的份额"(跨日分批时尾腿未卖, 不能算进本笔)
            denom = (h["units"] - _defer_units) * h["buy_eff"] * (1 + comm)
            r = proceeds / denom - 1 if denom and denom > 0 else 0.0
            # 第一百二十三轮修正: sell_px 曾经是"开盘腿的价格"(sold_at_close 为假时取 row["open"]),
            # 而 ret_pct 是"开盘腿+收盘腿"加权后的收益 —— 分批出场时两列不同源, 页面会出现
            # "卖价低于买价却是正收益"(开盘腿亏、收盘腿赚)。这里改成**成交加权均价**,
            # 并把两条腿的价格一并落盘, 让收益可由两列价格复现。
            _px_o, _px_c = float(row["open"]), float(row["close"])
            _u_tot = u_o + u_c
            _px_avg = ((u_o * _px_o + u_c * _px_c) / _u_tot) if _u_tot > 0 else _px_c
            trades.append({
                "signal_date": h["signal_date"], "code": h["code"], "name": h.get("name", ""),
                "rank": h.get("_rank"), "pool": h.get("pool", "main"),
                "strategy": h.get("strategy", "dip"),
                "buy_date": h["buy_date"], "buy_px": h["buy_px_raw"],
                "fill": h.get("fill", "open"),
                "sell_date": sell_d, "sell_px": round(_px_avg, 4),
                "sell_px_open": _px_o, "sell_px_close": _px_c,
                "sell_frac_open": (round(u_o / _u_tot, 4) if _u_tot > 0 else 0.0),
                "ret_pct": round(r * 100, 2), "note": note,
                # 第九十七轮: 成交明细记录"下注依据"(评分/竞价缺口/当日候选池深度),
                # 便于事后审计"为什么这笔下多重" —— 此前 trades 只有结果没有依据。
                "score": h.get("score"),
                "gap_pct": (round(float(h["_gap"]) * 100, 2) if h.get("_gap") is not None else None),
                "pool_n": h.get("_pool_n"),
                "pos_frac": h.get("pos_frac"),
            })
            if _defer_units > 0:
                # 尾腿: 保留原成本价与前收, 仅把应卖日推到 defer_days 之后(仍按同样规则卖出)
                remaining.append({**h, "units": _defer_units,
                                  "due_idx": cur_idx + max(1, int(ex.get("defer_days", 1) or 1)),
                                  "_deferred": True, "buy_idx": cur_idx})
            if r <= 0:
                loss_streak += 1
            else:
                loss_streak = 0
        holding[:] = remaining

    print("[3/4] 逐日模拟交易...")
    t0 = time.time()
    for idx, d in enumerate(main_dates):
        ym = d[:6]
        if ym != month_key:
            month_key = ym
            # 月初基准 = 上一交易日收盘估值(首日用开盘前现金+持仓估值近似, 此处用前一curve值)
            prev = curve.get(main_dates[idx - 1]) if idx > 0 else None
            month_start_nav = prev if prev else float(bt_cfg["initial_capital"])
        # 次日开盘: 前一交易日收盘卖出回笼的资金此时方可投入(T+2收盘卖模式)
        cash += pending_cash
        pending_cash = 0.0
        # 闲置现金计息(第一百七十三轮, 可选; cash_yield=0 时逐位等价):
        # 只对正余额计提 —— 透支部分已由下面的融资利息单独扣, 两者不可叠加。
        if cash_yield > 0 and cash > 0:
            _y = cash * cash_yield / 252.0
            cash += _y
            yield_total += _y
        # 融资利息: 对透支额按年化逐日计提(自然日近似为交易日 252)
        if cash < 0 and lev_rate > 0:
            _int = -cash * lev_rate / 252.0
            cash -= _int
            interest_total += _int
            fee_total += _int
        # 开盘: 先卖出昨日到期持仓(T-2信号的池), 再买入(T-1信号的池)
        do_sell(d, idx, has_buy=bool(pending_buy is not None and pending_buy[1]))
        if pending_buy is not None:
            _, picks = pending_buy
            do_buy(picks, d, idx)
            pending_buy = None
        # 收盘估值
        total = cash + pending_cash + sum(h["units"] * float(per_code[h["code"]].loc[d]["close_adj"])
                                          for h in holding if d in per_code[h["code"]].index)
        curve[d] = total
        if not holding:
            flat_days += 1   # 收盘账户无持仓(当日买入失败/全部到期卖出且无新票)
        # 收盘后: 决定明日计划
        reg_row = reg.loc[d] if d in reg.index else None
        # 分档参与(第一百一十三轮): 软参与日也照常出计划(买入端按 soft_trend_frac 打折)
        soft_today = (reg_row is not None and bool(reg_row.get("soft_ok", False))
                      and float(_rm.get("soft_trend_frac", 0.0) or 0.0) > 0)
        if day_frac:
            eff_today = day_frac.get(d, 1.0)
        else:
            eff_today = 0.0 if (reg_row is not None and not bool(reg_row["trade_ok"])) \
                and not ex.get("blocked_frac", 0.0) and not soft_today else 1.0
        blocked = reg_row is not None and not bool(reg_row["trade_ok"]) and eff_today <= 0
        # 连亏熔断: 连续亏损达阈值后, 从当日起再暂停 pause_days 个交易日(买入端)
        paused = False
        if ps_streak > 0 and loss_streak >= ps_streak:
            if idx > pause_until:
                pause_until = idx + ps_days
            paused = idx <= pause_until
        if blocked or paused:
            risk_empty_days += 1
            pending_buy = (d, [])
            # 注: 大盘风控触发仅限制次一交易日不买入; 持仓(含当日买入票)
            #     均按既定 T+2(或之后)开盘卖出, 遵守T+1制度, 无任何当日卖出。
        else:
            # 第一百一十七轮 PIT 修正: 回测不传"当前名称"。
            # stock_names() 返回的是**今天**的名称, 用它过滤历史行等于提前知道"这票后来会被 ST/退市"
            # (实测: 当前 198 只 ST 中平均 148 只在 1500 日窗口的历史某日还是非 ST, 约占当时全市场 3.3%)。
            # 当日的 ST 状态由 is_st 字段给足(逐日快照真实值), 回测只用它; 实盘 daily_run 仍传 names,
            # 因为实盘当天用当天名称不构成前视。
            picks, _sel_stats = select_stocks(d, by_date[d], cfg, names, name_filter=False)
            # 第九十七轮: 当日候选池深度(通过 dip 筛选并完成评分的只数)随计划下发,
            # 用于诊断"机会丰度"; 纯观测字段, 不改变任何买入判定。
            _pn = int((_sel_stats or {}).get("scored") or 0)
            pending_buy = (d, [{**p, "signal_date": d, "_pool_n": _pn} for p in picks])
        # 若持仓未在开盘卖出(停牌顺延中), 继续每日估值
    print(f"  模拟完成, 耗时 {time.time()-t0:.0f}s")

    print("[4/4] 汇总绩效...")
    nav = pd.Series(curve).sort_index()
    if nav.empty:
        raise RuntimeError("回测主窗口为空, 无净值数据(请检查交易日/预热配置)")
    ret = nav.pct_change().dropna()
    n = len(nav)
    total_ret = nav.iloc[-1] / nav.iloc[0] - 1
    cagr = (nav.iloc[-1] / nav.iloc[0]) ** (252 / max(n, 1)) - 1
    dd = nav / nav.cummax() - 1
    mdd = float(dd.min())
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() else 0.0
    real = [t for t in trades if t["ret_pct"] is not None]
    wins = [t for t in real if t["ret_pct"] > 0]
    losses = [t for t in real if t["ret_pct"] <= 0]
    gross_win = sum(t["ret_pct"] for t in wins)
    gross_loss = -sum(t["ret_pct"] for t in losses)
    bench = reg.loc[reg.index.isin(main_dates), "mkt_idx"]
    bench_norm = (bench / bench.iloc[0] * float(bt_cfg["initial_capital"])) if len(bench) else bench
    # 期末未平仓持仓披露: 买入记录延迟至卖出日(do_sell)才写入trades, 期末未平仓的
    # 持仓在trades明细中天然缺席——显式列出, 防止"明细全skip"被误读为"未买入"
    open_positions = []
    for h in holding:
        sub = per_code.get(h["code"])
        last = sub.iloc[-1] if sub is not None and len(sub) else None
        unreal = (float(last["close_adj"]) / h["buy_eff"] - 1) * 100 if last is not None else None
        open_positions.append({
            "signal_date": h.get("signal_date", ""), "code": h["code"], "name": h.get("name", ""),
            "buy_date": h.get("buy_date", ""), "buy_px": h.get("buy_px_raw"),
            "cost_pct": round(h["units"] * h["buy_eff"] / float(nav.iloc[-1]) * 100, 1),
            "last_px": float(last["close"]) if last is not None else None,
            "unrealized_pct": round(unreal, 2) if unreal is not None else None,
        })

    result = {
        "meta": {
            "run_time": datetime.now(TZ_CN).strftime("%Y-%m-%d %H:%M:%S"),
            "start": main_dates[0], "end": main_dates[-1], "days": n,
            "initial_capital": bt_cfg["initial_capital"],
            # 第一百二十五轮: 口径与成本均按 config 生成, 不再写死"T+2开盘卖出"。
            "rules": rules_line(
                ex, f"复利/佣金万{comm * 10000:g}+印花税万{stamp * 10000:g}+滑点千{slip * 1000:g}"),
        },
        "metrics": {
            "总收益率%": round(total_ret * 100, 2),
            "年化收益率%": round(cagr * 100, 2),
            "最大回撤%": round(mdd * 100, 2),
            "卡玛比率": round(cagr / abs(mdd), 2) if mdd else None,
            "夏普比率": round(sharpe, 2),
            "交易次数": len(real),
            "胜率%": round(len(wins) / len(real) * 100, 2) if real else 0,
            "平均单笔收益%": round(np.mean([t["ret_pct"] for t in real]), 3) if real else 0,
            "盈亏比": round(gross_win / gross_loss, 2) if gross_loss else None,
            "空仓天数": flat_days,
            "风控空仓天数": risk_empty_days,
            "跳过买入次数": sum(1 for t in trades if t["ret_pct"] is None),
            "总费用(元)": round(fee_total, 0),
            "同期市场等权基准%": round(float(bench_norm.iloc[-1] / bt_cfg["initial_capital"] - 1) * 100, 2) if len(bench_norm) else None,
        },
        "equity": [{"d": _fmt_date(d), "nav": round(float(v), 0)} for d, v in nav.items()],
        "benchmark": [{"d": _fmt_date(d), "nav": round(float(v), 0)} for d, v in bench_norm.items()] if len(bench_norm) else [],
        "trades_recent": trades[-60:],
        # 已执行成交单列(2026-10-24第三十九轮): skip密度高(全窗3777条), 混排尾部窗口
        # 装不下成交记录, 失效监测回溯(tools/monitor_retro)需要独立的成交序列
        "trades_executed_recent": [t for t in trades if t.get("ret_pct") is not None][-200:],
        **({"trades_all": trades} if __import__("os").environ.get("BT_FULL_TRADES") else {}),
        **({"cash_blocked": cash_blocked} if os.environ.get("BT_DEBUG_CASH") else {}),
        **({"open_positions": open_positions} if open_positions else {}),
        "monthly": _monthly(nav),
    }
    return result


def sced_lbl(x):
    return f"{x:g}%"


def _trade(p: dict, kind: str, d: str, note: str = "") -> dict:
    return {"signal_date": p.get("signal_date", ""), "code": p["code"], "name": p.get("name", ""),
            "rank": p.get("_rank"), "pool": p.get("pool", "main"),
            "buy_date": d, "buy_px": None, "sell_date": None, "sell_px": None,
            "ret_pct": None, "note": note}


def _fmt_date(d: str) -> str:
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


def _monthly(nav: pd.Series) -> list[dict]:
    m = nav.copy()
    m.index = [str(i)[:6] for i in m.index]
    g = m.groupby(level=0).agg(["first", "last"])
    out = []
    prev_last = None
    for idx, row in g.iterrows():
        base = prev_last if prev_last else row["first"]
        out.append({"month": idx, "ret_pct": round((row["last"] / base - 1) * 100, 2)})
        prev_last = row["last"]
    return out


def save_result(result: dict, out: str | None = None):
    out_path = Path(out) if out else ROOT / "output" / "backtest.json"
    out_path.parent.mkdir(parents=True, exist_ok=True)
    # default=float: float32面板(长窗口)下指标为np.float32, 兜底转float
    out_path.write_text(json.dumps(result, ensure_ascii=False, indent=1, default=float),
                        encoding="utf-8")
    return out_path


def main():
    ap = argparse.ArgumentParser(description="A股次日候选池策略回测")
    ap.add_argument("--days", type=int, default=None, help="回测交易日数")
    ap.add_argument("--config", type=str, default=None)
    args = ap.parse_args()
    cfg = load_config(args.config)
    result = run_backtest(cfg, args.days)
    path = save_result(result)
    m = result["metrics"]
    print("\n========== 回测结果 ==========")
    print(f"区间: {result['meta']['start']} ~ {result['meta']['end']} ({result['meta']['days']}个交易日)")
    for k, v in m.items():
        print(f"  {k}: {v}")
    print(f"报告已写入: {path}")


if __name__ == "__main__":
    main()
