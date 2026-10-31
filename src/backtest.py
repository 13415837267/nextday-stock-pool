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
  - T+2 开盘竞价卖出
    * T+2 停牌 -> 顺延至后续交易日开盘卖出(最多5日, 否则按最后收盘价)
    * T+2 开盘跌停且开启保守模式 -> 按当日收盘价卖出
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
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from .config import ROOT, load_config
from .data_api import ZZClient
from .selector import apply_regime, build_indicators, regime_base, regime_series, select_stocks, split_by_date


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
    per_code, names = ctx["per_code"], ctx["names"]
    bt_cfg = cfg["backtest"]

    ex = cfg["execution"]
    comm, stamp, slip = ex["commission_rate"], ex["stamp_tax"], ex["slippage"]
    gap_tiers = ex.get("gap_size_tiers") or []   # 缺口分级下注: [{below: -4.0, mult: 1.0}, ...]
    gap_above_tiers = ex.get("gap_above_tiers") or []   # 高开降仓: [{above: 0, mult: 0.5}, ...]
    pool_tiers = ex.get("pool_n_tiers") or []           # 池深分档下注: [{min: 41, mult: 1.0}, ...]
    min_top1_score = ex.get("min_top1_score")     # 可选: 当日top1评分低于该值则放弃买入(质量门槛)
    gate_scale = bool(ex.get("score_gate_pool_scale", False))   # 第一百轮: 门槛按池内z分理论上限缩放
    gate_ref = float(ex.get("score_gate_ref_pool", 40) or 40)    # 缩放基准池深(该池深下门槛=min_top1_score)
    # 第九十六轮: 评分分档下注 [{min: 1.5, mult: 1.0}, {min: 1.1, mult: 0.6}, ...] 降序匹配
    # —— 门槛降低后不再"一刀切空仓", 低分票只给小仓, 高分票才满仓; 兼顾笔数与期望。
    score_tiers = ex.get("score_tiers") or []
    skip_gap_deep = ex.get("skip_gap_deep")       # 可选: 缺口深于该值(更恐慌)也放弃(崩塌保护)
    cash = float(bt_cfg["initial_capital"])
    curve: dict[str, float] = {}
    trades: list[dict] = []
    pending_buy: tuple[str, list[dict]] | None = None   # (buy_date, picks)
    holding: list[dict] = []                            # 未平仓持仓
    empty_days = 0
    fee_total = 0.0
    pending_cash = 0.0   # T+2收盘卖模式: 当日收盘卖出回笼、次日开盘方可投入的资金
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
        nonlocal cash, fee_total
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
        if (not day_ok) and day_frac and day_frac.get(d, 0) > 0:
            floor = ex.get("block_gap_below", floor)
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
                p = {**p, "_gap_below": fb.get("gap_below")}
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
            if gap * 100 >= ex["skip_open_gain"]:
                trades.append(_trade(p, "skip", d, note=f"开盘高开{gap*100:.1f}%"))
                continue
            if ex.get("skip_open_loss", 0) and gap * 100 <= -abs(ex["skip_open_loss"]):
                trades.append(_trade(p, "skip", d, note=f"开盘低开{gap*100:.1f}%"))
                continue
            # dip 承接过滤: 仅买缺口 <= floor% 的深低开票(0=不过滤); 备用池可配专属承接线
            _fl = p.get("_gap_below")
            _fl = floor if _fl is None else _fl
            # 第八十轮: 名次梯度承接线(可选) —— 池内名次越靠后要求缺口越深,
            # 用"更激进的折价"换深名次的质量, 让扩池新增的成交只发生在最极端的恐慌日。
            # 配置形如 [{rank_above: 5, gap_below: -3.0}, ...], 取第一个满足条件的档。
            for _t in (ex.get("gap_rank_tiers") or []):
                if int(p.get("_rank") or 0) >= int(_t.get("rank_above", 1)):
                    _tg = float(_t.get("gap_below", _fl))
                    _fl = min(_tg, float(floor)) if floor else _tg
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
            if _gate is not None and p.get("score", 0) < _gate:
                trades.append(_trade(p, "skip", d, note=f"top1评分{p.get('score',0):.2f}低于门槛{_gate:.2f}"))
                continue
            qualified.append({**p, "_row": row, "_gap": gap, "_lim_px": _lim_px})
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
            else:
                frac *= ex.get("blocked_frac", 0.0)  # 大盘风控期参与系数: 0=空仓(默认), 0.5=半仓
        if frac <= 0:
            return   # 第五十二轮修复: 信号日可买但买入日突发风控/恐慌(frac==0)时,
                     # 原实现alloc=0且"0<0"不拦截, 产生units=0的幽灵成交(ret_pct=0.0,
                     # 污染交易次数/胜率/连亏计数; 资金路径本就无影响, 此处不产生任何记录)
        equity = cash + sum(
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
            if cash <= 0:
                break
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
            alloc = min(pick_frac * equity, cash)
            # 剩余现金不足"本笔目标仓位"的30%才放弃(第九十七轮修正: 原以满仓 frac 为基准,
            # 在评分档×缺口档叠加把目标仓位压到 30% 以下时会把本可成交的单直接丢弃 ——
            # 高开降仓的实验里因此凭空少掉 159 笔交易; 判据应与本笔实际目标仓位同尺度)。
            if alloc < pick_frac * equity * 0.3:
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
            })
        # 未成交部分留在现金中, 复利滚动
        # 注: A股T+1制度, 当日买入的股票当日不可卖出, 因此不存在任何
        #     "买入当日止损/止盈"逻辑; 所有持仓最早于次一交易日开盘卖出。

    def do_sell(d: str, idx: int = 0):
        hold_extra = int(ex.get("hold_extra_days", 0) or 0)  # T+2开盘后再多持有N个交易日(0=原口径)
        frac_open = min(1.0, max(0.0, float(ex.get("sell_open_frac", 1.0) or 0.0)))  # 开盘卖出占比
        nonlocal cash, fee_total, pending_cash, loss_streak
        remaining = []
        for h in holding:
            if idx - h.get("buy_idx", 0) < 1 + hold_extra:
                remaining.append(h)   # 未到卖出时点(持有期延长口径)
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
            u_o = 0.0 if sold_at_close else h["units"] * frac_open
            u_c = h["units"] - u_o
            gross_o = u_o * open_adj * (1 - slip)
            gross_c = u_c * close_adj * (1 - slip)
            net_o = gross_o * (1 - stamp) * (1 - comm)
            net_c = gross_c * (1 - stamp) * (1 - comm)
            proceeds = net_o + net_c
            fee = gross_o * (stamp + comm) + gross_c * (stamp + comm)
            fee_total += fee
            # 开盘卖出: 资金当日开盘即回笼可再投入; 收盘卖出(整日/跌停回退/延卖/分批尾腿):
            # 资金收盘后回笼, 次一交易日开盘才可用于买入(防止虚假资金复用)
            if u_o:
                cash += net_o
            if u_c:
                pending_cash += net_c
            if 0 < frac_open < 1 and not sold_at_close:
                note = (note + f" 分批{frac_open:.0%}开盘/{1 - frac_open:.0%}收盘").strip()
            denom = h["units"] * h["buy_eff"] * (1 + comm)
            r = proceeds / denom - 1 if denom and denom > 0 else 0.0
            trades.append({
                "signal_date": h["signal_date"], "code": h["code"], "name": h.get("name", ""),
                "rank": h.get("_rank"), "pool": h.get("pool", "main"),
                "strategy": h.get("strategy", "dip"),
                "buy_date": h["buy_date"], "buy_px": h["buy_px_raw"],
                "fill": h.get("fill", "open"),
                "sell_date": sell_d, "sell_px": float(row["close" if sold_at_close else "open"]),
                "ret_pct": round(r * 100, 2), "note": note,
                # 第九十七轮: 成交明细记录"下注依据"(评分/竞价缺口/当日候选池深度),
                # 便于事后审计"为什么这笔下多重" —— 此前 trades 只有结果没有依据。
                "score": h.get("score"),
                "gap_pct": (round(float(h["_gap"]) * 100, 2) if h.get("_gap") is not None else None),
                "pool_n": h.get("_pool_n"),
            })
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
        # 开盘: 先卖出昨日到期持仓(T-2信号的池), 再买入(T-1信号的池)
        do_sell(d, idx)
        if pending_buy is not None:
            _, picks = pending_buy
            do_buy(picks, d, idx)
            pending_buy = None
        # 收盘估值
        total = cash + pending_cash + sum(h["units"] * float(per_code[h["code"]].loc[d]["close_adj"])
                                          for h in holding if d in per_code[h["code"]].index)
        curve[d] = total
        # 收盘后: 决定明日计划
        reg_row = reg.loc[d] if d in reg.index else None
        if day_frac:
            eff_today = day_frac.get(d, 1.0)
        else:
            eff_today = 0.0 if (reg_row is not None and not bool(reg_row["trade_ok"])) \
                and not ex.get("blocked_frac", 0.0) else 1.0
        blocked = reg_row is not None and not bool(reg_row["trade_ok"]) and eff_today <= 0
        # 连亏熔断: 连续亏损达阈值后, 从当日起再暂停 pause_days 个交易日(买入端)
        paused = False
        if ps_streak > 0 and loss_streak >= ps_streak:
            if idx > pause_until:
                pause_until = idx + ps_days
            paused = idx <= pause_until
        if blocked or paused:
            empty_days += 1
            pending_buy = (d, [])
            # 注: 大盘风控触发仅限制次一交易日不买入; 持仓(含当日买入票)
            #     均按既定 T+2(或之后)开盘卖出, 遵守T+1制度, 无任何当日卖出。
        else:
            picks, _sel_stats = select_stocks(d, by_date[d], cfg, names)
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
            "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "start": main_dates[0], "end": main_dates[-1], "days": n,
            "initial_capital": bt_cfg["initial_capital"],
            "rules": "T+1开盘买入/T+2开盘卖出/复利/佣金万2.5+印花税万5+滑点千1",
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
            "空仓天数": empty_days,
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
