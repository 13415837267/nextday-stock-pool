# -*- coding: utf-8 -*-
"""每日运行入口(供 GitHub Actions 每日收盘后调用)。

流程:
  0. 信号日幂等闸门: 非交易日不空转, 按"最新完整交易日"补缺(防 schedule 延迟跨日丢数据)
  1. 确定最新有完整数据的交易日 T
  2. 抓取近 regime.warmup_days 个交易日全市场日线 -> 计算指标与大盘风控
  3. 在 T 日横截面上执行选股, 得到"次日(T+1)计划候选池"
  4. 对最终候选逐票核查上市天数(过滤次新股)
  5. 结算 T-2 日发布池的实盘战绩(T-1开盘买入, T开盘卖出), 同步落持久账本:
     监测样本(append-only, 不受history 60条展示裁剪限制) + 风控空仓日虚拟对照
  6. 输出 output/latest.json / output/history.json, 并在 CHANGELOG.md 追加记录
"""
from __future__ import annotations

import json
import time
from datetime import datetime, timedelta, timezone

TZ_CN = timezone(timedelta(hours=8))  # 北京时间: Actions 运行于 UTC, 展示型时间戳统一 +8

import pandas as pd

from .config import ROOT, load_config
from .data_api import ZZClient
from .monitor import compute_monitor
from .rule_text import sell_rule_text
from .selector import STRATEGY_CN, build_indicators, regime_series, select_stocks, split_by_date
from .backtest import fetch_panel

DISCLAIMER = "本候选池为量化模型自动生成, 仅供研究参考, 不构成任何投资建议。股市有风险, 入市需谨慎。"


def _fmt(d: str) -> str:
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


def _config_fp(cfg: dict) -> str:
    """选股/执行关键参数的指纹: 参数一变, 同一信号日也要重算(三端同步的开关)。

    只纳入会改变选股结果或成交判定的字段; 注释、README、仓位小数等不影响结果的
    改动不会触发重算, 避免无意义地重刷 output。
    """
    import hashlib

    u = cfg.get("universe") or {}
    s = cfg.get("selection") or {}
    e = cfg.get("execution") or {}
    key = {
        "boards": list(u.get("boards") or []),
        "price": [u.get("price_min"), u.get("price_max")],
        "turnover": [u.get("turnover_min"), u.get("turnover_max")],
        "amount_min_yi": u.get("amount_min_yi"),
        "listed_min_bars": u.get("listed_min_bars"),
        "mode": s.get("mode"),
        "disp_n": s.get("display_top_n"),
        "exec_n": s.get("exec_top_n"),
        "dip": [s.get("dip_pct_min"), s.get("dip_pct_max"), s.get("dip_vol_ratio_max"),
                s.get("dip_mom20_min"), s.get("dip_mom20_max"), s.get("dip_dist_low20_min"),
                s.get("dip_dist_low20_max"),   # 第一百一十三轮: 距20日低点涨幅上限(可选)
                s.get("dip_close_pos_max"), s.get("dip_weights")],
        "bias20_max": s.get("bias20_max"),
        "fallbacks": [(f.get("mode"), f.get("top_n"), f.get("gap_below"), f.get("segment"))
                      for f in (s.get("fallbacks") or [])],
        "slot": bool(s.get("fallback_slot")),
        "gap": e.get("buy_gap_below"),
        "block_gap": e.get("block_gap_below"),
        "scd": e.get("sell_close_if_open_drop"),
        "spb": e.get("sell_close_if_profit_below"),   # 第九十八轮: 浮亏延卖阈值
        "sell_frac": e.get("sell_open_frac"),          # 第九十八轮: 分批出场(开盘卖出占比)
        "max_buy": e.get("max_buy_per_day"),
        "alloc": e.get("alloc_frac_pct"),
        "score_gate": e.get("min_top1_score"),          # 第九十五轮: 质量门槛
        "score_tiers": [(t.get("min"), t.get("mult")) for t in (e.get("score_tiers") or [])],  # 第九十六轮: 分档下注
        "gap_above": [(t.get("above"), t.get("mult")) for t in (e.get("gap_above_tiers") or [])],  # 第九十七轮: 高开降仓
        "pool_tiers": [(t.get("min"), t.get("mult")) for t in (e.get("pool_n_tiers") or [])],  # 第九十九轮: 池深分档
        # 第一百七十五轮: 加仓档必须进指纹 —— 它直接改变成交后的仓位, 属于"会改变成交判定的字段"
        # (第一百七十三轮的 mkt_boost_tiers 当初漏了, 一并补上)。
        "mkt_boost": [(t.get("mkt_above"), t.get("mkt_below"), t.get("mult"))
                      for t in (e.get("mkt_boost_tiers") or [])],
        "pool_boost": [(t.get("min"), t.get("max"), t.get("mult"))
                       for t in (e.get("pool_n_boost_tiers") or [])],
        "gap_floor": [(t.get("below"), t.get("min_pool"), t.get("floor")) for t in (e.get("gap_floor_tiers") or [])],  # 第一百零一轮: 深低开下限
        "max_block": (cfg.get("regime") or {}).get("max_block_days"),  # 第九十五轮: 强制复参与上限
        "blocked_frac": e.get("blocked_frac"),          # 第一百零二轮: 风控日参与系数
        "soft_trend": (cfg.get("regime") or {}).get("soft_trend_frac"),  # 第一百一十三轮: 分档参与(可选)
        # 第一百七十六轮: 产物结构版本 —— plans(明日计划)是纯展示层, 不进"成交判定",
        # 但结构一变就必须重算一次(否则幂等跳过会留下没有 plans 的旧 latest.json)。
        "plan_ver": 1,
    }
    raw = json.dumps(key, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


def _pos_tier_mult(score, ex: dict) -> float:
    """评分分档下注乘数(与回测 do_buy 的 score_tiers 同口径, 降序匹配)。"""
    for t in (ex.get("score_tiers") or []):
        if score is not None and float(score) >= float(t["min"]):
            return float(t.get("mult", 1.0))
    return 1.0


def _gap_tier_mult(gap: float, ex: dict, score=None) -> tuple[float, str]:
    """高开降仓乘数(与回测 do_buy 的 gap_above_tiers 同口径, 降序匹配)。

    gap 单位为百分数(T+1 开盘相对 T 日收盘)。返回 (乘数, 说明文案)。
    只降仓、不拒单 —— 每日计划票仍按开盘价成交, 交易次数不变。
    档内 `score_below` 与回测一致: 评分达到该值的票跳过本档(强势股抢筹不降仓)。
    """
    for t in (ex.get("gap_above_tiers") or []):
        if "score_below" in t and score is not None and float(score) >= float(t["score_below"]):
            continue
        if float(gap) >= float(t.get("above", 0.0)):
            _m = float(t.get("mult", 1.0))
            return _m, f"缺口{gap:+.1f}%→{_m:.0%}仓({float(t['above']):g}档)"
    return 1.0, ""


def _pool_tier_mult(pool_n, ex: dict) -> tuple[float, str]:
    """池深分档乘数(与回测 do_buy 的 pool_n_tiers 同口径, 降序匹配)。

    pool_n = T 日收盘后满足 dip 全部条件的票数(盘后即知, 无未来函数)。
    池越宽 = 越多同风格票同步被错杀 = 次日修复有板块共振; 池越窄 = 孤立回调。
    与回测一致: pool_n 为 0/None(备用池补位或历史记录缺字段) 不匹配任何档 -> 满仓。
    """
    if pool_n is None:
        return 1.0, ""
    _n = int(pool_n)
    for t in (ex.get("pool_n_tiers") or []):
        if _n >= int(t["min"]):
            _m = float(t.get("mult", 1.0))
            # 第一百轮文案修正: pool_n 是"当日达标票数"(通过全部 dip 条件并完成评分),
            # 不是"买入只数"(恒为 1) —— 原写作"池深N只"与页面"候选 1 只"打架。
            return _m, (f"达标{_n}只→{_m:.0%}仓(仅买评分第1名)" if _m != 1.0 else "")
    return 1.0, ""


def _gap_floor_mult(gap: float, pool_n, ex: dict) -> tuple[float, str]:
    """深低开加仓下限(第一百零一轮, 与回测 do_buy 的 gap_floor_tiers 同口径)。

    深低开是"错杀"的直接证据, 池深折价衡量的是"共振不足"的间接代理。
    当直接证据足够强(缺口 <= below)且池子不算太窄(>= min_pool)时, 撤销池深折价回到满仓。
    返回 (下限系数, 文案); 0 表示不适用(沿用叠乘结果)。
    """
    for t in (ex.get("gap_floor_tiers") or []):
        try:
            if gap <= float(t["below"]) and int(pool_n or 0) >= int(t.get("min_pool", 0)):
                return float(t.get("floor", 1.0)), f"深低开{gap:.1f}%→撤销池深折价"
        except (KeyError, TypeError, ValueError):
            continue
    return 0.0, ""


def _boost_mult(pool_n, mkt_ret: float, ex: dict) -> tuple[float, str]:
    """加仓乘数(第一百七十五轮, 与回测 do_buy 的 mkt_boost_tiers / pool_n_boost_tiers 同口径)。

    两个维度独立叠乘, 都可以把仓位推过 100%(允许透支, 受 leverage 与 margin_rate 约束):
      · 市场维度 `mkt_boost_tiers`: 信号日全市场涨跌中位数落在 [mkt_above, mkt_below]
        —— 系统性错杀但还没到趋势崩塌(第一百七十三轮), 乘 mult;
      · 策略维度 `pool_n_boost_tiers`: 当日达标票数 >= min —— 池越深说明越多同风格票
        同时被打成缩量阴线, 这是策略自己看到的恐慌强度(第一百七十五轮)。
    与 `pool_n_tiers` 的分工必须写清: 后者是"池浅降仓"(第 99 轮), 本函数是"池深加仓",
    方向相反 —— 它们是同一组数字的两端, 不是同一个规则的两个写法。
    """
    mult, notes = 1.0, []
    try:
        _m = float(mkt_ret)
    except (TypeError, ValueError):
        _m = 0.0
    for t in (ex.get("mkt_boost_tiers") or []):
        try:
            _hi = t.get("mkt_below")
            _lo = t.get("mkt_above")
            if _hi is not None and _m > float(_hi):
                continue
            if _lo is not None and _m < float(_lo):
                continue
            _k = float(t.get("mult", 1.0))
        except (TypeError, ValueError):
            continue
        if _k != 1.0:
            mult *= _k
            notes.append(f"全市场{_m:+.1f}%错杀→{_k:.0%}仓")
        break
    for t in (ex.get("pool_n_boost_tiers") or []):
        try:
            _n = int(pool_n or 0)
            if _n < int(t.get("min", 0)):
                continue
            _mx = t.get("max")
            if _mx is not None and _n > int(_mx):
                continue
            _k = float(t.get("mult", 1.0))
        except (TypeError, ValueError):
            continue
        if _k != 1.0:
            mult *= _k
            notes.append(f"达标{_n}只(池深)→{_k:.0%}仓")
        break
    return mult, "·".join(notes)


def _market_median_pct(by_date: dict, date: str) -> float:
    """信号日全市场涨跌幅中位数("典型个股"当天涨跌): 只用 T 日收盘横截面, 盘后即知。"""
    try:
        df = by_date.get(date)
        if df is None or not len(df):
            return 0.0
        return float(pd.Series(df["pct"]).median())
    except Exception:  # noqa: BLE001 数据异常时不启用市场语境修正
        return 0.0


def _mkt_context(pct_stock, mkt_ret: float, ex: dict) -> tuple[float, str, float, str]:
    """第一百零三轮 市场语境修正, 与回测 do_buy 的 knife_tiers / mkt_floor_tiers 同口径。

    返回 (刀口乘数, 刀口文案, 恐慌日仓位下限, 恐慌日文案):
      - 刀口: 大盘平稳/上涨(中位数 > mkt_above)而个股超额下跌(个股-市场) <= res_below
        → 个股自己的事, 不是系统性错杀, 乘 mult 缩仓(第一百一十四轮起 mult=0 → 直接剔除);
      - 恐慌日: 全市场跌到 mkt_below 以下 → 系统性错杀, 仓位不低于 frac*floor。
    """
    km, kt = 1.0, ""
    try:
        _pct = float(pct_stock or 0.0)
    except (TypeError, ValueError):
        _pct = 0.0
    for t in (ex.get("knife_tiers") or []):
        if mkt_ret > float(t.get("mkt_above", -0.5)) and (_pct - mkt_ret) <= float(t.get("res_below", -4.0)):
            km = float(t.get("mult", 1.0))
            kt = (f"大盘中位数{mkt_ret:+.1f}%而个股超额{_pct - mkt_ret:+.1f}pp"
                  + ("→不参与(个股独跌)" if km <= 0 else f"→{km:.0%}仓(个股独跌)"))
            break
    mf, ft = 0.0, ""
    for t in (ex.get("mkt_floor_tiers") or []):
        if mkt_ret <= float(t.get("mkt_below", -1.5)):
            mf = float(t.get("floor", 1.0))
            ft = f"全市场下跌(中位数{mkt_ret:+.1f}%)→回{mf:.0%}仓"
            break
    return km, kt, mf, ft


def _exec_gate(rb, gap: float, ex: dict, floor: float) -> str | None:
    """非风控执行闸(与回测口径一致, 第49轮自内联elif链提取复用): 返回拦截原因, None=放行。

    语义与提取前逐条一致: 停牌/一字涨停/高开放弃/低开过深/缺口未达承接线。
    供实盘结算与风控空仓日虚拟对照共用 —— 对照口径="完全放行风控, 其余闸不变"。
    """
    if rb is None or rb["is_paused"] == 1 or rb["vol"] <= 0:
        return "停牌/无成交"
    if rb["open"] >= rb["high_limit"] - 1e-9:
        return "开盘一字涨停"
    if gap >= ex["skip_open_gain"]:
        return f"高开{gap:.1f}%放弃"
    if ex.get("skip_open_loss", 0) and gap <= -abs(ex["skip_open_loss"]):
        return f"低开{gap:.1f}%过深"
    if floor and gap > floor:
        return f"缺口{gap:.1f}%未达承接线{floor}%"
    return None


def _position_note(ex: dict) -> str:
    """按执行配置生成仓位说明(与回测口径一致)。"""
    frac = float(ex.get("alloc_frac_pct", 5.0))
    maxn = int(ex.get("max_buy_per_day", 0))
    gap = float(ex.get("buy_gap_below", 0) or 0)
    tiers = ex.get("score_tiers") or []
    tier_txt = ""
    if tiers:
        seen = []
        for t in tiers:
            seen.append(f"≥{float(t['min']):g}→{float(t.get('mult', 1.0)) * frac:g}%")
        tier_txt = "(评分分档: " + "/".join(seen) + " 总资产)"
    if maxn == 1:
        pos = f"若开仓: 买入评分第1名 1 只, 最高 {frac:g}% 总资产仓位" + tier_txt
    elif maxn > 1:
        pos = f"若开仓: 按评分买入最多 {maxn} 只, 每笔约 {frac:g}% 总资产"
    else:
        pos = f"若开仓: 达标票每笔约 {frac:g}% 总资产"
    if gap:
        pos += f"; 仅买开盘缺口 <= {gap:g}% 的低开票(风控日以判定卡承接线为准)"
    else:
        pos += "; 平日不设缺口门槛, T+1 开盘价直接成交(仅停牌/一字涨停买不到)"
    gat = ex.get("gap_above_tiers") or []
    if gat:
        seg = "/".join(f"≥{float(t['above']):g}%→{float(t.get('mult', 1.0)):.0%}"
                       for t in gat)
        pos += f"; 高开减仓: 竞价缺口 {seg}(照常成交, 只降仓位)"
    ptiers = ex.get("pool_n_tiers") or []
    if ptiers:
        seg = "/".join(f"≥{int(t['min'])}只→{float(t.get('mult', 1.0)):.0%}"
                       for t in ptiers)
        pos += f"; 池深分档: 当日达标票 {seg}(只降仓位, 不弃单; 达标≠买入只数, 恒买评分第1名)"
    for _t in (ex.get("gap_floor_tiers") or []):
        pos += (f"; 深低开回满仓: 竞价缺口 ≤{float(_t['below']):g}% 且达标≥{int(_t.get('min_pool', 0))}只"
                f" → 撤销池深折价、回{float(_t.get('floor', 1.0)):.0%}仓")
        break
    for _t in (ex.get("knife_tiers") or []):
        _km = float(_t.get("mult", 1.0))
        pos += (f"; 刀口保护: 信号日全市场涨跌中位数 >{float(_t['mkt_above']):g}% 且个股超额下跌"
                f"≤{float(_t['res_below']):g}pp → "
                + ("不参与(个股独跌≠系统性错杀)" if _km <= 0 else f"{_km:.0%}仓(个股独跌≠系统性错杀)"))
        break
    for _t in (ex.get("mkt_floor_tiers") or []):
        pos += (f"; 恐慌日回满仓: 全市场涨跌中位数 ≤{float(_t['mkt_below']):g}% → 撤销评分/池深折价回"
                f"{float(_t.get('floor', 1.0)):.0%}仓")
        break
    for _t in (ex.get("mkt_boost_tiers") or []):
        _hi, _lo = _t.get("mkt_below"), _t.get("mkt_above")
        _rng = (f"[{float(_lo):g}%, {float(_hi):g}%]" if _lo is not None
                else f"≤{float(_hi):g}%" if _hi is not None else "全区间")
        pos += (f"; 恐慌加仓: 信号日全市场涨跌中位数落在 {_rng} → 仓位×{float(_t.get('mult', 1.0)):g}"
                f"(超过100%的部分按融资利率计息)")
        break
    for _t in (ex.get("pool_n_boost_tiers") or []):
        pos += (f"; 池深加仓: 当日达标票 ≥{int(_t.get('min', 0))} 只 → 仓位×"
                f"{float(_t.get('mult', 1.0)):g}(池越深=越多票同步被错杀)")
        break
    if float(ex.get("blocked_frac", 0) or 0) > 0 and ex.get("block_gap_below"):
        pos += (f"; 风控日: 只在竞价缺口 ≤{float(ex['block_gap_below']):g}% 时按"
                f"{float(ex['blocked_frac']):.0%}仓接恐慌深跌, 否则空仓")
    pos += "; 出场: " + sell_rule_text(ex) + "(遵守A股T+1制度, 无当日卖出)"
    return pos


def _month_gain_state(led_exec: list, history: list, ym: str, close_of: dict | None,
                      cfg: dict, include_hold: bool = True) -> tuple[float, float, str]:
    """当月净值增益与月度盈利熔断状态(第五十三轮, 回测 risk.month_profit_cap 的实盘对齐)。

    背景: 回测买入端有"当月净值增益超cap_pct%后买入仓位×cut_frac"(第八轮引入),
    此前实盘指引缺失该机制 —— 回测收益数字包含降仓行为(500日窗口NO_CAP对照
    +985pp), 网页指引却恒显示全仓, 实盘照做将与回测口径分叉。
    口径对齐回测"买入时点equity/月初nav-1"(单吊先卖后买, 买入时点无持仓):
      include_hold=True(指引侧, T晚为T+1买入准备): 当月已结算复利 × (1+当前持仓浮盈)
        —— 当前持仓将于T+1开盘卖出实现浮盈, 恰好计入买入时点;
      include_hold=False(结算侧, 买入日当天重算): 仅已结算复利 —— 昨日浮盈已实现入账,
        唯一持仓即本记录自身, 其浮盈不属于买入时点状态。
    已结算源: 持久账本exec优先(不随history裁剪), 空则回退history已结算记录
    (与监测器同款回退)。返回 (month_gain_pct, cap_frac, note)。
    """
    _mcap = (cfg.get("risk") or {}).get("month_profit_cap") or {}
    mc_cap, mc_cut = float(_mcap.get("cap_pct", 0) or 0), float(_mcap.get("cut_frac", 1.0) or 1.0)
    if mc_cap <= 0 or mc_cut >= 1.0:
        return 0.0, 1.0, ""
    rets = [float(s["ret_pct"]) for s in led_exec
            if str(s.get("publish_date", "")).replace("-", "")[:6] == ym]
    if not rets:
        rets = [float(r["realized"]["avg_ret_pct"]) for r in history
                if r.get("status") == "已结算" and r.get("realized")
                and str(r.get("publish_date", "")).replace("-", "")[:6] == ym
                and r.get("regime_ok", True) is not False]
    gain = 1.0   # 净值倍数起步(空样本=0%增益); 此前误用100起步导致空样本虚增9900%
    for r in rets:
        gain *= (1 + r / 100)
    # 未平仓浮盈(单吊口径: 持仓市值波动直接计入当月净值; 浮盈按最新可得收盘价)
    if include_hold:
        hold = next((r["executed"] for r in history
                     if (r.get("executed") or {}).get("holding")
                     and str((r["executed"].get("buy_date") or "")).replace("-", "")[:6] == ym), None)
        if hold and close_of and close_of.get(hold.get("code")) and hold.get("buy_px"):
            gain *= (1 + (float(close_of[hold["code"]]) / float(hold["buy_px"]) - 1))
    gain = (gain - 1) * 100
    if gain > mc_cap:
        return gain, mc_cut, f"月度盈利熔断生效: 当月净值+{gain:.1f}%>{mc_cap:g}%, 买入仓位×{mc_cut:g}"
    return gain, 1.0, ""


def _sell_exit(rs, row_of, cal, sell_d: str, code: str, ex: dict, buy_adj: float | None = None):
    """卖出日退出判定(第五十四轮, 与回测 do_sell 逐条对齐):
    T+2停牌 -> 向后顺延最多5个交易日开盘卖; 开盘触及跌停且sell_limit_down_fallback
    (config已启用) -> 保守按当日收盘卖。返回 (sell_row, sell_dd, sold_at_close, note);
    None = 顺延窗口内无可交易行(未恢复/无数据), 留待后续结算 —— 实盘每日重跑天然
    顺延, 与回测"模拟内即时顺延"语义等价。差异说明: 回测"退市按末收盘"强平在实盘
    端不可复刻(daily_run面板仅近期窗口, 无全史末收盘); 十年回测该情形0笔、停牌顺延
    0笔、跌停保守2笔(002881.SZ/002612.SZ), 影响面见CHANGELOG第五十四轮量化。
    """
    row, dd, sold_close, note = rs, sell_d, False, ""
    frac_open = min(1.0, max(0.0, float(ex.get("sell_open_frac", 1.0) or 0.0)))
    if row is not None and (row["is_paused"] == 1 or row["vol"] <= 0):
        row = None
    if row is None:
        sd = str(sell_d).replace("-", "")
        if sd in cal:
            pos = cal.index(sd)
            for j in range(1, 6):
                if pos + j >= len(cal):
                    break
                r2 = row_of(cal[pos + j], code)
                if r2 is not None and r2["is_paused"] == 0 and r2["vol"] > 0:
                    row, dd, note = r2, cal[pos + j], "停牌顺延"
                    break
        if row is None:
            return None
    if ex.get("sell_at_open", True) and ex.get("sell_limit_down_fallback", True) \
            and float(row["open"]) <= float(row["low_limit"]) + 1e-9:
        sold_close = True
        note = (note + " 开盘跌停按收盘").strip()
    # 第八十九轮: T+2开盘深跌(未跌停)延卖至收盘(与 backtest do_sell 同口径,
    # 仅用T+2开盘/昨收瞬时信息做决策, 无未来函数)
    scd = float(ex.get("sell_close_if_open_drop") or 0)
    if (not sold_close) and scd > 0 and float(row["pre_close"]) > 0 \
            and (float(row["open"]) / float(row["pre_close"]) - 1) * 100 <= -scd:
        sold_close = True
        note = (note + f" 开盘深跌{scd:g}%按收盘").strip()
    # 第九十八轮: T+2开盘相对**买入价**浮亏 -> 延卖至收盘(与 backtest do_sell 同口径)。
    # 与上面 scd 的区别: scd 看"相对昨收的跳空", 这里看"相对我的成本"——同一个跳空,
    # T+1 大涨后是浮盈、T+1 崩盘后是深亏, 日内修复概率完全不同。仅用 T+2 开盘瞬时信息。
    spb = ex.get("sell_close_if_profit_below")
    if (not sold_close) and spb is not None and buy_adj and float(buy_adj) > 0 \
            and (float(row["open_adj"]) / float(buy_adj) - 1) * 100 < float(spb):
        sold_close = True
        note = (note + " 开盘浮亏按收盘").strip()
    return row, dd, sold_close, note, (0.0 if sold_close else frac_open)


# ----------------------------------------------------------------------
# 第一百七十六轮: 明日计划(主抄底 / 副追高, 每日各 1 只)
# ----------------------------------------------------------------------
# 第一百七十七轮: T+1 竞价缺口 → 实际成交单笔收益(1500 日, 330 笔实成交,
# T+1 开盘买 / T+2 收盘卖, 含佣金万2.5×2 + 印花税万5 + 滑点千1×2)。
# 来源: tools/exp_line.py 复算 output/backtest.json 的 trades_all(逐笔), 与引擎同一份面板。
# 用途: 计划卡上的"缺口速查", 回答"高开还要不要买"——四档全正, 深低开最肥。
# 反直觉处: 高开 132 笔 +1.20%/t=3.24 依然显著为正; 全样本(含未成交的虚拟笔)
# 高开档却是 -0.243% —— 差别在评分门槛, 不在缺口。不要把这张表当成"挑缺口"的依据。
_GAP_LADDER = [
    {"band": "≤ -2.5%", "n": 31, "mean": 2.17, "win": 71.0},
    {"band": "-2.5% ~ -1%", "n": 65, "mean": 2.28, "win": 66.2},
    {"band": "-1% ~ 0", "n": 102, "mean": 1.28, "win": 56.9},
    {"band": "> 0（高开）", "n": 132, "mean": 1.20, "win": 54.5},
]

# 第一百七十八轮: 评分档位速查(实成交逐笔, T+1开→T+2收, 含双边成本)。
# lo/hi 是 score 区间(横截面 z 分), 两个窗口的样本与均值分开列 —— 因为**它们不一致**:
#   1500 日单调(0.69/1.25/1.62/2.12), 全史不单调(1.46/1.02/0.87/1.64, 1.5~1.8 反而最差)。
# 不一致本身就是结论: 评分排序**没有稳定预测力**(横截面 IC +0.0026, t=+0.28, 第 1 名
# 甚至低于全池均值)。所以本表只回答"这一笔落在哪个质量档", 不作为挑单或加仓依据。
_SCORE_BANDS = [
    {"lo": 1.0, "hi": 1.2, "n15": 65, "m15": 0.686, "w15": 47.7, "nall": 148, "mall": 1.464, "wall": 49.3},
    {"lo": 1.2, "hi": 1.5, "n15": 97, "m15": 1.245, "w15": 56.7, "nall": 223, "mall": 1.019, "wall": 55.2},
    {"lo": 1.5, "hi": 1.8, "n15": 89, "m15": 1.620, "w15": 57.3, "nall": 184, "mall": 0.866, "wall": 53.8},
    {"lo": 1.8, "hi": 99.0, "n15": 101, "m15": 1.950, "w15": 60.4, "nall": 170, "mall": 1.639, "wall": 60.0},
]


def _score_band(score) -> dict | None:
    """把当日评分映射到历史档位(不改变任何执行)。"""
    try:
        s = float(score)
    except (TypeError, ValueError):
        return None
    if s < 1.0:
        # 低于 min_top1_score 的票系统本来就不买(1499 日零成交)。卡片仍要能说清"它在门槛下",
        # 而不是留空 —— 否则用户看到"评分 0.95"却没有任何参照。
        return {"band": "< 1.0（低于质量门槛）", "n15": 0, "m15": None, "w15": None,
                "nall": 0, "mall": None, "wall": None, "below_gate": True}
    for b in _SCORE_BANDS:
        if b["lo"] <= s < b["hi"]:
            return {"band": f"{b['lo']:g} ~ {b['hi']:g}" if b["hi"] < 90 else f"≥ {b['lo']:g}",
                    "n15": b["n15"], "m15": b["m15"], "w15": b["w15"],
                    "nall": b["nall"], "mall": b["mall"], "wall": b["wall"]}
    return None


def _build_plans(cfg: dict, day, signal_date: str, verified: list, ranked: list,
                 ex: dict, t1: dict, regime_info: dict, names: dict | None = None) -> dict:
    """生成"明日计划"双方案卡片: 给人工盯盘执行, 不是自动下单指令。

    为什么单独一块而不是复用 picks:
      picks 是候选池(可多只、带仓位分档), 回答"买多少"; plans 是**单票作业单**,
      回答"看哪只、什么价位动手、什么情况放弃"。用户的实际用法是盘后看一眼、
      次日自己盯盘决定, 两件事的信息密度不同, 混在一张卡里会互相淹没。

    两条纪律:
      1) 计划票 != 建议买入。系统风控/评分不达标时, 卡片必须原样写出"系统口径是空仓",
         不能因为"人工可以自己决定"就把风险提示藏起来。
      2) 观察清单里每条能追到 config 已回测参数的, 标"回测口径"; 追不到的(分时确认、
         盘口承接)一律标"未回测"—— 不许把经验包装成统计结论。
    """
    plans = {"as_of": signal_date, "main": None, "alt": None,
             "manual": {
                 "steps": [
                     "09:15-09:25 看集合竞价: 缺口落在卡片区间内才挂单, 超出即放弃",
                     "09:25 成交价 = 开盘价; 高开触发降仓档时按卡片仓位执行, 不追单",
                     "09:30-09:35 只看不动: 若分时跌破开盘价 1% 且无量反抽, 可放弃(标[未回测])",
                     "T+2 14:55 前清仓; 副方案按持有天数/移动止损离场",
                 ],
                 "caveat": "分时确认与盘口承接两条无日线数据可回测, 属执行建议, 不计入任何统计结论。",
             }}

    # ---------- 主方案: 抄底(缩量阴线回调) ----------
    mp = verified[0] if verified else (ranked[0] if ranked else None)
    if mp:
        _gat = t1.get("gap_tier_note")
        _observe = []
        if not t1.get("allow_buy"):
            _observe.append(f"系统口径: 空仓 —— {t1.get('buy_note') or '当日不参与'}")
        else:
            _observe.append(
                f"系统口径: 可按 {t1.get('position_pct')}% 仓位参与"
                + (f"; {_gat}" if _gat else ""))
        _observe.append(
            f"开盘涨幅 < {float(ex.get('skip_open_gain', 10.0)):g}% 且非一字/非停牌才买得到"
            " [回测口径]")
        plans["main"] = {
            "strategy": "dip",
            "strategy_cn": "主方案 · 抄底(缩量阴线回调)",
            "code": mp.get("code"), "name": mp.get("name"),
            "close": mp.get("close"), "pct": mp.get("pct"),
            "score": mp.get("score"), "score_kind": "横截面 z 分(当日同池比较, 非 0~100)",
            # 第一百七十八轮: 评分档位体检 —— 实成交逐笔, 两个窗口分开列。
            # 必须一起给: 1500 日单调、全史不单调, 单给任何一个都会误导。
            "score_band": _score_band(mp.get("score")),
            "signal": mp.get("reasons"),
            "observe": _observe,
            "abort": [
                f"开盘涨幅 >= {float(ex.get('skip_open_gain', 10.0)):g}%(一字/接近涨停) [回测口径]",
                "停牌或全天无成交 [回测口径]",
                "开盘跌幅 <= -5%: 深破位, 修复概率显著下降 [未回测, 经验规则]",
            ],
            "exit": "T+2 当日 14:55 收盘卖出(持满整个 T+2 交易日)",
            "system_ok": bool(t1.get("allow_buy")),
            "position_pct": (t1.get("position_pct") if t1.get("allow_buy") else 0.0),
            # 第一百七十七轮: 缺口速查表 —— 人工盯盘最常问的是"明天高开还要不要买",
            # 这张表用实成交逐笔给答案(1500 日 330 笔, T+1开→T+2收, 含双边成本)。
            # 结论出乎直觉: 四档**全为正**, 深低开只是更肥(2.17%/71%)而不是唯一的正期望;
            # 高开 132 笔仍有 +1.20%/t=3.24 —— 因为评分门槛已经把质量筛过一遍, 缺口只是第二层。
            # 所以本表不用于"挑单", 只用于"知道自己在赚什么钱"。
            "gap_ladder": _GAP_LADDER,
        }

    # ---------- 副方案: 追高(低波池内风险调整动量) ----------
    try:
        from .satellite import STRATEGY_CN as _SAT_CN
        from .satellite import daily_params as _sat_params
        from .satellite import pick_list as _sat_pick
        # 副策略要求 bars>=250(上市满一年), 而主流程面板只预热 30~60 天 —— 直接在
        # 当日面板上跑, m_listed 会全 False、静默返回空池(2026-11-09 实测: alt 恒为 null)。
        # 这里按需补一份长面板; 命中 data/panel 缓存时耗时约 5 秒, 有缓存缺失才会联网。
        _sday = day
        try:
            if day is None or (len(day) and float(day["bars"].max()) < 260):
                from .backtest import prepare_context as _prep
                _sctx = _prep(cfg, 320)
                _sday = _sctx["by_date"].get(signal_date.replace("-", ""), day)
        except Exception:  # noqa: BLE001 补面板失败不阻断, 退回当日面板
            _sday = day
        sl = _sat_pick(_sday, cfg, (), 4)
    except Exception as _e:  # noqa: BLE001  副策略异常不能拖垮主流程
        sl, _SAT_CN, _sat_params = [], {}, None
        plans["alt_error"] = f"{type(_e).__name__}: {_e}"
    if sl:
        s = sl[0]
        P = _sat_params(cfg) if _sat_params else {}
        _gate_ok = bool(regime_info.get("ok"))
        plans["alt"] = {
            "strategy": P.get("mode", "lowvol"),
            "strategy_cn": "副方案 · 追高(低波池内风险调整动量)",
            "code": s.get("code"),
            "name": (names or {}).get(s.get("code")) or s.get("name") or s.get("code"),
            "close": s.get("close"), "pct": s.get("pct"),
            # 评分与执行排序同源: 池内排序分位(第 1 名=100), 不是 ivol 分
            "score": s.get("rank_pct"), "score_kind": "池内排序分位(0~100, 第1名=100)",
            "ivol20": s.get("ivol20"), "dist_ma5": s.get("dist_ma5"), "mom20": s.get("mom20"),
            "signal": (f"20日动量{s.get('mom20')}% / 特质波动{s.get('ivol20')}"
                       f" / 距5日线{s.get('dist_ma5')}%"),
            "observe": [
                (f"系统口径: {'大盘闸门已开, 可参与' if _gate_ok else '大盘闸门关闭(指数在MA20下), 系统不开新仓'}"
                 " [回测口径]"),
                f"竞价缺口须在 [{P.get('gap_min')}%, +{P.get('gap_max')}%] 内"
                " [回测口径: 低开破位不接、高开过多不追]",
                (f"挂单口径: 限价 {round(float(s.get('close') or 0) * (1 - float(P.get('pullback_pct') or 0) / 100), 2)}"
                 f"（= 收盘 {s.get('close')} × (1-{float(P.get('pullback_pct') or 0):g}%)）, "
                 f"{int(P.get('pullback_days') or 1)} 个交易日内触及即成交; 开盘低于限价按开盘价成交"
                 " [不是追开盘价: 1500 日实测按开盘价买只有 19.01%, 挂回踩单 32.38%]")
                if float(P.get("pullback_pct") or 0) > 0 else "竞价后按开盘价买入",
            ],
            "abort": [
                f"低开 <= {P.get('gap_min')}%: 破位不接 [回测口径]",
                f"高开 >= +{P.get('gap_max')}%: 追高成本过高 [回测口径]",
            ],
            "exit": (f"持有 {P.get('hold_days')} 个交易日后收盘卖; 移动止损 "
                     f"{float(P.get('trail') or 0) * 100:g}% / 固定止损 "
                     f"{float(P.get('stop') or 0) * 100:g}% [回测口径]"),
            # 第一百七十九轮: 回踩限价 —— 计划卡必须给出"挂哪个价、挂几天", 否则用户
            # 只能按开盘价追, 回测的 32% 在实盘就落不到(回测口径是限价成交)。
            "order": ({
                "limit": round(float(s.get("close") or 0) * (1 - float(P.get("pullback_pct") or 0) / 100), 2),
                "pct": float(P.get("pullback_pct") or 0),
                "days": int(P.get("pullback_days") or 1),
                "base": s.get("close"),
            } if float(P.get("pullback_pct") or 0) > 0 and s.get("close") else None),
            "system_ok": _gate_ok,
            "position_pct": round(100.0 / max(1, int(P.get("max_pos") or 1)), 1),
            # 第一百七十六轮: 只买第 1 名的代价(1500 日实测, 同引擎同成本)
            "position_note": (
                "系统口径是一次买满 4 只、每只 25%; 只买第 1 名(单吊) 1500 日年化 9.09% / "
                "回撤 -43.71%, 每日 1 只滚动 8.10% / -17.87%, 都不如 4 只并发的 18.99% / -18.67%"
                " —— 差距来自资金利用率与分散, 不是选股(第 1 名的单笔并不更强)。"),
            "peers": [{"code": x.get("code"),
                       "name": (names or {}).get(x.get("code")) or x.get("name") or x.get("code"),
                       "rank_pct": x.get("rank_pct"), "mom20": x.get("mom20")}
                      for x in sl[1:4]],
        }
    elif "alt_error" not in plans:
        plans["alt_note"] = "当日副策略无达标候选(池子不足或大盘闸门关闭)"

    # 第一百七十六轮: 主腿空窗时, 副腿是当日唯一可执行方案 —— 这不是"凑数的备选",
    # 1500 日实测它在主策略空仓日的表现(292 笔 / 均值 +1.612% / 胜率 51.0%)
    # 反而高于自身整体(392 笔 / +1.366% / 50.3%)。第 133 轮 trend 补位是 -0.35%,
    # 差别在池子: 低波池抗跌, 追涨池在风控日正好是最差的一格。
    if plans.get("main") and not plans["main"].get("system_ok") and plans.get("alt"):
        plans["note"] = ("主方案今日系统口径为空仓 → 副方案是唯一可执行方案。"
                         "1500 日实测: 副策略在主策略空仓日的 292 笔均值 +1.612%, "
                         "高于其自身整体 +1.366% —— 补位不是凑数。")
    return plans


def main():
    cfg = load_config()
    client = ZZClient(cfg)
    t0 = time.time()

    # ---- 0. 交易日历标记: 非交易日不再硬性空转, 改由"信号日幂等闸门"决定是否补算 ----
    # 历史教训(2026-10-31): 原实现在执行日非交易日时直接 return。但 GitHub schedule 排队延迟
    # 常达数小时(实测 6h+), 18:00/20:00/22:00 三个槽会被推迟到次日 00:2x 才执行 —— 此时"执行日"
    # 已变成周六, 闸门判定非交易日空转, 前一周期的收盘数据(2026-10-30 龙虎榜/选股/账本)便永久
    # 丢弃, 而 latest.json 停在上一日, 页面无任何报错, 只能靠人工比对 data/lhb 日期才发现。
    # 现改为: 无论执行日是否交易日, 都先解析"最新完整交易日", 再由信号日幂等闸门判断该日是否
    # 已处理; 未处理则照常补算(延迟跨日自愈), 已处理则快速跳过(防重复计算)。
    try:
        from zoneinfo import ZoneInfo
        _now = datetime.now(ZoneInfo("Asia/Shanghai"))
    except Exception:  # noqa: BLE001 时区库不可用时退回系统本地时间
        _now = datetime.now()
    _today = _now.strftime("%Y%m%d")
    try:
        _today_is_trading = client.is_trading_day(_today)
    except Exception as _e:  # noqa: BLE001 日历不可用时按非交易日处理(继续走补缺路径, 不误跳过)
        print(f"⚠ 交易日历查询失败({_e}), 按非交易日继续执行补缺检查")
        _today_is_trading = False
    if _today_is_trading:
        print(f"今日{_fmt(_today)}为A股交易日, 开始运行")
    else:
        print(f"今日{_fmt(_today)}非A股交易日(周末或法定假日), 继续执行补缺检查(覆盖延迟跨日的上一交易日)")

    # ---- 1. 最新完整交易日 ----
    # 历史: trade_days只含已过去交易日; 未来: 确定信号日后用区间查询补齐——
    # 否则最新发布日的T+1/T+2越界为None → 持仓sell_date为空,
    # 竞价确认卖出侧永不触发(2026-10-24修复)
    cal = client.trade_days(days=cfg["regime"].get("warmup_days", 30) + 45)
    signal_date = None
    day_df = None
    # 先探测最近两个交易日的行数, 识别"当日行情只发布了部分股票"的半成品状态。
    # 背景(2026-10-31): 主槽提前到 16:00(收盘后仅 1 小时), 数据源可能尚未发布完整全市场行情。
    # 原判据 len(df)>=3000 过松——全市场约 5400 只, 只出 4000 只也会通过, 从而把残缺行情当作
    # 完整收盘数据参与选股(静默错误, 比"没更新"更难发现)。故增加相对判据: 当日行数不足前一
    # 交易日的 95% 时视为未出全, 顺延到上一交易日, 由后续补数据槽在行情出全后重新产出。
    _recent = [(d, client.market_day(d)) for d in reversed(cal[-2:])]
    for _i, (_d, _df) in enumerate(_recent):
        _n = len(_df)
        if _n < 3000:
            continue
        if _i + 1 < len(_recent):
            _base_n = len(_recent[_i + 1][1])
            if _base_n >= 3000 and _n < _base_n * 0.95:
                print(f"⚠ {_fmt(_d)} 行情仅 {_n} 行(前一交易日 {_base_n} 行, {_n / _base_n:.1%}),"
                      f" 疑似当日未出全, 顺延到上一交易日")
                continue
        signal_date, day_df = _d, _df
        break
    if signal_date is None:  # 回落: 原逐日探测(应对连续多日缺失/半成品的极端情况)
        for d in reversed(cal[-8:]):
            df = client.market_day(d)
            if len(df) >= 3000:
                signal_date = d
                day_df = df
                break
    if signal_date is None:
        raise RuntimeError("近8个交易日均无完整行情数据, 终止")
    try:
        _fut = client.trade_calendar(_today, (_now + timedelta(days=20)).strftime("%Y%m%d"))
        cal = sorted(set(cal) | set(_fut))
    except Exception as _e:  # noqa: BLE001 未来日历失败退回旧行为(仅最新一期卖出日为空)
        print(f"⚠ 未来交易日历获取失败(最新持仓卖出日可能为空): {_e}")
    pos = cal.index(signal_date)
    print(f"信号日(最新完整交易日): {_fmt(signal_date)}")

    # ---- 1.2 龙虎榜当日榜单落盘(数据积累管道, 2026-10-24第三十八轮) ----
    # 目的: 为"T-5上榜基因"(第三十一轮观察级苗头)积累每日榜单样本; 放在幂等跳过闸
    # 之前, 15:30首跑若榜单未发布(空), 后续触发自动重试; 非空落盘后不重复请求。
    try:
        _lhb = client.lhb_day(signal_date)
        _n_lhb = 0 if _lhb is None else len(_lhb)
        print(f"龙虎榜榜单: {_fmt(signal_date)} {_n_lhb}条"
              + ("" if _n_lhb else "(未发布或为空, 下次运行重试)"))
    except Exception as _e:  # noqa: BLE001 榜单失败不影响选股主流程
        print(f"龙虎榜榜单获取失败(不影响主流程): {_e}")

    # ---- 1.2b 择时层数据管道(第四十轮, 2026-10-24): 两融账户统计+期指基差日增量 ----
    # 全量拉取+本地补缺(幂等), 兼作历史回填; 纯数据积累不改执行(第四十轮检验:
    # 两融4变量无增量预测力, 基差5日斜率为观察级苗头t=1.64, 均不接入执行层)。
    try:
        from .timing_data import sync_future, sync_margin
        print(f"两融账户统计: 补缺{sync_margin()}天; 期指基差: 补缺{sync_future()}天")
    except Exception as _e:  # noqa: BLE001 择时数据失败不影响选股主流程
        print(f"择时数据抓取失败(不影响主流程): {_e}")

    # ---- 1.2c PIT财务数据管道(第四十一轮, 2026-10-24): 业绩报表按报告期落盘 ----
    # 每期全市场快照含公告日期(PIT锚点); 已有报告期跳过, 新报告期披露后自动落盘;
    # 服务观察级苗头③(营收同比-10~0)的持续积累, 不改执行。
    try:
        from .finance_data import sync_finance
        print(f"PIT财务: 新增{sync_finance()}个报告期")
    except Exception as _e:  # noqa: BLE001 财务数据失败不影响选股主流程
        print(f"财务数据抓取失败(不影响主流程): {_e}")

    # ---- 1.4 同信号日幂等跳过: 该交易日收盘信号已处理过则不重复计算(防重复触发) ----
    # 判定口径就是"信号日": signal_date 本身已是"最新完整交易日", 只要 latest.json 就是这一天,
    # 说明该日已处理完毕, 再算一遍结果必然相同。因此不再附加 signal_date>=_today 的日期比较 ——
    # 那个条件会让"数据尚未出全的早槽"(如 16:00 主槽)反复重算上一交易日, 白烧 runner 且可能
    # 用旧数据刷新 output。数据没出全时这里直接跳过, 交给后续重试槽; 数据出全后自然放行。
    _latest_path = ROOT / "output" / "latest.json"
    _prev_sig = ""
    _prev_fp = ""
    if _latest_path.exists():
        try:
            _prev = json.loads(_latest_path.read_text(encoding="utf-8")) or {}
            _prev_sig = str(_prev.get("signal_date") or "")
            _prev_fp = str(_prev.get("config_fp") or "")
        except Exception:  # noqa: BLE001 latest.json损坏时视为未处理, 正常重算
            _prev_sig, _prev_fp = "", ""
    # 口径指纹: 选股/执行的关键参数一变, 同一信号日也必须重算 —— 否则改完策略后,
    # 幂等闸门会以"这天已处理过"为由继续展示旧口径的结果, 三端静默不一致。
    _fp = _config_fp(cfg)
    if _prev_sig == _fmt(signal_date) and _prev_fp == _fp:
        # 第一百四十八轮: 幂等跳过时仍刷新 update_time —— 数据内容不变不重算,
        # 但"更新于"时间戳应反映最近一次检查(北京时间), 否则多次手动触发后页面时间停在首跑时刻
        try:
            _prev["update_time"] = datetime.now(TZ_CN).strftime("%Y-%m-%d %H:%M:%S")
            _latest_path.write_text(json.dumps(_prev, ensure_ascii=False, indent=2), encoding="utf-8")
        except Exception:  # noqa: BLE001 时间戳刷新失败不影响幂等语义
            pass
        print(f"信号日{_fmt(signal_date)}已处理过(latest.json一致, 口径{_fp}), 幂等跳过(仅刷新更新时间戳)")
        return
    if _prev_sig == _fmt(signal_date):
        print(f"信号日{_fmt(signal_date)}已处理过但口径已变更({_prev_fp or '无'}→{_fp}), 按新口径重算")

    names_df = client.stock_names()
    names = dict(zip(names_df["ts_code"], names_df["name"]))

    # ---- 1.5 数据质量校验(异常检测, 防脏数据进信号) ----
    anomalies = []
    if len(day_df) < 3000:
        anomalies.append(f"全市场行数仅{len(day_df)}(<3000)")
    # 新股上市首日: pre_close=0, pct为脏值(如+7199%)属正常现象, 单独列示不计为异常
    ipo_rows = day_df[day_df["pre_close"] <= 0] if "pre_close" in day_df.columns else day_df.iloc[:0]
    if len(ipo_rows):
        _codes = ", ".join(ipo_rows["ts_code"].head(3))
        anomalies.append(f"新股上市首日{len(ipo_rows)}只({_codes}等, pct脏值已在指标层置NaN, 正常)")
    # 真实异常: 收盘价超出自身涨跌停价(按各板自身限制判断, 北交所±30%/科创板±20%不再误报;
    # 限价=0为无涨跌幅限制的次新股, 不参与判定)
    lim = day_df[(day_df["pre_close"] > 0) & (day_df["high_limit"] > 0) & (day_df["low_limit"] > 0)]
    out_lim = lim[(lim["close"] > lim["high_limit"] + 0.005) | (lim["close"] < lim["low_limit"] - 0.005)]
    # 退市/退市整理期豁免(2026-10-24): 面值退市区(close<1元)或名称含"退"的票, 退市整理期首日
    # 无涨跌幅限制而数据源限价字段仍按常规板别计算, 越界属正常退市流程波动, 不计为真实异常
    _delist = (out_lim["close"] < 1) | out_lim["ts_code"].map(lambda c: "退" in str(names.get(c, "")))
    out_real, out_delist = out_lim[~_delist], out_lim[_delist]
    if len(out_real):
        _codes = ", ".join(out_real["ts_code"].head(5))
        anomalies.append(f"真实异常: 收盘超出自身涨跌停价共{len(out_real)}条({_codes})")
    if len(out_delist):
        _codes = ", ".join(out_delist["ts_code"].head(3))
        anomalies.append(f"退市整理期波动{len(out_delist)}只({_codes}等, 无涨跌幅限制属正常)")
    # 疑似除权: pct与收盘/昨收隐含收益矛盾但收盘在限价内(除权日quote_rate对未复权昨收, 属正常)
    mism = lim[lim["close"] > 0]
    _implied = (mism["close"] / mism["pre_close"] - 1) * 100
    mism = mism[((mism["pct"] - _implied).abs() > 1.5)
                & (mism["close"] <= mism["high_limit"] + 0.005)
                & (mism["close"] >= mism["low_limit"] - 0.005)]
    if len(mism):
        _codes = ", ".join(mism["ts_code"].head(5))
        anomalies.append(f"疑似除权: pct与收盘昨收不一致共{len(mism)}条({_codes}等, 除权日属正常)")
    nan_close = int(day_df["close"].isna().sum())
    if nan_close:
        anomalies.append(f"收盘价缺失{nan_close}条")
    dup = int(day_df["ts_code"].duplicated().sum())
    if dup:
        anomalies.append(f"代码重复{dup}条")
    if anomalies:
        print("⚠ 数据质量异常: " + "; ".join(anomalies))
    else:
        print("数据质量校验通过")

    # ---- 2. 近期面板与指标 ----
    warm = max(cfg["regime"].get("warmup_days", 30), 25)
    panel_dates = cal[max(0, pos - warm + 1): pos + 1]
    panel = fetch_panel(client, panel_dates)
    if not len(panel):
        raise RuntimeError("面板抓取为空, 指标无法计算, 终止(请检查数据源)")
    got = set(panel["trade_date"].astype(str))
    miss = [d for d in panel_dates if str(d) not in got]
    if miss:
        print(f"⚠ 面板缺日 {len(miss)}/{len(panel_dates)} 天: {', '.join(miss[:5])}"
              + (" ..." if len(miss) > 5 else "")
              + " —— 指标与风控基于实际可得日期计算, 请核查数据源")
    panel = build_indicators(panel)
    by_date = split_by_date(panel)
    reg = regime_series(panel, cfg)
    names_df = client.stock_names()
    names = dict(zip(names_df["ts_code"], names_df["name"]))

    # ---- 3. 选股(多取缓冲用于次新核查) ----
    # 注意: 日常面板仅warmup天, 无法用bars判断上市天数, 此处关闭该硬过滤,
    #       改由第4步逐票拉取扩展历史核查(回测路径不受影响)。
    sel_cfg = cfg["selection"]
    cfg_sel = json.loads(json.dumps(cfg))
    cfg_sel["universe"]["listed_min_bars"] = 0
    # 发布池数量: 人工审核用, 默认取评分前 display_top_n 只(与回测的 top_n 资金部署口径解耦)
    disp_n = int(sel_cfg.get("display_top_n") or sel_cfg["top_n"])
    # 第八十七轮: 绩效门控池深(与 src/backtest.py::do_buy 完全同口径)——
    # 近 window 笔"已结算"成交均值 > min_avg 时扩到 hot_top_n, 否则收缩到 cold_top_n;
    # 已结算样本不足 window 笔时保持 exec_top_n(与回测冷启动一致)。仅用账本中已实现成交,
    # 无未来函数: 今日出池时点之前的成交才是已知的。
    _dg = (cfg.get("execution") or {}).get("depth_gate") or {}
    if _dg:
        _w = int(_dg.get("window", 20))
        _rets: list[float] = []
        _led_p = ROOT / "output" / "monitor_samples.json"
        if _led_p.exists():
            try:
                _d = json.loads(_led_p.read_text(encoding="utf-8"))
                _rets = [float(s["ret_pct"]) for s in (_d.get("exec") or [])
                         if s.get("ret_pct") is not None]
            except Exception:  # noqa: BLE001 账本损坏按冷启动处理
                _rets = []
        if len(_rets) >= _w:
            _avg = sum(_rets[-_w:]) / _w
            _hot = _avg > float(_dg.get("min_avg", 0.0))
            disp_n = int(_dg.get("hot_top_n" if _hot else "cold_top_n") or disp_n)
            print(f"  绩效门控: 近{_w}笔均值{_avg:+.2f}% "
                  f"{'≥' if _hot else '<'} {float(_dg.get('min_avg', 0.0)):+.2f}% "
                  f"→ 发布池{'扩至' if _hot else '收缩至'}{disp_n}只")
        else:
            disp_n = int(sel_cfg.get("exec_top_n") or disp_n)
            print(f"  绩效门控: 已结算{len(_rets)}笔<{_w} → 冷启动按{disp_n}只")
    ranked, stats = select_stocks(signal_date, by_date[signal_date], cfg_sel,
                                  names, top_n=sel_cfg["top_n"] + 8)
    # ---- 3b. 多策略合并池(第七十九轮) ----
    # 主力池 + 备用池(fb1/fb2...)合成一个池, 每只候选带 strategy/strategy_cn 标签。
    # 展示配额: 主力 disp_n 只(次新剔除时向后顺延), 各备用池 top_n 只;
    # 顺序=主力在前、备用在后, 与 src/backtest.py::do_buy 的买入顺延顺序完全一致。
    _fbs = sel_cfg.get("fallbacks") or []
    _main_all = [p for p in ranked if p.get("pool", "main") == "main"]
    _fb_all = [p for p in ranked if p.get("pool", "main") != "main"] if _fbs else []
    ranked = _main_all + _fb_all
    quota = {"main": disp_n}
    for i, fb in enumerate(_fbs, 1):
        if fb.get("enabled", True):
            quota[f"fb{i}"] = int(fb.get("top_n") or 0)
    disp_cap = sum(quota.values())
    # ---- 4. 次新股/上市天数核查 ----
    start_far = (pd.Timestamp(signal_date) - pd.Timedelta(days=540)).strftime("%Y%m%d")
    verified, _pool_cnt = [], {}
    for p in ranked:
        if len(verified) >= disp_cap:
            break
        _pool = p.get("pool", "main")
        if _pool_cnt.get(_pool, 0) >= quota.get(_pool, 0):
            continue
        hist = client.stock_history(p["code"], start_far, signal_date)
        bars = len(hist)
        if bars < cfg["universe"]["listed_min_bars"]:
            print(f"  剔除次新/数据不足: {p['code']} {p['name']} ({bars}根K线)")
            continue
        verified.append(p)
        _pool_cnt[_pool] = _pool_cnt.get(_pool, 0) + 1
    # 第九十九轮: 把"当日主力池候选数"挂到每只票上 —— 池深分档下注的唯一输入,
    # 与回测 do_buy 的 _pool_n(= select 的 scored) 完全同源, 盘后即知。
    _pn_main = int(stats.get("scored") or 0)
    for _p in verified:
        _p["pool_n"] = _pn_main if _p.get("pool", "main") == "main" else 0

    # ---- 5. 结算 T-2 发布池 ----
    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    hist_path = out_dir / "history.json"
    if hist_path.exists():
        _data = json.loads(hist_path.read_text(encoding="utf-8"))
        history = _data.get("records", []) if isinstance(_data, dict) else _data
    else:
        history = []
    # 第一百零二轮一次性口径修复: 风控日"幻影成交"。
    # 旧代码在 max_block_days=0 时把每个风控日算成"全额参与日", 于是账本给风控期的计划
    # 记了成交(2026-10-22/10-28/10-29 三笔), 而当日决策卡写的是"空仓观望" —— 实际未执行。
    # 这里只回填 2026-11-01 之前、且 regime_ok=False 的已成交记录; 新口径下风控日若真按
    # 深低开成交(blocked_frac>0), 记录里会带 pos_note, 不会被误清。
    for _r in history:
        if (_r.get("regime_ok") is False and _r.get("executed")
                and str(_r.get("buy_date", "")).replace("-", "") < "20261101"):
            _r["status"] = "无成交"
            _r.pop("executed", None)
            _r.pop("realized", None)
            _r["skip_chain"] = ["大盘风控空仓(第一百零二轮口径修复: 旧版风控日幻影成交)"]
    ex = cfg["execution"]

    # 买入日风控状态(与回测口径一致): 超限参与日用专属承接线与参与系数(含渐近恢复)
    _rm = cfg["regime"]
    max_block = int(_rm.get("max_block_days", 0) or 0)
    reenter = float(_rm.get("block_reenter_frac", 0) or 0)
    rf_ramp = bool(_rm.get("rf_ramp", False))
    day_frac_map: dict[str, float] = {}
    day_block_map: dict[str, bool] = {}   # 该日是否处于风控期(trade_ok=False), 供承接线口径判定
    _consec = 0
    for _d in reg.index:
        if bool(reg.loc[_d, "trade_ok"]):
            _consec = 0
            day_frac_map[_d] = 1.0
        else:
            day_block_map[_d] = True
            _consec += 1
            if max_block <= 0:
                # 第一百零二轮修复: max_block_days=0 的语义是"不设上限、风控期一直空仓"。
                # 原实现直接落进 rf_ramp 分支 → min(1, 0.95+0.05*consec)=1.0, 把每个风控日
                # 都当成"全额参与日"(还带着严承接线), 于是账本在风控期凭空记出成交,
                # 与回测(blocked_frac=0 → 不买)和页面"空仓观望"自相矛盾。
                # 现在: 风控日改用风控参与系数 blocked_frac(0=空仓), 承接线取 block_gap_below。
                day_frac_map[_d] = float(ex.get("blocked_frac", 0.0) or 0.0)
            elif _consec <= max_block:
                day_frac_map[_d] = 0.0
            elif rf_ramp:
                day_frac_map[_d] = min(1.0, reenter + 0.05 * (_consec - max_block))
            else:
                day_frac_map[_d] = reenter

    def row_of(d: str, code: str):
        # 日期归一化: 调用方可能传入带横杠(history)或无横杠(cal)格式
        sub = by_date.get(str(d).replace("-", ""))
        if sub is None:
            return None
        r = sub[sub["ts_code"] == code]
        return r.iloc[0] if len(r) else None

    def _settle_holding(rec, buy_d, sell_d, signal_date, cal, day_frac_map, row_of, ex, cfg):
        """买入日已发生但卖出日未到: 按当日真实竞价判定成交, 返回持仓中明细(无卖出价)。"""
        bfrac = day_frac_map.get(buy_d, 1.0) if buy_d else 0.0
        day_ok = bfrac > 0
        floor = ex.get("buy_gap_below", 0)
        if day_ok and day_block_map.get(buy_d, False):
            # 超限小仓参与日(风控期但允许小仓): 用专属承接线, 与回测口径一致
            floor = ex.get("block_gap_below", floor)
        # 注意(第一百二十五轮): 实盘结算端只支持"单吊"—— _settle_holding 命中第一只可成交票即
        # return, 配 max_buy_per_day > 1 不会生效(与回测的多只口径不同)。当前 config 为 1,
        # 行为正确; 此处原先的无用赋值 maxn 已删, 避免让人误以为改配置就能多买。
        chain = []
        for rank_i, p in enumerate(rec.get("picks", []), 1):
            rb = row_of(buy_d, p["code"]) if buy_d else None
            rp = row_of(rec["publish_date"], p["code"])
            gap = (rb["open_adj"] / rp["close_adj"] - 1) * 100 \
                if (rb is not None and rp is not None and rp["close_adj"] > 0) else 999
            reason = None
            if not day_ok:
                reason = "大盘风控空仓"
            elif rb is None or rb["is_paused"] == 1 or rb["vol"] <= 0:
                reason = "停牌/无成交"
            elif rb["open"] >= rb["high_limit"] - 1e-9:
                reason = "开盘一字涨停"
            elif gap >= ex["skip_open_gain"]:
                reason = f"高开{gap:.1f}%放弃"
            elif ex.get("skip_open_loss", 0) and gap <= -abs(ex["skip_open_loss"]):
                reason = f"低开{gap:.1f}%过深"
            elif floor and gap > floor:
                reason = f"缺口{gap:.1f}%未达承接线{floor}%"
            if reason:
                chain.append(f"第{rank_i}名 {p.get('name', p['code'])}: {reason}")
                continue
            # 第五十三轮: 买入日仓位含月度盈利熔断系数(仅已结算口径, 本记录浮盈不属买入时点)
            _, _cf, _ = _month_gain_state([], history, (buy_d or "")[:6], None, cfg, include_hold=False)
            # 第九十七轮: 仓位口径补齐 —— 评分档 × 高开降仓档(与回测 do_buy 同尺度),
            # 此前只记 bfrac*熔断系数, 页面显示的执行仓位与回测实际仓位不同。
            _sm = _pos_tier_mult(p.get("score"), ex)
            _gm, _gt = _gap_tier_mult(gap, ex, p.get("score"))
            _pm, _pt = _pool_tier_mult(p.get("pool_n"), ex)
            _fl, _ft = _gap_floor_mult(gap, p.get("pool_n"), ex)   # 第一百零一轮: 深低开下限
            _mkt = _market_median_pct(by_date, rec["publish_date"])
            _km, _kt, _mf, _mft = _mkt_context(p.get("pct"), _mkt, ex)   # 第一百零三轮: 市场语境
            _bm, _bt = _boost_mult(p.get("pool_n"), _mkt, ex)            # 第一百七十五轮: 加仓档
            _base = bfrac * _cf
            _pos = round(max(_base * _sm * _gm * _pm * _km * _bm, _base * _fl, _base * _mf)
                         * ex.get("alloc_frac_pct", 100.0), 1)
            if _fl and _pm < _fl:
                _pt = (f"{_pt}·{_ft}" if _pt else _ft)
            return {
                "rank": rank_i, "code": p["code"], "name": p.get("name", p["code"]),
                "buy_date": _fmt(buy_d), "buy_px": round(float(rb["open"]), 2),
                "sell_date": _fmt(sell_d) if sell_d else "", "sell_px": None,
                "ret_pct": None,
                "position_pct": _pos,
                "pos_note": ("·".join(x for x in [
                    (f"评分{float(p['score']):.2f}→{_sm:.0%}" if (p.get("score") is not None and _sm != 1.0) else ""),
                    _gt, _pt, _kt, _mft, _bt] if x) or None),
                "gap_pct": round(gap, 2),
                "pool_n": p.get("pool_n"),
                "chain": chain,
                "holding": True,
            }
        rec["skip_chain"] = chain   # 总是覆盖: 顺延链为空即清空, 防历史运行残留与executed矛盾
        return None

    new_exec, new_blocked = [], []   # 第49轮: 本轮结算新产生的监测样本/空仓对照
    for rec in history:
        if rec.get("status") == "已结算":
            continue
        # 第九十二轮: 账本不因口径变更改写 —— 已终结为"无成交"的记录若来自旧口径
        # (config_fp 不同或旧版无该字段), 保持原判, 不再用当前执行规则重判。
        # 否则每次调参都会把历史"缺口不达标空仓"翻成"已成交", 与页面留档说明矛盾。
        # 仅未终结记录(待结算/持仓中)继续按当前口径推进。
        if rec.get("status") == "无成交" and str(rec.get("config_fp") or "") != _fp:
            continue
        pdate = rec["publish_date"]
        # 日期格式归一化: history 的 publish_date 带横杠, cal/by_date/reg 为无横杠
        # (此前格式不一致导致结算循环静默跳过所有记录 —— 2026-10-23 发现并修复)
        pdate_n = str(pdate).replace("-", "")
        if pdate_n not in cal:
            continue
        ip = cal.index(pdate_n)
        buy_d = cal[ip + 1] if ip + 1 < len(cal) else None
        sell_d = cal[ip + 2] if ip + 2 < len(cal) else None
        if not sell_d or sell_d > signal_date:
            # 卖出日未到: 若买入日数据已产生, 记录"持仓中"(买入事实当日可见)
            if buy_d and buy_d <= signal_date and rec.get("picks"):
                _holding = _settle_holding(rec, buy_d, sell_d, signal_date, cal, day_frac_map, row_of, ex, cfg)
                if _holding:
                    rec["executed"] = _holding
                    rec["status"] = "持仓中"
                    rec["skipped_codes"] = []
                    rec.pop("skip_chain", None)   # 清除历史运行残留的旧顺延链(与executed矛盾)
                else:
                    rec.pop("executed", None)     # 判定未成交: 清除可能的旧持仓记录
            continue
        # 买入日仓位系数与承接线(与回测口径一致)
        bfrac = day_frac_map.get(buy_d, 1.0) if buy_d else 0.0
        day_ok = bfrac > 0
        floor = ex.get("buy_gap_below", 0)
        if day_ok and day_block_map.get(buy_d, False):
            # 超限小仓参与日(风控期但允许小仓): 用专属承接线, 与回测口径一致
            floor = ex.get("block_gap_below", floor)
        rets, skipped, chain = [], [], []
        last_sell_dd = sell_d   # 第五十四轮: 实际卖出日(停牌顺延/收盘卖时≠T+2)
        vrets, vpick = [], None   # 第49轮: 风控空仓日虚拟对照(除风控闸外同口径)
        executed = None
        maxn = int(ex.get("max_buy_per_day", 0))
        gap_fee = (1 - ex["slippage"]) * (1 - ex["commission_rate"] - ex["stamp_tax"]) \
                  / ((1 + ex["slippage"]) * (1 + ex["commission_rate"]))
        for rank_i, p in enumerate(rec.get("picks", []), 1):
            if maxn > 0 and len(rets) >= maxn:
                break  # 与回测口径一致: 竞价过滤后仅按评分买入前 maxn 只
            rb = row_of(buy_d, p["code"]) if buy_d else None
            rs = row_of(sell_d, p["code"])
            rp = row_of(pdate, p["code"])
            gap = (rb["open_adj"] / rp["close_adj"] - 1) * 100 \
                if (rb is not None and rp is not None and rp["close_adj"] > 0) else 999
            if not day_ok:
                reason = "大盘风控空仓"
                # 第49轮: 空仓日虚拟结算 —— 若当日完全放行(承接线等其余执行闸不变),
                # top1虚拟收益逐日入账, 前瞻验证风控机会成本(第45轮回顾: era后贡献+1183pp),
                # 只记录不参与决策。
                if maxn <= 0 or len(vrets) < maxn:
                    if _exec_gate(rb, gap, ex, floor) is None \
                            and rs is not None and rs["is_paused"] != 1 and rs["vol"] > 0:
                        vrets.append(float((rs["open_adj"] / rb["open_adj"]) * gap_fee - 1))
                        vpick = vpick or p
            else:
                reason = _exec_gate(rb, gap, ex, floor)
            if reason:
                chain.append(f"第{rank_i}名 {p.get('name', p['code'])}: {reason}")
                skipped.append(p["code"])
                continue
            # 第五十四轮: 卖出端对齐回测 do_sell(停牌顺延5日+开盘跌停保守收盘卖),
            # 原实现sell_d固定T+2且无条件按开盘价 —— 跌停日按开盘高估收益,
            # 停牌日真实成交回报永久丢失(记录卡"无成交")。顺延后每日重跑自然续结。
            sx = _sell_exit(rs, row_of, cal, sell_d, p["code"], ex,
                            buy_adj=(float(rb["open_adj"]) if rb is not None else None))
            if sx is None:
                continue  # 顺延窗口内未恢复: 留待后续结算
            sell_row, sell_dd, sold_close, sell_note, _sfo = sx
            # 第九十八轮: 分批出场(与回测 do_sell 同口径) —— 开盘腿按开盘价、收盘腿按收盘价,
            # 加权得卖出净价; 开盘跌停/延卖日 _sfo=0(全按收盘)。
            _so, _sc = float(sell_row["open_adj"]), float(sell_row["close_adj"])
            _spx = _sc if sold_close else (_sfo * _so + (1 - _sfo) * _sc)
            r = (_spx / rb["open_adj"]) * gap_fee - 1
            rets.append(float(r))
            last_sell_dd = sell_dd
            # 第五十三轮: 结算仓位含月度盈利熔断系数(买入日口径, 仅已结算复利)
            _, _cf, _ = _month_gain_state([], history, (buy_d or "")[:6], None, cfg, include_hold=False)
            # 第九十七轮: 同 _settle_holding —— 补上评分档与高开降仓档, 执行记录与回测同尺度
            _sm = _pos_tier_mult(p.get("score"), ex)
            _gm, _gt = _gap_tier_mult(gap, ex, p.get("score"))
            _pm, _pt = _pool_tier_mult(p.get("pool_n"), ex)
            _fl, _ft = _gap_floor_mult(gap, p.get("pool_n"), ex)   # 第一百零一轮
            _mkt = _market_median_pct(by_date, rec["publish_date"])
            _km, _kt, _mf, _mft = _mkt_context(p.get("pct"), _mkt, ex)   # 第一百零三轮
            _bm, _bt = _boost_mult(p.get("pool_n"), _mkt, ex)            # 第一百七十五轮
            _base = bfrac * _cf
            _pos_final = round(max(_base * _sm * _gm * _pm * _km * _bm, _base * _fl, _base * _mf)
                               * ex.get("alloc_frac_pct", 100.0), 1)
            if _fl and _pm < _fl:
                _pt = (f"{_pt}·{_ft}" if _pt else _ft)
            executed = {
                "rank": rank_i, "code": p["code"], "name": p.get("name", p["code"]),
                "buy_date": _fmt(buy_d), "buy_px": round(float(rb["open"]), 2),
                "sell_date": _fmt(sell_dd),
                "sell_px": round(_spx / (float(sell_row["close_adj"]) / float(sell_row["close"]))
                                 if float(sell_row["close"]) else _spx, 2),
                "ret_pct": round(float(r) * 100, 2),
                "position_pct": _pos_final,
                "pos_note": ("·".join(x for x in [
                    (f"评分{float(p['score']):.2f}→{_sm:.0%}" if (p.get("score") is not None and _sm != 1.0) else ""),
                    _gt, _pt, _kt, _mft, _bt] if x) or None),
                "gap_pct": round(gap, 2),
                "pool_n": p.get("pool_n"),
                "chain": chain,
            }
            if sell_note:
                executed["sell_note"] = sell_note
        if rets:
            rec["realized"] = {
                "buy_date": _fmt(buy_d), "sell_date": _fmt(last_sell_dd),
                "n_traded": len(rets), "n_skipped": len(skipped),
                "avg_ret_pct": round(sum(rets) / len(rets) * 100, 2),
                "win": sum(1 for r in rets if r > 0),
            }
            rec["executed"] = executed
            rec.pop("skip_chain", None)   # 顺延链已在executed.chain内, 清除顶层残留
        elif executed is None and (chain or not day_ok):
            rec["skip_chain"] = chain if day_ok else ["大盘风控空仓"]
        else:
            rec.pop("skip_chain", None)
        rec["status"] = "已结算" if rets else "无成交"
        rec["skipped_codes"] = skipped
        # 第49轮: 样本入账 —— exec与回测trades_executed_recent同口径(实际执行,
        # 含超限小仓参与日); blocked为空仓日虚拟对照, 只记录不参与决策
        if rets and executed:
            new_exec.append({"publish_date": rec["publish_date"], "sell_date": _fmt(last_sell_dd),
                             "code": executed["code"], "name": executed["name"],
                             "ret_pct": round(sum(rets) / len(rets) * 100, 2)})
        elif not day_ok and vrets:
            new_blocked.append({"publish_date": rec["publish_date"], "sell_date": _fmt(sell_d),
                                "code": (vpick or {}).get("code", ""),
                                "name": (vpick or {}).get("name", ""),
                                "ret_pct": round(sum(vrets) / len(vrets) * 100, 2)})

    # ---- 5.45 持久账本落盘(第49轮): 监测样本append-only + 风控空仓对照 ----
    # 此前监测样本直接取自history.json, 而history仅保留最近60条展示记录 —— 执行日
    # 占比低时(风控连续空仓/候选稀疏), 滚动窗口内可能永远攒不满window笔, 实盘监测灯
    # 形同虚设。账本按publish_date去重追加(同信号日重算幂等覆盖), 不随history裁剪,
    # 为监测器提供全量样本源。
    led_path = out_dir / "monitor_samples.json"
    led = {"exec": [], "blocked": []}
    if led_path.exists():
        try:
            _led = json.loads(led_path.read_text(encoding="utf-8"))
            led = {"exec": _led.get("exec") or [], "blocked": _led.get("blocked") or []}
        except Exception:  # noqa: BLE001 账本损坏时重建
            pass
    for _key, _items in (("exec", new_exec), ("blocked", new_blocked)):
        _idx = {s.get("publish_date"): s for s in led[_key]}
        for it in _items:
            _idx[it["publish_date"]] = it
        led[_key] = sorted(_idx.values(), key=lambda s: s.get("sell_date", ""))
    led["updated"] = _fmt(signal_date)
    led_path.write_text(json.dumps(led, ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 5.5 滚动失效监测(纯告警层, 2026-10-24第三十九轮) ----
    # 数据源(第49轮起): 持久账本 monitor_samples.json 的exec样本(全量, 不随history
    # 展示裁剪丢失, 与回测trades_executed_recent同口径); 账本为空时回退history口径。
    # 阈值事前注册于 config.monitor, 不做任何拟合。
    mon_cfg = cfg.get("monitor") or {}
    monitor_info = None
    observe_info = None
    if mon_cfg.get("enabled"):
        _settled_rets = [float(s["ret_pct"]) for s in led["exec"]]
        if not _settled_rets:
            _settled_rets = [float(r["realized"]["avg_ret_pct"]) for r in history
                             if r.get("status") == "已结算" and r.get("realized")
                             and r.get("regime_ok", True) is not False]
        monitor_info = compute_monitor(_settled_rets, mon_cfg)
        print(f"失效监测: {monitor_info['level']} | {monitor_info['note']}")
        if str(monitor_info["level"]).startswith("红") or str(monitor_info["level"]).startswith("深红"):
            print("⚠⚠ 失效监测告警 —— 请人工复检后再决定是否继续按既有纪律执行")
        # 空仓对照前瞻指标(第49轮): 验证风控过滤器的机会成本方向, 仅记录不参与决策
        if led["blocked"]:
            _b = [float(s["ret_pct"]) for s in led["blocked"]]
            observe_info = {"n": len(_b), "avg_pct": round(sum(_b) / len(_b), 2),
                            "win_pct": round(100 * sum(r > 0 for r in _b) / len(_b), 1),
                            "note": ("风控空仓日对照: 若当日完全放行(其余执行闸不变)的虚拟top1"
                                     "收益, 前瞻验证风控机会成本(第45轮回顾: era后贡献+1183pp), "
                                     "不参与决策")}
            print(f"空仓对照: n={observe_info['n']} 均值{observe_info['avg_pct']:+.2f}%/笔 "
                  f"胜率{observe_info['win_pct']:.0f}%")

    # ---- 6. 写出候选池与历史(空仓时也展示观察池, 并明确标注风控状态) ----
    reg_row = reg.loc[signal_date] if signal_date in reg.index else None
    # 最大连续空仓上限: 统计截至信号日的连续风控天数, 超限后按小仓位参与
    _rm = cfg["regime"]
    max_block, reenter = int(_rm.get("max_block_days", 0) or 0), float(_rm.get("block_reenter_frac", 0) or 0)
    consec = 0
    if reg_row is not None and not bool(reg_row["trade_ok"]):
        for d_i in reversed(list(reg.index[:reg.index.get_loc(signal_date) + 1])):
            if bool(reg.loc[d_i, "trade_ok"]):
                break
            consec += 1
    over_limit = max_block > 0 and reenter > 0 and consec > max_block
    # 第一百零二轮: 风控日参与系数(0=严格空仓) —— 只在"缺口 ≤ block_gap_below"时执行
    _risk_frac = max(0.0, min(1.0, float(ex.get("blocked_frac", 0.0) or 0.0)))
    # 第五十二轮口径修正: 回测rf_ramp公式为min(1, reenter+0.05*(consec-max_block)),
    # 首个超限日(consec=max_block+1)即 0.95+0.05=1.0 封顶 —— "95%仓位"从未实际生效,
    # 回测中超限参与日实为"全额参与+严承接线"。此处按买入日(明日, consec+1)对齐同款公式。
    over_frac = min(1.0, reenter + 0.05 * (consec + 1 - max_block)) if over_limit else 1.0
    # 第九十二轮: 无缺口门槛时, 文案统一为"开盘价直接成交", 不再输出承接线百分比
    _bg = cfg["execution"].get("block_gap_below", cfg["execution"].get("buy_gap_below"))
    _bg_txt = f"(执行承接线{_bg}%)" if _bg else "(不设缺口门槛, 按开盘价买入)"
    regime_info = {
        "ok": bool(reg_row["trade_ok"]) if reg_row is not None else True,
        "breadth": round(float(reg_row["breadth"]), 3) if reg_row is not None else None,
        "mkt_above_ma": bool(reg_row["mkt_idx"] >= reg_row["mkt_ma"]) if reg_row is not None else None,
        "block_days": consec,
        "max_block_days": max_block,
        "reenter_frac": over_frac if over_limit else None,
        "note": "宽度/市场指数达标, 可正常开仓" if (reg_row is not None and reg_row["trade_ok"])
                else (f"大盘风控连续第{consec}日, 超过{max_block}日上限: 次日按{over_frac:.0%}仓位参与候选池"
                      f"{_bg_txt}"
                      if over_limit else
                      ((f"大盘风控触发(连续第{consec}日): 次日只在竞价缺口 ≤{float(ex.get('block_gap_below') or 0):g}% "
                        f"时按{_risk_frac:.0%}仓位接恐慌深跌, 否则空仓")
                       if (_risk_frac > 0 and ex.get("block_gap_below")) else
                       (f"大盘风控触发(连续第{consec}日): 次日建议空仓观望" if reg_row is not None else ""))),
    }
    buy_d = cal[pos + 1] if pos + 1 < len(cal) else None
    sell_d = cal[pos + 2] if pos + 2 < len(cal) else None
    if buy_d is None:   # 信号日=日历最后一天: 推算下一交易日(跳过周末; 法定节假日顺延, 误差可接受)
        _bd = pd.bdate_range(start=_fmt(signal_date), periods=3)
        buy_d = _bd[1].strftime("%Y%m%d")
        sell_d = _bd[2].strftime("%Y%m%d")
        print(f"交易日历无未来日, 推算买入日{buy_d}/卖出日{sell_d}(遇节假日顺延)")

    # ---- T+1 执行判定卡(每晚生成, 次日照卡执行) ----
    # 月度盈利熔断状态(第五十三轮实盘对齐): 回测买入端有"当月净值增益超cap后仓位×cut",
    # 指引此前缺失 —— 回测收益含降仓行为, 网页指引恒全仓, 实盘照做将与回测口径分叉
    close_of = day_df.set_index("ts_code")["close"].to_dict() if day_df is not None and len(day_df) else {}
    month_gain_pct, cap_frac, cap_note = _month_gain_state(
        led["exec"], history, signal_date[:6], close_of, cfg)
    regime_ok_now = bool(reg_row["trade_ok"]) if reg_row is not None else True
    over_limit_now = over_limit
    panic_day = False
    if reg_row is not None and cfg["regime"].get("vol_max"):
        _vol20 = float(reg["mkt_pct"].rolling(20).std().loc[signal_date]) if signal_date in reg.index else 0.0
        _m20 = float((reg["mkt_idx"].loc[signal_date] / reg["mkt_idx"].shift(20).loc[signal_date] - 1) * 100) \
            if signal_date in reg.index else 0.0
        panic_day = (_vol20 > cfg["regime"]["vol_max"]) and (_m20 < cfg["regime"].get("vol_mkt_ret_min", 0.0))
    plan_frac = (over_frac if (over_limit_now and not regime_ok_now)
                 else (_risk_frac if not regime_ok_now else 1.0)) * cap_frac
    floor_now = ex.get("buy_gap_below", 0)
    if not regime_ok_now:
        # 第九十二轮: 承接线判定提前到文案之前, 便于按"有无门槛"生成不同指引
        floor_now = ex.get("block_gap_below", floor_now)
    _fill_txt = f"开盘缺口≤{floor_now:g}%才买" if floor_now else "开盘价直接买入(不设缺口门槛)"
    if panic_day:
        allow_buy = False
        buy_note = "恐慌型波动日(高波动+市场20日下跌): 全额空仓, 不参与任何买入"
    elif regime_ok_now:
        allow_buy = True
        buy_note = f"风控通过: 可正常开仓({_fill_txt})"
    elif over_limit_now:
        allow_buy = True
        buy_note = f"超限参与日(连续风控第{consec}日): 按{over_frac:.0%}仓位参与, {_fill_txt}"
    elif _risk_frac > 0 and floor_now:
        # 第一百零二轮: 风控日只接"恐慌深跌" —— 缺口是错杀的直接证据, 宽度/均线只是间接代理
        allow_buy = True
        buy_note = (f"风控中(连续第{consec}日): 只接恐慌深跌 —— {_fill_txt}, "
                    f"仓位按{_risk_frac:.0%}执行; 缺口不够深则当日空仓")
    else:
        allow_buy = False
        buy_note = f"大盘风控中(连续第{consec}日): 空仓观望"
    # 第九十五轮: 质量门槛 —— 首选评分(横截面 z 分)不足则当日不参与, 与回测 min_top1_score 同口径。
    # 独立于风控判定: 即使当日已因风控空仓, 也在文案里点出评分状态, 避免"风控放开才发现没票"。
    _gate = ex.get("min_top1_score")
    _top_score = verified[0].get("score") if verified else None
    _gate_blocked = bool(_gate is not None and verified and
                         (_top_score is None or float(_top_score) < float(_gate)))
    # 池深 1 只时横截面 z 分的标准差为 0, selector._zs 直接返回 0 —— 这是"没有可比对象",
    # 不是"这只票质量差"。原先文案写"当日质量不足", 会让人以为选股出了问题, 故分岔说明。
    if _top_score is not None and _gate is not None:
        if _pn_main <= 1:
            _gate_txt = (f"当日主力池仅 {_pn_main} 只达标, 横截面 z 分无参照"
                         f"(标准差为 0, 按 0 计) < 门槛 {float(_gate):g}")
        else:
            _gate_txt = f"首选评分 {float(_top_score):.2f} < 门槛 {float(_gate):g}(横截面 z 分)"
    else:
        _gate_txt = "首选评分缺失"
    if _gate_blocked and allow_buy:
        allow_buy = False
        buy_note = (f"{_gate_txt}: 当日无横截面可比, 空仓不参与(与回测 min_top1_score 同口径)"
                    if _pn_main <= 1 else
                    f"{_gate_txt}: 当日质量不足, 空仓不参与(与回测 min_top1_score 同口径)")
    elif _gate_blocked:
        buy_note += f"; 另: {_gate_txt}, 即使风控解除也不参与"
    # 第九十六轮: 评分分档下注 —— 门槛降到 1.0 后, 1.0~1.6 的票不再一刀切空仓, 而是按档位减仓
    _tiers = ex.get("score_tiers") or []
    _tier_mult, _tier_txt = 1.0, ""
    if _tiers and _top_score is not None and not _gate_blocked:
        for _t in _tiers:
            if float(_top_score) >= float(_t["min"]):
                _tier_mult = float(_t.get("mult", 1.0))
                _tier_txt = f"评分{float(_top_score):.2f}→{_tier_mult:.0%}仓({float(_t['min']):g}档)"
                break
        if _tier_mult < 1.0 and allow_buy:
            buy_note += f"; {_tier_txt}"
    if allow_buy and cap_frac < 1.0:
        buy_note += f"; {cap_note}"
    # 第九十七轮: 高开降仓 —— 缺口数据 T+1 开盘才有, 盘前只给规则, 9:25 竞价确认时按实际缺口执行
    _gat = ex.get("gap_above_tiers") or []
    _gat_txt = "/".join(f"≥{float(t['above']):g}%→{float(t.get('mult', 1.0)):.0%}" for t in _gat)
    _gat_rule = (f"; 若竞价缺口 {_gat_txt} 则按该比例减仓(照常成交, 不弃单)" if _gat else "")
    if _gat and allow_buy:
        buy_note += _gat_rule
    # 第九十九轮: 池深分档 —— 与"评分档""高开档"同尺度叠乘, 只降仓不弃单。
    _pn_today = int(stats.get("scored") or 0)
    _pool_mult, _pool_txt = _pool_tier_mult((_pn_today if verified and verified[0].get("pool", "main") == "main" else None), ex)
    if _pool_mult < 1.0 and allow_buy:
        buy_note += f"; {_pool_txt}(当日候选池偏窄=孤立回调、缺共振)"
    # 第一百零一轮: 深低开撤销池深折价 —— 缺口 T+1 开盘才知道, 盘前只给规则
    _gft = ex.get("gap_floor_tiers") or []
    if _gft and allow_buy and _pool_mult < 1.0:
        _t0 = _gft[0]
        buy_note += (f"; 若竞价缺口 ≤{float(_t0['below']):g}% 且达标≥{int(_t0.get('min_pool', 0))}只"
                     f" → 撤销池深折价、回{float(_t0.get('floor', 1.0)):.0%}仓")
    _pos_frac = plan_frac * _tier_mult * _pool_mult
    # 第一百零三轮: 市场语境 —— 把"深低开/回调"拆成市场成分与个股成分
    #   (a) 刀口保护: 大盘平稳/上涨(全市场涨跌中位数 > mkt_above)而首选超额下跌
    #       (个股 pct - 市场中位数) <= res_below → 个股自己的事, 不是系统性错杀, 缩仓;
    #   (b) 恐慌日回满仓: 全市场跌到 mkt_below 以下 → 系统性错杀, 次日修复有全市场共振,
    #       撤销评分/池深折价回满仓(仍以 plan_frac 为上限, 不越过风控降仓)。
    # 两个输入都是 T 日收盘后的横截面统计, 盘后即知, 无未来函数。
    _mkt_ret = _market_median_pct(by_date, signal_date)
    _knife_mult, _knife_txt, _mf_frac, _mkt_floor_txt = _mkt_context(
        (verified[0].get("pct") if verified else None), _mkt_ret, ex)
    if allow_buy:
        _pos_frac *= _knife_mult
        if _mf_frac and _pos_frac < plan_frac * _mf_frac:
            _pos_frac = plan_frac * _mf_frac
    # 第一百一十四轮: 刀口乘数 0 = 剔除(与回测 knife_tiers mult=0 同口径) ——
    # 个股独立下跌属"自己的事", 零期望票占用单吊唯一的名额是负贡献, 当日直接不参与。
    if allow_buy and _knife_mult <= 0:
        allow_buy = False
        buy_note = (f"{_knife_txt}: 个股独立下跌(非系统性错杀), 当日不参与"
                    f"(与回测 knife_tiers mult=0 同口径)")
    elif allow_buy and _knife_txt:
        buy_note += f"; {_knife_txt}"
    if allow_buy and _mkt_floor_txt:
        buy_note += f"; {_mkt_floor_txt}"
    steps = []
    _rk: dict[str, int] = {}   # 池内名次(主力/备用各自计数, 与回测 _rank 一致)
    # 顺延指引覆盖整个发布池(第六十一轮: 5→display_top_n只): 网页展示N只=指引N只=回测exec_top_n,
    # 三端一致——回测在发布池内顺延, 实盘按指引在发布池内顺延, 口径不再分叉
    for i, p in enumerate(verified, 1):
        # 备用池票有各自承接线(与回测 _gap_below 一致), 指引逐只按来源策略标注
        _strategy = p.get("strategy", "dip")
        _pool = p.get("pool", "main")
        _rk[_pool] = _rk.get(_pool, 0) + 1
        _floor = floor_now
        if _pool != "main":
            i_fb = int(str(_pool)[2:]) - 1
            fb_cfg = _fbs[i_fb] if 0 <= i_fb < len(_fbs) else {}
            if fb_cfg.get("gap_below") is not None:
                _floor = fb_cfg["gap_below"]
        # 第八十七轮: 名次梯度承接线(与 src/backtest.py 同口径) —— 池内名次越靠后,
        # 要求缺口越深; 取所有满足条件的档中最深者(更严)。
        for _t in (ex.get("gap_rank_tiers") or []):
            if _rk[_pool] >= int(_t.get("rank_above", 1)):
                _tg = float(_t.get("gap_below", _floor))
                _floor = min(_tg, float(_floor)) if _floor else _tg
        steps.append({
            "rank": i, "code": p["code"], "name": p["name"], "score": p.get("score"),
            "close": p.get("close"), "pct": p.get("pct"),
            "pool": p.get("pool", "main"), "strategy": _strategy,
            "strategy_cn": p.get("strategy_cn", _strategy),
            "pool_n": p.get("pool_n"),
            "rule": ((f"评分 {float(_top_score):.2f} < 门槛 {float(_gate):g} → 不参与(空仓)"
                      if _gate_blocked else
                      (f"当日不参与 —— {buy_note}"
                       if not allow_buy else
                       (f"9:25集合竞价缺口 ≤ {_floor:g}% 且涨幅 < {float(ex['skip_open_gain']):g}%、未停牌未一字 → 按{_pos_frac:.0%}仓位买入{_gat_rule}"
                        if _floor else
                        f"9:25集合竞价按开盘价直接买入(不设缺口门槛; 涨幅 < {float(ex['skip_open_gain']):g}%、未停牌未一字) → 按{_pos_frac:.0%}仓位买入{_gat_rule}")))),
            "fallback": f"不达标 → 顺延第{i+1}名" if i < len(verified) else "",
        })
    if not verified:
        steps.append({"rank": 0, "code": "", "name": "（今日无合格候选）", "rule": "明日空仓, 无需操作", "fallback": ""})
    t1_decision = {
        "allow_buy": allow_buy,
        "buy_note": buy_note,
        "panic_day": panic_day,
        "floor_pct": (floor_now if (allow_buy and floor_now) else None),
        "position_pct": round(_pos_frac * ex.get("alloc_frac_pct", 100.0), 1),
        "month_gain_pct": round(month_gain_pct, 1),
        "month_cap_active": cap_frac < 1.0,
        "cap_note": cap_note,
        "score_gate": (float(_gate) if _gate is not None else None),
        "top_score": (round(float(_top_score), 3) if _top_score is not None else None),
        "gate_blocked": _gate_blocked,
        "score_tier_mult": (round(_tier_mult, 3) if _tiers else None),
        "score_tier_note": (_tier_txt or None),
        "pool_n": _pn_today,                       # 第九十九轮: 当日候选池深度
        "pool_tier_mult": (round(_pool_mult, 3) if (ex.get("pool_n_tiers") or []) else None),
        "pool_tier_note": (_pool_txt or None),
        # 第一百零三轮: 市场语境(信号日全市场中位数涨跌)与两条仓位修正
        "mkt_ret_pct": round(_mkt_ret, 2),
        "knife_mult": (round(_knife_mult, 3) if (ex.get("knife_tiers") or []) else None),
        "knife_note": (_knife_txt or None),
        "mkt_floor_note": (_mkt_floor_txt or None),
        "gap_tier_note": (_gat_txt or None),
        "steps": steps,
        "sell_plan": sell_rule_text(ex),
        "empty_funnel": ({"全市场": stats.get("universe"), "流动性/次新": stats.get("liquidity"),
                          "回调形态": stats.get("price_action"), "趋势条件": stats.get("trend_ok"),
                          "评分入池": stats.get("scored")} if not verified else None),
    }

    # 第一百七十六轮: 明日计划双方案(主抄底 / 副追高)。放在 latest 之前算好,
    # 异常在其中被吞掉并写 alt_error, 不影响主流程出池。
    _plans = _build_plans(cfg, by_date[signal_date], _fmt(signal_date), verified, ranked,
                          ex, t1_decision, regime_info, names)

    latest = {
        "update_time": datetime.now(TZ_CN).strftime("%Y-%m-%d %H:%M:%S"),
        "signal_date": _fmt(signal_date),
        "config_fp": _fp,
        "plan": {
            "buy_date": _fmt(buy_d) if buy_d else "",
            "buy_point": "开盘集合竞价(09:15-09:25挂单, 09:25成交)",
            "sell_date": _fmt(sell_d) if sell_d else "",
            "sell_point": sell_rule_text(ex),
            "position": _position_note(ex),
        },
        # 第一百二十五轮: 出场口径禁止硬编码。原文"再次日开盘卖出"写死于第一百一十五轮
        # 之前的开盘腿, 而现行 config 为 sell_open_frac=0(全额 T+2 收盘卖), 与同一屏的
        # plan.sell_point / position 直接矛盾。改为与回测/回放共用 rule_text 的短文案。
        "strategy": "次日开盘买入 · {}（{}{}）".format(
            sell_rule_text(ex, short=True),
            "趋势内缩量回调+次日开盘价成交" if cfg["selection"].get("mode", "momentum") == "dip" else "隔夜动量(备用)",
            ("＋备选补位:" + "/".join(STRATEGY_CN.get(fb.get("mode"), str(fb.get("mode")))
                                   for fb in (cfg["selection"].get("fallbacks") or [])))
            if cfg["selection"].get("fallbacks") else "",
        ),
        "regime": regime_info,
        "stats": {k: v for k, v in stats.items()},
        "picks": verified,
        "plans": _plans,          # 第一百七十六轮: 明日计划(双方案, 人工执行)
        "t1_decision": t1_decision,
        "monitor": monitor_info,
        "observe": observe_info,
        # 第一百二十五轮: 移除 "audit"(9:26 竞价确认 vs 晚间 panel 对账)。
        # auction.json 自 2026-11-01 起已停止更新(生成器不在仓库内), 该对账永远返回
        # checked=False, 属"永不执行却天天写一条 note"的僵尸路径; 前端 au.checked
        # 判空后不渲染, 删除字段对页面无影响。
        "data_quality": {"anomalies": anomalies, "rows": int(len(day_df))},
        "disclaimer": DISCLAIMER,
    }
    (out_dir / "latest.json").write_text(
        json.dumps(latest, ensure_ascii=False, indent=1), encoding="utf-8")

    new_rec = {
        "publish_date": _fmt(signal_date),
        "buy_date": _fmt(buy_d) if buy_d else "",
        "sell_date": _fmt(sell_d) if sell_d else "",
        "regime_ok": regime_info["ok"],
        "picks": [{"code": p["code"], "name": p["name"], "close": p["close"],
                   "pct": p["pct"], "reasons": p["reasons"],
                   "pool": p.get("pool", "main"),
                   "strategy_cn": p.get("strategy_cn", p.get("strategy", "dip"))} for p in verified],
        "status": "待结算",
        "config_fp": _fp,   # 第九十二轮: 记录生成时的执行口径, 供后续区分"历史留档"与"当前口径"
    }
    history = [r for r in history if r["publish_date"] != new_rec["publish_date"]]
    history.insert(0, new_rec)
    keep = cfg["pages"].get("history_keep_days", 60)
    # 汇总仅统计风控通过(实际可开仓)的已结算池; 空仓期观察池不计入胜率
    done = [r for r in history
            if r.get("status") == "已结算" and r.get("realized") and r.get("regime_ok", True) is not False]
    win_n = sum(1 for r in done if r["realized"]["avg_ret_pct"] > 0)
    summary = {
        "n_done": len(done),
        "win_rate": round(win_n / len(done) * 100, 1) if done else None,
        "avg_ret": round(sum(r["realized"]["avg_ret_pct"] for r in done) / len(done), 2) if done else None,
    }
    hist_path.write_text(json.dumps({"summary": summary, "records": history[:keep]},
                                    ensure_ascii=False, indent=1), encoding="utf-8")

    # ---- 7. CHANGELOG 追加运行记录 ----
    cl = ROOT / "CHANGELOG.md"
    if regime_info["ok"]:
        regime_tag = "风控通过"
    elif regime_info.get("reenter_frac") is not None:
        regime_tag = f"超限复参与({regime_info['reenter_frac']:.0%})"
    else:
        regime_tag = "触发空仓"
    line = (f"- {datetime.now(TZ_CN).strftime('%Y-%m-%d %H:%M')} 自动运行: 信号日 {_fmt(signal_date)}, "
            f"候选 {len(verified)} 只, {regime_tag}, "
            f"耗时 {time.time()-t0:.0f}s\n")
    text = cl.read_text(encoding="utf-8") if cl.exists() else "# 更新日志\n\n## 自动运行记录\n"
    marker = "## 自动运行记录\n"
    if marker in text:
        text = text.replace(marker, marker + line, 1)
    else:
        text += "\n## 自动运行记录\n" + line
    cl.write_text(text, encoding="utf-8")

    print(f"候选池({len(verified)}只): " + ", ".join(f"{p['code']} {p['name']}" for p in verified))
    print(f"风控: {regime_info['note']}")
    print(f"完成, 耗时 {time.time()-t0:.0f}s")


if __name__ == "__main__":
    main()
