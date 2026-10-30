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
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

from .config import ROOT, load_config
from .data_api import ZZClient
from .monitor import compute_monitor
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
                s.get("dip_close_pos_max"), s.get("dip_weights")],
        "bias20_max": s.get("bias20_max"),
        "fallbacks": [(f.get("mode"), f.get("top_n"), f.get("gap_below"), f.get("segment"))
                      for f in (s.get("fallbacks") or [])],
        "slot": bool(s.get("fallback_slot")),
        "gap": e.get("buy_gap_below"),
        "block_gap": e.get("block_gap_below"),
        "scd": e.get("sell_close_if_open_drop"),
        "max_buy": e.get("max_buy_per_day"),
        "alloc": e.get("alloc_frac_pct"),
    }
    raw = json.dumps(key, sort_keys=True, ensure_ascii=False, default=str)
    return hashlib.sha1(raw.encode("utf-8")).hexdigest()[:12]


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
    if maxn == 1:
        pos = f"若开仓: 买入评分第1名 1 只, 约 {frac:g}% 总资产仓位"
    elif maxn > 1:
        pos = f"若开仓: 按评分买入最多 {maxn} 只, 每笔约 {frac:g}% 总资产"
    else:
        pos = f"若开仓: 达标票每笔约 {frac:g}% 总资产"
    if gap:
        pos += f"; 仅买开盘缺口 <= {gap:g}% 的低开票(风控日以判定卡承接线为准)"
    else:
        pos += "; 不设缺口门槛, T+1 开盘价直接成交(仅停牌/一字涨停买不到)"
    pos += "; T+1买入, T+2开盘卖出(遵守A股T+1制度, 无当日卖出)"
    scd = float(ex.get("sell_close_if_open_drop") or 0)
    if scd > 0:
        pos += f"; 若T+2开盘跌幅≥{scd:g}%且未跌停, 改为当日收盘卖出"
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


def _sell_exit(rs, row_of, cal, sell_d: str, code: str, ex: dict):
    """卖出日退出判定(第五十四轮, 与回测 do_sell 逐条对齐):
    T+2停牌 -> 向后顺延最多5个交易日开盘卖; 开盘触及跌停且sell_limit_down_fallback
    (config已启用) -> 保守按当日收盘卖。返回 (sell_row, sell_dd, sold_at_close, note);
    None = 顺延窗口内无可交易行(未恢复/无数据), 留待后续结算 —— 实盘每日重跑天然
    顺延, 与回测"模拟内即时顺延"语义等价。差异说明: 回测"退市按末收盘"强平在实盘
    端不可复刻(daily_run面板仅近期窗口, 无全史末收盘); 十年回测该情形0笔、停牌顺延
    0笔、跌停保守2笔(002881.SZ/002612.SZ), 影响面见CHANGELOG第五十四轮量化。
    """
    row, dd, sold_close, note = rs, sell_d, False, ""
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
    return row, dd, sold_close, note


def _refresh_auction_positions(out_dir: Path, day_df: pd.DataFrame, now_str: str) -> None:
    """盘后将 auction.json 的持仓浮盈刷新为收盘价口径。

    9:26竞价确认写入的positions为实时现价(9:26时=开盘价, 即买入价, 浮盈恒0);
    收盘后应以T日收盘价刷新, 避免网页盘后一直显示买入开盘价。
    auction.json 不存在时跳过(不伪造竞价确认数据)。
    """
    ap = out_dir / "auction.json"
    if not ap.exists():
        return
    try:
        a = json.loads(ap.read_text(encoding="utf-8"))
        hist = json.loads((out_dir / "history.json").read_text(encoding="utf-8"))
        holds = [(r.get("executed") or {}) for r in hist.get("records", [])
                 if (r.get("executed") or {}).get("holding") and (r.get("executed") or {}).get("code")]
        close_of = day_df.set_index("ts_code")["close"].to_dict() if day_df is not None and len(day_df) else {}
        positions = []
        for e in holds:
            close = close_of.get(e["code"])
            ok = close is not None and pd.notna(close) and close > 0 and e.get("buy_px")
            positions.append({"code": e["code"], "name": e.get("name", ""),
                              "buy_px": e.get("buy_px"), "buy_date": e.get("buy_date"),
                              "cur_px": round(float(close), 2) if ok else None,
                              "float_pct": round((close / e["buy_px"] - 1) * 100, 2) if ok else None})
        a["positions"] = positions
        a["positions_time"] = now_str
        ap.write_text(json.dumps(a, ensure_ascii=False, indent=1), encoding="utf-8")
        print(f"持仓浮盈已刷新为收盘价口径({len(positions)}只, 截至 {now_str})")
    except Exception as e:  # noqa: BLE001 刷新失败不影响主流程
        print(f"auction.json持仓浮盈刷新失败(不影响主流程): {e}")


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
        print(f"信号日{_fmt(signal_date)}已处理过(latest.json一致, 口径{_fp}), 幂等跳过")
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

    # ---- 5. 结算 T-2 发布池 ----
    out_dir = ROOT / "output"
    out_dir.mkdir(exist_ok=True)
    hist_path = out_dir / "history.json"
    if hist_path.exists():
        _data = json.loads(hist_path.read_text(encoding="utf-8"))
        history = _data.get("records", []) if isinstance(_data, dict) else _data
    else:
        history = []
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
            if _consec <= max_block:
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
        maxn = int(ex.get("max_buy_per_day", 0))
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
            return {
                "rank": rank_i, "code": p["code"], "name": p.get("name", p["code"]),
                "buy_date": _fmt(buy_d), "buy_px": round(float(rb["open"]), 2),
                "sell_date": _fmt(sell_d) if sell_d else "", "sell_px": None,
                "ret_pct": None,
                "position_pct": round(bfrac * _cf * ex.get("alloc_frac_pct", 100.0), 1),
                "gap_pct": round(gap, 2),
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
            sx = _sell_exit(rs, row_of, cal, sell_d, p["code"], ex)
            if sx is None:
                continue  # 顺延窗口内未恢复: 留待后续结算
            sell_row, sell_dd, sold_close, sell_note = sx
            r = (float(sell_row["close_adj" if sold_close else "open_adj"]) / rb["open_adj"]) * gap_fee - 1
            rets.append(float(r))
            last_sell_dd = sell_dd
            # 第五十三轮: 结算仓位含月度盈利熔断系数(买入日口径, 仅已结算复利)
            _, _cf, _ = _month_gain_state([], history, (buy_d or "")[:6], None, cfg, include_hold=False)
            executed = {
                "rank": rank_i, "code": p["code"], "name": p.get("name", p["code"]),
                "buy_date": _fmt(buy_d), "buy_px": round(float(rb["open"]), 2),
                "sell_date": _fmt(sell_dd),
                "sell_px": round(float(sell_row["close" if sold_close else "open"]), 2),
                "ret_pct": round(float(r) * 100, 2),
                "position_pct": round(bfrac * _cf * ex.get("alloc_frac_pct", 100.0), 1),
                "gap_pct": round(gap, 2),
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
                      (f"大盘风控触发(连续第{consec}日): 次日建议空仓观望" if reg_row is not None else "")),
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
    plan_frac = (over_frac if (over_limit_now and not regime_ok_now) else 1.0) * cap_frac
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
    else:
        allow_buy = False
        buy_note = f"大盘风控中(连续第{consec}日): 空仓观望"
    if allow_buy and cap_frac < 1.0:
        buy_note += f"; {cap_note}"
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
            "rule": (f"9:25集合竞价缺口 ≤ {_floor:g}% 且涨幅 < {float(ex['skip_open_gain']):g}%、未停牌未一字 → 按{plan_frac:.0%}仓位买入"
                     if _floor else
                     f"9:25集合竞价按开盘价直接买入(不设缺口门槛; 涨幅 < {float(ex['skip_open_gain']):g}%、未停牌未一字) → 按{plan_frac:.0%}仓位买入"),
            "fallback": f"不达标 → 顺延第{i+1}名" if i < len(verified) else "",
        })
    if not verified:
        steps.append({"rank": 0, "code": "", "name": "（今日无合格候选）", "rule": "明日空仓, 无需操作", "fallback": ""})
    t1_decision = {
        "allow_buy": allow_buy,
        "buy_note": buy_note,
        "panic_day": panic_day,
        "floor_pct": (floor_now if (allow_buy and floor_now) else None),
        "position_pct": round(plan_frac * ex.get("alloc_frac_pct", 100.0), 1),
        "month_gain_pct": round(month_gain_pct, 1),
        "month_cap_active": cap_frac < 1.0,
        "cap_note": cap_note,
        "steps": steps,
        "sell_plan": (f"T+2 开盘集合竞价卖出(开盘跌幅≥{float(ex.get('sell_close_if_open_drop') or 0):g}%或跌停则当日收盘卖出)"
                      if float(ex.get("sell_close_if_open_drop") or 0) > 0
                      else "T+2 开盘集合竞价卖出(开盘跌停则当日收盘卖出)"),
        "empty_funnel": ({"全市场": stats.get("universe"), "流动性/次新": stats.get("liquidity"),
                          "回调形态": stats.get("price_action"), "趋势条件": stats.get("trend_ok"),
                          "评分入池": stats.get("scored")} if not verified else None),
    }

    def _audit_vs_auction(out_dir, today_fmt, history):
        """两道判定一致性对账: 9:26竞价确认(auction.json, 腾讯实时) vs 晚间panel结算(权威)。
        两道判定应一致; 不一致说明数据源异常, 返回match=False供告警。"""
        ap = out_dir / "auction.json"
        if not ap.exists():
            return {"checked": False, "note": "无竞价确认记录"}
        try:
            a = json.loads(ap.read_text(encoding="utf-8"))
        except Exception:
            return {"checked": False, "note": "auction.json 解析失败"}
        if a.get("buy_date", "").replace("-", "") != today_fmt.replace("-", ""):
            return {"checked": False, "note": f"auction非今日数据({a.get('buy_date')})"}
        # 今日实际结算结果(executed: 已结算或持仓中)
        actual = None
        for rec in history:
            e = rec.get("executed") or {}
            if str(e.get("buy_date", "")).replace("-", "") == today_fmt.replace("-", "") and not e.get("holding"):
                actual = e  # 已结算(卖出也完成了)
                break
            if str(e.get("buy_date", "")).replace("-", "") == today_fmt.replace("-", ""):
                actual = e
                break
        a_rank, a_code = a.get("rank"), a.get("code")
        if a.get("action") == "buy":
            if actual and str(actual.get("code")) == str(a_code):
                return {"checked": True, "match": True,
                        "note": f"两道判定一致: 买入{a.get('name')} 第{a_rank}名 (竞价缺口{a.get('gap_pct')}%)"}
            if actual:
                return {"checked": True, "match": False,
                        "note": f"不一致! 竞价确认买{a.get('name')}, panel结算为{actual.get('name')} —— 请检查数据源"}
            return {"checked": True, "match": False,
                    "note": "不一致! 竞价确认有买入但panel未结算到成交 —— 请检查数据源"}
        # auction 认为空仓
        if actual:
            return {"checked": True, "match": False,
                    "note": f"不一致! 竞价确认空仓但panel结算有成交({actual.get('name')}) —— 请检查数据源"}
        return {"checked": True, "match": True, "note": "两道判定一致: 今日空仓"}

    latest = {
        "update_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "signal_date": _fmt(signal_date),
        "config_fp": _fp,
        "plan": {
            "buy_date": _fmt(buy_d) if buy_d else "",
            "buy_point": "开盘集合竞价(09:15-09:25挂单, 09:25成交)",
            "sell_date": _fmt(sell_d) if sell_d else "",
            "sell_point": (f"次日开盘集合竞价卖出(开盘跌幅≥{float(ex.get('sell_close_if_open_drop') or 0):g}%且未跌停则当日收盘卖出)"
                           if float(ex.get("sell_close_if_open_drop") or 0) > 0
                           else "次日开盘集合竞价卖出"),
            "position": _position_note(ex),
        },
        "strategy": "次日开盘买入-再次日开盘卖出({}{})".format(
            "趋势内缩量回调+次日低开承接" if cfg["selection"].get("mode", "momentum") == "dip" else "隔夜动量(备用)",
            ("＋备选补位:" + "/".join(STRATEGY_CN.get(fb.get("mode"), str(fb.get("mode")))
                                   for fb in (cfg["selection"].get("fallbacks") or [])))
            if cfg["selection"].get("fallbacks") else "",
        ),
        "regime": regime_info,
        "stats": {k: v for k, v in stats.items()},
        "picks": verified,
        "t1_decision": t1_decision,
        "monitor": monitor_info,
        "observe": observe_info,
        "audit": _audit_vs_auction(out_dir, _fmt(signal_date), history),
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

    # ---- 6.5 盘后刷新 auction.json 持仓浮盈为收盘价口径 ----
    _refresh_auction_positions(out_dir, day_df, datetime.now().strftime("%Y-%m-%d %H:%M:%S"))

    # ---- 7. CHANGELOG 追加运行记录 ----
    cl = ROOT / "CHANGELOG.md"
    if regime_info["ok"]:
        regime_tag = "风控通过"
    elif regime_info.get("reenter_frac") is not None:
        regime_tag = f"超限复参与({regime_info['reenter_frac']:.0%})"
    else:
        regime_tag = "触发空仓"
    line = (f"- {datetime.now().strftime('%Y-%m-%d %H:%M')} 自动运行: 信号日 {_fmt(signal_date)}, "
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
