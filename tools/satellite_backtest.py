# -*- coding: utf-8 -*-
"""副策略回测(第一百四十轮) —— 给页面提供与主策略同构的「回测绩效」模块。

为什么单独一个脚本
------------------
主策略的数值来自 src/backtest.py(多票组合/分档仓位/评分门槛), 副策略是单吊满仓、
持 10 日、大盘闸门, 两套引擎无法共用一次模拟。但**输出结构必须一致**:
页面用同一个渲染器画卡片/净值曲线/月度收益/期末持仓, 只有数字来源不同。

口径与实盘账本(tools/satellite_run.py)逐条对齐, 不是"另写一套回测":
  · 选股 rank_pool()/pick_one()、买 entry_gate()、卖 should_exit() 全部 import 自 src.satellite
  · T 日收盘出信号(大盘闸门 reg.trade_ok 在信号日收盘判定) → T+1 开盘价买入
  · 100% 仓位, 单吊 1 只, 卖出资金当日收盘后再出信号(次日照常买)
  · 单边成本 0.15%(滑点千1 + 佣金万2.5 + 印花税万5 的一半), 与账本 HALF 同一常数
  · 持有 held>=hold_days 个交易日收盘卖; 收盘相对买入价跌破 stop_pct 当日收盘止损

无未来函数: 指标全部 rolling/shift(1); 买入用 T+1 开盘、卖出用 T+N 收盘;
风控闸门只用信号日收盘可知的 trade_ok。

用法: python tools/satellite_backtest.py [--days 1500] [--out output/satellite_bt.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timedelta, timezone

TZ_CN = timezone(timedelta(hours=8))  # 北京时间: Actions 运行于 UTC, 展示型时间戳统一 +8
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context  # noqa: E402
from src.config import load_config  # noqa: E402
from src.satellite import (STRATEGY_CN, daily_params, entry_gate, entry_ok_series,  # noqa: E402
                           pick_list, rank_qualified, should_exit)

HALF = 0.0015          # 单边成本, 与 tools/satellite_run.py 同一常数
REPLAY_DAYS = 250      # 回放区间: 最近约 12 个月交易日(与主策略 replay 的区间长度一致)


def _row(day, code):
    if not code:
        return None
    r = day.loc[day["ts_code"] == code]
    return r.iloc[0] if len(r) else None


def _fmt(d: str) -> str:
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


def _d8(s) -> str:
    return str(s).replace("-", "")[:8]


def _monthly(nav: pd.Series) -> list[dict]:
    """与 src.backtest._monthly 同口径: 每月收益 = 月末净值 / 上月末净值 - 1。"""
    m = nav.copy()
    m.index = [str(i)[:6] for i in m.index]
    g = m.groupby(level=0).agg(["first", "last"])
    out, prev = [], None
    for idx, row in g.iterrows():
        base = prev if prev else row["first"]
        out.append({"month": idx, "ret_pct": round((row["last"] / base - 1) * 100, 2)})
        prev = row["last"]
    return out


def _monthly_from(base_nav: float, curve: list[dict]) -> list[dict]:
    """区间月度收益(与 tools/replay.py 同一口径): 首月基准取 base_nav, 之后取上月末净值。"""
    out, prev, i, n = [], float(base_nav), 0, len(curve)
    while i < n:
        ym = _d8(curve[i]["d"])[:6]
        j = i
        while j + 1 < n and _d8(curve[j + 1]["d"])[:6] == ym:
            j += 1
        last = float(curve[j]["nav"])
        out.append({"month": ym, "ret_pct": round((last / prev - 1) * 100, 2)})
        prev, i = last, j + 1
    return out


def _build_replay(dates: list, nav: pd.Series, trades: list[dict],
                  daily: list[dict], init: float, P: dict) -> dict:
    """把 1500 日回测切出「近 12 个月逐日回放」, 结构对齐 output/replay.json。

    回放与回测同一引擎、同一次模拟 —— 只切片不重跑, 所以两者永远不会打架。
    """
    # 区间口径与主策略 tools/replay.py 一致: 最新交易日前推 12 个月(不是"最近 N 个交易日"),
    # 这样两块回放的起点日期相同, 并排看不会出现"一个 10-27、一个 11-05"的错位。
    end = dates[-1]
    since = f"{int(end[:4]) - 1}{end[4:]}"
    span = [x for x in dates if x >= since] or dates[-REPLAY_DAYS:]
    start = span[0]
    pre = [x for x in dates if x < start]
    base_date = pre[-1] if pre else start
    base_nav = float(nav.loc[base_date]) if pre else float(init)
    curve = [{"d": _fmt(x), "nav": round(float(nav.loc[x]), 0)} for x in span]
    end_nav = float(curve[-1]["nav"])

    navs = [float(nav.loc[x]) for x in span]
    peak, mdd = navs[0], 0.0
    for v in navs:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)

    # 区间内平仓的交易(卖出日落在区间), 这样逐笔之和能解释区间收益
    rtr = [t for t in trades if _d8(t["exit_date"]) >= start]
    rets = [float(t["ret_pct"]) for t in rtr]
    n, win = len(rets), sum(1 for r in rets if r > 0)
    summary = {
        "n_trades": n, "n_win": win,
        "win_rate": round(win / n * 100, 1) if n else None,
        "avg_ret_pct": round(sum(rets) / n, 2) if n else None,
        "sum_ret_pct": round(sum(rets), 2) if n else None,
        "best": round(max(rets), 2) if n else None,
        "worst": round(min(rets), 2) if n else None,
        "range_ret_pct": round((end_nav / base_nav - 1) * 100, 2),
        "range_mdd_pct": round(mdd * 100, 2),
        "base_date": _fmt(base_date), "base_nav": round(base_nav, 0),
        "end_nav": round(end_nav, 0),
    }
    r_trades = []
    for t in rtr:                      # 日期统一成 8 位, 与主策略 replay 同格式
        r_trades.append({**t,
                         "signal_date": _d8(t["signal_date"]),
                         "entry_date": _d8(t["entry_date"]),
                         "exit_date": _d8(t["exit_date"])})
    return {
        "meta": {"start": start, "end": span[-1], "days": len(span),
                 "base_date": _fmt(base_date),
                 "strategy": "低波单吊", "window": f"近 {len(span)} 个交易日"},
        "summary": summary,
        "monthly": _monthly_from(base_nav, curve),
        "curve": curve,
        "trades": r_trades,
        "daily": [x for x in daily if x["date"] >= start],
    }


def run(ctx: dict, cfg: dict, defer_on_limit_down: bool | None = None,
        stamp_2023: bool | None = None,
        entry_ok: pd.Series | None = None) -> dict:
    # 第 162 轮: 两个执行摩擦开关默认从 config 读, 脚本可显式覆盖(用于对照实验)。
    if defer_on_limit_down is None:
        defer_on_limit_down = bool((cfg.get("satellite") or {}).get("defer_on_limit_down", False))
    if stamp_2023 is None:
        stamp_2023 = bool((cfg.get("execution") or {}).get("stamp_tax_time_varying", False))
    P = daily_params(cfg)
    # 第一百七十一轮: 融资杠杆(默认 1.0, 与旧口径逐位一致)。
    #   副策略的"仓位效率"只有 1/max_pos(4 只各 25%), 拉高 lev 等于把每个槽位放大。
    #   lev>1 时允许透支 (lev-1)*nav, 透支额按 margin_rate 年化逐日计提。
    #   与主策略 execution.leverage 同一套约定, 便于两条腿并排比较。
    _sat = cfg.get("satellite") or {}
    LEV = float(_sat.get("leverage", 1.0) or 1.0)
    RATE = float(_sat.get("margin_rate", 0.055) or 0.0) if LEV > 1.0 else 0.0
    _interest = 0.0
    # 第一百五十六轮修复: cost_pct 此前是**死参数** —— config 里写 0.8 也不生效,
    # 因为买卖两侧直接用了硬编码常量 HALF。现在真正读 config: 往返成本 cost_pct%
    # 对半分到买卖两侧。默认 cost_pct=0.3 -> 单边 0.0015 = HALF, 旧结果逐位不变。
    # 这解释了为什么"把成本从 0.1% 调到 0.5%, 回测纹丝不动" —— 参数根本没接上。
    _half = (P["cost"] / 2.0) if P["cost"] > 0 else HALF
    # 第一百六十二轮: 印花税时变。2023-08-28 起 A 股卖出印花税由千 1 降至万 5,
    # 而 cost_pct=0.3 的构成注释里用的是**万 5**(降税后口径)。也就是说 2023-08-28 之前
    # 的每一笔都少算了 0.05% 的卖出成本 —— 一个从未被审过的、方向单一的系统性乐观偏差。
    # 打开后只在卖出侧按日期加 0.05%, 买入侧不变(印花税只在卖出时收)。
    _STAMP_CUT = "20230828"

    def _sell_net(d: str, gross: float) -> float:
        extra = 0.0005 if (stamp_2023 and str(d) < _STAMP_CUT) else 0.0
        return gross * (1 - _half - extra)
    dates = ctx["main_dates"]
    by_date = ctx["by_date"]
    reg = ctx.get("reg")
    names = ctx.get("names") or {}
    init = 100000.0
    # 风控序列的索引在不同装载路径下可能是 int(本地 csv)或 str(线上 json) —— 统一成
    # "20261102" 字符串, 否则闸门查不到、基准也取不到(会静默变成"全程放行 + 无基准")。
    if reg is not None and len(reg) and not isinstance(reg.index[0], str):
        reg = reg.copy()
        reg.index = reg.index.astype(str)

    cash, positions = init, []
    # 第一百七十九轮: 回踩限价挂单。pb=0 时不启用(pending 恒为空, 行为与旧口径逐位一致)。
    _PB, _PBD, _PB_MODE = P["pullback_pct"], P["pullback_days"], P["pullback_mode"]
    pending: list[dict] = []
    sig = None
    navs, trades, daily = [], [], []
    blocked = skipped = 0
    last_px = {}

    for j, d in enumerate(dates):
        day = by_date.get(d)
        if day is None:
            navs.append((d, cash + sum(p["units"] * last_px.get(p["code"], p["entry_px"])
                                       for p in positions)))
            daily.append({"date": d, "trade_ok": None, "alloc": None, "pool_n": None,
                          "holding": bool(positions), "fills": [], "skipped": [],
                          "note": "面板缺该日行情"})
            continue

        # 融资利息: 对透支余额按年化逐日计提
        if cash < 0 and RATE > 0:
            _i = -cash * RATE / 252.0
            cash -= _i
            _interest += _i

        # 当日收盘的闸门(只用当日可知信息): 决定明日能否开新仓
        gate_today = True
        if P["gate"] and reg is not None and d in reg.index:
            gate_today = bool(reg.loc[d, "trade_ok"])
        rec = {"date": d, "trade_ok": gate_today,
               "alloc": round(min(len(positions) / P["max_pos"], 1.0) * 100, 1),
               "pool_n": None, "holding": bool(positions),
               "fills": [], "skipped": [], "note": ""}
        # 第一百八十一轮: 入场闸门与清仓闸门分离(仅实验路径, 默认 None = 两者同源)。
        #   动机: 单吊 120 笔里 28 笔"买入当日就被大盘闸门清仓", 说明信号日 gate_ok 但
        #   T+1 收盘就破位 —— 买在了临界点。允许外层传一条更严的入场序列(如"连续 2 日
        #   trade_ok"), 清仓仍用当日 gate_today, 从而只过滤入场、不放松风控。
        _entry_today = gate_today if entry_ok is None else bool(entry_ok.get(d, gate_today))
        ev = []

        # ---------- 0) 开盘: 昨日跌停未能卖出的挂单(第一百六十二轮) ----------
        for p in list(positions):
            if not p.get("_pending_exit"):
                continue
            r = _row(day, p["code"])
            if r is None or bool(r["is_paused"] == 1) or float(r["vol"]) <= 0:
                continue                      # 停牌, 继续挂着
            op = float(r["open"])
            if op <= float(r["low_limit"]) + 1e-9:
                p["_defer_n"] = int(p.get("_defer_n") or 0) + 1
                ev.append(f"{p['name']} 开盘仍跌停，继续挂单")
                continue                      # 开盘一字跌停, 还是卖不出
            px_adj = float(r["open_adj"])
            last_px[p["code"]] = float(r["close_adj"])
            ret2 = (px_adj / p["entry_px"] - 1) * 100
            cash += _sell_net(d, p["units"] * px_adj)
            trades.append({"code": p["code"], "name": p["name"],
                           "signal_date": p["signal_date"], "entry_date": p["entry_date"],
                           "exit_date": d, "entry_px": round(p["entry_px_raw"], 3),
                           "exit_px": round(op, 3),
                           "gap_pct": p.get("gap_pct"),
                           "hold_days": j - p["entry_j"],
                           "ret_pct": round(ret2, 2),
                           "note": f"{p['_pending_exit']} 跌停顺延{p.get('_defer_n', 1)}日开盘",
                           "entry_px_adj": round(p["entry_px"], 4),
                           "exit_px_adj": round(px_adj, 4)})
            ev.append(f"卖出 {p['name']}（跌停顺延，开盘成交）")
            positions.remove(p)

        # ---------- 1) 开盘: 限价挂单撮合(第一百七十九轮) ----------
        # pb=0 时等价旧口径(T+1 开盘无条件买, _exp 当日失效); pb>0 时挂"信号日收盘-pb%"的
        # 限价单, pullback_days 个交易日内有效, 到期未触及则放弃(信号衰减)。
        for _s in (sig if isinstance(sig, list) else ([sig] if sig else [])):
            if _PB > 0:
                _lim = _s.get("_lim")
                if _lim is None:
                    continue
                _exp = j + _PBD - 1
            else:
                _lim = float("inf")          # 不回踩: 开盘价必然低于无穷大, 等价旧口径
                _exp = j                     # 不回踩时挂单当日失效, 不跨日补买
            pending.append({"_s": _s, "_lim": _lim, "_exp": _exp})
        sig = None
        for o in list(pending):
            if len(positions) >= P["max_pos"]:
                break
            if j > o["_exp"]:
                pending.remove(o)
                continue
            s2 = o["_s"]
            r = _row(day, s2["code"])
            nm = names.get(s2["code"], s2.get("name", ""))
            if r is None or bool(r["is_paused"] == 1) or float(r["vol"]) <= 0:
                if j >= o["_exp"]:
                    pending.remove(o)
                    skipped += 1
                    rec["skipped"].append({"code": s2["code"], "name": nm, "note": "停牌或无行情，买不到"})
                continue
            op = float(r["open"])
            if op >= float(r["high_limit"]) - 1e-9:
                pending.remove(o)
                skipped += 1
                rec["skipped"].append({"code": s2["code"], "name": nm, "note": "一字涨停，买不到"})
                continue
            ratio = float(r["open_adj"]) / op if op else 1.0
            low_adj = float(r["low"]) * ratio
            high_adj = float(r["high"]) * ratio
            gap = (op / float(r["pre_close"]) - 1) * 100 if float(r["pre_close"]) else 0.0
            ok, why = entry_gate(gap, cfg)
            fill_adj = fill_raw = None
            if ok:
                if _PB_MODE == "breakout":
                    # 右侧买入: 只在价格**向上突破**信号日收盘 +pb% 时成交(停损买单)。
                    # 开盘已高开过线 -> 按开盘价(更差), 否则按限价; 全天最高都没到线 -> 不成交。
                    if float(r["open_adj"]) >= o["_lim"] - 1e-9:
                        fill_adj, fill_raw = float(r["open_adj"]), op
                    elif high_adj >= o["_lim"] - 1e-9:
                        fill_adj, fill_raw = o["_lim"], o["_lim"] / ratio
                elif _PB_MODE == "lowopen" and gap <= 0:
                    # 低开即买: 开盘已比昨收便宜, 不再等更深; 只有高开才挂限价等回踩。
                    fill_adj, fill_raw = float(r["open_adj"]), op
                elif _PB <= 0 or float(r["open_adj"]) <= o["_lim"] + 1e-9:
                    fill_adj, fill_raw = float(r["open_adj"]), op      # 开盘价已在限价之下 -> 开盘成交
                elif _PB_MODE != "open" and low_adj <= o["_lim"] + 1e-9:
                    fill_adj, fill_raw = o["_lim"], o["_lim"] / ratio  # 盘中回踩触及 -> 限价成交
            if fill_adj is None:
                if not ok:
                    pending.remove(o)
                    skipped += 1
                    rec["skipped"].append({"code": s2["code"], "name": nm, "note": why})
                elif j >= o["_exp"]:
                    pending.remove(o)
                    skipped += 1
                    rec["skipped"].append({"code": s2["code"], "name": nm, "note": "未回踩到限价，放弃"})
                continue
            px = fill_adj * (1 + _half)
            # 单笔预算 = 总资产 / max_pos。总资产用"现金 + 持仓成本"估算:
            # 买入发生在开盘, 当日收盘价还不可知, 用成本不引入未来函数。
            nav_now = cash + sum(p["units"] * p["entry_px"] for p in positions)
            _slots = P["budget_slots"] or P["max_pos"]
            _cap = cash + max(0.0, (LEV - 1.0) * nav_now) if LEV > 1.0 else cash
            budget = min(nav_now / _slots * LEV, _cap * 0.995)
            if budget < 1000:
                pending.remove(o)
                skipped += 1
                rec["skipped"].append({"code": s2["code"], "name": nm, "note": "现金不足"})
                continue
            units = budget / px
            cash -= units * px
            positions.append({"code": s2["code"], "name": nm,
                              "signal_date": s2["date"], "entry_date": d, "entry_j": j,
                              "entry_px": px, "entry_px_raw": fill_raw,
                              "gap_pct": round(gap, 2), "units": units,
                              "hwm": px})
            rec["fills"].append({"code": s2["code"], "name": nm, "ret_pct": None})
            pending.remove(o)

        # ---------- 2) 收盘: 逐只卖出判定 ----------
        for p in list(positions):
            if p.get("_pending_exit"):
                continue                      # 已在挂单队列, 不再重复判定(否则 _defer_n 会重复计数)
            r = _row(day, p["code"])
            if r is None or bool(r["is_paused"] == 1) or float(r["vol"]) <= 0:
                continue
            cl = float(r["close_adj"])
            last_px[p["code"]] = cl
            p["hwm"] = max(float(p.get("hwm") or p["entry_px"]), cl)
            ret = (cl / p["entry_px"] - 1) * 100
            # 第一百八十轮: 大盘闸门已搬进 should_exit(与实盘账本同一套判定), 这里只传状态。
            ex, why = should_exit(j - p["entry_j"], ret, cfg, hwm=p.get("hwm"),
                                  close=cl, gate_ok=gate_today)
            # 持仓层风控(第一百五十二轮): exit_ma 仍在引擎内(实盘未启用该旋钮, config 为 0)
            if not ex and P["exit_ma"] and j - p["entry_j"] >= 1:
                _ma = float(r.get("ma%d" % P["exit_ma"]) or 0)
                if _ma > 0 and cl < _ma:
                    ex, why = True, "跌破 MA%d" % P["exit_ma"]
            if ex:
                # 第一百六十二轮: 收盘封死跌停 -> 卖不出。实盘只能次日开盘挂单。
                # 判定用未复权收盘价 vs 跌停价(交易所口径), 与复权无关。
                # 注意这不是"保守假设": 跌停 -10% 必然击穿 -12% 止损以外的一切阈值,
                # 移动止损(-8%)触发的 63 笔里有 5 笔落在跌停日 —— 回测把"卖不掉"当成了"卖在跌停价"。
                if defer_on_limit_down and float(r["close"]) <= float(r["low_limit"]) + 1e-9:
                    p["_pending_exit"] = why
                    p["_defer_n"] = int(p.get("_defer_n") or 0) + 1
                    ev.append(f"{p['name']} 收盘跌停，挂单顺延（{why}）")
                    continue
                cash += _sell_net(d, p["units"] * cl)
                trades.append({"code": p["code"], "name": p["name"],
                               "signal_date": p["signal_date"], "entry_date": p["entry_date"],
                               "exit_date": d, "entry_px": round(p["entry_px_raw"], 3),
                               "exit_px": round(float(r["close"]), 3),
                               "gap_pct": p.get("gap_pct"),
                               "hold_days": j - p["entry_j"],
                               "ret_pct": round(ret, 2), "note": why,
                               # 复权口径的成交价(与 ret_pct 自洽; entry_px/exit_px 是屏幕原始价)
                               "entry_px_adj": round(p["entry_px"], 4),
                               "exit_px_adj": round(cl, 4)})
                ev.append(f"卖出 {p['name']}（{why}）")
                positions.remove(p)

        # ---------- 3) 收盘: 出明日信号(先过大盘闸门) ----------
        if len(positions) < P["max_pos"]:
            if not _entry_today:
                blocked += 1
                ev.append("大盘闸门关闭（指数在 MA20 下方），不开新仓")
            else:
                # 取 max(buys_per_day, min_cands) 只: 既要够买, 也要能判断"信号宽度"
                cands = pick_list(day, cfg, held=[p["code"] for p in positions],
                                  k=max(P["buys_per_day"], P["min_cands"]))
                if len(cands) >= P["min_cands"]:
                    cands = cands[:P["buys_per_day"]]
                    for cand in cands:
                        cand["date"] = d
                        cand["name"] = names.get(cand["code"], "")
                        if _PB > 0:
                            # 限价基准必须锁在**信号日收盘**(T 日已知), 不能读 T+1 行情
                            _r0 = _row(day, cand["code"])
                            cand["_lim"] = (float(_r0["close_adj"]) *
                                            (1 + _PB / 100.0 if _PB_MODE == "breakout"
                                             else 1 - _PB / 100.0)
                                            if _r0 is not None else None)
                    sig = cands if P["buys_per_day"] > 1 else cands[0]
                elif cands:
                    ev.append("候选 %d 只 < 宽度门槛 %d，不出票" % (len(cands), P["min_cands"]))
                    rec["pool_n"] = len(rank_qualified(day, cfg))
                else:
                    ev.append("无票跌破 MA5 或候选池不足，不出票")
        else:
            ev.append(f"持仓已满 {len(positions)}/{P['max_pos']}")

        # ---------- 4) 收盘估值 ----------
        mv = 0.0
        for p in positions:
            r = _row(day, p["code"])
            if r is not None and bool(r["is_paused"] == 0) and float(r["vol"]) > 0:
                last_px[p["code"]] = float(r["close_adj"])
            mv += p["units"] * last_px.get(p["code"], p["entry_px"])
            p["held_days"] = j - p["entry_j"]
            p["unrealized_pct"] = round(
                (last_px.get(p["code"], p["entry_px"]) / p["entry_px"] - 1) * 100, 2)
        if positions:
            ev.append("持仓 " + "、".join(
                "%s(%d/%d日)" % (p["name"], p["held_days"], P["hold_days"]) for p in positions))
        rec["note"] = "；".join(ev)
        rec["holding"] = bool(positions)
        rec["alloc"] = round(min(len(positions) / P["max_pos"], 1.0) * 100, 1)
        navs.append((d, cash + mv))
        daily.append(rec)

    nav = pd.Series({d: v for d, v in navs}, dtype=float)
    ret = nav.pct_change().dropna()
    years = len(nav) / 250.0
    total = nav.iloc[-1] / init - 1
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and (1 + total) > 0 else -1.0
    mdd = float((nav / nav.cummax() - 1).min())
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() else 0.0
    real = [t for t in trades if t["ret_pct"] is not None]
    wins = [t for t in real if t["ret_pct"] > 0]
    losses = [t for t in real if t["ret_pct"] <= 0]
    gw = sum(t["ret_pct"] for t in wins)
    gl = -sum(t["ret_pct"] for t in losses)

    bench = reg.loc[reg.index.isin(dates), "mkt_idx"] if reg is not None else pd.Series(dtype=float)
    bench_norm = bench / bench.iloc[0] * init if len(bench) else bench

    open_positions = []
    for p in positions:
        mv = p["units"] * last_px.get(p["code"], p["entry_px"])
        open_positions.append({
            "signal_date": _fmt(p["signal_date"]), "code": p["code"], "name": p["name"],
            "buy_date": _fmt(p["entry_date"]), "buy_px": round(p["entry_px_raw"], 3),
            "cost_pct": round(100.0 / P["max_pos"], 1),
            "last_px": round(last_px.get(p["code"], 0), 3),
            "unrealized_pct": round((mv / (p["units"] * p["entry_px"]) - 1) * 100, 2)
            if p["units"] else None,
        })

    return {
        "meta": {
            "run_time": datetime.now(TZ_CN).strftime("%Y-%m-%d %H:%M:%S"),
            "start": dates[0], "end": dates[-1], "days": len(dates),
            "initial_capital": init, "strategy": STRATEGY_CN.get(P["mode"], P["mode"]),
            "rules": (f"最多 {P['max_pos']} 只并发 · 每日最多买 {P['buys_per_day']} 只 · 持 {P['hold_days']} 个交易日收盘卖"
                      + (f" · 只在跌破 MA5 {abs(P['dist_ma5_max']):g}% 时买" if P["dist_ma5_max"] else "")
                      + (f" · 低波前 {P['ivol_pct'] * 100:g}%" if P["ivol_pct"] else "")
                      # 第一百八十轮: 候选池是"展示几只", 执行是"买前几名" —— 单吊时两者不同,
                      # 旧文案只写"候选池≤4只", 与"最多 1 只并发"并排看会自相矛盾。
                      + (f" · 候选池展示≤{P['pool_show']}只 / 只买前 {P['buys_per_day']} 名(评分≥{P['min_score']:g})"
                         if P["min_score"] > 0
                         else f" · 候选池展示≤{P['pool_show']}只 / 只买前 {P['buys_per_day']} 名")
                      + (f" / 止损 {P['stop'] * 100:g}%" if P["stop"] < 0 else "")
                      + (" · 大盘 MA20 闸门" if P["gate"] else "")
                      + f" · 单边成本 {_half * 100:g}%"),
        },
        "metrics": {
            "总收益率%": round(total * 100, 2),
            "年化收益率%": round(cagr * 100, 2),
            "最大回撤%": round(mdd * 100, 2),
            "卡玛比率": round(cagr / abs(mdd), 2) if mdd else None,
            "夏普比率": round(sharpe, 2),
            "交易次数": len(real),
            "胜率%": round(len(wins) / len(real) * 100, 2) if real else 0,
            "平均单笔收益%": round(float(np.mean([t["ret_pct"] for t in real])), 3) if real else 0,
            "盈亏比": round(gw / gl, 2) if gl else None,
            # 第一百六十二轮修复: 原式 len(nav) - sum(hold_days+1) 在并发持仓时把重叠
            # 日期重复扣减, 1400 日窗口算出 -1336 天(负值)。改为按日统计无持仓天数。
            "空仓天数": sum(1 for x in daily if not x["holding"]),
            "跳过买入次数": skipped,
            "同期市场等权基准%": round(float(bench_norm.iloc[-1] / init - 1) * 100, 2)
            if len(bench_norm) else None,
            "融资利息(元)": round(_interest, 0) if LEV > 1.0 else 0,
        },
        "equity": [{"d": _fmt(d), "nav": round(float(v), 0)} for d, v in nav.items()],
        "benchmark": [{"d": _fmt(d), "nav": round(float(v), 0)} for d, v in bench_norm.items()]
        if len(bench_norm) else [],
        "monthly": _monthly(nav),
        "trades": trades,
        # 回放: 从同一次模拟里切出近 12 个月, 不重跑 -- 回测与回放不可能对不上
        "replay": _build_replay(dates, nav, trades, daily, init, P),
        **({"open_positions": open_positions} if open_positions else {}),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=None)
    ap.add_argument("--out", default="output/satellite_bt.json")
    args = ap.parse_args()

    cfg = load_config()
    days = args.days or int((cfg.get("backtest") or {}).get("trading_days", 1500))
    ctx = prepare_context(cfg, days)
    print(f"窗口 {len(ctx['main_dates'])} 日 {ctx['main_dates'][0]} ~ {ctx['main_dates'][-1]}", flush=True)

    # 第一百八十一轮: 入场闸门(指数站上 MA20+0.3%)。buf=0 时该函数返回 None, 行为与旧口径一致。
    _entry = entry_ok_series(ctx.get("reg"), cfg)
    res = run(ctx, cfg, entry_ok=_entry)
    m = res["metrics"]
    print(f"总收益 {m['总收益率%']:+.2f}%  年化 {m['年化收益率%']:+.2f}%  回撤 {m['最大回撤%']:.2f}%  "
          f"卡玛 {m['卡玛比率']}  笔数 {m['交易次数']}  胜率 {m['胜率%']}%  单笔 {m['平均单笔收益%']:+.3f}%",
          flush=True)

    out = Path(args.out)
    if not out.is_absolute():
        out = ROOT / out
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"写出 {out}", flush=True)


if __name__ == "__main__":
    main()
