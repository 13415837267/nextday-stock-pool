# -*- coding: utf-8 -*-
"""独立副策略账本 v3 —— 单吊(与主策略同构)。

数据：复用 src.backtest.prepare_context（与主策略同一个 ZZClient、同一份面板缓存
data/cache/market_*.pkl），不单独抓数、不单独建指标。每日流程仍是"取一次数据 →
主策略算一遍 → 副策略算一遍"。

账本状态机（与主策略 paper.json 同构）：
    T 日收盘  -> 写 signal（今日唯一候选）
    T+1 开盘  -> 若在 gap 区间内、未一字涨停/停牌, 用开盘价买入（100% 仓位）
    持有 N 日 -> 第 N 个交易日收盘价卖出（或触发止损）
    空仓      -> 不买, 等下一次信号（"不到标准不成交"）

幂等：updated == 今日 时直接退出, 重复运行不重复买卖。

用法: python tools/satellite_run.py [--rebuild]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context  # noqa: E402
from src.config import load_config  # noqa: E402
from src.satellite import (daily_params, entry_gate, pick_list, pick_one,  # noqa: E402
                           pool_funnel, rank_pool, rank_qualified, score_of,
                           should_exit, STRATEGY_CN)

# 面板窗口: 主窗口 100 日 + 预热 160 日 = 260 日。
# 必须是 251 日以上: build_indicators 的 bars 是**窗口内**累计行号, 不是真实上市天数,
# 而 MIN_BARS=250 —— 窗口恰好 250 日时最后一行的 bars 只有 249, 全市场会被判成"上市不足",
# 票池归零(2026-11-06 实测: 90+160=250 日窗口下 base=0、候选池 0 只)。留 10 日余量。
DAYS = 100
OUT = ROOT / "output" / "satellite.json"
HALF = 0.0015          # 单边成本 0.15%（佣金万2.5 + 印花税万5 的一半 + 千1滑点）


def _load() -> dict:
    if OUT.exists():
        try:
            return json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _fmt(code: str) -> str:
    return code


def _px(day, code):
    """返回 (收盘价, 原始收盘价, 开盘价, 原始开盘价, 是否可交易)。"""
    r = day.loc[day["ts_code"] == code] if code else None
    if r is None or not len(r):
        return None
    r = r.iloc[0]
    if bool(r["is_paused"] == 1) or float(r["vol"]) <= 0:
        return None
    return (float(r["close_adj"]), float(r["close"]), float(r["open_adj"]),
            float(r["open"]), float(r["pre_close"]), float(r["high_limit"]),
            float(r["low_limit"]))


def _f(v, nd=2, default=None):
    try:
        x = float(v)
    except Exception:
        return default
    return None if x != x else round(x, nd)       # NaN -> default


def _pool_snapshot(day, cfg, names: dict, altn: int = None) -> dict:
    """当日候选池快照(第一百五十四轮) —— 低波池里风险调整动量最强的票, 最多 3 只。

    口径(与页面 legend / 漏斗逐字对应, 不各算各的):
      base   = 基础票池: 通过上市/流动性/板块等硬性条件的只数(rank_pool 的池子)
      rows   = 候选池: 基础池里 ivol 最低 10%(池边界) → 按 mom20/ivol(风险调整动量)
               最强排序, 不设涨幅上限(追高只追在低波池里), 最多 pool_show 只
      n      = 候选池只数(= len(rows)); 基础票池不足 min_pool 或全部不合格时为 0
      alt    = 已废弃, 恒空数组
      funnel = 全市场 → 上市状态 → 成交额 → 板块 的逐道剩余只数

    执行: 最多同时持 max_pos 只, 每日最多买 buys_per_day 只(第一百六十三轮起 =max_pos,
          即"一次买满") —— 候选池前 buys_per_day 名就是执行标的,
    第 3 名是候补(持仓满员时不买)。
    """
    P = daily_params(cfg)
    fun = pool_funnel(day, cfg)
    base = int(fun[-1]["n"])
    rk = rank_pool(day, cfg)          # 不足 min_pool 时返回空表
    q = rank_qualified(day, cfg, rk)  # 评分达标的前 N 名

    def _row(i, r):
        code = str(r["ts_code"])
        return {"rank": i, "code": code, "name": names.get(code, ""),
                "ivol20": _f(r["_ivol"], 3), "score": round(score_of(r["_ivol"]), 1),
                "dist_ma5": _f(r["dist_ma5"]),
                "pct": _f(r["pct"]),
                "close": _f(r["close"]), "amount_yi": _f(float(r["amount"]) / 1e8),
                "vol_ratio": _f(r["vol_ratio"]), "turnover_rate": _f(r["turnover_rate"]),
                "mom20": _f(r["mom20"], 1),
                # 风险调整动量(第一百五十四轮排序键): 页面直接展示这个数, 与后端排序同源
                "ra": _f(float(r["mom20"]) / float(r["_ivol"]) if float(r["_ivol"]) else None, 2)}

    rows = [_row(i, r) for i, (_, r) in enumerate(q.head(P["pool_show"]).iterrows(), 1)]
    return {"n": len(rows), "base": base, "show": P["pool_show"],
            "alt": [], "funnel": fun, "rows": rows,
            "score_gate": round(P["min_score"], 1),
            "ivol_cap": round(P["ivol_cap"], 3)}


def _migrate(tape: dict, day, today: str) -> float:
    """v2(40 只低波批次) -> v3(单吊): 按今日收盘价全部清仓, 净值结转。"""
    batches = tape.get("batches") or []
    cash = float(tape.get("cash") or 0.0)
    left = [b for b in batches if b.get("status") == "holding"]
    closed = []
    for b in left:
        p = _px(day, b["code"])
        if not p:
            continue
        proceeds = float(b.get("shares") or 0) * p[0] * (1 - HALF)
        cash += proceeds
        closed.append({"code": b["code"], "name": b.get("name", ""),
                       "entry_date": b.get("signal_date"), "exit_date": today,
                       "entry_px": b.get("entry_px_raw"), "exit_px": round(p[1], 3),
                       "ret_pct": round((p[0] / float(b.get("entry_px") or p[0]) - 1) * 100, 2),
                       "note": "v2 归档"})
        b["status"] = "closed"
        b["exit_date"] = today
    tape["legacy_batches"] = len(left)
    tape["legacy_trades"] = (tape.get("legacy_trades") or []) + closed
    print("  迁移 v2: 清仓 %d 只, 结转现金 %.2f" % (len(closed), cash), flush=True)
    return cash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    P = daily_params(cfg)
    tape = {} if args.rebuild else _load()

    print("[1/2] 加载面板(与主策略共用缓存) %d 日 ..." % DAYS, flush=True)
    ctx = prepare_context(cfg, DAYS)
    by_date = ctx["by_date"]
    dates = sorted(by_date)
    main_dates = ctx["main_dates"]
    names = ctx.get("names") or {}
    today = main_dates[-1]
    day = by_date[today]

    # ---------- 初始化 / 迁移 ----------
    ver = int(tape.get("version") or 0)
    if ver < 3:
        # v2 的 capital 是"当前总资产", init_capital 才是本金 —— 取后者, 否则净值会被归一化错
        init_cap = float(tape.get("init_capital") or 100000.0)
        cash = _migrate(tape, day, today) if ver == 2 else init_cap
        tape = {"version": 3, "strategy": "trend_single", "engine": "single",
                "init_capital": init_cap, "capital": init_cap, "cash": cash,
                "nav": round(cash / init_cap, 4) if init_cap else 1.0,
                "signal": None, "positions": [], "trades": [],
                "daily": [], "legacy_batches": tape.get("legacy_batches", 0),
                "legacy_trades": tape.get("legacy_trades") or [],
                "updated": None}
        print("  v3 初始化: 本金 %.0f, 现金 %.2f" % (init_cap, cash), flush=True)

    # 第一百五十一轮: 单持仓(position) -> 多持仓(positions)。老账本最多只有 1 只,
    # 直接搬进数组即可; 空仓就是空数组。
    if "positions" not in tape:
        _p = tape.pop("position", None)
        tape["positions"] = [_p] if _p else []

    if tape.get("updated") == today:
        # 幂等跳过, 但有一个例外: 候选池快照是"展示层", 口径变过一次(第一百四十轮新增 pool,
        # 第一百四十二轮改成 base/show/funnel + 20 行)。旧快照缺字段时重算一次 ——
        # 只补写展示字段, 状态机一律不动(重跑状态机会重复记账)。
        _snap = tape.get("pool") or {}
        # 展示格式升级(第一百四十三轮): 老快照可能是 20 行候选池 / 缺 alt 字段。
        # 只在格式对不上时重算展示层, 状态机一律不动 —— 重跑状态机会重复记账。
        if (_snap.get("date") != today or "funnel" not in _snap
                or "alt" not in _snap or _snap.get("show") != P["pool_show"]
                or _snap.get("score_gate") != round(P["min_score"], 1)
                or (tape.get("params") or {}).get("hold_days") != P["hold_days"]
                or (tape.get("params") or {}).get("max_pos") != P["max_pos"]
                or (tape.get("params") or {}).get("buys_per_day") != P["buys_per_day"]
                or (tape.get("params") or {}).get("dist_ma5_max") != P["dist_ma5_max"]
                or (tape.get("params") or {}).get("sort_by") != P["sort_by"]
                # 第一百八十轮: 持仓闸门开关也是"披露口径", 老账本没有该字段 -> 补写一次
                or (tape.get("params") or {}).get("gate_exit") != P["gate_exit"]
                # 第一百八十一轮: 入场缓冲也是披露口径, 老账本没有该字段 -> 补写一次
                or (tape.get("params") or {}).get("gate_entry_buffer") != round(P["gate_entry_buf"], 2)
                or "min_score" not in (tape.get("params") or {})
                or "min_pool" not in (tape.get("params") or {})):
            same_day = _snap.get("date") == today
            snap = _pool_snapshot(day, cfg, names)
            snap["date"] = today
            gate_ok = True
            reg = ctx.get("reg")
            if P["gate"] and reg is not None and today in reg.index:
                gate_ok = bool(reg.loc[today, "trade_ok"])
            # 同一天只升级展示格式时, 沿用旧快照里已有的判定/主选, 不重新推断
            snap["gate_ok"] = _snap.get("gate_ok", gate_ok) if same_day else gate_ok
            # 第一百八十一轮: 入场口径随快照补写(旧快照没有该字段时页面会退回 gate_ok)
            _entry0 = gate_ok
            _b0 = float(P.get("gate_entry_buf") or 0.0)
            if gate_ok and _b0 > 0 and reg is not None and today in reg.index:
                _r0 = reg.loc[today]
                _entry0 = bool(float(_r0["mkt_idx"]) >= float(_r0["mkt_ma"]) * (1 + _b0 / 100.0))
            snap["entry_ok"] = bool(_entry0)
            snap["picked"] = (_snap.get("picked") if same_day else "") or (
                (tape.get("signal") or {}).get("code")
                or ((tape.get("positions") or [{}])[0] or {}).get("code") or "")
            snap["pick_note"] = _snap.get("pick_note", "") if same_day else ""
            tape["pool"] = snap
            # 池口径参数同样补写: 老账本的 params 没有这三个字段, 若只升级 pool 不升级 params,
            # 页面就得靠前端 fallback 猜"样本下限多少" —— 配置一改就成假披露。
            _pr = dict(tape.get("params") or {})
            _pr.update({"min_pool": P["min_pool"], "pool_show": P["pool_show"],
                        "pool_alt": P["pool_alt"], "min_score": round(P["min_score"], 1),
                        "ivol_cap": round(P["ivol_cap"], 3), "hold_days": P["hold_days"],
                        "max_pos": P["max_pos"], "buys_per_day": P["buys_per_day"],
                        "dist_ma5_max": P["dist_ma5_max"],
                        "sort_by": P["sort_by"], "ivol_pct_max": P["ivol_pct"],
                        "gate_exit": bool(P["gate_exit"]), "gate_exit_mode": P["gate_exit_mode"],
                        "gate_entry_buffer": round(P["gate_entry_buf"], 2)})
            tape["params"] = _pr
            OUT.write_text(json.dumps(tape, ensure_ascii=False, indent=1), encoding="utf-8")
            print("  升级候选池快照(候选池 %d 只 / 评分阈值 %g / 基础票池 %d 只), 状态机未动"
                  % (snap["n"], P["min_score"], snap["base"]), flush=True)
        print("  今日(%s)已更新, 跳过" % today, flush=True)
        return

    cash = float(tape.get("cash") or 0.0)
    positions = tape.get("positions") or []
    sig = tape.get("signal")
    # 第一百六十三轮: buys_per_day>1 时, 信号是一个列表(一次买满 max_pos 只)。
    # tape["signal"] 仍保留第一只 —— 页面/报告/旧账本都按单只读, 不能改动它们的契约。
    sigs = tape.get("signals")
    if sigs is None:
        sigs = [sig] if sig else []
    trades = tape.get("trades") or []
    daily = tape.get("daily") or []
    params = {"mode": P["mode"], "mode_cn": STRATEGY_CN.get(P["mode"], P["mode"]),
              "hold_days": P["hold_days"], "max_ivol": P["max_ivol"],
              # 第一百四十九轮: 评分阈值(min_score)与换算后的 ivol 上限(ivol_cap)一起发布,
              # 页面"评分 ≥ X 达标"必须与选股同一口径, 不能在页面里另写一个数。
              "min_score": round(P["min_score"], 1), "ivol_cap": round(P["ivol_cap"], 3),
              "amount_min_yi": round(P["amt_min"] / 1e8, 2),
              "gap_min": P["gap_min"], "gap_max": P["gap_max"],
              "stop_pct": round(P["stop"] * 100, 2), "market_gate": P["gate"],
              # 第一百五十五轮: 移动止损随参数发布, 页面"自高点回撤 X% 即卖"与配置同源
              "trail_pct": round(P["trail"] * 100, 2),
              # 第一百八十轮: 持仓闸门随参数发布 —— 否则页面只看到"大盘闸门"(开新仓),
              # 不知道"闸门关闭时还会清老仓", 账面回撤会被误读成"没风控"。
              "gate_exit": bool(P["gate_exit"]), "gate_exit_mode": P["gate_exit_mode"],
              # 第一百八十一轮: 入场缓冲随参数发布 —— 页面"闸门开=可买"要能解释
              # "闸门开了但指数只高 0.1%, 仍不追"的情况。
              "gate_entry_buffer": round(P["gate_entry_buf"], 2),
              "cost_pct": round(P["cost"] * 100, 3),
              # 池口径随参数一起发布: 页面 legend 的"候选池 N 只 / 样本下限 M 只"必须
              # 跟配置一致, 不能在页面里写死 20 —— 改了 config 页面不跟着变就成了假披露。
              "min_pool": P["min_pool"], "pool_show": P["pool_show"], "pool_alt": P["pool_alt"],
              # 第一百五十一轮: 并发上限与形态阈值随参数发布, 页面"最多 N 只""跌破 MA5 X%"
              # 必须跟配置同源 —— 改配置不改页面就是假披露。
              "max_pos": P["max_pos"], "buys_per_day": P["buys_per_day"],
              "dist_ma5_max": P["dist_ma5_max"],
              "sort_by": P["sort_by"], "ivol_pct_max": P["ivol_pct"],
              # 第一百七十九轮: 回踩限价随参数发布 —— 页面"挂单 -2%"必须与执行同源
              "pullback_pct": P["pullback_pct"], "pullback_days": P["pullback_days"],
              "pullback_mode": P["pullback_mode"]}
    tape["params"] = params

    action, note, code_today = "none", "", ""

    # ---------- 1) T+1 开盘: 执行昨日信号(每日最多 buys_per_day 笔, 持仓上限 max_pos) ----------
    bought = 0
    for _s in sigs:
        if len(positions) >= P["max_pos"]:
            break
        code_today = _s["code"]
        p = _px(day, code_today)
        if p is None:
            note = note or "停牌, 未成交"
            continue
        _cl, _clr, _op, op_raw, pre, hi_lim, _lo = p
        gap = (op_raw / pre - 1) * 100 if pre else 0.0
        if op_raw >= hi_lim - 1e-9:
            note = note or "一字涨停, 买不到"
            continue
        ok, why = entry_gate(gap, cfg)
        if not ok:
            note = note or why
            continue
        # 第一百七十九轮: 回踩限价。挂"信号日收盘 ×(1-pb%)"的买单, 盘中触及才成交。
        # 账本按日推进, pullback_days=1 时挂单当日有效, 因此不需要跨日队列。
        _fill_adj = _op
        _ratio = 1.0
        if P["pullback_pct"] > 0:
            _lim = float(_s.get("limit_adj") or 0.0)
            if _lim <= 0:
                note = note or "信号缺少限价基准, 不买"
                continue
            _rr = day.loc[day["ts_code"] == code_today].iloc[0]
            _o_raw = float(_rr["open"])
            _ratio = _op / _o_raw if _o_raw else 1.0
            _low_adj = float(_rr["low"]) * _ratio
            if _op <= _lim + 1e-9:
                _fill_adj = _op                      # 开盘已低于限价 -> 开盘成交
            elif P["pullback_mode"] != "open" and _low_adj <= _lim + 1e-9:
                _fill_adj = _lim                     # 盘中回踩触及 -> 限价成交
            else:
                note = note or ("未回踩到 -%g%% 限价(最低 %+.2f%%), 今日不买"
                                % (P["pullback_pct"], (_low_adj / _cl - 1) * 100))
                continue
        px = _fill_adj * (1 + HALF)
        # 单笔预算 = 总资产 / max_pos。总资产用"现金 + 已有持仓成本"估算:
        # 买入发生在开盘, 当日收盘价还不可知, 用成本估值不引入未来函数。
        budget = min((cash + sum(float(x.get("cost") or 0) for x in positions))
                     / P["max_pos"], cash * 0.995)
        if budget < 1000:
            note = note or "现金不足, 不买入"
            continue
        shares = budget / px
        pos = {"code": code_today, "name": names.get(code_today, _s.get("name", "")),
               "signal_date": _s["date"], "entry_date": today,
               "entry_px": round(px, 4),
               "entry_px_raw": round(_fill_adj / _ratio, 3),
               "gap_pct": round(gap, 2), "shares": round(shares, 4),
               "cost": round(budget, 2), "hold_days": P["hold_days"],
               # 第一百五十五轮: 移动止损需要"持仓期最高收盘价"(复权口径, 与 entry_px 同源)
               "hwm": round(px, 4),
               "status": "holding"}
        cash -= budget
        positions.append(pos)
        tape["positions"] = positions
        action = "buy"
        note = "开盘 %.3f 买入" % op_raw if not note else note
        print("  买入 %s %s @%.3f (缺口 %+.2f%%, 预算 %.0f)"
              % (code_today, pos["name"], op_raw, gap, budget), flush=True)
        bought += 1
    if bought:
        tape["signal"] = None
        tape["signals"] = None

    # 第一百八十轮: 大盘闸门提前算 —— 持仓清仓判定与"明日是否开新仓"必须用同一个值,
    # 否则会出现"实盘因闸门不开新仓、但老仓位不清"的行为, 与回测口径不一致。
    gate_ok, entry_ok, gate_note = True, True, ""
    _gate_missing = False
    _buf = float(P.get("gate_entry_buf") or 0.0)
    if P["gate"]:
        reg = ctx.get("reg")
        if reg is not None and today in reg.index:
            gate_ok = bool(reg.loc[today, "trade_ok"])
            entry_ok = gate_ok
            if gate_ok and _buf > 0:
                # 第一百八十一轮: 入场缓冲 —— 与 src.satellite.entry_ok_series 同一口径。
                # 只在"闸门已开"之后再收紧, 所以不会放大风险, 只过滤"刚翻多就买"。
                _rr = reg.loc[today]
                entry_ok = bool(float(_rr["mkt_idx"]) >=
                                float(_rr["mkt_ma"]) * (1 + _buf / 100.0))
        elif reg is not None:
            # 第 192 轮: 旧实现在 "today 不在 reg.index" 时让 gate_ok 保持默认 True ——
            # 数据没就绪反而放行, 是方向性错误。改成显式拒绝, 并在 note 里写明原因。
            gate_ok = entry_ok = False
            _gate_missing = True
        if not gate_ok:
            gate_note = (("风控序列未覆盖 %s(数据未就绪), 今日按不开新仓处理" % today)
                         if _gate_missing
                         else "大盘闸门关闭(指数在 MA20 下方 / 宽度不足), 不开新仓")
        elif not entry_ok:
            _r0 = reg.loc[today]
            gate_note = ("大盘闸门虽开, 但指数只在 MA20 上方 %.2f%%(< %.2f%% 入场缓冲), "
                         "今日不追新仓" % ((float(_r0["mkt_idx"]) / float(_r0["mkt_ma"]) - 1) * 100, _buf))

    # ---------- 2) 收盘: 逐只卖出判定 ----------
    for pos in list(positions):
        code_today = pos["code"]
        p = _px(day, code_today)
        if p is None:
            note = (note + " %s 停牌" % pos.get("name", code_today)).strip()
        else:
            cl, clr, *_ = p
            ret = (cl / float(pos["entry_px"]) - 1) * 100
            # 移动止损: 先按当日收盘刷新持仓期最高价, 再判定(创新高当日不会触发)
            pos["hwm"] = round(max(float(pos.get("hwm") or pos["entry_px"]), cl), 4)
            held = dates.index(today) - dates.index(pos["entry_date"]) if pos["entry_date"] in dates else 0
            ex, why = should_exit(held, ret, cfg, hwm=pos["hwm"], close=cl, gate_ok=gate_ok)
            if ex:
                proceeds = float(pos["shares"]) * cl * (1 - HALF)
                cash += proceeds
                trades.append({"code": code_today, "name": pos.get("name", ""),
                               "signal_date": pos.get("signal_date"),
                               "entry_date": pos["entry_date"], "exit_date": today,
                               "entry_px": pos.get("entry_px_raw"), "exit_px": round(clr, 3),
                               "hold_days": held, "ret_pct": round(ret, 2), "note": why})
                positions.remove(pos)
                action = "sell"
                note = (note + " %s %s" % (pos.get("name", code_today), why)).strip()
                print("  卖出 %s %s @%.3f %+.2f%% (%s)"
                      % (code_today, pos.get("name", ""), clr, ret, why), flush=True)
    tape["positions"] = positions

    # ---------- 3) 收盘: 选明日信号(大盘闸门已在上面算好) ----------
    if len(positions) < P["max_pos"] and not entry_ok:
        tape["signal"] = None
        tape["signals"] = None
        action, note = "none", gate_note
    elif len(positions) < P["max_pos"]:
        # 第一百六十三轮: 一次取满 buys_per_day 只。取 max(...) 是为了让"信号宽度门槛"
        # 仍能判定(候选不足 min_cands 就不出票), 与回测 pick_list(k=max(buys,min_cands)) 同口径。
        cands = pick_list(day, cfg, held=[x["code"] for x in positions],
                          k=max(P["buys_per_day"], P["min_cands"]))
        if len(cands) < P["min_cands"]:
            cands = []
        else:
            cands = cands[:P["buys_per_day"]]
        for _c in cands:
            _c["date"] = today
            _c["name"] = names.get(_c["code"], "")
            if P["pullback_pct"] > 0:
                # 限价基准锁在**信号日收盘**(今日已知), 次日执行时不再读新行情
                _r0 = day.loc[day["ts_code"] == _c["code"]].iloc[0]
                _c["limit_adj"] = round(float(_r0["close_adj"]) * (1 - P["pullback_pct"] / 100.0), 4)
        tape["signals"] = cands
        cand = cands[0] if cands else None
        if cand:
            cand["date"] = today
            cand["name"] = names.get(cand["code"], "")
            tape["signal"] = cand
            code_today = code_today or cand["code"]
            if action == "none":
                action = "signal"
                note = note or "低波池内风险调整动量最强(20日 %+.1f%% / 波动 %.2f / 距MA5 %+.2f%%), 明日开盘买入" % (
                    cand["mom20"], cand["ivol20"], cand["dist_ma5"])
            print("  信号 %s %s (mom20 %+.1f%%, dist_ma5 %+.2f%%, ivol %.2f)"
                  % (cand["code"], cand["name"], cand["mom20"], cand["dist_ma5"],
                     cand["ivol20"]), flush=True)
        else:
            tape["signal"] = None
            tape["signals"] = None
            note = note or "候选池不足/无符合条件的票(低波前 %g%%), 明日不买" % (
                P["ivol_pct"] * 100)
    else:
        tape["signal"] = None
        tape["signals"] = None

    # ---------- 3.5) 候选池快照(页面用; 与是否成交无关) ----------
    snap = _pool_snapshot(day, cfg, names)
    snap["date"] = today
    held_code = (positions[0] or {}).get("code") if positions else ""
    sig_code = (tape.get("signal") or {}).get("code")
    snap["picked"] = sig_code or held_code or ""
    snap["gate_ok"] = bool(gate_ok)
    snap["entry_ok"] = bool(entry_ok)   # 第一百八十一轮: 闸门开 != 允许开新仓
    snap["pick_note"] = note
    tape["pool"] = snap

    # ---------- 4) 净值 ----------
    mv = 0.0
    for pos in positions:
        p = _px(day, pos["code"])
        if p:
            mv += float(pos["shares"]) * p[0]
            pos["held_days"] = dates.index(today) - dates.index(pos["entry_date"]) \
                if pos["entry_date"] in dates else 0
            pos["px"] = round(p[1], 3)
            pos["ret_pct"] = round((p[0] / float(pos["entry_px"]) - 1) * 100, 2)
        else:
            mv += float(pos.get("cost") or 0)
    nav = (cash + mv) / float(tape["init_capital"])
    tape["cash"] = round(cash, 2)
    tape["nav"] = round(nav, 4)
    tape["capital"] = round(cash + mv, 2)
    tape["trades"] = trades
    daily.append({"date": today, "nav": round(nav, 4), "action": action,
                  "code": code_today, "note": note})
    tape["daily"] = daily[-500:]
    tape["updated"] = today

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(tape, ensure_ascii=False, indent=1), encoding="utf-8")
    print("  净值 %.4f · 现金 %.2f · 持仓 %s · %s"
          % (nav, cash, ",".join(x["code"] for x in positions) or "空", note), flush=True)
    print("  写出 %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
