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
from src.satellite import (daily_params, entry_gate, pick_one,  # noqa: E402
                           rank_pool, should_exit, STRATEGY_CN)

DAYS = 90
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
                "signal": None, "position": None, "trades": [],
                "daily": [], "legacy_batches": tape.get("legacy_batches", 0),
                "legacy_trades": tape.get("legacy_trades") or [],
                "updated": None}
        print("  v3 初始化: 本金 %.0f, 现金 %.2f" % (init_cap, cash), flush=True)

    if tape.get("updated") == today:
        print("  今日(%s)已更新, 跳过" % today, flush=True)
        return

    cash = float(tape.get("cash") or 0.0)
    pos = tape.get("position")
    sig = tape.get("signal")
    trades = tape.get("trades") or []
    daily = tape.get("daily") or []
    params = {"mode": P["mode"], "mode_cn": STRATEGY_CN.get(P["mode"], P["mode"]),
              "hold_days": P["hold_days"], "max_ivol": P["max_ivol"],
              "amount_min_yi": round(P["amt_min"] / 1e8, 2),
              "gap_min": P["gap_min"], "gap_max": P["gap_max"],
              "stop_pct": round(P["stop"] * 100, 2), "market_gate": P["gate"],
              "cost_pct": round(P["cost"] * 100, 3)}
    tape["params"] = params

    action, note, code_today = "none", "", ""

    # ---------- 1) T+1 开盘: 执行昨日信号 ----------
    if sig and not pos:
        code_today = sig["code"]
        p = _px(day, code_today)
        if p is None:
            note = "停牌, 未成交"
            tape["signal"] = None
        else:
            _cl, _clr, _op, op_raw, pre, hi_lim, _lo = p
            gap = (op_raw / pre - 1) * 100 if pre else 0.0
            if op_raw >= hi_lim - 1e-9:
                note = "一字涨停, 买不到"
                tape["signal"] = None
            else:
                ok, why = entry_gate(gap, cfg)
                if not ok:
                    note = why
                    tape["signal"] = None
                else:
                    px = _op * (1 + HALF)
                    shares = cash / px
                    pos = {"code": code_today, "name": names.get(code_today, sig.get("name", "")),
                           "signal_date": sig["date"], "entry_date": today,
                           "entry_px": round(px, 4), "entry_px_raw": round(op_raw, 3),
                           "gap_pct": round(gap, 2), "shares": round(shares, 4),
                           "cost": round(cash, 2), "hold_days": P["hold_days"],
                           "status": "holding"}
                    cash = 0.0
                    tape["position"] = pos
                    tape["signal"] = None
                    action = "buy"
                    note = note or "开盘 %.3f 买入" % op_raw
                    print("  买入 %s %s @%.3f (缺口 %+.2f%%)" % (code_today, pos["name"], op_raw, gap), flush=True)

    # ---------- 2) 收盘: 卖出判定 ----------
    pos = tape.get("position")
    if pos:
        code_today = pos["code"]
        p = _px(day, code_today)
        if p is None:
            note = (note + " 持仓停牌").strip()
        else:
            cl, clr, *_ = p
            ret = (cl / float(pos["entry_px"]) - 1) * 100
            held = dates.index(today) - dates.index(pos["entry_date"]) if pos["entry_date"] in dates else 0
            ex, why = should_exit(held, ret, cfg)
            if ex:
                proceeds = float(pos["shares"]) * cl * (1 - HALF)
                cash += proceeds
                trades.append({"code": code_today, "name": pos.get("name", ""),
                               "signal_date": pos.get("signal_date"),
                               "entry_date": pos["entry_date"], "exit_date": today,
                               "entry_px": pos.get("entry_px_raw"), "exit_px": round(clr, 3),
                               "hold_days": held, "ret_pct": round(ret, 2), "note": why})
                action = "sell"
                note = (note + " " + why).strip()
                tape["position"] = None
                print("  卖出 %s %s @%.3f %+.2f%% (%s)" % (code_today, pos.get("name", ""), clr, ret, why), flush=True)

    # ---------- 3) 收盘: 选明日信号(先过大盘闸门) ----------
    gate_ok, gate_note = True, ""
    if P["gate"]:
        reg = ctx.get("reg")
        if reg is not None and today in reg.index:
            gate_ok = bool(reg.loc[today, "trade_ok"])
        if not gate_ok:
            gate_note = "大盘闸门关闭(指数在 MA20 下方 / 宽度不足), 不开新仓"
    if not tape.get("position") and not gate_ok:
        tape["signal"] = None
        action, note = "none", gate_note
    elif not tape.get("position"):
        cand = pick_one(day, cfg)
        if cand:
            cand["date"] = today
            cand["name"] = names.get(cand["code"], "")
            tape["signal"] = cand
            code_today = code_today or cand["code"]
            if action == "none":
                action = "signal"
                note = note or "ivol %.2f, 明日开盘买入" % cand["ivol20"]
            print("  信号 %s %s (ivol %.2f)" % (cand["code"], cand["name"], cand["ivol20"]), flush=True)
        else:
            tape["signal"] = None
            note = note or "候选池不足/无合格票, 明日空仓"
    else:
        tape["signal"] = None

    # ---------- 4) 净值 ----------
    pos = tape.get("position")
    mv = 0.0
    if pos:
        p = _px(day, pos["code"])
        if p:
            mv = float(pos["shares"]) * p[0]
            pos["held_days"] = dates.index(today) - dates.index(pos["entry_date"]) \
                if pos["entry_date"] in dates else 0
            pos["px"] = round(p[1], 3)
            pos["ret_pct"] = round((p[0] / float(pos["entry_px"]) - 1) * 100, 2)
        else:
            mv = float(pos.get("cost") or 0)
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
          % (nav, cash, (pos or {}).get("code", "空"), note), flush=True)
    print("  写出 %s" % OUT, flush=True)


if __name__ == "__main__":
    main()
