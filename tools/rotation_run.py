# -*- coding: utf-8 -*-
"""副策略 v4(低波组合轮动)每日账本。

与 v3(单吊)的区别: 账本里是 N 只持仓而不是 1 只; "信号"是一张调仓清单
(卖哪些/买哪些/留哪些), 而不是一只票。

时点(与回测逐条对齐, 无未来函数):
  T 日收盘 → 按 ivol 升序取前 n_hold 只 → 生成调仓清单
  T+1 开盘 → 按开盘价执行(出榜的卖、新进榜的买、留榜的不动)
  净值按收盘价估值; 单边成本 0.15%

用法: python tools/rotation_run.py [--rebuild]
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
from src.rotation import (STRATEGY_CN, diff_basket, funnel, params,  # noqa: E402
                          pick_basket)

DAYS = 300
HALF = 0.0015
OUT = ROOT / "output" / "rotation.json"


def _load() -> dict:
    if OUT.exists():
        try:
            return json.loads(OUT.read_text(encoding="utf-8"))
        except Exception:
            pass
    return {}


def _px(day, code):
    r = day.loc[day["ts_code"] == code]
    if not len(r):
        return None
    r = r.iloc[0]
    if bool(r["is_paused"] == 1) or float(r["vol"]) <= 0:
        return None
    return float(r["close_adj"]), float(r["close"]), float(r["open_adj"]), float(r["open"]), float(r["high_limit"])


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()
    cfg = load_config()
    P = params(cfg)
    tape = {} if args.rebuild else _load()

    print("[1/3] 加载面板 %d 日 ..." % DAYS, flush=True)
    ctx = prepare_context(cfg, DAYS)
    by_date, main_dates = ctx["by_date"], ctx["main_dates"]
    names = ctx.get("names") or {}
    today = main_dates[-1]
    day = by_date[today]
    idx = {d: i for i, d in enumerate(main_dates)}

    if not tape:
        tape = {"version": 1, "strategy": "lowvol_rotation", "engine": "basket",
                "init_capital": 100000.0, "cash": 100000.0, "nav": 1.0,
                "holdings": {}, "pending": {}, "basket": [], "last_rebalance": None,
                "next_rebalance": None,
                "rounds": [], "daily": [], "updated": None}
    # 幂等守卫必须覆盖"本版本写出的所有字段": 只看 updated 会让旧账本(无 funnel/无 params.n_hold)
    # 永远补不上新字段 —— 旧库更新于今天就直接跳过, 页面漏斗便一直只剩最后一步。
    if (tape.get("updated") == today and tape.get("funnel") and "basket" in tape
            and "n_hold" in (tape.get("params") or {})):
        print("  今日(%s)已更新, 跳过" % today, flush=True)
        return

    holds = tape.get("holdings") or {}
    # 上一张清单: pending 会在本次执行后被清空, 先把清单本体捞出来, 保证账本里
    # 永远有一份"当前目标组合"可展示(pending 为空 ≠ 没有目标组合)。
    last_basket = tape.get("basket") or (tape.get("pending") or {}).get("basket") or []
    cash = float(tape.get("cash") or 0.0)
    init = float(tape.get("init_capital") or 100000.0)
    rounds = tape.get("rounds") or []
    daily = tape.get("daily") or []
    note = ""

    # ---------- 1) 执行昨日清单(今日开盘) ----------
    pend = tape.get("pending") or {}
    if pend.get("codes") is not None and pend.get("signal_date") and pend["signal_date"] < today:
        target = list(pend["codes"])
        prev_codes = set(holds)      # 调仓前持仓; "留几只"必须用它算, 买入后再算恒等于 N
        px = {}
        for c in set(list(holds) + target):
            p = _px(day, c)
            if p:
                px[c] = p
        sold, bought, sold_list, buy_list = 0.0, 0.0, [], []
        for c in list(holds):
            if c not in target and c in px:
                amt = float(holds[c]["shares"]) * px[c][2]
                cash += amt * (1 - HALF)
                sold += amt
                sold_list.append({"code": c, "name": holds[c].get("name", ""), "px": round(px[c][3], 3)})
                del holds[c]
        equity = cash + sum(float(h["shares"]) * (px[c][0] if c in px else h.get("last_px", h["entry_px"]))
                            for c, h in holds.items())
        need = [c for c in target if c not in holds and c in px
                and px[c][3] < px[c][4] - 1e-9]
        if need:
            each = equity / max(len(target), 1)
            for c in need:
                budget = min(each, cash)
                if budget <= 0:
                    break
                sh = budget / (px[c][2] * (1 + HALF))
                cash -= budget
                holds[c] = {"shares": sh, "entry_px": px[c][2], "entry_date": today,
                            "entry_px_raw": px[c][3], "name": names.get(c, ""),
                            "last_px": px[c][0]}
                bought += budget
                buy_list.append({"code": c, "name": names.get(c, ""), "px": round(px[c][3], 3)})
        turn = (sold + bought) / equity if equity else 0.0
        keep_n = len([c for c in target if c in prev_codes])
        rounds.append({"date": today, "sold": sold_list, "bought": buy_list,
                       "keep": keep_n,
                       "turn_pct": round(turn * 100, 1),
                       "hold_n": len(holds)})
        note = "执行调仓: 卖%d 买%d 留%d" % (len(sold_list), len(buy_list), keep_n)
        tape["pending"] = {}

    # ---------- 2) 收盘估值 ----------
    mv = 0.0
    for c, h in holds.items():
        p = _px(day, c)
        if p:
            h["last_px"] = p[0]
        mv += float(h["shares"]) * h.get("last_px", h["entry_px"])
    nav = (cash + mv) / init if init else 1.0

    # ---------- 3) 生成下一次清单 ----------
    # 用"上次调仓日 + 交易日计数"判断, 不能看 next_rebalance: 它的日期一旦落在
    # 已知日历窗口外就是 None, 而 `if not due` 会让它每天都成立(天天调仓)。
    last = tape.get("last_rebalance")
    gap = idx.get(today, 0) - idx.get(last, -9999) if last else 9999
    due = gap >= P["hold_days"]
    if due:
        basket = pick_basket(day, cfg, names)
        if basket:
            codes = [b["code"] for b in basket]
            tape["pending"] = {"signal_date": today, "codes": codes, "basket": basket}
            # 顶层再存一份"最近一次目标组合": pending 执行完就被清空, 页面若无条件读 pending,
            # 次日建仓后就会显示"今日无目标组合"——明明手上正拿着这 20 只。
            tape["basket"] = basket
            tape["last_rebalance"] = today
            nxt = idx.get(today, 0) + P["hold_days"]
            # 只用于展示; 窗口外写 None, 判定一律走 last_rebalance(见上)
            tape["next_rebalance"] = main_dates[nxt] if nxt < len(main_dates) else None
            d = diff_basket([c for c in holds], codes)
            note = (note + "；" if note else "") + \
                "生成调仓清单(卖%d 买%d 留%d)" % (len(d["sell"]), len(d["buy"]), len(d["keep"]))
        else:
            note = (note + "；" if note else "") + "候选池不足 %d 只, 不出清单" % P["min_pool"]

    tape.update({"holdings": holds, "cash": round(cash, 2), "nav": round(nav, 4),
                 "basket": tape.get("basket") or last_basket,
                 "updated": today,
                 "rounds": rounds[-60:],
                 "daily": (daily + [{"date": today, "holding_n": len(holds),
                                     "nav": round(nav, 4), "note": note}])[-260:],
                 "funnel": funnel(day, cfg),
                 "params": {"strategy_cn": STRATEGY_CN, "n_hold": P["n_hold"],
                            "hold_days": P["hold_days"],
                            "amount_min_yi": round(P["amt_min"] / 1e8, 2),
                            "min_pool": P["min_pool"]}})
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(tape, ensure_ascii=False, indent=1), encoding="utf-8")
    print("  %s 持仓 %d 只 现金 %.0f 净值 %.4f %s" % (today, len(holds), cash, nav, note), flush=True)


if __name__ == "__main__":
    main()
