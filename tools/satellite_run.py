# -*- coding: utf-8 -*-
"""独立副策略每日维护 v2: 每天推一只低波票, 持有 40 个交易日(2026-11-05)。

账本 output/satellite.json 结构(v2)
-----------------------------------
{
 "strategy": "lowvol_daily", "version": 2,
 "init_capital": 100000, "cash": 12000.5, "nav": 1.03, "capital": 103000,
 "params": {"hold_days":40, ...},
 "batches": [ {id, signal_date, code, name, ivol20, entry_px, entry_px_raw,
               shares, cost, exp_date, status, exit_date, exit_px, ret_pct} ],
 "daily":   [ {date, code, name, ivol20, action, note} ],   # 最近 120 条
 "legacy_records": [ ...v1 月度记录, 原样保留... ],
 "updated": "20261105"
}

三条铁律: 只增不改(已结算批次不改)、同日幂等、不碰 config/latest/paper。

用法
----
  python3 -m tools.satellite_run              # 每日维护
  python3 -m tools.satellite_run --rebuild    # 丢弃 v2 账本重建(调试用)
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
from src.satellite import daily_params, pick_one  # noqa: E402

TAPE = ROOT / "output" / "satellite.json"
INIT_CAP = 100000.0
DAYS = 400          # 40 日持有 + 300 余日缓冲, 足够算到期日
HALF = 0.0015       # 单边成本(往返 0.3% 的一半)


def _load() -> dict:
    if TAPE.exists():
        try:
            d = json.loads(TAPE.read_text(encoding="utf-8"))
            if isinstance(d, dict):
                return d
        except Exception:  # noqa: BLE001
            pass
    return {"strategy": "lowvol_daily", "version": 2, "init_capital": INIT_CAP,
            "cash": INIT_CAP, "nav": 1.0, "capital": INIT_CAP,
            "params": {}, "batches": [], "daily": [], "legacy_records": [],
            "updated": None}


def _px(day):
    """code -> (close_adj, close)"""
    return {str(c): (float(a), float(b)) for c, a, b in
            zip(day["ts_code"], day["close_adj"], day["close"])}


def _migrate_v1(tape: dict, day, today: str) -> float:
    """把 v1 月度持仓按今日收盘归档, 返回可用现金。"""
    recs = tape.get("records") or []
    px = _px(day)
    cash = INIT_CAP
    if recs:
        last = recs[-1]
        if last.get("status") == "holding":
            entry_nav = float(last.get("entry_nav") or last.get("nav") or 1.0)
            r = 0.0
            wsum = 0.0
            for h in last.get("holdings") or []:
                p = px.get(str(h.get("code")))
                if not p or not h.get("entry_px"):
                    continue
                r += float(h.get("weight") or 0) * (p[0] / float(h["entry_px"]) - 1)
                wsum += float(h.get("weight") or 0)
            if wsum > 0:
                r = r / wsum
            nav = entry_nav * (1 + r)
            last["status"] = "closed"
            last["exit_date"] = today
            last["exit_nav"] = round(nav, 6)
            last["exit_ret_pct"] = round((nav / max(1e-9, last.get("prev_nav", 1.0)) - 1) * 100, 2)
            last["archived_by"] = "v2 日频切换"
            cash = INIT_CAP * nav * (1 - HALF)
            print("  归档 v1 %s: 净值 %.4f (%+.2f%%), 转出现金 %.0f"
                  % (last.get("month"), nav, last["exit_ret_pct"], cash), flush=True)
        tape["legacy_records"] = recs
    tape.pop("records", None)
    return cash


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--rebuild", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    P = daily_params(cfg)
    tape = _load()
    if args.rebuild:
        tape = {"strategy": "lowvol_daily", "version": 2, "init_capital": INIT_CAP,
                "cash": INIT_CAP, "nav": 1.0, "capital": INIT_CAP,
                "params": {}, "batches": [], "daily": [], "legacy_records": [], "updated": None}

    print("[1/2] 加载面板 %d 日 ..." % DAYS, flush=True)
    ctx = prepare_context(cfg, DAYS)
    by_date, dates = ctx["by_date"], sorted(ctx["by_date"])
    main_dates = ctx["main_dates"]
    names = ctx.get("names") or {}
    today = main_dates[-1]
    day = by_date[today]
    px = _px(day)

    # ---- 版本迁移 ----
    if int(tape.get("version") or 1) < 2:
        tape["cash"] = _migrate_v1(tape, day, today)
        tape["version"] = 2
        tape["strategy"] = "lowvol_daily"
    cash = float(tape.get("cash", INIT_CAP))
    batches = tape.get("batches") or []
    daily = tape.get("daily") or []

    book = {"hold_days": P["hold_days"], "cost_pct": round(P["cost"] * 100, 3),
            "amount_min_yi": P["amt_min"] / 1e8, "ma20_filter": P["ma20_filter"],
            "universe": "非ST/上市满250日/成交额>=1亿"}
    tape["params"] = book

    # ---- 1) 结算到期批次 ----
    n_close = 0
    for b in batches:
        if b.get("status") != "holding":
            continue
        hd = int(b.get("hold_days") or P["hold_days"])
        elapsed = _held_days(dates, b.get("signal_date"), today)
        if elapsed is None or elapsed < hd:
            continue
        p = px.get(b["code"])
        if not p:
            continue                      # 停牌/退市: 顺延到有价格的日子
        proceeds = float(b["shares"]) * p[0] * (1 - HALF)
        cash += proceeds
        b["status"] = "closed"
        b["exit_date"] = today
        b["exit_px"] = round(p[1], 3)
        b["ret_pct"] = round((p[0] / float(b["entry_px"]) - 1) * 100, 2)
        n_close += 1
    if n_close:
        print("  结算 %d 笔" % n_close, flush=True)

    held = [b["code"] for b in batches if b.get("status") == "holding"]

    # ---- 2) 选票 ----
    cand = pick_one(day, cfg, held)
    action, note = "none", ""
    if not cand:
        note = "股票池无合格标的"
    elif cand["code"] in held:
        action = "skip"
        note = "已在持仓, 今日不加仓"
    else:
        action = "buy"

    # ---- 3) 建仓 / 初始化 ----
    today_done = bool(daily and daily[-1].get("date") == today)
    nav_before = (cash + sum(float(b["shares"]) * px.get(b["code"], (b["entry_px"], 0))[0]
                             for b in batches if b.get("status") == "holding")) / INIT_CAP
    n_buy = 0

    def _buy(c: dict, num: int, hd: int | None = None):
        nonlocal cash, n_buy
        p = px.get(c["code"])
        if not p or p[0] <= 0:
            return False
        nav_now = (cash + sum(float(b["shares"]) * px.get(b["code"], (b["entry_px"], 0))[0]
                              for b in batches if b.get("status") == "holding")) / INIT_CAP
        amount = nav_now * INIT_CAP / max(1, num)
        amount = min(amount, cash * 0.999)
        if amount < 100:
            return False
        shares = amount * (1 - HALF) / p[0]
        batches.append({"id": "%s-%03d" % (today, len(batches) + 1),
                        "signal_date": today, "code": c["code"],
                        "name": names.get(c["code"], ""), "ivol20": c["ivol20"],
                        "entry_px": round(p[0], 4), "entry_px_raw": round(p[1], 3),
                        "shares": round(shares, 4), "cost": round(amount, 2),
                        "hold_days": int(hd or P["hold_days"]), "status": "holding",
                        "exit_date": None, "exit_px": None, "ret_pct": None})
        cash -= amount
        n_buy += 1
        return True

    if not batches and not today_done:
        # 首次上线: 一次性建满 hold_days 只, 之后每日 1 只滚动
        from src.satellite import rank_lowvol
        rk = rank_lowvol(day, cfg)
        k = 0
        for _, row in rk.iterrows():
            if k >= P["hold_days"]:
                break
            # 到期日错开: 第 1 只 1 日后到期 ... 第 40 只 40 日后, 之后每天到期 1 只
            if _buy({"code": str(row["ts_code"]), "ivol20": round(float(row["_ivol"]), 2)},
                    P["hold_days"], hd=k + 1):
                k += 1
        print("  初始化建仓 %d 只 (每只 %.1f%%, 到期日 1~%d 日错开)"
              % (n_buy, 100.0 / P["hold_days"], P["hold_days"]), flush=True)
    elif action == "buy" and not today_done:
        _buy(cand, P["hold_days"])
        print("  买入 %s %s  ivol %.2f" % (cand["code"], names.get(cand["code"], ""), cand["ivol20"]), flush=True)

    # ---- 4) 推荐留痕 ----
    if not today_done:
        daily.append({"date": today, "code": cand["code"] if cand else "",
                      "name": names.get(cand["code"], "") if cand else "",
                      "ivol20": cand["ivol20"] if cand else None,
                      "action": action, "note": note,
                      "bought": bool(n_buy)})
        daily = daily[-120:]

    # ---- 5) 重估 ----
    hold_val = 0.0
    for b in batches:
        if b.get("status") != "holding":
            continue
        p = px.get(b["code"])
        if not p:
            continue
        b["px"] = round(p[1], 3)
        b["ret_pct"] = round((p[0] / float(b["entry_px"]) - 1) * 100, 2)
        b["held_days"] = _held_days(dates, b.get("signal_date"), today)
        hold_val += float(b["shares"]) * p[0]

    nav = (cash + hold_val) / INIT_CAP
    tape["cash"] = round(cash, 2)
    tape["nav"] = round(nav, 6)
    tape["capital"] = round(INIT_CAP * nav, 2)
    tape["batches"] = batches
    tape["daily"] = daily
    tape["updated"] = today

    n_hold = len([b for b in batches if b.get("status") == "holding"])
    print("  持仓 %d 只, 现金 %.0f, 净值 %.4f (%+.2f%%)"
          % (n_hold, cash, nav, (nav / nav_before - 1) * 100), flush=True)

    TAPE.parent.mkdir(parents=True, exist_ok=True)
    TAPE.write_text(json.dumps(tape, ensure_ascii=False, indent=1), encoding="utf-8")
    print("已写 %s | %d 批 / %d 条推荐" % (TAPE, len(batches), len(daily)), flush=True)


def _idx(dates, d):
    try:
        return dates.index(d)
    except ValueError:
        return -1


def _held_days(dates, d0, d1):
    i0, i1 = _idx(dates, d0), _idx(dates, d1)
    if i0 < 0 or i1 < 0:
        return None
    return i1 - i0


if __name__ == "__main__":
    main()
