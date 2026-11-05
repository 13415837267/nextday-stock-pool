# -*- coding: utf-8 -*-
"""独立副策略每日维护: 月初建仓 / 每日更新浮盈 / 跨月结算(2026-11-05)。

与 paper_tape.py(主策略台账)完全并列、互不读写。三条铁律相同:
  1. 只增不改 —— 已结算的记录不重写;
  2. 幂等 —— 同一 month 不重复建仓(月初漏跑则次日补建);
  3. 只读 —— 不碰 config / latest.json / paper.json。

口径
----
  · 建仓价 = 信号日收盘价(close_adj); 结算价 = 换仓日收盘价;
  · 实盘按次日开盘执行, 与收盘口径的偏差约 ±0.3%, 记账不模拟;
  · 成本 = 换手率 × 0.30%(与回测同口径), 只在换仓时扣;
  · 初始虚拟本金 10 万, 与主策略资金完全独立。

用法
----
  python3 -m tools.satellite_run          # 正常维护
  python3 -m tools.satellite_run --force   # 强制本月重建(调试用, 会覆盖当月未结算记录)
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context  # noqa: E402
from src.config import load_config  # noqa: E402
from src.satellite import apply_cost, params, pick, turnover  # noqa: E402

TAPE = ROOT / "output" / "satellite.json"
INIT_CAP = 100000.0
DAYS = 300          # 只需 300 日: ivol20 用 20 日, 留足缓冲


def _load() -> dict:
    if TAPE.exists():
        try:
            d = json.loads(TAPE.read_text(encoding="utf-8"))
            if isinstance(d, dict) and isinstance(d.get("records"), list):
                return d
        except Exception:  # noqa: BLE001
            pass
    return {"strategy": "lowvol_buf", "version": 1, "init_capital": INIT_CAP,
            "nav": 1.0, "capital": INIT_CAP, "updated": None, "params": {}, "records": []}


def _px_map(day):
    """ts_code -> (close_adj, close)"""
    return {c: (float(a), float(b)) for c, a, b in
            zip(day["ts_code"], day["close_adj"], day["close"])}


def _mark(rec: dict, day, today: str) -> float:
    """按今日收盘更新记录浮盈, 返回该期净值。"""
    px = _px_map(day)
    tot = 0.0
    wsum = 0.0
    for h in rec["holdings"]:
        p = px.get(h["code"])
        if not p or h["entry_px"] <= 0:
            continue
        r = p[0] / h["entry_px"] - 1
        h["px"] = round(p[1], 3)
        h["ret_pct"] = round(r * 100, 2)
        tot += h["weight"] * r
        wsum += h["weight"]
    if wsum <= 0:
        rec["nav"] = rec.get("entry_nav", rec.get("nav", 1.0))
        return rec["nav"]
    r = tot / wsum
    rec["nav"] = round(rec.get("entry_nav", 1.0) * (1 + r), 6)
    rec["ret_pct"] = round((rec["nav"] / max(1e-9, rec.get("prev_nav", 1.0)) - 1) * 100, 2)
    rec["marked"] = today
    return rec["nav"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true")
    args = ap.parse_args()

    cfg = load_config()
    P = params(cfg)
    tape = _load()
    tape["params"] = {"top_n": P["top_n"], "buf_n": P["buf_n"],
                      "cost_pct": round(P["cost"] * 100, 3),
                      "amount_min_yi": P["amt_min"] / 1e8,
                      "universe": "非ST/上市满250日/成交额>=1亿"}
    recs = tape["records"]

    print("[1/2] 加载面板 %d 日 ..." % DAYS, flush=True)
    ctx = prepare_context(cfg, DAYS)
    by_date, dates = ctx["by_date"], ctx["main_dates"]
    names = ctx.get("names") or {}
    today = dates[-1]
    day = by_date[today]
    month = today[:6]

    last = recs[-1] if recs else None
    new_month = (last is None) or (last["month"] != month) or args.force

    # ---- 跨月: 先结算上一期 ----
    if new_month and last and last.get("status") == "holding":
        nav = _mark(last, day, today)
        last["status"] = "closed"
        last["exit_date"] = today
        last["exit_nav"] = round(nav, 6)
        last["exit_ret_pct"] = round((nav / max(1e-9, last.get("prev_nav", 1.0)) - 1) * 100, 2)
        print("  结算 %s: 净值 %.4f (%+.2f%%)"
              % (last["month"], nav, last["exit_ret_pct"]), flush=True)
        tape["nav"] = nav

    # ---- 本月建仓 ----
    if new_month:
        prev_codes = [h["code"] for h in last["holdings"]] if last else None
        picks = pick(day, cfg, prev_codes, names)
        if not picks:
            print("  股票池不足, 本月不建仓", flush=True)
            return
        turn = turnover([p["code"] for p in picks], prev_codes)
        nav_before = float(tape.get("nav", 1.0))
        nav_entry = apply_cost(nav_before, turn, P["cost"])
        px = _px_map(day)
        for p in picks:
            p["entry_px"] = round(px.get(p["code"], (0, 0))[0], 4)
            p["px"] = round(px.get(p["code"], (0, 0))[1], 3)
            p["ret_pct"] = 0.0
        rec = {"month": month, "signal_date": today, "status": "holding",
               "prev_nav": round(nav_before, 6), "entry_nav": round(nav_entry, 6),
               "nav": round(nav_entry, 6), "ret_pct": round((nav_entry / nav_before - 1) * 100, 2),
               "turnover": round(turn, 3), "holdings": picks, "marked": today}
        recs.append(rec)
        tape["nav"] = nav_entry
        print("  建仓 %s: %d 只, 换手 %.0f%%, 净值 %.4f"
              % (month, len(picks), turn * 100, nav_entry), flush=True)
        for p in picks:
            print("    %s %s  ivol %.2f  w %.3f" % (p["code"], p["name"], p["ivol20"], p["weight"]), flush=True)
    else:
        nav = _mark(last, day, today)
        tape["nav"] = nav
        print("  更新 %s: 净值 %.4f (%+.2f%%)" % (month, nav, last.get("ret_pct", 0)), flush=True)

    tape["capital"] = round(INIT_CAP * float(tape["nav"]), 2)
    tape["updated"] = today
    TAPE.parent.mkdir(parents=True, exist_ok=True)
    TAPE.write_text(json.dumps(tape, ensure_ascii=False, indent=1), encoding="utf-8")
    print("\n已写 %s | 累计净值 %.4f (%.2f 万)"
          % (TAPE, tape["nav"], tape["capital"] / 1e4), flush=True)


if __name__ == "__main__":
    main()
