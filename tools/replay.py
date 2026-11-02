"""历史回放数据生成器 —— 为站点「历史回放」页产出 output/replay.json。

为什么单独一个脚本（而不是直接复用 output/backtest.json）:
  1. 站点要的是**区间内逐笔**成交（含每笔的买入价/两条卖出腿/收益），
     而 backtest.json 默认只留最近 60 笔（trades_recent），不够回放用；
  2. 站点要的净值曲线是**区间内重新归一**的（起点=区间首日估值），
     而 backtest.json 的 equity 是整个回测窗口的复利曲线；
  3. 成交明细必须能自证：`sell_px` 是(开盘腿+收盘腿)的加权均价，
     另带 `sell_px_open/sell_px_close/sell_frac_open`，
     否则分批出场时会出现"卖价低于买价却是正收益"的读不懂。

用法:
    python tools/replay.py                        # 默认 2025-11-01 ~ 2026-10-31
    python tools/replay.py --start 20250101 --end 20251231
"""
from __future__ import annotations

import argparse
import json
import os
import sys
from datetime import datetime
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

os.environ.setdefault("BT_FULL_TRADES", "1")   # 需要 trades_all

import pandas as pd

from src.backtest import prepare_context, run_backtest
from src.config import load_config
from src.rule_text import rules_line, sell_rule_text


def _fmt(d: str) -> str:
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


def _monthly(curve: list[dict], base_nav: float, base_date: str) -> list[dict]:
    """区间月度收益：每月末净值 / 上月末净值 - 1（首月用 base_nav）。"""
    months: dict[str, list[float]] = {}
    for p in curve:
        months.setdefault(p["d"][:7], []).append(p["nav"])
    out, prev = [], base_nav
    for m in sorted(months):
        vals = months[m]
        # 首月若起点在月中，基准用 base_nav；否则用上月末
        last = vals[-1]
        out.append({"month": m.replace("-", ""), "ret_pct": round((last / prev - 1) * 100, 2)})
        prev = last
    return out


def build(start: str, end: str) -> dict:
    cfg = load_config()
    bt = cfg["backtest"]
    ex = cfg["execution"]
    warmup = int(bt["warmup_days"])

    # 先量出区间需要多少交易日，再多留 10 天缓冲（避免边界缺失）
    from src.data_api import ZZClient
    cal = ZZClient(cfg).trade_days(days=1200)
    cal = sorted(str(d) for d in cal)
    span = [d for d in cal if start <= d <= end]
    if not span:
        raise SystemExit(f"区间 {start}~{end} 内没有交易日")
    need = len(span) + 10

    print(f"[replay] 区间 {start}~{end}: {len(span)} 个交易日, 回测 {need} 日(+预热 {warmup})")
    ctx = prepare_context(cfg, need)
    res = run_backtest(cfg, need, ctx=ctx)

    trades_all = res.get("trades_all") or []
    equity = res.get("equity") or []
    benchmark = res.get("benchmark") or []

    # 区间内全部动作（含 skip），用于「逐日决策」；成交明细只取真正成交的
    acts = [t for t in trades_all if t.get("buy_date") and start <= t["buy_date"] <= end]
    tr = [t for t in acts if t.get("ret_pct") is not None]
    # float32 面板会带出 10.050000190734863 这种噪声, 落盘前收敛到 4 位
    for t in tr:
        for k in ("buy_px", "sell_px", "sell_px_open", "sell_px_close"):
            if t.get(k) is not None:
                t[k] = round(float(t[k]), 4)
    done = tr

    eq = [p for p in equity if start <= p["d"].replace("-", "") <= end]
    if not eq:
        raise SystemExit("区间内没有净值数据")
    # 起点估值 = 区间首日之前的最后一个净值点（没有就用首日）
    pre = [p for p in equity if p["d"].replace("-", "") < start]
    base_nav = float(pre[-1]["nav"]) if pre else float(eq[0]["nav"])
    base_date = pre[-1]["d"] if pre else eq[0]["d"]
    end_nav = float(eq[-1]["nav"])

    navs = [p["nav"] for p in eq]
    peak, mdd = navs[0], 0.0
    for v in navs:
        peak = max(peak, v)
        mdd = min(mdd, v / peak - 1)

    rets = [t["ret_pct"] for t in done]
    n = len(rets)
    win = sum(1 for r in rets if r > 0)
    summary = {
        "n_trades": n,
        "n_win": win,
        "win_rate": round(win / n * 100, 1) if n else None,
        "avg_ret_pct": round(sum(rets) / n, 2) if n else None,
        "sum_ret_pct": round(sum(rets), 2) if n else None,
        "best": round(max(rets), 2) if n else None,
        "worst": round(min(rets), 2) if n else None,
        "range_ret_pct": round((end_nav / base_nav - 1) * 100, 2),
        "range_mdd_pct": round(mdd * 100, 2),
        "base_date": base_date,
        "base_nav": base_nav,
        "end_nav": end_nav,
    }

    # 逐日决策（解释"这天为什么没买"）：按买入日聚合（含 skip，故用 acts 而非 tr）
    # trade_ok / alloc 必须带上: 前端用它区分"风控空仓"与"评分不足",
    # 缺字段会让 `!x.trade_ok` 把每一天都判成空仓。
    reg = ctx.get("reg")
    days: dict[str, dict] = {}
    for t in acts:
        d = t["buy_date"]
        ok = bool(reg.loc[d, "trade_ok"]) if (reg is not None and d in reg.index) else None
        rec = days.setdefault(d, {"date": d, "trade_ok": ok, "alloc": None,
                                  "fills": [], "skipped": []})
        if t.get("ret_pct") is None:
            rec["skipped"].append({"code": t["code"], "name": t.get("name", ""),
                                   "note": t.get("note", "")})
        else:
            rec["fills"].append({"code": t["code"], "name": t.get("name", ""),
                                 "ret_pct": t["ret_pct"]})
            if rec["alloc"] is None and t.get("pos_frac") is not None:
                rec["alloc"] = float(t["pos_frac"])
    # 补齐"完全没有动作"的交易日: 否则「N 个交易日」只统计到有候选/有跳过的日子,
    # 读者会以为漏跑; 也让"当天没有合格候选"这一类能被单独计数。
    for d in span:
        if d not in days:
            ok = bool(reg.loc[d, "trade_ok"]) if (reg is not None and d in reg.index) else None
            days[d] = {"date": d, "trade_ok": ok, "alloc": None, "fills": [], "skipped": []}
    daily = [days[k] for k in sorted(days)]

    # 同期市场等权基准（区间口径）
    bmk = [p for p in benchmark if start <= p["d"].replace("-", "") <= end]
    bench_pct = None
    if bmk:
        b0 = float(bmk[0]["nav"])
        bench_pct = round((float(bmk[-1]["nav"]) / b0 - 1) * 100, 2)

    return {
        "meta": {
            "start": _fmt(span[0]),
            "end": _fmt(span[-1]),
            # 第一百二十五轮: 出场口径按 config 生成。原文"T+2收盘卖出(14:55；开盘浮盈≥阈值
            # 则开盘兑现)"同时错了两处: ①"浮盈"应为"浮亏"(sell_close_if_profit_below 是浮亏
            # 保护); ②该保护只在存在开盘腿时才有意义, 现行 sell_open_frac=0 全额收盘卖。
            "strategy": "T日收盘选股 → T+1开盘竞价买入 → " + sell_rule_text(ex, short=True),
            "initial_capital": float(bt["initial_capital"]),
            "base_date": base_date,
            "rules": rules_line(ex, "复利/" + (
                f"佣金万{ex['commission_rate'] * 10000:g}+印花税万{ex['stamp_tax'] * 10000:g}"
                f"+滑点千{ex['slippage'] * 1000:g}")),
            "generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "note": ("复用回测引擎, 指标仅用截至当日数据; 买入取 T+1 真实开盘价, "
                     "卖出按 T+2 开盘/收盘两条腿加权(比例见每笔 sell_frac_open), "
                     "收益含佣金万2.5+印花税万5+滑点千1; "
                     "跨区间边界的持仓(区间前买入/区间后卖出)不计入成交明细, 但其盈亏已体现在起点与终点净值中"),
        },
        "summary": summary,
        "trades": tr,
        "curve": eq,
        "base_nav": base_nav,
        "daily": daily,
        "monthly": _monthly(eq, base_nav, base_date),
        "perf": {"同期市场等权基准%": bench_pct},
    }


def main():
    ap = argparse.ArgumentParser(description="生成站点历史回放数据")
    ap.add_argument("--start", default="20251101", help="区间起(含), YYYYMMDD")
    ap.add_argument("--end", default="20261031", help="区间止(含), YYYYMMDD")
    ap.add_argument("--out", default=str(ROOT / "output" / "replay.json"))
    args = ap.parse_args()

    payload = build(args.start, args.end)
    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, ensure_ascii=False, indent=1, default=float), encoding="utf-8")

    s = payload["summary"]
    print("\n========== 回放 ==========")
    print(f"区间: {payload['meta']['start']} ~ {payload['meta']['end']}")
    print(f"笔数 {s['n_trades']} / 胜率 {s['win_rate']}% / 平均 {s['avg_ret_pct']}% / "
          f"最好 {s['best']}% / 最差 {s['worst']}%")
    print(f"区间收益 {s['range_ret_pct']}% / 最大回撤 {s['range_mdd_pct']}% "
          f"({s['base_nav']:,.0f} → {s['end_nav']:,.0f})")
    print(f"已写入: {out}")


if __name__ == "__main__":
    main()
