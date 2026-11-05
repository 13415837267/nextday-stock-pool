# -*- coding: utf-8 -*-
"""副策略回测(第一百四十轮) —— 给页面提供与主策略同构的「回测绩效」模块。

为什么单独一个脚本
------------------
主策略的数值来自 src/backtest.py(多票组合/分档仓位/评分门槛), 副策略是单吊满仓、
持 40 日、大盘闸门, 两套引擎无法共用一次模拟。但**输出结构必须一致**:
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
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context  # noqa: E402
from src.config import load_config  # noqa: E402
from src.satellite import (STRATEGY_CN, daily_params, entry_gate,  # noqa: E402
                           pick_one, should_exit)

HALF = 0.0015          # 单边成本, 与 tools/satellite_run.py 同一常数


def _row(day, code):
    if not code:
        return None
    r = day.loc[day["ts_code"] == code]
    return r.iloc[0] if len(r) else None


def _fmt(d: str) -> str:
    return f"{d[:4]}-{d[4:6]}-{d[6:]}"


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


def run(ctx: dict, cfg: dict) -> dict:
    P = daily_params(cfg)
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

    cash, units, pos = init, 0.0, None
    sig = None
    navs, trades = [], []
    blocked = skipped = 0
    last_px = {}

    for j, d in enumerate(dates):
        day = by_date.get(d)
        if day is None:
            navs.append((d, cash + units * last_px.get(pos["code"], 0) if pos else cash))
            continue

        # ---------- 1) T+1 开盘: 执行昨日信号 ----------
        if sig and pos is None:
            r = _row(day, sig["code"])
            if r is None or bool(r["is_paused"] == 1) or float(r["vol"]) <= 0:
                skipped += 1
            elif float(r["open"]) >= float(r["high_limit"]) - 1e-9:
                skipped += 1          # 一字涨停买不到
            else:
                gap = (float(r["open"]) / float(r["pre_close"]) - 1) * 100 if float(r["pre_close"]) else 0.0
                ok, _why = entry_gate(gap, cfg)
                if not ok:
                    skipped += 1
                else:
                    px = float(r["open_adj"]) * (1 + HALF)
                    units, cash = cash / px, 0.0
                    pos = {"code": sig["code"], "name": names.get(sig["code"], sig.get("name", "")),
                           "signal_date": sig["date"], "entry_date": d, "entry_j": j,
                           "entry_px": px, "entry_px_raw": float(r["open"]),
                           "gap_pct": round(gap, 2), "shares": units}
            sig = None

        # ---------- 2) 收盘: 卖出判定 ----------
        if pos is not None:
            r = _row(day, pos["code"])
            if r is not None and bool(r["is_paused"] == 0) and float(r["vol"]) > 0:
                cl = float(r["close_adj"])
                last_px[pos["code"]] = cl
                ret = (cl / pos["entry_px"] - 1) * 100
                ex, why = should_exit(j - pos["entry_j"], ret, cfg)
                if ex:
                    got = units * cl * (1 - HALF)
                    cash += got
                    trades.append({"code": pos["code"], "name": pos["name"],
                                   "signal_date": pos["signal_date"], "entry_date": pos["entry_date"],
                                   "exit_date": d, "entry_px": round(pos["entry_px_raw"], 3),
                                   "exit_px": round(float(r["close"]), 3),
                                   "hold_days": j - pos["entry_j"],
                                   "ret_pct": round(ret, 2),
                                   "note": why})
                    pos, units = None, 0.0

        # ---------- 3) 收盘: 出明日信号(先过大盘闸门) ----------
        if pos is None:
            gate_ok = True
            if P["gate"] and reg is not None and d in reg.index:
                gate_ok = bool(reg.loc[d, "trade_ok"])
            if not gate_ok:
                blocked += 1
            else:
                cand = pick_one(day, cfg)
                if cand:
                    cand["date"] = d
                    cand["name"] = names.get(cand["code"], "")
                    sig = cand

        # ---------- 4) 收盘估值 ----------
        if pos is not None:
            r = _row(day, pos["code"])
            if r is not None and bool(r["is_paused"] == 0) and float(r["vol"]) > 0:
                last_px[pos["code"]] = float(r["close_adj"])
            mv = units * last_px.get(pos["code"], pos["entry_px"])
            if pos.get("entry_j") is not None:
                pos["held_days"] = j - pos["entry_j"]
                pos["unrealized_pct"] = round((mv / (units * pos["entry_px"]) - 1) * 100, 2) \
                    if units else None
        else:
            mv = 0.0
        navs.append((d, cash + mv))

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
    if pos is not None:
        mv = units * last_px.get(pos["code"], pos["entry_px"])
        open_positions.append({
            "signal_date": _fmt(pos["signal_date"]), "code": pos["code"], "name": pos["name"],
            "buy_date": _fmt(pos["entry_date"]), "buy_px": round(pos["entry_px_raw"], 3),
            "cost_pct": 100.0, "last_px": round(last_px.get(pos["code"], 0), 3),
            "unrealized_pct": round((mv / (units * pos["entry_px"]) - 1) * 100, 2) if units else None,
        })

    return {
        "meta": {
            "run_time": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
            "start": dates[0], "end": dates[-1], "days": len(dates),
            "initial_capital": init, "strategy": STRATEGY_CN.get(P["mode"], P["mode"]),
            "rules": (f"单吊 100% 仓位 · 每日最多 1 只 · 持 {P['hold_days']} 个交易日收盘卖"
                      + (f" / 止损 {P['stop'] * 100:g}%" if P["stop"] < 0 else "")
                      + (" · 大盘 MA20 闸门" if P["gate"] else "")
                      + f" · 单边成本 {HALF * 100:g}%"),
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
            "空仓天数": len(nav) - sum(t["hold_days"] + 1 for t in real),
            "跳过买入次数": skipped,
            "同期市场等权基准%": round(float(bench_norm.iloc[-1] / init - 1) * 100, 2)
            if len(bench_norm) else None,
        },
        "equity": [{"d": _fmt(d), "nav": round(float(v), 0)} for d, v in nav.items()],
        "benchmark": [{"d": _fmt(d), "nav": round(float(v), 0)} for d, v in bench_norm.items()]
        if len(bench_norm) else [],
        "monthly": _monthly(nav),
        "trades": trades,
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

    res = run(ctx, cfg)
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
