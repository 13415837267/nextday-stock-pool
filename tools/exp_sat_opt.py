# -*- coding: utf-8 -*-
"""副策略结构性变体实验(第一百四十四轮)。

为什么是结构性变体、不是调参
----------------------------
副策略 1500 日只有 32 笔, 参数网格随便调都能"调好", 那是过拟合。
所以这里只测**有独立理论依据、不改选股因子**的规则:

  V1 持仓期大盘闸门: 持仓中指数跌破 MA20(reg.trade_ok=False) 当日收盘清仓
     —— 当前闸门只管"开新仓", 不管"已持仓", 40 日满仓承受全部 beta, 是 -49% 回撤的主因
  V2 持有期 20 日: 缩短暴露时间, 交易次数翻倍, 看成本与回撤的交换比
  V3 20 日动量过滤: 只买 mom20>0 的低波票 —— 低波 + 阴跌 = 低波陷阱
  V4 移动止盈: 从持仓期最高收盘回撤 10% 卖 —— 保住右尾(2025 招商轮船 +39% 这类)
  V5 = V1 + V4 组合
  V6 = V1 + V2 组合

口径与 tools/satellite_backtest.py 逐条一致(同一 context、同一成本、同一成交时点),
只替换被实验的那一条规则; 选股仍走 rank_pool(), 不另写一套筛选。

用法: python tools/exp_sat_opt.py [--days 1500] [--json /tmp/exp_sat_opt.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context  # noqa: E402
from src.config import load_config  # noqa: E402
from src.satellite import daily_params, entry_gate, rank_pool, should_exit  # noqa: E402

HALF = 0.0015


def _row(day, code):
    if not code:
        return None
    r = day.loc[day["ts_code"] == code]
    return r.iloc[0] if len(r) else None


def _rank_one(day, cfg, mom20_min=None, depth=None, order_by=None, pick_nth=0):
    """与 pick_one 同口径, 只是可选地"先取 ivol 前 depth 名, 再按 order_by 排序取第 pick_nth 个"。

    order_by=None 时就是原始口径(全池 ivol 升序第 1 名)。
    depth=20 + order_by="mom20"/"amount"/"bias20" = "在最低波的 20 只里挑最超跌/最大额/最偏离均线"。
    """
    rk = rank_pool(day, cfg)
    P = daily_params(cfg)
    if len(rk) < P["min_pool"]:
        return None
    if mom20_min is not None:
        rk = rk.loc[rk["mom20"] > mom20_min]
        if not len(rk):
            return None
    if depth:
        rk = rk.head(int(depth))
    if order_by:
        # amount 要"最大额"(降序), mom20/bias20 要"最超跌"(升序)
        rk = rk.sort_values(order_by, ascending=(order_by != "amount"))
    if pick_nth:
        rk = rk.iloc[int(pick_nth):]
        if not len(rk):
            return None
    r = rk.iloc[0]
    return {"code": str(r["ts_code"]), "ivol20": round(float(r["ivol20"]), 3),
            "close": round(float(r["close"]), 2)}


def simulate(ctx, cfg, *, hold=None, exit_gate=False, mom20_min=None, trail=None,
             depth=None, order_by=None, pick_nth=0):
    """跑一遍单吊模拟, 返回指标 + 分年 + 逐笔。"""
    by_date, dates, reg, names = ctx["by_date"], ctx["main_dates"], ctx["reg"], ctx["names"]
    cfg = json.loads(json.dumps(cfg))
    if hold:
        cfg.setdefault("satellite", {})["hold_days"] = int(hold)
    P = daily_params(cfg)

    if reg is not None and len(reg) and not isinstance(reg.index[0], str):
        reg = reg.copy()
        reg.index = reg.index.astype(str)

    cash, units, pos, sig = 100000.0, 0.0, None, None
    navs, trades = [], []
    last_px = {}

    for j, d in enumerate(dates):
        day = by_date.get(d)
        if day is None:
            navs.append((d, cash + units * last_px.get(pos["code"], 0) if pos else cash))
            continue

        gate = True
        if P["gate"] and reg is not None and d in reg.index:
            gate = bool(reg.loc[d, "trade_ok"])

        # 1) 开盘执行信号
        if sig and pos is None:
            r = _row(day, sig["code"])
            if r is not None and bool(r["is_paused"] == 0) and float(r["vol"]) > 0 \
                    and float(r["open"]) < float(r["high_limit"]) - 1e-9:
                gap = (float(r["open"]) / float(r["pre_close"]) - 1) * 100 if float(r["pre_close"]) else 0.0
                ok, _ = entry_gate(gap, cfg)
                if ok:
                    px = float(r["open_adj"]) * (1 + HALF)
                    units, cash = cash / px, 0.0
                    pos = {"code": sig["code"], "name": names.get(sig["code"], ""),
                           "signal_date": sig["date"], "entry_date": d, "entry_j": j,
                           "entry_px": px, "entry_px_raw": float(r["open"]), "peak": px}
            sig = None

        # 2) 收盘卖出
        if pos is not None:
            r = _row(day, pos["code"])
            if r is not None and bool(r["is_paused"] == 0) and float(r["vol"]) > 0:
                cl = float(r["close_adj"])
                last_px[pos["code"]] = cl
                pos["peak"] = max(pos["peak"], cl)
                ret = (cl / pos["entry_px"] - 1) * 100
                why = ""
                if exit_gate and not gate:
                    why = "大盘跌破 MA20 清仓"
                elif trail and (cl / pos["peak"] - 1) * 100 <= -float(trail):
                    why = "移动止盈(自高点 %.1f%%)" % ((cl / pos["peak"] - 1) * 100)
                if not why:
                    ex, w = should_exit(j - pos["entry_j"], ret, cfg)
                    why = w if ex else ""
                if why:
                    cash += units * cl * (1 - HALF)
                    trades.append({"entry": pos["entry_date"], "exit": d, "code": pos["code"],
                                   "name": pos["name"], "ret_pct": round(ret, 2),
                                   "hold_days": j - pos["entry_j"], "note": why})
                    pos, units = None, 0.0

        # 3) 收盘信号
        if pos is None:
            if gate:
                c = _rank_one(day, cfg, mom20_min, depth=depth, order_by=order_by,
                              pick_nth=pick_nth)
                if c:
                    c["date"] = d
                    sig = c

        # 4) 估值
        mv = 0.0
        if pos is not None:
            r = _row(day, pos["code"])
            if r is not None and bool(r["is_paused"] == 0) and float(r["vol"]) > 0:
                last_px[pos["code"]] = float(r["close_adj"])
            mv = units * last_px.get(pos["code"], pos["entry_px"])
        navs.append((d, cash + mv))

    nav = pd.Series({d: v for d, v in navs}, dtype=float)
    ret = nav.pct_change().dropna()
    years = len(nav) / 250.0
    total = nav.iloc[-1] / 100000.0 - 1
    cagr = (1 + total) ** (1 / years) - 1 if years > 0 and (1 + total) > 0 else -1.0
    mdd = float((nav / nav.cummax() - 1).min())
    sharpe = float(ret.mean() / ret.std() * np.sqrt(252)) if ret.std() else 0.0
    real = [t for t in trades if t["ret_pct"] is not None]
    wins = [t for t in real if t["ret_pct"] > 0]
    gw = sum(t["ret_pct"] for t in wins)
    gl = -sum(t["ret_pct"] for t in real if t["ret_pct"] <= 0)

    # 分年
    yr = {}
    prev = nav.iloc[0]
    for y in sorted({str(d)[:4] for d in nav.index}):
        seg = nav[[d for d in nav.index if str(d)[:4] == y]]
        yr[y] = round((seg.iloc[-1] / prev - 1) * 100, 1) if len(seg) else 0.0
        prev = seg.iloc[-1]

    return {"total": round(total * 100, 2), "cagr": round(cagr * 100, 2),
            "mdd": round(mdd * 100, 2),
            "calmar": round(cagr / abs(mdd), 2) if mdd else 0.0,
            "sharpe": round(sharpe, 2), "n": len(real),
            "win": round(len(wins) / len(real) * 100, 1) if real else 0.0,
            "avg": round(sum(t["ret_pct"] for t in real) / len(real), 2) if real else 0.0,
            "pl_ratio": round(gw / gl, 2) if gl else 0.0,
            "years": yr, "trades": trades}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--json", default="")
    a = ap.parse_args()

    cfg = load_config()
    print(f"[ctx] 载入 {a.days} 日面板...", flush=True)
    ctx = prepare_context(cfg, a.days)
    print(f"[ctx] {len(ctx['main_dates'])} 个交易日, {ctx['main_dates'][0]} ~ {ctx['main_dates'][-1]}",
          flush=True)

    variants = [
        ("V0 基线", dict()),
        ("V1 持仓期闸门", dict(exit_gate=True)),
        ("V2 持有20日", dict(hold=20)),
        ("V3 mom20>0", dict(mom20_min=0)),
        ("V4 移动止盈10%", dict(trail=10)),
        ("V5 V1+移动止盈", dict(exit_gate=True, trail=10)),
        ("V6 V1+持有20日", dict(exit_gate=True, hold=20)),
        # 第二轮: 不动交易规则, 只换"低波池内怎么挑那 1 只"
        ("V7 低波20里最超跌", dict(depth=20, order_by="mom20")),
        ("V8 低波20里最大额", dict(depth=20, order_by="amount")),
        ("V9 低波20里最偏离MA20", dict(depth=20, order_by="bias20")),
        ("V10 ivol第2名", dict(pick_nth=1)),
        ("V11 低波5里最超跌", dict(depth=5, order_by="mom20")),
    ]
    out = []
    for name, kw in variants:
        r = simulate(ctx, cfg, **kw)
        out.append({"name": name, **{k: v for k, v in r.items() if k != "trades"},
                    "trades": r["trades"]})
        print("%-18s 总%8.2f%% 年化%6.2f%% 回撤%7.2f%% 卡玛%6.2f 夏普%5.2f 笔%3d 胜%5.1f%% 均%6.2f%%"
              % (name, r["total"], r["cagr"], r["mdd"], r["calmar"], r["sharpe"], r["n"], r["win"], r["avg"]),
              flush=True)

    years = sorted({y for r in out for y in r["years"]})
    print("\n分年收益(%):")
    print("变体".ljust(18) + "".join(y.rjust(7) for y in years))
    for r in out:
        print(r["name"].ljust(18) + "".join(str(r["years"].get(y, "")).rjust(7) for y in years))

    if a.json:
        Path(a.json).write_text(json.dumps({"days": a.days, "result": out}, ensure_ascii=False, indent=1),
                                encoding="utf-8")
        print("\n写出", a.json)


if __name__ == "__main__":
    main()
