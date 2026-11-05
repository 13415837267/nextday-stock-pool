# -*- coding: utf-8 -*-
"""副策略回测口径对照(2026-11-05, 第一百四十轮)。

起因: 页面要与主策略同构, 需要一个 1500 日副策略回测。新写的
tools/satellite_backtest.py(严格对齐实盘账本)得到 +50.9%/-49.1%,
而第一百三十九轮 exp_trend_single4 的同一组参数是 +6.3%/-42.8%。
两者差 45pp —— 必须查清是"规则不同"还是"引擎有 bug", 否则页面数字不可信。

逐条对比两套实现, 只有三处不同(选股/闸门/成本完全一致):
  A. 卖出时点: 实盘"持满第 N 个交易日**收盘**卖"; exp "同一日**开盘**卖"
  B. 止损执行: 实盘在触发止损的**当日收盘**成交; exp 标记后在**次日开盘**卖
  C. 卖出后衔接: 实盘卖出当日收盘才出信号(次日开盘买, 空一天);
     exp 每天都算候选, 卖出当日开盘就用手里的旧候选**立即再买**(不空仓)

本脚本把 A/B/C 拆成开关, 在同一份面板上跑 2x2x2, 看每个口径各贡献多少。

用法: python3 tools/exp_sat_exit.py --days 1500 --out output/exp_sat_exit.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context  # noqa: E402
from src.config import load_config  # noqa: E402
from src.satellite import daily_params, entry_gate, pick_one, should_exit  # noqa: E402
import tools.exp_trend_single4 as EXP  # noqa: E402

HALF = 0.0015


def _row(day, code):
    if not code:
        return None
    r = day.loc[day["ts_code"] == code]
    return r.iloc[0] if len(r) else None


def _ok(r):
    return r is not None and bool(r["is_paused"] == 0) and float(r["vol"]) > 0


def _stats(nav, trades):
    nav = np.asarray(nav, dtype=float)
    years = len(nav) / 250.0
    total = nav[-1] - 1
    ann = (1 + total) ** (1 / years) - 1 if total > -1 else -1.0
    mdd = float((nav / np.maximum.accumulate(nav) - 1).min())
    r = nav[1:] / nav[:-1] - 1
    rets = [t["ret_pct"] for t in trades]
    return {"total": round(total * 100, 2), "ann": round(ann * 100, 2), "mdd": round(mdd * 100, 2),
            "calmar": round(ann / abs(mdd), 2) if mdd else None,
            "sharpe": round(float(np.mean(r) / (np.std(r) + 1e-12) * np.sqrt(250)), 2),
            "trades": len(trades),
            "win": round(100 * float(np.mean([x > 0 for x in rets])), 1) if rets else None,
            "avg": round(float(np.mean(rets)), 3) if rets else None}


def sim(ctx, cfg, exit_at_close=True, stop_at_close=True, same_day_entry=False):
    """exit_at_close: 持满当日收盘卖(True=实盘) / 当日开盘卖(False=exp)
    stop_at_close: 止损当日收盘成交(True=实盘) / 次日开盘卖(False=exp)
    same_day_entry: 卖出当日是否用手里的旧候选立即再买(False=实盘, 空一天)
    """
    P = daily_params(cfg)
    dates, by_date, reg = ctx["main_dates"], ctx["by_date"], ctx.get("reg")
    cash, units, pos = 1.0, 0.0, None
    pick = None                    # 昨日收盘生成的候选
    pending_stop = False           # 昨日收盘触发止损, 等今日开盘卖
    navs, trades = [], []

    for j, d in enumerate(dates):
        day = by_date.get(d)
        if day is None:
            navs.append(cash)
            continue
        sold_today = False

        # ---------- 1) 开盘: 出场(仅当出场价取开盘) ----------
        if pos is not None:
            r = _row(day, pos["code"])
            if _ok(r):
                due_hold = (j - pos["entry_j"]) >= P["hold_days"]
                if (pending_stop and not stop_at_close) or (due_hold and not exit_at_close):
                    px = float(r["close_adj"]) if float(r["open"]) <= float(r["low_limit"]) + 1e-9 \
                        else float(r["open_adj"])
                    cash = units * px * (1 - HALF)
                    trades.append({"ret_pct": round((px / pos["entry_px"] - 1) * 100, 2),
                                   "hold": j - pos["entry_j"], "stopped": bool(pending_stop)})
                    pos, units, pending_stop, sold_today = None, 0.0, False, True

        # ---------- 2) 开盘: 买入 ----------
        if pos is None and pick is not None and cash > 0 and (not sold_today or same_day_entry):
            ok = True
            if P["gate"] and reg is not None:
                sday = dates[j - 1] if j > 0 else d      # 信号日 = 前一日收盘
                ok = bool(reg.loc[sday, "trade_ok"]) if sday in reg.index else True
            r = _row(day, pick["code"])
            if ok and _ok(r) and float(r["open"]) < float(r["high_limit"]) - 1e-9:
                gap = (float(r["open"]) / float(r["pre_close"]) - 1) * 100 if float(r["pre_close"]) else 0.0
                g_ok, _ = entry_gate(gap, cfg)
                if g_ok:
                    px = float(r["open_adj"]) * (1 + HALF)
                    units, cash = cash / px, 0.0
                    pos = {"code": pick["code"], "entry_j": j, "entry_px": px}

        # ---------- 3) 收盘: 出场(出场价取收盘) ----------
        if pos is not None:
            r = _row(day, pos["code"])
            if _ok(r):
                cl = float(r["close_adj"])
                ret = (cl / pos["entry_px"] - 1) * 100
                due_hold = (j - pos["entry_j"]) >= P["hold_days"]
                if due_hold and exit_at_close:
                    cash = units * cl * (1 - HALF)
                    trades.append({"ret_pct": round(ret, 2), "hold": j - pos["entry_j"],
                                   "stopped": False})
                    pos, units, sold_today = None, 0.0, True
                elif P["stop"] < 0 and ret <= P["stop"] * 100:
                    if stop_at_close:
                        cash = units * cl * (1 - HALF)
                        trades.append({"ret_pct": round(ret, 2), "hold": j - pos["entry_j"],
                                       "stopped": True})
                        pos, units, sold_today = None, 0.0, True
                    else:
                        pending_stop = True

        # ---------- 4) 收盘: 出明日候选 ----------
        if pos is None or same_day_entry:
            gate_ok = True
            if P["gate"] and reg is not None and d in reg.index:
                gate_ok = bool(reg.loc[d, "trade_ok"])
            pick = pick_one(day, cfg) if gate_ok else None
        else:
            pick = None

        r = _row(day, pos["code"]) if pos is not None else None
        navs.append(cash + units * float(r["close_adj"]) if r is not None and _ok(r) else cash)

    return _stats(navs, trades), trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--out", default="output/exp_sat_exit.json")
    args = ap.parse_args()

    cfg = load_config()
    ctx = prepare_context(cfg, args.days)
    print(f"窗口 {len(ctx['main_dates'])} 日", flush=True)

    out = {"days": args.days, "matrix": {}}
    for eac in (True, False):
        for sac in (True, False):
            for sde in (False, True):
                key = f"{'close' if eac else 'open'}exit_{'close' if sac else 'open'}stop_" \
                      f"{'reentry' if sde else 'gap1'}"
                st, _ = sim(ctx, cfg, eac, sac, sde)
                out["matrix"][key] = st
                print(f"{key:36s} 总{st['total']:+9.2f}% 回撤{st['mdd']:+7.2f}% "
                      f"卡玛{st['calmar'] or 0:5.2f} 笔{st['trades']:3d} 胜{st['win'] or 0:5.1f}% "
                      f"单笔{st['avg'] or 0:+6.2f}%", flush=True)

    # 交叉验证: 直接调用第一百三十九轮实验脚本的 run()
    st_exp, _ = EXP.run(ctx, cfg, trend=False, hold=40, gate=True, stop=-0.12)
    out["exp_trend_single4_pure_H40_gate_s12"] = {
        "total": st_exp["total"], "ann": st_exp["ann"], "mdd": st_exp["mdd"],
        "calmar": st_exp["calmar"], "trades": st_exp["trades"],
        "win": st_exp["win"], "avg": st_exp["avg"]}
    print(f"{'exp_trend_single4(旧口径)':36s} 总{st_exp['total']:+9.2f}% 回撤{st_exp['mdd']:+7.2f}% "
          f"卡玛{st_exp['calmar'] or 0:5.2f} 笔{st_exp['trades']:3d} 胜{st_exp['win'] or 0:5.1f}% "
          f"单笔{st_exp['avg'] or 0:+6.2f}%", flush=True)

    p = Path(args.out)
    if not p.is_absolute():
        p = ROOT / p
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print(f"写出 {p}", flush=True)


if __name__ == "__main__":
    main()
