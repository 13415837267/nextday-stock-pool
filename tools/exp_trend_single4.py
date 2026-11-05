# -*- coding: utf-8 -*-
"""单吊 + 大盘风控 + 止损(2026-11-05, 第一百三十九轮之四)。

前三轮把"单吊"这条路的边界量清楚了:
  · 追突破/趋势内回调/纯均线多头   回撤 -70%~-88%, 不可用
  · 纯低波单吊 H20                1500日 +7.8%/-42.7%, 全史 +6.1%/-57.6%
  · 趋势+低波单吊 H40             全史 +5.4%/-63.6% —— 趋势过滤没有增量
共同症结不是选股, 是**满仓单吊把个股尾部风险全额放大到净值**。

本轮只测一件事: 加两道防守能不能把回撤压到 -35% 以内, 且收益不明显缩水。
  ① 大盘风控闸门: 指数在 MA20 上方(trade_ok)才开新仓 —— 与主策略同一份 reg
  ② 个股止损: 收盘相对买入价跌破 -12%/-15% -> 次日开盘卖

用法: python3 tools/exp_trend_single4.py --days 1500 --out output/exp_trend_single4.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context  # noqa: E402
from src.config import board_of, load_config  # noqa: E402

COMM, STAMP, SLIP = 0.00025, 0.0005, 0.001
AMT_MIN, MIN_BARS, MIN_POOL = 1e8, 250, 20


def _stats(nav, trades):
    nav = np.asarray(nav, dtype=float)
    if len(nav) < 2:
        return {}
    years = len(nav) / 250.0
    ann = float(nav[-1] ** (1 / years) - 1) if nav[-1] > 0 else -1.0
    peak = np.maximum.accumulate(nav)
    mdd = float((nav / peak - 1).min())
    r = nav[1:] / nav[:-1] - 1
    rets = [t["ret_pct"] for t in trades]
    return {"ann": round(ann * 100, 2), "mdd": round(mdd * 100, 2),
            "calmar": round(ann / abs(mdd), 2) if mdd < 0 else None,
            "sharpe": round(float(np.mean(r) / (np.std(r) + 1e-12) * np.sqrt(250)), 2),
            "total": round((nav[-1] - 1) * 100, 2), "days": len(nav),
            "trades": len(trades),
            "win": round(100 * float(np.mean([x > 0 for x in rets])), 1) if rets else None,
            "avg": round(float(np.mean(rets)), 3) if rets else None,
            "expo": round(100.0 * sum(t["hold"] for t in trades) / max(1, len(nav)), 1),
            "stops": sum(1 for t in trades if t.get("stopped"))}


def _cand(day, cfg, trend):
    uni = cfg["universe"]
    boards = set(uni["boards"])
    d = day
    m = ((d["is_paused"] == 0) & (d["is_st"] == 0) & (d["bars"] >= MIN_BARS)
         & (d["amount"] >= AMT_MIN) & (d["ivol20"] > 0) & (d["ivol20"] <= 2.5))
    m &= d["ts_code"].map(lambda c: board_of(c) in boards).to_numpy()
    if trend:
        m &= ((d["close_adj"] > d["ma5"]) & (d["ma5"] > d["ma10"])
              & (d["ma10"] > d["ma20"]) & (d["ma20"] > d["ma20_prev"])).to_numpy()
        m &= (d["mom20"] > 0).to_numpy()
    df = d.loc[m]
    if len(df) < MIN_POOL:
        return None
    return str(df.loc[df["ivol20"].idxmin(), "ts_code"])


def _row(day, code):
    r = day.loc[day["ts_code"] == code] if code else None
    return r.iloc[0] if (r is not None and len(r)) else None


def run(ctx, cfg, trend, hold, gate, stop, gmin=-3.0, gmax=5.0):
    dates, by_date, reg = ctx["main_dates"], ctx["by_date"], ctx.get("reg")
    cash, units, held = 1.0, 0.0, None
    entry_j, entry_px, pending_exit, blocked = None, None, False, 0
    nav_curve, trades = [], []
    pick = None
    for j, d in enumerate(dates):
        day = by_date.get(d)
        if day is None:
            nav_curve.append(cash)
            continue
        if held and (pending_exit or (j - entry_j) >= hold):
            r = _row(day, held)
            if r is not None and bool(r["is_paused"] == 0) and float(r["vol"]) > 0:
                px = float(r["close_adj"]) if float(r["open"]) <= float(r["low_limit"]) + 1e-9 \
                    else float(r["open_adj"])
                got = units * px * (1 - SLIP) * (1 - COMM - STAMP)
                trades.append({"code": held, "entry_j": entry_j, "exit_j": j, "hold": j - entry_j,
                               "stopped": bool(pending_exit),
                               "ret_pct": round((got / (units * entry_px) - 1) * 100, 3)})
                cash, units, held, entry_j, entry_px, pending_exit = got, 0.0, None, None, None, False
        if held is None and pick is not None and cash > 0:
            ok = True
            if gate and reg is not None:
                # 用**信号日**的风控状态(收盘后可知), 无未来函数
                sday = dates[j - 1] if j > 0 else d
                ok = bool(reg.loc[sday, "trade_ok"]) if sday in reg.index else True
            r = _row(day, pick)
            if ok and r is not None and bool(r["is_paused"] == 0) and float(r["vol"]) > 0 \
                    and float(r["open"]) < float(r["high_limit"]) - 1e-9:
                gap = (float(r["open"]) / float(r["pre_close"]) - 1) * 100
                if gmin <= gap <= gmax:
                    px = float(r["open_adj"]) * (1 + SLIP) * (1 + COMM)
                    units, cash = cash / px, 0.0
                    held, entry_j, entry_px = pick, j, px
            elif not ok:
                blocked += 1
        r = _row(day, held) if held else None
        if r is not None:
            nav = cash + units * float(r["close_adj"])
            if stop and not pending_exit and (float(r["close_adj"]) / entry_px - 1) <= stop:
                pending_exit = True
        else:
            nav = cash
        nav_curve.append(nav)
        pick = _cand(day, cfg, trend)
    st = _stats(nav_curve, trades)
    st["blocked"] = blocked
    return st, trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--out", default="output/exp_trend_single4.json")
    args = ap.parse_args()

    cfg = load_config()
    ctx = prepare_context(cfg, args.days)
    print(f"窗口 {len(ctx['main_dates'])} 日", flush=True)

    out = {"days": args.days, "n_days": len(ctx["main_dates"]), "matrix": {}, "yearly": {}}
    for trend in (True, False):
        for hold in (20, 40):
            for gate in (False, True):
                for stop in (0.0, -0.12, -0.15):
                    key = f"{'trend' if trend else 'pure'}_H{hold}_{'gate' if gate else 'free'}_s{int(stop*100)}"
                    st, tr = run(ctx, cfg, trend, hold, gate, stop)
                    out["matrix"][key] = st
                    by = {}
                    for t in tr:
                        by.setdefault(str(ctx["main_dates"][t["exit_j"]])[:4], []).append(t["ret_pct"])
                    out["yearly"][key] = {y: {"n": len(v), "avg": round(float(np.mean(v)), 2)}
                                          for y, v in sorted(by.items())}
                    print(f"{key:30s} 年化{st['ann']:+7.1f}% 回撤{st['mdd']:+7.1f}% "
                          f"卡玛{st['calmar'] or 0:5.2f} 笔{st['trades']:4d} 胜{st['win'] or 0:5.1f}% "
                          f"单笔{st['avg'] or 0:+6.2f}% 止损{st['stops']:3d} 空{st['blocked']:4d}", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
