# -*- coding: utf-8 -*-
"""单吊趋势 第二轮: 三种趋势定义 x 止损 x 大盘过滤(2026-11-05, 第一百三十九轮之二)。

第一轮(exp_trend_single)结论: 现有"阳线突破 + 接近20日新高"的趋势票, 单吊全是负期望 ——
H=1 单笔 -0.50%、H=5 -0.28%、H=10 +0.43%、H=20 +1.26%(回撤 -70%)。说明**追突破**在 A 股
短线是均值回复的对手盘。本轮换三种"上涨趋势"的读法, 并加止损与大势过滤:

  A breakout : 均线多头 + 当日阳线接近20日新高(第一轮同款) —— 追突破
  B pullback : 均线多头 + 当日缩量阴线回调(不破MA20)   —— 趋势内低吸
  C stack    : 只要均线多头 + 20日动量为正                —— 纯趋势持有

止损: 收盘相对买入价跌破 -6%/-10% -> 次日开盘卖(仅用当日收盘信息)。
大势过滤: 指数在 MA20 上方才开仓(用 ctx["reg"] 的 trade_ok, 与主策略同一份)。

用法: python3 tools/exp_trend_single2.py --days 1500 --out output/exp_trend_single2.json
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context  # noqa: E402
from src.config import board_of, load_config  # noqa: E402

COMM, STAMP, SLIP = 0.00025, 0.0005, 0.001
AMT_MIN, MIN_BARS, MIN_POOL = 1e8, 250, 5


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
            "med": round(float(np.median(rets)), 3) if rets else None,
            "expo": round(100.0 * sum(t["hold"] for t in trades) / max(1, len(nav)), 1),
            "stops": sum(1 for t in trades if t.get("stopped"))}


def _base(day, cfg):
    uni = cfg["universe"]
    boards = set(uni["boards"])
    m = ((day["is_paused"] == 0) & (day["is_st"] == 0) & (day["bars"] >= MIN_BARS)
         & (day["amount"] >= AMT_MIN) & (day["turnover_rate"] <= uni["turnover_max"]))
    m &= day["ts_code"].map(lambda c: board_of(c) in boards).to_numpy()
    m &= ((day["close_adj"] > day["ma5"]) & (day["ma5"] > day["ma10"])
          & (day["ma10"] > day["ma20"]) & (day["ma20"] > day["ma20_prev"])).to_numpy()
    return day.loc[m]


def _score(df, kind):
    z = lambda s: (s - s.mean()) / (s.std() + 1e-9) if s.std() > 0 else s * 0
    if kind == "breakout":
        return (0.30 * z(df["dist_high20"]) + 0.25 * z(df["mom20"]) + 0.20 * z(df["close_pos"])
                + 0.15 * z(-df["bias20"]) + 0.10 * z(-np.log1p(df["amount"])))
    if kind == "pullback":
        # 趋势内低吸: 回调越深越好(但不破位)、缩量、收盘位置低
        return (0.35 * z(-df["pct"]) + 0.25 * z(-df["vol_ratio"]) + 0.20 * z(-df["close_pos"])
                + 0.10 * z(df["mom20"]) + 0.10 * z(-np.abs(df["bias20"] - 3)))
    # stack: 纯趋势强度
    return 0.45 * z(df["mom20"]) + 0.30 * z(df["close_adj"] / df["ma20"]) + 0.25 * z(-df["bias20"])


def _pick(day, cfg, kind):
    df = _base(day, cfg)
    if len(df) < MIN_POOL:
        return None
    if kind == "breakout":
        sel = cfg["selection"]
        df = df[(df["pct"] >= sel.get("tr_pct_min", 1.5)) & (df["pct"] <= sel.get("tr_pct_max", 7.0))]
        df = df[(df["close"] < df["high_limit"] - 1e-9) & (df["low"] < df["high"])
                & (df["close"] > df["open"])]
        df = df[df["close_pos"] >= sel.get("tr_close_pos_min", 0.55)]
        df = df[df["mom20"].between(sel.get("tr_mom20_min", 3.0), sel.get("tr_mom20_max", 35.0))]
        df = df[df["dist_high20"] >= sel.get("tr_dist_high20_min", -1.0)]
        df = df[df["vol_ratio"].between(sel.get("tr_vol_ratio_min", 1.0), sel.get("tr_vol_ratio_max", 3.0))]
        df = df[df["bias20"] <= sel.get("tr_bias20_max", 15.0)]
    elif kind == "pullback":
        df = df[(df["pct"] >= -5.0) & (df["pct"] <= 0.5)]          # 阴线或微涨
        df = df[df["close_pos"] <= 0.55]                           # 收在下沿
        df = df[df["vol_ratio"] <= 1.5]                            # 缩量
        df = df[df["close_adj"] >= df["ma20"] * 0.995]             # 不破20日线
        df = df[df["mom20"].between(2.0, 40.0)]
    else:
        df = df[df["mom20"].between(3.0, 40.0)]
        df = df[df["bias20"] <= 18.0]
    if len(df) < MIN_POOL:
        return None
    sc = _score(df, kind).sort_values(ascending=False)
    return str(df.loc[sc.index[0], "ts_code"])


def _row(day, code):
    r = day.loc[day["ts_code"] == code] if code else None
    return r.iloc[0] if (r is not None and len(r)) else None


def run(ctx, cfg, kind, hold, stop, mkt_gate, gmin=-2.0, gmax=5.0):
    dates, by_date = ctx["main_dates"], ctx["by_date"]
    reg = ctx.get("reg")
    cash, units, held = 1.0, 0.0, None
    entry_j, entry_px, pending_exit = None, None, False
    nav_curve, trades = [], []
    pick = None
    for j, d in enumerate(dates):
        day = by_date.get(d)
        if day is None:
            nav_curve.append(cash)
            continue
        # ---- 开盘: 卖出(到期 或 昨日触发止损) ----
        if held and (pending_exit or (j - entry_j) >= hold):
            r = _row(day, held)
            if r is not None and bool(r["is_paused"] == 0) and float(r["vol"]) > 0:
                if float(r["open"]) <= float(r["low_limit"]) + 1e-9:
                    px = float(r["close_adj"])
                else:
                    px = float(r["open_adj"])
                got = units * px * (1 - SLIP) * (1 - COMM - STAMP)
                trades.append({"code": held, "entry_j": entry_j, "exit_j": j,
                               "hold": j - entry_j, "stopped": bool(pending_exit),
                               "ret_pct": round((got / (units * entry_px) - 1) * 100, 3)})
                cash, units, held, entry_j, entry_px, pending_exit = got, 0.0, None, None, None, False
        # ---- 开盘: 买入 ----
        if held is None and pick is not None and cash > 0:
            ok_reg = True
            if mkt_gate and reg is not None and d in reg.index:
                ok_reg = bool(reg.loc[d, "trade_ok"])
            r = _row(day, pick)
            if ok_reg and r is not None and bool(r["is_paused"] == 0) and float(r["vol"]) > 0 \
                    and float(r["open"]) < float(r["high_limit"]) - 1e-9:
                gap = (float(r["open"]) / float(r["pre_close"]) - 1) * 100
                if gmin <= gap <= gmax:
                    px = float(r["open_adj"]) * (1 + SLIP) * (1 + COMM)
                    units, cash = cash / px, 0.0
                    held, entry_j, entry_px = pick, j, px
        # ---- 收盘: 止损检查 + 选股 + 净值 ----
        r = _row(day, held) if held else None
        if r is not None:
            nav = cash + units * float(r["close_adj"])
            if stop and not pending_exit:
                if (float(r["close_adj"]) / entry_px - 1) <= stop:
                    pending_exit = True
        else:
            nav = cash
        pick = _pick(day, cfg, kind)
        nav_curve.append(nav)
    return _stats(nav_curve, trades), trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--out", default="output/exp_trend_single2.json")
    args = ap.parse_args()

    cfg = load_config()
    ctx = prepare_context(cfg, args.days)
    print(f"窗口 {len(ctx['main_dates'])} 日", flush=True)

    out = {"days": args.days, "n_days": len(ctx["main_dates"]), "matrix": {}, "yearly": {}}
    for kind in ("breakout", "pullback", "stack"):
        for hold in (3, 5, 10, 20):
            for stop in (0.0, -0.06, -0.10):
                for mg in (False, True):
                    key = f"{kind}_H{hold}_s{int(stop*100)}_{'mkt' if mg else 'all'}"
                    st, tr = run(ctx, cfg, kind, hold, stop, mg)
                    out["matrix"][key] = st
                    by = {}
                    for t in tr:
                        by.setdefault(str(ctx["main_dates"][t["exit_j"]])[:4], []).append(t["ret_pct"])
                    out["yearly"][key] = {y: {"n": len(v), "avg": round(float(np.mean(v)), 2)}
                                          for y, v in sorted(by.items())}
                    print(f"{key:32s} 年化{st['ann']:+7.1f}% 回撤{st['mdd']:+7.1f}% "
                          f"卡玛{st['calmar'] or 0:5.2f} 笔{st['trades']:4d} 胜{st['win'] or 0:5.1f}% "
                          f"单笔{st['avg'] or 0:+6.3f}% 止损{st['stops']:3d} 仓位{st['expo']:5.1f}%", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, default=float), encoding="utf-8")
    pos = {k: v for k, v in out["matrix"].items() if v["ann"] > 0}
    print("\n正收益组合:", len(pos), "/", len(out["matrix"]))
    for k, v in sorted(pos.items(), key=lambda x: -x[1]["ann"])[:8]:
        print(f"  {k:32s} {v['ann']:+6.1f}% 回撤{v['mdd']:+6.1f}% 卡玛{v['calmar']}")


if __name__ == "__main__":
    main()
