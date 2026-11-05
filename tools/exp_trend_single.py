# -*- coding: utf-8 -*-
"""单吊趋势策略回测(2026-11-05, 第一百三十九轮)。

用户要求: 副策略与主策略架构一致 —— **单吊**(全仓、每天最多买1只、卖出后资金当日
再投), 但选股换成"上涨趋势票"。此前(第一百三十二/三轮)测的是"趋势票补 dip 的位",
持有 T+2, 单笔 -0.35%; 那是**短持有 + 补位**口径, 与"单吊趋势"不是一回事。

本脚本只做单吊趋势:
  第 T 日收盘  -> 在趋势池里取评分第 1 名
  第 T+1 日开盘 -> 若 gap 在 [gap_min, gap_max] 内、未一字涨停, 全仓买入
  持有 H 个交易日 -> 第 T+1+H 日开盘卖出(开盘跌停则按收盘), 卖出当日可再买
  无候选/不达标 -> 空仓等待(与主策略"不到标准不成交"同构)

扫描: H x gap 上限 x 卖出时点。

用法: python3 tools/exp_trend_single.py --days 1500 --out output/exp_trend_single.json
      python3 tools/exp_trend_single.py --days 3378 --out output/exp_trend_single_full.json
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

COMM = 0.00025
STAMP = 0.0005
SLIP = 0.001
AMT_MIN = 1e8
MIN_BARS = 250
MIN_POOL = 5          # 趋势池少于该数则视为"没票"


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
            "expo": round(100.0 * sum(t["hold"] for t in trades) / max(1, len(nav)), 1)}


def _trend_score(day, cfg):
    """返回 (codes, scores) —— 趋势池内按评分降序。条件与 selector.trend 对齐。"""
    sel = cfg["selection"]
    uni = cfg["universe"]
    boards = set(uni["boards"])
    d = day
    m = ((d["is_paused"] == 0) & (d["is_st"] == 0) & (d["bars"] >= MIN_BARS)
         & (d["amount"] >= AMT_MIN) & (d["turnover_rate"] <= uni["turnover_max"]))
    m &= d["ts_code"].map(lambda c: board_of(c) in boards).to_numpy()
    m &= (d["pct"] >= sel.get("tr_pct_min", 1.5)).to_numpy() & (d["pct"] <= sel.get("tr_pct_max", 7.0)).to_numpy()
    m &= (d["close"] < d["high_limit"] - 1e-9).to_numpy()
    m &= (d["low"] < d["high"]).to_numpy()
    m &= (d["close"] > d["open"]).to_numpy()
    m &= (d["close_pos"] >= sel.get("tr_close_pos_min", 0.55)).to_numpy()
    m &= ((d["close_adj"] > d["ma5"]) & (d["ma5"] > d["ma10"])
          & (d["ma10"] > d["ma20"]) & (d["ma20"] > d["ma20_prev"])).to_numpy()
    m &= d["mom20"].between(sel.get("tr_mom20_min", 3.0), sel.get("tr_mom20_max", 35.0)).to_numpy()
    m &= (d["dist_high20"] >= sel.get("tr_dist_high20_min", -1.0)).to_numpy()
    m &= d["vol_ratio"].between(sel.get("tr_vol_ratio_min", 1.0), sel.get("tr_vol_ratio_max", 3.0)).to_numpy()
    m &= (d["bias20"] <= sel.get("tr_bias20_max", 15.0)).to_numpy()
    df = d.loc[m]
    if len(df) < MIN_POOL:
        return None
    z = lambda s: (s - s.mean()) / (s.std() + 1e-9) if s.std() > 0 else s * 0
    score = (0.30 * z(df["dist_high20"]) + 0.25 * z(df["mom20"]) + 0.20 * z(df["close_pos"])
             + 0.15 * z(-df["bias20"]) + 0.10 * z(-np.log1p(df["amount"])))
    order = score.sort_values(ascending=False)
    return [(str(c), float(s)) for c, s in zip(df.loc[order.index, "ts_code"], order)]


def _row(day, code):
    if not code:
        return None
    r = day.loc[day["ts_code"] == code]
    return r.iloc[0] if len(r) else None


def run(ctx, cfg, hold, gap_min, gap_max, sell_mode, cost=True):
    """单吊模拟。sell_mode: open=开盘卖, close=收盘卖。"""
    dates, by_date = ctx["main_dates"], ctx["by_date"]
    cash, units, held = 1.0, 0.0, None
    entry_j, entry_px = None, None
    nav_curve, trades = [], []
    pick = None                      # 昨日收盘选出的票
    for j, d in enumerate(dates):
        day = by_date.get(d)
        if day is None:
            nav_curve.append(cash + units * (entry_px or 1.0))
            continue
        # ---------- 开盘: 到期卖出 ----------
        if held and (j - entry_j) >= hold:
            r = _row(day, held)
            if r is not None and bool(r["is_paused"] == 0) and float(r["vol"]) > 0:
                if float(r["open"]) <= float(r["low_limit"]) + 1e-9:
                    px = float(r["close_adj"])        # 开盘跌停 -> 按收盘
                elif sell_mode == "close":
                    px = float(r["close_adj"])
                else:
                    px = float(r["open_adj"])
                got = units * px
                if cost:
                    got *= (1 - SLIP) * (1 - COMM - STAMP)
                trades.append({"code": held, "entry_j": entry_j, "exit_j": j,
                               "hold": j - entry_j,
                               "ret_pct": round((got / (units * entry_px) - 1) * 100, 3)})
                cash, units, held, entry_j, entry_px = got, 0.0, None, None, None
            # 停牌: 顺延到下一交易日
        # ---------- 开盘: 空仓买入 ----------
        if held is None and pick is not None and cash > 0:
            r = _row(day, pick[0])
            if r is not None and bool(r["is_paused"] == 0) and float(r["vol"]) > 0 \
                    and float(r["open"]) < float(r["high_limit"]) - 1e-9:
                gap = (float(r["open"]) / float(r["pre_close"]) - 1) * 100
                if gap_min <= gap <= gap_max:
                    px = float(r["open_adj"])
                    if cost:
                        px *= (1 + SLIP) * (1 + COMM)
                    units, cash = cash / px, 0.0
                    held, entry_j, entry_px = pick[0], j, px
        # ---------- 收盘: 选股 + 净值 ----------
        cand = _trend_score(day, cfg)
        pick = cand[0] if cand else None
        r = _row(day, held) if held else None
        nav_curve.append(cash + units * float(r["close_adj"]) if (r is not None and units) else cash)
    return _stats(nav_curve, trades), trades, nav_curve


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--out", default="output/exp_trend_single.json")
    args = ap.parse_args()

    cfg = load_config()
    ctx = prepare_context(cfg, args.days)
    print(f"窗口 {len(ctx['main_dates'])} 日", flush=True)

    GRID = []
    for hold in (1, 2, 3, 5, 10, 20):
        for gm in (-2.0, -5.0, -99.0):
            for gx in (2.0, 5.0, 99.0):
                GRID.append((hold, gm, gx, "open"))
    GRID += [(5, -99.0, 99.0, "close"), (10, -99.0, 99.0, "close"), (20, -99.0, 99.0, "close")]

    out = {"days": args.days, "n_days": len(ctx["main_dates"]), "matrix": {},
           "trades": {}, "yearly": {}}
    for hold, gm, gx, sm in GRID:
        key = f"H{hold}_g{gm:g}~{gx:g}_{sm}"
        st, tr, nav = run(ctx, cfg, hold, gm, gx, sm)
        st["gap_min"], st["gap_max"], st["sell"] = gm, gx, sm
        out["matrix"][key] = st
        out["trades"][key] = tr
        print(f"{key:26s} 年化{st['ann']:+7.1f}% 回撤{st['mdd']:+7.1f}% "
              f"卡玛{st['calmar'] or 0:5.2f} 笔{st['trades']:4d} 胜{st['win'] or 0:5.1f}% "
              f"单笔{st['avg'] or 0:+6.3f}% 仓位{st['expo']:5.1f}%", flush=True)

    best = max(out["matrix"], key=lambda k: (out["matrix"][k].get("calmar") or -9))
    tr = out["trades"][best]
    by = {}
    for t in tr:
        y = str(ctx["main_dates"][t["exit_j"]])[:4]
        by.setdefault(y, []).append(t["ret_pct"])
    out["yearly"][best] = {y: {"n": len(v), "avg": round(float(np.mean(v)), 2),
                               "win": round(100 * float(np.mean([x > 0 for x in v])), 1)}
                           for y, v in sorted(by.items())}
    out["best"] = best
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, default=float), encoding="utf-8")
    print("best", best, out["matrix"][best], flush=True)
    print("yearly", out["yearly"][best], flush=True)


if __name__ == "__main__":
    main()
