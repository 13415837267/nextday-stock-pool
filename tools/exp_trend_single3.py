# -*- coding: utf-8 -*-
"""单吊 + 趋势过滤 + 低波排序(2026-11-05, 第一百三十九轮之三)。

前两轮已证伪"单吊趋势":
  · 突破(阳线+近新高)     全史 H20 最好 -1.2%/回撤 -73.5%
  · 趋势内回调(pullback)   1500日最好 +12.4%/回撤 -76.1%/卡玛 0.16
  · 纯均线多头(stack)      1500日最好 +3.2%/回撤 -88.6%
止损(-6%/-10%)与大势过滤都救不回来 —— 单吊满仓下, 个股尾部风险全额落到净值上。

本轮换一个问法: **同样单吊, 把"上涨趋势"当过滤器而不是选股器**——
  池 = 均线多头(趋势向上) ∩ 非ST ∩ 成交额≥1亿
  排序 = 20日特质波动(ivol) 升序, 取最稳的那只
  对照 = 不要求均线多头, 纯低波排序(看趋势过滤是加分还是减分)

若"趋势+低波"优于"纯低波", 说明"在上涨趋势里挑最稳的票"这个读法成立,
副策略就能同时满足"单吊架构"与"选上涨趋势票"; 若劣于, 则趋势过滤是负贡献,
必须如实告诉用户: 单吊 + 上涨趋势 在 A 股全时段不成立。

用法: python3 tools/exp_trend_single3.py --days 1500 --out output/exp_trend_single3.json
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
            "expo": round(100.0 * sum(t["hold"] for t in trades) / max(1, len(nav)), 1)}


def _cand(day, cfg, trend, max_ivol):
    uni = cfg["universe"]
    boards = set(uni["boards"])
    d = day
    m = ((d["is_paused"] == 0) & (d["is_st"] == 0) & (d["bars"] >= MIN_BARS)
         & (d["amount"] >= AMT_MIN) & (d["ivol20"] > 0))
    m &= d["ts_code"].map(lambda c: board_of(c) in boards).to_numpy()
    if trend:
        m &= ((d["close_adj"] > d["ma5"]) & (d["ma5"] > d["ma10"])
              & (d["ma10"] > d["ma20"]) & (d["ma20"] > d["ma20_prev"])).to_numpy()
        m &= (d["mom20"] > 0).to_numpy()
    df = d.loc[m]
    if max_ivol:
        df = df[df["ivol20"] <= max_ivol]
    if len(df) < MIN_POOL:
        return None
    return str(df.loc[df["ivol20"].idxmin(), "ts_code"])


def _row(day, code):
    r = day.loc[day["ts_code"] == code] if code else None
    return r.iloc[0] if (r is not None and len(r)) else None


def run(ctx, cfg, trend, hold, max_ivol=None, gmin=-3.0, gmax=5.0):
    dates, by_date = ctx["main_dates"], ctx["by_date"]
    cash, units, held = 1.0, 0.0, None
    entry_j, entry_px = None, None
    nav_curve, trades = [], []
    pick = None
    for j, d in enumerate(dates):
        day = by_date.get(d)
        if day is None:
            nav_curve.append(cash)
            continue
        if held and (j - entry_j) >= hold:
            r = _row(day, held)
            if r is not None and bool(r["is_paused"] == 0) and float(r["vol"]) > 0:
                px = float(r["close_adj"]) if float(r["open"]) <= float(r["low_limit"]) + 1e-9 \
                    else float(r["open_adj"])
                got = units * px * (1 - SLIP) * (1 - COMM - STAMP)
                trades.append({"code": held, "entry_j": entry_j, "exit_j": j, "hold": j - entry_j,
                               "ret_pct": round((got / (units * entry_px) - 1) * 100, 3)})
                cash, units, held, entry_j, entry_px = got, 0.0, None, None, None
        if held is None and pick is not None and cash > 0:
            r = _row(day, pick)
            if r is not None and bool(r["is_paused"] == 0) and float(r["vol"]) > 0 \
                    and float(r["open"]) < float(r["high_limit"]) - 1e-9:
                gap = (float(r["open"]) / float(r["pre_close"]) - 1) * 100
                if gmin <= gap <= gmax:
                    px = float(r["open_adj"]) * (1 + SLIP) * (1 + COMM)
                    units, cash = cash / px, 0.0
                    held, entry_j, entry_px = pick, j, px
        r = _row(day, held) if held else None
        nav_curve.append(cash + units * float(r["close_adj"]) if (r is not None and units) else cash)
        pick = _cand(day, cfg, trend, max_ivol)
    return _stats(nav_curve, trades), trades


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--out", default="output/exp_trend_single3.json")
    args = ap.parse_args()

    cfg = load_config()
    ctx = prepare_context(cfg, args.days)
    print(f"窗口 {len(ctx['main_dates'])} 日", flush=True)

    out = {"days": args.days, "n_days": len(ctx["main_dates"]), "matrix": {}, "yearly": {}}
    for trend in (True, False):
        for hold in (10, 20, 40, 60):
            for miv in (None, 2.5):
                key = f"{'trend' if trend else 'pure'}_H{hold}_iv{miv or 'all'}"
                st, tr = run(ctx, cfg, trend, hold, miv)
                out["matrix"][key] = st
                by = {}
                for t in tr:
                    by.setdefault(str(ctx["main_dates"][t["exit_j"]])[:4], []).append(t["ret_pct"])
                out["yearly"][key] = {y: {"n": len(v), "avg": round(float(np.mean(v)), 2)}
                                      for y, v in sorted(by.items())}
                print(f"{key:24s} 年化{st['ann']:+7.1f}% 回撤{st['mdd']:+7.1f}% "
                      f"卡玛{st['calmar'] or 0:5.2f} 笔{st['trades']:4d} 胜{st['win'] or 0:5.1f}% "
                      f"单笔{st['avg'] or 0:+6.3f}% 仓位{st['expo']:5.1f}%", flush=True)
    Path(args.out).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(out, ensure_ascii=False, default=float), encoding="utf-8")


if __name__ == "__main__":
    main()
