# -*- coding: utf-8 -*-
"""每日一只副策略: 持有期细扫 + 质量过滤(2026-11-05, 第一百三十八轮之三)。

第二轮结论(1500 日, 同窗口基准 -5.86%):
  monthly_buf(月度20只+缓冲) +7.5% / -28.7% / 卡玛 0.26 / 换手 15.6
  dedupe_H40(每天一只未持有低波票, 持40日) +7.7% / -27.4% / 卡玛 0.28 / 换手 7.7
  dedupe_H20 +6.0% / dedupe_H30 +6.4% / dedupe_H60 +5.8% / dedupe_H120 +7.3%
H=40 与相邻档差 1.3~1.9pp, 可能是噪声峰值。本轮:
  1. H 细扫 25/30/35/40/45/50/60/80/100/120, 看是否存在稳定平台而非尖峰;
  2. 在 H=40 上叠加 6 个预设质量过滤(成交额/上市年限/动量/均线/振幅), 看是否一致提升。

口径同 exp_sat_daily2: 金额制、T 日收盘、往返 0.30%、股票池同主策略。
用法: python3 tools/exp_sat_daily3.py --days 1500
"""
import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context  # noqa: E402
from src.config import board_of, load_config  # noqa: E402

COST = 0.0030
AMT_MIN = 1e8
MIN_POOL = 200
H_FINE = [25, 30, 35, 40, 45, 50, 60, 80, 100, 120]

FILTERS = {
    "base":      ("无过滤", lambda d: d),
    "amt2":      ("成交额≥2亿", lambda d: d[d["amount"] >= 2e8]),
    "amt3":      ("成交额≥3亿", lambda d: d[d["amount"] >= 3e8]),
    "bars500":   ("上市≥500日", lambda d: d[d["bars"] >= 500]),
    "mom_gt_-10": ("20日涨幅>-10%", lambda d: d[d["mom20"] > -10]),
    "mom_gt_0":  ("20日涨幅>0", lambda d: d[d["mom20"] > 0]),
    "above_ma20": ("站上20日线", lambda d: d[d["close_adj"] > d["ma20"]]),
    "amp_lt5":   ("当日振幅<5%", lambda d: d[d["amp_pct"] < 5]),
    "amt2_mom":  ("≥2亿 且 20日>-10%", lambda d: d[(d["amount"] >= 2e8) & (d["mom20"] > -10)]),
}


def _stats(rets):
    r = np.asarray(rets, dtype=float)
    if len(r) == 0:
        return {}
    nav = np.cumprod(1 + r)
    years = len(r) / 250.0
    ann = float(nav[-1] ** (1 / years) - 1) if nav[-1] > 0 else -1.0
    peak = np.maximum.accumulate(nav)
    mdd = float((nav / peak - 1).min())
    sharpe = float(np.mean(r) / (np.std(r) + 1e-12) * np.sqrt(250))
    return {"ann": round(ann * 100, 2), "mdd": round(mdd * 100, 2),
            "calmar": round(ann / abs(mdd), 2) if mdd < 0 else None,
            "sharpe": round(sharpe, 2), "days": len(r), "nav": round(float(nav[-1]), 4),
            "win": round(100 * float((r > 0).mean()), 1)}


def _universe(day, cfg):
    boards = set(cfg["universe"]["boards"])
    m = ((day["is_paused"] == 0) & (day["is_st"] == 0) & (day["bars"] >= 250)
         & (day["amount"] >= AMT_MIN))
    df = day.loc[m]
    return df.loc[df["ts_code"].map(lambda c: board_of(c) in boards)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--out", default="output/exp_sat_daily3.json")
    args = ap.parse_args()

    cfg = load_config()
    print("[1/4] 构建上下文 %d 日 ..." % args.days, flush=True)
    t0 = time.time()
    ctx = prepare_context(cfg, args.days)
    by_date, main_dates = ctx["by_date"], ctx["main_dates"]
    print("      %.0fs, %d 日" % (time.time() - t0, len(main_dates)), flush=True)

    dates = sorted(by_date)
    dpos = {d: j for j, d in enumerate(dates)}
    codes = sorted({c for d in dates for c in by_date[d]["ts_code"]})
    cpos = {c: i for i, c in enumerate(codes)}
    close_m = np.full((len(codes), len(dates)), np.nan, dtype=np.float32)
    for j, d in enumerate(dates):
        df = by_date[d]
        close_m[df["ts_code"].map(cpos).to_numpy(), j] = df["close_adj"].to_numpy()
    ret_m = close_m[:, 1:] / close_m[:, :-1] - 1.0
    ret_m = np.where(np.isfinite(ret_m), ret_m, np.nan).astype(np.float32)

    print("[2/4] 按过滤条件逐日排序 ...", flush=True)
    ranks = {k: {} for k in FILTERS}
    pools = {k: [] for k in FILTERS}
    for k, (d) in enumerate(main_dates):
        j = dpos[d]
        if j < 1 or j >= len(dates) - 1:
            continue
        base = _universe(by_date[d], cfg)
        if len(base) < MIN_POOL:
            continue
        for key, (_, fn) in FILTERS.items():
            day = fn(base)
            v = pd.to_numeric(day["ivol20"], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(v) & (v > 0)
            if ok.sum() < 40:
                continue
            cc = np.asarray(day["ts_code"], dtype=object)[ok]
            vv = v[ok]
            order = np.argsort(vv, kind="mergesort")[:50]
            ranks[key][d] = [str(cc[i]) for i in order]
            pools[key].append(int(ok.sum()))
        if (k + 1) % 300 == 0:
            print("      %d 日" % (k + 1), flush=True)
    sig = [d for d in main_dates if d in ranks["base"]]
    print("      有效 %d 日; 各过滤平均候选数: %s" % (
        len(sig), {k: int(np.mean(v)) if v else 0 for k, v in pools.items()}), flush=True)

    def sim(rank_map, H):
        cash, hold, turn, nav_prev, rets = 1.0, {}, 0.0, 1.0, []
        for d in sig:
            j = dpos[d]
            for c, h in list(hold.items()):
                r = ret_m[cpos[c], j - 1]
                if np.isfinite(r):
                    h["v"] *= (1 + float(r))
            for c in [c for c, h in hold.items() if h["exp"] <= j]:
                v = hold.pop(c)["v"]
                cash += v * (1 - COST / 2)
                turn += v
            lst = rank_map.get(d) or []
            nav = cash + sum(h["v"] for h in hold.values())
            amt = nav / H
            for c in lst:
                if c in hold:
                    continue
                if cash < amt * 0.2:
                    break
                use = min(amt, cash)
                cash -= use
                hold[c] = {"v": use * (1 - COST / 2), "exp": j + H}
                turn += use
                break
            nav = cash + sum(h["v"] for h in hold.values())
            rets.append(nav / nav_prev - 1.0)
            nav_prev = nav
        st = _stats(rets)
        st["turnover_yr"] = round(turn / 2.0 / max(1e-9, len(sig) / 250.0), 2)
        return st

    res = {"days": args.days, "cost_pct": COST * 100, "h_fine": H_FINE,
           "filters": {k: v[0] for k, v in FILTERS.items()}, "strategies": {}}

    print("[3/4] H 细扫(无过滤) ...", flush=True)
    for H in H_FINE:
        st = sim(ranks["base"], H)
        st["H"] = H
        res["strategies"]["base_H%d" % H] = st
        print("      H=%-4d %+7.1f%%/年 回撤 %6.1f%% 卡玛 %.2f 换手 %.1f" % (
            H, st["ann"], st["mdd"], st["calmar"] or 0, st["turnover_yr"]), flush=True)

    print("[4/4] 过滤变体(H=40) ...", flush=True)
    for key in FILTERS:
        if key == "base":
            continue
        st = sim(ranks[key], 40)
        st["H"] = 40
        res["strategies"][key] = st
        print("      %-12s %+7.1f%%/年 回撤 %6.1f%% 卡玛 %.2f 换手 %.1f" % (
            key, st["ann"], st["mdd"], st["calmar"] or 0, st["turnover_yr"]), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("结果已写 %s" % out, flush=True)


if __name__ == "__main__":
    main()
