# -*- coding: utf-8 -*-
"""每日一只副策略: 趋势过滤的稳健性矩阵(2026-11-05, 第一百三十八轮之四)。

第三轮在 H=40 上试了 8 个过滤, 只有"站上 20 日线"同时改善收益与回撤
(+7.7%→+9.5%, -27.4%→-23.8%, 卡玛 0.28→0.40)。单一 H 上的单点提升很容易是噪声,
本轮把它放到 8 个持有期上做矩阵: 若 above_ma20 在**每一档 H** 都优于 base,
才认定是真效应; 若只在 H=40 好, 就是挑参数。

同时输出分年度收益(2012~2026), 看是否集中在某一段。

用法: python3 tools/exp_sat_daily4.py --days 1500
      python3 tools/exp_sat_daily4.py --days 3378
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
HS = [20, 30, 40, 50, 60, 80, 100, 120]
VARIANTS = ["base", "above_ma20", "mom_gt_0", "ma20_mom"]


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
    ap.add_argument("--out", default="output/exp_sat_daily4.json")
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

    print("[2/4] 逐日排序(4 个变体) ...", flush=True)
    ranks = {v: {} for v in VARIANTS}
    for d in main_dates:
        j = dpos[d]
        if j < 1 or j >= len(dates) - 1:
            continue
        base = _universe(by_date[d], cfg)
        if len(base) < MIN_POOL:
            continue
        cond = {
            "base": pd.Series(True, index=base.index),
            "above_ma20": base["close_adj"] > base["ma20"],
            "mom_gt_0": base["mom20"] > 0,
            "ma20_mom": (base["close_adj"] > base["ma20"]) & (base["mom20"] > 0),
        }
        for v in VARIANTS:
            day = base.loc[cond[v]]
            x = pd.to_numeric(day["ivol20"], errors="coerce").to_numpy(dtype=float)
            ok = np.isfinite(x) & (x > 0)
            if ok.sum() < 40:
                continue
            cc = np.asarray(day["ts_code"], dtype=object)[ok]
            vv = x[ok]
            order = np.argsort(vv, kind="mergesort")[:50]
            ranks[v][d] = [str(cc[i]) for i in order]
    sig = [d for d in main_dates if d in ranks["base"]]
    print("      有效 %d 日" % len(sig), flush=True)

    def sim(rank_map, H):
        cash, hold, turn, nav_prev = 1.0, {}, 0.0, 1.0
        rets, yrets, y = [], {}, None
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
            nav = cash + sum(h["v"] for h in hold.values())
            amt = nav / H
            for c in (rank_map.get(d) or []):
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
            r = nav / nav_prev - 1.0
            rets.append(r)
            yr = d[:4]
            yrets[yr] = yrets.get(yr, 1.0) * (1 + r)
            nav_prev = nav
        st = _stats(rets)
        st["turnover_yr"] = round(turn / 2.0 / max(1e-9, len(sig) / 250.0), 2)
        st["by_year"] = {k: round((v - 1) * 100, 1) for k, v in sorted(yrets.items())}
        return st

    print("[3/4] H x 变体矩阵 ...", flush=True)
    res = {"days": args.days, "cost_pct": COST * 100, "h": HS, "variants": VARIANTS,
           "matrix": {}, "n_days": len(sig)}
    print("  %-11s %s" % ("变体\\H", "".join("%8d" % h for h in HS)), flush=True)
    for v in VARIANTS:
        row = []
        for H in HS:
            st = sim(ranks[v], H)
            res["matrix"]["%s_H%d" % (v, H)] = st
            row.append(st["ann"])
        print("  %-11s %s" % (v, "".join("%+7.1f%%" % x for x in row)), flush=True)

    print("\n  回撤矩阵")
    print("  %-11s %s" % ("变体\\H", "".join("%8d" % h for h in HS)), flush=True)
    for v in VARIANTS:
        row = [res["matrix"]["%s_H%d" % (v, h)]["mdd"] for h in HS]
        print("  %-11s %s" % (v, "".join("%+7.1f%%" % x for x in row)), flush=True)

    print("\n  逐年(base/above_ma20, H=40)")
    for v in ["base", "above_ma20"]:
        st = res["matrix"]["%s_H40" % v]
        print("  %-11s %s" % (v, st["by_year"]), flush=True)

    out = Path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(res, ensure_ascii=False, indent=1, default=float), encoding="utf-8")
    print("结果已写 %s" % out, flush=True)


if __name__ == "__main__":
    main()
