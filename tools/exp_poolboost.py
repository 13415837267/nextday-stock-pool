# -*- coding: utf-8 -*-
"""第一百七十五轮(主攻): 池深加仓 —— 策略自己看到的恐慌强度。

为什么是它
----------
前 174 轮所有的仓位规则都建立在**市场维度**上(涨跌幅、波动率、宽度、指数均线)。
但 dip 策略有一个自己独有的、一直只被用来"降仓"的指标: **当日合格池深**。
第 99 轮加的 `pool_n_tiers` 只做一件事 —— 池子浅就降仓(≥21 满仓 / 11~20 半仓 / 1~10 三成),
即"池子浅说明形态不成立, 别下重手"。反方向"池子深就加仓"从来没人测过。

200 笔样本按池深分档(2023-04~2026-10):
    ≤5 只   2 笔 +3.99%      5~10 只  15 笔 +0.45%
    10~20   46 笔 +0.63%     20~40 只 58 笔 +0.84%
    40 只以上 66 笔 **+2.21%**
最高档与最低正档差 1.58pp, 且是单调的。

为什么它可能不是 mkt_boost 的重复
---------------------------------
池深度量的是"**有多少票同时被打成缩量阴线**", 市场涨跌幅度量的是"**平均跌了多少**"。
两者在两种情形下会分开:
  · 普跌但幅度温和(-0.8%): 池深很大, 但 mkt_boost 的 [-2.5,-1.0] 区间不触发;
  · 少数权重股砸指数(-2%): 池深很小, mkt_boost 触发但池子空了。
所以本轮的诊断必须给出两者的交叉表 —— 若池深高等价于 mkt 大跌, 那就没有增量。

用法: python tools/exp_poolboost.py [days] [组名,...]
输出: output/exp_poolboost.json
"""
from __future__ import annotations

import copy
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["BT_FULL_TRADES"] = "1"

from src.backtest import prepare_context, run_backtest       # noqa: E402
from src.config import load_config                          # noqa: E402

OUT = os.path.join(ROOT, "output")

GROUPS = [
    ("base",    1.25, []),
    ("p30x13",  1.25, [{"min": 30, "mult": 1.3}]),
    ("p40x13",  1.25, [{"min": 40, "mult": 1.3}]),
    ("p40x15",  1.25, [{"min": 40, "mult": 1.5}]),
    ("p60x15",  1.25, [{"min": 60, "mult": 1.5}]),
    ("p100x15", 1.25, [{"min": 100, "mult": 1.5}]),
    # 等效敞口对照: p40x15 的均敞口 1.373 vs base 1.121, 即 +22.5%。
    # 1.25 * 1.373/1.121 = 1.531 —— 用纯杠杆达到同样敞口, 才能判断池深是真 alpha 还是杠杆变形。
    ("L153",    1.53, []),
]


def yearly(equity):
    by: dict[str, list] = {}
    for e in equity:
        by.setdefault(str(e["d"])[:4], []).append((str(e["d"]), e["nav"]))
    out, prev = {}, None
    for y in sorted(by):
        pts = by[y]
        start = prev if prev is not None else pts[0][1]
        out[y] = round((pts[-1][1] / start - 1) * 100, 2)
        prev = pts[-1][1]
    return out


def seg_split(trades):
    real = [t for t in trades if t.get("ret_pct") is not None]
    out = {}
    for k, f in (("pre", lambda d: d < "20240930"), ("post", lambda d: d >= "20240930")):
        v = [float(t["ret_pct"]) for t in real if f(str(t.get("sell_date", "")))]
        out[k] = {"n": len(v), "mean": round(float(np.mean(v)), 3) if v else None}
    return out


def mkt_pct_map(ctx, dates):
    """信号日全市场个股涨跌中位数(百分点), 用于与池深做交叉。只算成交出现的日期。"""
    out = {}
    for d in dates:
        df = ctx["by_date"].get(d)
        if df is None:
            continue
        v = df["pct"].to_numpy(dtype="float64")
        v = v[~np.isnan(v)]
        if len(v):
            out[d] = float(np.median(v))
    return out


def diagnose(trades, mkt):
    real = pd.DataFrame([t for t in trades if t.get("ret_pct") is not None])
    if real.empty:
        return {}
    real["r"] = real["ret_pct"].astype(float) / 100.0
    real["pn"] = real["pool_n"].astype(float)
    out = {"n": int(len(real))}

    bins = [0, 10, 20, 30, 40, 60, 100, 10**9]
    lab = ["0-10", "10-20", "20-30", "30-40", "40-60", "60-100", "100+"]
    real["pb"] = pd.cut(real["pn"], bins, labels=lab)
    g = real.groupby("pb", observed=False).agg(n=("r", "size"), mean=("r", "mean"),
                                               std=("r", "std"), win=("r", lambda x: (x > 0).mean()))
    out["pool"] = {str(k): {"n": int(g.loc[k, "n"]),
                            "mean_pct": round(float(g.loc[k, "mean"]) * 100, 3),
                            "std_pct": round(float(g.loc[k, "std"]) * 100, 3),
                            "win_pct": round(float(g.loc[k, "win"]) * 100, 1)}
                   for k in lab if not pd.isna(g.loc[k, "mean"])}

    # 与市场跌幅交叉: 池深>=40 且 mkt 在 boost 区间内 / 外
    real["m"] = real["signal_date"].astype(str).map(mkt)
    real = real.dropna(subset=["m"])
    boost_in = (real["m"] <= -1.0) & (real["m"] >= -2.5)
    deep = real["pn"] >= 40
    cells = {
        "deep&boost": [real["r"][deep & boost_in], "池深>=40 且 市场跌1~2.5%"],
        "deep&other": [real["r"][deep & ~boost_in], "池深>=40 但市场不在该区间"],
        "shallow&boost": [real["r"][~deep & boost_in], "池深<40 且 市场跌1~2.5%"],
        "shallow&other": [real["r"][~deep & ~boost_in], "池深<40 且市场不在该区间"],
    }
    out["cross"] = {k: {"desc": v[1], "n": int(len(v[0])),
                        "mean_pct": round(float(v[0].mean()) * 100, 3) if len(v[0]) else None}
                    for k, v in cells.items()}
    out["corr_pool_r"] = round(float(real[["pn", "r"]].corr().iloc[0, 1]), 3)
    out["corr_pool_mkt"] = round(float(real[["pn", "m"]].corr().iloc[0, 1]), 3)
    return out


def main(days: int = 1500, only: list[str] | None = None) -> None:
    cfg0 = load_config()
    print(f"\n[prepare] days={days}", flush=True)
    t0 = time.time()
    ctx = prepare_context(cfg0, days)
    print(f"[prepare] done in {time.time() - t0:.0f}s\n", flush=True)

    rows, diag = [], {}
    print("%-10s %5s %11s %9s %8s %6s %6s %5s %7s %8s"
          % ("组", "lev", "总收益%", "年化%", "回撤%", "卡玛", "夏普", "笔数", "单笔%", "均杠杆"))
    print("-" * 96, flush=True)

    for name, lev, boost in GROUPS:
        if only and name not in only:
            continue
        cfg = copy.deepcopy(cfg0)
        cfg["execution"]["leverage"] = lev
        cfg["execution"]["pool_n_boost_tiers"] = boost
        t1 = time.time()
        r = run_backtest(cfg, days, ctx)
        m = r["metrics"]
        trades = r.get("trades_all") or []
        real = [t for t in trades if t.get("ret_pct") is not None]
        avg_lev = float(np.mean([float(t.get("pos_frac") or 1.0) for t in real])) * lev if real else 0.0
        yr = yearly(r["equity"])
        sg = seg_split(trades)
        rows.append({"name": name, "lev": lev, "pool_boost": boost, "metrics": m,
                     "year": yr, "seg": sg, "avg_eff_lev": round(avg_lev, 3),
                     "secs": round(time.time() - t1)})
        print("%-10s %5.2f %11.1f %9.2f %8.2f %6.2f %6.2f %5d %7.3f %8.3f"
              % (name, lev, m["总收益率%"], m["年化收益率%"], m["最大回撤%"],
                 m["卡玛比率"] or 0, m["夏普比率"], m["交易次数"],
                 m["平均单笔收益%"], avg_lev), flush=True)
        print("           前段 n=%d %s%%  后段 n=%d %s%%   耗时 %.0fs"
              % (sg["pre"]["n"], sg["pre"]["mean"], sg["post"]["n"], sg["post"]["mean"],
                 time.time() - t1), flush=True)

        if name == "base" and not diag:
            _sig_dates = sorted({str(t["signal_date"]) for t in trades})
            diag = diagnose(trades, mkt_pct_map(ctx, _sig_dates))
            if diag:
                print("\n  === 诊断: %d 笔按当日合格池深 ===" % diag["n"], flush=True)
                for k, v in diag["pool"].items():
                    print("    池深 %-7s n=%3d 单笔 %+7.3f%%  std %5.3f%%  胜率 %4.1f%%"
                          % (k, v["n"], v["mean_pct"], v["std_pct"], v["win_pct"]), flush=True)
                print("    corr(池深, 单笔) = %+.3f   corr(池深, 市场跌幅) = %+.3f"
                      % (diag["corr_pool_r"], diag["corr_pool_mkt"]), flush=True)
                print("    --- 与市场跌幅交叉 ---", flush=True)
                for k, v in diag["cross"].items():
                    print("      %-28s n=%3d 单笔 %s%%" % (v["desc"], v["n"], v["mean_pct"]),
                          flush=True)
                print("", flush=True)

        os.makedirs(OUT, exist_ok=True)
        fn = "exp_poolboost.json" if days <= 1600 else f"exp_poolboost_{days}.json"
        with open(os.path.join(OUT, fn), "w", encoding="utf-8") as f:
            json.dump({"days": days, "rows": rows, "diag": diag}, f, ensure_ascii=False, indent=1)

    print("\n=== 分年收益 ===")
    years = sorted({y for x in rows for y in x["year"]})
    print("%-6s " % "年" + " ".join(f"{x['name']:>10s}" for x in rows))
    for y in years:
        print("%-6s " % y + " ".join(f"{x['year'].get(y, 0):>10.1f}" for x in rows))
    print("\nsaved output/exp_poolboost*.json")


if __name__ == "__main__":
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    o = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else None
    main(d, o)
