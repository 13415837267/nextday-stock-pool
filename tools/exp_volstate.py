# -*- coding: utf-8 -*-
"""第一百七十五轮: 波动率状态加仓 —— 从"事件"维度换到"状态"维度。

为什么试这个
------------
第一百七十三轮的 `mkt_boost_tiers` 用的是**信号日单日跌幅**(事件): 全市场跌 1%~2.5% 时
仓位 ×1.5, 双窗口回撤逐位不变、年化 +22~28%。它是"当天被错杀"。
但还有另一种情形它抓不到: **一段持续阴跌**里每天都只跌 0.3%, 单日跌幅永远够不上 -1%,
可市场 20 日波动率已经抬到高位, 抛压是持续释放的。这就是"状态"维度。

先验证据(2023-04~2026-10 的 200 笔, 按信号日市场 20 日波动率五分位):
    最低档 +1.05%   第二档 +0.11%   中档 +1.24%   第四档 +1.70%   最高档 +2.24%
    corr(vol20, 单笔收益) = +0.163, Spearman +0.103
单调性不完美(第二档塌陷), 但高波动档显著更好, 且这个方向与 vol-targeting
(低波加仓、高波减仓)完全相反 —— 所以"控制波动"在这个策略上会直接砍掉收益。

本轮做的事
----------
1. 引擎新增 `execution.vol_boost_tiers = [{vol_above, vol_below, mult}]`,
   信号日市场 N 日波动率(百分点)落在区间内时仓位 ×mult。与 mkt_boost_tiers 独立叠乘。
2. 诊断: 全量样本按 vol 分层、以及 vol × 是否命中事件加仓 的交叉, 判断两者是否重复。
3. 扫阈值与倍数, 并做 2024-09 前后切分。

用法: python tools/exp_volstate.py [days] [组名,...]
输出: output/exp_volstate.json
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

# name, leverage, vol_boost_tiers
GROUPS = [
    ("base",        1.25, []),
    ("v10x12",      1.25, [{"vol_above": 1.0, "mult": 1.2}]),
    ("v15x13",      1.25, [{"vol_above": 1.5, "mult": 1.3}]),
    ("v15x15",      1.25, [{"vol_above": 1.5, "mult": 1.5}]),
    ("v20x15",      1.25, [{"vol_above": 2.0, "mult": 1.5}]),
    ("r15_25x15",   1.25, [{"vol_above": 1.5, "vol_below": 2.5, "mult": 1.5}]),
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


def vol_series(ctx, win: int = 20) -> pd.Series:
    """市场 20 日波动率(百分点)。用风控基础序列的 mkt_pct(全市场等权日收益)。"""
    rb = ctx["reg_base"]
    return rb["mkt_pct"].rolling(win, min_periods=win // 2).std()


def diagnose(trades, vol, cfg0):
    """全量成交样本的条件分层: 波动率 / 与事件加仓的交叉。"""
    real = pd.DataFrame([t for t in trades if t.get("ret_pct") is not None])
    if real.empty:
        return {}
    real["sig"] = real["signal_date"].astype(str)
    real["r"] = real["ret_pct"].astype(float) / 100.0
    real["v20"] = real["sig"].map(vol)
    real = real.dropna(subset=["v20"])

    out: dict = {"n": int(len(real)), "span": [real["sig"].min(), real["sig"].max()]}

    q = pd.qcut(real["v20"], 5, labels=False, duplicates="drop")
    g = real.groupby(q).agg(n=("r", "size"), mean=("r", "mean"), std=("r", "std"),
                            win=("r", lambda x: (x > 0).mean()))
    out["vol5"] = {
        "edges": [round(float(x), 3) for x in real.groupby(q)["v20"].max().values],
        "n": [int(x) for x in g["n"]],
        "mean_pct": [round(float(x) * 100, 3) for x in g["mean"]],
        "std_pct": [round(float(x) * 100, 3) for x in g["std"]],
        "win_pct": [round(float(x) * 100, 1) for x in g["win"]],
    }

    # 八分位: 看末端是否反转(崩塌期)
    q8 = pd.qcut(real["v20"], 8, labels=False, duplicates="drop")
    g8 = real.groupby(q8).agg(n=("r", "size"), mean=("r", "mean"))
    out["vol8_mean_pct"] = [round(float(x) * 100, 3) for x in g8["mean"]]
    out["vol8_n"] = [int(x) for x in g8["n"]]
    out["vol8_hi"] = [round(float(x), 2) for x in real.groupby(q8)["v20"].max()]

    # 分年 × 波动率高低: 排除"某一年恰好高波高收益"的年度效应
    real["yr"] = real["sig"].str[:4]
    med = float(real["v20"].median())
    real["hi_vol"] = real["v20"] > med
    piv = real.pivot_table(index="yr", columns="hi_vol", values="r", aggfunc=["size", "mean"])
    out["by_year"] = {
        str(y): {
            "lo_n": int(piv.loc[y, ("size", False)]), "lo_pct": round(float(piv.loc[y, ("mean", False)]) * 100, 3),
            "hi_n": int(piv.loc[y, ("size", True)]), "hi_pct": round(float(piv.loc[y, ("mean", True)]) * 100, 3),
        } for y in piv.index
    }

    out["corr"] = round(float(real[["v20", "r"]].corr().iloc[0, 1]), 3)
    out["spearman"] = round(float(real[["v20", "r"]].corr(method="spearman").iloc[0, 1]), 3)
    return out


def build_groups(spec: str):
    """解析 "V=1.5:1.3,2.0:1.5" -> 组定义; 可选 "L=2.0" 前缀。区间用 "1.5:2.5:1.5"。"""
    lev = 1.25
    out = []
    for part in spec.split(";"):
        part = part.strip()
        if part.startswith("L="):
            lev = float(part[2:])
            out.append((f"L{lev:g}", lev, []))      # 纯杠杆对照(无 vol_boost)
            continue
        if part.startswith("V="):
            for x in part[2:].split(","):
                seg = x.split(":")
                if len(seg) == 2:
                    lo, mult = float(seg[0]), float(seg[1])
                    tier = {"vol_above": lo, "mult": mult}
                    tag = f"v{lo:.1f}"
                else:
                    lo, hi, mult = float(seg[0]), float(seg[1]), float(seg[2])
                    tier = {"vol_above": lo, "vol_below": hi, "mult": mult}
                    tag = f"r{lo:.1f}_{hi:.1f}"
                out.append((f"{tag}x{mult:.2f}_L{lev:g}", lev, [tier]))
    return out


def main(days: int = 1500, only: list[str] | None = None, spec: str = "") -> None:
    cfg0 = load_config()
    print(f"\n[prepare] days={days}", flush=True)
    t0 = time.time()
    ctx = prepare_context(cfg0, days)
    print(f"[prepare] done in {time.time() - t0:.0f}s\n", flush=True)

    vol = vol_series(ctx)
    groups = build_groups(spec) if spec else GROUPS
    rows = []
    diag = {}
    print("%-12s %5s %11s %9s %8s %6s %6s %5s %7s %8s"
          % ("组", "lev", "总收益%", "年化%", "回撤%", "卡玛", "夏普", "笔数", "单笔%", "均杠杆"))
    print("-" * 96, flush=True)

    for name, lev, boost in groups:
        if only and name not in only:
            continue
        cfg = copy.deepcopy(cfg0)
        cfg["execution"]["leverage"] = lev
        cfg["execution"]["vol_boost_tiers"] = boost
        t1 = time.time()
        r = run_backtest(cfg, days, ctx)
        m = r["metrics"]
        trades = r.get("trades_all") or []
        real = [t for t in trades if t.get("ret_pct") is not None]
        avg_lev = float(np.mean([float(t.get("pos_frac") or 1.0) for t in real])) * lev if real else 0.0
        yr = yearly(r["equity"])
        sg = seg_split(trades)
        rows.append({"name": name, "lev": lev, "vol_boost": boost, "metrics": m,
                     "year": yr, "seg": sg, "avg_eff_lev": round(avg_lev, 3),
                     "secs": round(time.time() - t1)})
        print("%-12s %5.2f %11.1f %9.2f %8.2f %6.2f %6.2f %5d %7.3f %8.3f"
              % (name, lev, m["总收益率%"], m["年化收益率%"], m["最大回撤%"],
                 m["卡玛比率"] or 0, m["夏普比率"], m["交易次数"],
                 m["平均单笔收益%"], avg_lev), flush=True)
        print("             前段 n=%d %s%%  后段 n=%d %s%%   耗时 %.0fs"
              % (sg["pre"]["n"], sg["pre"]["mean"], sg["post"]["n"], sg["post"]["mean"],
                 time.time() - t1), flush=True)
        if name == "base" and not diag:
            diag = diagnose(trades, vol, cfg0)
            if diag:
                print("\n  === 诊断: 全量 %d 笔按市场20日波动率五分位 ===" % diag["n"], flush=True)
                print("    边界 %s" % diag["vol5"]["edges"], flush=True)
                for i in range(len(diag["vol5"]["n"])):
                    print("    Q%d n=%3d  单笔 %+7.3f%%  std %5.3f%%  胜率 %4.1f%%"
                          % (i + 1, diag["vol5"]["n"][i], diag["vol5"]["mean_pct"][i],
                             diag["vol5"]["std_pct"][i], diag["vol5"]["win_pct"][i]), flush=True)
                print("    八分位均值 %s" % diag["vol8_mean_pct"], flush=True)
                print("    八分位上界 %s" % diag["vol8_hi"], flush=True)
                print("    corr %+.3f  spearman %+.3f" % (diag["corr"], diag["spearman"]), flush=True)
                print("    分年 高波/低波 单笔:", flush=True)
                for y, v in diag["by_year"].items():
                    print("      %s  低波 n=%3d %+7.3f%%   高波 n=%3d %+7.3f%%"
                          % (y, v["lo_n"], v["lo_pct"], v["hi_n"], v["hi_pct"]), flush=True)
                print("", flush=True)

        os.makedirs(OUT, exist_ok=True)
        fn = "exp_volstate.json" if days <= 1600 else f"exp_volstate_{days}.json"
        with open(os.path.join(OUT, fn), "w", encoding="utf-8") as f:
            json.dump({"days": days, "rows": rows, "diag": diag}, f, ensure_ascii=False, indent=1)

    print("\n=== 分年收益 ===")
    years = sorted({y for x in rows for y in x["year"]})
    print("%-6s " % "年" + " ".join(f"{x['name']:>12s}" for x in rows))
    for y in years:
        print("%-6s " % y + " ".join(f"{x['year'].get(y, 0):>12.1f}" for x in rows))
    print("\nsaved output/exp_volstate*.json")


if __name__ == "__main__":
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    a2 = sys.argv[2] if len(sys.argv) > 2 else ""
    if "=" in a2:
        main(d, None, a2)
    else:
        main(d, a2.split(",") if a2 else None, "")
