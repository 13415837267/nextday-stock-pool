# -*- coding: utf-8 -*-
"""第一百七十五轮: 缺口深度加仓 —— 从"是否达标"到"达标多少"。

来自哪里
--------
本轮先做了一次"被跳过信号的反事实体检"(tools/exp_skipped.py): 979 次跳过里,
**702 次是"开盘缺口未达承接线 -1.0%"**。对它们重算"假如买了"的收益:

    缺口 > -1.0% 被剔除的 702 笔:  单笔 -0.606%   胜率 41.3%   t = -4.12
    门槛内成交的 352 笔:           单笔 +1.439%

分离度 2.05pp, t 值 -4.12 —— 这是这批因子里最强的单变量, 比市场跌幅(mkt_boost)
的分离度还大。既然门槛两侧差这么多, 门槛内部是否继续单调就是自然的下一步:
**开盘跌得越深, 是不是越该重仓**。现在承接线只做二值判断(达标/不达标), 达标之后
无论 gap 是 -1.1% 还是 -2.4% 仓位都一样, 这是明显的浪费。

本轮做的事
----------
1. 引擎新增 `execution.gap_boost_tiers = [{gap_below, gap_above, mult}]`,
   个股 T+1 开盘缺口落在区间内时仓位 ×mult。与 mkt_boost(全市场维度)正交, 独立叠乘。
2. 诊断: 成交笔按 gap 五分位 / 与市场跌幅交叉, 判断是否存在"深跌 + 弱市"的双重叠加。
3. 扫阈值与倍数, 2024-09 前后切分。

风险提示
--------
gap 越深越接近 `block_gap_below` (-2.5%), 那一侧是"个股有坏消息"。所以必须带下限,
否则会变成"在暴跌股上加杠杆"。第一百七十三轮的教训: 不加下限的加仓在全史会崩。

用法: python tools/exp_gapboost.py [days] [组名,...]
输出: output/exp_gapboost.json
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
    ("base",       1.25, []),
    ("g12x13",     1.25, [{"gap_below": -1.2, "mult": 1.3}]),
    ("g15x13",     1.25, [{"gap_below": -1.5, "mult": 1.3}]),
    ("g15x15",     1.25, [{"gap_below": -1.5, "mult": 1.5}]),
    ("g20x15",     1.25, [{"gap_below": -2.0, "mult": 1.5}]),
    ("r15_25x15",  1.25, [{"gap_above": -2.5, "gap_below": -1.5, "mult": 1.5}]),
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


def diagnose(trades):
    real = pd.DataFrame([t for t in trades if t.get("ret_pct") is not None])
    if real.empty:
        return {}
    real["g"] = real["gap_pct"].astype(float)
    real["r"] = real["ret_pct"].astype(float) / 100.0
    out: dict = {"n": int(len(real))}

    q = pd.qcut(real["g"], 6, labels=False, duplicates="drop")
    g = real.groupby(q).agg(n=("r", "size"), mean=("r", "mean"), std=("r", "std"),
                            win=("r", lambda x: (x > 0).mean()), gmin=("g", "min"), gmax=("g", "max"))
    out["gap6"] = {
        "lo": [round(float(x), 2) for x in g["gmin"]],
        "hi": [round(float(x), 2) for x in g["gmax"]],
        "n": [int(x) for x in g["n"]],
        "mean_pct": [round(float(x) * 100, 3) for x in g["mean"]],
        "std_pct": [round(float(x) * 100, 3) for x in g["std"]],
        "win_pct": [round(float(x) * 100, 1) for x in g["win"]],
    }
    # 固定阈值切: 与配置里的承接线对齐
    for lo in (-1.2, -1.5, -2.0):
        sub = real[real["g"] <= lo]
        out[f"le{lo}"] = {"n": int(len(sub)),
                          "mean_pct": round(float(sub["r"].mean()) * 100, 3) if len(sub) else None}
    out["corr"] = round(float(real[["g", "r"]].corr().iloc[0, 1]), 3)
    return out


def main(days: int = 1500, only: list[str] | None = None) -> None:
    cfg0 = load_config()
    print(f"\n[prepare] days={days}", flush=True)
    t0 = time.time()
    ctx = prepare_context(cfg0, days)
    print(f"[prepare] done in {time.time() - t0:.0f}s\n", flush=True)

    rows, diag = [], {}
    print("%-12s %5s %11s %9s %8s %6s %6s %5s %7s %8s"
          % ("组", "lev", "总收益%", "年化%", "回撤%", "卡玛", "夏普", "笔数", "单笔%", "均杠杆"))
    print("-" * 96, flush=True)

    for name, lev, boost in GROUPS:
        if only and name not in only:
            continue
        cfg = copy.deepcopy(cfg0)
        cfg["execution"]["leverage"] = lev
        cfg["execution"]["gap_boost_tiers"] = boost
        t1 = time.time()
        r = run_backtest(cfg, days, ctx)
        m = r["metrics"]
        trades = r.get("trades_all") or []
        real = [t for t in trades if t.get("ret_pct") is not None]
        avg_lev = float(np.mean([float(t.get("pos_frac") or 1.0) for t in real])) * lev if real else 0.0
        yr = yearly(r["equity"])
        sg = seg_split(trades)
        rows.append({"name": name, "lev": lev, "gap_boost": boost, "metrics": m,
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
            diag = diagnose(trades)
            if diag:
                d = diag["gap6"]
                print("\n  === 诊断: 成交 %d 笔按开盘缺口六分位 ===" % diag["n"], flush=True)
                for i in range(len(d["n"])):
                    print("    [%+6.2f%%, %+6.2f%%]  n=%3d  单笔 %+7.3f%%  std %5.3f%%  胜率 %4.1f%%"
                          % (d["lo"][i], d["hi"][i], d["n"][i], d["mean_pct"][i],
                             d["std_pct"][i], d["win_pct"][i]), flush=True)
                for k in ("le-1.2", "le-1.5", "le-2.0"):
                    v = diag[k]
                    print("    gap <= %s%%:  n=%3d  单笔 %s%%"
                          % (k[2:], v["n"], v["mean_pct"]), flush=True)
                print("    corr(gap, r) = %+.3f" % diag["corr"], flush=True)
                print("", flush=True)

        os.makedirs(OUT, exist_ok=True)
        fn = "exp_gapboost.json" if days <= 1600 else f"exp_gapboost_{days}.json"
        with open(os.path.join(OUT, fn), "w", encoding="utf-8") as f:
            json.dump({"days": days, "rows": rows, "diag": diag}, f, ensure_ascii=False, indent=1)

    print("\n=== 分年收益 ===")
    years = sorted({y for x in rows for y in x["year"]})
    print("%-6s " % "年" + " ".join(f"{x['name']:>12s}" for x in rows))
    for y in years:
        print("%-6s " % y + " ".join(f"{x['year'].get(y, 0):>12.1f}" for x in rows))
    print("\nsaved output/exp_gapboost*.json")


if __name__ == "__main__":
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    o = sys.argv[2].split(",") if len(sys.argv) > 2 and sys.argv[2] else None
    main(d, o)
