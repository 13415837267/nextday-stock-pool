# -*- coding: utf-8 -*-
"""第一百七十三轮: 恐慌日超配(mkt_boost_tiers) —— 唯一还没试过的"加仓"方向。

为什么试这个
------------
前面 172 轮里, 所有"仓位调节"都是**减仓**: 评分档降仓(第 96 轮)、池深档降仓(第 99 轮)、
高开降仓(第 97 轮)、刀口保护降仓(第 103 轮)、perf_gate 降仓(第 97 轮)、回撤控制。
第 119 轮把这批统一否掉, 理由只有一句: **复利策略里自我降仓 = 减少在场次数, 绝对收益必崩**。
于是"恐慌日加仓"这个反方向从来没被认真测过 —— 现有 `mkt_floor_tiers` 的语义只是
"把仓位补回当日 frac 基准"(风控日 frac=0.4, 最多补到 0.4), 并没有超过满仓。

数据说了什么
------------
用 200 笔已成交样本, 按**信号日**全市场典型个股涨跌(中位数)分桶:

    mkt <= -2.5%   6 笔   -1.61%
    mkt <= -2.0%  11 笔   +0.81%
    mkt <= -1.5%  21 笔   +1.75%
    mkt <= -1.0%  35 笔   +1.92%
    mkt <= -0.5%  78 笔   +2.06%
    mkt >  -1.5% 179 笔   +1.21%

**不是单调的**: 跌到 -2.5% 以下反而是负的(接飞刀), 最优点是 -0.5%~-1.0% 这一段。
这正是 `knife_tiers` 与 `mkt_floor_tiers` 同时存在的原因 —— 深跌要保护, 中跌要加码。

本轮做的事
----------
1. 引擎新增 `execution.mkt_boost_tiers = [{mkt_below, mult}]`: 信号日市场典型跌幅达到阈值时,
   **最终仓位再乘 mult(>1)**, 允许突破满仓。与 mkt_floor_tiers(补回基准) 语义不同、可叠加。
2. 修 `margin_avail`: 原来按 `(lev-1)*equity` 算融资额度, pick_frac>1 时会被卡死,
   超配永远不生效。改为按 `lev*max(1,pick_frac)` 算。pick_frac<=1 时逐位等价。
3. 扫阈值×倍数, 并做 2024-09 前后切分(第 172 轮发现的结构断点)。

用法: python tools/exp_mktboost.py [days] [组名,...]
输出: output/exp_mktboost.json
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

# name, leverage, mkt_boost_tiers
GROUPS = [
    ("base",      1.25, []),
    ("b05x13",    1.25, [{"mkt_below": -0.5, "mult": 1.3}]),
    ("b05x15",    1.25, [{"mkt_below": -0.5, "mult": 1.5}]),
    ("b10x13",    1.25, [{"mkt_below": -1.0, "mult": 1.3}]),
    ("b10x15",    1.25, [{"mkt_below": -1.0, "mult": 1.5}]),
    ("b15x15",    1.25, [{"mkt_below": -1.5, "mult": 1.5}]),
    ("b05x15_L2", 2.00, [{"mkt_below": -0.5, "mult": 1.5}]),
    ("b10x15_L2", 2.00, [{"mkt_below": -1.0, "mult": 1.5}]),
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
    """按卖出日切 2024-09-30 前后(第 172 轮的断点), 看加仓是否只在某一段有效。"""
    real = [t for t in trades if t.get("ret_pct") is not None]
    out = {}
    for k, f in (("pre", lambda d: d < "20240930"), ("post", lambda d: d >= "20240930")):
        v = [float(t["ret_pct"]) for t in real if f(str(t.get("sell_date", "")))]
        out[k] = {"n": len(v), "mean": round(float(np.mean(v)), 3) if v else None}
    return out


def build_groups(spec: str):
    """解析 "B=-1.0:1.2,-0.8:1.5" -> 组定义;  可选 "L=2.0" 前缀指定杠杆。"""
    lev = 1.25
    out = []
    for part in spec.split(";"):
        part = part.strip()
        if part.startswith("L="):
            lev = float(part[2:])
            continue
        if part.startswith("B="):
            for x in part[2:].split(","):
                seg = x.split(":")
                thr, mult = float(seg[0]), float(seg[1])
                tier = {"mkt_below": thr, "mult": mult}
                tag = f"b{abs(thr):.1f}"
                if len(seg) > 2:                     # lo:hi:mult -> 带下限的区间档
                    tier["mkt_above"] = float(seg[0])
                    tier["mkt_below"] = float(seg[1])
                    mult = float(seg[2])
                    tier["mult"] = mult
                    tag = f"r{abs(float(seg[0])):.1f}_{abs(float(seg[1])):.1f}"
                out.append((f"{tag}x{mult:.2f}_L{lev:g}".replace(".", ""), lev, [tier]))
    return out


def main(days: int = 1500, only: list[str] | None = None, spec: str = "") -> None:
    cfg0 = load_config()
    print(f"\n[prepare] days={days}", flush=True)
    t0 = time.time()
    ctx = prepare_context(cfg0, days)
    print(f"[prepare] done in {time.time() - t0:.0f}s\n", flush=True)

    groups = build_groups(spec) if spec else GROUPS
    rows = []
    print("%-10s %5s %11s %9s %8s %6s %6s %5s %7s %8s"
          % ("组", "lev", "总收益%", "年化%", "回撤%", "卡玛", "夏普", "笔数", "单笔%", "均杠杆"))
    print("-" * 92, flush=True)

    for name, lev, boost in groups:
        if only and name not in only:
            continue
        cfg = copy.deepcopy(cfg0)
        cfg["execution"]["leverage"] = lev
        cfg["execution"]["mkt_boost_tiers"] = boost
        t1 = time.time()
        r = run_backtest(cfg, days, ctx)
        m = r["metrics"]
        trades = r.get("trades_all") or []
        real = [t for t in trades if t.get("ret_pct") is not None]
        avg_lev = float(np.mean([float(t.get("pos_frac") or 1.0) for t in real])) * lev if real else 0.0
        yr = yearly(r["equity"])
        sg = seg_split(trades)
        rows.append({"name": name, "lev": lev, "boost": boost, "metrics": m,
                     "year": yr, "seg": sg, "avg_eff_lev": round(avg_lev, 3),
                     "secs": round(time.time() - t1)})
        print("%-10s %5.2f %11.1f %9.2f %8.2f %6.2f %6.2f %5d %7.3f %8.3f"
              % (name, lev, m["总收益率%"], m["年化收益率%"], m["最大回撤%"],
                 m["卡玛比率"] or 0, m["夏普比率"], m["交易次数"],
                 m["平均单笔收益%"], avg_lev), flush=True)
        print("          前段 n=%d %s%%  后段 n=%d %s%%   耗时 %.0fs"
              % (sg["pre"]["n"], sg["pre"]["mean"], sg["post"]["n"], sg["post"]["mean"],
                 time.time() - t1), flush=True)
        os.makedirs(OUT, exist_ok=True)
        with open(os.path.join(OUT, "exp_mktboost.json"), "w", encoding="utf-8") as f:
            json.dump({"days": days, "rows": rows}, f, ensure_ascii=False, indent=1)

    print("\n=== 分年收益 ===")
    years = sorted({y for x in rows for y in x["year"]})
    print("%-6s " % "年" + " ".join(f"{x['name']:>10s}" for x in rows))
    for y in years:
        print("%-6s " % y + " ".join(f"{x['year'].get(y, 0):>10.1f}" for x in rows))
    print("\nsaved output/exp_mktboost.json")


if __name__ == "__main__":
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    a2 = sys.argv[2] if len(sys.argv) > 2 else ""
    if "=" in a2:
        main(d, None, a2)
    else:
        main(d, a2.split(",") if a2 else None, "")
