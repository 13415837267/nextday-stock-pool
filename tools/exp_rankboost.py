#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第一百七十四轮: 恐慌加仓的阈值该用固定百分点还是滚动分位?

问题
----
第一百七十三轮上线的 mkt_boost_tiers 用的是**固定百分点**:
信号日全市场个股涨跌中位数落在 [-2.5%, -1.0%] 时仓位 x1.5。
这个判据隐含一个假设 —— "跌 1%" 在任何年份都代表同样的稀有程度。
实际不是: 2015 与 2024 的市场日波动率差一倍以上, 同一个 -1% 在低波动年
是事件(约占 8% 的交易日), 在高波动年是日常(约占 25%)。
于是同一条规则在不同年份筛出的样本密度不同, 相当于"规则强度随年份漂移"。

分位把阈值标准化: 今天的 mkt_pct 在过去 N 日里跌得比多少比例的日子更狠。
rank 越低 = 跌得越极端。若"错杀修复"这个机制是真的, 它应当在**分位**上
比在百分点上更稳定 —— 这正是本实验要证伪的东西。

用法
----
    python tools/exp_rankboost.py 1500                # 默认组
    python tools/exp_rankboost.py 3053 BR=250:0.02:0.20:1.5    # 自定义
    python tools/exp_rankboost.py 1500 BR=250:0.05:0.30:1.5,250:0.02:0.20:1.5
"""
from __future__ import annotations

import gc
import json
import os
import sys
import time

import numpy as np
import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
OUT = os.path.join(ROOT, "output")

# name, rank_win, rank_above(下限, 可 None), rank_below(上限), mult
GROUPS: list[tuple] = [
    ("base",      None,        None,  None,  1.5),   # 现行: 固定 -2.5%~-1.0%
    ("r250_05_20", 250,        0.05,  0.20,  1.5),
    ("r250_05_30", 250,        0.05,  0.30,  1.5),
    ("r250_02_20", 250,        0.02,  0.20,  1.5),
    ("r250_10_30", 250,        0.10,  0.30,  1.2),
]


def build_groups(spec: str):
    out = []
    for part in spec.split(","):
        seg = part.strip().split(":")
        if len(seg) == 4:
            win, lo, hi, mult = seg
            out.append((f"r{win}_{lo.replace('.','')}_{hi.replace('.','')}",
                        int(win), float(lo), float(hi), float(mult)))
    return out


def run_one(cfg: dict, ctx: dict, dates: list[str], ex_override: dict) -> dict:
    from src.backtest import run_backtest

    c2 = dict(cfg)
    c2["execution"] = dict(cfg["execution"])
    c2["execution"].update(ex_override)
    ctx2 = dict(ctx)
    ctx2["main_dates"] = dates
    t0 = time.time()
    res = run_backtest(c2, None, ctx2)
    m = dict(res["metrics"])
    eq = res.get("equity") or []
    del res
    gc.collect()

    return {"metrics": m, "secs": round(time.time() - t0, 1),
            "years": _yearly(eq), "n_eq": len(eq)}


def _yearly(equity: list[dict]) -> dict:
    pts = [(str(e["d"]), float(e["nav"])) for e in equity if e.get("nav") is not None]
    if not pts:
        return {}
    out, anchor, cur, last = {}, pts[0][1], pts[0][0][:4], pts[0][1]
    for d, v in pts:
        y = d[:4]
        if y != cur:
            out[cur] = round((last / anchor - 1) * 100, 2) if anchor else 0.0
            anchor, cur = last, y
        last = v
    out[cur] = round((last / anchor - 1) * 100, 2) if anchor else 0.0
    return out


def main(days: int = 1500, groups=None) -> None:
    from src.backtest import prepare_context
    from src.config import load_config

    groups = groups or GROUPS
    cfg = load_config(os.path.join(ROOT, "config.yaml"))
    base_ex = dict(cfg["execution"])

    print(f"[exp] 构建上下文 (含 warmup), 目标 {days} 交易日", flush=True)
    t0 = time.time()
    ctx = prepare_context(cfg, days)
    md = ctx["main_dates"]
    print(f"[exp] 上下文 {md[0]}~{md[-1]}  {len(md)} 日  {time.time()-t0:.0f}s", flush=True)

    rows = []
    hdr = f"{'组':<14}{'年化%':>9}{'回撤%':>8}{'卡玛':>7}{'夏普':>7}{'笔数':>6}{'耗时':>6}"
    print("\n" + hdr)
    print("-" * len(hdr))

    for name, win, lo, hi, mult in groups:
        ex = dict(base_ex)
        if win is None:
            # 现行绝对阈值
            ex["mkt_boost_tiers"] = base_ex.get("mkt_boost_tiers") or []
            ex["mkt_rank_win"] = 0
        else:
            tier = {"mult": mult}
            if lo is not None:
                tier["rank_above"] = lo
            tier["rank_below"] = hi
            ex["mkt_boost_tiers"] = [tier]
            ex["mkt_rank_win"] = win
        r = run_one(cfg, ctx, md, ex)
        m = r["metrics"]
        rows.append({"name": name, "win": win, "lo": lo, "hi": hi, "mult": mult,
                     "metrics": m, "secs": r["secs"], "years": r["years"]})
        print(f"{name:<14}{m['年化收益率%']:>9.2f}{m['最大回撤%']:>8.2f}"
              f"{m['卡玛比率']:>7.2f}{m['夏普比率']:>7.2f}{m['交易次数']:>6}"
              f"{r['secs']:>6.0f}", flush=True)

    # 分年：把 base 与每个分位组的年度收益并排放，看"稳定性"这件事本身
    print("\n分年收益（%）")
    base_y = rows[0]["years"] if rows else {}
    yrs = sorted(base_y)
    print(f"{'组':<14}" + "".join(f"{y:>9}" for y in yrs))
    for r in rows:
        print(f"{r['name']:<14}" + "".join(f"{r['years'].get(y, float('nan')):>9.1f}" for y in yrs))

    os.makedirs(OUT, exist_ok=True)
    fn = "exp_rankboost.json" if days <= 1600 else f"exp_rankboost_{days}.json"
    with open(os.path.join(OUT, fn), "w", encoding="utf-8") as f:
        json.dump({"days": days, "rows": rows}, f, ensure_ascii=False, indent=1)
    print(f"\nsaved output/{fn}")


if __name__ == "__main__":
    d = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    g = build_groups(sys.argv[2]) if len(sys.argv) > 2 and ":" in sys.argv[2] else None
    main(d, g)
