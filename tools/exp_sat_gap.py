# -*- coding: utf-8 -*-
"""第一百七十一轮: 副策略"只买低开"的阈值稳健性 —— 它是平台还是又一根尖峰?

背景
----
exp_sat_exec 里 `gap_max: 5.0 -> 0`(只买低开) 让副策略年化 18.99% -> 26.90%,
`gap_max=0 + gap_min=-2` 到 28.58%, 而笔数只从 392 变 384 —— 这不像"多交易换收益",
更像"把高开那部分负期望的成交剔掉了"。单笔 +1.366% -> +1.891%。

但第 160/163 轮的教训是: 副策略**所有**参数都是尖峰不是平台(hold 相邻一天年化 4% <-> 27%)。
所以一个 +9.6pp 的改进, 必须先用"相邻取值是否同向"来验伪:
  · 如果 gap_max 从 -1 到 +1 是一条平缓的坡, 那就是真效应;
  · 如果只有 0 这一点好、-0.5 和 +0.5 都塌, 那就是又一次过拟合。

用法: python tools/exp_sat_gap.py [天窗口, 默认 1500]
输出: output/exp_sat_gap.json
"""
from __future__ import annotations

import copy
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.backtest import prepare_context        # noqa: E402
from src.config import load_config              # noqa: E402
from tools.satellite_backtest import run        # noqa: E402

OUT = os.path.join(ROOT, "output")
GRID = [-4.0, -3.0, -2.0, -1.5, -1.0, -0.5, 0.0, 0.5, 1.0, 2.0, 5.0]   # 扫 gap_max, gap_min 固定 -3
GMIN = [-4.0, -3.0, -2.5, -2.0, -1.5, -1.0]                          # 扫 gap_min, gap_max 固定 0


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    base = load_config()
    ctx = prepare_context(base, days)
    print(f"[ctx] {len(ctx['main_dates'])} 日", flush=True)

    rows = []
    trials = [(g, -3.0) for g in GRID] + [(0.0, g) for g in GMIN]
    for gmax, gmin in trials:
        cfg = copy.deepcopy(base)
        cfg.setdefault("satellite", {}).update({"gap_max": gmax, "gap_min": gmin})
        r = run(ctx, cfg)
        m = r["metrics"]
        tr = [t for t in r["trades"] if t.get("ret_pct") is not None]
        by = {}
        for t in tr:
            y = str(t["exit_date"])[:4]
            by.setdefault(y, []).append(float(t["ret_pct"]))
        rows.append({"gap_max": gmax, "gap_min": gmin, "ann": m["年化收益率%"], "mdd": m["最大回撤%"],
                     "calmar": m["卡玛比率"], "n": m["交易次数"], "avg": m["平均单笔收益%"],
                     "win": m["胜率%"], "total": m["总收益率%"],
                     "by_year": {k: round(float(np.mean(v)), 2) for k, v in sorted(by.items())}})
        print("gap_max=%5.1f gap_min=%5.1f 年化%7.2f 回撤%7.2f 卡玛%5.2f 笔数%4d 单笔%6.3f"
              % (gmax, gmin, m["年化收益率%"], m["最大回撤%"], m["卡玛比率"] or 0,
                 m["交易次数"], m["平均单笔收益%"]), flush=True)

    print("\n分年单笔(%):")
    yrs = sorted({y for r in rows for y in r["by_year"]})
    print("%-17s %s" % ("max / min", " ".join("%7s" % y for y in yrs)))
    for r in rows:
        print("%6.1f / %6.1f %s" % (r["gap_max"], r["gap_min"],
                                    " ".join("%7.2f" % r["by_year"].get(y, float("nan")) for y in yrs)))

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "exp_sat_gap.json")
    json.dump({"window": {"days": len(ctx["main_dates"]),
                          "start": ctx["main_dates"][0], "end": ctx["main_dates"][-1]},
               "grid": rows}, open(p, "w"), ensure_ascii=False, indent=1)
    print("\nsaved", p)


if __name__ == "__main__":
    main()
