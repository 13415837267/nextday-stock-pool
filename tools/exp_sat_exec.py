# -*- coding: utf-8 -*-
"""第一百七十一轮: 副策略执行层过滤 —— 把主策略"不买"的那部分搬过来。

为什么测这个
------------
横评发现一个反直觉的事实: 主策略官方 top1 在 **T+1 开盘买 / T+2 开盘卖** 的口径下
只有 **-0.137%/笔**(831 笔), 而官方报 1.435%(353 笔)。同样一批票、同一个卖点,
差 1.57pp/笔 —— 差的那部分**全部来自执行过滤**:
  · buy_gap_below=0  → 只在 T+1 **低开**(gap<=0)时才买
  · min_top1_score / score_tiers → 分数不够不买
  · pool_n_tiers → 候选池太窄自动降档
即 **"不买"本身就是 alpha 的一半**。

副策略这边这些过滤几乎全开: config 里 `min_score: 0`(不筛评分)、`gap_max: 5.0`
(高开 5% 以内都买)。本轮把主策略的执行层逐条搬过来, 看能不能在不换因子的前提下
把 18.99% 抬上去。

用法: python tools/exp_sat_exec.py [天窗口, 默认 1500]
输出: output/exp_sat_exec.json
"""
from __future__ import annotations

import copy
import json
import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)

from src.backtest import prepare_context        # noqa: E402
from src.config import load_config              # noqa: E402
from tools.satellite_backtest import run        # noqa: E402

OUT = os.path.join(ROOT, "output")

# (标签, {satellite 覆盖项})
VARIANTS = [
    ("基线(gap[-3,5], 无评分)",     {}),
    ("只买低开 gap<=0",             {"gap_max": 0.0}),
    ("只买低开 gap<=-1",            {"gap_max": -1.0}),
    ("低开 gap<=0 且 gap>=-2",      {"gap_max": 0.0, "gap_min": -2.0}),
    ("评分门槛 20",                 {"min_score": 20.0}),
    ("评分门槛 40",                 {"min_score": 40.0}),
    ("低开 + 评分20",               {"gap_max": 0.0, "min_score": 20.0}),
    ("低开 + 评分40",               {"gap_max": 0.0, "min_score": 40.0}),
    ("低开 + 评分20 + 8只",         {"gap_max": 0.0, "min_score": 20.0, "max_pos": 8,
                                     "buys_per_day": 8}),
]


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    base = load_config()
    ctx = prepare_context(base, days)
    print(f"[ctx] {len(ctx['main_dates'])} 日", flush=True)

    rows = []
    for tag, over in VARIANTS:
        cfg = copy.deepcopy(base)
        cfg.setdefault("satellite", {}).update(over)
        try:
            r = run(ctx, cfg)
        except Exception as e:                      # noqa: BLE001
            print(f"{tag}: ERR {type(e).__name__} {e}", flush=True)
            continue
        m = r["metrics"]
        rows.append({"tag": tag, **over, **{k: m[k] for k in (
            "总收益率%", "年化收益率%", "最大回撤%", "卡玛比率", "夏普比率",
            "交易次数", "胜率%", "平均单笔收益%")}})
        print("%-24s 年化%8.2f 回撤%8.2f 卡玛%5.2f 笔数%5d 单笔%6.3f 胜率%5.1f"
              % (tag, m["年化收益率%"], m["最大回撤%"], m["卡玛比率"] or 0,
                 m["交易次数"], m["平均单笔收益%"], m["胜率%"]), flush=True)

    print("\n%-24s %8s %8s %6s %5s %8s %7s" % ("变体", "年化%", "回撤%", "卡玛", "笔数", "单笔%", "胜率%"))
    for x in sorted(rows, key=lambda z: -z["年化收益率%"]):
        print("%-24s %8.2f %8.2f %6.2f %5d %8.3f %7.2f" % (
            x["tag"], x["年化收益率%"], x["最大回撤%"], x["卡玛比率"] or 0,
            x["交易次数"], x["平均单笔收益%"], x["胜率%"]))

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "exp_sat_exec.json")
    json.dump({"window": {"days": len(ctx["main_dates"]),
                          "start": ctx["main_dates"][0], "end": ctx["main_dates"][-1]},
               "variants": rows}, open(p, "w"), ensure_ascii=False, indent=1)
    print("\nsaved", p)


if __name__ == "__main__":
    main()
