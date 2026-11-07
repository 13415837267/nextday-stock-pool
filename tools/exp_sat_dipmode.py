# -*- coding: utf-8 -*-
"""第一百七十一轮: 副策略改用"缩量阴线回调"形态(dip), 换到创业板/科创板。

动机
----
副策略的低波持 10 日在 1500 日只能做到 18.99%(单笔 +1.366%, 资金效率 25%)。
而主策略的"缩量阴线回调 + T+1 开盘买 + 隔日卖"单笔 +1.435%(资金效率 100%)。
同一套形态如果搬到**另一块板**, 就同时满足"不同选股"和"高密度"两件事:
第 155 轮的板块实验里, 仅创业板 33.08% / 285 笔 / 单笔 1.051% —— 是副策略现值的 1.7 倍。

本脚本用**副策略引擎**(含跌停顺延/闸门/成本)重跑, 而不是沿用主策略引擎的数字,
因为两个引擎的仓位/卖出/成本口径不同, 不能直接搬。

变体维度: 板块 / 持有期 / 并发数 / 排序键 / 缺口过滤

用法: python tools/exp_sat_dipmode.py [天窗口, 默认 1500]
输出: output/exp_sat_dipmode.json
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

BASE_SAT = {"mode": "dip", "hold_days": 1, "max_pos": 1, "buys_per_day": 1,
            "min_pool": 1, "ivol_pct_max": 0.0, "min_score": 0.0,
            "gap_min": -3.0, "gap_max": 0.0, "sort_by": "ivol",
            "dist_ma5_max": None, "dist_ma5_min": None, "dist_high20_max": None,
            "stop_pct": 0.0, "trail_pct": 0.0, "market_gate": True}

VARIANTS = [
    ("创业板 持1 单吊 低开",      {"boards": ["创业板"]}),
    ("创业板 持1 单吊 gap5",      {"boards": ["创业板"], "gap_max": 5.0}),
    ("创业板 持1 四只 低开",      {"boards": ["创业板"], "max_pos": 4, "buys_per_day": 4}),
    ("创业板 持3 单吊 低开",      {"boards": ["创业板"], "hold_days": 3}),
    ("创业板 持1 单吊 跌最深",    {"boards": ["创业板"], "sort_by": "mom20_asc"}),
    ("科创板 持1 单吊 低开",      {"boards": ["科创板"], "dip_pct_min": -8.0, "dip_pct_max": -4.0}),
    ("主板   持1 单吊 低开",      {"boards": ["主板"]}),
]


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    base = load_config()
    ctx = prepare_context(base, days)
    print(f"[ctx] {len(ctx['main_dates'])} 日", flush=True)

    rows = []
    for tag, over in VARIANTS:
        boards = over.pop("boards")
        cfg = copy.deepcopy(base)
        cfg["universe"] = {**cfg.get("universe", {}), "boards": boards}
        cfg.setdefault("satellite", {}).update(BASE_SAT)
        cfg["satellite"].update(over)
        try:
            r = run(ctx, cfg)
        except Exception as e:                       # noqa: BLE001
            print(f"{tag}: ERR {type(e).__name__} {e}", flush=True)
            continue
        m = r["metrics"]
        tr = [t for t in r["trades"] if t.get("ret_pct") is not None]
        by = {}
        for t in tr:
            by.setdefault(str(t["exit_date"])[:4], []).append(float(t["ret_pct"]))
        rows.append({"tag": tag, "boards": boards, **over,
                     **{k: m[k] for k in ("总收益率%", "年化收益率%", "最大回撤%", "卡玛比率",
                                          "夏普比率", "交易次数", "胜率%", "平均单笔收益%")},
                     "by_year": {k: round(float(np.mean(v)), 2) for k, v in sorted(by.items())}})
        print("%-22s 年化%8.2f 回撤%8.2f 卡玛%5.2f 笔数%5d 单笔%6.3f 胜率%5.1f"
              % (tag, m["年化收益率%"], m["最大回撤%"], m["卡玛比率"] or 0,
                 m["交易次数"], m["平均单笔收益%"], m["胜率%"]), flush=True)
        over["boards"] = boards

    print("\n%-22s %8s %8s %6s %5s %8s" % ("变体", "年化%", "回撤%", "卡玛", "笔数", "单笔%"))
    for x in sorted(rows, key=lambda z: -z["年化收益率%"]):
        print("%-22s %8.2f %8.2f %6.2f %5d %8.3f" % (
            x["tag"], x["年化收益率%"], x["最大回撤%"], x["卡玛比率"] or 0,
            x["交易次数"], x["平均单笔收益%"]))

    print("\n分年单笔(%):")
    yrs = sorted({y for r in rows for y in r["by_year"]})
    print("%-22s %s" % ("变体", " ".join("%7s" % y for y in yrs)))
    for x in rows:
        print("%-22s %s" % (x["tag"], " ".join("%7.2f" % x["by_year"].get(y, float("nan"))
                                               for y in yrs)))

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "exp_sat_dipmode.json")
    json.dump({"window": {"days": len(ctx["main_dates"]),
                          "start": ctx["main_dates"][0], "end": ctx["main_dates"][-1]},
               "variants": rows}, open(p, "w"), ensure_ascii=False, indent=1)
    print("\nsaved", p)


if __name__ == "__main__":
    main()
