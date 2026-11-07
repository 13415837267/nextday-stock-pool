# -*- coding: utf-8 -*-
"""第一百七十一轮: 副策略"池子"横向评测。

为什么不是参数扫描
------------------
第 160/163 轮已证明副策略的所有阈值都是尖峰不是平台(hold 相邻一天, 年化 4% <-> 27%)。
本轮不再动阈值, 只换**池子本身** —— 即"在同一个基础池里按什么键排序、要不要先裁波动分位",
这两件事决定了"买的是哪一类票", 是策略身份而不是参数微调。

变体全部走官方副策略引擎(tools/satellite_backtest.run), 含跌停顺延/闸门/成本/印花税时变,
与页面报的是同一套账。

用法: python tools/exp_sat_pool.py [天窗口, 默认 1500]
输出: output/exp_sat_pool.json
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

# (标签, sort_by, ivol_pct_max, hold_days, max_pos)
VARIANTS = [
    ("基线 lowvol H10",        "mom20_ivol",     0.10, 10, 4),
    ("lowvol H5",              "mom20_ivol",     0.10, 5, 4),
    ("lowvol H20",             "mom20_ivol",     0.10, 20, 4),
    ("小成交额 H10",           "amount",         0.00, 10, 4),
    ("小成交额 H5",            "amount",         0.00, 5, 4),
    ("小成交额 H20",           "amount",         0.00, 20, 4),
    ("超跌反转 H10",           "mom20_asc",      0.00, 10, 4),
    ("超跌反转 H5",            "mom20_asc",      0.00, 5, 4),
    ("高振幅 H10",             "amp_desc",       0.00, 10, 4),
    ("高换手 H10",             "turnover_desc",  0.00, 10, 4),
    ("距高点最远 H10",         "dist_high20",    0.00, 10, 4),
    ("低波·单吊 max_pos=1",    "mom20_ivol",     0.10, 10, 1),
    ("低波·6 只并发",          "mom20_ivol",     0.10, 10, 6),
    ("小成交额·单吊",          "amount",         0.00, 10, 1),
]


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    base = load_config()
    print(f"[ctx] 准备 {days} 日上下文 ...", flush=True)
    ctx = prepare_context(base, days)
    print(f"[ctx] 主窗口 {len(ctx['main_dates'])} 日", flush=True)

    rows = []
    for tag, key, ipct, hold, mpos in VARIANTS:
        cfg = copy.deepcopy(base)
        s = cfg.setdefault("satellite", {})
        s["sort_by"] = key
        s["ivol_pct_max"] = ipct
        s["hold_days"] = hold
        s["max_pos"] = mpos
        s["buys_per_day"] = min(mpos, int(s.get("buys_per_day") or mpos))
        try:
            r = run(ctx, cfg)
        except Exception as e:                      # noqa: BLE001
            print(f"{tag}: ERR {type(e).__name__} {e}", flush=True)
            continue
        m = r["metrics"]
        rows.append({"tag": tag, "sort_by": key, "ivol_pct": ipct, "hold": hold,
                     "max_pos": mpos, **{k: m[k] for k in (
                         "总收益率%", "年化收益率%", "最大回撤%", "卡玛比率", "夏普比率",
                         "交易次数", "胜率%", "平均单笔收益%", "空仓天数")}})
        print("%-22s 年化%8.2f 回撤%8.2f 卡玛%5.2f 笔数%5d 单笔%6.3f 空仓%5d"
              % (tag, m["年化收益率%"], m["最大回撤%"], m["卡玛比率"] or 0,
                 m["交易次数"], m["平均单笔收益%"], m["空仓天数"]), flush=True)

    print("\n%-22s %8s %8s %6s %5s %8s %7s" % (
        "变体", "年化%", "回撤%", "卡玛", "笔数", "单笔%", "胜率%"))
    for x in sorted(rows, key=lambda z: -z["年化收益率%"]):
        print("%-22s %8.2f %8.2f %6.2f %5d %8.3f %7.2f" % (
            x["tag"], x["年化收益率%"], x["最大回撤%"], x["卡玛比率"] or 0,
            x["交易次数"], x["平均单笔收益%"], x["胜率%"]))

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "exp_sat_pool.json")
    json.dump({"window": {"days": len(ctx["main_dates"]),
                          "start": ctx["main_dates"][0], "end": ctx["main_dates"][-1]},
               "variants": rows}, open(p, "w"), ensure_ascii=False, indent=1)
    print("\nsaved", p)


if __name__ == "__main__":
    main()
