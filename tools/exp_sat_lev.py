# -*- coding: utf-8 -*-
"""第一百七十一轮: 副策略的总敞口 —— 六条 alpha 路全否之后, 唯一还动得动的旋钮。

为什么到这里才动杠杆
--------------------
本轮先试了六条"提高 alpha 密度"的路(换池子/缩短持有/换事件/换超跌域/加执行过滤/阈值扫描),
全部证伪。副策略 19% 的瓶颈不是参数, 是"每天只有 1/max_pos 的资金在工作":
  · 392 笔 × 25% 仓位 × 1.366% ≈ 134% 的六年资金加权收益
  · 主策略 353 笔 × 100% 仓位 × 1.435% ≈ 507%
差 3.8 倍的**不是单笔, 是仓位**。要让副策略的数字接近主策略, 数学上只剩杠杆一条路 ——
代价是回撤同向放大(第 170 轮对主策略已验过: 卡玛不因杠杆改变)。

本脚本给出一张"敞口-收益-回撤"的对照表, 由用户决定停在哪一档。默认仍是 1.0(不改行为)。

用法: python tools/exp_sat_lev.py [天窗口, 默认 1500]
输出: output/exp_sat_lev.json
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
LEVS = [1.0, 1.25, 1.5, 2.0, 2.5]


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    base = load_config()
    ctx = prepare_context(base, days)
    print(f"[ctx] {len(ctx['main_dates'])} 日", flush=True)

    rows = []
    for lev in LEVS:
        cfg = copy.deepcopy(base)
        cfg.setdefault("satellite", {}).update({"leverage": lev, "margin_rate": 0.055})
        r = run(ctx, cfg)
        m = r["metrics"]
        tr = [t for t in r["trades"] if t.get("ret_pct") is not None]
        rets = np.array([float(t["ret_pct"]) for t in tr])
        # 杠杆后的单笔分布(近似: 收益 x lev, 再扣利息)
        lev_rets = rets * lev
        worst = float(lev_rets.min()) if len(lev_rets) else 0.0
        p25 = float(np.percentile(lev_rets, 2.5)) if len(lev_rets) else 0.0
        row = {"lev": lev, "ann": m["年化收益率%"], "mdd": m["最大回撤%"],
               "calmar": m["卡玛比率"], "sharpe": m["夏普比率"], "n": m["交易次数"],
               "avg": m["平均单笔收益%"], "total": m["总收益率%"],
               "interest": m.get("融资利息(元)", 0), "worst_lev": round(worst, 2),
               "p25_lev": round(p25, 2)}
        rows.append(row)
        print("lev=%.2f 年化%8.2f 回撤%8.2f 卡玛%5.2f 笔数%4d 单笔%6.3f 最差单笔%7.2f 利息%9.0f"
              % (lev, row["ann"], row["mdd"], row["calmar"] or 0, row["n"], row["avg"],
                 row["worst_lev"], row["interest"]), flush=True)

    print("\n%-6s %9s %9s %7s %8s %11s %10s" % ("杠杆", "年化%", "回撤%", "卡玛", "期末倍数", "最差单笔%", "2.5%分位%"))
    for x in rows:
        print("%-6.2f %9.2f %9.2f %7.2f %8.2f %11.2f %10.2f" % (
            x["lev"], x["ann"], x["mdd"], x["calmar"] or 0,
            1 + x["total"] / 100.0, x["worst_lev"], x["p25_lev"]))

    os.makedirs(OUT, exist_ok=True)
    p = os.path.join(OUT, "exp_sat_lev.json")
    json.dump({"window": {"days": len(ctx["main_dates"]),
                          "start": ctx["main_dates"][0], "end": ctx["main_dates"][-1]},
               "rows": rows}, open(p, "w"), ensure_ascii=False, indent=1)
    print("\nsaved", p)


if __name__ == "__main__":
    main()
