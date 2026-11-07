# -*- coding: utf-8 -*-
"""第 170 轮: 引擎级杠杆验证 —— 用改后的 backtest.py 跑真实路径(含融资利息)。

与 exp_leverage.py(事后按日收益放大)的区别: 这里杠杆进入**买入额与现金约束**,
利息逐日从 cash 扣, 复利路径由引擎自己滚 —— 是落地口径, 不是近似。

用法: python tools/exp_lev_engine.py 1500
"""
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["BT_FULL_TRADES"] = "1"

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

OUT = os.path.join(ROOT, "output")
LEVS = (1.0, 1.25, 1.5, 1.75, 2.0)


def yearly(equity):
    """按自然年切净值, 返回 {年: 收益率%}。"""
    by = {}
    for e in equity:
        by.setdefault(str(e["d"])[:4], []).append((str(e["d"]), e["nav"]))
    out = {}
    prev_end = None
    for y in sorted(by):
        pts = by[y]
        start = prev_end if prev_end else pts[0][1]
        out[y] = round((pts[-1][1] / start - 1) * 100, 2)
        prev_end = pts[-1][1]
    return out


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    cfg0 = load_config()
    print(f"\n[prepare] days={days}")
    ctx = prepare_context(cfg0, days)
    res = []
    print("\nL      总收益%      年化%    回撤%    卡玛    夏普   笔数   单笔%   总费用")
    base_year = None
    for L in LEVS:
        cfg = load_config()
        cfg["execution"]["leverage"] = L
        r = run_backtest(cfg, days, ctx)
        m = r["metrics"]
        yr = yearly(r["equity"])
        if base_year is None:
            base_year = yr
        res.append(dict(L=L, **{k: m[k] for k in m}, year=yr))
        print(f"{L:<6.2f} {m['总收益率%']:>10.1f} {m['年化收益率%']:>10.2f} "
              f"{m['最大回撤%']:>9.2f} {m['卡玛比率']:>7.2f} {m['夏普比率']:>7.2f} "
              f"{m['交易次数']:>5d} {m['平均单笔收益%']:>7.3f} {m['总费用(元)']:>12.0f}")

    print("\n=== 分年对比(1.0x -> 2.0x) ===")
    years = sorted(base_year)
    print("年     " + "  ".join(f"{L:>7.2f}x" for L in LEVS))
    for y in years:
        vals = []
        for x in res:
            vals.append(f"{x['year'].get(y, 0):>8.2f}")
        print(f"{y}  " + "  ".join(vals))

    json.dump({"days": days, "res": res}, open(os.path.join(OUT, "exp_lev_engine.json"), "w"),
              ensure_ascii=False, indent=1)
    print("\nsaved output/exp_lev_engine.json")


if __name__ == "__main__":
    main()
