# -*- coding: utf-8 -*-
"""第 191 轮: 保险丝式节流 —— 深阈值 + 一次清掉杠杆, 能否解锁 2.0x。

第一百九十轮的节流失败, 原因是**触发太早、降得太少**:
  -8% 就降到 1.5x, 而 -8% 在 352 笔里发生了上百次, 等于长期在用 1.5x 跑, 收益腰斩,
  可最坏担保比例只从 134% 抬到 145%(因为 -8% 那一刻亏损已经实亏)。
本轮换一种设计: 把节流当**保险丝** —— 正常时期完全不干扰(base 杠杆拉满),
只在逼近强平线前 5~10pp 时一次性把杠杆打到 1.0(清掉全部融资)。

2.0x 的强平回撤阈值 = 1.3(2-1)/2 - 1 = -35.0%, 2.5x = -22.0%。
故触发线取 -22/-25/-28/-30(2.0x) 与 -15/-18/-20(2.5x)。

用法: python3 tools/exp_fuse_191.py [days]
输出: output/fuse_191.json
"""
from __future__ import annotations

import copy
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
os.environ["BT_FULL_TRADES"] = "1"

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import ROOT, load_config  # noqa: E402

CASES = [
    ("1.50x 无保险丝", 1.5, []),
    ("1.70x 无保险丝", 1.7, []),
    ("1.70x | -30 -> 1.0", 1.7, [{"dd_below": -30.0, "lev": 1.0}]),
    ("1.70x | -35 -> 1.0", 1.7, [{"dd_below": -35.0, "lev": 1.0}]),
    ("1.70x | -40 -> 1.0", 1.7, [{"dd_below": -40.0, "lev": 1.0}]),
    ("2.00x 无保险丝", 2.0, []),
    ("2.00x | -28 -> 1.0", 2.0, [{"dd_below": -28.0, "lev": 1.0}]),
    ("2.00x | -30 -> 1.0", 2.0, [{"dd_below": -30.0, "lev": 1.0}]),
]


def ratio(lev: float, mdd: float) -> float:
    if lev <= 1.0:
        return float("inf")
    return lev * (1.0 + mdd / 100.0) / (lev - 1.0) * 100.0


def main() -> None:
    base = load_config()
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    print(f"== 保险丝节流: {days} 日 ==", flush=True)
    ctx = prepare_context(base, days)

    rows = []
    for name, L, tiers in CASES:
        cfg = copy.deepcopy(base)
        cfg["execution"]["leverage"] = float(L)
        cfg["execution"]["dd_cut_tiers"] = tiers
        r = run_backtest(cfg, days, ctx)
        m = r["metrics"]
        row = {"name": name, "lev": L, "tiers": tiers,
               "total": m["总收益率%"], "ann": m["年化收益率%"],
               "mdd": m["最大回撤%"], "calmar": m["卡玛比率"],
               "n": m["交易次数"], "win": m["胜率%"], "avg": m["平均单笔收益%"],
               "days": r["meta"]["days"], "ratio": ratio(L, m["最大回撤%"])}
        rows.append(row)
        print(f"{name:<22s} {format(row['total'], ',.0f'):>16}%  年化{row['ann']:6.1f}%  "
              f"回撤{row['mdd']:7.2f}%  卡玛{row['calmar']:6.2f}  {row['n']:4d}笔  "
              f"单笔{row['avg']:+.3f}%  担保{row['ratio']:6.0f}%", flush=True)

    p = ROOT / "output" / "fuse_191.json"
    p.write_text(json.dumps({"days": days, "rows": rows}, ensure_ascii=False, indent=1))
    print(f"\n已写 {p}")


if __name__ == "__main__":
    main()
