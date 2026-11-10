# -*- coding: utf-8 -*-
"""第一百九十轮: 杠杆的回撤节流 —— 用"回撤到线就降杠杆"换 2.0x 的实盘资格。

第一百八十九轮把杠杆前沿算清楚了: 卡玛随杠杆单调上升, 唯一的墙是两融维持担保比例
(总资产/负债 < 130% 强平)。把这条线换算成净值回撤:

    ratio = L * (1 + dd) / (L - 1)

  L=1.5: 强平于 mdd -45%, 历史最坏 -25.6% -> 担保 223%(安全)
  L=2.0: 强平于 mdd -35%, 历史最坏 -32.8% -> 担保 134%(贴线, 等于赌"历史最坏不再重演")
  L=2.5: 强平于 mdd -25%, 历史最坏 -39.5% -> 担保 101%(历史最坏那一刻早已被平)

**注意那个公式的残酷之处**: L=2.0 时只要净值有任何回撤, 担保比例就低于 200%。
所以"2.0x 且 200% 担保"在数学上不存在 —— 2.0x 的可比目标是"把 mdd 从 -32.8% 压到 -22%,
即担保 156%", 与 1.5x 的 223% 相比安全边际更薄, 但收益若能从 1,465x 涨到 5,000x+ 就值得。

实现: execution.dd_cut_tiers = [{dd_below: -8.0, lev: 1.0}] —— 净值从高点回撤超过 8% 时,
后续买入的杠杆降到 1.0, 回撤修复后自动加回基础杠杆。只看截至前一日的曲线, 无未来函数。

用法: python3 tools/exp_dd_throttle_190.py [days]
输出: output/dd_throttle_190.json
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
    ("回归检查: 1.25x 无节流", 1.25, []),
    ("2.0x 无节流", 2.0, []),
    ("2.0x | dd<-6 -> 1.5x", 2.0, [{"dd_below": -6.0, "lev": 1.5}]),
    ("2.0x | dd<-8 -> 1.5x", 2.0, [{"dd_below": -8.0, "lev": 1.5}]),
    ("2.0x | dd<-8 -> 1.0x", 2.0, [{"dd_below": -8.0, "lev": 1.0}]),
    ("2.0x | dd<-12 -> 1.0x", 2.0, [{"dd_below": -12.0, "lev": 1.0}]),
    ("2.0x | -6->1.5 / -12->1.0", 2.0, [{"dd_below": -12.0, "lev": 1.0},
                                        {"dd_below": -6.0, "lev": 1.5}]),
    ("2.5x | dd<-8 -> 1.5x", 2.5, [{"dd_below": -8.0, "lev": 1.5}]),
    ("2.5x | dd<-10 -> 1.0x", 2.5, [{"dd_below": -10.0, "lev": 1.0}]),
]


def ratio(lev: float, mdd: float) -> float:
    if lev <= 1.0:
        return float("inf")
    return lev * (1.0 + mdd / 100.0) / (lev - 1.0) * 100.0


def main() -> None:
    base = load_config()
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    print(f"== 回撤节流 x 杠杆: {days} 日 ==", flush=True)
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
               "sharpe": m["夏普比率"], "n": m["交易次数"],
               "win": m["胜率%"], "ratio": ratio(L, m["最大回撤%"])}
        rows.append(row)
        print(f"{name:<30s} {format(row['total'], ',.0f'):>14}%  年化{row['ann']:6.1f}%  "
              f"回撤{row['mdd']:7.2f}%  卡玛{row['calmar']:5.2f}  {row['n']:4d}笔  "
              f"担保{row['ratio']:5.0f}%", flush=True)

    p = ROOT / "output" / "dd_throttle_190.json"
    p.write_text(json.dumps({"days": days, "rows": rows}, ensure_ascii=False, indent=1))
    print(f"\n已写 {p}")


if __name__ == "__main__":
    main()
