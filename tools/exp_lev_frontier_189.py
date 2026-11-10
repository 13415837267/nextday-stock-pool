# -*- coding: utf-8 -*-
"""第一百八十九轮: 杠杆前沿在当前参数下重算(不落地, 供用户拍板)。

第 170 轮做过这条前沿, 但此后又改了 10 多轮参数(154→189), 且当时的基线是年化 104%/卡玛 5.87。
本轮用现行配置真实回测 1.0/1.25/1.5/1.75/2.0 五档, 只改 execution.leverage, 不改任何别的。

为什么要重算: "卡玛随杠杆单调上升"这个反直觉结论(1.0x 5.87 → 2.0x 8.37)如果在新参数下仍成立,
那说明"下注额"是唯一还没挖尽、且不损伤选股的维度 —— 因为策略 76% 的交易日空仓, 空仓期是
免费的风险缓冲, 杠杆只作用于剩下的 24% 敞口时间, 回撤增幅天然小于收益增幅。

用法: python3 tools/exp_lev_frontier_189.py
输出: output/lev_frontier_189.json
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

LEVELS = [1.0, 1.25, 1.5, 1.75, 2.0, 2.5]


def main() -> None:
    base = load_config()
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    print(f"== 杠杆前沿重算: {days} 日 ==", flush=True)
    ctx = prepare_context(base, days)
    rows = []
    for L in LEVELS:
        cfg = copy.deepcopy(base)
        cfg["execution"]["leverage"] = float(L)
        r = run_backtest(cfg, days, ctx)
        m = r["metrics"]
        row = {
            "lev": L,
            "total": m["总收益率%"],
            "ann": m["年化收益率%"],
            "mdd": m["最大回撤%"],
            "calmar": m["卡玛比率"],
            "sharpe": m["夏普比率"],
            "n": m["交易次数"],
            "win": m["胜率%"],
            "avg": m["平均单笔收益%"],
        }
        rows.append(row)
        print(f"L={L:.2f}  {row['total']:>14,.0f}%  年化{row['ann']:6.1f}%  "
              f"回撤{row['mdd']:6.2f}%  卡玛{row['calmar']:5.2f}  "
              f"夏普{row['sharpe']:5.2f}  {row['n']}笔 单笔{row['avg']:.3f}%", flush=True)

    p = ROOT / "output" / "lev_frontier_189.json"
    p.write_text(json.dumps({"days": days, "rows": rows}, ensure_ascii=False, indent=1))
    print(f"\n已写 {p}")


if __name__ == "__main__":
    main()
