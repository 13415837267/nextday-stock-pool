# -*- coding: utf-8 -*-
"""第一百九十轮(全史校验): 只跑三格 —— 1.5x 基线 / 2.0x 无节流 / 2.0x+节流。

1500 日窗口可能带红利(第 187 轮右侧策略就是这么塌的), 决定"能不能上钱"必须看全史。
用法: python3 tools/exp_dd_full_190.py [days]
输出: output/dd_throttle_full_190.json
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
    ("1.50x 无节流", 1.5, []),
    ("1.60x 无节流", 1.6, []),
    ("1.70x 无节流", 1.7, []),
    ("1.75x 无节流", 1.75, []),
    ("2.00x | dd<-8 -> 1.5x", 2.0, [{"dd_below": -8.0, "lev": 1.5}]),
]


def ratio(lev: float, mdd: float) -> float:
    if lev <= 1.0:
        return float("inf")
    return lev * (1.0 + mdd / 100.0) / (lev - 1.0) * 100.0


def main() -> None:
    base = load_config()
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 3000
    print(f"== 全史校验: {days} 日 ==", flush=True)
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
        print(f"{name:<24s} {format(row['total'], ',.0f'):>16}%  年化{row['ann']:6.1f}%  "
              f"回撤{row['mdd']:7.2f}%  卡玛{row['calmar']:6.2f}  {row['n']:4d}笔  "
              f"单笔{row['avg']:+.3f}%  担保{row['ratio']:6.0f}%", flush=True)

    p = ROOT / "output" / "dd_throttle_full_190.json"
    p.write_text(json.dumps({"days": days, "rows": rows}, ensure_ascii=False, indent=1))
    print(f"\n已写 {p}")


if __name__ == "__main__":
    main()
