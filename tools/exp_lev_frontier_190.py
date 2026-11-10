# -*- coding: utf-8 -*-
"""第一百九十轮: 分散 x 杠杆 的二维前沿 —— 凯利公式说"降波动 = 抬上限"。

第一百八十九轮证明: 杠杆是纯线性放大器(笔数 352 / 单笔 1.439% / 夏普 2.52 全程不变),
卡玛随杠杆单调上升, 唯一约束是两融担保比例(130% 强平): 1.5x 最坏 223%, 2.0x 只剩 134%。

于是问题变成: **能不能把最大回撤压下来, 从而在同样的担保比例下用更高的杠杆?**

凯利公式 f* = mu / sigma^2 给出方向: 单吊 1 只 mu=1.435% / sigma=4.871% -> f*=6.05。
分 2 只各半(第 2 名单笔 0.67%), mu=(1.44+0.67)/2=1.05%, sigma 因分散下降约 30% -> 3.4%,
f*=9.0 —— **sigma^2 掉得比 mu 快, 凯利的"该下多重"反而上升**。而担保比例看的是回撤,
分散天然压回撤。两条效应同向, 所以"分散+高杠杆"值得实测, 这是历史上从未交叉过的二维格。

历史留档的口径差异: 第 114 轮"2×50% = 1229% vs 6392%"是在 **1.0x 杠杆**下测的,
那时杠杆不是自由度, 分散只是纯粹稀释 alpha; 本轮把它和杠杆一起放开。

用法: python3 tools/exp_lev_frontier_190.py [days]
输出: output/lev_frontier_190.json
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

# (exec_top_n, alloc_frac_pct) —— 单吊满仓 / 2 只各半 / 3 只各 1/3 / 4 只各 1/4
SLOTS = [(1, 100.0), (2, 50.0), (3, 33.3), (4, 25.0)]
LEVS = [1.5, 2.0, 2.5, 3.0]


def worst_ratio(lev: float, mdd_pct: float) -> float:
    """最坏时刻的担保比例: 自有 100 + 融资 (L-1)*100, 回撤后总资产 L*100*(1+mdd)。

    与两融口径一致: 维持担保比例 = 总资产 / 负债, <130% 触发强平。
    """
    if lev <= 1.0:
        return float("inf")
    eq = lev * 100.0 * (1.0 + mdd_pct / 100.0)
    debt = (lev - 1.0) * 100.0
    return eq / debt * 100.0


def main() -> None:
    base = load_config()
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    print(f"== 分散 x 杠杆二维前沿: {days} 日 ==", flush=True)
    ctx = prepare_context(base, days)

    rows = []
    for top_n, alloc in SLOTS:
        for L in LEVS:
            cfg = copy.deepcopy(base)
            cfg["selection"]["exec_top_n"] = int(top_n)
            cfg["execution"]["alloc_frac_pct"] = float(alloc)
            cfg["execution"]["leverage"] = float(L)
            r = run_backtest(cfg, days, ctx)
            m = r["metrics"]
            row = {
                "top_n": top_n, "alloc": alloc, "lev": L,
                "total": m["总收益率%"], "ann": m["年化收益率%"],
                "mdd": m["最大回撤%"], "calmar": m["卡玛比率"],
                "sharpe": m["夏普比率"], "n": m["交易次数"],
                "win": m["胜率%"], "avg": m["平均单笔收益%"],
                "ratio": worst_ratio(L, m["最大回撤%"]),
            }
            rows.append(row)
            print(f"top{top_n} L={L:.2f}  {row['total']:>16,.0f}%  "
                  f"年化{row['ann']:6.1f}%  回撤{row['mdd']:7.2f}%  "
                  f"卡玛{row['calmar']:5.2f}  {row['n']:4d}笔 单笔{row['avg']:.3f}%  "
                  f"担保{row['ratio']:5.0f}%", flush=True)

    p = ROOT / "output" / "lev_frontier_190.json"
    p.write_text(json.dumps({"days": days, "rows": rows}, ensure_ascii=False, indent=1))
    print(f"\n已写 {p}")


if __name__ == "__main__":
    main()
