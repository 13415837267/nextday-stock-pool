"""第255轮: "只做评分顶尖的" —— 把严格化这条路真正跑一遍

用户的问题: "为什么一直在加股票, 我要的是评分顶尖的, 而不是继续加"。

这句质疑对应的硬事实:
  · 253 (pool_n_tiers 11-20 只 0.3→1.0) 与 254 (11-15 只 0→0.5) 两轮的方向都是**放宽**;
  · 254 的放宽只换来 +0.11pp 年化 (96.43→96.54) 和 +12 笔, 价格极低, 但方向确实是"加"。

本组把"严格化"的每一个旋钮单独拆开, 看各自的代价:
  A 254 现状       {21:1, 16:1, 11:0.5, 1:0}   补位开
  B 退回 253       {21:1, 11:1, 1:0}          补位开   ← 只退掉 254 的放宽
  C 只做 21 只以上  {21:1, 1:0}                补位开
  D 只做 31 只以上  {31:1, 1:0}                补位开
  E 分母 40 只以上  {41:1, 1:0}                补位开
  F 关补位(A 口径)  {21:1, 16:1, 11:0.5, 1:0}  补位关   ← 纯主策略
  G 评分门 1.2      B 口径 + min_top1_score 1.2 补位开
  H 评分门 1.4      B 口径 + min_top1_score 1.4 补位开
"""

from __future__ import annotations

import copy
import os
import pickle
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

os.environ["BT_FULL_TRADES"] = "1"

T254 = [{"min": 21, "mult": 1.0}, {"min": 16, "mult": 1.0},
        {"min": 11, "mult": 0.5}, {"min": 1, "mult": 0.0}]
T253 = [{"min": 21, "mult": 1.0}, {"min": 11, "mult": 1.0}, {"min": 1, "mult": 0.0}]

CASES = [
    ("A 254现状", T254, True, 1.0),
    ("B 退回253", T253, True, 1.0),
    ("C 只做21+", [{"min": 21, "mult": 1.0}, {"min": 1, "mult": 0.0}], True, 1.0),
    ("D 只做31+", [{"min": 31, "mult": 1.0}, {"min": 1, "mult": 0.0}], True, 1.0),
    ("E 只做41+", [{"min": 41, "mult": 1.0}, {"min": 1, "mult": 0.0}], True, 1.0),
    ("F 关补位", T254, False, 1.0),
    ("G 评分门1.2", T253, True, 1.2),
    ("H 评分门1.4", T253, True, 1.4),
]


def main() -> None:
    cfg0 = load_config()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    only = sys.argv[2].split(",") if len(sys.argv) > 2 else None
    ctx = prepare_context(cfg0, n)

    out = []
    for nm, tiers, fill, gate in CASES:
        if only and nm.split()[0] not in only:
            continue
        c = copy.deepcopy(cfg0)
        c["execution"]["pool_n_tiers"] = copy.deepcopy(tiers)
        c["execution"]["pool_rate_tiers"] = []
        c["execution"]["min_top1_score"] = gate
        c["fillin"]["enabled"] = fill
        r = run_backtest(c, n, ctx)
        m = r["metrics"]
        tr = pd.DataFrame(r.get("trades_all") or [])
        out.append({"name": nm, "m": m, "trades": tr, "equity": r.get("equity")})
        print(f"\n【{nm}】 年化{m['年化收益率%']} 回撤{m['最大回撤%']} 卡玛{m['卡玛比率']}"
              f" 笔数{m['交易次数']} 胜率{m['胜率%']} 单笔{m['平均单笔收益%']}", flush=True)

    print("\n" + "=" * 96)
    print(f"{'方案':<14}{'年化':>9}{'回撤':>9}{'卡玛':>7}{'笔数':>6}{'胜率':>7}{'单笔':>8}{'空仓':>7}")
    for o in out:
        m = o["m"]
        print(f"{o['name']:<14}{m['年化收益率%']:>9.2f}{m['最大回撤%']:>9.2f}"
              f"{m['卡玛比率']:>7.2f}{m['交易次数']:>6}{m['胜率%']:>7.2f}"
              f"{m['平均单笔收益%']:>8.2f}{m['空仓天数']:>7}")

    tag = f"w255_strict_{n}{('_' + ''.join(only)) if only else ''}.pkl"
    with open(ROOT / "output" / tag, "wb") as f:
        pickle.dump(out, f)
    print(f"\n明细已存 output/{tag}")


if __name__ == "__main__":
    main()
