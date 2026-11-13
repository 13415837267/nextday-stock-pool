#!/usr/bin/env python3
"""第224轮(下): 成本假设值多少钱, 以及 3 万本金的真实天花板。

历史上所有年化数字都在两套假设上: 滑点 0.10% + 佣金 0.025%, 现金收益 1.5%。
但这两项在实盘里都不是固定的:
  - 集合竞价按开盘价成交, 3 万单在 2.5 亿成交额里参与率 0.012%, 真实滑点接近 0
  - 佣金万 1 是现在的主流报价(万 2.5 是不还价的默认值)
  - 空仓期 56% 的现金, 国债逆回购/货币 ETF 能到 1.8%, 不是 1.5%

口径: 3 万本金 + 100 股整数(第224轮新增约束), 1500 日, 其余参数不变。
"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    cfg = load_config()
    t0 = time.time()
    ctx = prepare_context(cfg, days)
    print(f"上下文 {len(ctx['main_dates'])} 日, {time.time()-t0:.0f}s\n", flush=True)

    base = {"execution": {"slippage": 0.001, "commission_rate": 0.00025, "cash_yield": 0.015}}
    scenes = [
        ("A 现行假设 滑千1+佣万2.5+现金1.5%", {}),
        ("B 免滑点   滑0+佣万2.5+现金1.5%", {"slippage": 0.0}),
        ("C 低费     滑千0.2+佣万1+现金1.5%", {"slippage": 0.0002, "commission_rate": 0.0001}),
        ("D 低费+现金1.8%  (完全可执行)", {"slippage": 0.0002, "commission_rate": 0.0001,
                                            "cash_yield": 0.018}),
        ("E 零成本  (理论上限)", {"slippage": 0.0, "commission_rate": 0.0,
                                   "stamp_tax": 0.0, "cash_yield": 0.018}),
    ]
    rows = []
    for name, over in scenes:
        c = copy.deepcopy(cfg)
        c["backtest"]["initial_capital"] = 30_000.0
        c["execution"]["lot_size"] = 100
        for k in ("slippage", "commission_rate", "cash_yield", "stamp_tax"):
            if k in over:
                c["execution"][k] = over[k]
        r = run_backtest(c, days, ctx)
        m = r["metrics"]
        rows.append({
            "场景": name, "年化%": m["年化收益率%"], "回撤%": m["最大回撤%"],
            "卡玛": m["卡玛比率"], "笔数": m["交易次数"], "单笔%": m["平均单笔收益%"],
        })
        print(f"  {name}: {m['年化收益率%']}% / {m['最大回撤%']}% / 单笔 {m['平均单笔收益%']}%", flush=True)

    df = pd.DataFrame(rows)
    print("\n=== 成本假设的年化价值 (3万本金/1500日) ===")
    print(df.to_string(index=False))
    a, dd, e = df.iloc[0], df.iloc[3], df.iloc[4]
    print(f"\n现行 -> 完全可执行: +{dd['年化%']-a['年化%']:.1f}pp ({a['年化%']}% -> {dd['年化%']}%)")
    print(f"现行 -> 理论零成本: +{e['年化%']-a['年化%']:.1f}pp  ← 成本全部空间的上限")
    df.to_csv(ROOT / "output" / "cost_224.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
