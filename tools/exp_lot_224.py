#!/usr/bin/env python3
"""第224轮: 3 万本金 + 100 股整数倍 —— 回测里的"纸面收益"有多少来自买不满。

回测一直按连续股数下单(units = 金额/价格)。现实中:
  - 必须 100 股整数倍
  - 3 万 × 40% 仓位 = 1.2 万, 买不起 150 元的股票(1 手 1.5 万)
  - 3 万 × 100% = 3 万, 买 117 元的江南新材只能买 200 股(2.34 万), 实际仓位 78%

四个场景对照, 同一份面板、同一套信号:
  A 100万 + 无整数约束    = 历史口径(基准)
  B 100万 + 100股整数     = 只加约束, 看约束本身值多少钱
  C 3万   + 无整数约束    = 只改本金
  D 3万   + 100股整数     = 你的真实情况
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

    scenes = [
        ("A 100万 无整数约束", 1_000_000, 0),
        ("B 100万 100股整数", 1_000_000, 100),
        ("C 3万 无整数约束", 30_000, 0),
        ("D 3万 100股整数(真实)", 30_000, 100),
    ]
    rows = []
    for name, cap, lot in scenes:
        c = copy.deepcopy(cfg)
        c["backtest"]["initial_capital"] = float(cap)
        c["execution"]["lot_size"] = int(lot)
        t = time.time()
        r = run_backtest(c, days, ctx)
        m = r["metrics"]
        rows.append({
            "场景": name, "年化%": m["年化收益率%"], "回撤%": m["最大回撤%"],
            "卡玛": m["卡玛比率"], "笔数": m["交易次数"], "胜率%": m["胜率%"],
            "单笔%": m["平均单笔收益%"], "跳过": m["跳过买入次数"],
            "期末(万)": round(r["equity"][-1]["nav"] / 10000, 2),
        })
        print(f"  {name}: 年化 {m['年化收益率%']}% 回撤 {m['最大回撤%']}% "
              f"笔数 {m['交易次数']} 单笔 {m['平均单笔收益%']}% ({time.time()-t:.0f}s)", flush=True)

    df = pd.DataFrame(rows)
    print("\n=== 四场景对照 (1500日) ===")
    print(df.to_string(index=False))
    a, d = df.iloc[0], df.iloc[3]
    print(f"\n纸面 {a['年化%']}% -> 真实 {d['年化%']}%  "
          f"损耗 {a['年化%']-d['年化%']:.1f}pp, 笔数 {a['笔数']} -> {d['笔数']}")
    df.to_csv(ROOT / "output" / "lot_224.csv", index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
