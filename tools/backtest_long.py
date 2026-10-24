# -*- coding: utf-8 -*-
"""长窗口(十年)回测入口: 结果独立落盘, 不覆盖500日基准 output/backtest.json。

第四十二轮内存改造后长窗口可在受限环境跑通(见 src/backtest.py):
  - 面板数值列 float32(内存减半)
  - 逐票历史懒加载 _LazyPerCode(消除全量groupby复制)

用法: python3 -m tools.backtest_long [days]      默认 2500
输出: output/backtest_long.json (含 trades_all 全量成交, 供长窗口检验复用)
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import prepare_context, run_backtest, save_result
from src.config import ROOT, load_config


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    os.environ["BT_FULL_TRADES"] = "1"   # 全量成交入结果(供长窗口检验逐笔复用)
    cfg = load_config()
    print(f"== 长窗口回测: {days} 交易日(float32面板+懒加载逐票索引) ==", flush=True)
    ctx = prepare_context(cfg, days)
    result = run_backtest(cfg, days, ctx)
    path = save_result(result, str(ROOT / "output" / "backtest_long.json"))
    m = result["metrics"]
    print("\n========== 长窗口回测结果 ==========")
    print(f"区间: {result['meta']['start']} ~ {result['meta']['end']} ({result['meta']['days']}个交易日)")
    for k, v in m.items():
        print(f"  {k}: {v}")
    print(f"报告已写入: {path}")


if __name__ == "__main__":
    main()
