"""第253轮 B: 2500 日定稿方案的 1500 日窗口复核

A/B 在 2500 日上: 89.47/5.20 -> 96.43/5.60。
252 定稿(现行)的 1500 日基线是 144.69% / -17.21% / 8.41, 这里同窗口单跑 B 一组。
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

TIERS = [{"min": 21, "mult": 1.0}, {"min": 11, "mult": 1.0}, {"min": 1, "mult": 0.0}]


def main() -> None:
    cfg0 = load_config()
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    c = copy.deepcopy(cfg0)
    c["execution"]["pool_n_tiers"] = copy.deepcopy(TIERS)
    c["backtest"]["trading_days"] = n
    ctx = prepare_context(c, n)
    r = run_backtest(c, n, ctx)
    m = r["metrics"]
    print(f"\n【B 1.0/1.0/0 @ {n}日】 年化{m['年化收益率%']} 回撤{m['最大回撤%']} "
          f"卡玛{m['卡玛比率']} 笔数{m['交易次数']} 胜率{m['胜率%']} 单笔{m['平均单笔收益%']}")
    with open(ROOT / "output" / f"w253b_b_{n}.pkl", "wb") as f:
        pickle.dump([{"name": f"B {n}", "m": m, "equity": r.get("equity")}], f)


if __name__ == "__main__":
    main()
