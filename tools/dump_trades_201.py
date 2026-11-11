"""第 201 轮：把主策略 355/717 笔的全字段导出来，看有没有"该跳过的一档"。

回测引擎每笔都记了 score（打分）、gap_pct（T+1 开盘缺口）、pool_n（当日池深）、
pos_frac（下注比例）。此前 195 轮 GA 只在参数空间里搜，从没单独检验过
"score 低的那一档是不是负期望" —— 如果低档确实是拖累，跳过它就是净增益，
而且不需要新数据、不需要新逻辑（引擎已有 score_gate 实现）。
"""
import json
import os
import sys
from pathlib import Path

import pandas as pd
import yaml

os.environ["BT_FULL_TRADES"] = "1"
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402


def main(days):
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    ctx = prepare_context(cfg, days)
    r = run_backtest(cfg, days=days, ctx=ctx)
    print(days, json.dumps(r["metrics"], ensure_ascii=False))
    t = pd.DataFrame([x for x in r["trades_all"] if x.get("ret_pct") is not None])
    t = t.rename(columns={"ret_pct": "ret", "pool_n": "pn"})
    cols = ["code", "signal_date", "buy_date", "sell_date", "ret", "score", "gap_pct", "pn", "pos_frac"]
    t[cols].to_csv(ROOT / "output" / f"trades_201_{days}.csv", index=False)
    print("saved", len(t))


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 1500)
