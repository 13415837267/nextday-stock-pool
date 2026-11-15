"""第232轮 · 导出 2500 日回测的逐笔明细（带 code 与信号日），供分钟线特征使用"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

OUT = ROOT / "output" / "w232_trades.csv"


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    os.environ["BT_FULL_TRADES"] = "1"
    cfg = load_config()
    ctx = prepare_context(cfg, days)
    res = run_backtest(cfg, None, ctx)
    ts = [t for t in res["trades_all"] if t.get("ret_pct") is not None]
    d = pd.DataFrame([{
        "signal": t["signal_date"], "buy": t["buy_date"], "sell": t["sell_date"],
        "code": t["code"], "name": t.get("name"), "pool": t.get("pool"),
        "ret": t["ret_pct"], "score": t.get("score"), "margin": t.get("margin"),
        "pool_n": t.get("pool_n"), "rank": t.get("rank"),
    } for t in ts])
    d.to_csv(OUT, index=False, encoding="utf-8-sig")
    print(f"{len(d)} 笔 -> {OUT}")
    print(d.groupby("pool")["ret"].agg(["count", "mean"]).round(2).to_string())


if __name__ == "__main__":
    main()
