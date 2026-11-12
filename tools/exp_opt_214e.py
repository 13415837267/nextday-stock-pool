#!/usr/bin/env python3.11
"""第 214 轮: 卖出端提前止损参数扫描(sell_close_if_open_drop)。"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import prepare_context, run_backtest
from src.config import load_config


def _M(m: dict) -> dict:
    return {"年化%": m["年化收益率%"], "回撤%": m["最大回撤%"], "卡玛": m["卡玛比率"],
            "笔数": m["交易次数"], "胜率%": m["胜率%"], "均单%": m["平均单笔收益%"]}


def main() -> None:
    base = load_config()
    ctx = prepare_context(base, 1500)
    rows = []
    for v in [0.0, 1.0, 2.0, 3.0, 4.0]:
        cfg = copy.deepcopy(base)
        cfg["execution"]["sell_close_if_open_drop"] = v
        r = run_backtest(cfg, 1500, ctx)
        row = _M(r["metrics"])
        row["阈值"] = v
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    out = Path(__file__).resolve().parent.parent / "output" / "opt_214_sell.json"
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
