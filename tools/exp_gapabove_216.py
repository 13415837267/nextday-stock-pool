#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 216 轮: gap_above 高开降仓倍率细扫(找收益/回撤平衡点)"""
import copy
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402


def M(m):
    return {
        "年化%": round(m["年化收益率%"], 2),
        "回撤%": round(m["最大回撤%"], 2),
        "卡玛": round(m["卡玛比率"], 2),
        "笔数": m["交易次数"],
        "胜率%": round(m["胜率%"], 2),
        "均单%": round(m["平均单笔收益%"], 3),
    }


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    base = load_config()
    ctx = prepare_context(base, days)
    # 基线 gap_above: above=0.5, mult=0.5, score_below=1.6
    levels = [("基线(0.5)", 0.5), ("mult0.6", 0.6), ("mult0.7", 0.7),
              ("mult0.8", 0.8), ("mult0.9", 0.9), ("关闭(1.0)", None)]
    rows = []
    for name, mult in levels:
        c = copy.deepcopy(base)
        if mult is None:
            c["execution"]["gap_above_tiers"] = []
        else:
            c["execution"]["gap_above_tiers"] = [{"above": 0.5, "mult": mult, "score_below": 1.6}]
        try:
            r = run_backtest(c, days, ctx)
            row = M(r["metrics"]); row["变体"] = name
        except Exception as e:  # noqa: BLE001
            row = {"变体": name, "错误": str(e)[:150]}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    Path(f"output/gapabove_216_{days}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
