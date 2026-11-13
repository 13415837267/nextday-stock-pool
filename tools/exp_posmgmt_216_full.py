#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""第 216 轮: 仓位管理亮点变体的全史(3664日)稳健性验证"""
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
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 3664
    base = load_config()
    ctx = prepare_context(base, days)
    gapoff = {"gap_above_tiers": []}
    pool11 = {"pool_n_tiers": [{"min": 21, "mult": 1.0}, {"min": 11, "mult": 1.0}, {"min": 1, "mult": 0.5}]}
    poolmid = {"pool_n_tiers": [{"min": 21, "mult": 1.0}, {"min": 11, "mult": 0.7}, {"min": 1, "mult": 0.5}]}
    poolfull = {"pool_n_tiers": [{"min": 1, "mult": 1.0}]}

    V = [
        ("基线", {}),
        ("gap_above_关闭", gapoff),
        ("池11满仓", pool11),
        ("池温和", poolmid),
        ("池全满仓", poolfull),
        ("gap关+池温和", {**gapoff, **poolmid}),
    ]
    rows = []
    for name, patch in V:
        c = copy.deepcopy(base)
        c["execution"].update(patch)
        try:
            r = run_backtest(c, days, ctx)
            row = M(r["metrics"]); row["变体"] = name
        except Exception as e:  # noqa: BLE001
            row = {"变体": name, "错误": str(e)[:150]}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    Path(f"output/posmgmt_216_full_{days}.json").write_text(
        json.dumps(rows, ensure_ascii=False, indent=1))


if __name__ == "__main__":
    main()
