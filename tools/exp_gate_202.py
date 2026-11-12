"""第 202 轮: 闸门长度复扫(面板先验说短端更好, 历史网格只测过 15~30).

用法: python3 tools/exp_gate_202.py <days> <ma1> [ma2 ...]
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

os.environ["BT_FULL_TRADES"] = "1"

import yaml  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402


def run(days: int, ma: int, breadth: float | None = None) -> dict:
    cfg = yaml.safe_load(open(ROOT / "config.yaml"))
    cfg["regime"]["market_ma"] = ma
    if breadth is not None:
        cfg["regime"]["breadth_min"] = breadth
    ctx = prepare_context(cfg, days)
    r = run_backtest(cfg, days=days, ctx=ctx)
    m = r["metrics"]
    return {"days": days, "ma": ma, "breadth": breadth,
            "年化%": m["年化收益率%"], "回撤%": m["最大回撤%"],
            "卡玛": m["卡玛比率"], "笔数": m["交易次数"], "单笔%": m["平均单笔收益%"],
            "总收益%": m["总收益率%"]}


if __name__ == "__main__":
    days = int(sys.argv[1])
    mas = [int(x) for x in sys.argv[2:]] or [20]
    f = ROOT / "output" / f"exp_gate_202_{days}.json"
    rows = json.load(open(f)) if f.exists() else []
    have = {r["ma"] for r in rows}
    for ma in mas:
        if ma in have:
            continue
        try:
            row = run(days, ma)
        except Exception as e:  # noqa: BLE001
            print(days, ma, "ERR", type(e).__name__, str(e)[:120], flush=True)
            continue
        rows.append(row)
        json.dump(rows, open(f, "w"), ensure_ascii=False, indent=1)
        print(f'  {days}日 MA{ma}: 年化 {row["年化%"]:.2f}% 回撤 {row["回撤%"]:.2f}% '
              f'卡玛 {row["卡玛"]:.2f} 笔数 {row["笔数"]}', flush=True)
