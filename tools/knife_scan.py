# -*- coding: utf-8 -*-
"""第一百零三轮扫描: 阴跌刀口保护(mkt_ret 中位数 + 个股超额跌幅)阈值/乘数。

一次 prepare_context, 多次 run_backtest; 只改 execution.knife_tiers。
用法: python3 tools/knife_scan.py --days 1500
输出: 控制台 + output/knife_scan.json
"""
from __future__ import annotations
import os
os.environ.setdefault("BT_FULL_TRADES", "1")

import argparse
import copy
import json
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

OUT = ROOT / "output" / "knife_scan.json"


def metrics(res: dict) -> dict:
    m = res["metrics"]
    return {"ret": m["总收益率%"], "mdd": m["最大回撤%"], "calmar": m["卡玛比率"],
            "sharpe": m["夏普比率"], "n": m["交易次数"], "wr": m["胜率%"],
            "avg": m["平均单笔收益%"]}


def yearly(res: dict) -> dict:
    """按年复利——必须用净值曲线(仓位变化不改单笔收益, 只改资金路径)。"""
    eq = res.get("equity") or []
    if not eq:
        return {}
    s = pd.Series({e["d"][:4]: e["nav"] for e in eq})
    out = {}
    for y, g in s.groupby(level=0):
        first, last = g.iloc[0], g.iloc[-1]
        # 年内首日净值是"上一年末", 用日收益连乘更准
        out[y] = round((last / first - 1) * 100, 1)
    return out


VARIANTS = [
    ("base", None, None),
    ("knife m-0.5 r-4 x0.00", {"mkt_above": -0.5, "res_below": -4.0, "mult": 0.00}, None),
    ("knife m-0.5 r-4 x0.15", {"mkt_above": -0.5, "res_below": -4.0, "mult": 0.15}, None),
    ("knife m-0.5 r-4 x0.30", {"mkt_above": -0.5, "res_below": -4.0, "mult": 0.30}, None),
    ("knife m-0.5 r-4 x0.40", {"mkt_above": -0.5, "res_below": -4.0, "mult": 0.40}, None),
    ("knife m-0.5 r-3.5 x0.30", {"mkt_above": -0.5, "res_below": -3.5, "mult": 0.30}, None),
    ("knife m0.0 r-4 x0.30", {"mkt_above": 0.0, "res_below": -4.0, "mult": 0.30}, None),
    ("knife m-0.5 r-4 x0.30 + floor-1", {"mkt_above": -0.5, "res_below": -4.0, "mult": 0.30},
     {"mkt_below": -1.0, "floor": 1.0}),
    ("floor-1 only", None, {"mkt_below": -1.0, "floor": 1.0}),
]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    args = ap.parse_args()
    cfg0 = load_config()
    ctx = prepare_context(cfg0, args.days)
    out = {"days": args.days, "rows": []}
    base_y = None
    for key, tier, floor in VARIANTS:
        cfg = copy.deepcopy(cfg0)
        cfg["execution"].pop("knife_tiers", None)
        cfg["execution"].pop("mkt_floor_tiers", None)
        if tier is not None:
            cfg["execution"]["knife_tiers"] = [tier]
        if floor is not None:
            cfg["execution"]["mkt_floor_tiers"] = [floor]
        t0 = time.time()
        res = run_backtest(cfg, args.days, ctx)
        m = metrics(res)
        y = yearly(res)
        if base_y is None:
            base_y = y
        row = {"key": key, "tier": tier, "floor": floor, **m, "yearly": y, "secs": round(time.time() - t0)}
        out["rows"].append(row)
        diff = {k: round(y.get(k, 0) - base_y.get(k, 0), 1) for k in y}
        print(f"{key:<30s} {m['ret']:>9.2f}% mdd {m['mdd']:>7.2f}% calmar {m['calmar']:>5.2f} "
              f"n {m['n']:>4d} wr {m['wr']:>5.2f} avg {m['avg']:>5.2f} | 分年差 {diff}", flush=True)
        OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
