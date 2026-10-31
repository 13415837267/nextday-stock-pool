# -*- coding: utf-8 -*-
"""第一百零三轮: 刀口保护全史/分窗验证。

用法: python3 tools/knife_validate.py --days 3378 --variants base,k30,k15
输出: 控制台 + output/knife_validate_<days>.json
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

V = {
    "base": (None, None),
    "k30": ({"mkt_above": -0.5, "res_below": -4.0, "mult": 0.30}, None),
    "k15": ({"mkt_above": -0.5, "res_below": -4.0, "mult": 0.15}, None),
    "k30f": ({"mkt_above": -0.5, "res_below": -4.0, "mult": 0.30}, {"mkt_below": -1.0, "floor": 1.0}),
    "f05": (None, {"mkt_below": -0.5, "floor": 1.0}),
    "f10": (None, {"mkt_below": -1.0, "floor": 1.0}),
    "f15": (None, {"mkt_below": -1.5, "floor": 1.0}),
    "k30f05": ({"mkt_above": -0.5, "res_below": -4.0, "mult": 0.30}, {"mkt_below": -0.5, "floor": 1.0}),
    "k30f15": ({"mkt_above": -0.5, "res_below": -4.0, "mult": 0.30}, {"mkt_below": -1.5, "floor": 1.0}),
    "k15f10": ({"mkt_above": -0.5, "res_below": -4.0, "mult": 0.15}, {"mkt_below": -1.0, "floor": 1.0}),
    "k30f12": ({"mkt_above": -0.5, "res_below": -4.0, "mult": 0.30}, {"mkt_below": -1.2, "floor": 1.0}),
    "k30f20": ({"mkt_above": -0.5, "res_below": -4.0, "mult": 0.30}, {"mkt_below": -2.0, "floor": 1.0}),
    "k20f15": ({"mkt_above": -0.5, "res_below": -4.0, "mult": 0.20}, {"mkt_below": -1.5, "floor": 1.0}),
    "k40f15": ({"mkt_above": -0.5, "res_below": -4.0, "mult": 0.40}, {"mkt_below": -1.5, "floor": 1.0}),
}


def yearly(res: dict) -> dict:
    eq = res.get("equity") or []
    s = pd.Series([e["nav"] for e in eq], index=[e["d"] for e in eq])
    out = {}
    for y, g in s.groupby(lambda i: i[:4]):
        out[y] = round((g.iloc[-1] / g.iloc[0] - 1) * 100, 1)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=3378)
    ap.add_argument("--variants", type=str, default="base,k30,k15")
    args = ap.parse_args()
    cfg0 = load_config()
    t0 = time.time()
    ctx = prepare_context(cfg0, args.days)
    print(f"ctx ready {time.time()-t0:.0f}s", flush=True)
    rows = []
    base_y = None
    for key in args.variants.split(","):
        tier, floor = V[key]
        c = copy.deepcopy(cfg0)
        c["execution"].pop("knife_tiers", None)
        c["execution"].pop("mkt_floor_tiers", None)
        if tier:
            c["execution"]["knife_tiers"] = [tier]
        if floor:
            c["execution"]["mkt_floor_tiers"] = [floor]
        t1 = time.time()
        r = run_backtest(c, args.days, ctx)
        m = r["metrics"]
        y = yearly(r)
        if base_y is None:
            base_y = y
        print(f"{key:<6s} {m['总收益率%']:>9.2f}% mdd {m['最大回撤%']:>7.2f}% calmar {m['卡玛比率']:>5.2f} "
              f"sharpe {m['夏普比率']:>4.2f} n {m['交易次数']:>4d} wr {m['胜率%']:>5.2f} avg {m['平均单笔收益%']:>5.2f} "
              f"({time.time()-t1:.0f}s) | {y} | 差 {({k: round(y[k]-base_y.get(k,0),1) for k in y})}", flush=True)
        rows.append({"key": key, "ret": m["总收益率%"], "mdd": m["最大回撤%"], "calmar": m["卡玛比率"],
                     "sharpe": m["夏普比率"], "n": m["交易次数"], "wr": m["胜率%"], "yearly": y})
        (ROOT / "output" / f"knife_validate_{args.days}.json").write_text(
            json.dumps({"days": args.days, "rows": rows}, ensure_ascii=False, indent=1), encoding="utf-8")


if __name__ == "__main__":
    main()
