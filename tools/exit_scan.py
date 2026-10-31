# -*- coding: utf-8 -*-
"""第一百零四轮扫描: 出场端"高浮盈开盘全卖"阈值 + 分批比例复核。

用法: python3 tools/exit_scan.py --days 1500 --variants base,sfg4,sfg5,sfg6,sfg8,of04,of06
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

OUT = ROOT / "output" / "exit_scan.json"


def yearly(res: dict) -> dict:
    s = pd.Series({e["d"]: e["nav"] for e in res.get("equity") or []})
    if s.empty:
        return {}
    return {y: round((g.iloc[-1] / g.iloc[0] - 1) * 100, 1) for y, g in s.groupby(s.index.str[:4])}


V = {
    "base":  ({}, {}),
    "sfg3":  ({"sell_full_if_open_gain": 3.0}, {}),
    "sfg4":  ({"sell_full_if_open_gain": 4.0}, {}),
    "sfg5":  ({"sell_full_if_open_gain": 5.0}, {}),
    "sfg6":  ({"sell_full_if_open_gain": 6.0}, {}),
    "sfg8":  ({"sell_full_if_open_gain": 8.0}, {}),
    "of04":  ({}, {"sell_open_frac": 0.4}),
    "of06":  ({}, {"sell_open_frac": 0.6}),
    "of03":  ({}, {"sell_open_frac": 0.3}),
    "of02":  ({}, {"sell_open_frac": 0.2}),
    "of00":  ({}, {"sell_open_frac": 0.0}),
    "sfg5of03": ({"sell_full_if_open_gain": 5.0}, {"sell_open_frac": 0.3}),
    "sfg5of02": ({"sell_full_if_open_gain": 5.0}, {"sell_open_frac": 0.2}),
    "of04scd15": ({}, {"sell_open_frac": 0.4, "sell_close_if_open_drop": 1.5}),
    "of04scd25": ({}, {"sell_open_frac": 0.4, "sell_close_if_open_drop": 2.5}),
    "of04scd30": ({}, {"sell_open_frac": 0.4, "sell_close_if_open_drop": 3.0}),
    "of04scd00": ({}, {"sell_open_frac": 0.4, "sell_close_if_open_drop": 0.0}),
    "bof08": ({}, {"sell_open_frac": 0.4, "buy_open_frac": 0.8}),
    "bof06": ({}, {"sell_open_frac": 0.4, "buy_open_frac": 0.6}),
    "bof04": ({}, {"sell_open_frac": 0.4, "buy_open_frac": 0.4}),
    "sfg5of04": ({"sell_full_if_open_gain": 5.0}, {"sell_open_frac": 0.4}),
    "sfg5of06": ({"sell_full_if_open_gain": 5.0}, {"sell_open_frac": 0.6}),
}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--variants", default="base,sfg4,sfg5,sfg6,sfg8")
    args = ap.parse_args()
    want = [x.strip() for x in args.variants.split(",") if x.strip()]
    cfg0 = load_config()
    t0 = time.time()
    ctx = prepare_context(cfg0, args.days)
    print(f"面板就绪 {time.time()-t0:.0f}s", flush=True)
    out = {"days": args.days, "rows": []}
    base_y = None
    for k in want:
        add, over = V[k]
        cfg = copy.deepcopy(cfg0)
        for kk, vv in over.items():
            cfg["execution"][kk] = vv
        for kk, vv in add.items():
            cfg["execution"][kk] = vv
        if k == "base":
            cfg["execution"].pop("sell_full_if_open_gain", None)
        t1 = time.time()
        res = run_backtest(cfg, args.days, ctx)
        m = res["metrics"]
        y = yearly(res)
        if base_y is None:
            base_y = y
        diff = {kk: round(y.get(kk, 0) - base_y.get(kk, 0), 1) for kk in y}
        print(f"{k:<12s} {m['总收益率%']:>9.2f}% mdd {m['最大回撤%']:>7.2f}% calmar {m['卡玛比率']:>5.2f} "
              f"n {m['交易次数']:>4d} wr {m['胜率%']:>5.2f} avg {m['平均单笔收益%']:>5.2f} "
              f"| 分年差 {diff} ({time.time()-t1:.0f}s)", flush=True)
        out["rows"].append({"key": k, **{kk: m[kk] for kk in
                                        ["总收益率%", "最大回撤%", "卡玛比率", "夏普比率", "交易次数", "胜率%", "平均单笔收益%"]},
                            "yearly": y})
        OUT.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"\n写入 {OUT}")


if __name__ == "__main__":
    main()
