# -*- coding: utf-8 -*-
"""归因实验：alpha 到底在"缺口"还是在"选股方向"？（第186轮）

假设
----
右侧（追强/回踩）之所以全面失败，不是因为选股差，而是因为**成交价没有折价**。
主策略 dip 买在 T+1 深低开，赚的是隔夜恐慌抛售造成的流动性折价；
右侧买在平开/高开，等于在别人也要买的位置成交。

如果这个假设成立，那么把**任何一个模式的成交**按 T+1 开盘缺口分组，都会看到
「缺口越低 → 单笔越高」的单调关系，而且右侧模式在低开箱里的单笔也不差。

做法
----
对 dip / trend / pullback / limitup 四个模式各跑一次完整回测，取全部成交
（含 gap_pct），按缺口分 8 箱，输出每箱笔数 / 单笔均值 / 胜率 / 贡献总收益。
同一模式内部横向比较，可以排除"不同模式选股不同"的干扰。

用法: python3 tools/exp_gap_attrib.py --days 1500
"""
import argparse
import copy
import json
import os
import sys
from pathlib import Path

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["BT_FULL_TRADES"] = "1"

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

BINS = [(-99, -4), (-4, -3), (-3, -2), (-2, -1), (-1, 0), (0, 1), (1, 2), (2, 99)]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--out", default="")
    args = ap.parse_args()

    cfg0 = load_config()
    ctx = prepare_context(cfg0, args.days)
    print(f"窗口 {len(ctx['main_dates'])} 日\n", flush=True)

    modes = ["dip", "trend", "pullback", "limitup"]
    out = {"days": args.days, "modes": {}}

    for mode in modes:
        c = copy.deepcopy(cfg0)
        c["selection"]["mode"] = mode
        c["selection"]["fallbacks"] = []
        c["selection"]["fallback_slot"] = True
        if mode in ("trend", "pullback", "limitup"):
            c["execution"]["buy_gap_below"] = 0        # 不挑缺口, 让全部箱子都能成交
        res = run_backtest(c, args.days, ctx)
        tr = [t for t in res.get("trades_all", [])
              if t.get("ret_pct") is not None and t.get("gap_pct") is not None]
        if not tr:
            print(f"{mode}: 无成交", flush=True)
            continue
        gaps = np.array([t["gap_pct"] for t in tr], dtype=float)
        rets = np.array([t["ret_pct"] for t in tr], dtype=float)

        rows = []
        print(f"=== {mode} ({len(tr)} 笔, 全样本单笔 {rets.mean():+.2f}%, "
              f"胜率 {(rets > 0).mean() * 100:.1f}%) ===", flush=True)
        print(f"  {'缺口箱':>12s}{'笔数':>7s}{'占比':>7s}{'单笔%':>9s}"
              f"{'胜率%':>8s}{'累计贡献%':>10s}", flush=True)
        for lo, hi in BINS:
            m = (gaps >= lo) & (gaps < hi)
            if m.sum() == 0:
                rows.append({"bin": f"[{lo},{hi})", "n": 0})
                continue
            r = rets[m]
            # 累计贡献: 复利近似 = sum(ret), 反映该箱对总收益的推力
            contrib = float(r.sum())
            rows.append({"bin": f"[{lo},{hi})", "n": int(m.sum()),
                         "share%": round(100 * m.mean(), 1),
                         "avg%": round(float(r.mean()), 3),
                         "win%": round(100 * float((r > 0).mean()), 1),
                         "wsum%": round(contrib, 1)})
            print(f"  {f'[{lo},{hi})':>12s}{m.sum():>7d}{100 * m.mean():>6.1f}%"
                  f"{r.mean():>+9.2f}%{100 * (r > 0).mean():>7.1f}%{contrib:>+10.1f}",
                  flush=True)
        # 低开 vs 非低开
        lo_m = gaps < 0
        out["modes"][mode] = {
            "n": len(tr), "avg%": round(float(rets.mean()), 3),
            "win%": round(100 * float((rets > 0).mean()), 1),
            "low_open": {"n": int(lo_m.sum()), "avg%": round(float(rets[lo_m].mean()), 3) if lo_m.sum() else None,
                         "win%": round(100 * float((rets[lo_m] > 0).mean()), 1) if lo_m.sum() else None},
            "high_open": {"n": int((~lo_m).sum()), "avg%": round(float(rets[~lo_m].mean()), 3) if (~lo_m).sum() else None,
                          "win%": round(100 * float((rets[~lo_m] > 0).mean()), 1) if (~lo_m).sum() else None},
            "bins": rows,
        }
        print(f"  → 低开 {int(lo_m.sum())} 笔 单笔 {rets[lo_m].mean():+.2f}% | "
              f"平/高开 {int((~lo_m).sum())} 笔 单笔 {rets[~lo_m].mean():+.2f}%\n", flush=True)

    p = Path(args.out or f"output/exp_gap_attrib_{args.days}.json")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(f"写出 {p}", flush=True)


if __name__ == "__main__":
    main()
