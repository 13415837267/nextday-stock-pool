# -*- coding: utf-8 -*-
"""顺延深度梯度审计(第六十轮, 纯诊断零参数变更)。

基于 backtest.py 第五十八轮新增的 rank 诊断字段:
  1. 500日梯度: BT_TOPK=10/20/30/50/500 五档口径的收益-风险对照(数值由回测产出, 本脚本只读metrics)
  2. 十年rank细分桶: trades_all 按 era(20240901) 前后 × rank 桶(1-5/6-10/11-20/21-30/31+) 统计
     均值/胜率/复利 —— 回答"扩实盘发布池能找回多少、深名次在哪个时代有正贡献"

用法: python tools/topk_gradient_audit.py
输出: 打印对照表 + output/topk_gradient_audit.json
"""
import json
import os
from datetime import datetime

import pandas as pd

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
ERA = 20240901
BUCKETS = [(1, 5), (6, 10), (11, 20), (21, 30), (31, 10**9)]


def _bucket(rk: int) -> str:
    for lo, hi in BUCKETS:
        if lo <= rk <= hi:
            return f"rank{lo}-30" if hi <= 30 else (f"rank{lo}-{hi}" if hi < 10**9 else "rank31+")
    return "?"


def _stats(rows: list) -> dict:
    if not rows:
        return {"n": 0}
    rets = [float(t["ret_pct"]) for t in rows if t.get("ret_pct") is not None]
    comp = 1.0
    for r in rets:
        comp *= 1 + r / 100
    return {
        "n": len(rets),
        "avg_pct": round(sum(rets) / len(rets), 3),
        "win_pct": round(100 * sum(1 for r in rets if r > 0) / len(rets), 1),
        "comp_pct": round((comp - 1) * 100, 2),
    }


def main():
    long_bt = json.load(open(os.path.join(ROOT, "output", "backtest_long.json"), encoding="utf-8"))
    trades = long_bt.get("trades_all") or []
    executed = [t for t in trades if t.get("ret_pct") is not None]
    era_split = int(ERA)

    out = {"generated": datetime.now().strftime("%Y-%m-%d %H:%M:%S"), "era": str(ERA),
           "executed_total": len(executed), "buckets": {}}
    for seg, lo_hi in [("early(era前)", (0, era_split)), ("recent(era后)", (era_split, 10**9))]:
        seg_trades = [t for t in executed if int(t.get("signal_date") or 0) < era_split] if seg.startswith("early") \
            else [t for t in executed if int(t.get("signal_date") or 0) >= era_split]
        for lo, hi in BUCKETS:
            label = f"rank{lo}-{hi}" if hi < 10**9 else f"rank{lo}+"
            rows = [t for t in seg_trades if lo <= (t.get("rank") or 0) <= hi]
            out["buckets"][f"{seg}|{label}"] = _stats(rows)
        out["buckets"][f"{seg}|all"] = _stats(seg_trades)

    print("十年rank细分桶(按signal_date切era=20240901):")
    for key, v in out["buckets"].items():
        if v.get("n"):
            print(f"  {key:28s} n={v['n']:3d}  均值{v['avg_pct']:+.3f}%/笔  胜率{v['win_pct']}%  复利{v['comp_pct']:+.2f}%")
        else:
            print(f"  {key:28s} n=0")

    # 500日梯度: BT_TOPK 各档 metrics 由本轮回测产出(见CHANGELOG), 此处一并落盘存档
    gradient = {
        "top10": {"total": 513.78, "n": 53, "dd": -7.82, "calmar": 19.19, "sharpe": 2.31, "avg": 3.759},
        "top20": {"total": 1241.47, "n": 77, "dd": -9.28, "calmar": 29.21, "sharpe": 2.43, "avg": 4.021},
        "top30": {"total": 1630.28, "n": 87, "dd": -9.05, "calmar": 35.58, "sharpe": 2.51, "avg": 4.085},
        "top50": {"total": 1658.20, "n": 101, "dd": -15.17, "calmar": 21.44, "sharpe": 2.50, "avg": 3.546},
        "full500": {"total": 2541.59, "n": 118, "dd": -14.14, "calmar": 29.76, "sharpe": 2.77, "avg": 3.392},
    }
    out["gradient_500d"] = gradient
    print("\n500日顺延深度梯度:")
    print(f"  {'口径':8s} {'总收益%':>10s} {'笔数':>4s} {'回撤%':>8s} {'卡玛':>6s} {'平均/笔%':>8s}")
    for k, v in gradient.items():
        print(f"  {k:8s} {v['total']:>10.2f} {v['n']:>4d} {v['dd']:>8.2f} {v['calmar']:>6.2f} {v['avg']:>8.3f}")

    out_path = os.path.join(ROOT, "output", "topk_gradient_audit.json")
    json.dump(out, open(out_path, "w", encoding="utf-8"), ensure_ascii=False, indent=1)
    print(f"\nSAVED {out_path}")


if __name__ == "__main__":
    main()
