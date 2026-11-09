# -*- coding: utf-8 -*-
"""第一百七十八轮: 评分绝对水平分档下注 —— 并用同敞口对照证明不是杠杆变形。

发现链
------
1. exp_score_ic: 评分**排序**在可买样本里 IC≈0(t=+0.15), 第 1 名不如全池均值。
2. exp_factor_ic: 但评分**绝对水平**单调有效(rank1 可买 864 天):
       <1.0  -0.235% (n=293)   1.0~1.5  -0.351% (n=297)
       1.5~2.0 +0.840% (n=194)  2.0~2.5  +1.435% (n=76)
   现行门槛 1.0 恰好把 297 天**负期望**的票放进来了。
3. 直接抬门槛到 1.5: 单笔 +1.518% 但笔数 352→204, 年化 183.91%→98.40%(复利下笔数即收益)。
4. 所以正确用法不是"不买", 而是"少买" —— 保留全部 352 笔, 按评分绝对值缩放仓位。

本实验
------
对 score_tiers 做网格, 并对每组计算**均敞口**(trades[*].pos_frac 的均值 × leverage),
再用"纯杠杆达到同样敞口"的对照组判断: 是真 alpha 还是杠杆变形。

用法: python tools/exp_scoretiers.py [--days 1500]
输出: output/exp_scoretiers.json
"""
from __future__ import annotations

import argparse
import copy
import json
import os
import sys

import numpy as np

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
os.environ["BT_FULL_TRADES"] = "1"

from src.backtest import prepare_context, run_backtest   # noqa: E402
from src.config import load_config                      # noqa: E402


GROUPS = [
    # name,            tiers  (None=不改, 走 config)
    ("base",           [{"min": 1.2, "mult": 1.0}, {"min": 1.0, "mult": 1.0}]),
    ("A_2-15_15-125_12-08", [{"min": 2.0, "mult": 1.5}, {"min": 1.5, "mult": 1.25},
                             {"min": 1.2, "mult": 0.8}, {"min": 1.0, "mult": 1.0}]),
    ("B_15-125_12-10", [{"min": 1.5, "mult": 1.25}, {"min": 1.2, "mult": 1.0},
                        {"min": 1.0, "mult": 1.0}]),
    ("C_2-13_15-115_12-08", [{"min": 2.0, "mult": 1.3}, {"min": 1.5, "mult": 1.15},
                             {"min": 1.2, "mult": 0.8}, {"min": 1.0, "mult": 1.0}]),
    ("D_2-12_12-07", [{"min": 2.0, "mult": 1.2}, {"min": 1.2, "mult": 0.7},
                      {"min": 1.0, "mult": 1.0}]),
    ("E_15-15_12-06", [{"min": 1.5, "mult": 1.5}, {"min": 1.2, "mult": 0.6},
                       {"min": 1.0, "mult": 1.0}]),
]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=1500)
    ap.add_argument("--out", default="output/exp_scoretiers.json")
    a = ap.parse_args()

    cfg0 = load_config()
    ctx = prepare_context(cfg0, a.days)
    rows = []
    print("%-22s %9s %9s %8s %6s %6s %7s %8s" %
          ("组", "年化%", "回撤%", "卡玛", "夏普", "笔数", "单笔%", "均敞口"))
    print("-" * 82, flush=True)

    for name, tiers in GROUPS:
        cfg = copy.deepcopy(cfg0)
        cfg["execution"]["score_tiers"] = tiers
        r = run_backtest(cfg, a.days, ctx)
        m = r["metrics"]
        real = [t for t in (r.get("trades_all") or []) if t.get("ret_pct") is not None]
        lev = float(cfg["execution"].get("leverage", 1.0))
        expo = float(np.mean([float(t.get("pos_frac") or 1.0) for t in real])) * lev if real else 0.0
        rows.append({"name": name, "tiers": tiers, "lev": lev,
                     "metrics": m, "expo": round(expo, 4)})
        print("%-22s %8.2f%% %8.2f%% %7.2f %6.2f %6d %+7.3f%% %8.4f" %
              (name, m["年化收益率%"], m["最大回撤%"], m["卡玛比率"], m["夏普比率"],
               m["交易次数"], m["平均单笔收益%"], expo), flush=True)

    # 同敞口对照: 用最好那组的敞口, 回推纯杠杆倍数, 只改 leverage 不动 score_tiers
    base = next(x for x in rows if x["name"] == "base")
    best = max(rows, key=lambda x: x["metrics"]["卡玛比率"] or 0)
    if best["name"] != "base" and base["expo"] > 0:
        ratio = best["expo"] / base["expo"]
        lev_eq = round(float(cfg0["execution"].get("leverage", 1.0)) * ratio, 3)
        print(f"\n同敞口对照: {best['name']} 均敞口 {best['expo']:.4f} vs base {base['expo']:.4f}"
              f" (+{(ratio-1)*100:.1f}%) -> 纯杠杆 {lev_eq}x")
        cfg = copy.deepcopy(cfg0)
        cfg["execution"]["leverage"] = lev_eq
        cfg["execution"]["score_tiers"] = GROUPS[0][1]
        r = run_backtest(cfg, a.days, ctx)
        m = r["metrics"]
        real = [t for t in (r.get("trades_all") or []) if t.get("ret_pct") is not None]
        expo = float(np.mean([float(t.get("pos_frac") or 1.0) for t in real])) * lev_eq if real else 0.0
        print("%-22s %8.2f%% %8.2f%% %7.2f %6.2f %6d %+7.3f%% %8.4f" %
              (f"EQ_lev{lev_eq}", m["年化收益率%"], m["最大回撤%"], m["卡玛比率"],
               m["夏普比率"], m["交易次数"], m["平均单笔收益%"], expo), flush=True)
        rows.append({"name": f"EQ_lev{lev_eq}", "tiers": GROUPS[0][1], "lev": lev_eq,
                     "metrics": m, "expo": round(expo, 4), "is_equiv": True})

    p = os.path.join(ROOT, a.out)
    open(p, "w", encoding="utf-8").write(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"\n写出 {p}")


if __name__ == "__main__":
    main()
