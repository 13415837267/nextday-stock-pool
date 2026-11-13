#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
第 216 轮: 仓位管理参数系统扫描(第214轮遗漏的维度)
  第214轮只扫了"选股门槛/择股/备用池/卖出端", 从未系统扫过 execution 里的仓位 tiers:
    pool_n_tiers    池深分档降仓(当前 21+->1.0, 11-20->0.5, 1-10->0.3)
    knife_tiers     阴跌刀口保护(当前 market> -0.5% 且个股超额跌>4% -> 仓位*0.0 弃单)
    gap_above_tiers 高开降仓(当前 高开>=0.5% 且评分<1.6 -> *0.5)
    mkt_boost_tiers 恐慌日加仓(当前 market -2.5~-1.0 -> *1.5, 受 pos_frac_cap=1.0 截断)
    score_tiers     评分分档下注(当前全 1.0, 等于未启用)
  不加杠杆(pos_frac_cap=1.0 保持不动)。
"""
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


def variants(base):
    ex = "execution"
    V = []
    V.append(("基线", {}))
    # --- knife 阴刀弃单 ---
    V.append(("knife_关闭", {"knife_tiers": []}))
    V.append(("knife_减半", {"knife_tiers": [{"mkt_above": -0.5, "mult": 0.5, "res_below": -4.0}]}))
    # --- gap_above 高开降仓 ---
    V.append(("gap_above_关闭", {"gap_above_tiers": []}))
    V.append(("gap_above_0.3", {"gap_above_tiers": [{"above": 0.3, "mult": 0.5, "score_below": 1.6}]}))
    # --- pool_n 池深降仓 ---
    V.append(("池11满仓", {"pool_n_tiers": [{"min": 21, "mult": 1.0}, {"min": 11, "mult": 1.0}, {"min": 1, "mult": 0.5}]}))
    V.append(("池温和", {"pool_n_tiers": [{"min": 21, "mult": 1.0}, {"min": 11, "mult": 0.7}, {"min": 1, "mult": 0.5}]}))
    V.append(("池更狠", {"pool_n_tiers": [{"min": 21, "mult": 1.0}, {"min": 11, "mult": 0.3}, {"min": 1, "mult": 0.1}]}))
    V.append(("池全满仓", {"pool_n_tiers": [{"min": 1, "mult": 1.0}]}))
    # --- mkt_boost 恐慌日加仓 ---
    V.append(("mkt_boost_关闭", {"mkt_boost_tiers": []}))
    # --- score 评分分档 ---
    V.append(("score_分档", {"score_tiers": [{"min": 1.5, "mult": 1.0}, {"min": 1.0, "mult": 0.7}]}))
    return V


def main():
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    base = load_config()
    ctx = prepare_context(base, days)
    rows = []
    for name, patch in variants(base):
        c = copy.deepcopy(base)
        c["execution"].update(patch)
        try:
            r = run_backtest(c, days, ctx)
            row = M(r["metrics"])
            row["变体"] = name
        except Exception as e:  # noqa: BLE001
            row = {"变体": name, "错误": str(e)[:150]}
        rows.append(row)
        print(json.dumps(row, ensure_ascii=False), flush=True)
    out = Path(f"output/posmgmt_216_{days}.json")
    out.write_text(json.dumps(rows, ensure_ascii=False, indent=1))
    print(f"== 已写 {out} ==", flush=True)


if __name__ == "__main__":
    main()
