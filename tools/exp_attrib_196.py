# -*- coding: utf-8 -*-
"""第 196 轮归因: 空仓日到底是"浪费"还是"避雷"?

资金复用(exp_reuse_196)的结果是: 把主策略空档期的钱借给副策略, 零成本下年化 +5.58pp,
但回撤从 -17.60% 扩到 -25.18%, 卡玛 7.59 -> 5.72; 一旦计入让位换手(0.3%/次),
年化掉 20pp。三个档位(25/50/75/100% 仓位上限)全部是"收益小增 + 卡玛大降"。

本脚本回答"为什么", 用的是同一批数据, 只做切分不做新回测:
  A. 主策略在场日 vs 空仓日, **市场等权基准**的日均收益(择时是否真的躲开了下跌);
  B. 副策略在"主策略在场日" vs "主策略空仓日"的日均收益(空档期的质量);
  C. 副策略 106 笔按"持仓期是否与主策略冲突"分组(冲突笔 vs 独占笔的单笔收益);
  D. 主策略 365 笔按"信号日前 5 日市场涨跌"分组, 看 edge 是否只在特定市况。

用法: python3 -u tools/exp_attrib_196.py [days]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.config import ROOT                       # noqa: E402
from tools.exp_reuse_196 import _d8, _leg         # noqa: E402

OUT = ROOT / "output"


def _bench(res: dict) -> dict[str, float]:
    eq = res.get("benchmark") or []
    nav = {_d8(x["d"]): float(x["nav"]) for x in eq}
    ds = sorted(nav)
    return {ds[i]: nav[ds[i]] / nav[ds[i - 1]] - 1 for i in range(1, len(ds)) if nav[ds[i - 1]]}


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    rm_res = json.loads((OUT / f"reuse_196_m{days}.json").read_text(encoding="utf-8"))
    rs_res = json.loads((OUT / f"reuse_196_s{days}.json").read_text(encoding="utf-8"))

    dates, rm, wm = _leg(rm_res, "buy_date", "sell_date", "pos_frac")
    _, rs, ws = _leg(rs_res, "entry_date", "exit_date", None)
    bm = _bench(rm_res)

    hold = [d for d in dates[1:] if wm.get(d, 0) > 0]
    flat = [d for d in dates[1:] if wm.get(d, 0) <= 0]
    pct = lambda v: round(float(np.mean(v)) * 100, 3) if len(v) else None  # noqa: E731

    A = {
        "在场日数": len(hold), "空仓日数": len(flat),
        "在场日_市场日均%": pct([bm[d] for d in hold if d in bm]),
        "空仓日_市场日均%": pct([bm[d] for d in flat if d in bm]),
        "在场日_市场胜率%": round(float(np.mean([bm[d] > 0 for d in hold if d in bm])) * 100, 1),
        "空仓日_市场胜率%": round(float(np.mean([bm[d] > 0 for d in flat if d in bm])) * 100, 1),
    }

    # B: 副策略持仓日, 按主策略当日是否在场分组
    s_hold = [d for d in dates[1:] if ws.get(d, 0) > 0]
    s_overlap = [d for d in s_hold if wm.get(d, 0) > 0]
    s_alone = [d for d in s_hold if wm.get(d, 0) <= 0]
    B = {
        "副持仓日": len(s_hold), "与主重叠": len(s_overlap), "独占空档": len(s_alone),
        "重叠日_副日均%": pct([rs[d] for d in s_overlap]),
        "空档日_副日均%": pct([rs[d] for d in s_alone]),
        "空档日_市场日均%": pct([bm[d] for d in s_alone if d in bm]),
        "重叠日_市场日均%": pct([bm[d] for d in s_overlap if d in bm]),
    }

    # C: 副策略逐笔, 按持仓期是否与主策略冲突
    idx = {d: i for i, d in enumerate(dates)}
    conf, solo = [], []
    for t in rs_res["trades"]:
        if t.get("ret_pct") is None:
            continue
        b, s = _d8(t["entry_date"]), _d8(t["exit_date"])
        if b not in idx or s not in idx:
            continue
        days_rng = dates[idx[b]:idx[s] + 1]
        hit = any(wm.get(d, 0) > 0 for d in days_rng)
        (conf if hit else solo).append(float(t["ret_pct"]))
    C = {"冲突笔数": len(conf), "独占笔数": len(solo),
         "冲突笔_单笔%": round(float(np.mean(conf)), 3) if conf else None,
         "独占笔_单笔%": round(float(np.mean(solo)), 3) if solo else None,
         "冲突笔_胜率%": round(float(np.mean([x > 0 for x in conf])) * 100, 1) if conf else None,
         "独占笔_胜率%": round(float(np.mean([x > 0 for x in solo])) * 100, 1) if solo else None}

    # D: 主策略逐笔, 按信号日前的市场 5 日动量分组
    m5 = {}
    ds = sorted(bm)
    for i, d in enumerate(ds):
        if i >= 5:
            m5[d] = float(np.prod([1 + bm[ds[j]] for j in range(i - 4, i + 1)]) - 1)
    buckets = {"跌>3%": [], "跌0~3%": [], "涨0~3%": [], "涨>3%": []}
    for t in rm_res["trades_all"]:
        if t.get("ret_pct") is None:
            continue
        sd = _d8(t["signal_date"])
        # 信号日收盘时的前 5 日动量: 取 signal_date 当日的 m5
        if sd not in m5:
            continue
        v = m5[sd]
        k = "跌>3%" if v < -0.03 else "跌0~3%" if v < 0 else "涨0~3%" if v < 0.03 else "涨>3%"
        buckets[k].append(float(t["ret_pct"]))
    D = {k: {"笔数": len(v), "单笔%": round(float(np.mean(v)), 3) if v else None}
         for k, v in buckets.items()}

    out = {"days": days, "A_择时是否躲开下跌": A, "B_副策略在空档期": B,
           "C_副策略逐笔冲突": C, "D_主策略按信号前5日动量": D}
    fp = OUT / f"exp_attrib_196_{days}.json"
    fp.write_text(json.dumps(out, ensure_ascii=False, indent=1), encoding="utf-8")
    print(json.dumps(out, ensure_ascii=False, indent=1))
    print(f"写出 {fp}")


if __name__ == "__main__":
    main()
