# -*- coding: utf-8 -*-
"""第 196 轮: "名义无杠杆"审计 —— leverage=1.0 的账户, 实际有没有在用融资?

起因
----
第 195 轮按用户指令把 execution.leverage 从 1.25 降到 1.0。但第 173 轮落地的
mkt_boost_tiers(信号日全市场涨跌中位数落在 [-2.5%, -1.0%] 时仓位 ×1.5)会产出
pick_frac=1.5, 而引擎在 alloc 前的额度公式是:
    _lev_eff = lev_now * max(1.0, pick_frac)
    margin_avail = (_lev_eff - 1.0) * equity
也就是说 **pick_frac>1 会自动打开融资额度**, 与 leverage 写的是 1.25 还是 1.0 无关。
1500 日里 pos_frac=1.5 的成交有 71 笔(19%), 均敞口 1.121 —— 账户实际上仍在用 0.5 倍融资。

本轮把三条路并排跑:
  A 现行:   boost ×1.5, 允许透支(均敞口 ~1.12)
  B 关掉:   mkt_boost_tiers = []  (纯自有资金, 但"恐慌日加仓"的择时价值一并丢掉)
  C 截断:   boost ×1.5 但 pos_frac_cap=1.0 (恐慌日只把"池浅减掉的仓"补回来,
            永远不超过自有资金)
C 是"保留择时、去掉杠杆"的唯一形态, 也是本轮要判断的东西。

用法: python3 -u tools/exp_lev_196.py [days]
"""
from __future__ import annotations

import copy
import json
import os
import sys
from collections import Counter
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from src.backtest import prepare_context, run_backtest   # noqa: E402
from src.config import ROOT, load_config                # noqa: E402

OUT = ROOT / "output"


def _exposure(res: dict) -> dict:
    tr = [t for t in res.get("trades_all", []) if t.get("ret_pct") is not None]
    pf = [float(t.get("pos_frac") or 0) for t in tr]
    over = [x for x in pf if x > 1.0001]
    return {"成交笔数": len(tr), "pos_frac>1笔数": len(over),
            "超配占比%": round(len(over) / len(tr) * 100, 1) if tr else 0,
            "均目标仓位": round(float(np.mean(pf)), 4) if pf else None,
            "最大目标仓位": round(max(pf), 3) if pf else None,
            "分布": {str(k): v for k, v in sorted(Counter(pf).items())}}


def _run(cfg, days, ctx_cache):
    if days not in ctx_cache:
        ctx_cache[days] = prepare_context(cfg, days)
    return run_backtest(cfg, days, ctx_cache[days])


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 1500
    os.environ["BT_FULL_TRADES"] = "1"
    base = load_config()
    ctx_cache: dict = {}

    variants = {}
    a = copy.deepcopy(base)
    variants["A_现行_boost1.5可透支"] = a
    b = copy.deepcopy(base)
    b["execution"]["mkt_boost_tiers"] = []
    variants["B_关闭boost"] = b
    c = copy.deepcopy(base)
    c["execution"]["pos_frac_cap"] = 1.0
    variants["C_boost截断到1.0"] = c

    rows = {}
    for name, cfg in variants.items():
        res = _run(cfg, days, ctx_cache)
        m = res["metrics"]
        rows[name] = {"总收益率%": m["总收益率%"], "年化收益率%": m["年化收益率%"],
                      "最大回撤%": m["最大回撤%"], "卡玛比率": m["卡玛比率"],
                      "夏普比率": m["夏普比率"], "交易次数": m["交易次数"],
                      "平均单笔%": m["平均单笔收益%"], "总费用": m["总费用(元)"],
                      **_exposure(res)}
        print(f"{name}: 总 {m['总收益率%']:.2f}% 年化 {m['年化收益率%']:.2f}% "
              f"回撤 {m['最大回撤%']:.2f}% 卡玛 {m['卡玛比率']} 笔 {m['交易次数']} "
              f"| 超配 {rows[name]['pos_frac>1笔数']} 笔 均仓 {rows[name]['均目标仓位']} "
              f"费用 {m['总费用(元)'] / 1e4:.0f}万", flush=True)

    fp = OUT / f"exp_lev_196_{days}.json"
    fp.write_text(json.dumps({"days": days, "variants": rows}, ensure_ascii=False, indent=1),
                  encoding="utf-8")
    print(f"写出 {fp}")


if __name__ == "__main__":
    main()
