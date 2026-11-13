"""第225轮E · 候选改动在多窗口验收（防止单窗口过拟合）

候选:
  A 现行
  B neg_bias20 0.70 -> 0.85      （真实回测内坐标搜索的最优）
  C 执行层扫描最优（若已产出 w225c_gap.json 则读入）
窗口: 1500 日 / 近 750 日 / 全史(4100 日)
判定: 三个窗口年化都不低于现行 且 卡玛都不低于现行*0.98 才算通过。
"""
from __future__ import annotations

import copy
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402


def one(cfg, days, ctx) -> dict:
    m = run_backtest(cfg, days, ctx)["metrics"]
    return {"年化": m["年化收益率%"], "回撤": m["最大回撤%"],
            "卡玛": m["卡玛比率"], "笔数": m["交易次数"], "胜率": m["胜率%"]}


def main() -> None:
    base = load_config()
    cands = {"A 现行": base}

    c = copy.deepcopy(base)
    c["selection"]["dip_weights"]["neg_bias20"] = 0.85
    cands["B neg_bias20=0.85"] = c

    gp = ROOT / "output" / "w225c_gap.json"
    if gp.exists():
        d = json.load(open(gp))
        best = max(d["rows"], key=lambda r: r["卡玛"])
        c3 = copy.deepcopy(base)
        c3["execution"]["block_gap_below"] = best["block_gap"]
        c3["execution"]["blocked_frac"] = best["frac"]
        cands[f"C 执行最优(gap{best['block_gap']}/frac{best['frac']})"] = c3
        # B+C 组合
        c4 = copy.deepcopy(c3)
        c4["selection"]["dip_weights"]["neg_bias20"] = 0.85
        cands["D B+C"] = c4

    for days in (1500, 750, 4100):
        ctx = prepare_context(base, days)
        print(f"\n===== 窗口 {len(ctx['main_dates'])} 日 =====", flush=True)
        rows = []
        for nm, cfg in cands.items():
            try:
                r = one(cfg, days, ctx)
            except Exception as e:  # noqa: BLE001
                print(f"  [FAIL] {nm}: {e}", flush=True)
                continue
            rows.append({"方案": nm, **r})
            print(f"  {nm:28s} 年化 {r['年化']:7.2f}% 回撤 {r['回撤']:7.2f}% "
                  f"卡玛 {r['卡玛']:5.2f} 笔 {r['笔数']:4d} 胜率 {r['胜率']:4.1f}%", flush=True)
        pd.DataFrame(rows).to_csv(ROOT / "output" / f"w225e_{len(ctx['main_dates'])}.csv",
                                 index=False, encoding="utf-8-sig")


if __name__ == "__main__":
    main()
