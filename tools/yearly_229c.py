"""第229轮 C · 全史分年核查：候选条件的收益到底长在哪几年

单维扫描只看"全史合计"，而 227/228 两轮已经证明：全史合计常被 2021/2022 或最近一段撑起来。
本脚本把每个候选的**逐年平均单笔收益**拉出来，与基准逐年对照：
  · 若 17 年里有 12 年以上不低于基准 → 条件是真结构；
  · 若只有 2~3 年暴涨、其余持平或下降 → 是年份彩票，不落地。

用法: BT_FULL_TRADES=1 python3 tools/yearly_229c.py <days> <case1> <case2> ...
"""
from __future__ import annotations

import copy
import os
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["BT_FULL_TRADES"] = "1"

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

OUT = ROOT / "output" / "w229_yearly.csv"

CASES = {
    "close_pos0.55": {"selection__dip_close_pos_max": 0.55},
    "bars250": {"universe__listed_min_bars": 250},
    "vr6": {"selection__dip_vol_ratio_max": 6.0},
    "bars250+cp0.55": {"universe__listed_min_bars": 250, "selection__dip_close_pos_max": 0.55},
    "bars250+vr6": {"universe__listed_min_bars": 250, "selection__dip_vol_ratio_max": 6.0},
}


def by_year(res: dict) -> dict:
    ts = res.get("trades_all") or []
    rows = [{"y": str(t.get("signal_date") or t.get("buy_date"))[:4], "ret": t["ret_pct"]}
            for t in ts if t.get("ret_pct") is not None]
    d = pd.DataFrame(rows)
    if not len(d):
        return {}
    g = d.groupby("y")["ret"].agg(["count", "mean"])
    return {y: (int(r["count"]), round(float(r["mean"]), 3)) for y, r in g.iterrows()}


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 4201
    picks = sys.argv[2:] or list(CASES)
    base = load_config()

    def mk(patch: dict) -> dict:
        c = copy.deepcopy(base)
        for k, v in patch.items():
            node, parts = c, k.split("__")
            for p in parts[:-1]:
                node = node.setdefault(p, {})
            node[parts[-1]] = v
        return c

    t0 = time.time()
    ctx = prepare_context(base, days)
    print(f"ctx {time.time()-t0:.0f}s, {len(ctx['main_dates'])} 日\n", flush=True)

    out = []
    for name, patch in [("基准", {})] + [(n, CASES[n]) for n in picks if n in CASES]:
        r = run_backtest(mk(patch), None, ctx)
        y = by_year(r)
        m = r["metrics"]
        print(f"--- {name}: 年化{m['年化收益率%']}% 卡玛{m['卡玛比率']} 单笔{m['平均单笔收益%']}% "
              f"{m['交易次数']}笔", flush=True)
        for yr, (n, avg) in sorted(y.items()):
            out.append({"方案": name, "年份": yr, "笔数": n, "平均单笔": avg})
            print(f"    {yr} {n:4d}笔 {avg:+.3f}%", flush=True)

    d = pd.DataFrame(out)
    d.to_csv(OUT, index=False, encoding="utf-8-sig")
    p = d.pivot(index="年份", columns="方案", values="平均单笔")
    n = d.pivot(index="年份", columns="方案", values="笔数")
    print("\n=== 逐年平均单笔(%) ===")
    print(p.to_string())
    print("\n=== 逐年笔数 ===")
    print(n.to_string())
    if "基准" in p.columns:
        print("\n=== 相对基准胜年数 ===")
        for c in p.columns:
            if c == "基准":
                continue
            ok = (p[c] >= p["基准"]).sum()
            print(f"  {c:<16} {ok}/{len(p)} 年不低于基准  "
                  f"Δ均值{(p[c] - p['基准']).mean():+.3f}pp")


if __name__ == "__main__":
    main()
