"""第二百三十四轮: 选股区间整段下移 —— 前面 200 多轮从没测过的方向

全市场 817 万低开样本的曲线是单调的: 跌 3.75% 是 0.35%, 跌 10.25% 是 1.07%。
但策略从早期版本起就固定买 [-6%, -3%]。229 轮只测过"两头各收一格"
(dip_pct_max -3->-4.5, dip_pct_min -6->-5), 从来没测过"整段住下移"。

风险(必须先看清楚): 深跌票 T+2 若跌停则卖不掉, 引擎按收盘价成交会高估。
所以本脚本同时记录笔数与单笔, 若单笔被少数极端样本拉高, 会在报告里点明。
"""
from __future__ import annotations

import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest      # noqa: E402
from src.config import load_config                          # noqa: E402

OUT = ROOT / "output" / "w234_range.csv"


def brief(res: dict) -> dict:
    m = res["metrics"]
    nav = pd.DataFrame(res["equity"])["nav"].astype(float).values
    n = len(nav)
    seg = {}
    for i, tag in enumerate(["s1", "s2", "s3"]):
        x = nav[int(n * i / 3): int(n * (i + 1) / 3)]
        yrs = max(0.3, len(x) / 244.0)
        seg[tag] = round(((x[-1] / x[0]) ** (1 / yrs) - 1) * 100, 1) if len(x) > 40 else None
    return {"年化": round(float(m["年化收益率%"]), 2),
            "回撤": round(float(m["最大回撤%"]), 2),
            "卡玛": round(float(m["卡玛比率"] or 0), 2),
            "笔数": int(m["交易次数"]),
            "胜率": round(float(m["胜率%"]), 1),
            "单笔": round(float(m["平均单笔收益%"]), 3), **seg}


def mk(base: dict, lo: float, hi: float, extra: dict | None = None) -> dict:
    c = dict(base)
    c["selection"] = dict(base["selection"])
    c["selection"]["dip_pct_min"] = lo
    c["selection"]["dip_pct_max"] = hi
    if extra:
        c["selection"].update(extra)
    return c


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    base = load_config()
    t0 = time.time()
    ctx = prepare_context(base, days)
    print(f"ctx {time.time()-t0:.0f}s, {len(ctx['main_dates'])} 日\n", flush=True)

    plan = [
        ("F0 [-6,-3] 现行", -6.0, -3.0, None),
        ("F1 [-7,-3]", -7.0, -3.0, None),
        ("F2 [-8,-5]", -8.0, -5.0, None),
        ("F3 [-9,-4]", -9.0, -4.0, None),
        ("F4 [-10,-6]", -10.0, -6.0, None),
        ("F5 [-11,-6]", -11.0, -6.0, None),
        ("F6 [-9,-3]", -9.0, -3.0, None),
        ("F7 [-10,-6] 放宽度距低点", -10.0, -6.0, dict(dip_dist_low20_min=0.0)),
    ]
    rows = []
    for name, lo, hi, ex in plan:
        t = time.time()
        r = brief(run_backtest(mk(base, lo, hi, ex), None, ctx))
        r["方案"] = name
        if rows:
            r["Δ年化"] = round(r["年化"] - rows[0]["年化"], 2)
            r["Δ笔数"] = r["笔数"] - rows[0]["笔数"]
        else:
            r["Δ年化"], r["Δ笔数"] = 0.0, 0
        rows.append(r)
        print(f"{name:24s} 年化 {r['年化']:7.2f}% ({r['Δ年化']:+6.2f})  "
              f"回撤 {r['回撤']:7.2f}%  卡玛 {r['卡玛']:5.2f}  "
              f"笔数 {r['笔数']:4d} ({r['Δ笔数']:+4d})  单笔 {r['单笔']:+.3f}%  "
              f"[{time.time()-t:.0f}s]", flush=True)
        pd.DataFrame(rows)[["方案", "年化", "回撤", "卡玛", "笔数", "胜率", "单笔",
                            "s1", "s2", "s3", "Δ年化", "Δ笔数"]].to_csv(
            OUT, index=False, encoding="utf-8-sig")

    print("\n" + pd.DataFrame(rows)[["方案", "年化", "回撤", "卡玛", "笔数", "胜率",
                                     "单笔", "s1", "s2", "s3"]].to_string(index=False), flush=True)


if __name__ == "__main__":
    main()
