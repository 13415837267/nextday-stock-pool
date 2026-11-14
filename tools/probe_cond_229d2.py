"""第229轮 D' · 分池探查（2500日）：把主池与备用池拆开再分箱

上一版（4201日）的一个疏漏：952 笔里有 25.9% 来自**备用池**
（momentum/oversold/pullback，pct 门槛 1~7%，与主池 dip 的 −6~−3% 完全不同）。
混在一起分箱，等于把两种策略的样本倒进一个桶，形状会被稀释。

本版按 pool 拆分，只对 main（dip 主池）做分箱 —— 主池条件只影响主池票，
所以判断"某个观测量能不能当条件用"，必须只看主池。
"""
from __future__ import annotations

import os
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
os.environ["BT_FULL_TRADES"] = "1"

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

PANEL = ROOT / "data" / "panel"


def load_day(d: str) -> pd.DataFrame:
    p = PANEL / f"{d}.csv.gz"
    if not p.exists():
        return pd.DataFrame()
    df = pd.read_csv(p)
    df["circ_mv"] = df["amount"] / (df["turnover_rate"].replace(0, float("nan")) / 100) / 1e8
    df["amp"] = (df["high"] - df["low"]) / df["pre_close"].replace(0, float("nan")) * 100
    df["gap"] = (df["open"] / df["pre_close"].replace(0, float("nan")) - 1) * 100
    return df.set_index("ts_code")[["circ_mv", "amp", "gap", "pct", "amount", "turnover_rate"]]


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    out = ROOT / "output" / f"w229_probe_{days}.csv"
    cfg = load_config()
    ctx = prepare_context(cfg, days)
    res = run_backtest(cfg, None, ctx)
    m = res["metrics"]
    print(f"{len(ctx['main_dates'])} 日 · 年化{m['年化收益率%']}% 卡玛{m['卡玛比率']} "
          f"单笔{m['平均单笔收益%']}% {m['交易次数']}笔\n", flush=True)

    rows, cache = [], {}
    for t in res["trades_all"]:
        if t.get("ret_pct") is None:
            continue
        d, code = str(t["signal_date"]), t["code"]
        if d not in cache:
            cache[d] = load_day(d)
        day = cache[d]
        if code not in day.index:
            continue
        r = day.loc[code]
        rows.append({"pool": t.get("pool", "main"), "year": d[:4], "ret": t["ret_pct"],
                     "circ_mv": float(r["circ_mv"]), "amp": float(r["amp"]),
                     "gap": float(r["gap"]), "pct": float(r["pct"]),
                     "amount_yi": float(r["amount"]) / 1e8, "turnover": float(r["turnover_rate"])})
    df = pd.DataFrame(rows)
    df.to_csv(out, index=False, encoding="utf-8-sig")

    print("=== 分池构成 ===", flush=True)
    g = df.groupby("pool")["ret"].agg(["count", "mean", lambda s: (s > 0).mean() * 100])
    g.columns = ["笔数", "平均单笔", "胜率"]
    print(g.round(3).to_string(), flush=True)

    main = df[df["pool"] == "main"].reset_index(drop=True)
    print(f"\n主池 {len(main)} 笔, 平均 {main['ret'].mean():.3f}%", flush=True)

    for col in ["circ_mv", "turnover", "amp", "gap", "amount_yi"]:
        print(f"\n=== 主池 · {col} 五分位 ===", flush=True)
        q = pd.qcut(main[col], 5, duplicates="drop")
        g = main.groupby(q, observed=True)["ret"].agg(["count", "mean", lambda s: (s > 0).mean() * 100])
        g.columns = ["笔数", "平均单笔", "胜率"]
        print(g.round(3).to_string(), flush=True)

    print("\n=== 主池 · circ_mv 十档 ===", flush=True)
    q = pd.qcut(main["circ_mv"], 10, duplicates="drop")
    g = main.groupby(q, observed=True)["ret"].agg(["count", "mean", lambda s: (s > 0).mean() * 100])
    g.columns = ["笔数", "平均单笔", "胜率"]
    print(g.round(3).to_string(), flush=True)

    print("\n=== 主池逐年 ===", flush=True)
    y = main.groupby("year")["ret"].agg(["count", "mean"])
    for yr, r in y.iterrows():
        print(f"  {yr} {int(r['count']):4d}笔 {r['mean']:+.3f}%", flush=True)


if __name__ == "__main__":
    main()
