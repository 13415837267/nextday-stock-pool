"""第233轮 · 买入端缺口门槛：唯一可能"加笔数而不砍仓位"的杠杆

漏斗显示 2500 日里 1319 次放弃属于"缺口分档"（远多于评分门槛的 594 次），
即：每天都有候选，但 T+1 开盘的缺口经常不落在可买区间内，于是空仓。

策略的 alpha 有一半来自"低开折扣"（232 轮实测 0.61pp/笔）。放宽缺口
= 用更低的开盘折扣换更多笔数。年化 ≈ 单笔 × 笔数，临界点就在两者
的相对变化上，必须实测而不能猜。
"""
from __future__ import annotations

import copy
import os
import re
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

OUT = ROOT / "output" / "w233d_gap.csv"


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


def mk(base: dict, **ex) -> dict:
    c = copy.deepcopy(base)
    c["execution"] = dict(c["execution"])
    c["execution"].update(ex)
    return c


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    os.environ["BT_FULL_TRADES"] = "1"
    cfg = load_config()
    ctx = prepare_context(cfg, days)

    # ---- 诊断：把被"缺口分档"拒掉的缺口值全部抽出来看分布 ----
    res = run_backtest(cfg, None, ctx)
    base = brief(res)
    print(f"基准 {base}\n")
    gs = []
    for t in res["trades_all"]:
        if t.get("ret_pct") is not None:
            continue
        n = t.get("note") or ""
        m = re.match(r"缺口(-?\d+\.?\d*)%", n)
        if m:
            gs.append(float(m.group(1)))
    s = pd.Series(gs)
    print(f"缺口类放弃 {len(s)} 条；缺口分布：")
    edges = [-99, -6, -5, -4, -3, -2, -1, -0.5, 0, 0.5, 1, 2, 3, 5, 99]
    print(pd.cut(s, edges).value_counts().sort_index().to_string())
    print(f"  其中 gap>0（高开被拒）{(s > 0).sum()} 条 = {(s > 0).mean():.1%}")
    print(f"  gap>1% {(s > 1).sum()}；gap<-5%（深低开被拒）{(s < -5).sum()}\n")

    prev = pd.read_csv(OUT, encoding="utf-8-sig") if OUT.exists() else None
    have = set(prev["方案"]) if prev is not None else set()

    plan = [
        ("基准(现行)", {}),
        ("gap<=+0.5", dict(buy_gap_below=0.5, gap_floor_tiers=None)),
        ("gap<=+1.0", dict(buy_gap_below=1.0, gap_floor_tiers=None)),
        ("gap<=+2.0", dict(buy_gap_below=2.0, gap_floor_tiers=None)),
        ("gap<=+3.0", dict(buy_gap_below=3.0, gap_floor_tiers=None)),
        ("gap<=+5.0", dict(buy_gap_below=5.0, gap_floor_tiers=None)),
        ("floor=2.0", dict(gap_floor_tiers=[{"below": -1.0, "min_pool": 5, "floor": 2.0}])),
        ("floor=3.0", dict(gap_floor_tiers=[{"below": -1.0, "min_pool": 5, "floor": 3.0}])),
        ("gap<=-0.5(收紧)", dict(buy_gap_below=-0.5, gap_floor_tiers=None)),
        ("gap<=-1.0(收紧)", dict(buy_gap_below=-1.0, gap_floor_tiers=None)),
    ]

    rows = []
    if prev is None:
        rows.append({"方案": "基准(现行)", **base, "Δ年化": 0.0})
        pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
    for name, kw in plan:
        if name in have or name == "基准(现行)":
            continue
        c = mk(cfg, **kw)
        t0 = time.time()
        r = brief(run_backtest(c, None, ctx))
        row = {"方案": name, **r, "Δ年化": round(r["年化"] - base["年化"], 2)}
        rows.append(row)
        pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
        print(f"{name:16s} 年化{r['年化']:7.2f}% ({row['Δ年化']:+6.2f}) 回撤{r['回撤']:7.2f} "
              f"卡玛{r['卡玛']:5.2f} 笔数{r['笔数']:4d} 单笔{r['单笔']:6.3f}% "
              f"[{time.time()-t0:.0f}s]", flush=True)

    df = pd.DataFrame(rows)
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
