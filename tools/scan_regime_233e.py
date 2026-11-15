"""第233轮 · 风控的"形式"：笔数的上限由它决定

漏斗实测：2500 个交易日里风控只放行 1339 天（53.6%）。持仓 2 天，
故笔数理论上限 ≈ 670，当前 559 = 83%。放宽缺口门槛去填满这 17%，
实测 −21pp（买进来的都是差票）。于是"笔数"这个乘数只剩一个没动过的旋钮：
**风控本身把 45% 的日子关在门外，这个比例合理吗？**

过去只调过 breadth_min（0.20~0.35 不敏感）与 market_ma（20 硬最优），
从未改过风控的**组合形式**：and / or / 分档参与。
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

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

OUT = ROOT / "output" / "w233e_regime.csv"


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


def mk(base: dict, **rg) -> dict:
    c = copy.deepcopy(base)
    c["regime"] = dict(c["regime"])
    c["regime"].update(rg)
    return c


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 2500
    cfg = load_config()
    ctx = prepare_context(cfg, days)
    base = brief(run_backtest(cfg, None, ctx))
    print(f"基准 {base}\n")

    prev = pd.read_csv(OUT, encoding="utf-8-sig") if OUT.exists() else None
    rows = [] if prev is None else prev.to_dict("records")
    if prev is None:
        rows.append({"方案": "基准(and)", **base, "Δ年化": 0.0})
    have = {r["方案"] for r in rows}

    plan = [
        ("风控全关", dict(enabled=False)),
        ("mode=or", dict(mode="or")),
        ("soft 0.5", dict(soft_trend_frac=0.5)),
        ("soft 0.3", dict(soft_trend_frac=0.3)),
        ("breadth 0.15", dict(breadth_min=0.15)),
        ("breadth 0.10", dict(breadth_min=0.10)),
        ("ma=10", dict(market_ma=10)),
        ("ma=30", dict(market_ma=30)),
        ("vol_max 关闭", dict(vol_max=None)),
        ("vol_max 1.8", dict(vol_max=1.8)),
    ]
    for name, kw in plan:
        if name in have:
            continue
        t0 = time.time()
        r = brief(run_backtest(mk(cfg, **kw), None, ctx))
        rows.append({"方案": name, **r, "Δ年化": round(r["年化"] - base["年化"], 2)})
        pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
        print(f"{name:14s} 年化{r['年化']:7.2f}% ({r['年化']-base['年化']:+6.2f}) "
              f"回撤{r['回撤']:7.2f} 卡玛{r['卡玛']:5.2f} 笔数{r['笔数']:4d} "
              f"单笔{r['单笔']:6.3f}% [{time.time()-t0:.0f}s]", flush=True)

    print("\n" + pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
