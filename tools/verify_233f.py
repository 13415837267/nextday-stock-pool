"""第233轮 · 全史复验：风控到底值多少钱（基准 / 全关 / ma=30）"""
from __future__ import annotations

import copy
import sys
import time
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402

OUT = ROOT / "output" / "w233f_fullhist.csv"


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
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 4201
    cfg = load_config()
    ctx = prepare_context(cfg, days)
    base = brief(run_backtest(cfg, None, ctx))
    rows = [{"方案": "基准(MA20)", **base, "Δ年化": 0.0}]
    print(f"基准 {base}\n", flush=True)
    for name, kw in [("风控全关", dict(enabled=False)), ("ma=30", dict(market_ma=30)),
                     ("ma=15", dict(market_ma=15))]:
        t0 = time.time()
        r = brief(run_backtest(mk(cfg, **kw), None, ctx))
        rows.append({"方案": name, **r, "Δ年化": round(r["年化"] - base["年化"], 2)})
        pd.DataFrame(rows).to_csv(OUT, index=False, encoding="utf-8-sig")
        print(f"{name:10s} 年化{r['年化']:7.2f}% ({r['年化']-base['年化']:+6.2f}) "
              f"回撤{r['回撤']:7.2f} 卡玛{r['卡玛']:5.2f} 笔数{r['笔数']:4d} "
              f"单笔{r['单笔']:6.3f}% [{time.time()-t0:.0f}s]", flush=True)
    print("\n" + pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
