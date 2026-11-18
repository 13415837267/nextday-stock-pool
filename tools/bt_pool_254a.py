"""第254轮: 导出每个交易日的"机会丰度"漏斗 —— scored 绝对值 vs 达标率。

动机(253 结论的尾巴): pool_n 是绝对只数, 21 只在 2016 年(全市场约 2800 只)与
2026 年(约 5300 只)含义差一倍。若"达标率"(scored / 当日可交易只数)在各年之间
分布更稳、与次日收益的关系更单调, 那阈值就该用相对口径。

产出 output/w254_funnel_<days>.pkl:
  DataFrame[date, universe, liquid, trend_ok, scored, idx_dist]
"""
from __future__ import annotations

import os
import pickle
import sys
import time

import pandas as pd

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from src.backtest import prepare_context            # noqa: E402
from src.config import load_config                  # noqa: E402
from src.fillin import idx_dist_at                 # noqa: E402
from src.selector import select_stocks              # noqa: E402


def main(days: int) -> None:
    cfg = load_config()
    ctx = prepare_context(cfg, days)
    dates, by_date = ctx["main_dates"], ctx["by_date"]
    fidx = ctx.get("fill_idx")
    rows = []
    t0 = time.time()
    for i, d in enumerate(dates):
        day = by_date[d]
        picks, st = select_stocks(d, day, cfg, names=None, name_filter=False)
        rows.append({
            "date": d,
            "universe": int(len(day)),
            "liquid": int(st.get("liquidity") or 0),
            "price_action": int(st.get("price_action") or 0),
            "trend_ok": int(st.get("trend_ok") or 0),
            "scored": int(st.get("scored") or 0),
            "picks": len(picks),
            "idx_dist": (idx_dist_at(fidx, d) if fidx is not None else None),
        })
        if (i + 1) % 200 == 0:
            print(f"  {i + 1}/{len(dates)}  {time.time() - t0:.0f}s", flush=True)
    df = pd.DataFrame(rows)
    out = f"output/w254_funnel_{days}.pkl"
    with open(out, "wb") as f:
        pickle.dump(df, f)
    print(df.tail(3).to_string())
    print(f"→ {out}  {len(df)} 日  {time.time() - t0:.0f}s")


if __name__ == "__main__":
    main(int(sys.argv[1]) if len(sys.argv) > 1 else 2500)
