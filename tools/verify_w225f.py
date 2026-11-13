"""第225轮F · 全史窗口单独验收 B(neg_bias20 0.70->0.85)

1500/750 两窗口已通过(年化+2.4pp, 回撤改善1.5~4pp)。全史窗口取数约 200s, 单独跑以避开超时。
"""
from __future__ import annotations

import copy
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.backtest import prepare_context, run_backtest  # noqa: E402
from src.config import load_config  # noqa: E402


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 4100
    base = load_config()
    ctx = prepare_context(base, days)
    print(f"窗口 {len(ctx['main_dates'])} 日", flush=True)

    rows = []
    for nm, w in [("A 现行", 0.70), ("B neg_bias20=0.85", 0.85)]:
        c = copy.deepcopy(base)
        c["selection"]["dip_weights"]["neg_bias20"] = w
        m = run_backtest(c, days, ctx)["metrics"]
        r = {"方案": nm, "年化": m["年化收益率%"], "回撤": m["最大回撤%"],
             "卡玛": m["卡玛比率"], "笔数": m["交易次数"], "胜率": m["胜率%"]}
        rows.append(r)
        print(f"  {nm:20s} 年化 {r['年化']:7.2f}% 回撤 {r['回撤']:7.2f}% "
              f"卡玛 {r['卡玛']:5.2f} 笔 {r['笔数']:4d} 胜率 {r['胜率']:4.1f}%", flush=True)
    pd.DataFrame(rows).to_csv(ROOT / "output" / f"w225e_{len(ctx['main_dates'])}.csv",
                             index=False, encoding="utf-8-sig")
    print(pd.DataFrame(rows).to_string(index=False))


if __name__ == "__main__":
    main()
