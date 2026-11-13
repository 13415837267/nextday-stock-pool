"""第226轮B · price_max / turnover_max 的多窗口验收 + 组合

1500 日粗筛结果:
  price_max 150->100: 年化 +6.52pp, 卡玛 6.31->6.68 (双改善)
  turnover_max 12->20: 年化 +9.07pp, 卡玛 6.31->5.93 (收益换风险)
验收窗口: 1500 / 750 / 全史, 任一窗口劣化即不落地。
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


def variants(cfg):
    out = {"A 现行": cfg}

    c = copy.deepcopy(cfg); c["universe"]["price_max"] = 100.0
    out["B price_max=100"] = c

    c = copy.deepcopy(cfg); c["universe"]["turnover_max"] = 20.0
    out["C to_max=20"] = c

    c = copy.deepcopy(cfg); c["universe"]["price_max"] = 100.0
    c["universe"]["turnover_max"] = 20.0
    out["D B+C"] = c

    c = copy.deepcopy(cfg); c["universe"]["price_max"] = 100.0
    c["universe"]["amount_min_yi"] = 1.5
    out["E B+amt1.5"] = c
    return out


def main() -> None:
    days = int(sys.argv[1]) if len(sys.argv) > 1 else 750
    base = load_config()
    ctx = prepare_context(base, days)
    print(f"窗口 {len(ctx['main_dates'])} 日", flush=True)

    rows = []
    for nm, cfg in variants(base).items():
        m = run_backtest(cfg, days, ctx)["metrics"]
        r = {"方案": nm, "年化": float(m["年化收益率%"]), "回撤": float(m["最大回撤%"]),
             "卡玛": float(m["卡玛比率"]), "笔数": int(m["交易次数"]),
             "胜率": float(m["胜率%"])}
        rows.append(r)
        print(f"  {nm:16s} 年化 {r['年化']:7.2f}% 回撤 {r['回撤']:7.2f}% "
              f"卡玛 {r['卡玛']:5.2f} 笔 {r['笔数']:4d} 胜率 {r['胜率']:4.1f}%", flush=True)

    df = pd.DataFrame(rows)
    df.to_csv(ROOT / "output" / f"s226b_{len(ctx['main_dates'])}.csv", index=False,
              encoding="utf-8-sig")
    print("\n" + df.to_string(index=False))


if __name__ == "__main__":
    main()
