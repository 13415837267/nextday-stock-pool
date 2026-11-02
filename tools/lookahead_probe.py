# -*- coding: utf-8 -*-
"""前视探针(回归测试): 截断面板重算指标+选股, 必须与全量面板逐位相同。

原理: 若指标/选股只依赖截至 D 日的信息, 那么"把 D 之后的行整段删掉再算一遍"
在 D 日应当得到**完全相同**的候选、评分与风控状态。任何差异都说明有未来信息泄漏
(典型伪装: 相对面板末日归一化的量、用当前状态回填历史、分子分母不同复权口径)。

用法:
    python tools/lookahead_probe.py [--days 420] [--probes 250,320,-1]
退出码 0 = 全部一致; 1 = 发现差异。
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402

from src.backtest import ZZClient, fetch_panel  # noqa: E402
from src.config import load_config  # noqa: E402
from src.selector import build_indicators, regime_series, select_stocks  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--days", type=int, default=420, help="面板交易日数(含预热)")
    ap.add_argument("--probes", default="250,320,-1", help="探针日下标, 逗号分隔, 支持负数")
    ap.add_argument("--end", default=None, help="面板最后一日上限 YYYYMMDD")
    a = ap.parse_args()

    cfg = load_config()
    client = ZZClient(cfg)
    days = client.trade_days(days=a.days + 10)
    if a.end:
        days = [d for d in days if d <= a.end]
    days = days[-a.days:]
    print(f"面板 {len(days)} 日: {days[0]} ~ {days[-1]}", flush=True)

    raw = fetch_panel(client, days, float32=False)
    full = build_indicators(raw.copy())
    reg_full = regime_series(full, cfg)
    by_full = {str(k): v for k, v in full.groupby("trade_date", sort=True)}
    print(f"全量面板 {len(raw):,} 行, 指标就绪", flush=True)

    probes = [days[int(i)] for i in a.probes.split(",")]
    bad = 0
    for d in probes:
        picks_a, st_a = select_stocks(d, by_full[d], cfg, names={})
        ra = reg_full.loc[d]

        sub = build_indicators(raw.loc[raw["trade_date"] <= d].copy())
        reg_b = regime_series(sub, cfg)
        picks_b, st_b = select_stocks(d, sub.loc[sub["trade_date"] == d], cfg, names={})
        rb = reg_b.loc[d]

        ka = [(p["code"], round(float(p["score"]), 8)) for p in picks_a]
        kb = [(p["code"], round(float(p["score"]), 8)) for p in picks_b]
        same = ka == kb
        risk_same = (bool(ra["trade_ok"]) == bool(rb["trade_ok"])
                     and abs(float(ra["breadth"]) - float(rb["breadth"])) < 1e-12
                     and abs(float(ra["mkt_ma"]) - float(rb["mkt_ma"])) < 1e-12
                     and bool(ra["panic"]) == bool(rb["panic"]))
        flag = "OK" if (same and risk_same) else "LEAK"
        print(f"[{flag}] {d}: 候选 {len(ka)} vs {len(kb)}, 逐位一致={same}, 风控一致={risk_same}",
              flush=True)
        if not same:
            bad += 1
            sa = dict(ka)
            sb = dict(kb)
            print("  仅全量版:", [x for x in ka if x[0] not in sb][:5])
            print("  仅截断版:", [x for x in kb if x[0] not in sa][:5])
            print("  同码评分差:",
                  [(c, sa[c], sb[c]) for c in sa if c in sb and abs(sa[c] - sb[c]) > 1e-9][:5])
        if not risk_same:
            bad += 1
            print(f"  风控: trade_ok {ra['trade_ok']}/{rb['trade_ok']}, "
                  f"breadth {ra['breadth']}/{rb['breadth']}, mkt_ma {ra['mkt_ma']}/{rb['mkt_ma']}")

    print(f"\n{'ALL PASS' if bad == 0 else f'{bad} 处差异'}")
    return 0 if bad == 0 else 1


if __name__ == "__main__":
    raise SystemExit(main())
